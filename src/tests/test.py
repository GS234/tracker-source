from __future__ import annotations # da delajo tut type hint-i znotraj istega class-a
import numpy as np
import random
import math
import cv2 as cv

# global vars:
R = 10

# S1, S2 = 180, 60 # default s1,s2
# S1, S2 = 80, 40 # default s1,s2
# S1, S2 = 120, 60 # default s1,s2
S1, S2 = 12, 6 # default s1,s2

# test trajectories:
t1 = [
    [(122, 52)],
    [(115, 64)],
    [(120, 79),(112, 81),(116, 83),(116, 84),(114, 87)],
    [(110, 98),(104, 103),(105, 105),(105, 106),(101, 109),],
    [(104, 126),(99, 128) ,(98, 129) ,(99, 129) ,(100, 132)],
    [(114, 146),(114, 147)],
    [(128, 173),(131, 174),(132, 174)],
    [(148, 191),(152, 191)],
    [(146, 207),(145, 210),(148, 210),(145, 211),(148, 211)],
    [(146, 226)],
    [(141, 232)],
    [(133, 232),(133, 234),(133, 236)],
    [(123, 234)],
    [(114, 224)],
    [(103, 210),(103, 211),(104, 212)],
    [(96, 215)],
    [(89, 223),(89, 226),(87, 228),(91, 228),(91, 230),(90, 231)],
    [(93, 242)],
    [(105, 256)],
    [(113, 270),(113, 272)],
    [(118, 285)],
    [(123, 287)],
    [(134, 284)],
    [(145, 294),(143, 295),(146, 296)],
    [(158, 312)],
    [(156, 326)],
    [(160, 338),(160, 339)],
    [(155, 352),(155, 354)],
    [(150, 364)],
    [(144, 373),(143, 374),(142, 375),(140, 376)],
    [(126, 384)],
    [(109, 387),(108, 388),(110, 389)],
    [(97, 398)],
    [(96, 408),(96, 409)],
    [(85, 424),(85, 425),(84, 426),(85, 426)],
    [(80, 439)],
    [(94, 446)],
    [(116, 452),(117, 452)],
    [(139, 440),(137, 441),(136, 442),(137, 442)],
    [(149, 439)],
    [(163, 436)],
    [(188, 426)],
    [(204, 409)]
]


def coords2det2(X):
    detections = []
    i = 0
    for x_t in X:
        det_arr = []
        
        for x in x_t:
            det_arr.append(Detection(x,0,i,0))
        detections.append(det_arr)
        i = i+1
    return detections



# -----------------




# detection class; if becomes too complicated, move it to other file
class Detection:
    # constructor: position, velocity, timestamp
    def __init__(self, x, v = 0, t=0, theta=0):
        self.x = np.array(x)
        self.v = v #initial velocity is 0
        self.t = t #time stamp
        self.theta = theta

    # calculates rotation matrix based on detection orientation (calculate each time, as theta might change)
    def getRotationMatrix(self):
        return np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta)],
                [np.sin(self.theta), np.cos(self.theta)]
            ])
    
    def getRotationMatrix2(self, theta=None):
        if(theta is None):
            theta = self.theta
        return np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
    
    # calculate next detection estimate from current info (x_p)
    # POMEMBNA ZADEVA
    def estimateNext(self, dt = 1) -> Detection:
        x_t1 = self.x[0] + int(dt*self.v*math.cos(self.theta))
        y_t1 = self.x[1] + int(dt*self.v*math.sin(self.theta))
        return Detection((x_t1, y_t1), self.v, self.t+dt, self.theta)
    
    # calculate search region bound coordinates (for visualization)
    def getSearchRegionBounds(self, r=R):
        phi_ = np.arange(0, 2*math.pi, math.pi/20.0) # angles on a ring of bounding points
        x_ = np.cos(phi_)*r + self.x[0]
        y_ = np.sin(phi_)*r + self.x[1]
        return [(int(x_[i]), int(y_[i])) for i in range(len(phi_))] # return list of tuples of bounding points around detection
    
    # calculate search region bound coordinates (ellipse)
    # TUDI TO JE POMEMBNA ZADEVA
    def getSearchRegionBounds2(self, s1=S1, s2=S2, theta=None, n=50):
        # rotation matrix
        rot_mat = self.getRotationMatrix2(theta)
        
        full_circle = np.arange(0, 2*np.pi, 2*np.pi/n)

        X = np.array(
            [
                s1*np.sin(full_circle),
                s2*np.cos(full_circle)
            ])
        
        X_rot = np.dot(rot_mat, X) # rotate it!
        # X_rot = np.dot(rot_mat.T, X_rot) # rotate back same amount (inverse rotation)
        X_rot[0] += self.x[0]
        X_rot[1] += self.x[1]

        # print(X_rot)
        # return X_rot.astype(np.int32)
        # nnn = [(x[0], x[1]) for x in X_rot.astype(np.int32).T]
        # print("nnn",nnn)
        return [(x[0], x[1]) for x in X_rot.astype(np.int32).T]

    # !!! POMEMBNO:
    # check if detection d is within this detection's search region (seems to work fine)
    def isWithin(self, d: Detection, s1=S1, s2=S2):
        rot_mat = self.getRotationMatrix() # get rotation matrix to rotate detection (easier calculation)
        v = d.x - self.x # representation relative to ellipsis center (da se prav obrne)
        v = np.dot(rot_mat.T, v)
        
        # check if within (enacba elipse):
        # x,y = v[0], v[1] #float
        x,y = int(v[0]), int(v[1]) #int (bolj clanky)
        return ((x*x)/(s1*s1) + (y*y)/(s2*s2) <= 1)

    # !!! POMEMBNO:
    # probability density function (bivariate normal distribution) (seems to work fine)
    def getProb(self, d: Detection, s1=S1, s2=S2) -> float:
        # cov_mat = np.array([[s1,0.0],[0.0,s2]]).astype(np.float32)
        inv_cov_mat = np.array([[1.0/(3*s1),0.0],[0.0,1.0/(3*s2)]])
        
        rot_mat = self.getRotationMatrix()
        v = d.x - self.x # x - mu
        v = np.dot(rot_mat.T, v) #un-rotate, so that it can be evaluated over un-rotated distribution

        pi_2, cov_mat_det = 2*np.pi , 9*s1*s2
        # return (1.0 / (np.sqrt( pi_2*pi_2 * cov_mat_det))) * np.exp(-0.5* np.dot( np.dot(v, inv_cov_mat), v)) # probability
        return np.exp(-0.5* np.dot( np.dot(v, inv_cov_mat), v)) # score (unscaled prob) (za vizualizacijo)
    # ----------------------------------------------------------

    def __str__(self):
        return "d{x="+str(self.x)+",v="+str(self.v)+",t="+str(self.t)+"}"
    
    def __repr__(self):
        return self.__str__()


# detection space
class DetectionSpace:
    def __init__(self, n, D: list = None):
        self.n = n # map size (dimensions)
        self.map = np.zeros((n,n)).astype(np.float32) # init empty map
        self.window_name = "detection space"
        
        self.D = []
        if D is not None:
            self.D.append(D) # append detections at time t = 0
    

    def showSpace(self):
        if(self.D):
            for d in self.D:
                detections2map(d, self.map)
        # print(map)
        cv.imshow(self.window_name, self.map)
        while cv.getWindowProperty(self.window_name, cv.WND_PROP_VISIBLE) >= 1:
            cv.waitKey(1)
        cv.destroyAllWindows()
    
    def estimateNext(self):
        self.D.append([d.estimateNext() for d in self.D[-1]])
    
    def estimateNext_t(self, t=1):
        self.D.append([d.estimateNext(t) for d in self.D[-1]])
    
    # !!! POMEMBNO [TODO - fix/adjust]
    # estimates next point in trajectory given collected detections and prediction (x_t -> x_t+1)
    def estimateNext2(self, d: Detection, dt=1) -> Detection:
        # 1. predict next point from current detection
        x_p = d.estimateNext(dt) # oftype Detection
        
        # 2. collect detections around prediction:
        t_i1 = x_p.t # already contains new time

        # search among detections in next time moment (next frame, that is (whichever, usually immediate successor (dt = 1)))
        next_detections = [] # store 'em in list
        next_detections_probs = [] # weights: sampled from distribution (bivariate normal dist, see Detection.getProb())
        
        if(t_i1 < len(self.D)): # check only if has detections in this layer
            for d_i in self.D[t_i1]:
                if(x_p.isWithin(d_i)):
                    next_detections.append(d_i) # store detections, for now
                    next_detections_probs.append(x_p.getProb(d_i)) # get probability score from nearby point
        
        # change detection list into coordinate list:
        n_det = len(next_detections) # number of detections (i)
        next_detections_X = np.zeros((n_det, 2))
        for i in range(n_det):
            next_detections_X[i] = next_detections[i].x
        print(next_detections_X)
        print(next_detections_probs)

        # 3. compute weighted mean (prediction + all detections) to determine actual next point
        
        # temporal discount, as used in paper [pami, leibe et al. ...] = e^-lambda
        L = 0 # lambda: temporal discount ([TODO] - un-hardcode) (should be large)
        p_tempDisc = np.exp(-L)

        # calculate normalization factor Z (sum of all weights):
        Z = np.sum(next_detections_probs)+p_tempDisc
        print("Z: ",Z)

        x_t1 = np.array((1/Z) * ( p_tempDisc * x_p.x + np.dot(next_detections_probs, next_detections_X) )).astype(np.int32)
        print("x(t+1): ",x_t1)

        # 4. also estimate velocity, angle (based on x_t+1: no need to calculate weights, estimates again, as they are the same)
        # velocity: v_t1 = sqrt( dolzina vektorja (x_t1 - x_t) )
        x_dif = d.x-x_t1
        x_dif = x_dif/dt # we do that here, velocity is then simply it's length
        v_t1 = np.sqrt(np.dot(x_dif,x_dif))

        # theta:
        # calculate relative to unit base vector x_i
        # base_x = np.array([1.0,0.0]) # -> not needed, see notes
        theta_t1 = d.theta
        if(v_t1 != 0): # only if it has speed this is relevant
            cos_theta = x_dif[0] / v_t1  # this is it, just trust me bro
            theta_t1 = np.arccos(cos_theta) # co-domain is only from 0-pi, not a problem, because ellipse is symmetrical (so essentialy v ~ -v)
        if(x_dif[1] < 0): # same angle is computed for both sides, because we only compare magnitude, so correction is needed in some cases
            theta_t1 = -theta_t1
        
        d_t1 = Detection(x_t1,v_t1,t_i1,theta_t1) # next detection, it should probably be something else
        # return x_t1
        return d_t1
        
    # might be needed to move it elsewhere
    def drawDsearchRegionAroundLastDetections(self):
        for d in self.D[-1]:
            # bounds = d.getSearchRegionBounds()
            bounds = d.getSearchRegionBounds2()
            coords2map(bounds, self.map)
    
    def drawDsearchRegionAroundDetection(self, D: Detection, a=None, b=None, theta=None):
        # bounds = D.getSearchRegionBounds()
        s1, s2 = 20,20
        if((b is not None) and (a is not None)):
            s1,s2 = a,b
        bounds = D.getSearchRegionBounds2(s1,s2, theta)
        coords2map(bounds, self.map)
    
    # method draws x instead of .
    def drawX(self, c:list, brightness=0.5) -> None:
        x_shape = np.array([
            [-3,-3],
            [-2,-2],
            [-1,-1],
            [0,0],
            [1,1],
            [2,2],
            [3,3],
            [-3,3],
            [-2,2],
            [-1,1],
            [1,-1],
            [2,-2],
            [3,-3],
        ])
        x_shape = x_shape + c # add origin
        p_list = [(x[0], x[1]) for x in x_shape]
        coords2map(p_list, self.map, brightness=brightness)

    
    # should be used for visualization only, is slow (O( (2*max(S1, S2)) ^2))
    def drawProbDistAroundDetection(self, D: Detection):
        # draw probability distribution function within detection area:
        s1s2 = np.max([S1,S2])
        for i in range(D.x[0]-s1s2,D.x[1]+s1s2,1):
            for j in range(D.x[0]-s1s2,D.x[1]+s1s2,1):
                detection_ij = Detection((i,j),0,0,0)
                # print((i,j))
                if(D.isWithin(detection_ij)):
                    prob = D.getProb(detection_ij)
                    self.map[i][j] = prob
                    # print(prob)
                    # print("is within")
    
    # POMEMBNO!!
    def findNextDetections(self, D: Detection, r=R):
        # print("method findNextDetections gets called")
        # 2. estimate search region in next timestamp
        d_est = D.estimateNext()
        print("d_est: ",d_est)
        print("D: ",D)
        # 3. collect all detections within search region (use simple l2 distance, for now)
        detections = self.D[d_est.t]
        for d in detections:
            diff = d.x-d_est.x
            l2_dist = np.sqrt(np.dot(diff, diff))
            print(l2_dist , end=", ") # flush=True -> da se izpise
            if(l2_dist <= R):
                # highlight detection
                self.map[d.x[0]][d.x[1]] = 5.0 # ne deluje, ker se vrednost na novo prepise
                # self.drawDsearchRegionAroundDetection(d)
        print()
    
    # POMEMBNO!!
    # method takes the trajectory and builds it [TODO]
    def buildTrajectory(self, t: Trajectory):
        # 1. detection
        x_t = t.X[-1] # current point in trajectory (last in array)


        # 2. look for next detections (current v, theta) (calculate estimation, look for detections inside its region)
        self.drawDsearchRegionAroundDetection(x_t, 20, 20, 0)
        # self.findNextDetections()
        

        # 3. estimate next detection: weighted mean of detections
        # 4. add calculated estimate to trajectory
        # repeat loop
        
        


class Trajectory:
    def __init__(self, d0):
        self.v = 0 # initial velocity is 0
        self.theta = 0 # theta is also 0
        self.X = [d0] # trajectory points (detections)
        self.holes = 0 # counter to count how many trajectory points have been added considering only estimate of next detection

    def estimateNext(self, dt = 1) -> Detection:
        x = self.X[-1]
        x_t1 = x[0] + int(dt*self.v*math.cos(self.theta))
        y_t1 = x[1] + int(dt*self.v*math.sin(self.theta))
        return Detection((x_t1, y_t1), self.v, self.t+dt, self.theta)
        
                
        

# helper methods:
# takes list of tuples, returns list of detections
def coords2detect(X):
    return [Detection(x) for x in X]

def detections2map(D, map):
    n = np.shape(map)[0]
    t_now = len(D) # which time instant is it
    for d in D:
        x, y = d.x
        # out of bounds is possible, this is the easiest quick fix, should probably be made different
        x = x%n
        y = y%n
        
        # map[x,y]= 1.0 - (1.0 / (1.0+(d.t/20.0)))*0.9 # more correct, as more recent detections should be brighter
        map[x,y]= 1.0
        # map[x,y]= (1.0 / (1.0+(d.t/8))) # I like this more, but is not correct because of ^

def coords2map(X, map, brightness=0.5):
    n = np.shape(map)[0]
    for point in X:
        x, y = point
        x = x%n
        y = y%n
        map[x,y]= brightness



def random_coords(n):
    x = int(random.random()*n)
    y = int(random.random()*n)
    return (x,y)

# like random_coords, but generates all at once and can use seed
def random_matrix(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.random(shape)

def n_random_coords(n, N=1, seed=None):
    X = (random_matrix((N,2), seed)*n).astype(np.int32).T # generate random x and y coordinates
    print(X)
    return list(zip(X[0],X[1]))


# -------------------------




# main:
def main():
    n = 500 # canvas size
    seed = 42
    
    D = coords2det2(t1)
    
    # create detection space object:
    dspace = DetectionSpace(n)
    dspace.D = D

    d0 = D[0][0]
    print(d0)

    th1 = Trajectory(d0)
    dspace.buildTrajectory(th1)
    


    dspace.showSpace()

# -----------

if __name__ == "__main__":
    main()


# def main():
#     n = 500 # canvas size
#     seed = 42
#     show_detections = False
    

#     # some detections:
    
#     # coordinates:
#     # X = [random_coords(n) for i in range(10)]
#     X = n_random_coords(n, 10, seed)
#     print(X)

#     # detections:
#     D = coords2detect(X)
#     # add some velocity and orientation to detections:
#     v=4
#     n_D = len(D)
#     n_theta = random_matrix((n_D,),seed)*math.pi*2.0
    
#     for i in range(n_D):
#         d = D[i]
#         d.v=v
#         d.theta = n_theta[i]

    
#     if show_detections:
#         print(D)
    
#     # create detection space object:

#     dspace = DetectionSpace(n, D)
#     for i in range(30):
#         dspace.estimateNext_t(1)
#     dspace.drawDsearchRegionAroundLastDetections()

#     # show detection space:
#     # print(dspace.D)


#     # add some detections around estimate (257, 463):
#     detections_around_coords = [(259, 464), (261, 461), (255, 468)]
#     detections_around = coords2detect(detections_around_coords)
#     for d in detections_around:
#         d.t = 24
#         dspace.D[24].append(d)
#     # detections2map

#     # take 1 detection from map:
#     detection = dspace.D[23][5]
#     print(detection)
#     # dspace.drawDsearchRegionAroundDetection(detection)
#     dspace.findNextDetections(detection)
    

# new detection
#     new_d = Detection((150,200),0,0,0)
#     new_d1 = Detection((-108+250,-26+250),0,0,0)

#     D.append(new_d) # add to space
#     D.append(new_d1) # add to space new_d1

#     # d0.theta = (np.pi/180)*192
#     dspace.drawDsearchRegionAroundDetection(d0)
#     d0.theta = (np.pi/180)*13

#     dspace.drawDsearchRegionAroundDetection(d0)
#     # dspace.drawDsearchRegionAroundDetection(new_d)

#     # some test prints:
#     print(d0.isWithin(new_d))

#     dspace.showSpace()


# probability distribution visualization testing
# def main():
#     n = 500 # canvas size
#     seed = 42
#     show_detections = False
    
#     # some detections:
    
#     # coordinates:
#     # X = [random_coords(n) for i in range(10)]
#     X = [(250,250)]
#     # print(X)

#     # detections:
#     D = coords2detect(X)
#     d0 = D[0]
#     # add some velocity and orientation to detections:
#     v=4
#     n_D = len(D)
#     n_theta = random_matrix((n_D,),seed)*math.pi*2.0
    
#     for i in range(n_D):
#         d = D[i]
#         d.v=v
#         # d.theta = n_theta[i]
#         d.theta = (np.pi/180.0)* 0.0

#     if show_detections:
#         print(D)
    
#     # create detection space object:
#     dspace = DetectionSpace(n, D)

#     dspace.drawDsearchRegionAroundDetection(d0)

#     # draw probability distribution function within detection area:
#     dspace.drawProbDistAroundDetection(d0)

#     dspace.showSpace()

# def main():
#     n = 500 # canvas size
#     seed = 42
    
#     d0 = Detection((250,250), -4, 0, (np.pi/180.0)*0)
#     d1_1 = Detection((240,240), 0, 1, 0)
#     # d1_2 = Detection((250,240), 0, 1, 0)
#     # d1_3 = Detection((250,220), 0, 1, 0)
#     # d1_4 = Detection((210,140), 0, 1, 0)
#     # d1_5 = Detection((270,200), 0, 1, 0)
#     D0 = [d0]
    
#     # create detection space object:
#     dspace = DetectionSpace(n, D0)

#     # add some more detections to space (at time t=1)
#     # D1 = [d1_1,d1_2,d1_3,d1_4,d1_5]
#     # D1 = [d1_1]
#     # dspace.D.append(D1)

#     # print("is within?")
#     # for d in D1:
#     #     print(d,d0.isWithin(d))
#     # print("---")

#     # dspace.drawProbDistAroundDetection(d0)

#     d1 = dspace.estimateNext2(d0)
#     dspace.drawX(d1.x)
#     dspace.drawX(d0.x, 0.1)

#     dspace.drawDsearchRegionAroundDetection(d0)
#     dspace.drawDsearchRegionAroundDetection(d1)

    
#     dspace.showSpace()