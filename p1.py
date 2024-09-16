from __future__ import annotations # da delajo tut type hint-i znotraj istega class-a
import numpy as np
import random
import math
import cv2 as cv

# global vars:
S1, S2 = 30,30 # default s1,s2

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

t3 = [
    [(275, 114)],
    [(265, 135), (266, 135), (266, 136), (266, 137)],
    [(273, 152), (274, 152)],
    [(282, 170)],
    [(283, 194)],
    [(291, 214), (292, 214)],
    [(306, 224)],
    [(313, 236), (314, 237), (314, 238)],
    [(314, 254), (314, 255), (314, 256), (314, 257)],
    [(304, 277), (304, 278), (303, 279), (301, 280), (302, 280), (300, 281), (304, 281), (305, 289)],
    [(287, 296)],
    [(277, 309), (277, 310)],
    [(286, 319), (287, 320), (288, 320), (288, 321), (289, 321)],
    [(306, 330), (306, 331), (306, 332)],
    [(297, 349), (298, 349)],
    [(303, 362)],
    [(289, 374), (290, 374), (287, 375), (288, 375), (286, 376)]
]



t2 = [
    [(250,249)],
    # [(240,240),(260,240)]
    [(240,240),(260,240),(253,240),(256,240)]
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
    # constructor: position, velocity, timestamp (position, velocity are only used for visualization and legacy reasons)
    def __init__(self, x, v = 0, t=0, theta=0):
        self.x = np.array(x)
        self.v = v #initial velocity is 0
        self.t = t #time stamp
        self.theta = theta

    def __str__(self):
        return "d{x="+str(self.x)+",v="+str(self.v)+",t="+str(self.t)+",theta="+str(self.theta)+"}"
    
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
    
    # detection-specific methods (MIGHT NEED TO MOVE THEM ELSEWHERE [TODO: consider doing so!])
    # calculates rotation matrix based on detection orientation (calculate each time, as theta might change)
    def getRotationMatrix(self, theta=0):
        return np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
    
    # calculate search region bound coordinates (ellipse)
    def getSearchRegionBounds(self, x, a=S1, b=S2, theta=0, n=50):
        # rotation matrix
        rot_mat = self.getRotationMatrix(theta)
        full_circle = np.arange(0, 2*np.pi, 2*np.pi/n)
        X = np.array(
            [
                a*np.sin(full_circle),
                b*np.cos(full_circle)
            ])
        X_rot = np.dot(rot_mat, X) # rotate it!
        X_rot[0] += x[0]
        X_rot[1] += x[1]
        return [(b[0], b[1]) for b in X_rot.astype(np.int32).T]

    # !!! POMEMBNO:
    # check if detection d is within this detection's search region (seems to work fine)
    def isWithin(self, x, x_ref, a=S1, b=S2):
        rot_mat = self.getRotationMatrix() # get rotation matrix to rotate detection (easier calculation)
        v = x - x_ref # representation relative to ellipsis center (da se prav obrne)
        v = np.dot(rot_mat.T, v)

        # print("--")
        # print(rot_mat,v)
        # print(a,b)
        # print("--")
        
        # check if within (enacba elipse):
        xx,yy = int(v[0]), int(v[1]) #int (bolj clanky kot float)
        return ((xx*xx)/(a*a) + (yy*yy)/(b*b) <= 1)

    # !!! POMEMBNO:
    # probability density function (bivariate normal distribution) (seems to work fine)
    def getProb(self, x, x_ref, a=S1, b=S2) -> float:
        # cov_mat = np.array([[s1,0.0],[0.0,s2]]).astype(np.float32)
        inv_cov_mat = np.array([[1.0/(3*a),0.0],[0.0,1.0/(3*b)]])
        
        rot_mat = self.getRotationMatrix()
        v = x - x_ref # x - mu
        v = np.dot(rot_mat.T, v) #un-rotate, so that it can be evaluated over un-rotated distribution

        pi_2, cov_mat_det = 2*np.pi , 9*a*b
        # return (1.0 / (np.sqrt( pi_2*pi_2 * cov_mat_det))) * np.exp(-0.5* np.dot( np.dot(v, inv_cov_mat), v)) # probability
        return np.exp(-0.5* np.dot( np.dot(v, inv_cov_mat), v)) # score (unscaled prob) (za vizualizacijo)
    # ----------------------------------------------------------------------------------------
    

    def showSpace(self):
        if(self.D):
            for d in self.D:
                detections2map(d, self.map)
        # print(map)
        cv.imshow(self.window_name, self.map)
        while cv.getWindowProperty(self.window_name, cv.WND_PROP_VISIBLE) >= 1:
            cv.waitKey(1)
        cv.destroyAllWindows()
    
    # collects detections within search region at time t
    # returns: coordinates, probabilities
    def collectWithin(self, t, d: Detection):
        # search among detections in next time moment (next frame, that is (whichever, usually immediate successor (dt = 1)))
        next_detections = [] # store 'em in list
        next_detections_probs = [] # weights: sampled from distribution (bivariate normal dist, see Detection.getProb())
        
        if(t < len(self.D)): # check only if has detections in this layer
            for d_i in self.D[t]:
                if(self.isWithin(d_i.x, d.x)):
                    next_detections.append(d_i) # store detections, for now
                    next_detections_probs.append(self.getProb(d_i.x, d.x)) # get probability score from nearby point
        
        n_det = len(next_detections) # number of detections (i)
        next_detections_X = np.zeros((n_det, 2))
        for i in range(n_det):
            next_detections_X[i] = next_detections[i].x
        return next_detections_X, next_detections_probs



    # !!! POMEMBNO [TODO - fix/adjust/modify/test]
    # estimates next point in trajectory given collected detections and prediction (x_t -> x_t+1)
    def estimateNext(self, t:Trajectory, dt=1) -> Detection:
        # 1. predict next point from current detection
        x_p = t.estimateNext(dt) # oftype Detection
        x_t = t.X[-1]
        # print("estimated: ", x_p)
        
        # 2. collect detections around prediction:
        t_i1 = x_p.t # already contains new time
        next_detections_X, next_detections_probs = self.collectWithin(t_i1, x_t)

        # 3. compute weighted mean (prediction + all detections) to determine actual next point
        # temporal discount, as used in paper [pami, leibe et al. ...] = e^-lambda
        L = 40 # lambda: temporal discount ([TODO] - un-hardcode) (should be large)
        p_tempDisc = np.exp(-L)

        # calculate normalization factor Z (sum of all weights):
        Z = np.sum(next_detections_probs)+p_tempDisc
        # print("Z: ",Z)

        x_t1 = np.array((1/Z) * ( p_tempDisc * x_p.x + np.dot(next_detections_probs, next_detections_X) )).astype(np.int32)
        # print(x_t1)
        # print("x(t+1): ",x_t1)

        # 4. also estimate velocity, angle (based on x_t+1: no need to calculate weights, estimates again, as they are the same)
        # velocity: v_t1 = sqrt( dolzina vektorja (x_t1 - x_t) )
        
        x_dif = x_t1 - x_t.x
        
        # print(x_t.x, x_t1, x_dif)
        x_dif = x_dif/dt # we do that here, velocity is then simply it's length
        v_t1 = np.sqrt(np.dot(x_dif,x_dif))

        # theta:
        # calculate relative to unit base vector x_i
        # base_x = np.array([1.0,0.0]) # -> not needed, see notes
        # print("theta: ",t.theta)
        theta_t1 = t.theta
        if(v_t1 != 0): # only if it has speed this is relevant
            cos_theta = x_dif[0] / v_t1  # this is it, just trust me bro
            theta_t1 = np.arccos(cos_theta) # co-domain is only from 0-pi, not a problem, because ellipse is symmetrical (so essentialy v ~ -v)
        if(x_dif[1] < 0): # same angle is computed for both sides, because we only compare magnitude, so correction is needed in some cases
            theta_t1 = -theta_t1
        
        d_t1 = Detection(x_t1,v_t1,t_i1,theta_t1) # next detection, it should probably be something else
        # return x_t1
        return (d_t1,x_p)
    
    # ----------------------------------
        
    
    def drawDsearchRegionAroundDetection(self, x, a=None, b=None, theta=None):
        # bounds = D.getSearchRegionBounds()
        s1, s2 = S1,S2
        if((b is not None) and (a is not None)):
            s1,s2 = a,b
        bounds = self.getSearchRegionBounds(x,s1,s2, theta)
        coords2map(bounds, self.map, overwrite=False)
    
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
    
    def drawLine(self, x1, x2, brightness=1):
        x1 = np.array(x1)
        x2 = np.array(x2)
        n = (x2 - x1) # normal from x1 to x2
        # print(n.reshape( (2,1) ))
        n_len = np.sqrt(np.dot(n.T,n))
        n = n/n_len
        n = n.reshape((2,1))
        
        values = np.arange(0, int(n_len), 0.1)
        values = values.reshape((1,len(values)))
        
        points = np.dot(n, values) + x1.reshape((2,1))
        p_list = [(int(x[0]), int(x[1])) for x in points.T]
        coords2map(p_list, self.map, brightness, overwrite=False)



    # should be used for visualization only, is slow (O( (2*max(S1, S2)) ^2))
    def drawProbDistAroundDetection(self, x):
        # draw probability distribution function within detection area:
        s1s2 = np.max([S1,S2])
        for i in range(x[0]-s1s2,x[1]+s1s2,1):
            for j in range(x[0]-s1s2,x[1]+s1s2,1):
                if(self.isWithin((i,j),x)):
                    prob = self.getProb((i,j),x)
                    self.map[i][j] = prob
    
    
    # [TODO - fix/finish]
    # POMEMBNO!!
    # method takes the trajectory and builds it [TODO]
    def buildTrajectory(self, t: Trajectory):
        t_n = len(self.D)
        
        for i in range(t_n):
            # 1. detection
            x_t = t.X[-1].x # current point in trajectory (last in array)

            # 2. look for next detections (current v, theta) (calculate estimation, look for detections inside its region)
            self.drawDsearchRegionAroundDetection(x_t, S1, S2, t.theta)
            # 3. estimate next detection: weighted mean of detections
            d_next,d_p = self.estimateNext(t)
            # self.drawX(d_next.x,1.0)
            self.drawLine(x_t, d_next.x, 0.4)
            # self.drawX(d_p.x, 0.8)
            # print(d_next)
            # print(d_next)

            # 4. add calculated estimate to trajectory
            t.X.append(d_next)
            t.v = d_next.v
            t.theta = d_next.theta
            # t.theta = 0
            # repeat loop
        

class Trajectory:
    def __init__(self, d0):
        self.v = 0 # initial velocity is 0
        self.theta = 0 # theta is also 0
        self.X = [d0] # trajectory points (detections)
        self.holes = 0 # counter to count how many trajectory points have been added considering only estimate of next detection

    def estimateNext(self, dt = 1) -> Detection:
        x = self.X[-1].x
        t = self.X[-1].t
        # print(self.X)
        # print(self.theta)
        # print(self.v)
        x_t1 = x[0] + int(dt*self.v*math.cos(self.theta))
        y_t1 = x[1] + int(dt*self.v*math.sin(self.theta))
        return Detection((x_t1, y_t1), self.v, t+dt, self.theta)
        
                
        

# helper functions:
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

def coords2map(X, map, brightness=0.5, overwrite=True):
    n = np.shape(map)[0]
    for point in X:
        x, y = point
        x = x%n
        y = y%n
        if(overwrite or map[x,y] == 0):
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
    
    # D = coords2det2(t1)
    D = coords2det2(t1)
    D2 = coords2det2(t3)

    offset = 5
    for i in range(len(D2)):
        for j in range(len(D2[i])):
            D2[i][j].t = i+offset # lval + assignment: mem = mov
            D[i+offset].append(D2[i][j])

    
    # create detection space object:
    dspace = DetectionSpace(n)
    dspace.D = D

    d0 = D[0][0]
    # print(d0)
    d0_1 = D2[0][0]
    th1 = Trajectory(d0)
    th2 = Trajectory(d0_1)
    dspace.buildTrajectory(th1)
    dspace.buildTrajectory(th2)

    # dspace.drawLine((10,10), (100,70))
    
    dspace.showSpace()

# -----------

if __name__ == "__main__":
    main()
