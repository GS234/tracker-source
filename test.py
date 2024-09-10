import numpy as np
import random
import math
import cv2 as cv
from __future__ import annotations # da delajo tut type hint-i znotraj istega class-a

# global vars:
R = 10



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
    
    # calculate next detection estimate from current info
    # POMEMBNA ZADEVA
    def estimateNext(self, dt = 1):
        x_t1 = self.x[0] + int(dt*self.v*math.cos(self.theta))
        y_t1 = self.x[1] + int(dt*self.v*math.sin(self.theta))
        return Detection((x_t1, y_t1), self.v, self.t+dt, self.theta)
    
    # calculate search region bound coordinates (for visualization)
    # TUDI TO JE POMEMBNA ZADEVA
    def getSearchRegionBounds(self, r=R):
        phi_ = np.arange(0, 2*math.pi, math.pi/20.0) # angles on a ring of bounding points
        x_ = np.cos(phi_)*r + self.x[0]
        y_ = np.sin(phi_)*r + self.x[1]
        return [(int(x_[i]), int(y_[i])) for i in range(len(phi_))] # return list of tuples of bounding points around detection
    
    def getSearchRegionBounds2(self, s1=10, s2=5, n=30):
        # rotation matrix
        rot_mat = self.getRotationMatrix()
        
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

    # TODO: preglej delovanje spodnjih funkcij (kr na eni tocki najbols)
    # POMEMBNO: mogoce ne dela, poglej!! [TODO]
    # check if detection d is within this detection's search region
    def isWithin(self, d: Detection, s1=10, s2=5):
        rot_mat = self.getRotationMatrix() # get rotation matrix to rotate detection (easier calculation)
        v = d.x - self.x # representation relative to ellipsis center (da se prav obrne)
        v = np.dot(rot_mat, v)

        # check if within (enacba elipse):
        x,y = v[0], v[1]
        return ((x*x)/(s1*s1) + (y*y)/(s2*s2) <= 1)

    # POMEMBNO [TODO] (mogoce ne deluje se, FIX)
    # probability density function (bivariate normal distribution)
    def getProb(self, d: Detection, s1=10, s2=5):
        cov_mat = np.array([[s1,0],[0,s2]])
        x_dif = d.x - self.x # x - mu
        return (1/(np.sqrt( ((2*np.pi)**2)) * s1*s2)) * np.exp(-0.5* np.dot( np.dot(x_dif, 1/cov_mat), x_dif))


    # ----------------------------------------------------------

    def __str__(self):
        return "d{x="+str(self.x)+",v="+str(self.v)+",t="+str(self.t)+"}"
    
    def __repr__(self):
        return self.__str__()


# detection space
class DetectionSpace:
    def __init__(self, n, D = None):
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
    

    # might be needed to move it elsewhere
    def drawDsearchRegionAroundLastDetections(self):
        for d in self.D[-1]:
            # bounds = d.getSearchRegionBounds()
            bounds = d.getSearchRegionBounds2()
            coords2map(bounds, self.map)
    
    def drawDsearchRegionAroundDetection(self, D: Detection):
        # bounds = D.getSearchRegionBounds()
        bounds = D.getSearchRegionBounds2()
        coords2map(bounds, self.map)
    
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
        
        map[x,y]= 1.0 - (1.0 / (1.0+(d.t/20.0)))*0.9 # more correct, as more recent detections should be brighter
        # map[x,y]= (1.0 / (1.0+(d.t/8))) # I like this more, but is not correct because of ^

def coords2map(X, map):
    n = np.shape(map)[0]
    for point in X:
        x, y = point
        x = x%n
        y = y%n
        map[x,y]= 1.0/2.0 



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
    show_detections = False
    

    # some detections:
    
    # coordinates:
    # X = [random_coords(n) for i in range(10)]
    X = n_random_coords(n, 10, seed)
    print(X)

    # detections:
    D = coords2detect(X)
    # add some velocity and orientation to detections:
    v=4
    n_D = len(D)
    n_theta = random_matrix((n_D,),seed)*math.pi*2.0
    
    for i in range(n_D):
        d = D[i]
        d.v=v
        d.theta = n_theta[i]

    
    if show_detections:
        print(D)
    
    # create detection space object:

    dspace = DetectionSpace(n, D)
    for i in range(30):
        dspace.estimateNext_t(1)
    dspace.drawDsearchRegionAroundLastDetections()

    # show detection space:
    # print(dspace.D)


    # add some detections around estimate (257, 463):
    detections_around_coords = [(259, 464), (261, 461), (255, 468)]
    detections_around = coords2detect(detections_around_coords)
    for d in detections_around:
        d.t = 24
        dspace.D[24].append(d)
    # detections2map


    # take 1 detection from map:
    detection = dspace.D[23][5]
    print(detection)
    # dspace.drawDsearchRegionAroundDetection(detection)
    dspace.findNextDetections(detection)


    
    dspace.showSpace()


# -----------

if __name__ == "__main__":
    main()
