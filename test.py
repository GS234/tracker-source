import numpy as np
import random
import math
import cv2 as cv

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
            bounds = d.getSearchRegionBounds()
            coords2map(bounds, self.map)
    
    def drawDsearchRegionAroundDetection(self, D):
        bounds = D.getSearchRegionBounds()
        coords2map(bounds, self.map)
    
    # POMEMBNO!!
    def findNextDetections(self, D: Detection, r=R):
        print("method findNextDetections gets called")
        # 2. estimate search region in next timestamp
        d_est = D.estimateNext()
        print(d_est)
        print(D)
        # 3. collect all detections within search region (use simple l2 distance, for now)
        detections = self.D[d_est.t]
        for d in detections:
            diff = d.x-d_est.x
            l2_dist = np.sqrt(np.dot(diff, diff))
            # print(l2_dist , end=", ")
            if(l2_dist <= R):
                # highlight detection
                self.map[d.x[0]][d.x[1]] = 5.0 # ne deluje, ker se vrednost na novo prepise
                self.drawDsearchRegionAroundDetection(d)
                
        

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


# -------------------------




# main:
def main():
    n = 500 # canvas size
    show_detections = True
    

    # some detections:
    
    # coordinates:
    X = [random_coords(n) for i in range(10)]
    print(X)

    # detections:
    D = coords2detect(X)
    # add some velocity and orientation to detections:
    v=4
    for d in D:
        d.v=v
        d.theta = random.random()*math.pi*2.0

    
    if show_detections:
        print(D)
    
    # create detection space object:

    dspace = DetectionSpace(n, D)
    for i in range(30):
        dspace.estimateNext_t(1)
    # dspace.drawDsearchRegionAroundLastDetections()

    # show detection space:
    print(dspace.D)


    # take 1 detection from map:
    detection = dspace.D[23][5]
    print(detection)
    dspace.drawDsearchRegionAroundDetection(detection)
    dspace.findNextDetections(detection)
    
    dspace.showSpace()


# -----------

if __name__ == "__main__":
    main()
