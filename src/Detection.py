from __future__ import annotations # da delajo tut type hint-i znotraj istega class-a
import numpy as np

# detection class; if becomes too complicated, move it to other file
class Detection:
    Did = 0 # private static int
    # constructor: position, timestamp
    def __init__(self, x, t=0, bb=None, visual_feat=None, flow_vector=None):
        self.id = Detection.Did
        Detection.Did += 1
        self.x = np.array(x)
        self.bb = bb # bounding box [x, y, h, w]
        self.visual_feat = visual_feat # visual features
        self.flow_vector, self.has_flow_vector = flow_vector, True # optical flow vector
        self.t = t
        self.color = None # use different color

        # default values if not provided through constructor:
        if(bb is None): # ... if provided
            self.bb = (0,0,0,0)
        if(flow_vector is None):
            self.flow_vector = np.array([0,0])
            self.has_flow_vector = False
    

    # equals: compares detections based on coordinates and time
    def equalsCoordsTime(self, d2):
        coordsEq = (self.x[0] == d2.x[0]) and (self.x[1] == d2.x[1])
        timeEq = (self.t == d2.t)
        return (coordsEq and timeEq)
        # return (coordsEq)

    def __str__(self):
        # return "d{x="+str(self.x)+",t="+str(self.t)+"}"
        return "d{%d, x=%s,bb=(%.2f,%.2f),t=%d}"%(self.id, str(self.x),self.bb[0], self.bb[1], self.t)
    
    def __repr__(self):
        return self.__str__()

# detection of trajectory (also has theta, velocity)
class TDet(Detection):
    def __init__(self, x, t=0, v=0, theta=0):
        super().__init__(x, t)
        # motion model
        self.v=v
        self.theta = theta
        self.k = 1.4 # experimental: used in DetectionSpace.isWithin (if trajectory is stalled, then this coefficient is used to extend search region (for occlusions))
    
    # # FUJ
    # def fromDet(self, d:Detection):
    #     self.bb = d.bb
    #     self.hasHist = d.hasHist
    #     self.color_hist = d.color_hist
        
    
    def __str__(self):
        return "td{x="+str(self.x)+",t="+str(self.t)+"}"


def TDet_from_Detection(d: Detection) -> TDet:
    td = TDet(d.x, d.t)
    
    # copy everything
    td.bb = d.bb
    td.visual_feat, td.flow_vector, td.has_flow_vector, td.color = d.visual_feat, d.flow_vector, d.has_flow_vector, d.color
    return td

### detection-specific functions: ###
### ----------------------------- ###
