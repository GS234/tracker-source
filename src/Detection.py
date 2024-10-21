from __future__ import annotations # da delajo tut type hint-i znotraj istega class-a
import numpy as np

# detection class; if becomes too complicated, move it to other file
class Detection:
    # constructor: position, timestamp
    def __init__(self, x, t=0, bb=None):
        self.x = np.array(x)
        self.bb = bb # bounding box is also there, ...
        if(bb is None): # ... if provided
            self.bb = (0,0,0,0)
        self.t = t

    def __str__(self):
        return "d{x="+str(self.x)+",t="+str(self.t)+"}"
    
    def __repr__(self):
        return self.__str__()

# detection of trajectory (also has theta, velocity)
class TDet(Detection):
    def __init__(self, x, t=0, v=0, theta=0):
        super().__init__(x, t)
        self.v=v
        self.theta = theta
    
    def __str__(self):
        return "td{x="+str(self.x)+",t="+str(self.t)+"}"

### detection-specific functions: ###
### ----------------------------- ###
