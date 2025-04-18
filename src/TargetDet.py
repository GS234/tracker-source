import numpy as np
from Detection import Detection
from Trajectory import Trajectory
from helper_func import *

class TargetDet(Detection):
    def __init__(self, d:Detection, tr:Trajectory = None):
        super().__init__(d.x, d.t, d.bb, d.visual_feat, d.flow_vector)
        self.visual_avg = None
        self.visual_n = 0
        self.color=[50,200,255]
        self.tr = tr # trajectory to which this target belongs
    
    def updateVisual(self, visual_feat):
        if(self.visual_avg is None):
            self.visual_avg = visual_feat
            self.visual_n = 1
        else:
            self.visual_avg = addToAvg(self.visual_avg, visual_feat, self.visual_n)
            self.visual_n = self.visual_n+1
    
    def setVisual(self, visual_feat):
        self.visual_avg = visual_feat
        self.visual_n = 1
    
    def updateVisualFromTrajectory(self, tr:Trajectory):
        if(self.visual_avg is None):
            self.visual_avg = tr.visual_avg
            self.visual_n = tr.visual_n
        else:
            self.visual_avg = add2Avg(self.visual_avg, tr.visual_avg, self.visual_n, tr.visual_n)
            self.visual_n = self.visual_n+tr.visual_n
    
    
    def getNearestTr(self, tr_list:list[Trajectory]):
        dist = 0
        nearest_t = None
        for tr in tr_list:
            this_dist = getDistBetweenDets(self, tr.origin)
            print(this_dist)
            if(nearest_t is None):
                nearest_t = tr
                dist = this_dist
            else:
                if(this_dist < dist):
                    nearest_t = tr
                    dist = this_dist
        return nearest_t
    
    def getNearestTrVisual(self, tr_list:list[Trajectory]):
        dist = 0
        nearest_t = None
        for tr in tr_list:
            this_dist = getTrVisualSimilarity(self.tr, tr)
            printTrWithStats(tr, add_to_end="dist: %d"%(this_dist))
            if(nearest_t is None):
                nearest_t = tr
                dist = this_dist
            else:
                if(this_dist < dist):
                    nearest_t = tr
                    dist = this_dist
        return nearest_t
    
    def setLastTDetProps(self):
        tr_det = self.tr.X[-1]

        # set things to that of last tr det:
        self.x = tr_det.x
        self.bb = tr_det.bb
        self.t = tr_det.t
        # and of tr
        self.visual_avg = self.tr.visual_avg
        self.visual_n = self.tr.visual_n

    
    def isFin(self):
        if(self.tr is None):
            print("[WARN] This target has no trajectory. Returning false.")
            return False
        return self.tr.term