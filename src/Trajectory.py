from __future__ import annotations # da delajo tut type hint-i znotraj istega class-a
import numpy as np
from Detection import Detection, TDet
import math

# to avoid cyclic import (detection space imports trajectory, trajectory imports detection space)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from DetectionSpace import DetectionSpace

class Trajectory:
    # assumptions:
    # Hi_ti -> if there is only one detection inside of event cone when building trajectory, then this is used as Hi_ti
    # if there are more, then the one with maximum probability (according to Dt (is defined by trajectory point)) is selected
    # (could probably also use weighted average / average / random / build hypotheses for all of them (hard??)) --> discussion is needed
    Tid = 0 # apparently static? (want private static, maybe should be _Tid)

    def __init__(self, d0: Detection, detectionSpace: DetectionSpace):
        self.id = Trajectory.Tid # unique id of the trajectory
        self.origin = TDet(d0.x,d0.t,0,0)
        self.detectionSpace = detectionSpace # pointer to detection space in which trajectory lives (has detections)
        Trajectory.Tid = Trajectory.Tid+1
        self.X = [ ] # trajectory points ("trajectory" detections)
        self.holes = 0 # counter to count how many trajectory points have been added considering only estimate of next detection
        self.S = 0 # score/support of the trajectory
        self.D = set([d0]) # all detections in the trajectory (set: to determine intersecting detections with other trajectories to calculate penalty)

    # method estimates next position based on current position, velocity and orientation (theta)
    def estimateNext(self, td_current: TDet, dt: int = 1) -> TDet:
        x = td_current.x
        t = td_current.t
        x_t1 = x[0] + int(dt*td_current.v*math.cos(td_current.theta))
        y_t1 = x[1] + int(dt*td_current.v*math.sin(td_current.theta))
        return TDet((x_t1, y_t1), t+dt, td_current.v, td_current.theta)
    
    # method builds trajectory and returns list of trajectory detections (and holes)
    # [TODO] - is it really necessary to predict positions in the future? maybe only in the past?
    # HOLES: allow up to n holes, if no detections after n frames, discontinue (also delete all predictions)
    def connectPoints(self, td_orig: TDet, dt=1) -> list:
        # print("this is connect points:")
        t_n = len(self.detectionSpace.D)
        n_holes_total = 0 # skupno stevilo lukenj
        n_holes = 0 # stevilo zaporednih lukenj
        n_holes_max = 5 # najvecje steivlo zaporednih lukenj

        td_connected = [] # list of connected trajectory detections
        
        td_current = td_orig
        for _ in range(t_n):
            # 1. detection
            # td_current

            # 2. look for next detections (current v, theta) (calculate estimation, look for detections inside its region)
            # 2.1 estimate next point:
            td_pred = self.estimateNext(td_current, dt)
            # print(td_pred.v)
            # 2.2 find next detections:
            next_dets, next_probs = self.detectionSpace.collectWithin(td_pred.t, td_current) # collect in next frame (t+1)

            # print("current: ",td_current, "next dets: ", next_dets) # debug stuff


            # add detections (objects, not just coords) to trajectory detections set (for intersections with other trajectories)
            self.D.update(next_dets)

            if(len(next_dets) == 0): # either hole or discontinued
                # print("lukna je naprej od: ",td_current.x)
                n_holes += 1
                n_holes_total += 1
                # print("empty")
                if(n_holes >= n_holes_max):
                    n_holes = n_holes-1
                    break
            else:
                n_holes = 0

            # 3. estimate next detection: weighted mean of detections
            # d_next, _ = self.estimateNext(t)
            td_next, _ = self.detectionSpace.estimateNext2(td_current, td_pred, next_dets, next_probs, dt=dt)
            
            # 4. add calculated estimate to list
            td_connected.append(td_next)
            td_current = td_next # update current point (i = i+1, p = p.next() <- neki tazga)
            # repeat loop
        
        # prune predicted points in the future
        if(len(td_connected) >= n_holes):
            self.holes = n_holes_total - n_holes
            for _ in range(n_holes):
                td_connected.pop()
        return td_connected
    
    # metod builds trajectory from origin point
    def build(self):
        # [previous time frames] [self.origin] [next time frames]
        t_prev = self.connectPoints(self.origin, -1) # previous detections
        # print(t_prev)
        t_next = self.connectPoints(self.origin, 1) # next detections

        for i in range(len(t_prev)-1, -1, -1):
            self.X.append(t_prev[i])
        
        self.X.append(self.origin)

        for i in range(len(t_next)):
            self.X.append(t_next[i])
        # print(self.X)
        

    # method draws trajectory to detection space
    def drawToSpace(self):
        x_brightness = 1.0
        if(len(self.X) > 0): # if has one
            self.detectionSpace.drawX(self.X[0].x, x_brightness)
        if(len(self.X) > 1): # if has many
            for i in range(len(self.X)-1):
                xi = self.X[i].x
                xi1 = self.X[i+1].x
                self.detectionSpace.drawLine(xi,xi1,0.5)
                # self.detectionSpace.drawDsearchRegionAroundDetection(xi1)
            self.detectionSpace.drawX(self.X[-1].x, x_brightness)
    
    # POMEMBNO!! MOGOCE DELA NAROBE (klicemo iz build qbp matrix)
    # method calculates CUMULATIVE "error" of ALL detections around estimated trajectory point in time t (is this ok?)
    # [TODO] - is this ok? might not be
    # FUJ! - choose one of the following methods to calculate g and use only one everywhere, as they do essentially same thing
    def g(self, t: int):
        
        # 1. find detections around trajectory point at time t
        t_relative = t - self.X[0].t # need relative time, because trajectories might not start at time 0
        td_t = self.X[t_relative]
        dets, probs = self.detectionSpace.collectWithin(td_t.t, td_t)
        # print("current: ", td_t, "around: ", dets, probs)

        # 2. get each detection's probability in image (we get that from detector)
        p_hi = 1

        # 3. calculate g:
        #   option 1:  g = p*(Hk,tk|Itk) + SUM(a)[   log(p(Ha,tk|H))   ] -> cumulative log error of all detections around trajectory point at t
        result = p_hi 
        if(len(dets) == 0):
            return None # we should not add anything to this, this case should be handled
        
        result = result + np.sum( np.log(probs) )
        return result
    
    ##### how to use g:
    # tr is current trajectory
    # for td in tr.X:
    #     g_k = tr.g(td.t) # throws indexOutOfBounds!  (!! method collects all from its neighbourhood, and not just ones that were used to build trajectory)
    #     if(g_k is None):
    #         continue # do not add anything if has no detections (see method Trajectory.g(...))
    #     S_err = S_err + ((1.0 - e2) + e2*g_k)
    #####
    

    # method calculates g_k of detections in list (used for intersecting detections AND for detections used to build trajectory) (seems to work fine for now)
    def g_k(self, dets: list[Detection]):
        probs = self.getDetProbs(dets)
        # print(dets,probs)
        p_hi = 1
        result = p_hi 
        if(len(dets) == 0):
            return None # this case should be handled
        result = result + np.sum( np.log(probs) )
        return result
    
    
    # method gets probabilities of detections in list around corresponding trajectory point (seems to work fine for now)
    def getDetProbs(self, det_list: list[Detection]):
        t_off = self.X[0].t # time of first detection, is used to calculate relative index of point in trajectory
        probs = []
        
        for d in det_list:
            td = self.X[d.t - t_off]
            d_prob = self.detectionSpace.getProb2(d,td)
            probs.append(d_prob)
        return probs
    
    # POMEMBNO!!

    def __str__(self):
        return "{t"+str(self.id)+", len="+str(len(self.X))+"}"

    def __repr__(self):
        return self.__str__()