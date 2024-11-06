from __future__ import annotations # da delajo tut type hint-i znotraj istega class-a
import numpy as np
from Detection import Detection, TDet
import math
from helper_func import detSet2map # helper functions

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

    def __init__(self, d0: Detection, detectionSpace: DetectionSpace, color: list=None):
        self.id = Trajectory.Tid # unique id of the trajectory
        
        self.color = (np.random.rand(3)*150).astype(np.uint8)+10 # color of trajectory (for visualization)
        self.color[2]=255
        if(color is not None):
            self.color = color
        Trajectory.Tid = Trajectory.Tid+1
        self.origin = TDet(d0.x,d0.t,0,0)
        self.detectionSpace = detectionSpace # pointer to detection space in which trajectory lives (has detections)
        self.X = [ ] # trajectory points ("trajectory" detections)
        self.holes = 0 # counter: how many trajectory points have been added considering only estimate of next detection
        self.holes_ref = 0 # holes reference: used to calculate relative number of holes (set to self.holes first, then calculate difference) (used in extend)
        
        self.not_selected_strike = 0 # number of times the trajectory has not been selected but is in hypothesis set
        self.disable_grow = False # flag: disable trajectory to grow

        self.S = 0 # score/support of the trajectory (IS SET/UPDATED IN DetectionSpace.buildQBPMatrix() METHOD)
        self.D = set([d0]) # all detections in the trajectory (set: to determine intersecting detections with other trajectories to calculate penalty)

    # method estimates next position based on current position, velocity and orientation (theta)
    def estimateNext(self, td_current: TDet, dt: int = 1) -> TDet:
        x = td_current.x
        t = td_current.t
        x_t1 = x[0] + int(dt*td_current.v*math.cos(td_current.theta))
        y_t1 = x[1] + int(dt*td_current.v*math.sin(td_current.theta))
        return TDet((x_t1, y_t1), t+dt, td_current.v, td_current.theta)
    
    # method builds trajectory and returns list of trajectory detections (and holes)
    # HOLES: allow up to n holes, if no detections after n frames, discontinue (also delete all predictions); also prune tails from both ends
    def connectPoints(self, td_orig: TDet, dt=1) -> list[TDet]:
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
            # next_dets, next_probs = self.detectionSpace.collectWithin(td_pred.t, td_current) # collect in next frame (t+1) around CURRENT TDET (shouldn't it be around predicted?)
            next_dets, next_probs = self.detectionSpace.collectWithin(td_pred.t, td_pred) # collect in next frame (t+1) around prediction

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
    def drawToSpace(self, color=None):
        if(color is None):
            color = self.color
        x_brightness = 1.0
        if(len(self.X) > 0): # if has one
            self.detectionSpace.drawX(self.X[0].x, x_brightness)
        if(len(self.X) > 1): # if has many
            for i in range(len(self.X)-1):
                xi = self.X[i].x
                xi1 = self.X[i+1].x
                self.detectionSpace.drawLine(xi,xi1,color)
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
    def g_k(self, dets: list[Detection]) -> float:
        probs = self.getDetProbs(dets)
        # print(dets,probs)
        p_hi = 1
        result = p_hi 
        if(len(dets) == 0):
            return None # this case should be handled
        result = result + np.sum( np.log(probs) )
        return result
    
    
    # method gets probabilities of detections in list around corresponding trajectory point (seems to work fine for now)
    def getDetProbs(self, det_list: list[Detection]) -> list[float]:
        t_off = self.X[0].t # time of first detection, is used to calculate relative index of point in trajectory
        probs = []
        
        for d in det_list:
            td = self.X[d.t - t_off]
            d_prob = self.detectionSpace.getProb2(d,td)
            probs.append(d_prob)
        return probs
    
    # POMEMBNO!!


    # method is used to extend existing trajectory to time t+1 (effectively: one step of connect)
    # [TODO] - time? need current time (for reference); further testing needed!
    def extend(self, debug=False, n_empty=-1):
        if(self.disable_grow):
            return
        # 1. find detections around last detection
        # 2. estimate, add to trajectory, ...
        dt = 1
    
        td_current: TDet = self.X[-1] # current
        td_pred: TDet = self.estimateNext(td_current, dt) # prediction
        
        
        # next_dets, next_probs = self.detectionSpace.collectWithin(td_pred.t, td_current) # collect in next frame (t+1) !! IS THIS CORRECT? !! (collect around current)
        next_dets, next_probs = self.detectionSpace.collectWithin(td_pred.t, td_pred) # collect around prediction

        self.detectionSpace.drawX(td_pred.x, 0.5)
        self.detectionSpace.drawDsearchRegionAroundDetection(td_pred.x)
        
        add_estimate = True # if this is true, then estimate next value with method, else add detection with posiiton of last detection and velocity 0 (as if it did not move)
        if(len(next_dets) == 0): # hole
            if(self.holes_ref == 0): # set holes_ref for reference to determine relative holes
                self.holes_ref = self.holes
            
            self.holes = self.holes + 1
            
            # handle case for consecutive holes:
            if(n_empty != -1):
                add_estimate = False
                pass
            # else: pass
        else: # has detections
            # add detections (objects, not just coords) to trajectory detections set (for intersections with other trajectories)
            self.D.update(next_dets)
            self.holes_ref = 0 # reset holes_ref (logic in if block needs this)
        
        # might not need to add it (because)
        td_next = TDet(td_current.x, td_pred.t, 0, td_current.theta)
        # d_t1 = TDet(x_t1,t_i1,v_t1,theta_t1) # next detection, it should probably be something else
        if(add_estimate):
            # # 3. estimate next detection: weighted mean of detections
            td_next, _ = self.detectionSpace.estimateNext2(td_current, td_pred, next_dets, next_probs, dt=dt)
    
        # # 4. add calculated estimate to list (trajectory)
        self.X.append(td_next)

        if(debug):
            print("this is extend:")
            print(td_current)
            print(td_pred)
            print(next_dets)
            print("next: ", td_next)
            print(self.X)

    # method checks if trajectories is made from same points (some kind of equals)
    def basedOnSameDetections(self, t2: Trajectory) -> bool:
        return self.D == t2.D # is this equals?
    
    # equals2: check if it is based on same points AND if it has same score.
    # basically: IF it has greater score, returns false, otherwise, true
    # also compares trajectory scores - use case: if every trajectory is tested against every other in same list,
    # only ones with higher score will survive
    # WARNING: is kinda slow (because of getScoreUnbalanced (*2))
    def equalsDetScore(self, t2: Trajectory) -> bool:
        if(self.basedOnSameDetections(t2)):
            if(int(self.getScoreUnbalanced()) <= int(t2.getScoreUnbalanced())): # int comparison: might be problematic [TODO]
                return True
            # return self.equalsTrajectory(t2)
        return False
    
    # compares trajectory path (TDets)
    def equalsTrajectory(self, t2: Trajectory) -> bool:
        if(len(self.X) == len(t2.X)):
            i = 0
            while(i < len(self.X)):
                tdet1: TDet = self.X[i]
                tdet2: TDet = t2.X[i]
                if(not tdet1.equalsCoordsTime(tdet2)):
                    return False
                i = i+1
            return True
        else:
            return False
    
    # calculates unbalanced score of trajectory (should be used only for comparison of two trajectories, not for qbp matrix calculation)
    # (almost) duplicate of first part of DetectionSpace.buildQBPMatrix (for merit term calculation)
    def getScoreUnbalanced(self):
        q_ii = 0
        S_err = 0
        e1,e2=1,1
            
        dets_in_tr = self.D # detections, that are part of trajectory (this is set)
        dets_in_tr_map = detSet2map(dets_in_tr) # (this is map of ^)

        for dets_i in dets_in_tr_map:
            dets = dets_in_tr_map[dets_i]
            g_k = self.g_k(dets)
            if(g_k is None): # handled case (see method g_k)
                continue
            # add to sum:
            S_err = S_err + ((1.0 - e2) + e2*g_k)

        # 2. add holes (S_model)
        q_ii = q_ii - e1*self.holes + S_err
        return q_ii
        

    def __str__(self):
        disabled = ""
        if(self.disable_grow):
            disabled = " (d)"
        return "{t"+str(self.id)+", len="+str(len(self.X))+", S="+str(int(self.S))+disabled+"}"

    def __repr__(self):
        return self.__str__()