from __future__ import annotations # da delajo tut type hint-i znotraj istega class-a
import numpy as np
from Detection import Detection, TDet
import math
import copy
from helper_func import detSet2map, getTrColor, showHists, compareHists, drawX, drawLine, drawBoundingBox, getVecMagAng, getMotionVec, getRect, getRectBb # helper functions
import cv2 as cv

# to avoid cyclic import (detection space imports trajectory, trajectory imports detection space)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from DetectionSpace import DetectionSpace

FLOW_WINDOW_SIZE = 10

class Trajectory:
    # assumptions:
    # Hi_ti -> if there is only one detection inside of event cone when building trajectory, then this is used as Hi_ti
    # if there are more, then the one with maximum probability (according to Dt (is defined by trajectory point)) is selected
    # (could probably also use weighted average / average / random / build hypotheses for all of them (hard??)) --> discussion is needed
    Tid = 0 # apparently static? (want private static, maybe should be _Tid)

    def __init__(self, d0: Detection, detectionSpace: DetectionSpace, color: list=None):
        self.id = Trajectory.Tid # unique id of the trajectory
        Trajectory.Tid = Trajectory.Tid+1
        self.detectionSpace = detectionSpace # pointer to detection space in which trajectory lives (has detections)
        
        # self.color = (np.random.rand(3)*150).astype(np.uint8)+10 # color of trajectory (for visualization)
        # self.color[2]=255
        self.color = getTrColor()
        if(color is not None):
            self.color = color
        
        # set origin: fromDetection + some mods (also need bounding box, color histogram)
        self.origin = TDet(d0.x,d0.t,0,0)
        self.origin.bb = d0.bb
        self.origin.hasHist = d0.hasHist
        self.origin.color_hist = d0.color_hist
        # ---

        self.D = set([d0]) # all detections in the trajectory (set: to determine intersecting detections with other trajectories to calculate penalty)
        self.X = [ ] # trajectory points ("trajectory" detections)
        
        self.S = 0 # score/support of the trajectory - getScore
        self.holes = 0 # counter: how many trajectory points have been added considering only estimate of next detection
        
        # pruning variables, signals:
        self.holes_ref = 0 # holes reference: used to calculate relative number of holes (set to self.holes first, then calculate difference) (used in extend)
        self.not_selected_strike = 0 # number of times the trajectory has not been selected but is in hypothesis set
        self.disable_grow = False # flag: disable trajectory to grow (if not selected for a while)

        self.possibly_occluded = False # if this is true, trajectory should be extended by e1Nc instead of extend
        self.possible_next: list[Trajectory] = [] # this is list of trajectories that are possible after this one got occluded

        # -----------------

    # method estimates next position based on current position, velocity and orientation (theta)
    def estimateNext(self, td_current: TDet, dt: int = 1) -> TDet:
        x = td_current.x
        t = td_current.t
        x_t1 = x[0] + int(dt*td_current.v*math.cos(td_current.theta))
        y_t1 = x[1] + int(dt*td_current.v*math.sin(td_current.theta))

        nextTDet = TDet((x_t1, y_t1), t+dt, td_current.v, td_current.theta)
        nextTDet.color_hist = td_current.color_hist # assume current appearance
        nextTDet.hasHist = td_current.hasHist
        return nextTDet
    
    # method estimates next position using optical flow (is better to have separate method, maybe could also make wrapper)
    # [TODO] test it if works correctly
    def estimateNextUsingFlow(self, td_current: TDet, dt: int = 1) -> TDet:
        x = td_current.x
        t = td_current.t
        
        # get displacement vector:
        rect_a = FLOW_WINDOW_SIZE # 10*10 neighbourhood
        bb_x, bb_y = td_current.x[0]-rect_a//2, td_current.x[1]-rect_a//2
        bb = (bb_x, bb_y, rect_a, rect_a)
        flow_region = getRectBb(bb, self.detectionSpace.flow_map)
        flow_vec2 = (getMotionVec(flow_region))
        drawBoundingBox(self.detectionSpace.map, bb, [0,255,255])

        # construct next detection
        next_v,next_theta = getVecMagAng(flow_vec2)
        x_next = x+flow_vec2*dt

        nextTDet = TDet(x_next, t+dt, next_v,next_theta)
        nextTDet.color_hist = td_current.color_hist # assume current appearance
        nextTDet.hasHist = td_current.hasHist
        return nextTDet
        
    
    # method builds trajectory and returns list of trajectory detections (and holes)
    # HOLES: allow up to n holes, if no detections after n frames, discontinue (also delete all predictions); also prune tails from both ends
    def connectPoints(self, td_orig: TDet, dt=1) -> list[TDet]:
        # print("this is connect points:")
        t_n = len(self.detectionSpace.D) # number of 'detection frames'
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
            # td_next, _ = self.detectionSpace.estimateNext2UsingFlow(td_current, td_pred, next_dets, next_probs, dt=dt)
            
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
            drawX(self.detectionSpace.map, self.X[0].x, x_brightness)
        if(len(self.X) > 1): # if has many
            for i in range(len(self.X)-1):
                xi = self.X[i].x
                di1 = self.X[i+1] # next detection
                if(di1.color is not None):
                    color=di1.color
                else:
                    color = self.color
                xi1 = di1.x
                drawLine(self.detectionSpace.map, xi,xi1,color)
                # self.detectionSpace.drawDsearchRegionAroundDetection(xi1)
            drawX(self.detectionSpace.map, self.X[-1].x, x_brightness)
    
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
            # print("[",self.id,"]",d.t, t_off, d.t-t_off, "(",len(self.X),")")
            td = self.X[d.t - t_off]
            d_prob = self.detectionSpace.getProb2(d,td)
            probs.append(d_prob)
        return probs
    
    # POMEMBNO!!
    # extend methods:

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
        s_region_bias = td_current.k*self.holes_ref # search region bias: is added to extend search region (to recover from occlusion, hopefully)
        # s_region_bias = 0 # do not use bias
        next_dets, next_probs = self.detectionSpace.collectWithin(td_pred.t, td_pred, s_region_bias) # collect around prediction

        drawX(self.detectionSpace.map, td_pred.x, 0.5)
        self.detectionSpace.drawDsearchRegionAroundDetection(td_pred.x, 15+td_pred.v+s_region_bias, 15+s_region_bias, td_pred.theta)
        
        add_estimate = True # if this is true, then estimate next value with method, else add detection with posiiton of last detection and velocity 0 (as if it did not move)
        if(len(next_dets) == 0): # hole
            # if(self.holes_ref == 0): # set holes_ref for reference to determine relative holes -> zakaj to rabm?
            #     print("self.holes: ", self.holes)
            #     self.holes_ref = self.holes
            self.holes_ref = self.holes_ref + 1 # simply add 1 (to relative)
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
        # ALSO ADD COLOR MODEL
        td_next.color_hist = td_current.color_hist
        td_next.hasHist = td_current.hasHist
        # -----------

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
    
    # function collects next detections and extends (e1Nc - extend 1 and copy; returns list of new trajectories, that are extended by one detection) (does it solve occlusions?)
    # [TODO] - finish it
    def e1Nc(self, debug=False, n_empty=-1) -> list[Trajectory]:
        print("this is e1Nc")
        # return False # [TODO] - not finished yet
        # 1. find detections around last detection
        # 2. estimate, add to trajectory, ...
        dt = 1
    
        td_current: TDet = self.X[-1] # current (expects last detection to have most recent time (extrapolated TDet if not based on detections))
        td_pred: TDet = self.estimateNext(td_current, dt) # prediction
        
        s_region_bias = td_current.k*self.holes_ref # search region bias: is added to extend search region (to recover from occlusion, hopefully)
        # s_region_bias = 0 # do not use bias
        next_dets, next_probs = self.detectionSpace.collectWithin(td_pred.t, td_pred, s_region_bias, use_motion=False) # collect around prediction
        
        # debug info:
        print("[e1Nc] next dets: ",next_dets, next_probs)
        for d in next_dets:
            print("comparing: ", d, td_current, ": ", compareHists(d.color_hist, td_current.color_hist))
        # hists = [td_current.color_hist]
        # for d in next_dets:
        #     print("det:",d)
        #     hists.append(d.color_hist)
        #     # showHists()
        # if(len(hists) > 0):
        #     showHists(hists)
        # else:
        #     print("no histograms")
        # ..
            
            

        drawX(self.detectionSpace.map, td_pred.x, 0.5)
        self.detectionSpace.drawDsearchRegionAroundDetection(td_pred.x, 15+td_pred.v+s_region_bias, 15+s_region_bias, td_pred.theta)
        
        possible_tr = []
        if(len(next_dets) == 0): # no detections; just increase holes, holes_ref
            self.holes_ref = self.holes_ref + 1 # simply add 1 (to relative)
            self.holes = self.holes + 1
        else: # has detections, for each detection, copy this trajectory and extend it with detection
            for i in range(len(next_dets)):
            # for d in next_dets:
                # t_possible = copy.deepcopy(self) # copy of this trajectory
                t_possible = self.getCopy() # copy of this trajectory
                t_possible.holes = t_possible.holes-self.holes_ref # do not count holes from e1Nc
                t_possible.holes_ref = 0 # reset holes counter
                
                # 3. estimate next detection: weighted mean of detections
                d = next_dets[i]
                dp = next_probs[i]
                td_possible_next, _ = self.detectionSpace.estimateNext2(td_current, td_pred, [d], [dp], dt=dt)
                
                # 4. extend: add used detection to D, add new TDet to X
                t_possible.D.add(d)
                t_possible.X.append(td_possible_next)
                possible_tr.append(t_possible)
        
        # might not need to add it (because)
        td_next = TDet(td_current.x, td_pred.t, 0, td_current.theta)
        # ALSO ADD COLOR MODEL
        td_next.color_hist = td_current.color_hist
        td_next.hasHist = td_current.hasHist
        # -----------
        self.X.append(td_next)

        return possible_tr
    
    # OPTICAL FLOW SPECIFIC METHODS:
    
    # like extend, but uses optical flow to create smoother trajectories (idea: use flow when detections are near, if further, jump to detection)
    def extend2(self, debug=False, n_empty=-1):
        if(self.disable_grow):
            return
        # 1. find detections around last detection
        # 2. estimate, add to trajectory, ...
        dt = 1
    
        td_current: TDet = self.X[-1] # current
        td_pred: TDet = self.estimateNextUsingFlow(td_current, dt) # prediction #### use flow!
        
        
        # next_dets, next_probs = self.detectionSpace.collectWithin(td_pred.t, td_current) # collect in next frame (t+1) !! IS THIS CORRECT? !! (collect around current)
        s_region_bias = td_current.k*self.holes_ref # search region bias: is added to extend search region (to recover from occlusion, hopefully)
        # s_region_bias = 0 # do not use bias
        next_dets, next_probs = self.detectionSpace.collectWithin(td_pred.t, td_pred, s_region_bias) # collect around prediction

        drawX(self.detectionSpace.map, td_pred.x, 0.5)
        s1, s2 = self.detectionSpace.s1,self.detectionSpace.s2
        self.detectionSpace.drawDsearchRegionAroundDetection(td_pred.x, s1+td_pred.v+s_region_bias, s2+s_region_bias, td_pred.theta)
        
        has_dets = False
        if(len(next_dets) == 0): # hole
            self.holes_ref = self.holes_ref + 1 # simply add 1 (to relative)
            self.holes = self.holes + 1
        else: # has detections
            # add detections (objects, not just coords) to trajectory detections set (for intersections with other trajectories)
            self.D.update(next_dets)
            self.holes_ref = 0 # reset holes_ref (logic in if block needs this)
            has_dets = True
        
        td_next = td_pred
        self.detectionSpace.drawDsearchRegionAroundDetection(td_pred.x, s1+td_pred.v, s2, td_pred.theta, [0,255,255]) # to visualize
        if(has_dets):
            # # 3. estimate next detection: weighted mean of detections
            td_possibly_next, _ = self.detectionSpace.estimateNext2(td_current, td_pred, next_dets, next_probs, dt=dt)
            
            # check distance; if greater than FLOW_WINDOW_SIZE, then jump to td_next,else just use flow (smoothing trajectory)

            # if(dist >= FLOW_WINDOW_SIZE*self.detectionSpace.s1): ## somehow should determine this size # not ok, should search within 
            
            if(not self.detectionSpace.isWithin(td_possibly_next.x, td_pred.x, a = s1+td_pred.v, b=s2, theta=td_pred.theta)):
                td_possibly_next.color = [0,0,255] # so that path gets different color
                td_next = td_possibly_next
            
        # # 4. add calculated estimate to list (trajectory)
        self.X.append(td_next)

    # uses optical flow only (starting from initial detection)
    def extendUsingFlow2(self, n_empty=-1):
        # print("this is extend using flow 2")
        if(self.disable_grow):
            return
        # 1. find detections around last detection
        # 2. estimate, add to trajectory, ...
        dt = 1

        td_current: TDet = self.X[-1] # current
        # might not need to add it (because it has detections: that is normally the case)
        td_next = TDet(td_current.x, td_current.t+dt, 0, td_current.theta)
        # ALSO ADD COLOR MODEL
        td_next.color_hist = td_current.color_hist
        td_next.hasHist = td_current.hasHist
        
        rect_a = 10
        bb_x, bb_y = td_current.x[0]-rect_a//2, td_current.x[1]-rect_a//2
        td_next.bb = (bb_x, bb_y, rect_a, rect_a)
        # -----------

        # 3. get optical flow data of last point in trajectory
        flow_region = getRect(td_next, self.detectionSpace.flow_map)
        flow_vec2 = (getMotionVec(flow_region)) ## !!! flow is defined up to subpixel accuraccy, float needed
        print(flow_vec2)

        # getDetMotionVector()
        drawBoundingBox(self.detectionSpace.map, td_next.bb, [0,255,255])
        drawBoundingBox(self.detectionSpace.flow_img, td_next.bb, [0,255,255])
        
        mag,ang = getVecMagAng(flow_vec2)
        td_next.v = mag
        print(mag)
        if(mag >= 2.0):
            # setting color
            print("setting color (mag > 2)")
            td_next.color = [0,0,int(255*(mag/5.0))] # test
        if(not math.isclose(mag, 0.0)):
            td_next.theta = ang
        td_next.x = td_current.x + flow_vec2
    
        # # 4. add calculated estimate to list (trajectory)
        self.X.append(td_next)
    
    # method uses optical flow data to estimate next point in trajectory if there are no detections, else it uses extend as normally (collectWtihin + estimateNext2)
    def extend3(self, debug=False, n_empty=-1):
        if(self.disable_grow):
            return
        dt = 1
    
        td_current: TDet = self.X[-1] # current
        td_pred: TDet = self.estimateNext(td_current, dt) # prediction

        s_region_bias = 0 # do not use bias
        next_dets, next_probs = self.detectionSpace.collectWithin(td_pred.t, td_pred, s_region_bias) # collect around prediction (get detections and their probabilities)

        # draw it to space
        drawX(self.detectionSpace.map, td_pred.x, [80,160,255])
        s1, s2 = self.detectionSpace.s1,self.detectionSpace.s2
        self.detectionSpace.drawDsearchRegionAroundDetection(td_pred.x, s1+td_pred.v+s_region_bias, s2+s_region_bias, td_pred.theta)
        
        add_estimate = True # if this is true, then estimate next value with method, else add detection with posiiton of last detection and velocity 0 (as if it did not move)
        if(len(next_dets) == 0): # hole
            self.holes_ref = self.holes_ref + 1 # simply add 1 (to relative)
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
        td_next = self.estimateNextUsingFlow(td_current)
        # -----------

        # d_t1 = TDet(x_t1,t_i1,v_t1,theta_t1) # next detection, it should probably be something else
        if(add_estimate):
            # # 3. estimate next detection: weighted mean of detections
            td_next, _ = self.detectionSpace.estimateNext2(td_current, td_pred, next_dets, next_probs, dt=dt)
        else:
            print("[extend3] using flow estimated next")
    
        # # 4. add calculated estimate to list (trajectory)
        self.X.append(td_next)
        
        
        pass
    
    # ------------------------------


    # method checks if trajectories are made from same points (some kind of equals)
    def basedOnSameDetections(self, t2: Trajectory) -> bool:
        return self.D == t2.D # is this equals?
    
    # equals2: check if it is based on same points AND if it has same score.
    # basically: IF it has greater score, returns false, otherwise, true
    # also compares trajectory scores - use case: if every trajectory is tested against every other in same list,
    # only ones with higher score will survive
    # WARNING: is kinda slow (because of getScoreUnbalanced (*2))
    def equalsDetScore(self, t2: Trajectory, recalculate=True) -> bool:
        if(self.basedOnSameDetections(t2)):
            t1_s = int(self.S)
            t2_s = int(t2.S)

            if(recalculate):
                t1_s = int(self.getScoreUnbalanced())
                t2_s = int(t2.getScoreUnbalanced())

            if(t1_s <= t2_s): # int comparison: might be problematic [TODO]
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
    
    # !! METHODS ARE MAINLY USED TO BUILD QBP MATRIX
    # calculate score of trajectory (like ^, also use balancing constants e1/e2)
    def getScore(self, e1: float = 1.0, e2: float = 1.0) -> float:
        q_ii = 0 # merit term
        S_err = 0
        
        dets_in_tr = self.D # detections, that are part of trajectory (this is set)
        dets_in_tr_map = detSet2map(dets_in_tr) # (this is map of ^)
        # print(dets_in_tr)
        # print(dets_in_tr_map)

        # 1. calculate model error
        for dets_i in dets_in_tr_map:
            dets = dets_in_tr_map[dets_i]
            g_k = self.g_k(dets)
            if(g_k is None): # handled case (see method g_k)
                continue
            # add to sum:
            S_err = S_err + ((1.0 - e2) + e2*g_k)
        
        # 2. add holes (S_model) (model cost)
        q_ii = q_ii - e1*self.holes + S_err
        self.S = q_ii # set score
        return q_ii
    

    # [WARN] score of two trajectories must already be calculated for this function to work properly
    def getInteractionCost(self, other_t: Trajectory, e1: float = 1.0, e2: float = 1.0):
        # 1. get points in intersection
        det_intersect = self.D & other_t.D
        det_intersect_map = detSet2map(det_intersect)

        # choose weaker hypothesis
        tr_l = other_t # assume weaker is the other
        # if(Q_ii[n] > Q_ii[m]):
        if(other_t.S > self.S):
            tr_l = self # change if necessary
        # print(other_t.S, self.S, "selected: ", tr_l.S)

        # 2. calculate g of intersecting points (with D of the weaker hypothesis)
        q_ij = 0
        S_err = 0
        for dets_i in det_intersect_map:
            dets = det_intersect_map[dets_i]
            g_kl = tr_l.g_k(dets)
            if(g_kl is None): # handled case (see method g_k)
                continue
            # add to sum:
            S_err = S_err + ((1-e2) + e2*g_kl)

        q_ij = S_err * (-0.5)
        return q_ij
    
    # method returns copy of this trajectory
    def getCopy(self) -> Trajectory:
        t_ret = Trajectory(self.origin, self.detectionSpace)
        
        # things to not deepcopy
        t_ret.S = self.S
        t_ret.holes = self.holes
        t_ret.holes_ref = self.holes_ref
        t_ret.not_selected_strike = self.not_selected_strike
        
        # things to deepcopy
        t_ret.D = copy.deepcopy(self.D)
        t_ret.X = copy.deepcopy(self.X)

        return t_ret
        
        

    def __str__(self):
        disabled = ""
        possibly_occluded = ""
        if(self.disable_grow):
            disabled = " (d)"
        if(self.possibly_occluded):
            possibly_occluded  = " (|?|)"
        return "{t"+str(self.id)+", len="+str(len(self.X))+", S="+ f"{self.S:.4f}" +disabled+possibly_occluded+"}"

    def __repr__(self):
        return self.__str__()