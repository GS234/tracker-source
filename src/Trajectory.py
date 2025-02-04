from __future__ import annotations # da delajo tut type hint-i znotraj istega class-a
import numpy as np
from Detection import Detection, TDet, TDet_from_Detection
import math
import copy
from helper_func import bbDet2Det, detSet2map, getTrColor, showHists, compareHists, drawX, drawO, drawDot, drawLine, drawBoundingBox, getVecMagAng, getMotionVec, getRect, getRectBb, IoU, getFlowToFromAtI,getFlowAtI, getFlowToI, vecScore # helper functions
import cv2 as cv
# to avoid cyclic import (detection space imports trajectory, trajectory imports detection space)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from DetectionSpace import DetectionSpace

FLOW_WINDOW_SIZE = 10
# EST_SCORE = 0.001 # score of estimated det (if there is no detection)
EST_SCORE = 0.2 # score of estimated det (if there is no detection)
# DET_ADD_THR = 0.50 # threshold to add detection to trajectory
# DET_ADD_THR = 0.35
DET_ADD_THR = 0.2
MAX_HOLES = 10
E2 = 0.3


class Trajectory:
    # assumptions:
    # Hi_ti -> if there is only one detection inside of event cone when building trajectory, then this is used as Hi_ti
    # if there are more, then the one with maximum probability (according to Dt (is defined by trajectory point)) is selected
    # (could probably also use weighted average / average / random / build hypotheses for all of them (hard??)) --> discussion is needed
    Tid = 0 # apparently static? (want private static, maybe should be _Tid)

    def __init__(self, d0: Detection, detectionSpace: DetectionSpace, color: list=None):
        self.id = Trajectory.Tid # unique id of the trajectory (is used in T2 dicts)
        self.not_so_much_unique_id = self.id # this one is used to determine trajectory identity (of object - gets propagated at merging)
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

        self.D: set[Detection] = set([d0]) # all detections in the trajectory (set: to determine intersecting detections with other trajectories to calculate penalty) (kinda redundant, D2 is used mostly)
        
        self.D2: dict[Detection, float] = {} # 'new' scores
        self.T2: dict[Trajectory, float] = {} # trajectory -> merge score (these scores are added when merging trajectories)
        self.X: TDet = [ ] # trajectory points ("trajectory" detections)
        self.F: dict[int, list[Detection]] = {} # dictionary: used to store points where trajectory forked (used mostly at the end of tracking)
        self.scores: list[float] = [] # scores of detections (relative to estimate; H_t -> score of trajectory point) (kinda redundant, D2 is used mostly)
        
        self.S = 0 # score/support of the trajectory - getScore
        self.holes = 0 # counter: how many trajectory points have been added considering only estimate of next detection
        
        # pruning variables, signals:
        self.holes_ref = 0 # holes reference: used to calculate relative number of holes (set to self.holes first, then calculate difference) (used in extend)
        self.not_selected_strike = 0 # number of times the trajectory has not been selected but is in hypothesis set
        self.disable_grow = False # flag: disable trajectory to grow (if not selected for a while)

        self.possibly_occluded = False # if this is true, trajectory should be extended by e1Nc instead of extend
        self.possible_next: list[Trajectory] = [] # this is list of trajectories that are possible after this one got occluded

        # new things:
        self.term = False
        self.exited = False # true if entered exit zone

        # -----------------
    
    # draw it:
    # method draws trajectory to detection space
    def drawToSpace(self, color=None):
        color_is_set = True
        if(color is None):
            color_is_set = False
            color = self.color
        x_color = (np.array(self.color).astype(np.float32)*0.4).astype(np.uint8)  #[0,0,0]
        if(len(self.X) > 1): # if has many
            for i in range(len(self.X)-1):
                xi = self.X[i].x
                di1 = self.X[i+1] # next detection
                if(not color_is_set):
                    if(di1.color is not None):
                        color=di1.color
                    else:
                        color = self.color
                xi1 = di1.x
                drawLine(self.detectionSpace.map, xi,xi1,color)
                # self.detectionSpace.drawDsearchRegionAroundDetection(xi1)
            drawX(self.detectionSpace.map, self.X[-1].x, x_color) # end
        # draw on top of everything else
        if(len(self.X) > 0): # if has one
            drawO(self.detectionSpace.map, self.X[0].x, x_color) # start
    
    # method returns copy of this trajectory
    def getCopy(self, deep=True) -> Trajectory:
        t_ret = Trajectory(self.origin, self.detectionSpace)
        
        # things to not deepcopy
        t_ret.S = self.S
        t_ret.holes = self.holes
        t_ret.holes_ref = self.holes_ref
        t_ret.not_selected_strike = self.not_selected_strike
        t_ret.color = self.color
        t_ret.term = self.term # also this?
        # also not so much unique id?
        t_ret.not_so_much_unique_id = self.not_so_much_unique_id
        
        # things to deepcopy (do we really need to deepcopy this?)
        
        if(deep):
            t_ret.D = copy.deepcopy(self.D)
            t_ret.D2 = copy.deepcopy(self.D2)
            t_ret.X = copy.deepcopy(self.X)
            t_ret.F = copy.deepcopy(self.F)
        else:
            t_ret.D = copy.copy(self.D)
            t_ret.D2 = copy.copy(self.D2)
            t_ret.X = copy.copy(self.X)
            t_ret.F = copy.copy(self.F)
        

        return t_ret

    # method estimates next position using optical flow (is better to have separate method, maybe could also make wrapper)
    def estimateNextUsingFlow(self, td_current: TDet, dt: int = 1) -> TDet:
        x = td_current.x
        t = td_current.t
        
        # get displacement vector:
        # rect_a = FLOW_WINDOW_SIZE # 10*10 neighbourhood
        # bb_x, bb_y = td_current.x[0]-rect_a//2, td_current.x[1]-rect_a//2
        # bb = (bb_x, bb_y, rect_a, rect_a)
        bb = td_current.bb
        flow_region = getRectBb(bb, self.detectionSpace.flow_map)
        flow_vec2 = (getMotionVec(flow_region))
        disp_vec = flow_vec2*dt
        # drawBoundingBox(self.detectionSpace.map, bb, [0,255,255])

        # construct next detection
        next_v,next_theta = getVecMagAng(flow_vec2)
        x_next = x+disp_vec

        nextTDet = TDet(x_next, t+dt, next_v,next_theta)
        nextTDet.color_hist = td_current.color_hist # assume current appearance
        nextTDet.hasHist = td_current.hasHist
        next_bb = td_current.bb.copy() # current is best estimate for next
        next_bb[0:2] = next_bb[0:2]+disp_vec
        nextTDet.bb = next_bb
        return nextTDet
    
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
        bb_color = [0,255,255]
        if(self.id in {137,142,143}):
            bb_color = [100,255,255]
        if(self.id in {138,144}):
            bb_color = [100, 255, 100]
        drawBoundingBox(self.detectionSpace.map, td_next.bb, bb_color)
        drawBoundingBox(self.detectionSpace.flow_img, td_next.bb, bb_color)
        
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
    


    # [NEW METHODS]:

    # new main extend method
    # some refactoring is prob needed
    def extend4(self, det_add_thr=DET_ADD_THR, add_est=False, debug=False) -> tuple[set[Detection],list[Trajectory]]:
        if(self.term):
            return (set(),[self])
        td_current: TDet = self.X[-1] # current tdet (last trajectory point)
        # print("current of t"+str(self.id)+": ",td_current, self.X)
        td_next: TDet = self.estimateNextUsingFlow(td_current)

        # collectWithin2:
        next_dets, next_probs = self.detectionSpace.collectWithin2(td_next.t, td_next)

        # print("this is det probs: ", next_probs)

        bb_color = [0,255,255]
        # print("self.id: ",self.id)
        # if(self.id in {137,142,143}):
        #     bb_color = [0,0,255]
        # if(self.id in {138,144}):
        #     bb_color = [255, 0, 0]
        if(not self.term):
            drawBoundingBox(self.detectionSpace.map, td_next.bb, bb_color)
        

        # print("dets, probs of t"+str(self.id)+": ",next_dets, next_probs)

        n_dets = len(next_dets)
        used_dets: set[Detection] = set()
        next_tr: list[Trajectory] = list() # is used to collect forks
        # next_tr_origins: set[Detection] = set()
        if(n_dets > 0): # we have detections: first: continue this one, every else: copy&add (fork)
            this_current = self.getCopy(deep=False) # current trajectory
            i2 = 0 # secondary i - to determine if add to current or copy&add (and also allow using threshold)

            for i in range(n_dets):
                d_i = next_dets[i]
                d_i_prob = next_probs[i]

                if(d_i_prob >= det_add_thr):
                # if(d_i_prob >= 0):
                    self.holes_ref = 0
                    next_tdet = TDet_from_Detection(d_i) # has no v, theta, important is, that it has bounding box; should probably also compare color model, but when we have one
                    used_dets.add(d_i)
                    
                    if(i2 == 0): # first one continue this one, every else copy&add
                        self.X.append(next_tdet)
                        self.D.add(d_i)
                        self.D2[d_i]=d_i_prob # also add score to trajectory
                        
                    else:
                        next_t = this_current.getCopy(deep=False)
                        print("[!] FORKING t"+str(self.id)+" INTO t"+str(next_t.id), "(iou: ",d_i_prob," )")
                        next_t.X.append(next_tdet)
                        next_t.D2[d_i]=d_i_prob # also add score to trajectory
                        next_tr.append(next_t)
                        # next_tr_origins.add()
                    
                    i2 = i2+1 # increase if deteciton is appended
            if(i2 == 0): # it means that no detection has been added, so continue current trajectory with estimate
                self.holes_ref = self.holes_ref + 1
                self.X.append(td_next)
                self.D2[td_next]=EST_SCORE
            # elif(add_est): # add one additional trajectory: one that is continued with estimate
            #     next_tr.append(t_estimated)
                
        else:
            self.holes_ref = self.holes_ref + 1
            self.X.append(td_next) # continue current, with estimate
            self.D2[td_next]=EST_SCORE # as detection add estimate
        return (used_dets, next_tr)
    
    # testing/debug method - extend trajectory with optical flow only
    def extendFlowOnly(self) -> tuple[set[Detection], list[Trajectory]]:
        if(self.term):
            return (set(),[self])
        td_current: TDet = self.X[-1] # current tdet (last trajectory point)
        td_next: TDet = self.estimateNextUsingFlow(td_current)
        next_tr: list[Trajectory] = []

        self.X.append(td_next) # continue current, with estimate
        self.D2[td_next]=EST_SCORE # as detection add estimate
        return (set(), next_tr)
    
    # new build method (for now only append origin)
    def build2(self):
        self.X.append(self.origin)
        self.D2[self.origin] = 1.0
    
    # methods used to build QPB matrix:
    # WRAPPER METHODS TO CHOOSE CORRECT METHOD (should probably do differently) (FUJ)
    def getScoreX(self, type=1):
        if(type == 1):
            return self.getScore2()
        elif(type==2):
            return self.getScoreII()
        else:
            return 0
        
    def getInteractionCostX(self, other, type=1):
        if(type==1):
            return self.getInteractionCost2(other)
        elif(type==2):
            return self.getInteractionCostII(other)
        else:
            return 0
    # -----------------------------------------------



    # I. qbp1:
    # method calculates score of trajectory (simply sum of all detecion scores)
    def getScore2(self):
        return sum(self.D2.values())
    
    # method returns interaction cost of two trajectories
    def getInteractionCost2(self, other_t: Trajectory):
        # P1 = 0.05 # tie-breaker parameter
        P1 = 0.1 # tie-breaker parameter
        # 1. get points in intersection
        det_intersect = self.D2.keys() & other_t.D2.keys()

        # choose weaker hypothesis
        tr_l = other_t # assume weaker is the other
        if(other_t.getScore2() > self.getScore2()):
            tr_l = self # change if necessary
        
        # 2. calculate g of intersecting points (with D of the weaker hypothesis)
        q_ij = 0
        # if there is only one point in intersection, then trajectories have merged - keep one that has bigger score
        if(len(det_intersect) == 1):
            # print("JOIN FORK ----->")
            det = det_intersect.pop()
            score_of_this = self.D2[det]
            score_of_other = other_t.D2[det]

            print("[!] JOIN FORK: score of this (t"+str(self.id)+"): ",score_of_this, "score of other (t"+str(other_t.id)+"):", score_of_other)
            
            q_ij = other_t.getScore2() * -(0.5+P1) # penalize other
            # q_ij = -200
            if(score_of_other > score_of_this):
                q_ij = self.getScore2() * -(0.5+P1) # if this has weaker det prob, penalize this (is this really necessary? weaker does not get selected either way)
                # q_ij = -100.0
            # print("<---------")
        # if there are many, then it is probably fork or previous
        else:
            S_err = 0
            for dets_i in det_intersect:
                int_cost = tr_l.D2[dets_i]
                S_err = S_err + int_cost

            q_ij = S_err * -(0.5+P1)
        # print("ic t"+str(self.id)+" - t"+str(other_t.id)+":", q_ij," n intersect: ", len(det_intersect))
        return q_ij
    

    # II. qbp2: connecting trajectories
    def getScoreII(self):
        score = 0.0
        # score=self.getScore2()
        # score = score + sum((1.0-E2) + E2*np.log(list(self.T2.values())))
        
        iou_scores = np.array(list(self.T2.values()))[:,0]
        score = score + sum(iou_scores)
        # print(self.id, "this is getscoreii, trajectories: ",self.T2.keys())
        return score

    # method calculates connection cost of two trajectories (for 2nd stage: build Q for joining trajectories)
    def getInteractionCostII(self, other_t: Trajectory):
        # P1 = 0.05 # tie-breaker parameter
        # P1 = 0.10 # tie-breaker parameter
        P1 = 0.1 # tie-breaker parameter
        # 1. get points in intersection
        tr_intersect = self.T2.keys() & other_t.T2.keys()
        
        # choose weaker hypothesis
        tr_l = other_t # assume weaker is the other
        if(other_t.getScoreII() > self.getScoreII()):
            tr_l = self # change if necessary
        
        # 2. calculate g of intersecting points (with D of the weaker hypothesis)
        q_ij = 0
        S_err = 0
        for tr_i in tr_intersect:
            # int_cost = sum(tr_i.D2.values())+tr_l.T2[tr_i] ## !!! fix
            int_cost = tr_l.T2[tr_i][0] + (1.0-tr_l.T2[tr_i][0])
            # int_cost = tr_l.T2[tr_i][0]
            # S_err = S_err + ((1.0-E2) +  E2*np.log(int_cost))
            S_err = S_err + int_cost

        q_ij = S_err * -(0.5+P1)
        # print("ic t"+str(self.id)+" - t"+str(other_t.id)+":", q_ij," n intersect: ", len(tr_intersect))
        return q_ij
        
    # method calculates probability of connection of two trajectories (time and space distance)
    def getConnectionProb(self, other_t: Trajectory, time_window=20, max_space_diff=100.0):
        # print("this is get connection prob: ")
        first, second = t1t2ToFirstSecond(self, other_t)
        # print(first.X[0].t, first.X[-1].t, " - ", second.X[0].t, second.X[-1].t)
        end: TDet = first.X[-1] # end of first
        begining: TDet = second.X[0] # begining of second

        # 1. check if they live in the same time (first.X[-1].t < second.X[0].t)
        time_d = begining.t - end.t
        time_p = 1.0
        if(time_d >= 0): # it is ok
            time_rel = time_window-time_d
            if(time_rel < 0):
                time_rel = 0.0
            
            time_p = float(time_rel)/float(time_window)
        else: # it is not ok, they live at the same time
            time_p = 0.0
        
        # 2. distance:
        space_d = begining.x - end.x
        space_dd = np.sqrt(np.dot(space_d, space_d))
        space_p =  (max_space_diff - space_dd) / max_space_diff
        # print("probs: %1.4f, %1.4f; %1.4f" % (time_d, space_dd, time_p*space_p))
        return time_p*space_p
    
    # method returns list of possible next trajectories (of this)
    def getPossibleNext(self, tr_list, time_window=20, max_space_diff=100):
        possibleNext = []
        for t in tr_list:
            # skip this one
            if(t == self):
                continue
            
            first = self
            second = t
            # first_firstX = first.X[0]
            first_lastX = first.X[-1]
            
            second_firstX = second.X[0]
            # second_lastX = second.X[-1]
            time_diff = second_firstX.t-first_lastX.t
            if(time_diff >= 0 and time_diff < time_window):
                space_diff = first_lastX.x - second_firstX.x
                space_diff = np.sqrt(np.dot(space_diff, space_diff))
                if(space_diff < max_space_diff):
                    possibleNext.append(t)
        return possibleNext
    
    # method returns list of possible next trajectories, but instead of space difference, it uses extrapolated bb and iou of previous with first bb of possible next trajectory
    # it also returns iou-s for each connection
    def getPossibleNext2(self, tr_list, n_prev=5, n_ext=5, time_window=20):
        possibleNext = []
        # this_bb = self.getNextBbII(n_prev, n_ext)
        for t in tr_list:
            # skip this one
            if(t == self):
                continue
            
            first = self
            second = t
            # first_firstX = first.X[0]
            first_lastX:TDet = first.X[-1]
            
            second_firstX:TDet = second.X[0]
            # second_lastX = second.X[-1]
            time_diff = second_firstX.t-first_lastX.t
            if(time_diff >= 0 and ((time_diff < time_window))):
                # get iou, if it is not 0, add to possible next
                # create virtual detection with bb (IoU accepts only detections)
                # print("t%d, t%d: %d"%(first.id, second.id, time_diff))
                this_bb = self.getNextBbII(n_prev, n_ext)
                # this_bb = self.getNextBbII(n_prev, time_diff) # extrapolate for time_diff frames
                # drawBoundingBox(self.detectionSpace.map, this_bb, [0,255,150])
                # drawLine(self.detectionSpace.map, first_lastX.x, (np.array(this_bb[0:2])+np.array(this_bb[2:])/2))
                det_with_this_bb = Detection([0,0], bb=this_bb)
                this_other_iou = IoU(det_with_this_bb, second_firstX)

                # print(v1, v2)
                # drawLine(self.detectionSpace.map, detB_x, first_lastX.x)
                # drawLine(self.detectionSpace.map, second_firstX.x, detF_x)
                nextBb_x = (np.array(this_bb[0:2])+np.array(this_bb[2:])/2)
                # v1 = np.array(second_firstX.x)-nextBb_x
                # v2 = np.array([0,0])
                # vec_score = vecScore(v1,v2)
                vec_score=1.0
                # print("vec_score: ",vec_score)


                if(not math.isclose(this_other_iou, 0)):
                    # print(this_other_iou)
                    ext_tdets = [] # extrapolated frames
                    bb_ext = this_bb.copy()
                    v = np.array(bb_ext[0:2]) - np.array(first_lastX.bb[0:2])
                    step = np.array([0,0])
                    if(time_diff != 0):
                        step = v/time_diff
                    # also extrapolate frames
                    for i in range(time_diff):
                        # next_bb = bb_ext
                        bb_next = first_lastX.bb.copy()
                        bb_next[0:2] = np.array(bb_next[0:2])+step
                        next_det = bbDet2Det(bb_next, first_lastX.t+i+1)
                        next_tdet = TDet_from_Detection(next_det)
                        ext_tdets.append(next_tdet)
                    ext_tdets = ext_tdets[0:-1]

                    possibleNext.append((t, this_other_iou, vec_score, ext_tdets))
                
        return possibleNext


    
    # method fits line to last n-points in track
    # n: n last points (at most; if tracklet is shorter, then use as many as there are)
    # n_ext: n frames to extrapolate
    # [WARN] might have problems with vertical lines (untested)
    def getNextBbII(self, n:int = 5, n_ext = 5, debug=False):
        last_n:list[TDet] = self.X[-n:]
        ret_bb = last_n[-1].bb.copy()
        if(len(last_n) == 1): # we have only one point (origin)
            ret_val = last_n[0].x

            if(debug):
                return ret_val,ret_val,ret_val,None,last_n[0].bb
            # return last_n[0].bb # return only bb
            return ret_bb # return only bb
        
        last_n_x = []
        last_n_y = []
        d:Detection = None
        for d in last_n:
            last_n_x.append(d.x[0])
            last_n_y.append(d.x[1])
        
        # 1. get avg vector:
        len_last_n = len(last_n)
        x_dash = ( np.array(last_n[-1].x) -np.array(last_n[0].x)) / len_last_n # vector spanning from first to last point
        
        # 2. project it to line vector:
        # 2.1 get line vector:
        x = np.array([0,10])
        
        diff = [0,0] # 'pythonic way of declaring variables' (FUJ)
        try:
            poly = np.polynomial.Polynomial.fit(x=last_n_x,y=last_n_y,deg=1)
            y = poly(x)
            diff = np.array([x[-1], y[-1]]) - np.array([x[0], y[0]])
        except np.linalg.LinAlgError:
            print("[WARN] could not fit polynomial. Displacement might be vertical, using base vector [0,1]")
            diff = [0,1]
        
        
        diff_len = np.sqrt(np.dot(diff,diff))
        v = np.array([0,0]) # is zero, if length is zero
        if(not math.isclose(diff_len,0)):
            v = diff / diff_len # vector of size 1

        # 2.2 projection:
        a = np.dot(v, x_dash)
        pr_xv = v*a
        
        # 3. extrapolate from average point:
        x_avg = np.array([np.average(last_n_x), np.average(last_n_y)])
        x_next = x_avg + (n/2 + n_ext)*pr_xv
        
        
        # x_next is center of bb, so we need to subtract its h,w
        ret_bb[0:2] = list(np.array(x_next) - np.array(ret_bb[2:])/2)

        if(debug):
            return x_next, x_avg, pr_xv, poly, ret_bb
        return ret_bb # return only bb


    # III. other trajectories: connect with flow
    def getPossibleNext3(self, tr_list, flow_path, time_window=20, max_iou_diff=0.02, iou_thresh=0.1):
        possibleNext = []
        for t in tr_list:
            # t = tr_list[1]
            # skip this one
            if(t == self):
                continue
            
            first = self
            second = t
            # first_firstX = first.X[0]
            first_lastX:TDet = first.X[-1]
            second_firstX:TDet = second.X[0]

            # second_lastX = second.X[-1]
            time_diff = second_firstX.t-first_lastX.t
            # print(time_diff, time_window)
            if(time_diff >= 0 and time_diff < time_window):
                # drawBoundingBox(self.detectionSpace.map,first_lastX.bb,[54,181,255])
                # drawBoundingBox(self.detectionSpace.map,second_firstX.bb,[26,62,240])
                
                # print("t%d - t%d, time diff:%d"%(first.id, second.id, time_diff))
                # detF = Detection(first_firstX.x, first_firstX.t, first_firstX.bb)
                # detB = Detection(first_lastX.x, first_lastX.t, first_lastX.bb)
                detF = Detection(first_lastX.x, first_lastX.t, first_lastX.bb)
                detB = Detection(second_firstX.x, second_firstX.t, second_firstX.bb)
                
                # drawX(self.detectionSpace.map, np.array(detF.bb[0:2])+np.array(detF.bb[2:])/2, [54,181,255])
                # drawX(self.detectionSpace.map, np.array(detB.bb[0:2])+np.array(detB.bb[2:])/2, [26,62,240])
                
                ext_tdets= [] # extrapolated bounding boxes
                for i in range(time_diff+1):
                    next_bb = self.getNextBbIII(detF, detF.t+i, flow_path=flow_path)
                    prev_bb = self.getNextBbIII(detB, detB.t-i+1, flow_path=flow_path, forward=False)

                    next_det = bbDet2Det(next_bb.copy(), first_lastX.t+i+1)
                    ext_tdets.append(TDet_from_Detection(next_det))

                    # debug draw
                    # drawBoundingBox(self.detectionSpace.map, next_bb)
                    # drawDot(self.detectionSpace.map, np.array(next_bb[0:2])+np.array(next_bb[2:])/2, [54,181,255])
                    # drawDot(self.detectionSpace.map, np.array(prev_bb[0:2])+np.array(prev_bb[2:])/2, [26,62,240])
                    # self.detectionSpace.showSpace(dspace_winname="dspace_win2")
                    detF.bb = next_bb
                    detB.bb = prev_bb
                ext_tdets = ext_tdets[0:-2]
                # self.detectionSpace.showSpace(dspace_winname="dspace_win2", draw_dets=False, draw_last_dets_bb=False)
                
                detF_x = np.array(detF.bb[0:2])+np.array(detF.bb[2:])/2
                detB_x = np.array(detB.bb[0:2])+np.array(detB.bb[2:])/2
                # drawO(self.detectionSpace.map, detF_x, [54,181,255])
                # drawO(self.detectionSpace.map, detB_x, [26,62,240])
                
                this_other_iou = IoU(detF, second_firstX)
                other_this_iou = IoU(detB, first_lastX)
                
                v1 = detB_x-np.array(first_lastX.x)
                v2 = np.array(second_firstX.x)-detF_x
                # print(v1, v2)
                # drawLine(self.detectionSpace.map, detB_x, first_lastX.x)
                # drawLine(self.detectionSpace.map, second_firstX.x, detF_x)
                vec_score = vecScore(v1,v2)
                iou_diff = abs(this_other_iou - other_this_iou)
                if(not math.isclose(this_other_iou, 0) and iou_diff < max_iou_diff and this_other_iou >= iou_thresh):
                # if(True):
                    # print("t%d - t%d, time diff:%d, IoU->:%6.2f, IoU<-:%6.2f, vecs=%6.2f"%(first.id, second.id, time_diff, this_other_iou, other_this_iou, vec_score))
                    # print(this_other_iou)
                    possibleNext.append((t, this_other_iou,vec_score, ext_tdets))
        return possibleNext
    
    # method gets next bounding box using optical flow
    # forward: flag to indicate whether to compute flow forwards or backwards (forward: for continuing trajectory, backward: for searching previous)
    def getNextBbIII(self, d:Detection, t:int, forward:bool=True, flow_path=""):
        next_bb = d.bb.copy()
        # flow_to, flow_from = getFlowToFromAtI(t,flow_path)
        flow_vec2 = []
        if(forward):
            flow_from = getFlowAtI(t, flow_path)
            flow_region = getRectBb(next_bb, flow_from)
            flow_vec2 = (getMotionVec(flow_region))
        else:
            flow_to = getFlowToI(t, flow_path)
            flow_region = getRectBb(next_bb, flow_to)
            flow_vec2 = -(getMotionVec(flow_region))
        next_bb[0:2] = next_bb[0:2]+flow_vec2
        return next_bb
    
    # calculate score similar to how we do it in II.
        
        
    
    

    # [END NEW METHODS] ------------------------------
    
    # returns string with bounding boxes
    # n_pad: for missing frames
    def tr2bbStr(self, n_all: int=0):
        ret_str = ""
        for td in self.X:
            ret_str = ret_str + "%6.2f,%6.2f,%6.2f,%6.2f\n"%(td.bb[0],td.bb[1],td.bb[2],td.bb[3])
        # pad missing:
        for i in range(n_all-len(self.X)+1):
            td = self.X[-1]
            ret_str = ret_str + "%6.2f,%6.2f,%6.2f,%6.2f\n"%(td.bb[0],td.bb[1],td.bb[2],td.bb[3])
        return ret_str

    
    def __str__(self):
        disabled = ""
        possibly_occluded = ""
        if(self.disable_grow):
            disabled = " (d)"
        if(self.possibly_occluded):
            possibly_occluded  = " (|?|)"
        # return "{t"+str(self.id)+", len="+str(len(self.X))+", S="+ f"{self.S:.4f}" +disabled+possibly_occluded+", c="+str(self.color)+"}"
        return "{t"+str(self.id)+", len="+str(len(self.X)) +disabled+possibly_occluded+", c="+str(self.color)+"}"

    def __repr__(self):
        return self.__str__()
    

def t1t2ToFirstSecond(t1: Trajectory, t2: Trajectory):
        if(t1.X[0].t >= t2.X[0].t):
            return (t2, t1)
        return t1, t2