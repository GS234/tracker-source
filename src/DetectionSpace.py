from __future__ import annotations # da delajo tut type hint-i znotraj istega class-a
import numpy as np
from Trajectory import Trajectory
from Detection import Detection, TDet
import cv2 as cv
from helper_func import * # helper functions
from collections import deque # for queue (de - double ended)


# global vars:
S1, S2 = 15,15 # default s1,s2
DATA_ROOT = "../data/"
X_FFFFFF = [255,255,255]
GREEN = [0,255,0]
EXIT_ZONE_OFFSET = 5

class DetectionSpace:
    def __init__(self, h,w, D: list = None, time_offset = 0):
        # self.map = np.zeros((n,n)).astype(np.float32) # init empty map (old way)
        self.lastFrame = np.zeros((h,w,3)).astype(np.uint8) # last frame (contains no drawings)
        self.map = np.zeros((h,w,3)).astype(np.uint8) # init empty map
        
        self.flow_img = np.zeros((h,w,3)).astype(np.uint8) # optical flow image (for visualization)
        self.last_flow_img = np.zeros((h,w,3)).astype(np.uint8)
        
        self.flow_map = np.zeros((h,w,2)).astype(np.uint8) # optical flow
        self.use_flow = True # flag: use optical (also show it)
        
        self.exit_zone = EXIT_ZONE_OFFSET # offset from edge
        self.hw = (h,w) # map size (dimensions)
        self.window_name = "detection space"
        self.TR = [] # array for storing trajectories (unused)
        self.time_offset = time_offset # time offset constant: because self.D expects t0 at index 0 (t0 is not necessarily 0, so this constant is used to correct detection accesses)
        
        self.D: list[list[Detection]] = [] # detections (2d array, 1st dim. is time, subarrays contain detections)
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
        return X_rot.astype(np.int32).T # normal (x, y)
        # return [(b[0], b[1]) for b in X_rot.astype(np.int32).T]
        # return np.dot(X_rot.T, [[0,1],[1,0]]).astype(np.int32) # return switched coordinates (y, x)

    # !!! POMEMBNO:
    # check if detection d is within this detection's search region (seems to work fine)
    def isWithin(self, x, x_ref, a=S1, b=S2, theta=0):
        rot_mat = self.getRotationMatrix(theta) # get rotation matrix to rotate detection (easier calculation)
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
    # method calculates probability (score) of detection 
    def getProb2(self, d:Detection, td:TDet, a=S1, b=S2, use_motion=True, use_color=True) -> float:
        ret_val = 1
        if(use_motion):
            motion_model_prob = self.getMotionModelProb(d,td,a,b)
            ret_val = ret_val*motion_model_prob
        if(use_color):
            color_model_prob = self.getColorModelProb(d,td)
            ret_val = ret_val*color_model_prob
        return ret_val
        

    # methods to calculate probability of motion model and color model (used in getProb, getProb2)
    # probability density function (bivariate normal distribution) (seems to work fine)
    # probability dist. around trajectory detection (also uses velocity and orientation)
    # [TODO] - stretch it according to velocity [TODO TODO TODO]
    def getMotionModelProb(self, d:Detection, td:TDet, a=S1, b=S2) -> float:
        inv_cov_mat = np.array([[1.0/(3*a),0.0],[0.0,1.0/(3*b)]])
        
        rot_mat = self.getRotationMatrix(theta=td.theta)
        v = d.x - td.x # x - mu
        v = np.dot(rot_mat.T, v) #un-rotate, so that it can be evaluated over un-rotated distribution

        pi_2, cov_mat_det = 2*np.pi , 9*a*b
        # return (1.0 / (np.sqrt( pi_2*pi_2 * cov_mat_det))) * np.exp(-0.5* np.dot( np.dot(v, inv_cov_mat), v)) # probability
        return np.exp(-0.5* np.dot( np.dot(v, inv_cov_mat), v)) # score (unscaled prob) (za vizualizacijo)

    # method compares color models and returns similarity
    def getColorModelProb(self, d:Detection, td:TDet) -> float:
        color_model_prob = 1
        if(d.hasHist):
            if(td.hasHist):
                color_model_prob = compareHists(td.color_hist, d.color_hist)
                # print("model similarity: ", color_model_prob)
            else:
                print("[warn] (getColorModelProb) ", td, " has no color hist, setting to 1")
        else:
            print("[warn] (getColorModelProb) ", d," has no color hist, setting to 1")

        return color_model_prob

    # ----------------------------------------------------------------------------------------

    def showSpace(self, det_color=[0,0,0], draw_dets=True):
        # draw exit zone (border)
        off = self.exit_zone
        exit_zone_color=[0,255,255]
        ul,bl,ur,br = (off,off),(off,self.hw[0]-off),(self.hw[1]-off, off),(self.hw[1]-off, self.hw[0]-off)
        drawLine(self.map, ul,bl,color=exit_zone_color)
        drawLine(self.map, ul,ur,color=exit_zone_color)
        drawLine(self.map, ur,br,color=exit_zone_color)
        drawLine(self.map, bl,br,color=exit_zone_color)

        
        if(self.D):
            if(draw_dets):
                for d in self.D:
                    detections2map(d, self.map, color=det_color)
            last_dets = self.D[-1]
            
            # print(last_dets)
            # print()
            for d in last_dets: # draw bounding boxes around last detections
                drawBoundingBox(self.map, d.bb)
                drawX(self.map, d.x)
                if(self.use_flow):
                    drawBoundingBox(self.flow_img, d.bb)
                    drawX(self.flow_img, d.x)
                    # draw also line in which direction is region moving
                    

                    y, x, h, w = d.bb
                    flow_region = self.flow_map[x:x+w+1,y:y+h+1] # bounding box image
                    motion_vec = getMotionVec(flow_region)
                    drawLine(self.flow_img, d.x, (d.x + 10*motion_vec), [0,255,255])

        # print(map)
        # print(self.map)
        if(self.use_flow):
            cv.imshow(self.window_name+" - optical flow", self.flow_img)
        cv.imshow(self.window_name, self.map)
        cv.waitKey(0)
        # while cv.getWindowProperty(self.window_name, cv.WND_PROP_VISIBLE) >= 1:
        #     cv.waitKey(1)
        # cv.destroyAllWindows()
    
    # method clears detection space of all other things except for detections and last detections' boundingboxes
    def clearSpace(self):
        self.map = self.lastFrame.copy()
        self.flow_img = self.last_flow_img.copy()

    # collects detections within search region at time t (INFO: t is global time (t0 is not necessarily 0, so it is used to calculate offset: t0' = t0 - t_off))
    # returns: DETECTIONS (has changed from coordinates, as we need those objects for trajectories), their probabilities
    # region_bias: used to expand search region
    # use color/motion: parameters to pass on to method getProb2 (one can switch color model / motion model on or off)
    def collectWithin(self, t:int, td: TDet, region_bias=0, use_motion=True, use_color=False):
        # search among detections in next time moment (next frame, that is (whichever, usually immediate successor (dt = 1)))
        next_detections = [] # store 'em in list
        next_detections_probs = [] # weights: sampled from distribution (bivariate normal dist, see Detection.getProb())
        t = t - self.time_offset # to fix indexing of self.D

        # print(self.D)
        # print(t)
        if(t < len(self.D) and t >= 0): # check only if has detections in this layer (and not before 0 (negative indices overflow))
            # print("self.d: ",self.D[t], "t: ", t, "t + t_off: ", t+self.time_offset, "len(d): ", len(self.D))
            for d_i in self.D[t]:
                if(self.isWithin(d_i.x, td.x, a = S1+td.v+region_bias, b=S2+region_bias, theta=td.theta)):
                    # print(d_i, " is within ", td)
                    next_detections.append(d_i) # store detections, for now

                    detection_prob = self.getProb2(d_i, td, use_motion=use_motion, use_color=use_color)

                    next_detections_probs.append(detection_prob) # get probability score from nearby point
        
        return next_detections, next_detections_probs
        
        # old stuff
        # n_det = len(next_detections) # number of detections (i)
        # next_detections_X = np.zeros((n_det, 2))
        # for i in range(n_det):
        #     next_detections_X[i] = next_detections[i].x
        # return next_detections_X, next_detections_probs

    # estimate next point in trajectory
    # input: last detection (for reference: estimate velocity, ...), predicted position, collected detections and their probabilities
    # output: estimated detection
    def estimateNext2(self, d_last: TDet, d_pred: TDet, next_detections: list[Detection], next_detections_probs: list[float], dt=1) -> TDet:
        x_p = d_pred # predicted detection
        x_t = d_last # last detection in trajectory
        t_i1 = x_t.t+dt # next time

        # change list of detections to matrix of detections coordinates
        n_det = len(next_detections) # number of detections (i)
        next_detections_X = np.zeros((n_det, 2))
        next_detections_hist_weighted_sum = d_last.color_hist # init with current model
        for i in range(n_det):
            next_detections_X[i] = next_detections[i].x
            if(next_detections[i].hasHist):
                # print(next_detections_hist_weighted_sum, "  --  ", next_detections[i].color_hist)
                next_detections_hist_weighted_sum = next_detections_hist_weighted_sum + next_detections[i].color_hist*next_detections_probs[i] # do everything in one go
        
        # 1. compute weighted mean (prediction + all detections) to determine actual next point
        # temporal discount, as used in paper [pami, leibe et al. ...] = e^-lambda
        L = 40 # lambda: temporal discount ([TODO] - un-hardcode) (should be large (prediction should not be more significant than actual detection))
        p_tempDisc = np.exp(-L)

        # calculate normalization factor Z (sum of all weights):
        Z = np.sum(next_detections_probs)+p_tempDisc
        x_t1 = np.array((1/Z) * ( p_tempDisc * x_p.x + np.dot(next_detections_probs, next_detections_X) )).astype(np.int32)

        # 2. also estimate velocity, angle (based on x_t+1: no need to calculate weights, estimates again, as they are the same)

        # velocity:
        x_dif = x_t1 - x_t.x
        x_dif = x_dif/dt # we do that here, velocity is then simply it's length
        v_t1 = np.sqrt(np.dot(x_dif,x_dif))

        # theta:
        # calculate relative to unit base vector x_i = [1,0]
        theta_t1 = d_last.theta
        if(v_t1 != 0): # only if it has speed this is relevant
            cos_theta = x_dif[0] / v_t1  # this is it, just trust me bro
            theta_t1 = np.arccos(cos_theta) # co-domain is only from 0-pi, not a problem, because ellipse is symmetrical (so essentially v ~ -v)
        if(x_dif[1] < 0): # same angle is computed for both sides, because we only compare magnitude, so correction is needed in some cases
            theta_t1 = -theta_t1
        
        # appearance model: weighted sum of all models (did that above):
        color_hist_est = (1/(Z-p_tempDisc+1)) * next_detections_hist_weighted_sum # +1? current color model has weight 1 (maybe should be made differently)
        
        d_t1 = TDet(x_t1,t_i1,v_t1,theta_t1) # next detection, it should probably be something else
        d_t1.color_hist = color_hist_est
        d_t1.hasHist = True
        return (d_t1,x_p)



    # !!! POMEMBNO [TODO - fix/adjust/modify/test]
    # estimates next point in trajectory given collected detections and prediction (x_t -> x_t+1)
    def estimateNext(self, t:Trajectory, dt=1) -> Detection:
        # 1. predict next point from current detection
        x_p = t.estimateNext(dt) # oftype Detection
        x_t = t.X[-1]

        # print("estimated: ", x_p)
        
        # 2. collect detections around prediction:
        t_i1 = x_p.t # already contains new time
        next_detections, next_detections_probs = self.collectWithin(t_i1, x_t) ####### WARNING!! THIS CAN CAUSE PROBLEMS (LIST ELEMENT TYPES COULD BE DETECTIONS) #######
        
        # change list of detections to matrix of detections coordinates
        n_det = len(next_detections) # number of detections (i)
        next_detections_X = np.zeros((n_det, 2))
        for i in range(n_det):
            next_detections_X[i] = next_detections[i].x

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
        
        d_t1 = TDet(x_t1,t_i1,v_t1,theta_t1) # next detection, it should probably be something else
        # return x_t1
        return (d_t1,x_p)
    
    # ----------------------------------
        
    
    def drawDsearchRegionAroundDetection(self, x, a=None, b=None, theta=None):
        # bounds = D.getSearchRegionBounds()
        s1, s2 = S1,S2
        if((b is not None) and (a is not None)):
            s1,s2 = a,b
        if(theta is None):
            theta = 0
        bounds = self.getSearchRegionBounds(x,s1,s2, theta)
        bounds = np.dot(bounds, [[0,1],[1,0]]) # flip coordinates
        coords2map(bounds, self.map, color=[255,0,255])
    
    # should be used for visualization only, is slow (O( (2*max(S1, S2)) ^2))
    def drawProbDistAroundDetection(self, x):
        # draw probability distribution function within detection area:
        s1s2 = np.max([S1,S2])
        for i in range(x[0]-s1s2,x[1]+s1s2,1):
            for j in range(x[0]-s1s2,x[1]+s1s2,1):
                if(self.isWithin((i,j),x)):
                    prob = self.getProb2( Detection((i,j)), TDet(x))
                    self.map[i][j] = prob
    
    
    # ------------------------------
    
    # method builds trajectory interatction matrix
    # [TODO] - should be tested, needs refactoring (move some things to separate methods)
    def buildQBPMatrix(self, tr_list: list[Trajectory], e1: float = 1.0, e2: float = 1.0):
        print("this is build QBP (og)")
        # 1. calculate q_ii terms ("merit terms")
        Q_ii = [] # list of q_ii (trajectory scores, "merit terms")
        for tr in tr_list:
            q_ii = 0 # merit term
            # calculate for every trajectory point in trajectory:
            
            S_err = 0
            # print("this is g: -->")
            
            dets_in_tr = tr.D # detections, that are part of trajectory (this is set)
            dets_in_tr_map = detSet2map(dets_in_tr) # (this is map of ^)

            for dets_i in dets_in_tr_map:
                dets = dets_in_tr_map[dets_i]
                g_k = tr.g_k(dets)
                if(g_k is None): # handled case (see method g_k)
                    continue
                # add to sum:
                S_err = S_err + ((1.0 - e2) + e2*g_k)

            # print("<-- this is end of g")
            
            # 2. add holes (S_model)
            q_ii = q_ii - e1*tr.holes + S_err
            

            Q_ii.append(q_ii)
            tr.S = q_ii # set/update trajectory score
        Q = np.diag(Q_ii) # make diagonal matrix
        
        
        # 2. calculate q_ij terms (interaction terms (similar to q_ii, but only consider intersecting trajectory points))
        n_tr = len(Q_ii)
        m = 0 # row index
        n = 0 # column index

        # I miss good old for loops from java so much ...
        while( m <= (n_tr-1)):
            n = m+1 # calculate only terms above diagonal, because Q is symmetric (Q[i,j] = Q[j,i])
            while( n <= (n_tr -1)):
                # 1. get points in intersection

                det_intersect = tr_list[m].D & tr_list[n].D
                det_intersect_map = detSet2map(det_intersect)

                # choose weaker hypothesis
                tr_l = n
                if(Q_ii[n] > Q_ii[m]):
                    tr_l = m
                tr_l = tr_list[tr_l]

                # print(Q_ii[n], Q_ii[m], "selected: ", tr_l.S)

                # 2. calculate g of intersecting points (with D of the weaker hypothesis)

                q_ij = 0
                S_err = 0
                # print("this is g_k: -->")
                for dets_i in det_intersect_map:
                    dets = det_intersect_map[dets_i]
                    # print(det_intersect_map)
                    g_kl = tr_l.g_k(dets)
                    if(g_kl is None): # handled case (see method g_k)
                        continue
                    # add to sum:
                    S_err = S_err + ((1-e2) + e2*g_kl)
                # print("<-- end of g_k")

                q_ij = S_err * (-0.5)

                # 3. set q_ij term (q_ij, q_ji)
                print(q_ij, end=" ", flush=True)
                Q[m,n] = q_ij
                Q[n,m] = q_ij
                n = n+1
            m = m+1
        print()

        # print(Q)
        return Q
    
    # method builds trajectory interatction matrix (same as ^, more clean)
    # recalculate_scores: calculate merit terms again (otherwise use old ones stored in Trajectory.S)
    def buildQBPMatrix2(self, tr_list: list[Trajectory], e1: float = 1.0, e2: float = 1.0, recalculate_scores=True):
        print("this is build QBP 2")
        # 1. calculate q_ii terms ("merit terms")
        Q_ii = [] # list of q_ii (trajectory scores, "merit terms")
        for tr in tr_list:
            q_ii = tr.S
            if(recalculate_scores):
                q_ii = tr.S
                if(not tr.disable_grow): # discontinued trajectories have same score
                    q_ii = tr.getScore(e1,e2)
            Q_ii.append(q_ii)
        
        Q = np.diag(Q_ii) # make diagonal matrix
        
        # 2. calculate q_ij terms (interaction terms (similar to q_ii, but only consider intersecting trajectory points))
        n_tr = len(Q_ii)
        m = 0 # row index
        n = 0 # column index

        # I miss good old for loops from java so much ...
        while( m <= (n_tr-1)):
            n = m+1 # calculate only terms above diagonal, because Q is symmetric (Q[i,j] = Q[j,i])
            while( n <= (n_tr -1)):
                # 1. calculate intersection cost (weaker hypothesis is detected within function)
                q_ij = tr_list[m].getInteractionCost(tr_list[n], e1, e2)

                # 2. set q_ij term (q_ij, q_ji)
                Q[m,n] = q_ij
                Q[n,m] = q_ij
                n = n+1
            m = m+1

        return Q
    

    # method solves qbp (returns list of selected hypotheses)
    def solveQBP(self, Q):
        # 1. init indicator vector
        m, n = np.shape(Q)

        v = np.zeros((m,1))
        incIndVec(v, rev=True) # start with 1 selected, not with 0
        

        # 2. find maximum by calculating all possible combinations (brute force method, should use solveQBP2)
        maximum = 0
        max_v = 1
        i=0
        for i in range((1<<(m))-1):
            current = np.dot(np.dot(v.T, Q), v)
            # print(current)
            # print(current, end="", flush=True)
            if(current > maximum):
                maximum = current
                max_v = i+1
            incIndVec(v, rev=True)
        print(maximum, "n_iter: ",i)
        return (binArrFromInt(max_v, m), maximum)
    
    # multibranch-ascent qbp solver (seems to work fine, for now):
    # basically bfs over specifically generated 0-1 space + some special conditions (see working notes)
    def solveQBP2(self, Q: np.array, debug=False):
        # 1. init variables
        n_el,_ = np.shape(Q) # length of vector - number of elements
        v = np.zeros(n_el).astype(np.uint8)

        # max:
        D_max = 0 # previous maximum

        depth = 0 # current depth, each new node gets value depth+1
        local_max_d = 0 # local max D, when reached new depth, update global with that
        local_max_v = v

        # init queue:
        queue = deque([(v,0,depth)]) # (v, n, d_current_max)

        # 2. main loop:
        n_iter = 0
        while((len(queue) != 0)):
            V, n, current_depth = queue.popleft()
            # if reached new depth: update global maximum with that of current depth
            if(current_depth > depth):
                depth = current_depth
                D_max = local_max_d
                if(debug):
                    print("depth: ", depth, " new max: ", D_max) # debug stuff

            # calculate score of current selection:
            d_current = np.dot(np.dot(V, Q),V)
            if(debug):
                print(V, ", D: ", d_current) # debug stuff
            
            # check if current is better than any other from upper level, if it is, update&generate, else skip
            if(d_current < D_max):
                continue # discontinue branch

            # update current depth maximum, if exceeded
            if(d_current >= local_max_d):
                local_max_d = d_current
                local_max_v = V.copy()

            # generate next nodes, put them into queue
            for i in range(n_el-n):
                v_i = n+i
                V[v_i] = 1
                queue.append((V.copy(), v_i+1, depth+1))
                V[v_i] = 0
            n_iter += 1
        
        
        print("n_iter: " + str(n_iter), " n_combinations: ", (1 << n_el)) # some stats
        # return (v_max, D_max)
        return (local_max_v, local_max_d)
    

# method builds trajectory interatction matrix - backup
# def buildQBPMatrix(self, tr_list: list[Trajectory], e1: float = 1.0, e2: float = 1.0):
#     # 1. calculate q_ii terms ("merit terms")
#     Q_ii = [] # list of q_ii (trajectory scores, "merit terms")
#     for tr in tr_list:
#         q_ii = 0 # merit term
#         # calculate for every trajectory point in trajectory:
        
#         S_err = 0
#         # print("this is g: -->")
        
#         dets_in_tr = tr.D # detections, that are part of trajectory (this is set)
#         dets_in_tr_map = detSet2map(dets_in_tr) # (this is map of ^)

#         for dets_i in dets_in_tr_map:
#             dets = dets_in_tr_map[dets_i]
#             g_k = tr.g_k(dets)
#             if(g_k is None): # handled case (see method g_k)
#                 continue
#             # add to sum:
#             S_err = S_err + ((1.0 - e2) + e2*g_k)

#         # print("<-- this is end of g")
        
#         # 2. add holes (S_model)
#         q_ii = q_ii - e1*tr.holes + S_err
        

#         Q_ii.append(q_ii)
#         tr.S = q_ii # set/update trajectory score

#     Q = np.diag(Q_ii) # make diagonal matrix
    
    
#     # 2. calculate q_ij terms (interaction terms (similar to q_ii, but only consider intersecting trajectory points))
#     n_tr = len(Q_ii)
#     m = 0 # row index
#     n = 0 # column index

#     # I miss good old for loops from java so much ...
#     while( m <= (n_tr-1)):
#         n = m+1 # calculate only terms above diagonal, because Q is symmetric (Q[i,j] = Q[j,i])
#         while( n <= (n_tr -1)):
#             # 1. get points in intersection

#             det_intersect = tr_list[m].D & tr_list[n].D
#             det_intersect_map = detSet2map(det_intersect)

#             # choose weaker hypothesis
#             tr_l = n
#             if(Q_ii[n] > Q_ii[m]):
#                 tr_l = m
#             tr_l = tr_list[tr_l]

#             # 2. calculate g of intersecting points (with D of the weaker hypothesis)

#             q_ij = 0
#             S_err = 0
#             # print("this is g_k: -->")
#             for dets_i in det_intersect_map:
#                 dets = det_intersect_map[dets_i]
#                 # print(det_intersect_map)
#                 g_kl = tr_l.g_k(dets)
#                 if(g_kl is None): # handled case (see method g_k)
#                     continue
#                 # add to sum:
#                 S_err = S_err + ((1-e2) + e2*g_kl)
#             # print("<-- end of g_k")

#             q_ij = S_err * (-0.5)

#             # 3. set q_ij term (q_ij, q_ji)
            
#             Q[m,n] = q_ij
#             Q[n,m] = q_ij
#             n = n+1
#         m = m+1

#     # print(Q)
#     return Q