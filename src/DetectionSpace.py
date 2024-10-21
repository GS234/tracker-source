from __future__ import annotations # da delajo tut type hint-i znotraj istega class-a
import numpy as np
from Trajectory import Trajectory
from Detection import Detection, TDet
import cv2 as cv
from helper_func import * # helper functions


# global vars:
S1, S2 = 15,15 # default s1,s2
DATA_ROOT = "../data/"
X_FFFFFF = [255,255,255]
GREEN = [0,255,0]

class DetectionSpace:
    def __init__(self, h,w, D: list = None):
        # self.map = np.zeros((n,n)).astype(np.float32) # init empty map (old way)
        
        self.map = np.zeros((h,w,3)).astype(np.uint8) # init empty map
        self.hw = (h,w) # map size (dimensions)
        self.window_name = "detection space"
        self.TR = [] # array for storing trajectories
        
        self.D = [] # detections (2d array, 1st dim. is time, subarrays contain detections)
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
        return [(b[0], b[1]) for b in X_rot.astype(np.int32).T]

    # !!! POMEMBNO:
    # check if detection d is within this detection's search region (seems to work fine)
    def isWithin(self, x, x_ref, a=S1, b=S2):
        rot_mat = self.getRotationMatrix() # get rotation matrix to rotate detection (easier calculation)
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
    # probability density function (bivariate normal distribution) (seems to work fine)
    def getProb(self, x, x_ref, a=S1, b=S2) -> float:
        # cov_mat = np.array([[s1,0.0],[0.0,s2]]).astype(np.float32)
        inv_cov_mat = np.array([[1.0/(3*a),0.0],[0.0,1.0/(3*b)]])
        
        rot_mat = self.getRotationMatrix()
        v = x - x_ref # x - mu
        v = np.dot(rot_mat.T, v) #un-rotate, so that it can be evaluated over un-rotated distribution

        pi_2, cov_mat_det = 2*np.pi , 9*a*b
        # return (1.0 / (np.sqrt( pi_2*pi_2 * cov_mat_det))) * np.exp(-0.5* np.dot( np.dot(v, inv_cov_mat), v)) # probability
        return np.exp(-0.5* np.dot( np.dot(v, inv_cov_mat), v)) # score (unscaled prob) (za vizualizacijo)
    
    # probability dist. around trajectory detection (also uses velocity and orientation)
    # [TODO] - stretch it according to velocity [TODO TODO TODO]
    def getProb2(self, d:Detection, td:TDet, a=S1, b=S2) -> float:
        inv_cov_mat = np.array([[1.0/(3*a),0.0],[0.0,1.0/(3*b)]])
        
        rot_mat = self.getRotationMatrix(theta=td.theta)
        v = d.x - td.x # x - mu
        v = np.dot(rot_mat.T, v) #un-rotate, so that it can be evaluated over un-rotated distribution

        pi_2, cov_mat_det = 2*np.pi , 9*a*b
        # return (1.0 / (np.sqrt( pi_2*pi_2 * cov_mat_det))) * np.exp(-0.5* np.dot( np.dot(v, inv_cov_mat), v)) # probability
        return np.exp(-0.5* np.dot( np.dot(v, inv_cov_mat), v)) # score (unscaled prob) (za vizualizacijo)
    # ----------------------------------------------------------------------------------------
    

    def showSpace(self):
        if(self.D):
            for d in self.D:
                detections2map(d, self.map, color=[0,0,0])
            last_dets = self.D[-1]
            
            print(last_dets)
            print()
            for d in last_dets: # draw bounding boxes around last detections
                self.drawBoundingBox(d.bb)
                self.drawX(d.x,1.0)
        # print(map)
        # print(self.map)
        cv.imshow(self.window_name, self.map)
        cv.waitKey(0)
        # while cv.getWindowProperty(self.window_name, cv.WND_PROP_VISIBLE) >= 1:
        #     cv.waitKey(1)
        # cv.destroyAllWindows()
    
    # collects detections within search region at time t
    # returns: DETECTIONS (has changed from coordinates, as we need those objects for trajectories), their probabilities
    def collectWithin(self, t:int, td: TDet):
        # search among detections in next time moment (next frame, that is (whichever, usually immediate successor (dt = 1)))
        next_detections = [] # store 'em in list
        next_detections_probs = [] # weights: sampled from distribution (bivariate normal dist, see Detection.getProb())
        # print(self.D)
        # print(t)
        if(t < len(self.D) and t >= 0): # check only if has detections in this layer (and not before 0 (negative indices overflow))
            for d_i in self.D[t]:
                if(self.isWithin(d_i.x, td.x)):
                    next_detections.append(d_i) # store detections, for now
                    next_detections_probs.append(self.getProb(d_i.x, td.x)) # get probability score from nearby point
        
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
        for i in range(n_det):
            next_detections_X[i] = next_detections[i].x
        
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
        
        d_t1 = TDet(x_t1,t_i1,v_t1,theta_t1) # next detection, it should probably be something else
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
        coords2map(bounds, self.map, overwrite=False)
    
    # method draws x instead of .
    def drawX(self, c:list, brightness=0.5) -> None:
        x_shape = np.array([
            [-3,-3],
            [-2,-2],
            [-1,-1],
            [0,0],
            [1,1],
            [2,2],
            [3,3],
            [-3,3],
            [-2,2],
            [-1,1],
            [1,-1],
            [2,-2],
            [3,-3],
        ])
        x_shape = x_shape + c # add origin
        p_list = [(x[1], x[0]) for x in x_shape]
        coords2map(p_list, self.map, color=X_FFFFFF)
    
    def drawLine(self, x1, x2, brightness: float = 1):
        x1 = np.array(x1)
        x2 = np.array(x2)
        n = (x2 - x1) # normal from x1 to x2
        # print(n.reshape( (2,1) ))
        n_len = np.sqrt(np.dot(n.T,n))
        if(n_len == 0): # if the same point, no need to draw :)
            return
        n = n/n_len
        n = n.reshape((2,1))
        
        values = np.arange(0, int(n_len), 0.1)
        values = values.reshape((1,len(values)))
        
        points = np.dot(n, values) + x1.reshape((2,1))
        p_list = [(int(x[1]), int(x[0])) for x in points.T]
        coords2map(p_list, self.map, color=[0,0,255], overwrite=True)
    
    # method draws bounding box in detection space
    def drawBoundingBox(self, bb: tuple, color: list = X_FFFFFF) -> None:
        # print(W, H)
        # 1. starting coordinate:
        y, x, h, w = bb
        # print(bb)

        points = []

        # print(image)
        # 2. draw horizontally:
        for i in range(w):
            x_i, y1_i, y2_i = x+i, y, y+h

            points.append((x+i, y))
            points.append((x+i, y+h))

        # 3. draw vertically:
        for i in range(h):
            points.append((x, y+i))
            points.append((x+w, y+i))
        coords2map(points, self.map, color=GREEN, overwrite=True)

    # should be used for visualization only, is slow (O( (2*max(S1, S2)) ^2))
    def drawProbDistAroundDetection(self, x):
        # draw probability distribution function within detection area:
        s1s2 = np.max([S1,S2])
        for i in range(x[0]-s1s2,x[1]+s1s2,1):
            for j in range(x[0]-s1s2,x[1]+s1s2,1):
                if(self.isWithin((i,j),x)):
                    prob = self.getProb((i,j),x)
                    self.map[i][j] = prob
    
    
    # ------------------------------
    
    # method builds trajectory interatction matrix
    # [TODO] - should be tested, needs refactoring (move some things to separate methods)
    def buildQBPMatrix(self, tr_list: list[Trajectory], e1: float = 1.0, e2: float = 1.0):
        # 1. calculate q_ii terms ("merit terms")
        Q_ii = []
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

        Q = np.diag(Q_ii) # make diagonal matrix
        
        
        # 2. calculate q_ij terms (interaction terms (similar to q_ii, but only consider intersecting trajectory points))
        n_tr = len(Q_ii)
        m = 0 # row index
        n = 0 # column index

        # I miss good old for loops from java so much ...
        while( m <= (n_tr-1)):
            n = m+1
            while( n <= (n_tr -1)):
                # 1. get points in intersection

                det_intersect = tr_list[m].D & tr_list[n].D
                det_intersect_map = detSet2map(det_intersect)

                # choose weaker hypothesis
                tr_l = n
                if(Q_ii[n] > Q_ii[m]):
                    tr_l = m
                tr_l = tr_list[tr_l]

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
                
                Q[m,n] = q_ij
                Q[n,m] = q_ij
                n = n+1
            m = m+1

        # print(Q)
        return Q
    

    # method solves qbp (returns list of selected hypotheses)
    def solveQBP(self, Q):
        # 1. init indicator vector
        m, n = np.shape(Q)

        v = np.zeros((m,1))
        incIndVec(v, rev=True) # start with 1 selected, not with 0
        

        # 2. find maximum by calculating all possible combinations (brute force method, should try something else in the future - [TODO])
        maximum = 0
        max_v = 1
        for i in range((1<<(m))-1):
            current = np.dot(np.dot(v.T, Q), v)
            # print(current)
            # print(current, end="", flush=True)
            if(current > maximum):
                maximum = current
                max_v = i+1
            incIndVec(v, rev=True)
        
        return max_v