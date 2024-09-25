from __future__ import annotations # da delajo tut type hint-i znotraj istega class-a
import numpy as np
import random
import math
import cv2 as cv

# global vars:
S1, S2 = 30,30 # default s1,s2

# test trajectories:
t1 = [
    [(122, 52)],
    [(115, 64)],
    [(120, 79),(112, 81),(116, 83),(116, 84),(114, 87)],
    [(110, 98),(104, 103),(105, 105),(105, 106),(101, 109),],
    [(104, 126),(99, 128) ,(98, 129) ,(99, 129) ,(100, 132)],
    [(114, 146),(114, 147)],
    [(128, 173),(131, 174),(132, 174)],
    [(148, 191),(152, 191)],
    [(146, 207),(145, 210),(148, 210),(145, 211),(148, 211)],
    [(146, 226)],
    [(141, 232)],
    [(133, 232),(133, 234),(133, 236)],
    [(123, 234)],
    [(114, 224)],
    [(103, 210),(103, 211),(104, 212)],
    [(96, 215)],
    [(89, 223),(89, 226),(87, 228),(91, 228),(91, 230),(90, 231)],
    [(93, 242)],
    [(105, 256)],
    [(113, 270),(113, 272)],
    [(118, 285)],
    [(123, 287)],
    [(134, 284)],
    [(145, 294),(143, 295),(146, 296)],
    [(158, 312)],
    [(156, 326)],
    [(160, 338),(160, 339)],
    [(155, 352),(155, 354)],
    [(150, 364)],
    [(144, 373),(143, 374),(142, 375),(140, 376)],
    [(126, 384)],
    [(109, 387),(108, 388),(110, 389)],
    [(97, 398)],
    [(96, 408),(96, 409)],
    [(85, 424),(85, 425),(84, 426),(85, 426)],
    [(80, 439)],
    [(94, 446)],
    [(116, 452),(117, 452)],
    [(139, 440),(137, 441),(136, 442),(137, 442)],
    [(149, 439)],
    [(163, 436)],
    [(188, 426)],
    [(204, 409)]
]

t3 = [
    [(275, 114)],
    [(265, 135), (266, 135), (266, 136), (266, 137)],
    [(273, 152), (274, 152)],
    [(282, 170)],
    [(283, 194)],
    [(291, 214), (292, 214)],
    [(306, 224)],
    [(313, 236), (314, 237), (314, 238)],
    [(314, 254), (314, 255), (314, 256), (314, 257)],
    [(304, 277), (304, 278), (303, 279), (301, 280), (302, 280), (300, 281), (304, 281), (305, 289)],
    [(287, 296)],
    [(277, 309), (277, 310)],
    [(286, 319), (287, 320), (288, 320), (288, 321), (289, 321)],
    [(306, 330), (306, 331), (306, 332)],
    [(297, 349), (298, 349)],
    [(303, 362)],
    [(289, 374), (290, 374), (287, 375), (288, 375), (286, 376)],
    [],
    [],
    [],
    [(255,420)]
]



t2 = [
    [(250,249)],
    [(240,240),(260,240)],
    [(240,240),(260,240),(253,240),(256,240)]
]

t4 = [
[(262, 201)],
[(254, 206)],
[(247, 215)],
[(238, 221),(242, 226),(230, 210)],
[(229, 233)],
[(219, 245)],
[(207, 253)],
[(202, 262)],
[(198, 269)],
[(193, 274)],
[(182, 280)]
]


def coords2det2(X):
    detections = []
    i = 0
    for x_t in X:
        det_arr = []
        
        for x in x_t:
            det_arr.append(Detection(x,i))
        detections.append(det_arr)
        i = i+1
    return detections

# -----------------


# detection class; if becomes too complicated, move it to other file
class Detection:
    # constructor: position, velocity, timestamp (position, velocity are only used for visualization and legacy reasons)
    def __init__(self, x, t=0):
        self.x = np.array(x)
        self.t = t #time stamp

    def __str__(self):
        # return "d{x="+str(self.x)+",v="+str(self.v)+",t="+str(self.t)+",theta="+str(self.theta)+"}"
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
        # return "td{x="+str(self.x)+",t="+str(self.t)+",v="+str(self.v)+",theta="+str(self.theta)+"}"
        # return "td{x="+str(self.x)+",t="+str(self.t)+",v="+str(self.v)+"}"
        return "td{x="+str(self.x)+",t="+str(self.t)+"}"
    


# detection space
class DetectionSpace:
    def __init__(self, n, D: list = None):
        self.n = n # map size (dimensions)
        self.map = np.zeros((n,n)).astype(np.float32) # init empty map
        self.window_name = "detection space"
        self.TR = [] # array for storing trajectories
        
        self.D = []
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
    # [TODO] - stretch it according to velocity
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
                detections2map(d, self.map)
        # print(map)
        cv.imshow(self.window_name, self.map)
        while cv.getWindowProperty(self.window_name, cv.WND_PROP_VISIBLE) >= 1:
            cv.waitKey(1)
        cv.destroyAllWindows()
    
    # collects detections within search region at time t
    # returns: DETECTIONS (has changed from coordinates, as we need those objects for trajectories), their probabilities
    def collectWithin(self, t, td: TDet):
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
    def estimateNext2(self, d_last: TDet, d_pred: TDet, next_detections, next_detections_probs, dt=1) -> TDet:
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
        L = 40 # lambda: temporal discount ([TODO] - un-hardcode) (should be large)
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
            theta_t1 = np.arccos(cos_theta) # co-domain is only from 0-pi, not a problem, because ellipse is symmetrical (so essentialy v ~ -v)
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
        p_list = [(x[0], x[1]) for x in x_shape]
        coords2map(p_list, self.map, brightness=brightness)
    
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
        p_list = [(int(x[0]), int(x[1])) for x in points.T]
        coords2map(p_list, self.map, brightness, overwrite=False)


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
            for td in tr.X:
                g_k = tr.g(td.t) # throws indexOutOfBounds!
                if(g_k is None):
                    continue # do not add anything if has no detections (see method Trajectory.g(...))
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
            print(current, end="", flush=True)
            if(current > maximum):
                maximum = current
                max_v = i+1
            incIndVec(v, rev=True)
        print("\n")
        
        print(maximum)
        print(max_v)
        return max_v


        

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
        t_n = len(self.detectionSpace.D)
        n_holes_total = 0 # skupno stevilo lukenj
        n_holes = 0 # stevilo zaporednih lukenj
        n_holes_max = 5 # najvecje steivlo zaporednih lukenj

        td_connected = [] # list of connected trajectory detections
        
        td_current = td_orig
        for i in range(t_n):
            # 1. detection
            # td_current

            # 2. look for next detections (current v, theta) (calculate estimation, look for detections inside its region)
            # 2.1 estimate next point:
            td_pred = self.estimateNext(td_current, dt)
            # print(td_pred.v)
            # 2.2 find next detections:
            next_dets, next_probs = self.detectionSpace.collectWithin(td_pred.t, td_current) # collect in next frame (t+1)

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
        # print("lukne: ",n_holes)
        # print("td_with: ",td_connected, "holes: ", n_holes)
        if(len(td_connected) >= n_holes):
            self.holes = n_holes_total - n_holes
            for _ in range(n_holes):
                # print("brisem lukno")
                # if(len(td_connected) > 0):
                td_connected.pop()
        # print("td_connected: " ,td_connected)
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
        print(self.X)
        

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

        # 2. get each detection's probability in image (we get that from detector)
        p_hi = 1

        # 3. calculate g:
        #   option 1:  g = p*(Hk,tk|Itk) + SUM(a)[   log(p(Ha,tk|H))   ] -> cumulative log error of all detections around trajectory point at t
        result = p_hi 
        if(len(dets) == 0):
            return None # we should not add anything to this, this case should be handled
        
        result = result + np.sum( np.log(probs) )
        return result
    

    # method calculates g_k of detections in list (used for intersecting detections) ()
    # [TODO] - (untested)
    def g_k(self, dets: list[Detection]):
        probs = self.getDetProbs(dets)
        p_hi = 1
        result = p_hi 
        if(len(dets) == 0):
            return None # this case should be handled
        result = result + np.sum( np.log(probs) )
        return result
    
    
    # method gets probabilities of detections in list around corresponding trajectory point
    # [TODO] - might need to test it if it works (untested)
    def getDetProbs(self, det_list: list[Detection]):
        t_off = self.X[0].t # time of first detection, is used to calculate relative index of point in trajectory
        probs = []
        
        for d in det_list:
            td = self.X[d.t - t_off]
            # print(td, det_list)
            d_prob = self.detectionSpace.getProb2(d,td)
            probs.append(d_prob)
        return probs
    
    # POMEMBNO!!


        
    
    def __str__(self):
        return "{t"+str(self.id)+", len="+str(len(self.X))+"}"

    def __repr__(self):
        return self.__str__()
                
        

# helper functions:
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
        
        # map[x,y]= 1.0 - (1.0 / (1.0+(d.t/20.0)))*0.9 # more correct, as more recent detections should be brighter
        map[x,y]= 1.0
        # map[x,y]= (1.0 / (1.0+(d.t/8))) # I like this more, but is not correct because of ^

def coords2map(X, map, brightness=0.5, overwrite=True):
    n = np.shape(map)[0]
    for point in X:
        x, y = point
        x = x%n
        y = y%n
        if(overwrite or map[x,y] == 0):
            map[x,y]= brightness



def random_coords(n):
    x = int(random.random()*n)
    y = int(random.random()*n)
    return (x,y)

# like random_coords, but generates all at once and can use seed
def random_matrix(shape, seed):
    rng = np.random.default_rng(seed)
    return rng.random(shape)

def n_random_coords(n, N=1, seed=None):
    X = (random_matrix((N,2), seed)*n).astype(np.int32).T # generate random x and y coordinates
    print(X)
    return list(zip(X[0],X[1]))

def detSet2map(dets: set[Detection]):
    detMap = {}
    for d in dets:
        try:
            detMap[d.t].append(d)
        except KeyError:
            detMap[d.t] = [d]
    return detMap

def incIndVec(vec: list[int], rev=False):
    c = 1
    n = len(vec)

    rev_i = 1
    if(rev):
        rev_i = 0

    for i in range(n):
        # i_i = rev_i*(n-1-i) + (1-rev_i)*i # branchless
        i_i = rev_i*(n-1-(i<<1))+i # branchless optimized :)

        vec[i_i] = vec[i_i] + c
        if(vec[i_i] == 2):
            vec[i_i] = 0
            c = 1
        else:
            break





# -------------------------

# debug trajectory

# total points: 4
tdeb_1 = [
    [(250,250)], # 0
    [(255,255)], # 1
    [(260,255),(265,260)], # 2
    [(265,265)], # 3
    [],
    [(270,270)], # 4
]



# main:
def main():
    n = 500 # canvas size
    seed = 42
    
    # D = coords2det2(t1)
    # D = coords2det2(t4)
    D = coords2det2(tdeb_1)


    # create detection space object:
    dspace = DetectionSpace(n)
    dspace.D = D

    # d0 = D[3][0]
    # d1 = D[3][1]
    # d2 = D[3][2]
    # d3 = D[0][0]
    # print(d0)

    dd0 = D[3][0]

    # d1,d2 = D[3][0],D[3][1]

    # th1 = Trajectory(d0, dspace)
    # th1.build()
    # th1.drawToSpace()
    
    # th2 = Trajectory(d1, dspace)
    # th2.build()
    # th2.drawToSpace()
    
    # th3 = Trajectory(d2, dspace)
    # th3.build()
    # th3.drawToSpace()
    
    # th4 = Trajectory(d3, dspace)
    # th4.build()
    # th4.drawToSpace()

    # detIntersect = th1.D & th2.D
    # print(detIntersect)
    # detIntMap = detSet2map(detIntersect)
    # print(detIntMap)


    td1 = Trajectory(dd0, dspace)
    td1.build()
    td1.drawToSpace()

    # Q = dspace.buildQBPMatrix([th4, th1, th2, th3], 0.5, 0.1)
    # Q = dspace.buildQBPMatrix([th1, th4], 0.1, 0.001)
    Q = dspace.buildQBPMatrix([td1, td1], 1., 1.)
    # print("lukne: ",td1.holes)
    print("Q:\n", Q)
    dspace.solveQBP(Q)
    
    
    # dspace.drawLine((10,10), (100,70))
    # dspace.drawX([235,430], 1)
    
    dspace.showSpace()

    # v = np.zeros(8)
    # for i in range(257):
    #     print(v)
    #     incIndVec(v)




# -----------

if __name__ == "__main__":
    main()

# def main():
#     n = 500 # canvas size
#     seed = 42
    
#     # D = coords2det2(t1)
#     D = coords2det2(t1)
#     D2 = coords2det2(t3)

#     offset = 5
#     for i in range(len(D2)):
#         for j in range(len(D2[i])):
#             D2[i][j].t = i+offset # lval + assignment: mem = mov
#             D[i+offset].append(D2[i][j])

    
#     # create detection space object:
#     dspace = DetectionSpace(n)
#     dspace.D = D

#     d0 = D[0][0]
#     # print(d0)
#     d0_1 = D2[0][0]
#     th1 = Trajectory(d0)
#     th2 = Trajectory(d0_1)
#     dspace.buildTrajectory(th1)
#     dspace.buildTrajectory(th2)

#     # dspace.drawLine((10,10), (100,70))
#     # dspace.drawX([235,430], 1)
    
#     dspace.showSpace()



# backup
# # !!! POMEMBNO [TODO - fix/adjust/modify/test]
#     # estimates next point in trajectory given collected detections and prediction (x_t -> x_t+1)
#     def estimateNext(self, t:Trajectory, dt=1) -> Detection:
#         # 1. predict next point from current detection
#         x_p = t.estimateNext(dt) # oftype Detection
#         x_t = t.X[-1]
#         # print("estimated: ", x_p)
        
#         # 2. collect detections around prediction:
#         t_i1 = x_p.t # already contains new time
#         next_detections_X, next_detections_probs = self.collectWithin(t_i1, x_t)

#         # 3. compute weighted mean (prediction + all detections) to determine actual next point
#         # temporal discount, as used in paper [pami, leibe et al. ...] = e^-lambda
#         L = 40 # lambda: temporal discount ([TODO] - un-hardcode) (should be large)
#         p_tempDisc = np.exp(-L)

#         # calculate normalization factor Z (sum of all weights):
#         Z = np.sum(next_detections_probs)+p_tempDisc
#         # print("Z: ",Z)

#         x_t1 = np.array((1/Z) * ( p_tempDisc * x_p.x + np.dot(next_detections_probs, next_detections_X) )).astype(np.int32)
#         # print(x_t1)
#         # print("x(t+1): ",x_t1)

#         # 4. also estimate velocity, angle (based on x_t+1: no need to calculate weights, estimates again, as they are the same)
#         # velocity: v_t1 = sqrt( dolzina vektorja (x_t1 - x_t) )
        
#         x_dif = x_t1 - x_t.x
        
#         # print(x_t.x, x_t1, x_dif)
#         x_dif = x_dif/dt # we do that here, velocity is then simply it's length
#         v_t1 = np.sqrt(np.dot(x_dif,x_dif))

#         # theta:
#         # calculate relative to unit base vector x_i
#         # base_x = np.array([1.0,0.0]) # -> not needed, see notes
#         # print("theta: ",t.theta)
#         theta_t1 = t.theta
#         if(v_t1 != 0): # only if it has speed this is relevant
#             cos_theta = x_dif[0] / v_t1  # this is it, just trust me bro
#             theta_t1 = np.arccos(cos_theta) # co-domain is only from 0-pi, not a problem, because ellipse is symmetrical (so essentialy v ~ -v)
#         if(x_dif[1] < 0): # same angle is computed for both sides, because we only compare magnitude, so correction is needed in some cases
#             theta_t1 = -theta_t1
        
#         d_t1 = Detection(x_t1,v_t1,t_i1,theta_t1) # next detection, it should probably be something else
#         # return x_t1
#         return (d_t1,x_p)

# backup: buildTrajectory
# def buildTrajectory(self, t: Trajectory, dt=1):
#         t_n = len(self.D)
#         n_holes_total = 0 # skupno stevilo lukenj
#         n_holes = 0 # stevilo zaporednih lukenj
#         n_holes_max = 5 # najvecje steivlo zaporednih lukenj
        
#         dt_next = 1
#         dt_prev = -1

#         for i in range(t_n):
#             # 1. detection
#             d_current = t.X[-1] #take last/first                                                                                <-
#             x_t = d_current.x # current point in trajectory (last in array)

#             # 2. look for next detections (current v, theta) (calculate estimation, look for detections inside its region)
#             # 2.1 estimate next point:
#             # d_pred = t.estimateNext(-1)
#             d_pred = t.estimateNext(dt=dt_next)
#             # 2.2 find next detections:
#             # self.drawDsearchRegionAroundDetection(x_t, S1, S2, t.theta)
#             next_dets, next_probs = self.collectWithin(d_pred.t, d_current) # collect in next frame (t+1)

#             # add detections (objects, not just coords) to trajectory set (for intersections with other trajectories)
#             t.D.update(next_dets)

#             if(len(next_dets) == 0):
#                 n_holes += 1
#                 n_holes_total += 1
#                 # print("empty")
#                 if(n_holes >= n_holes_max):
#                     t.holes = n_holes_total
#                     # print("maximum no. of sequential holes reached, ending trajectory")
#                     break
#             else:
#                 n_holes = 0

#             # 3. estimate next detection: weighted mean of detections
#             # d_next, _ = self.estimateNext(t)
#             d_next, _ = self.estimateNext2(d_current,d_pred,next_dets,next_probs,t.theta, dt=dt_next)
#             # self.drawX(d_next.x,1.0)
#             self.drawLine(x_t, d_next.x, 0.4)
#             # self.drawX(d_p.x, 0.8)
#             # print(d_next)
#             # print(d_next)

#             # 4. add calculated estimate to trajectory
#             t.X.append(d_next)                                                                                                  #<-
#             t.v = d_next.v
#             t.theta = d_next.theta
#             # t.theta = 0
#             # repeat loop
        
#         self.TR.append(t)
#         # print(t.D)
#         # print(self.TR)