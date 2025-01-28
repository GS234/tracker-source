from helper_func import *
from DetectionSpace import DetectionSpace
from Trajectory import Trajectory
from Detection import Detection
import cv2 as cv
import numpy as np
import copy # for deepcopy (visualization purposes)
import pickle
np.set_printoptions(suppress=True, precision=3, linewidth=1000)

DATA_ROOT = "../data/"
FRAMES_PATH = "frames/LaSOT_bird-2/color/"

USE_PRECOMPUTED = True # use precomputed optical flow (set to False to calculate it on the go) (flows-path must be set)
FLOWS_PATH = "flow_est/LaSOT_bird-2_1/"

DRAW_DETS=False
MAIN_DEB= not  True
# EXT_THR=10 # maximum number of extrapolation of trajectories (# of consecutive frames without detections for that trajectory (number of relative holes, essentially))
EXT_THR=5 
PO_THR=7 # possibly occluded threshold: number of holes before trajectory is marked as possibly occluded
UNSELECTED_STRIKE_MAX=5 # maximum number of times that trajectory is not selected but included in set
SAVE_TRAJECTORIES=not True # switch to save trajectories on every selection step for vizualization/debug purposes
T_INIT_S = 0.5


# some main-specific functions:
def setFrameFlowAtI(dspace: DetectionSpace, i: int):
    # set new frame
    frame = getFrameAtI(i,DATA_ROOT+FRAMES_PATH)
    dspace.lastFrame = frame.copy()
    dspace.map = frame
    
    # set new flow
    flow_from = getFlowAtI(i, DATA_ROOT+FLOWS_PATH)
    dspace.flow_map = flow_from
    flow_img = flow2img(flow_from)
    dspace.last_flow_img = flow_img.copy()
    dspace.flow_img = flow_img
    
    #  also set flow to this image
    if(i >= 1):
        flow_to = getFlowAtI(i-1, DATA_ROOT+FLOWS_PATH)
        dspace.flow_to = flow_to
    else: # there is no flow into first frame
        h,w = dspace.hw
        dspace.flow_map=np.zeros((h,w,2)).astype(np.uint8) # optical flow (outta this frame)

# II. qbp-specific functions:
# function merges trajectories in tr_list into single trajectory
def mergeTrajectories2(tr_list:list[tuple[Trajectory, float, float]]) -> Trajectory:
    t_merged = None # first
    # sort 'em by time:
    tr_list.sort(key=lambda x: x[0].X[0].t)
    
    min_id = 0
    merge_scores = []
    for p in tr_list:
        t = p[0]
        score = p[1]
        # print(p)
        score2 = p[2] # penalty is calculated from this
        if(t_merged is None):
            t_merged = t.getCopy()
            min_id = t.id
        else:
            # must do: X, D2
            # should do: holes, holes_ref, term, color
            t.X[0].color = [10,10,150]
            t_merged.X = t_merged.X + t.X
            t_merged.D2 = {**t_merged.D2, **t.D2}
            t_merged.T2 = {**t_merged.T2, **t.T2} # also merge T2

            t_merged.holes = t_merged.holes + t.holes
            t_merged.term = t.term
            if(t.id < min_id):
                min_id = t.id
            
        t_merged.T2[t] = [score, score2]
        merge_scores.append(t_merged.T2[t])
    return t_merged

# function generates and returns all possible connections with other trajectories (similar to extend, but it works with whole trajectories now)
def getMergedHypotheses(tr_list: list[Trajectory], time_window=20, max_space_diff=100.0, type=1):
    # print("this is getMergedHypotheses: ")
    mtr_hypotheses = []
    for tr in tr_list:
        tr_possible_next = []
        if(type==1):
            tr_possible_next = tr.getPossibleNext2(tr_list, time_window=time_window)
        if(type == 2):
            tr_possible_next = tr.getPossibleNext3(tr_list, DATA_ROOT+FLOWS_PATH, time_window=time_window)
        all_possible_next = extendAllPossibleNext(tr, [(tr,T_INIT_S,1.0)],tr_possible_next,tr_list, type=type)
        mtr_hypotheses = mtr_hypotheses + all_possible_next
    return mtr_hypotheses

# metod extends trajectories
# t: current trajectory
# collected: to be merged
# possible next: possible next trajectories that current can 'see'
# all_trs: all trajectories (because we do not have it in dspace)
def extendAllPossibleNext(t:Trajectory, collected:list[tuple[Trajectory, float]], possible_next: list[tuple[Trajectory, float]], all_trs: list[Trajectory], type=1, time_window=20):
    # 1. check if there are no more possible next
    if(not possible_next):
        # merge 'em
        # print(collected)
        t_merged = mergeTrajectories2(collected)
        skipped_time = 0
        time_l = collected[0][0].X[0].t
        time_h = time_l
        for i in range(len(collected)):
            c = collected[i]
            t:Trajectory = collected[i][0]
            time_h = t.X[-1].t

            print("<t%d (c=%3d), %4.2f, f=%s>" % (t.id, t.color[2], c[1], str(t.term)), end=", ")
            if(i != 0):
                t_p = collected[i-1][0].X[-1].t
                t_c = collected[i][0].X[0].t
                skipped_time = skipped_time + (t_c - t_p)

        total_time = time_h-time_l
        print("-> <t%d, %6.2f>, total time: %d, time skipped: %d (%4.2f) " % (t_merged.id, t_merged.getScoreII(), total_time, skipped_time, (float(skipped_time) / (float(total_time)+0.0000001)) ))
        # return it
        return [t_merged]

    mtr_hypotheses = []
    # 2. possible_next_from_this = extendAllPossibleNext()
    # print(possible_next)
    for p in possible_next:
        # tr_possible_next = tr.getPossibleNext(all_trs, max_space_diff=50)
        tr = p[0]
        tr_possible_next = []
        if(type==1):
            tr_possible_next = tr.getPossibleNext2(all_trs, time_window=time_window)
        if(type==2):
            tr_possible_next = tr.getPossibleNext3(all_trs, DATA_ROOT+FLOWS_PATH, time_window=time_window)
        collected.append( p ) # add it
        possible_next_trs = extendAllPossibleNext(tr, collected, tr_possible_next, all_trs, time_window=time_window)
        collected.pop() # remove it
        mtr_hypotheses = mtr_hypotheses + possible_next_trs
    
    # 3. return all collected trajectories
    return mtr_hypotheses


# debug main:
def main():
    # INIT - get through first n frames and initiate (hopefully) strong trajectories:
    # init variables used in process
    
    # SEQUENCES: uncomment for different sequences: ------------------------
    
    # sample of penguins file:
    # FRAMES_PATH = "sample/"
    # FLOW_FILE = "sample/sample_flo_2.p"
    # DETS_FILE = "sample/dets.txt"
    # T_OFFSET = 0 # constant: starting time (t_global) (used also to align DetectionSpace.D indices (as it expects t0 at index 0))
    # n = 0 # number of previous frames for trajectory init
    
    # penguins:
    FRAMES_PATH = "frames/LaSOT_bird-2/color/"
    FLOWS_PATH = "flow_est/LaSOT_bird-2_1/"
    DETS_FILE = "detections/LaSOT_bird-2.txt"
    # T_OFFSET = 500 + 653 # constant: starting time (t_global) (used also to align DetectionSpace.D indices (as it expects t0 at index 0))
    # T_OFFSET = 1160 # constant: starting time (t_global) (used also to align DetectionSpace.D indices (as it expects t0 at index 0))
    T_OFFSET = 1 # constant: starting time (t_global) (used also to align DetectionSpace.D indices (as it expects t0 at index 0))
    # T_OFFSET = 798
    # T_OFFSET = 1
    n = 0 # number of previous frames for trajectory init

    # testing squares:
    # FRAMES_PATH = "testing/sequence/"
    # FLOWS_PATH = "testing/flows/"
    # DETS_FILE = "testing/sequence/dets.txt"
    # T_OFFSET = 0 # constant: starting time (t_global) (used also to align DetectionSpace.D indices (as it expects t0 at index 0))
    # n = 0 # number of previous frames for trajectory init

    # ----------------------------------------------------------------------

    D13 = readDetFile2(DATA_ROOT+DETS_FILE) # read detections from file
    t_global = T_OFFSET # current time (frame)
    
    # actual offset for detections list is: T_OFFSET - n (because there are some detections before global offset (initial trajectories))
    n_res = len(D13) - n - T_OFFSET
    tr: list[Trajectory] = []
    tr_fin: list[Trajectory] = []
    n_stage_II_III = 120 # 5 # 20 # 'time window' - # of frames between trajectory selections
    stage_II_III_timew = 20 # time window for merging of trajectories
    # ------------------------------
    
    
    # INIT DSPACE
    frame = getFrameAtI(t_global,DATA_ROOT+FRAMES_PATH)
    h,w,_ = np.shape(frame)
    dspace = DetectionSpace(h,w, time_offset=(T_OFFSET-n)) # also pass time offset for using correct indices
    dspace.use_flow = True
    
    # set last frame
    dspace.lastFrame = frame.copy()
    dspace.map = frame

    # set flow map
    flow = getFlowAtI(t_global, DATA_ROOT+FLOWS_PATH)
    flow_img = flow2img(flow)
    dspace.flow_map = flow
    dspace.last_flow_img = flow_img.copy()
    dspace.flow_img = flow_img
    
    print("init dspace: ", dspace.D)
    D13_init = D13[t_global] # init with those from current time step
    dspace.D.append(D13_init)
    t_global = t_global+1
    # END INIT DSPACE
    
    # INIT TRAJECTORIES
    for d in D13_init:
        t = Trajectory(d,dspace,getTrColor())
        t.build2()
        tr.append(t)
    # END INIT TRAJECTORIES
    
    # quick visualization
    for t in tr:
        t.drawToSpace()
    dspace.showSpace(draw_dets=DRAW_DETS)
    

    # EXTEND
    skip_n = 0
    try:
        sslm = 0 # steps since last merge
        for i in range(n_res):
            print("[@"+str(t_global)+"] \/\/\/\/\/\/\/\/")
            
            # 1. init new frame:
            next_dets=D13[t_global]
            print("next dets: ",next_dets)
            dspace.D.append(next_dets)
            setFrameFlowAtI(dspace,t_global)
            # ---

            # 2. extend trajectories
            print("n_tr: ",len(tr))
            used_dets: set[Detection] = set()
            tr_next: list[Trajectory] = []
            print("[EXTENDING TRAJECTORIES]")
            for t in tr:
                tr_next_from_same: list[Trajectory] = []
                if((not t.term) and (t.holes_ref <= EXT_THR)): # extend only, if not terminated
                    # 2.1. add current (note: this one gets extended)
                    tr_next_from_same.append(t)
                    
                    # 2.2. also include current trajectory (not extended), add it to hypothesis selection
                    t_prev = t.getCopy(deep=False) 
                    t_prev.term = True # terminate it, so it does not extend
                    tr_next_from_same.append(t_prev)

                    # 2.3. extend current, also add forks, if they occur
                    used_dets_current, forked_tr = t.extend4() # need used dets to start new trajectories from unused ones
                    tr_next_from_same = tr_next_from_same + forked_tr
                    
                    # 2.4. update set of used dets (obtained from extend method)
                    used_dets.update(used_dets_current)
                else:
                    # store terminated trajectories in separate list
                    t.term = True
                    tr_fin.append(t)
                tr_next = tr_next + tr_next_from_same
                
                # some debug prints
                first = True
                for tt in tr_next_from_same:
                    if(first):
                        first = False
                        print(">>> t"+str(tt.id)+" - origin:",tt.origin,"score:",tt.getScore2(), "color:",tt.color, tt.X[-2:], "len:",len(tt.X))
                    else:
                        print("|-> t"+str(tt.id)+" - origin:",tt.origin,"score:",tt.getScore2(), "color:",tt.color,tt.X[-2:], "len:",len(tt.X))
                # ---
            print("[DONE EXTENDING]")
            # ---

            tr = tr_next

            # 3. start new trajectories from unused dets
            unused_dets = set(next_dets) - used_dets # difference of sets
            for d in unused_dets:
                t_new = Trajectory(d, dspace)
                t_new.build2()
                tr.append(t_new)
            # ---

            # 4. hypothesis selection
            print("building Q")
            tr.sort(key=lambda x: x.getScore2(), reverse=True) # sort to minimize chance of getting stuck in some local minimum (there ARE issues with qbp-solver)
            
            # 4.1 build Q
            Q = dspace.buildQBPMatrix3(tr)
            print(Q)

            # 4.2 solve QBP
            v = dspace.solveQBP2(Q)
            print(v)
            # ---

            # 5. keep only selected trajectories for next frame:
            tr_next2: list[Trajectory] = []
            print("selected: ", end="")
            for ii in range(len(v[0])):
                if(v[0][ii] == 1):
                    tr_i = tr[ii]
                    tr_next2.append(tr_i)
                    print("t"+str(tr_i.id),end=" ", flush=True)
            tr = tr_next2
            print()
            # ---

            # 6. visualization
            print("[VISUALIZATION]")
            t_i = 0
            print("fin (%d):"%(len(tr_fin)))
            for t in tr_fin:
                print("%3d. t%-5s o=[%-3d,%-3d], c=%-3d, len=%-4d, score=%9.4f %5d - %-5d, f:%s" % (t_i,t.id, t.origin.x[0],t.origin.x[1], t.color[2], len(t.X), t.getScore2(), t.X[0].t, t.X[-1].t, str(t.term)))
                t.drawToSpace(color=[100,100,100])
                t_i = t_i+1
            print("not fin (%d):"%(len(tr)))
            for t in tr:
                print("%3d. t%-5s c=%-3d, len=%-4d, score=%9.4f %5d - %-5d, f:%s" % (t_i,t.id, t.color[2], len(t.X), t.getScore2(), t.X[0].t, t.X[-1].t, t.term))
                t.drawToSpace()
                t_i = t_i+1
            # ---
            
            # 7. on every n_stage_II_III-th frame, merge trajectories collected so far:
            order=[2,1]
            sslm = sslm+1
            print("[!]",sslm)
            if(n_stage_II_III == sslm or len(tr_fin) >= 25):
                print("merge trajectories: --------------------------------------------------------------------------------")
                merged_merged:list[Trajectory] = []
                tr_all = tr_fin+tr_next2
                
                # 7.1. stage II (bridged) (is faster)
                merged_trs1 = getMergedHypotheses(tr_all, type=order[0], time_window=stage_II_III_timew)
                

                # 7.2. get unused trajectories
                tr_left = set(tr_all)
                i = 0
                for t in merged_trs1:
                    key_set = set(t.T2.keys())
                    if(len(key_set) > 1):
                        tr_left = tr_left - key_set
                    i = i+1
                tr_left = list(tr_left)

                # 7.3. stage III (flow) with unused trajectories (is slower)
                merged_trs2 = getMergedHypotheses(tr_left, type=order[1], time_window=stage_II_III_timew)
                
                
                
                # 7.4. select best trajectories to be used in new round
                merged_merged = merged_trs1+merged_trs2
                # merged_merged = merged_trs2
                Q = dspace.buildQBPMatrixX(merged_merged, type=2)


                
                # SOLVE QBP:
                print(Q)
                np.savetxt("Q.txt",Q, fmt="%7.3f")
                res = dspace.solveQBP2(Q)
                # res = dspace.solveQBP(Q)
                print(res)
                v = res[0]
                
                # SHOW SELECTED:
                selected_merged = []
                dspace.clearSpace()
                for i in range(len(v)):
                    if(v[i] == 1):
                        selected_merged.append(merged_merged[i])
                        merged_merged[i].drawToSpace()
                        print("t%d"%(merged_merged[i].id), end=", ")
                print()

                i_t = 0
                for t in selected_merged:
                    print("%3d. t%-5s o=[%-3d,%-3d], c=%-3d, len=%-4d, score=%9.4f %5d - %-5d (fin:%s)" % (i_t,t.id, t.origin.x[0],t.origin.x[1], t.color[2], len(t.X), t.getScore2(), t.X[0].t, t.X[-1].t, str(t.term)))
                    i_t = i_t+1

                # 7.5. use new trajectories in next round
                tr = tr+selected_merged
                tr_fin = []


                print("------------------- --------------------------------------------------------------------------------")
                
                d2_winname = "merged"
                cv.namedWindow(d2_winname, cv.WINDOW_NORMAL)
                dspace.showSpace(draw_dets=DRAW_DETS, draw_last_dets_bb=False, dspace_winname=d2_winname)
                sslm = 0
            # ---

            # some loop-related technical stuff
            # if skip_n is set (non zero), algorithm runs for skip_n frames without visualization
            if(skip_n <= 0):
                dspace.showSpace(draw_dets=DRAW_DETS)
                dspace.clearSpace()
            else:
                skip_n = skip_n-1
            
            # increase time on each iteration (most important detail)
            t_global = t_global + 1
    except KeyboardInterrupt:
        # save tr, tr_fin and T_OFFSET
        print("fin")
        if(SAVE_TRAJECTORIES):
            with open('trs.p', 'wb') as fp: # fp: file pointer?
                abc = [T_OFFSET, t_global, tr_fin, tr]
                pickle.dump(abc, fp)
        
    # END EXTEND

# main:
def main_d():
    print("[INFO] This is main_d. To run main, set MAIN_DEB to False.")

# quick debug main:
def main_qd():
    print("[INFO] This is main_qd. To run main, set MAIN_DEB to False. To run main_deb, set MAIN_QD to False.")

if __name__ == "__main__":
    if(MAIN_DEB):
        main_d()
    else:
        main()