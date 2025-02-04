from helper_func import *
from DetectionSpace import DetectionSpace
from Trajectory import Trajectory
from OpticalFlow import OpticalFlow
from Detection import Detection, TDet, TDet_from_Detection
import cv2 as cv
import numpy as np
import copy # for deepcopy (visualization purposes)
import pickle
import argparse # for raft model argument parser
from pathlib import Path
np.set_printoptions(suppress=True, precision=3, linewidth=1000)

DATA_ROOT = "../data/"
FRAMES_PATH = "frames/LaSOT_bird-2/color/"

# tracker cache: tracker/flow -> flow, tracker -> result, mogoce tut trajektorije
COMPUTED_FLOW = "tracker/flow/"
RESULT_PATH = "tracker/"
# --------

USE_PRECOMPUTED = not True # use precomputed optical flow (set to False to calculate it on the go) (flows-path must be set)
ASK_BEFORE_FLOW_DELETE = not True # the '-y' kinda flag
FLOWS_PATH = "flow_est/LaSOT_bird-2_1/"
if(not USE_PRECOMPUTED):
    FLOWS_PATH = "flow_est/LaSOT_bird-2_1/computed/" # this is here to not overwrite existing data, in normal cases it would be same as ^ (FLOWS_PATH)
else:
    COMPUTED_FLOW = DATA_ROOT + FLOWS_PATH

DRAW_DETS=False
MAIN_DEB= not True
MAIN_QD = not True
# EXT_THR=10 # maximum number of extrapolation of trajectories (# of consecutive frames without detections for that trajectory (number of relative holes, essentially))
EXT_THR=5 
PO_THR=7 # possibly occluded threshold: number of holes before trajectory is marked as possibly occluded
UNSELECTED_STRIKE_MAX=5 # maximum number of times that trajectory is not selected but included in set
SAVE_TRAJECTORIES= not True # switch to save trajectories on every selection step for vizualization/debug purposes
T_INIT_S = 0.5


# some main-specific functions:
# function is used when new flow is computed (on every new frame, for stage III, getFlow is still used, but on files that this thing generated previously)
def computeOrGetFlowAtI(i:int, optical_flow_gen: OpticalFlow=None):
    if(not USE_PRECOMPUTED):
        # return optical_flow_gen.computeFlowAtI(i)
        return optical_flow_gen.computeFlowAtITest(i) # WARN: this is to test only, on real use cases, use ^
    return getFlowAtI(i, COMPUTED_FLOW)

def setFrameFlowAtI(dspace: DetectionSpace, i: int, optical_flow_gen: OpticalFlow = None):
    # set new frame
    frame = getFrameAtI(i,DATA_ROOT+FRAMES_PATH)
    dspace.lastFrame = frame.copy()
    dspace.map = frame
    
    # set new flow
    # flow_from = getFlowAtI(i, DATA_ROOT+FLOWS_PATH)
    flow_from = computeOrGetFlowAtI(i, optical_flow_gen=optical_flow_gen)
    dspace.flow_map = flow_from
    flow_img = flow2img(flow_from)
    dspace.last_flow_img = flow_img.copy()
    dspace.flow_img = flow_img
    
    #  also set flow to this image
    if(i >= 1):
        flow_to = getFlowAtI(i-1, COMPUTED_FLOW)
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
        ext_dets = p[3]
        if(t_merged is None):
            t_merged = t.getCopy()
            min_id = t.not_so_much_unique_id
        else:
            # must do: X, D2
            # should do: holes, holes_ref, term, color
            t.X[0].color = [10,10,150]
            t_merged.X = t_merged.X + ext_dets + t.X
            t_merged.D2 = {**t_merged.D2, **t.D2}
            t_merged.T2 = {**t_merged.T2, **t.T2} # also merge T2
            t_merged.F = {**t_merged.F, **t.F} # also merge F

            t_merged.holes = t_merged.holes + t.holes
            t_merged.term = t.term
            if(t.id < min_id):
                min_id = t.not_so_much_unique_id

        t_merged.T2[t] = [score, score2]
        merge_scores.append(t_merged.T2[t])
    t_merged.not_so_much_unique_id = min_id # identity is propagated
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
            flows_path = COMPUTED_FLOW
            tr_possible_next = tr.getPossibleNext3(tr_list, flows_path, time_window=time_window)
        all_possible_next = extendAllPossibleNext(tr, [(tr,T_INIT_S,1.0,[])],tr_possible_next,tr_list, type=type)
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

            # print("<t%d (c=%3d), %4.2f, f=%s>" % (t.id, t.color[2], c[1], str(t.term)), end=", ")
            if(i != 0):
                t_p = collected[i-1][0].X[-1].t
                t_c = collected[i][0].X[0].t
                skipped_time = skipped_time + (t_c - t_p)

        total_time = time_h-time_l
        # print("-> <t%d, %6.2f>, total time: %d, time skipped: %d (%4.2f) " % (t_merged.id, t_merged.getScoreII(), total_time, skipped_time, (float(skipped_time) / (float(total_time)+0.0000001)) ))
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
            flows_path = COMPUTED_FLOW
            tr_possible_next = tr.getPossibleNext3(all_trs, flows_path, time_window=time_window)
        collected.append( p ) # add it
        possible_next_trs = extendAllPossibleNext(tr, collected, tr_possible_next, all_trs, time_window=time_window)
        collected.pop() # remove it
        mtr_hypotheses = mtr_hypotheses + possible_next_trs
    
    # 3. return all collected trajectories
    return mtr_hypotheses

def printTrWithStats(t: Trajectory, i_t=None):
    if(i_t is None):
        print("t%-5s (t%-5s) o=[%-3d,%-3d], c=%-3d, len=%-4d, score=%9.4f %5d - %-5d, n_merged=%3d (fin:%s)" % \
          (t.not_so_much_unique_id,t.id, t.origin.x[0],t.origin.x[1], t.color[2], len(t.X), t.getScore2(), t.X[0].t, t.X[-1].t, len(t.T2.keys()), str(t.term)))
    else:
        print("%3d. t%-5s (t%-5s) o=[%-3d,%-3d], c=%-3d, len=%-4d, score=%9.4f %5d - %-5d, n_merged=%3d (fin:%s)" % \
              (i_t,t.not_so_much_unique_id,t.id, t.origin.x[0],t.origin.x[1], t.color[2], len(t.X), t.getScore2(), t.X[0].t, t.X[-1].t, len(t.T2.keys()), str(t.term)))



# main:
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
    DETS_FILE = "detections/LaSOT_bird-2.txt"
    GT_PATH = FRAMES_PATH+"../groundtruth.txt"
    RUN_ALL = not False
    # T_OFFSET = 500 + 653 # constant: starting time (t_global) (used also to align DetectionSpace.D indices (as it expects t0 at index 0))
    # T_OFFSET = 1160 # constant: starting time (t_global) (used also to align DetectionSpace.D indices (as it expects t0 at index 0))
    # T_OFFSET = 4095
    # T_OFFSET = 798
    # T_OFFSET = 1
    T_OFFSET = 1
    # T_OFFSET = 3380
    n = 0 # number of previous frames for trajectory init

    # testing squares:
    # FRAMES_PATH = "testing/sequence/"
    # FLOWS_PATH = "testing/flows/"
    # DETS_FILE = "testing/sequence/dets.txt"
    # T_OFFSET = 0 # constant: starting time (t_global) (used also to align DetectionSpace.D indices (as it expects t0 at index 0))
    # n = 0 # number of previous frames for trajectory init

    # ----------------------------------------------------------------------

    D13 = readDetFile2(DATA_ROOT+DETS_FILE) # read detections from file
    D13 = [[]]+D13 # offset it (because time starts at 1, there is nothing on 0)
    t_global = T_OFFSET # current time (frame)
    # print(t_global, D13[T_OFFSET: T_OFFSET+10])
    
    # actual offset for detections list is: T_OFFSET - n (because there are some detections before global offset (initial trajectories))
    n_res = len(D13) - n - T_OFFSET
    tr: list[Trajectory] = []
    tr_fin: list[Trajectory] = []
    idTr_map: dict[int, Trajectory] = {} # id -> tr mapping: when merging, merged trajectory gets least id - this is used in the end (it contains all id-s that had been tracked)
    n_stage_II_III = 120 # 5 # 20 # 'time window' - # of frames between trajectory selections
    stage_II_III_timew = 20 # time window for merging of trajectories

    # create tracker cache dir:
    Path("tracker/flow/").mkdir(parents=True, exist_ok=True) 
    # ------------------------------
    
    
    # INIT DSPACE
    frame = getFrameAtI(t_global,DATA_ROOT+FRAMES_PATH)
    h,w,_ = np.shape(frame)
    dspace = DetectionSpace(h,w, time_offset=(T_OFFSET-n), show_flow=False, disable_vis=RUN_ALL) # also pass time offset for using correct indices | no visualization on run all
    
    # set last frame
    dspace.lastFrame = frame.copy()
    dspace.map = frame

        # INIT RAFT MODEL FOR OPTICAL FLOW ESTIMATION, IF NEEDED (if USE_PRECOMPUTED is set to False)
    of: OpticalFlow = None
    if(not USE_PRECOMPUTED):
        # prepare model arguments:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint")
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
        args = parser.parse_args()
        # of:OpticalFlow = OpticalFlow(args,frames_path=FRAMES_PATH, save_path=FLOW_SAVE_PATH)
        of = OpticalFlow(args,frames_path=(DATA_ROOT+FRAMES_PATH), save_path=(COMPUTED_FLOW))
        # END INIT RAFT

    # set flow map
    # flow = getFlowAtI(t_global, DATA_ROOT+FLOWS_PATH)
    flow = computeOrGetFlowAtI(t_global, of)
    flow_img = flow2img(flow)
    dspace.flow_map = flow
    dspace.last_flow_img = flow_img.copy()
    dspace.flow_img = flow_img
    
    print("init dspace: ")
    D13_init = D13[t_global] # init with those from current time step
    dspace.D.append(D13_init)
    print(dspace.D)
    t_global = t_global+1
    n_res = n_res-1 # there is one frame less - [FIX]
    # END INIT DSPACE

    
    
    # INIT TRAJECTORIES
    for d in D13_init:
        t = Trajectory(d,dspace,getTrColor())
        t.build2()
        tr.append(t)
        idTr_map[t.not_so_much_unique_id] = t # add it to map
    # END INIT TRAJECTORIES
    
    # quick visualization
    if(not RUN_ALL):
        for t in tr:
            t.drawToSpace()
        dspace.showSpace(draw_dets=DRAW_DETS)
    

    # EXTEND
    skip_n = 0
    # skip_n = 119
    skip_n = 200
    # skip_n = 285
    # skip_n = 298
    # skip_n = 316
    # skip_n = 555
    # skip_n = 650
    # skip_n = 4110 # all

    try:
        sslm = 0 # steps since last merge
        print("[EXTEND]")
        # main extend loop
        for i in range(n_res):
            print("[@"+str(t_global)+"]")
            
            # 1. init new frame (also compute flow, if needed):
            next_dets=D13[t_global]
            # print("next dets: ",next_dets)
            dspace.D.append(next_dets)
            setFrameFlowAtI(dspace,t_global, of)
            # ---

            # 2. extend trajectories
            # print("n_tr: ",len(tr))
            used_dets: set[Detection] = set()
            tr_next: list[Trajectory] = []
            if(not RUN_ALL):
                print("[EXTENDING TRAJECTORIES]")
            for t in tr:
                tr_next_from_same: list[Trajectory] = []
                if((not t.term) and (not t.exited) and (t.holes_ref <= EXT_THR)): # extend only, if not terminated or exited
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
                
                # 2.5. mark next from same (current + all forks) as exited, if enter exit zone
                for tr_nxt in tr_next_from_same:
                    if(dspace.isInExitZone(tr_nxt.X[-1])):
                        tr_nxt.exited = True
                        tr_nxt.term = True
                
                # 2.6. update next
                tr_next = tr_next + tr_next_from_same
                
                if(not RUN_ALL):
                    # some debug prints
                    first = True
                    for tt in tr_next_from_same:
                        if(first):
                            first = False
                            # print(">>> t"+str(tt.id)+""+" - origin:",tt.origin,"score:",tt.getScore2(), "color:",tt.color, tt.X[-2:], "len:",len(tt.X))
                            # print(">>> t%d (%d) - origin:%s score:%6.2f color:%s len:%d"%(tt.id,tt.not_so_much_unique_id,str(tt.origin),tt.getScore2(), str(tt.color),str(tt.X[-2:]),len(tt.X)))
                            print(">>> t%d (%d) - origin:%s score:%6.2f color:%s %s len:%d"%(tt.id,tt.not_so_much_unique_id,str(tt.origin),tt.getScore2(), str(tt.color),str(tt.X[-2:]),len(tt.X)))
                        else:
                            print("|-> t%d (%d) - origin:%s score:%6.2f color:%s %s len:%d"%(tt.id,tt.not_so_much_unique_id,str(tt.origin),tt.getScore2(), str(tt.color),str(tt.X[-2:]),len(tt.X)))
                # ---
            if(not RUN_ALL):
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
            tr.sort(key=lambda x: x.getScore2(), reverse=True) # sort to minimize chance of getting stuck in some local minimum (there ARE issues with qbp-solver)
            
            # 4.1 build Q
            # print("building Q")
            Q = dspace.buildQBPMatrix3(tr)

            # 4.2 solve QBP
            v = dspace.solveQBP2(Q)
            
            if(not RUN_ALL):
                print(Q)
                print(v)
            # ---

            # 5. keep only selected trajectories for next frame:
            tr_next2: list[Trajectory] = []
            # print("selected: ", end="")
            for ii in range(len(v[0])):
                if(v[0][ii] == 1):
                    tr_i = tr[ii]
                    tr_next2.append(tr_i)
                    # print("t"+str(tr_i.id),end=" ", flush=True)
            tr = tr_next2
            # print()
            # ---

            # 6. visualization
            if(not RUN_ALL):
                print("[VISUALIZATION]")
                t_i = 0
                print("fin (%d):"%(len(tr_fin)))
                for t in tr_fin:
                    printTrWithStats(t, i_t=t_i)
                    # print("%3d. t%-5s (t%-5s) o=[%-3d,%-3d], c=%-3d, len=%-4d, score=%9.4f %5d - %-5d, f:%s" % (t_i,t.not_so_much_unique_id,t.id, t.origin.x[0],t.origin.x[1], t.color[2], len(t.X), t.getScore2(), t.X[0].t, t.X[-1].t, str(t.term)))
                    t.drawToSpace(color=[100,100,100])
                    t_i = t_i+1
                print("not fin (%d):"%(len(tr)))
                for t in tr:
                    printTrWithStats(t, i_t=t_i)
                    # print("%3d. t%-5s (t%-5s) c=%-3d, len=%-4d, score=%9.4f %5d - %-5d, f:%s" % (t_i,t.not_so_much_unique_id,t.id, t.color[2], len(t.X), t.getScore2(), t.X[0].t, t.X[-1].t, t.term))
                    t.drawToSpace()
                    t_i = t_i+1
            # ---
            
            # 7. on every n_stage_II_III-th frame, merge trajectories collected so far:
            order=[2,1]
            sslm = sslm+1
            # print("[!]",sslm)
            if(n_stage_II_III == sslm or len(tr_fin) >= 25):
                if(not RUN_ALL):
                    print("[MERGE TRAJECTORIES]")
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
                # print(Q)
                np.savetxt("Q.txt",Q, fmt="%7.3f")
                res = dspace.solveQBP2(Q)
                # res = dspace.solveQBP(Q)
                print(res)
                v = res[0]
                
                # SHOW SELECTED:
                selected_merged:list[Trajectory] = []
                # dspace.clearSpace()
                for i in range(len(v)):
                    if(v[i] == 1):
                        merged_at_i = merged_merged[i]
                        if(not (merged_at_i.term and len(merged_at_i.T2.keys()) == 1)):
                            selected_merged.append(merged_at_i)
                            idTr_map[merged_at_i.not_so_much_unique_id] = merged_at_i # update trajectories holding/representing that id
                            # merged_merged[i].drawToSpace()
                            # print("t%d (t%d)"%(merged_merged[i].id, merged_merged[i].not_so_much_unique_id), end=", ")
                # print()

                if(not RUN_ALL):
                    i_t = 0
                    for t in selected_merged:
                        printTrWithStats(t,i_t=i_t)
                        i_t = i_t+1

                # 7.5. use new trajectories in next round
                # tr = tr+selected_merged # ?? why add? how 'bout id switch?
                tr = selected_merged
                tr_fin = []


                # print("------------------- --------------------------------------------------------------------------------")
                if(skip_n <= 0 and not RUN_ALL):
                    d2_winname = "merged"
                    cv.namedWindow(d2_winname, cv.WINDOW_NORMAL)
                    for t in selected_merged:
                        t.drawToSpace()
                        print("t"+str(t.id), end=" ", flush=True)
                        dspace.showSpace(draw_dets=DRAW_DETS, draw_last_dets_bb=False, dspace_winname=d2_winname)
                        dspace.clearSpace()
                    for t in selected_merged:
                        t.drawToSpace()
                    print()
                    dspace.showSpace(draw_dets=DRAW_DETS, draw_last_dets_bb=False, dspace_winname=d2_winname)
                    # dspace.clearSpace()
                sslm = 0
            # ---

            # some loop-related technical stuff
            # if skip_n is set (non zero), algorithm runs for skip_n frames without visualization
            if(not RUN_ALL):
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
        # if(SAVE_TRAJECTORIES):
        #     with open('trs2.p', 'wb') as fp: # fp: file pointer?
        #         abc = [T_OFFSET, t_global, tr_fin, tr]
        #         pickle.dump(abc, fp)
    print("[END EXTEND]")
    # END EXTEND

    print("[FINAL MERGE]")
    # print("merge fin: ")
    merged_merged:list[Trajectory] = []
    merge_fin = list(idTr_map.values())
    if(not RUN_ALL):
        for i in range(len(merge_fin)):
            printTrWithStats(merge_fin[i],i)
    
    # 7.1. stage II (bridged) (is faster)
    merged_trs1 = getMergedHypotheses(merge_fin, type=order[0], time_window=120) 
    
    # 7.2. get unused trajectories
    tr_left = set(merge_fin)
    i = 0
    for t in merged_trs1:
        key_set = set(t.T2.keys())
        if(len(key_set) > 1):
            tr_left = tr_left - key_set
        i = i+1
    tr_left = list(tr_left)

    # 7.3. stage III (flow) with unused trajectories (is slower)
    merged_trs2 = getMergedHypotheses(tr_left, type=order[1], time_window=120)
    # [INFO] flow is no longer needed from here on, so delete possibly precomputed to save space
    if(not USE_PRECOMPUTED):
        deletePrecomputedFlow(COMPUTED_FLOW, confirm=ASK_BEFORE_FLOW_DELETE)

    # 7.4. select best trajectories to be used in new round
    merged_merged = merged_trs1+merged_trs2
    # merged_merged = merged_trs2
    Q = dspace.buildQBPMatrixX(merged_merged, type=2)


    
    # SOLVE QBP:
    # print(Q)
    np.savetxt("Q.txt",Q, fmt="%7.3f")
    res = dspace.solveQBP2(Q)
    # res = dspace.solveQBP(Q)
    # print(res)
    v = res[0]
    


    # SHOW RESULTS:
    if(not RUN_ALL):
        dspace.clearSpace()
        i_t = 0
        
        # for tr in idTr_map:
        #     trr = idTr_map[tr]
        i = 0
        for trr in merged_merged:
            if(v[i] == 1):
                printTrWithStats(trr, i_t=i_t)
                trr.drawToSpace()
                dspace.showSpace(draw_dets=DRAW_DETS, draw_last_dets_bb=False, dspace_winname="merged")
                dspace.clearSpace()
                i_t = i_t+1
            i = i+1
        
        # for tr in idTr_map:
            # trr = idTr_map[tr]
        i = 0
        for trr in merged_merged:
            if(v[i] == 1):
                trr.drawToSpace()
            i = i+1
        
        dspace.showSpace(draw_dets=DRAW_DETS, draw_last_dets_bb=False, dspace_winname="merged")
        dspace.clearSpace()
    
    # select object to track (from ground truth)
    # get first frame
    target_select = []
    with open(DATA_ROOT+GT_PATH) as fd:
        first_l = fd.readline()
        det = first_l[0:-1].split(",")
        target_select = [float(i) for i in det]
    # print(target_select)

    max_iou = 0
    max_score = 0
    selected_t = merged_merged[0]
    for trr in merged_merged:
        first_d = trr.X[0]
        if(first_d.t <= 10): # search only among few first frames
            det_with_gt_bb = bbDet2Det(target_select,0)
            iou = IoU(first_d, det_with_gt_bb)
            # print("t%5d (t%5d): %5.2f"%(trr.not_so_much_unique_id, trr.id, iou))
            trr_score = trr.getScore2()
            if(iou >= max_iou and trr_score > max_score): 
                max_iou = iou
                selected_t = trr
                max_score = trr_score
    
    if(not RUN_ALL):
        print("selected: ")
        printTrWithStats(selected_t)
        selected_t.drawToSpace()
        
        dspace.showSpace(draw_dets=DRAW_DETS, draw_last_dets_bb=False, dspace_winname="merged")
        dspace.clearSpace()
    
    # for d in selected_t.X:
    #     print(d)
    print("[END] saving results")

    lines_to_file = selected_t.tr2bbStr(n_res).split('\n')
    lines_to_file[0] = "1"
    str_to_file = '\n'.join(lines_to_file)
    with open(RESULT_PATH+"results.txt", 'w') as fp: # fp: file pointer?
        fp.write(str_to_file)
    
    if(SAVE_TRAJECTORIES):
        with open(RESULT_PATH+'trs.p', 'wb') as fp: # fp: file pointer?
            abc = [T_OFFSET, t_global, copy.deepcopy(idTr_map), copy.deepcopy(merged_merged)]
            pickle.dump(abc, fp)
    print("[END] results saved.")
    # print(to_file)


# debug main:
def main_d():
    print("[INFO] This is main_d. To run main, set MAIN_DEB to False.")

    FRAMES_PATH = "frames/LaSOT_bird-2/color/"
    FLOWS_PATH = "flow_est/LaSOT_bird-2_1/"
    # DETS_FILE = "testing_data/LaSOT_bird-2_001.txt"
    DETS_FILE = "../src/results.txt"
    GT_PATH = "frames/LaSOT_bird-2/groundtruth.txt"
    T_OFFSET = 1
    n = 0

    # ----------------------------------------------------------------------

    D13 = [[]]+readDetFile2(DATA_ROOT+DETS_FILE, T_OFFSET) # read detections from file
    D14 = [[]]+readDetFile2(DATA_ROOT+GT_PATH, T_OFFSET) # read ground truth
    # print("track: ",D13[0:10])
    # print("gt: ",D14[0:10])
    t_global = T_OFFSET # current time (frame)
    
    # actual offset for detections list is: T_OFFSET - n (because there are some detections before global offset (initial trajectories))
    n_res = len(D13) - n - T_OFFSET
    tr: list[Trajectory] = []
    tr_fin: list[Trajectory] = []
    idTr_map: dict[int, Trajectory] = {} # id -> tr mapping: when merging, merged trajectory gets least id - this is used in the end (it contains all id-s that had been tracked)
    n_stage_II_III = 120 # 5 # 20 # 'time window' - # of frames between trajectory selections
    stage_II_III_timew = 20 # time window for merging of trajectories
    # ------------------------------

    target_select = []
    with open(DATA_ROOT+GT_PATH) as fd:
        first_l = fd.readline()
        det = first_l[0:-1].split(",")
        target_select = [float(i) for i in det]
    print("selected det.:", target_select)
    
    
    # INIT DSPACE
    frame = getFrameAtI(t_global,DATA_ROOT+FRAMES_PATH)
    h,w,_ = np.shape(frame)
    dspace = DetectionSpace(h,w, time_offset=(T_OFFSET-n), show_flow=False) # also pass time offset for using correct indices
    
    # set last frame
    dspace.lastFrame = frame.copy()
    dspace.map = frame

    # set D:
    dspace.D.append(D14[t_global])

    # init trajectories:
    t_track = Trajectory(D14[t_global][0], dspace, [0,100,100])
    t_gt = Trajectory(D14[t_global][0], dspace, [0,255,0])
    t_global = t_global+1

    # extend: just add from file:
    for i in range(n_res):
        # set new frame
        frame = getFrameAtI(t_global,DATA_ROOT+FRAMES_PATH)
        dspace.lastFrame = frame.copy()
        dspace.map = frame
                
        next_dets=D13[t_global] + D14[t_global]
        print("next dets: ",next_dets)
        dspace.D.append(next_dets)
        
        next_det = D13[t_global][0]
        next_gt = D14[t_global][0]
            
        t_track_tdet: TDet = TDet_from_Detection(next_det)
        t_gt_tdet: TDet = TDet_from_Detection(next_gt)
        t_track.X.append(t_track_tdet)
        t_gt.X.append(t_gt_tdet)
        t_track.drawToSpace()
        t_gt.drawToSpace()
        dspace.showSpace(draw_dets=DRAW_DETS)
        dspace.clearSpace()

        t_global = t_global+1





# quick debug main:
def main_qd():
    print("[INFO] This is main_qd. To run main, set MAIN_DEB to False. To run main_deb, set MAIN_QD to False.")

    FRAMES_PATH = "frames/LaSOT_bird-2/color/"
    FLOWS_PATH = "flow_est/LaSOT_bird-2_1/"
    DETS_FILE = "testing_data/LaSOT_bird-2_001.txt"
    GT_PATH = "frames/LaSOT_bird-2/groundtruth.txt"
    T_OFFSET = 1
    n = 0

    # ----------------------------------------------------------------------
    D13 = [[]]+readDetFile2(DATA_ROOT+DETS_FILE, T_OFFSET) # read detections from file
    # print("track: ",D13[0:10])
    # print("gt: ",D14[0:10])
    t_global = T_OFFSET # current time (frame)
    
    # actual offset for detections list is: T_OFFSET - n (because there are some detections before global offset (initial trajectories))
    n_res = len(D13) - n - T_OFFSET
    tr: list[Trajectory] = []
    tr_fin: list[Trajectory] = []
    idTr_map: dict[int, Trajectory] = {} # id -> tr mapping: when merging, merged trajectory gets least id - this is used in the end (it contains all id-s that had been tracked)
    n_stage_II_III = 120 # 5 # 20 # 'time window' - # of frames between trajectory selections
    stage_II_III_timew = 20 # time window for merging of trajectories
    # ------------------------------

    # target_select = []
    # with open(DATA_ROOT+GT_PATH) as fd:
    #     first_l = fd.readline()
    #     det = first_l[0:-1].split(",")
    #     target_select = [float(i) for i in det]
    # print("selected det.:", target_select)
    

    
    # LOAD TRS:
    # ------------
    Trajectory.Tid = 1000 #
    abc = []
    with open('trs2.p', 'rb') as fp:
        abc = pickle.load(fp)
    T_OFFSET, t_global, idTr_map, tr_fin, tr = abc
    print("this is idtrmap: ")
    print(idTr_map)

    tr_all = tr_fin + tr
    # -----------------------------------

    # INIT DSPACE
    frame = getFrameAtI(t_global,DATA_ROOT+FRAMES_PATH)
    h,w,_ = np.shape(frame)
    dspace = DetectionSpace(h,w, time_offset=(T_OFFSET-n), show_flow=False) # also pass time offset for using correct indices
    
    # set last frame
    dspace.lastFrame = frame.copy()
    dspace.map = frame

    for k in idTr_map:
        tr_k = idTr_map[k]
        tr_k.detectionSpace = dspace
        printTrWithStats(tr_k, k)
        # tr_k.drawToSpace()
    print(idTr_map[0].T2)
    idTr_map[0].drawToSpace()
    dspace.showSpace(draw_dets=DRAW_DETS,draw_last_dets_bb=False)
        



    # for t in tr_all:
    #     t.detectionSpace = dspace
    #     t.drawToSpace()
    # dspace.showSpace(draw_dets=DRAW_DETS,draw_last_dets_bb=False)





if __name__ == "__main__":
    if(MAIN_DEB):
        main_d()
    elif(MAIN_QD):
        main_qd()
    else:
        main()

 