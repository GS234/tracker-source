# from helper_func import coords2det2, readDetFile2, coords2map, readTrajectoryFile, getTrColor, getFrameAtI, colorHist2Det, getColorHist, showHists, updateDetColorHistFromFrame, readFlowFile, flow2img, getFlowAtI, getDetMotionVector, flowVec2Det, updateDetMotionVecFromFlowMap
from helper_func import *
from DetectionSpace import DetectionSpace
from Trajectory import Trajectory
from Detection import Detection
import cv2 as cv
import numpy as np
import copy # for deepcopy (visualization purposes)
import pickle
# np.set_printoptions(threshold=np.inf)
# np.set_printoptions(threshold=10)
np.set_printoptions(suppress=True, precision=3, linewidth=1000)

DATA_ROOT = "../data/"
FRAMES_PATH = "frames/LaSOT_bird-2/color/"
FLOWS_PATH = "flow_est/LaSOT_bird-2_1/"

DRAW_DETS=False
MAIN_DEB= not  True
MAIN_QD =  True
# MAIN_DEB=False
# EXT_THR=10 # maximum number of extrapolation of trajectories (# of consecutive frames without detections for that trajectory (number of relative holes, essentially))
EXT_THR=5 
PO_THR=7 # possibly occluded threshold: number of holes before trajectory is marked as possibly occluded
UNSELECTED_STRIKE_MAX=5 # maximum number of times that trajectory is not selected but included in set
SAVE_TRAJECTORIES=not True # switch to save trajectories on every selection step for vizualization/debug purposes
T_INIT_S = 0.5

E1,E2 =  3.3,0.1
# E1,E2 =  2.3,0.1
# E1,E2 =  1.2,0.1

# some main-specific functions:

# combination of building matrix and selection of trajectories
def selectBest(tr: list, dspace: DetectionSpace, debug = False, recalculate_scores=True) -> list:
    print("building Q")
    # Q = dspace.buildQBPMatrix(tr, E1, E2)
    Q = dspace.buildQBPMatrix2(tr, E1, E2, recalculate_scores=recalculate_scores)
    if(debug):
        print(Q)
    print("solving")
    v = dspace.solveQBP2(Q) # this is slow
    print("solved")
    if(debug):
        print(v)
    return v

# function returns list of only those trajectories, that are selected
def getSelectedNvisualize(tr: list[Trajectory], v: list[int], dspace:DetectionSpace, debug=False, separately=False) -> list:
    # draw them
    tr_temp = [] # temporary tr (to store only those, that are selected)
    # print(len(tr), len(v))
    for i in range(len(tr)):
        # print("trajectory: ", tr[i].id," - ", tr[i].holes_ref)
        if(debug):
            print(i, end= ", ", flush=True)
        # if((v[0][i] != 0) and (tr[i].holes_ref <= 15)):
        tr_color = tr[i].color

        if((v[i] != 0) and (tr[i].holes_ref <= 15)):
            if(separately):
                print()
                tr[i].color = [255,255,0]
                print("showing: ", tr[i].id)
            tr[i].drawToSpace()
            if(separately):
                dspace.showSpace(draw_dets=DRAW_DETS)
                tr[i].color = tr_color
                tr[i].drawToSpace()
            tr[i].not_selected_strike = 0 # reset not selected strike if is selected
            tr_temp.append(tr[i])
        else:
            tr[i].not_selected_strike = tr[i].not_selected_strike + 1 # increase not selected strike
            
    if(not separately):
        dspace.showSpace(draw_dets=DRAW_DETS)
    if(debug):
        print()
    return tr_temp

# # function reads i-th frame
# def getFrameAtI(i: int, path=FRAMES_PATH):
#     frame_i = f'{i:08}'
#     frame = cv.imread(DATA_ROOT+path+str(frame_i)+".jpg")
#     return frame

def dropRedundant(tr: list[Trajectory], debug=False, recalculate=True) -> list[Trajectory]:
    tr_new: list[Trajectory] = []
    i = 0 # current trajectory
    while(i < len(tr)):
        j = 0 # comparing to trajectories at j
        add = True
        while(j < i):
            # if is based on same detections, then do not add
            # if(tr[i].basedOnSameDetections(tr[j])):
            if(tr[i].equalsDetScore(tr[j],recalculate=recalculate)):
                if(debug):
                    print(tr[i], "("+str(i)+")", " is based on same as: ", tr[j], "("+str(j)+"). NOT ADDING: ", tr[i])
                add = False
                break
            j = j+1
        if(add):
            tr_new.append(tr[i])
        i = i+1
    return tr_new

# similar as dropRedunant, but checks only trajectory tr1 against trajectories in list (returns bool)
def checkIfRedundant(tr_list: list[Trajectory], tr:Trajectory, recalculate=True, debug=False) -> bool:
    for t in tr_list:
        if(tr.equalsDetScore(t, recalculate=recalculate)):
            if(debug):
                    print(tr, " is based on same as: ", t, "). NOT ADDING: ", tr)
            return True
    return False

# breakpoint function: stop and wait for keyboard interrupt
def stopNwaitForKI():
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("continue")

# function reads file with saved trajectories and displays them on frames
def visualizeSaved():
    # INIT:
    # get through first n frames and initiate (hopefully) strong trajectories

    # init variables used in process
    D13 = readDetFile2(DATA_ROOT+"detections/LaSOT_bird-2.txt") # read detections from file
    # TR = readTrajectoryFile('debug_data/hypotheses.p')
    TR = readTrajectoryFile('hypotheses.p') # tr is [(t_global, [trajectories])]

    T_OFFSET = 20 # constant: starting time (t_global) (used also to align DetectionSpace.D indices (as it expects t0 at index 0))
    t_global = T_OFFSET # current time (frame)
    n = 19 # number of previous frames for trajectory init
    # actual offset for detections list is: T_OFFSET - n (because there are some detections before global offset (initial trajectories))
    n_res = len(D13) - n # [TODO: handle properly (new t_global mechanics)]
    tr: list[Trajectory] = []
    # ------------------------------
    
    # init dspace:
    frame = getFrameAtI(t_global, DATA_ROOT+FRAMES_PATH)
    h,w,_ = np.shape(frame)
    dspace = DetectionSpace(h,w)
    dspace.lastFrame = frame
    
    # append detections
    for d in D13[T_OFFSET-n:T_OFFSET]:
        dspace.D.append(d)
    
    tr_i = 0
    t_global, tr = TR[tr_i] # initial trajectories
    tr_i = tr_i + 1
    
    # set trajectories dspace pointers
    for _,tl in TR:
        for t in tl:
            t.detectionSpace = dspace # pointers from objects read from picle are no longer valid
    

    # choose best trajectories and draw them:
    v = selectBest(tr, dspace, debug=True, recalculate_scores=False) # already have scores
    # getSelectedNvisualize(tr, v[0], dspace, True, True)
    # # END INIT
    

    # ONLY SELECTED
    # selected_tr = [0:len(tr)]
    selected_tr = list(range(len(TR))) # select all
    # selected_tr = [16] # select all
    for tr_i in selected_tr:
        t_global, tr = TR[tr_i]

        if(tr_i < len(TR)):
            # 0. set new frame
            frame = getFrameAtI(t_global, DATA_ROOT+FRAMES_PATH)
            dspace.lastFrame = frame.copy()
            dspace.map = frame

            # 1. get detections in current frame
            latest_dets = D13[t_global]
            dspace.D.append(latest_dets)

            # 2. read trajectories
            print("-----------------> t: ", t_global, ", tr_i: ", tr_i)
            v = selectBest(tr, dspace, debug=True, recalculate_scores=True) # already have scores
            
            dspace.clearSpace()
            for ii in range(len(tr)):
                if(v[0][ii] == 1):
                    print("-> ",end="",flush=True)
                else:
                    print("   ", end="", flush=True)
                print(tr[ii]," holes: ", tr[ii].holes_ref, " nss: ", tr[ii].not_selected_strike)
            # for ii in range(len(tr)):
            #     if(v[0][ii] == 0):
            #         tr[ii].color = getTrColor()
            #         tr[ii].drawToSpace()
            # for ii in range(len(tr)):
            #     if(v[0][ii] == 1):
            #         tr[ii].color = [255,255,0]
            #         tr[ii].drawToSpace()
            #         print("-> ",end="",flush=True)
            #     print(tr[ii]," holes: ", tr[ii].holes_ref, " nss: ", tr[ii].not_selected_strike)
            #     input("continue")
            # dspace.showSpace(draw_dets=DRAW_DETS)
            
            
            # debug section
            print("--------- D --------------")
            if(tr_i == -1):
                tr5 = tr[1]
                tr203 = tr[7]

                print(tr5, tr203)
                Q = dspace.buildQBPMatrix2([tr5,tr203], E1, E2, recalculate_scores=True)
                print(Q)
                print(dspace.solveQBP(Q))
                print(dspace.solveQBP2(Q))

                # tr83d = tr83.D
                # tr84d = tr84.D
                # print(tr83,tr84)
                # print(tr83.basedOnSameDetections(tr84))
                # print(tr83d)
                # print("\n")
                # print(tr84d)
                # print("\n")
                # print(tr83d.difference(tr83d & tr84d))


            print("--------- ! --------------")
            getSelectedNvisualize(tr, v[0], dspace, True, separately=True)
            # input("continue?")


    # LOOP ALL:
    # for i in range(n_res):
    #     if(tr_i < len(TR)):
    #         # 0. set new frame
    #         frame = getFrameAtI(t_global)
    #         dspace.lastFrame = frame.copy()
    #         dspace.map = frame

    #         # 1. get detections in current frame
    #         latest_dets = D13[t_global]
    #         dspace.D.append(latest_dets)

    #         # 2. read trajectories
    #         t_tr, tr = TR[tr_i]
    #         if(t_tr == t_global):
    #             print("-----------------> t: ", t_global, ", tr_i: ",tr_i)
    #             v = selectBest(tr, dspace, debug=True, recalculate_scores=False) # already have scores
    #             dspace.clearSpace()
    #             for ii in range(len(tr)):
    #                 if(v[0][ii] == 0):
    #                     tr[ii].color = getTrColor()
    #                     tr[ii].drawToSpace()
    #             for ii in range(len(tr)):
    #                 if(v[0][ii] == 1):
    #                     tr[ii].color = [255,255,0]
    #                     tr[ii].drawToSpace()
    #                     print("-> ",end="",flush=True)
    #                 print(tr[ii]," holes: ", tr[ii].holes_ref, " nss: ", tr[ii].not_selected_strike)
    #             dspace.showSpace(draw_dets=DRAW_DETS)
    #             # getSelectedNvisualize(tr, v[0], dspace, True, False)
    #             tr_i = tr_i + 1

    #     t_global = t_global + 1


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
def mergeTrajectories(tr_list:list[Trajectory], time_window=20, max_space_diff=100.0) -> Trajectory:
    t_merged = None # first
    # sort 'em by time:
    tr_list.sort(key=lambda x: x.X[0].t)
    
    min_id = 0
    merge_scores = []
    for t in tr_list:
        if(t_merged is None):
            t_merged = t.getCopy()
            min_id = t.id
            t_merged.T2[t] = 1.0 # first one gets one

        else:
            # must do: X, D2
            # should do: holes, holes_ref, term, color
            
            connection_prob = t_merged.getConnectionProb(t, time_window, max_space_diff) # this should be before merging of X
            t.X[0].color = [10,10,150]
            t_merged.X = t_merged.X + t.X
            t_merged.D2 = {**t_merged.D2, **t.D2}
            t_merged.T2 = {**t_merged.T2, **t.T2} # also merge T2

            t_merged.holes = t_merged.holes + t.holes
            t_merged.term = t.term
            if(t.id < min_id):
                min_id = t.id
            
            # get connection score:
            t_merged.T2[t] = connection_prob
        merge_scores.append(t_merged.T2[t])
    # print("scores: ",merge_scores)
    # t_merged.id = min_id #keep id-s unique
    return t_merged

# function works same as ^, but uses list[tuple[Trajectory, float]]
def mergeTrajectories2(tr_list:list[tuple[Trajectory, float, float]]) -> Trajectory:
    t_merged = None # first
    # sort 'em by time:
    tr_list.sort(key=lambda x: x[0].X[0].t)
    
    min_id = 0
    merge_scores = []
    for p in tr_list:
        t = p[0]
        score = p[1]
        print(p)
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
    # print("scores: ",merge_scores)
    # t_merged.id = min_id #keep id-s unique
    return t_merged

# function generates and returns all possible connections with other trajectories (similar to extend, but it works with whole trajectories now)
def getMergedHypotheses(tr_list: list[Trajectory], time_window=20, max_space_diff=100.0, type=1):
    print("this is getMergedHypotheses: ")
    mtr_hypotheses = []
    for tr in tr_list:
        # print("first one:")
        # first_tr = tr
        # print(first_tr, end=", ")
        # print(first_tr)
        # tr_set: set[Trajectory] = set(tr_list) - set([first_tr])
        # print("this %s could connect to:" % (tr))
        tr_possible_next = []
        if(type==1):
            tr_possible_next = tr.getPossibleNext2(tr_list, time_window=time_window)
        if(type == 2):
            tr_possible_next = tr.getPossibleNext3(tr_list, DATA_ROOT+FLOWS_PATH, time_window=time_window)
        # print("possible next:",tr_possible_next)
        all_possible_next = extendAllPossibleNext(tr, [(tr,T_INIT_S,1.0)],tr_possible_next,tr_list, type=type)
        mtr_hypotheses = mtr_hypotheses + all_possible_next
        
        
    #     print("   -----   ")
    # print(mtr_hypotheses)
    # print(len(mtr_hypotheses))
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
        print(collected)
        t_merged = mergeTrajectories2(collected)
        skipped_time = 0
        time_l = collected[0][0].X[0].t
        time_h = time_l
        for i in range(len(collected)):
            c = collected[i]
            time_h = c[0].X[-1].t
            print("<t%d (c=%3d), %4.2f>" % (c[0].id, c[0].color[2], c[1]), end=", ")
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
            # [TODO] also calculate backward flow (from begining to end, compare results)
        collected.append( p ) # add it
        possible_next_trs = extendAllPossibleNext(tr, collected, tr_possible_next, all_trs, time_window=time_window)
        collected.pop() # remove it
        mtr_hypotheses = mtr_hypotheses + possible_next_trs
    
    # 3. return all collected trajectories
    return mtr_hypotheses

# main:
def main():
    stage = 2
    override =  True
    Trajectory.Tid = 1000 #
    # get saved data:
    abc = []
    with open('trs.p', 'rb') as fp:
        abc = pickle.load(fp)
    # print(abc)
    t_off, t_global, tr_fin, tr = abc
    # ------------


    # INIT DSPACE
    frame = getFrameAtI(t_global,DATA_ROOT+FRAMES_PATH)
    h,w,_ = np.shape(frame)
    dspace = DetectionSpace(h,w, time_offset=(t_global)) # also pass time offset for using correct indices
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
    # END INIT DSPACE


    # draw trajectories:
    i = 0
    print("all trajectories:")
    print("fin:")
    for t in tr_fin:
        t: Trajectory = t
        # print("t"+str(t.id)+" c="+str(t.color[2])+", len="+str(len(t.X)), "from:", t.X[0].t, "to:", t.X[-1].t)
        print("%3d. t%-5s o=[%-3d,%-3d], c=%-3d, len=%-4d, score=%9.4f %5d - %-5d" % (i,t.id, t.origin.x[0],t.origin.x[1], t.color[2], len(t.X), t.getScore2(), t.X[0].t, t.X[-1].t))
        t.detectionSpace = dspace
        t.T2 = {}
        t.drawToSpace()
        i = i+1
    print("not fin:")
    for t in tr:
        t: Trajectory = t
        print("%3d. t%-5s c=%-3d, len=%-4d, score=%9.4f %5d - %-5d" % (i,t.id, t.color[2], len(t.X), t.getScore2(), t.X[0].t, t.X[-1].t))
        t.detectionSpace = dspace
        t.T2 = {}
        t.drawToSpace()
        i = i+1
    print("--------------------------")
    dspace.showSpace()

    tr_all = tr_fin + tr

    # tr_all = tr_all[15:22]
    # print(tr_all)
    # h = 1
    # tr_all = tr_all[h:h+1]

    # for t in tr_all:
    #     t.drawToSpace()
    #     # draw first bb and extrapolated one:
    #     # drawBoundingBox(dspace.map, t.X[0].bb, [255,255,0])
    #     bb5 = t.getNextBbII(5,5)

    #     # bb10 = t.getNextBbII(5,10)
    #     # drawBoundingBox(dspace.map, bb10, [100,255,200])
        
    #     possibleNext: list[Trajectory] = t.getPossibleNext2(tr_all)
    #     for tt in possibleNext:
    #         tt.drawToSpace()
    #         drawBoundingBox(dspace.map, tt.X[0].bb, [255,255,0])
        
    #     drawBoundingBox(dspace.map, bb5, [100,255,150])
    #     dspace.showSpace()
    #     dspace.clearSpace()

    # for t in tr_all:
    #     # draw all, then draw each individually
    #     for tt in tr_all:
    #         tt.drawToSpace()
    #     # tr:Trajectory=tr_all[3]
    #     tr:Trajectory=t
    #     n = 5
    #     n_ext = 5
    #     p = tr.getNextBbII(n, n_ext, debug=True)

    #     # draw things to verify correctness:
    #     if(p[3] is not None):
    #         x_avg = p[1]
    #         x_start = x_avg[0]-5
    #         x_stop = x_avg[0]+5
    #         y_start = p[3](x_start)
    #         y_stop = p[3](x_stop)
    #         x_a = np.array([x_start, y_start])
    #         x_b = np.array([x_stop, y_stop])
    #         drawLine(dspace.map, x_a, x_b, [0,255,0])


    #     drawO(dspace.map, p[1], [0,0,0])
    #     drawO(dspace.map, p[1]+2.5*p[2], [20,20,20])
    #     drawX(dspace.map, p[1]+7.5*p[2], [20,20,20])
    #     drawO(dspace.map, p[0], [50,50,50])
    #     drawBoundingBox(dspace.map, p[4], [180,255,0])
    #     dspace.showSpace(draw_dets=False)
    #     dspace.clearSpace()
        # -----

    # return
    # stage II:
    # MAKE CONNECTION HYPOTHESES (bridged):
    # merged_trs:list[Trajectory] = tr_all
    merged_merged:list[Trajectory] = []
    merged_trs1 = []
    Q = np.diag(np.zeros(len(tr_all)))
    Q[len(tr_all)-1,len(tr_all)-1]=1
    if(stage == 2 or override):
        merged_trs1 = getMergedHypotheses(tr_all, time_window=30)
        # merged_trs1 = merged_trs1[0:7]+merged_trs1[15:22]
        # merged_trs1 = merged_trs1[15:22] + merged_trs1[0:7]
        # merged_trs1.sort(key=lambda x: x.)
        print(merged_trs1)
        print(len(merged_trs1))
        # Q = dspace.buildQBPMatrixX(merged_trs1, type=2)
    

    # GET TRAJECTORIES, THAT WERE NOT USED IN CONNECTIONS
    # print(set(tr_all))
    tr_left = set(tr_all)
    i = 0
    for t in merged_trs1:
        key_set = set(t.T2.keys())
        # print(i,key_set)
        if(len(key_set) > 1):
            tr_left = tr_left - key_set
        i = i+1
    # print("ostalo: ",list(tr_left))
    # also include just merged trajectories:
    tr_left = list(tr_left)
    # tr_left = tr_all
    # tr_left = tr_left + merged_trs1

    # stage III:
    # MAKE CONNECTION HYPOTHESES (flow)
    merged_trs2 = []
    if(stage == 3 or override):
        d2_winname = "dspace_win2"
        cv.namedWindow(d2_winname, cv.WINDOW_NORMAL)
        
        # tr_all2: list[Trajectory] = tr_all
        # tr_all2: list[Trajectory] = [tr_all[7],tr_all[9],tr_all[19],  tr_all[0],tr_all[2], tr_all[3],tr_all[12],tr_all[17]]
        # tr_all2: list[Trajectory] = [tr_all[0],tr_all[2],   tr_all[3],tr_all[12],tr_all[17]]
        # dspace.clearSpace()
        i = 0
        # for t in tr_all2:
        #     t.drawToSpace()
        #     print("%3d: showing t%d"%(i,t.id))
        #     dspace.showSpace(draw_dets=False, dspace_winname=d2_winname)
        #     dspace.clearSpace()
        #     i= i+1

        # t1 = tr_all2[0]
        # t1.drawToSpace()
        # possible_next = t1.getPossibleNext3(tr_all2, DATA_ROOT+FLOWS_PATH)
        # print(possible_next)
        # dspace.showSpace(dspace_winname=d2_winname)

        merged_trs2 = getMergedHypotheses(tr_left, type=2)
        # Q = dspace.buildQBPMatrixX(merged_trs2, type=2)
    
    merged_merged = merged_trs1+merged_trs2
    # merged_merged = merged_trs2
    Q = dspace.buildQBPMatrixX(merged_merged, type=2)

    # DEBUG: check for duplicates:
    for m in merged_merged:
        print("%5d -> "%(m.id), end="")
        m_trs = m.T2.keys()
        for mt in m_trs:
            print("%5d, "%(mt.id),end="")
        print()



    
    # SOLVE QBP:
    print(Q)
    np.savetxt("Q.txt",Q, fmt="%7.3f")
    res = dspace.solveQBP2(Q)
    # res = dspace.solveQBP(Q)
    print(res)
    v = res[0]
    
    # SHOW SELECTED:
    selected_merged = []
    for i in range(len(v)):
        if(v[i] == 1):
            selected_merged.append(merged_merged[i])
            print("t%d"%(merged_merged[i].id), end=", ")
    print()

   
        # # print(tr_to_merge)
        # tr_merged = mergeTrajectories(tr_to_merge)
        # # print("tr_merged:")
        # # print("t%-5s c=%-3d, len=%-4d, score=%9.4f %5d - %-5d" % (tr_merged.id, tr_merged.color[2], len(tr_merged.X), tr_merged.getScore2(), tr_merged.X[0].t, tr_merged.X[-1].t))
        # trs_merged.append(tr_merged)

    i = 0
    n = len(selected_merged)
    # draw to separate window
    d2_winname = "dspace_win2"
    cv.namedWindow(d2_winname, cv.WINDOW_NORMAL)
    d2_winname2 = "dspace_win3"
    cv.namedWindow(d2_winname2, cv.WINDOW_NORMAL)
    for tr in selected_merged:
        tr.drawToSpace()
    dspace.showSpace(draw_dets=False, dspace_winname=d2_winname2)
    dspace.clearSpace()
    try:
        while True:
            tr: Trajectory = selected_merged[i]
            dspace.clearSpace()
            tr.drawToSpace()
            dspace.showSpace(draw_dets=False, dspace_winname=d2_winname)
            i = (i+1) % n
    except KeyboardInterrupt:
        print()
        print("fin :>")




# debug main:
def main_d():
    print("[INFO] This is main_d. To run main, set MAIN_DEB to False.")
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
    T_OFFSET = 1160 # constant: starting time (t_global) (used also to align DetectionSpace.D indices (as it expects t0 at index 0))
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
    
    # dspace.D.append([]) # skip zero, because frames are read from 1 on
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
    
    for t in tr:
        t.drawToSpace()
    dspace.showSpace(draw_dets=DRAW_DETS)
    

    # EXTEND
    # skip_n = 790
    skip_n = 35
    skip_n = 5
    skip_n = 0
    try:
        for i in range(n_res):
            print("______________")
            print("[at time "+str(t_global)+"]")
            print("      \/ ")
            # init new frame:
            next_dets=D13[t_global]
            print("next dets: ",next_dets)
            dspace.D.append(next_dets)
            setFrameFlowAtI(dspace,t_global)
            # end init new frame

            # extend trajectories
            print("n_tr: ",len(tr))
            used_dets: set[Detection] = set()
            tr_next: list[Trajectory] = []
            print("[EXTENDING TRAJECTORIES]")
            for t in tr:
                tr_next_from_same: list[Trajectory] = []
                # [TODO] - finish
                
                # tr_next_from_same.append(t) # append current if not terminated
                # # used_dets_current, forked_tr = t.extend4()
                # used_dets_current, forked_tr = t.extend4(add_est=True)
                # tr_next_from_same = tr_next_from_same + forked_tr
                # used_dets.update(used_dets_current)
                
                # if((not t.term) and (t.holes_ref <= EXT_THR)): # if not terminated, extend it
                if((not t.term) and (t.holes_ref <= EXT_THR)): # if not terminated, extend it
                    tr_next_from_same.append(t) # append current if not terminated
                    # include current trajectory (not extended), mark it as terminated
                    t_prev = t.getCopy(deep=False) # also include current trajectory, add it to hypothesis selection
                    t_prev.term = True # previous is terminated
                    tr_next_from_same.append(t_prev)

                    # include current trajectory and extend it
                    # used_dets_current, forked_tr = t.extend4(det_add_thr=0)
                    used_dets_current, forked_tr = t.extend4() # get set of used detections and forks (for starting new trajectories from others), list of new trajectories (forks) (add 'em to tr)
                    tr_next_from_same = tr_next_from_same + forked_tr
                    used_dets.update(used_dets_current)
                else: # if terminated, do not include it in hypothesis selection, but separately
                    tr_fin.append(t)
                


                first = True
                for tt in tr_next_from_same:
                    if(first):
                        first = False
                        print(">>> t"+str(tt.id)+" - origin:",tt.origin,"score:",tt.getScore2(), "color:",tt.color, tt.X[-2:], "len:",len(tt.X))
                    else:
                        print("|-> t"+str(tt.id)+" - origin:",tt.origin,"score:",tt.getScore2(), "color:",tt.color,tt.X[-2:], "len:",len(tt.X))
                tr_next = tr_next + tr_next_from_same
            print("[DONE EXTENDING]")

            tr = tr_next

            # # all hypothesis of this iteration
            # for t in tr_next:
            #     # print("t"+str(t.id)+" - origin:",t.origin,"score:",t.getScore2(), t.X[-3:])
            #     print("t"+str(t.id)+" - origin:",t.origin,"score:",t.getScore2())

            
            # start new trajectories from points that do not belong to any trajectory
            # 1. get difference of used dets and next dets, start new trajectories from unused ones
            unused_dets = set(next_dets) - used_dets
            # print(used_dets)
            print("used dets: ", used_dets," unused dets: ", unused_dets)
            for d in unused_dets:
                t_new = Trajectory(d, dspace)
                t_new.build2()
                tr.append(t_new)

            # hypothesis selection (build qbp, )
            print("building Q")
            tr.sort(key=lambda x: x.getScore2(), reverse=True) #
            print("tr sorted: ", tr)
            Q = dspace.buildQBPMatrix3(tr)
            print(Q)
            v = dspace.solveQBP2(Q)
            print(v)

            # keep only selected:
            tr_next2 = []
            print("selected: ", end="")
            for i in range(len(v[0])):
                if(v[0][i] == 1):
                    tr_i = tr[i]
                    tr_next2.append(tr_i)
                    print("t"+str(tr_i.id),end=" ", flush=True)
                    # if(tr_i.id != 143):
                    #     tr_i.drawToSpace()
                    tr_i.drawToSpace()
            tr = tr_next2
            print()

            # print also terminated ones:
            print("len tr fin: ",len(tr_fin))
            for t in tr_fin:
                t.drawToSpace(color=[100,100,100])
                # t.drawToSpace()
            
            # on each n_stage_II_III-th frame, do stage II and III trajectory merge
            if(i % n_stage_II_III == 0):
                # [TODO]
                print("do stage II and III merging [TODO]")
                # first: stage II (is faster)
                # get unused trajectories
                # second: stage III with unused trajectories (is slower - computes flow)
                # use new trajectories in next round

            # to skip n frames (algorithm runs without visualization)
            if(skip_n <= 0):
                dspace.showSpace(draw_dets=DRAW_DETS)
                dspace.clearSpace()
            else:
                skip_n = skip_n-1
            
            # increase time on each iteration
            t_global = t_global + 1
    except KeyboardInterrupt:
        # save tr, tr_fin and T_OFFSET
        print("fin")
        if(SAVE_TRAJECTORIES):
            with open('trs.p', 'wb') as fp: # fp: file pointer?
                abc = [T_OFFSET, t_global, tr_fin, tr]
                pickle.dump(abc, fp)
        
    # END EXTEND

# quick debug main:
def main_qd():
    print("[INFO] This is main_qd. To run main, set MAIN_DEB to False. To run main_deb, set MAIN_QD to False.")
    # SEQUENCES: uncomment for different sequences: ------------------------
    
    # penguins:
    FRAMES_PATH = "frames/LaSOT_bird-2/color/"
    FLOWS_PATH = "flow_est/LaSOT_bird-2_1/"
    DETS_FILE = "detections/LaSOT_bird-2.txt"
    # T_OFFSET = 500 + 653 # constant: starting time (t_global) (used also to align DetectionSpace.D indices (as it expects t0 at index 0))
    T_OFFSET = 1160 # constant: starting time (t_global) (used also to align DetectionSpace.D indices (as it expects t0 at index 0))
    # T_OFFSET = 798
    # T_OFFSET = 1
    n = 0 # number of previous frames for trajectory init

    # ----------------------------------------------------------------------
    SAVE_TRAJECTORIES = not True
    EXTEND = not True
    D13 = readDetFile2(DATA_ROOT+DETS_FILE) # read detections from file
    t_global = T_OFFSET # current time (frame)
    
    # actual offset for detections list is: T_OFFSET - n (because there are some detections before global offset (initial trajectories))
    n_res = len(D13) - n - T_OFFSET
    tr: list[Trajectory] = []
    tr_fin: list[Trajectory] = []
    n_solve = 1 # 5 # 20 # 'time window' - # of frames between trajectory selections
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
    
    # dspace.D.append([]) # skip zero, because frames are read from 1 on
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
    
    # tr = [tr[2]]
    # print(tr)
    # for t in tr:
    #     t.drawToSpace()
    #     print(">>> t"+str(t.id)+" - origin:",t.origin,"score:",t.getScore2(), "color:",t.color, t.X[-2:], "len:",len(t.X))
    # dspace.showSpace(draw_dets=DRAW_DETS)
    

    # EXTEND (using only flow)
    if(EXTEND):
        tr_split:list[Trajectory] = []
        # t2 = tr[0]
        # tr_s = Trajectory(t2.origin,t2.detectionSpace)
        # tr_s.build2()
        t1_stop = 1288
        t2_start = 1333
        skip_n = 269
        skip_n = 0
        # SAVE_TRAJECTORIES = True
        try:
            for i in range(n_res):
                print("______________")
                print("[at time "+str(t_global)+"]")
                print("      \/ ")
                # init new frame:
                next_dets=D13[t_global]
                print("next dets: ",next_dets)
                dspace.D.append(next_dets)
                setFrameFlowAtI(dspace,t_global)
                # end init new frame
                if(t_global == 1184 or t_global == 1228 or t_global == 1219):
                    print(next_dets)
                    det_origin = next_dets[1]
                    if(t_global == 1228):
                        det_origin = next_dets[0]
                    if(t_global == 1219):
                        det_origin = next_dets[3]
                    tr_n = Trajectory(det_origin, dspace)
                    tr_n.build2()
                    tr.append(tr_n)



                # extend trajectories
                print("n_tr: ",len(tr))
                used_dets: set[Detection] = set()
                tr_next: list[Trajectory] = []
                print("[EXTENDING TRAJECTORIES]")
                
                for t in tr:
                    _,_ = t.extendFlowOnly() # get set of used detections and forks (for starting new trajectories from others), list of new trajectories (forks) (add 'em to tr)
                
                
                # if(t_global == t1_stop):
                #     tr_s1 = tr_s.getCopy()
                #     tr_split.append(tr_s1)
                
                # if(t_global == t2_start):
                #     tr_s =Trajectory(tr_s.X[-1], tr_s.detectionSpace)
                #     tr_s.build2()

                # if(t_global == 1430):
                #     tr_s2 = tr_s.getCopy()
                #     tr_split.append(tr_s2)
                if(t_global == 1194):
                    for t in tr:
                        print("t%d, o=%s" % (t.id, str(t.origin.x)))
                    tr_split.append(tr[2])
                    tr.pop(2)

                if(t_global == 1197):
                    for t in tr:
                        print("t%d, o=%s" % (t.id, str(t.origin.x)))
                    tr_split.append(tr[2])
                    tr.pop(2)

                if(t_global == 1300):
                    for t in tr:
                        print("t%d, o=%s" % (t.id, str(t.origin.x)))
                    tr_split.append(tr[2])
                    tr_split.append(tr[3])
                    tr.pop(3)
                    tr.pop(2)

                print("[DONE EXTENDING]")
                # tr = tr_next
                
                for t in tr:
                    t.drawToSpace(color=[100,100,100])
                # tr_s.drawToSpace()

                if(skip_n <= 0):
                    dspace.showSpace(draw_dets=DRAW_DETS)
                    dspace.clearSpace()
                else:
                    skip_n = skip_n-1
                
                t_global = t_global + 1 # increase time in each iter
                    
        except KeyboardInterrupt:
            # save tr, tr_fin and T_OFFSET
            print("fin")
            if(SAVE_TRAJECTORIES):
                with open('trs_flow.p', 'wb') as fp: # fp: file pointer?
                    # abc = [T_OFFSET, t_global, tr_fin, tr]
                    abc = [T_OFFSET, t_global, tr_split]
                    pickle.dump(abc, fp)
        # END EXTEND
    else:
        # LOAD TRS:
        # ------------
        Trajectory.Tid = 1000 #
        # get saved data:
        abc = []
        with open('trs_flow.p', 'rb') as fp:
            abc = pickle.load(fp)
        # print(abc)
        t_off, t_global, tr_split = abc

        
        # set last frame
        frame = getFrameAtI(t_global,DATA_ROOT+FRAMES_PATH)
        dspace.lastFrame = frame.copy()
        dspace.map = frame

        # set flow map
        flow = getFlowAtI(t_global, DATA_ROOT+FLOWS_PATH)
        flow_img = flow2img(flow)
        dspace.flow_map = flow
        dspace.last_flow_img = flow_img.copy()
        dspace.flow_img = flow_img

        for t in tr_split:
            t.detectionSpace = dspace
        # -----------------------------------

    dspace.clearSpace()

    # debug stuff:
    # tr_split = [tr_split[0],tr_split[2]]
    # tr_split = [tr_split[1],tr_split[3]]
    # tr_split = [tr_split[0],tr_split[3]]
    d2_winname = "dspace_win2"
    cv.namedWindow(d2_winname, cv.WINDOW_NORMAL)

    for t in tr_split:
        t.drawToSpace()
        print("t%d, o=%s" % (t.id, str(t.origin.x)))
        dspace.showSpace(draw_dets=DRAW_DETS, draw_last_dets_bb=False, dspace_winname=d2_winname)
        dspace.clearSpace()


    # stage III:
    # MAKE CONNECTION HYPOTHESES (flow)
    d2_winname = "dspace_win2"
    cv.namedWindow(d2_winname, cv.WINDOW_NORMAL)
    # merged_trs2 = getMergedHypotheses(tr_split, type=2, time_window=50)
    merged_trs2 = getMergedHypotheses(tr_split, type=1, time_window=50)
    
    # add also single trajectories
    # for t in tr_split:
    #     t.T2[t] = [T_INIT_S,1.0]
    #     merged_trs2.append(t)

    
    
    
    # for t in tr_split:
    #     t.drawToSpace()
    #     dspace.showSpace(draw_dets=DRAW_DETS, draw_last_dets_bb=False, dspace_winname=d2_winname)
    #     dspace.clearSpace()


    # for t in merged_trs2:
    #     t.drawToSpace()
    
    #     dspace.showSpace(draw_dets=DRAW_DETS, draw_last_dets_bb=False, dspace_winname=d2_winname)
    #     dspace.clearSpace()
    
    # hypothesis selection:
    # print(merged_trs2[0].T2, merged_trs2[1].T2)
    
    Q = dspace.buildQBPMatrixX(merged_trs2, type=2)
    # vv = np.array([1,0,0,0,1,0,1,0,1])
    # vv = np.array([1,0,0,0,0,1,0,0,1])
    # for i in range(len(Q)):
    #     print(Q[i,i], Q[0,i])
    # print(np.dot(vv,np.dot(Q,vv)))
    
    # SOLVE QBP:
    print(Q)
    # np.savetxt("Q.txt",Q, fmt="%7.3f")
    res = dspace.solveQBP2(Q)
    # res = dspace.solveQBP(Q)
    print(res)
    v = res[0]
    print()
    
    # SHOW SELECTED:
    selected_merged = []
    for i in range(len(v)):
        t = merged_trs2[i]
        if(v[i] == 1):
            # selected_merged.append(merged_trs2[i])
            # print("t%d"%(merged_trs2[i].id), end=", ")

    # for t in selected_merged:
            print("+ t%d"%(t.id))
            t.drawToSpace([0,255,0])
        else:
            print("  t%d"%(t.id))
            t.drawToSpace([100,100,100])
        dspace.showSpace(draw_dets=DRAW_DETS, draw_last_dets_bb=False, dspace_winname=d2_winname)
        dspace.clearSpace()



# other mains/mainds:
# def main_d():
#     print("[INFO] This is main_d. To run main, set MAIN_DEB to False.")
#     dspace = DetectionSpace(1,1)
#     Q=np.array([
#         [ 22.398, -11.782,  -0.,     -0.,     -0.,     -0.,     -0.   ],
#         [-11.782,  21.421,  -0.,     -0.,     -0.,     -0.,     -0.   ],
#         [ -0.,     -0.,      6.417,  -0.   ,  -0.   ,  -0.   ,  -0.   ],
#         [ -0.,     -0.,     -0.   ,   4.589,  -0.   ,  -0.   ,  -0.   ],
#         [ -0.,     -0.,     -0.   ,  -0.   ,  11.098,  -5.599,  -0.   ],
#         [ -0.,     -0.,     -0.   ,  -0.   ,  -5.599,  10.18 ,  -0.   ],
#         [ -0.,     -0.,     -0.   ,  -0.   ,  -0.   ,  -0.   ,   1.   ]])
    
#     Q=np.array([
#         [ 6.417,  -0.   ,  -0.   ,  -0.   ,  -0.   ],
#         [-0.   ,   4.589,  -0.   ,  -0.   ,  -0.   ],
#         [-0.   ,  -0.   ,  11.098,  -5.599,  -0.   ],
#         [-0.   ,  -0.   ,  -5.599,  10.18 ,  -0.   ],
#         [-0.   ,  -0.   ,  -0.   ,  -0.   ,   1.   ]])
#     Q=np.array([
#         [ 6,  -0.   ,  -0.   ,  -0.    ],
#         [-0.   ,   4,  -0.   ,  -0.    ],
#         [-0.   ,  -0.   ,  11,  -6 ],
#         [-0.   ,  -0.   ,  -6,  10  ]])
#     Q=np.array([
#         [ 11,  -6.   ,  -0.   ,  -0.    ],
#         [-6.   ,   10,  -0.   ,  -0.    ],
#         [-0.   ,  -0.   ,  6,  -0 ],
#         [-0.   ,  -0.   ,  -0,  4  ]])
#     print(Q)
#     print(dspace.solveQBP(Q))
#     print(dspace.solveQBP2(Q, debug=True))
#     pass
# def main_d():
#     print("[INFO] This is main_d. To run main, set MAIN_DEB to False.")

#     # INIT - get through first n frames and initiate (hopefully) strong trajectories:
#     # init variables used in process
    
#     # SEQUENCES: uncomment for different sequences: ------------------------
    
#     # sample of penguins file:
#     # FRAMES_PATH = "sample/"
#     # FLOW_FILE = "sample/sample_flo_2.p"
#     # DETS_FILE = "sample/dets.txt"
#     # T_OFFSET = 0 # constant: starting time (t_global) (used also to align DetectionSpace.D indices (as it expects t0 at index 0))
#     # n = 0 # number of previous frames for trajectory init
    
#     # penguins:
#     FRAMES_PATH = "frames/LaSOT_bird-2/color/"
#     FLOWS_PATH = "flow_est/LaSOT_bird-2_1/"
#     DETS_FILE = "detections/LaSOT_bird-2.txt"
#     T_OFFSET = 500 + 653 # constant: starting time (t_global) (used also to align DetectionSpace.D indices (as it expects t0 at index 0))
#     n = 20

#     # testing squares:
#     # FRAMES_PATH = "testing/sequence/"
#     # FLOWS_PATH = "testing/flows/"
#     # DETS_FILE = "testing/sequence/dets.txt"
#     # T_OFFSET = 0 # constant: starting time (t_global) (used also to align DetectionSpace.D indices (as it expects t0 at index 0))
#     # n = 0 # number of previous frames for trajectory init

#     # ----------------------------------------------------------------------

    
#     D13 = readDetFile2(DATA_ROOT+DETS_FILE) # read detections from file
#     t_global = T_OFFSET # current time (frame)
    
#     # actual offset for detections list is: T_OFFSET - n (because there are some detections before global offset (initial trajectories))
#     n_res = len(D13) - n
#     tr: list[Trajectory] = []
#     n_solve = 5 # 20 # 'time window' - # of frames between trajectory selections
#     # ------------------------------
    
#     # INIT DSPACE
#     frame = getFrameAtI(t_global,DATA_ROOT+FRAMES_PATH)
#     h,w,_ = np.shape(frame)
#     dspace = DetectionSpace(h,w, time_offset=(T_OFFSET-n)) # also pass time offset for using correct indices
#     dspace.use_flow = True
    
#     # set last frame
#     dspace.lastFrame = frame.copy()
#     dspace.map = frame

#     # set flow map
#     flow = getFlowAtI(t_global, DATA_ROOT+FLOWS_PATH)
#     flow_img = flow2img(flow)
#     dspace.flow_map = flow
#     dspace.last_flow_img = flow_img.copy()
#     dspace.flow_img = flow_img
    
#     # dspace.D.append([]) # skip zero, because frames are read from 1 on
#     print("init dspace: ", dspace.D)

#     # append detections
#     D13_init: list[Detection] = D13[T_OFFSET-n:T_OFFSET] # initial detection slice (relative to t_global; use previous n detections)
#     colorHist2Det(D13, DATA_ROOT+FRAMES_PATH, T_OFFSET-n,T_OFFSET) # also compute color hists for detections
#     flowVec2Det(D13, DATA_ROOT+FLOWS_PATH, T_OFFSET-n,T_OFFSET) # and also motion vectors (where they move in next frame)
#     print(D13[T_OFFSET-n-5:T_OFFSET+5 ])

#     for d in D13_init:
#         dspace.D.append(d)
#         # also calculate detection's motion vectors
#         print(d)
#     # END INIT DSPACE
 
#     # build trajectories from all detectinos
#     # 1. get list of all detections
#     det_list = []
#     for dl in D13_init:
#         for d in dl:
#             det_list.append(d)
    
#     # 2. build them
#     print("building: ")
#     for i in range(0,len(det_list), 1):
#         # print(i)
#         d = det_list[i]
#         new_tr: Trajectory = Trajectory(d, dspace)
#         # print(d)
#         new_tr.build()
#         # print("end of bild, get score:")
#         new_tr.getScore(E1,E2) # calculate score of trajectory (Trajectory.S is set) (merit term used in matrix)
#         tr.append(new_tr)
#         # input("continue?")
#     print("n_tr: ",len(tr))

#     # 3.1 drop redunant trajectories (ones that use same detections and have lower score than others with same detections)
#     tr = dropRedundant(tr, debug=False, recalculate=False) # no need to recalculate, because we already have calculated scores
    
#     # 3.2 choose best trajectories and draw them:
#     v = selectBest(tr, dspace, debug=True, recalculate_scores=False) # already have scores
#     tr_temp = getSelectedNvisualize(tr, v[0], dspace, True, True)
    
#     # 3.3 keep only selected (only in init phase)
#     # tr = [tr_temp[0]]
#     tr = tr_temp
#     print("tr: ",tr)
#     # END INIT
    
#     # EXTEND (step):
#     dspace.clearSpace()
#     dspace.showSpace(draw_dets=DRAW_DETS)
#     print("next time instant: ",t_global)

#     try:
#         tr_save = []
#         if(SAVE_TRAJECTORIES):
#             tr_save.append((t_global-1,copy.deepcopy(tr))) # init trajectories to save
#         # on every new frame, do:
#         for i in range(n_res):
#             # 0. set new frame
#             frame = getFrameAtI(t_global,DATA_ROOT+FRAMES_PATH)
#             dspace.lastFrame = frame.copy()
#             dspace.map = frame
#             # 0.1 set new flow image
#             flow = getFlowAtI(t_global, DATA_ROOT+FLOWS_PATH)
#             dspace.flow_map = flow
#             flow_img = flow2img(flow)
#             dspace.last_flow_img = flow_img.copy()
#             dspace.flow_img = flow_img
            

#             # 1. get detections in current frame
#             latest_dets = D13[t_global]
#             # 1.1 ALSO CALCULATE DETECTIONS' COLOR HISTOGRAMS AND MOTION VECTORS
#             # print("latest dets: ")
#             for d in latest_dets:
#                 updateDetColorHistFromFrame(d, frame)
#                 updateDetMotionVecFromFlowMap(d, flow)
#                 # print(d, end=" ", flush=True)
#             # print()
                
                
#             # -----------------------------------
#             dspace.D.append(latest_dets)

#             # 2. try to extend existing trajectories
#             dspace.clearSpace()
#             for t in tr:
#                 # check if possibly occluded; do e1Nc
#                 # if(t.possibly_occluded):
#                 #     possible_tr = t.e1Nc(n_empty=1)
#                 #     print(possible_tr)
#                 #     for tt in possible_tr:
#                 #         t.possible_next.append(tt)
#                 #         tt.getScore(E1,E2)
#                 #     # also extend possible next - [TODO]
#                 #     # t.getScore(E1,E2) # calculate score of existing trajectories (extended/extrapolated)
#                 # else:
#                 #     t.extend(n_empty=1)
#                 #     t.getScore(E1,E2) # calculate score of existing trajectories (extended/extrapolated)
#                 # t.extend(n_empty=1)
#                 # t.extendUsingFlow2(n_empty=1)
#                 # t.extend2(n_empty=1) # uses flow
#                 t.extend3(n_empty=1) # uses flow
#                 t.getScore(E1,E2) # calculate score of existing trajectories (extended/extrapolated)
                    
#             # 3. vizualization
#             for t in tr:
#                 t.drawToSpace()
#                 # also show possible occluded trajectory's possible next
#                 if(t.possibly_occluded):
#                     for tt in t.possible_next:
#                         tt.drawToSpace([0,255,0])
#             dspace.showSpace(draw_dets=DRAW_DETS)

#             # 3. for every new detection, start new trajectory:
#             for d in latest_dets:
#                 tr_new_det = Trajectory(d, dspace)
#                 tr_new_det.build()
#                 tr_new_det.getScore(E1,E2) # calculate score of new trajectories
#                 # check if redundant:
#                 if(not checkIfRedundant(tr, tr_new_det, recalculate=False, debug=False)): # if it is not redundant, then add it, otherwise do not add
#                     tr.append(tr_new_det)


#             # 4. trajectory pruning: if trajectory inactive*, remove it
#             # * inactive, if not updated for 10 consecutive frames

#             print("i, t_global: ",i, t_global)
#             # on every n_solve-th frame, do hypothesis selection again:
#             if(i%n_solve == 0):
#                 if(SAVE_TRAJECTORIES):
#                     tr_save.append((t_global,copy.deepcopy(tr)))

#                 v = selectBest(tr, dspace, debug=True, recalculate_scores=False) # no need to recalculate, because they are updated (t.getScore(E1, E2))
#                 tr_temp = []
#                 t_i = 0
#                 for t in tr:
#                     # if(v[0][t_i] == 1):
#                     #     print("-> ", end="", flush=True)
#                     # print(t," holes_ref: ", t.holes_ref, " holes_total: ", t.holes, " nss: ", t.not_selected_strike)

#                     # pruning is actually happening here
#                     if(t.holes_ref < EXT_THR):
#                         if(t.holes_ref > PO_THR):
#                             t.possibly_occluded = True # !! <------------------------ POSSIBLY OCCLUDED IS SET HERE
#                         tr_temp.append(t)
#                         # if((t.S >= 0) or ((t.S < 0) and (-t.S <= len(t.X))) ):
#                         #     if(t.not_selected_strike > UNSELECTED_STRIKE_MAX):
#                         #         t.disable_grow = True
#                         #     # exclude discontinued trajectory from hypothesis selection
#                         #     else:
#                         #         tr_temp.append(t)

#                     t_i = t_i+1

#                     # handle special (possible occluded)
                    
#                     # if(t.possibly_occluded):
#                     #     print("possibly occluded: ", t, ":")
#                     #     print("possible next: ",t.possible_next)
#                     #     getSelectedNvisualize(t.possible_next, np.ones(len(t.possible_next)), dspace, separately=True, debug=True)

#                 # tr = tr_temp
#                 # keep only selected (only for testing purposes)
#                 # tr_temp = getSelectedNvisualize(tr, v[0], dspace, separately=True, debug=True)
#                 getSelectedNvisualize(tr, v[0], dspace, separately=True, debug=True)
#                 tr = tr_temp
#             t_global = t_global + 1 # very important

#     except KeyboardInterrupt:
#         print("abort")
#         if(SAVE_TRAJECTORIES):
#             with open('hypotheses.p', 'wb') as fp: # fp: file pointer?
#                 pickle.dump(tr_save, fp)
#     # END EXTEND
# ------------

# def main():
#     # get saved data:
#     abc = []
#     with open('trs.p', 'rb') as fp:
#         abc = pickle.load(fp)
#     # print(abc)
#     t_off, t_global, tr_fin, tr = abc
#     # ------------


#     # INIT DSPACE
#     frame = getFrameAtI(t_global,DATA_ROOT+FRAMES_PATH)
#     h,w,_ = np.shape(frame)
#     dspace = DetectionSpace(h,w, time_offset=(t_global)) # also pass time offset for using correct indices
#     dspace.use_flow = True
    
#     # set last frame
#     dspace.lastFrame = frame.copy()
#     dspace.map = frame

#     # set flow map
#     flow = getFlowAtI(t_global, DATA_ROOT+FLOWS_PATH)
#     flow_img = flow2img(flow)
#     dspace.flow_map = flow
#     dspace.last_flow_img = flow_img.copy()
#     dspace.flow_img = flow_img
#     # END INIT DSPACE


#     # draw trajectories:
#     print("all trajectories:")
#     print("fin:")
#     for t in tr_fin:
#         t: Trajectory = t
#         # print("t"+str(t.id)+" c="+str(t.color[2])+", len="+str(len(t.X)), "from:", t.X[0].t, "to:", t.X[-1].t)
#         print("t%-5s c=%-3d, len=%-4d, score=%9.4f %5d - %-5d" % (t.id, t.color[2], len(t.X), t.getScore2(), t.X[0].t, t.X[-1].t))
#         t.detectionSpace = dspace
#         t.T2 = {}
#         # t.drawToSpace()
#     print("not fin:")
#     for t in tr:
#         t: Trajectory = t
#         print("t%-5s c=%-3d, len=%-4d, score=%9.4f %5d - %-5d" % (t.id, t.color[2], len(t.X), t.getScore2(), t.X[0].t, t.X[-1].t))
#         t.detectionSpace = dspace
#         t.T2 = {}
#         # t.drawToSpace()
#     print("--------------------------")

#     tr_all = tr_fin + tr

#     print(tr_all)
#     tr_all = tr_all[0:1]

#     for t in tr_all:
#         t.drawToSpace()
#         # draw first bb and extrapolated one:
#         drawBoundingBox(dspace.map, t.X[0].bb, [255,255,0])
#         bb5 = t.getNextBbII(5,5)
#         bb10 = t.getNextBbII(5,10)

#         drawBoundingBox(dspace.map, bb5, [100,255,150])
#         drawBoundingBox(dspace.map, bb10, [100,255,200])
#     dspace.showSpace()



#     # for t in tr_all:
#     #     # draw all, then draw each individually
#     #     for tt in tr_all:
#     #         tt.drawToSpace()
#     #     # tr:Trajectory=tr_all[3]
#     #     tr:Trajectory=t
#     #     n = 5
#     #     n_ext = 5
#     #     p = tr.getNextBbII(n, n_ext, debug=True)

#     #     # draw things to verify correctness:
#     #     if(p[3] is not None):
#     #         x_avg = p[1]
#     #         x_start = x_avg[0]-5
#     #         x_stop = x_avg[0]+5
#     #         y_start = p[3](x_start)
#     #         y_stop = p[3](x_stop)
#     #         x_a = np.array([x_start, y_start])
#     #         x_b = np.array([x_stop, y_stop])
#     #         drawLine(dspace.map, x_a, x_b, [0,255,0])


#     #     drawO(dspace.map, p[1], [0,0,0])
#     #     drawO(dspace.map, p[1]+2.5*p[2], [20,20,20])
#     #     drawX(dspace.map, p[1]+7.5*p[2], [20,20,20])
#     #     drawO(dspace.map, p[0], [50,50,50])
#     #     drawBoundingBox(dspace.map, p[4], [180,255,0])
#     #     dspace.showSpace(draw_dets=False)
#     #     dspace.clearSpace()
#         # -----

#     return
#     # MAKE CONNECTION HYPOTHESES:
#     merged_trs = getMergedHypotheses(tr_all)
#     print(merged_trs)
#     print(len(merged_trs))



#     # # print(tr_all)
#     Q = dspace.buildQBPMatrixX(merged_trs, type=2)
#     print(Q)
#     res = dspace.solveQBP2(Q)
#     print(res)
#     v = res[0]
    
#     selected_merged = []
#     for i in range(len(v)):
#         if(v[i] == 1):
#             selected_merged.append(merged_trs[i])

   
#     #     # print(tr_to_merge)
#     #     tr_merged = mergeTrajectories(tr_to_merge)
#     #     # print("tr_merged:")
#     #     # print("t%-5s c=%-3d, len=%-4d, score=%9.4f %5d - %-5d" % (tr_merged.id, tr_merged.color[2], len(tr_merged.X), tr_merged.getScore2(), tr_merged.X[0].t, tr_merged.X[-1].t))
#     #     trs_merged.append(tr_merged)

#     i = 0
#     n = len(selected_merged)
#     # draw to separate window
#     d2_winname = "dspace_win2"
#     cv.namedWindow(d2_winname, cv.WINDOW_NORMAL)
#     try:
#         while True:
#             tr: Trajectory = selected_merged[i]
#             dspace.clearSpace()
#             tr.drawToSpace()
#             dspace.showSpace(draw_dets=False, dspace_winname=d2_winname)
#             i = (i+1) % n
#     except KeyboardInterrupt:
#         print()
#         print("fin :>")



if __name__ == "__main__":
    if(MAIN_DEB):
        if(MAIN_QD):
            main_qd()
        else:
            main_d()
    else:
        main()