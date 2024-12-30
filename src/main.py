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
MAIN_DEB= not not True
# MAIN_DEB=False
EXT_THR=20 # maximum number of extrapolation of trajectories (# of consecutive frames without detections for that trajectory (number of relative holes, essentially))
PO_THR=7 # possibly occluded threshold: number of holes before trajectory is marked as possibly occluded
UNSELECTED_STRIKE_MAX=5 # maximum number of times that trajectory is not selected but included in set
SAVE_TRAJECTORIES=False # switch to save trajectories on every selection step for vizualization/debug purposes

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


# main:
def main():
    dspace = DetectionSpace(1,1)
    Q=np.array([
        [ 22.398, -11.782,  -0.,     -0.,     -0.,     -0.,     -0.   ],
        [-11.782,  21.421,  -0.,     -0.,     -0.,     -0.,     -0.   ],
        [ -0.,     -0.,      6.417,  -0.   ,  -0.   ,  -0.   ,  -0.   ],
        [ -0.,     -0.,     -0.   ,   4.589,  -0.   ,  -0.   ,  -0.   ],
        [ -0.,     -0.,     -0.   ,  -0.   ,  11.098,  -5.599,  -0.   ],
        [ -0.,     -0.,     -0.   ,  -0.   ,  -5.599,  10.18 ,  -0.   ],
        [ -0.,     -0.,     -0.   ,  -0.   ,  -0.   ,  -0.   ,   1.   ]])
    
    Q=np.array([
        [ 6.417,  -0.   ,  -0.   ,  -0.   ,  -0.   ],
        [-0.   ,   4.589,  -0.   ,  -0.   ,  -0.   ],
        [-0.   ,  -0.   ,  11.098,  -5.599,  -0.   ],
        [-0.   ,  -0.   ,  -5.599,  10.18 ,  -0.   ],
        [-0.   ,  -0.   ,  -0.   ,  -0.   ,   1.   ]])
    Q=np.array([
        [ 6,  -0.   ,  -0.   ,  -0.    ],
        [-0.   ,   4,  -0.   ,  -0.    ],
        [-0.   ,  -0.   ,  11,  -6 ],
        [-0.   ,  -0.   ,  -6,  10  ]])
    Q=np.array([
        [ 11,  -6.   ,  -0.   ,  -0.    ],
        [-6.   ,   10,  -0.   ,  -0.    ],
        [-0.   ,  -0.   ,  6,  -0 ],
        [-0.   ,  -0.   ,  -0,  4  ]])
    print(Q)
    print(dspace.solveQBP(Q))
    print(dspace.solveQBP2(Q, debug=True))
    pass

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
    T_OFFSET = 1160
    n = 0

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
    
    for t in tr:
        t.drawToSpace()
    dspace.showSpace(draw_dets=DRAW_DETS)
    

    # EXTEND
    skip_n = 35
    skip_n = 0
    for i in range(n_res):
        print("______________")
        print("[at time "+str(t_global)+"]")
        print("      \/ ")
        # init new frame:
        next_dets=D13[t_global]
        print("main: next dets: ",next_dets)
        dspace.D.append(next_dets)
        setFrameFlowAtI(dspace,t_global)
        # end init new frame

        # extend trajectories
        print(len(tr))
        used_dets: set[Detection] = set()
        tr_next: list[Trajectory] = []
        print("[EXTENDING TRAJECTORIES]")
        for t in tr:
            tr_next_from_same: list[Trajectory] = []
            tr_next_from_same.append(t) # append this
            if(not t.term): # if not terminated
                t_prev = t.getCopy(deep=False) # also include current trajectory, add it to hypothesis selection
                t_prev.term = True # previous is terminated
                tr_next_from_same.append(t_prev) # append copy of this (for merging forks)
                used_dets_current, new_tr = t.extend4() # get set of used detections (for starting new trajectories from others), list of new trajectories (add 'em to tr)
                tr_next_from_same = tr_next_from_same + new_tr
                used_dets.update(used_dets_current)

            first = True
            for tt in tr_next_from_same:
                if(first):
                    first = False
                    print(">>> t"+str(tt.id)+" - origin:",tt.origin,"score:",tt.getScore2(), tt.X[-2:])
                else:
                    print("|-> t"+str(tt.id)+" - origin:",tt.origin,"score:",tt.getScore2(),tt.X[-2:])
            tr_next = tr_next + tr_next_from_same
        print("[DONE EXTENDING]")

        tr = tr_next

        # # all hypothesis of this iteration
        # for t in tr_next:
        #     # print("t"+str(t.id)+" - origin:",t.origin,"score:",t.getScore2(), t.X[-3:])
        #     print("t"+str(t.id)+" - origin:",t.origin,"score:",t.getScore2())

        
        # start new trajectories from points that do not belong to any trajectory [TODO]
        # 1. get difference of used dets and next dets, start new trajectories from unused ones
        unused_dets = set(next_dets) - used_dets
        print(used_dets)
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
        tr_next2 = []

        # keep only selected:
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

        if(skip_n <= 0):
            dspace.showSpace(draw_dets=DRAW_DETS)
            dspace.clearSpace()
        else:
            skip_n = skip_n-1
        t_global = t_global + 1 # increase time in each iter
    # END EXTEND


# other mains:
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

if __name__ == "__main__":
    if(MAIN_DEB):
        main_d()
    else:
        main()