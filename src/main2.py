from helper_func import *
from DetectionSpace import DetectionSpace
from Trajectory import Trajectory
from OpticalFlow import OpticalFlow
from FeatureExtractor import FeatureExtractor, roiPool, patchesInBB, getFeaturesPca, ImagePadder
from Detection import Detection, TDet, TDet_from_Detection
from TargetDet import TargetDet
import cv2 as cv
import numpy as np
import copy # for deepcopy (visualization purposes)
import pickle
import argparse # for raft model argument parser
from pathlib import Path
from configparser import ConfigParser
np.set_printoptions(suppress=True, precision=3, linewidth=1000)

import warnings
warnings.simplefilter('ignore')

# run examples:
# python3 main2.py --sequence='/home/gasper/disk/Nedokumenti/Faks/didi_sequences/GOT-10k_GOT-10k_Val_000014' --dets='/home/gasper/Faks/3_letnik/diplomska/koda/data/detections/GOT-10k_GOT-10k_Val_000014.txt'
# python3 main2.py --sequence='/home/gasper/disk/Nedokumenti/Faks/didi_sequences/LaSOT_bird-2' --dets='/home/gasper/Faks/3_letnik/diplomska/koda/data/detections/LaSOT_bird-2.txt'
# python3 main2.py --sequence /home/gasper/disk/Nedokumenti/Faks/didi_sequences/LaSOT_bird-2/ --dets /home/gasper/Faks/3_letnik/diplomska/koda/data/detections/LaSOT_bird-2.txt --feat /home/gasper/disk/Nedokumenti/Faks/didi_sequences/LaSOT_bird-2_feat/
# CUDA_VISIBLE_DEVICES=2 python3 main2.py --sequence='/home/gasper/tracker_ws/workspace/sequences/LaSOT_bottle-12' --dets='/home/gasper/tracker_ws/dets/LaSOT_bottle-12.txt' --track='/home/gasper/tracker_ws/tracker/'
# CUDA_VISIBLE_DEVICES=2 python3 main2.py --sequence /home/gasper/tracker_ws/workspace/sequences/LaSOT_bottle-12 --dets /home/gasper/tracker_ws/dets/LaSOT_bottle-12.txt --track /home/gasper/tracker_ws/tracker/ --feat /home/gasper/precomputed/LaSOT_bottle-12_pr/LaSOT_bottle-12_feat/

OPTIONS = getOptionNamespaceWdefaultInit('main2', 'defaults.ini')
# OPTIONS = getOptionNamespaceWdefaultInit('main2', 'options.ini')
# OPTIONS = getOptionNamespaceWdefaultInit('main2', 'options2.ini') # on server
# print(OPTIONS)

# tracker cache: tracker/flow -> flow, tracker -> result, mogoce tut trajektorije
RESULT_PATH = OPTIONS['result_path']
COMPUTED_FLOW = RESULT_PATH+"flow/"
COMPUTED_FEAT = RESULT_PATH+"feat/"
ASK_BEFORE_FLOW_DELETE = OPTIONS['ask_before_flow_delete'] # the '-y' kinda flag
# --------


DRAW_DETS=OPTIONS['draw_dets']
MAIN_N=int(OPTIONS['main_n']) # 0: main, 1: main_d1, 2: main_d2, ...
SAVE_TRAJECTORIES=OPTIONS['save_trajectories'] # switch to save trajectories on every selection step for vizualization/debug purposes
EXT_THR=int(OPTIONS['ext_thr']) # maximum number of extrapolation of trajectories (# of consecutive frames without detections for that trajectory (number of relative holes, essentially)) (was 10, now 5)
T_INIT_S=float(OPTIONS['t_init_s']) # initial score of trajectory (when merge)
MAX_N_TR_MERGE_ONCE=int(OPTIONS['max_n_tr_merge_once']) # maximum number of merged trajectories at once (to limit recursion depth) (100)


# some main-specific functions and constants:
USE_PRECOMPUTED = OPTIONS['use_precomputed'] # use precomputed optical flow (set to False to calculate it on the go) (flows-path must be set)
USE_PRECOMPUTED_FLOW = OPTIONS['use_precomputed_flow']
USE_PRECOMPUTED_FEAT = OPTIONS['use_precomputed_feat']
if(USE_PRECOMPUTED):
    USE_PRECOMPUTED_FLOW = True
    USE_PRECOMPUTED_FEAT = True

# function is used when new flow is computed (on every new frame, for stage III, getFlow is still used, but on files that this thing generated previously)
PRECOMPUTED_FLOW_PATH = OPTIONS['precomputed_flow_path']
PRECOMPUTED_FEAT_PATH = OPTIONS['precomputed_feat_path']

# flow, features:
COMPUTE_OR_GET_TEST = OPTIONS['compute_or_get_test']
def computeOrGetFlowAtI(i:int, optical_flow_gen: OpticalFlow=None):
    if(USE_PRECOMPUTED_FLOW): return getFlowAtI(i, PRECOMPUTED_FLOW_PATH)
    if(COMPUTE_OR_GET_TEST): return computeFlowAtITest(i, save_flow=True) # WARN: this is to test only, on real use cases, use ^
    return optical_flow_gen.computeFlowAtI(i)

def computeOrGetPCAFeaturesAtI(i:int, image:np.ndarray, feature_ext:FeatureExtractor, save_feat=False):
    if(USE_PRECOMPUTED_FEAT): return getFeaturesAtI(i, PRECOMPUTED_FEAT_PATH)
    if(COMPUTE_OR_GET_TEST): return computePCAFeaturesAtItest(i, save_feat=True)
    print("[getOrComputePCAFeatures] computing features")
    frame_feat = feature_ext.getFeatures(image)['x_norm_patchtokens'].to('cpu').numpy() # get frame features (patchtokens)
    pca_feat = getFeaturesPca(frame_feat, (feature_ext.last_padder.patch_h, feature_ext.last_padder.patch_w))
    print("[getOrComputePCAFeatures] done computing")
    if(save_feat):
        np.save(COMPUTED_FEAT+i2frameI(i)+".npy",pca_feat)
    return pca_feat

def computeFlowAtITest(i:int, save_flow=False):
    flow = getFlowAtI(i, PRECOMPUTED_FLOW_PATH)
    if(save_flow):
        np.save(COMPUTED_FLOW+i2frameI(i)+".npy",flow) # save as nnnnnnnn.npy (numpy matrix)
    return flow

def computePCAFeaturesAtItest(i:int, save_feat=False):
    pca_feat = getFeaturesAtI(i, PRECOMPUTED_FEAT_PATH)
    if(save_feat):
        np.save(COMPUTED_FEAT+i2frameI(i)+".npy",pca_feat)
    return pca_feat
# -----------------

# function sets frame and flow to detection space and returns frame (get features function needs this)
def setFrameFlowAtI(dspace: DetectionSpace, i: int, optical_flow_gen: OpticalFlow = None, frames_path=""):
    # print("fpath: %s"%frames_path)
    # set new frame
    frame = getFrameAtI(i,frames_path)
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
    if(i > 1):
        flow_path = ""
        if(USE_PRECOMPUTED_FLOW): flow_path = PRECOMPUTED_FLOW_PATH
        else: flow_path = COMPUTED_FLOW
        flow_to = getFlowAtI(i-1, flow_path)
        # flow_to = computeOrGetFlowAtI(i, optical_flow_gen=optical_flow_gen)
        dspace.flow_to = flow_to
    else: # there is no flow into first frame
        h,w = dspace.hw
        dspace.flow_map=np.zeros((h,w,2)).astype(np.uint8) # optical flow (outta this frame)
    return frame

# pruning mechanisms:
# get number of pure trajectories: that only have merit term
def getPureTrN(Q):
    n, n = np.shape(Q)
    num = 0
    for i in range(n):
        if(isPure(Q,i)):
            num = num+1
        
    # print(num)
    return num

# Q matrix analyze
# function checks if trajectory in Q at i is pure
def isPure(Q,i):
    vrni = False
    n_nonzero = nNonzero(Q[:,i])
    if(n_nonzero == 1):
        vrni=True
    return vrni

# function checks number of interactions
def nNonzero(q_column):
    n_nonzero = np.sum(np.where(np.isclose(q_column, 0), 0, 1))
    return n_nonzero

# function gets list of trajectories, builds Q, analyzes it, and returns new Q and trajectory list with trajectories, that are selected
# returns (simple, all_else)
def analyzeTrsWithQ(tr_list: list[Trajectory], type=1):
    # 1. build Q:
    Q = buildQBPMatrixX(tr_list, type=type)
    # Q[0,19] = 0
    # Q[19,0] = 0
    # Q[0,18] = -1
    # Q[18,0] = -1
    # Q[0,2] = -1
    # Q[2,0] = -1
    n, _ = np.shape(Q)
    
    # 2. build list with number of interactions based on Q
    list_c = []
    for i in range(n):
        x = Q[:,i]
        list_c.append(nNonzero(x))
    # print(Q)
    # print(list_c)

    # marker array (mark which trs to include and which not to)
    marker_array = np.ones(n).astype(np.bool_)

    simple_trs = []
    # 1. pass: get pure trs to simple_trs
    for i in range(n):
        tr_i = tr_list[i]
        if(list_c[i] == 1):
            # this is pure tr, does not interact with any other
            simple_trs.append(tr_i)
            marker_array[i] = False # mark it as false

    # 2. pass: check those that have 2 interactions: if they interact with tr with same not_so_much_unique_id,
    # and also that one interacts with only 2, then use one that has higher score
    for i in range(n):
        if(not marker_array[i]): continue # skip those that have been filtered out in 1. pass
        tr_i = tr_list[i]
        if(list_c[i] == 2):
            # find next with same nsmuid, if it also has 2 trs, add it to simple trs:
            j = i+1
            while(j < n): # simulation of classic for loop from java/c
                if(not marker_array[j]): 
                    j = j + 1
                    continue # never mind those
                tr_j = tr_list[j]
                if(list_c[j] == 2):
                    # this is for those, that are actually same one, but not including last one (terminated, duplicated)
                    if(tr_j.not_so_much_unique_id == tr_i.not_so_much_unique_id):
                        # add one that has bigger score:
                        if(Q[i,i] > Q[j,j]):
                            simple_trs.append(tr_i)
                        else:
                            simple_trs.append(tr_j)

                        # mark both as false
                        marker_array[i] = False
                        marker_array[j] = False
                    elif(not np.isclose(Q[i,j], 0)):
                        # solve qbp for those two:
                        a = [Q[i,i],Q[j,j],(Q[i,i]+Q[j,j])+2*Q[i,j]]
                        max_a = 0
                        max_i = 0
                        for k in range(3):
                            if(a[k] >= max_a):
                                max_a = a[k]
                                max_i = k+1
                        if(max_i == 1): # if max is 1
                            simple_trs.append(tr_i)
                        if(max_i == 2): # if max is 2
                            simple_trs.append(tr_j)
                        if(max_i == 3): # if max is both
                            simple_trs.append(tr_i)
                            simple_trs.append(tr_j)
                        marker_array[i] = False
                        marker_array[j] = False
                j = j + 1
    
    # 3. pass: collect all trajectories, that are left:
    all_else: list[Trajectory] = []
    for i in range(n):
        if(marker_array[i]): all_else.append(tr_list[i])
    # print(marker_array)
    return simple_trs, all_else
# --------------------

# function checks whether trajectory tr is within max_dist radius around detections (usually one, but supports many)
def isWithinDets(tr:Trajectory, dets:list[Detection], max_dist=150, also_return_dists=False):
    if(not also_return_dists):
        for det in dets:
            dist = getDistBetweenDets(det, tr.X[-1]) # get distance between last trajectory point and detection
            if(dist < max_dist):
                return True
        return False
    else: # debug version:
        dists: list[float] = []
        ret_val = False
        for det in dets:
            dist = getDistBetweenDets(det, tr.X[-1]) # get distance between last trajectory point and detection
            dists.append(dist)
            if(dist < max_dist):
                ret_val = True
        return (ret_val, dists)

# function filters out all trajectories, that are not close enough to current target det (dets in list, could be more targets)
def getPruned(trs: list[Trajectory], dets: list[Detection], n_prune=5, max_dist=150):
    tr_pruned: list[Trajectory] = []
    if(dets):
        for t_i in range(len(trs)):
            tr_i = trs[t_i]
            if( (n_prune == -1) or (len(tr_i.X) == n_prune)):
                is_within, dists = isWithinDets(tr_i, dets, max_dist, also_return_dists=True)
                printTrWithStats(tr_i, t_i, add_to_end=str(dists)+" "+str(is_within))
                # is_within = isWithinDets(tr_i, dets, max_dist) # use this in end version
                # printTrWithStats(tr_i, t_i, add_to_end=str(is_within))
                if(is_within):
                    tr_pruned.append(tr_i)
                else:
                    tr_i.term = True
            else:
                tr_pruned.append(tr_i)

    return tr_pruned
# ------------------


def updateFinInIdTR(idTR_map: dict[int, Trajectory], t_current:int):
    for k in idTR_map.keys():
        tr = idTR_map[k]
        if(tr.X[-1].t < t_current):
            tr.term=True

# function to select best trajectory around detection at time t
def selectBestTrAroundDetAtTime(tr_list:list[Trajectory], bbDet, time:int=0, time_add:int=10):
    max_iou = 0
    selected_t = tr_list[0]
    for trr in tr_list:
        first_d = trr.X[0]
        if(first_d.t <= 10): # search only among few first frames
            selection_det = bbDet2Det(bbDet,0)
            iou = IoU(first_d, selection_det)
            print("t%5d (t%5d): %5.2f"%(trr.not_so_much_unique_id, trr.id, iou))
            if(iou > max_iou):
                max_iou = iou
                selected_t = trr
            elif(iou == max_iou):
                if(selected_t.getScore2() < trr.getScore2()):
                    selected_t = trr
    return selected_t

# function to select best trajectory around detection at time t
def selectBestTrAroundDetAtTime2(tr_list:list[Trajectory], dets: list[Detection]):
    max_iou = 0
    selected_t = tr_list[0]
    for detl in dets:
        if(detl):
            det = detl[0]
            for trr in tr_list:
                first_d = trr.X[0]
                if(first_d.t == det.t): # search only among few first frames
                    selection_det = det
                    iou = IoU(first_d, selection_det)
                    print("t%5d (t%5d): %5.2f"%(trr.not_so_much_unique_id, trr.id, iou))
                    if(iou > max_iou):
                        max_iou = iou
                        selected_t = trr
                    elif(iou == max_iou):
                        if(selected_t.getScore2() < trr.getScore2()):
                            selected_t = trr
    return selected_t

# II. qbp-specific functions:
# function merges trajectories in tr_list into single trajectory
def mergeTrajectories2(tr_list:list[tuple[Trajectory, float, float]]) -> Trajectory:
    t_merged = None # first
    # sort 'em by time:
    tr_list.sort(key=lambda x: x[0].X[0].t)

    min_id = 0
    merge_scores = []
    # print("merging:")
    # print([a[0].not_so_much_unique_id for a in tr_list])
    for p in tr_list:
        t = p[0]
        # print(t)
        # printTrWithStats(t)
        score = p[1]
        # print(p)
        score2 = p[2] # penalty is calculated from this
        ext_dets = p[3]
        if(t_merged is None):
            t_merged = t.getCopy() # origin is set here, and is correct
            min_id = t.not_so_much_unique_id
        else:    
            t.X[0].color = [10,10,150] # red color to mark merge gap
            if(t.X[0].t < t_merged.X[-1].t):
                # print("we have overlap")
                if(isOverlapValid(t_merged, t)):
                    # print("overlap merge")

                    # 1. find first that overlaps
                    i = len(t_merged.X)-1
                    times_of_overlap = set()
                    while(t_merged.X[i].t >= t.X[0].t):
                        times_of_overlap.add(t_merged.X[i].t)
                        i = i-1
                    new_x = t_merged.X[0:i+1] + t.X
                    t_merged.X = new_x # new X
                    # print(times_of_overlap)

                    # 2. get dets from D2 that have that time
                    dets_overlap = set()
                    for d in t_merged.D2.keys():
                        if(d.t in times_of_overlap):
                            dets_overlap.add(d)
                    
                    # 3. new D2 (is already set, just unset scores):
                    for d in dets_overlap:
                        # print("setting %.2f to %.2f (of %s)"%(t_merged.D2[d],0.0, str(d)))
                        t_merged.D2[d] = 0.0 # remove score from overlaping


                else:
                    print("invalid case, skipping")
                    continue
            else:
                tx_add = t.X
                if(t.X[0].t == t_merged.X[-1].t):
                    t_merged.X[-1] = t.X[0]
                    tx_add = t.X[1:]
                
                # t_merged.X = t_merged.X + ext_dets + t.X
                t_merged.X = t_merged.X + ext_dets + tx_add

            # all else:
            t_merged.D2 = {**t_merged.D2, **t.D2} # merge D2 (detection -> score (float))
            t_merged.T2 = {**t_merged.T2, **t.T2} # also merge T2 (Trajectory -> [score, penalty])
            t_merged.F = {**t_merged.F, **t.F} # also merge F
            # must do: X, D2, T2
            # should do: holes, holes_ref, term, color
            
            t_merged.holes = t_merged.holes + t.holes
            t_merged.term = t.term

            # also merge looks
            new_features, new_features_i = add2AvgTr(t_merged, t)
            t_merged.visual_avg = new_features
            t_merged.visual_n = new_features_i
            # ---------------


            if(t.id < min_id):
                min_id = t.not_so_much_unique_id

        t_merged.T2[t] = [score, score2]
        merge_scores.append(t_merged.T2[t])
    t_merged.not_so_much_unique_id = min_id # identity is propagated
    return t_merged

# map: tr -> list[tuple]: possible next
trToPossibleNext = {}
def getPossibleNextWithMemo(tr: Trajectory, tr_list:list[Trajectory], time_window=20, type=1, also_overlapping=False):
    global trToPossibleNext
    # print(tr.id)
    if(tr in trToPossibleNext):
        # print("is in memo, returning")
        return trToPossibleNext[tr]
    else:
        # also check overlaping trajectories
        # print("is not in memo")
        vrni = []
        if(also_overlapping):
            vrni = tr.getPossibleNextOverlap(tr_list, time_overlap=time_window)
        if(type == 1): # bridge (linear extrapolation)
            vrni = vrni + tr.getPossibleNext2(tr_list, time_window=time_window)
            trToPossibleNext[tr] = vrni
            # print(vrni)
            return vrni
        # elif(type == 2): # bridge (flow)
        else:
            flows_path = ""
            if(USE_PRECOMPUTED_FLOW):
                flows_path = PRECOMPUTED_FLOW_PATH
            else:
                flows_path = COMPUTED_FLOW
            # return tr.getPossibleNext3(tr_list, flows_path, time_window=time_window)
        
            vrni = vrni + tr.getPossibleNext3(tr_list, flows_path, time_window=time_window)
            trToPossibleNext[tr] = vrni
            # print(vrni)
            return vrni
        # else: # overlap
        #     vrni = tr.getPossibleNextOverlap(tr_list, time_overlap=time_window)
        #     trToPossibleNext[tr] = vrni
        #     # print(vrni)
        #     return vrni

# function generates all possible hypotheses from one specific hypothesis
# if with_qbp is set (it is reasonable to also set max_n to something like 5 or 10), function
# generates trajectories in sequences (not exceeding recursion depth of extendAllPossibleNext, which is value of max_n)
# resulting in max merged trajectory sizes of max_n. Those intermediate trajectries are then passed through qbp for 
# minimizing the need for searching through large number of less important trajectories (note that search space can
# grow uncontrollably, as observed during testing). The merging now continues for these new (selected) trajectories
def getMergedHypothesis(tr:Trajectory, tr_list: list[Trajectory], time_window=20, type=1, also_overlapping=False, reset_memo=True, also_drop_redundant=False, max_n=-1, with_qbp=False):
    # this is needed if used alone, when looping, this should be done outside of loop instead (if loop: reset_memo = False)
    global MAX_N_TR_MERGE_ONCE
    global T_INIT_S
    if(reset_memo):
        global trToPossibleNext
        trToPossibleNext = {}
    drop_after_n = -1
    if(also_drop_redundant): drop_after_n = 200
    max_n1 = MAX_N_TR_MERGE_ONCE
    if(max_n > 0): max_n1 = max_n
        
    tr_possible_next = getPossibleNextWithMemo(tr, tr_list, time_window=time_window,type=type, also_overlapping=also_overlapping)
    # tr_possible_next = getPossibleNextWithMemo(tr1, tr_list, time_window=time_window, type=1, also_overlapping=True)
    all_possible_next = extendAllPossibleNext(tr, [(tr,T_INIT_S,1.0,[])],tr_possible_next,tr_list, type=type, time_window=time_window, max_n=max_n1, also_overlapping=also_overlapping, drop_redundant_after_n=drop_after_n)
    if(not with_qbp):
        return all_possible_next
    else:
        Q = buildQBPMatrixX(all_possible_next, type=2)
        v = solveQBP2(Q)
        next_merged = getSelected(v[0], all_possible_next)
        continue_loop = True # flag is set to false, when none of the trajectories can be further extended (lack of presence of 'possible next')
        while(continue_loop):
            # select best:
            
            # 1. extend all possible:
            all_possible_next = []
            continue_loop = False
            for tr in next_merged:
                tr_possible_next = getPossibleNextWithMemo(tr, tr_list, time_window=time_window,type=type, also_overlapping=also_overlapping)
                tr_possible_next_bool = bool(tr_possible_next)
                continue_loop = continue_loop or tr_possible_next_bool # if all false, ...
                if(tr_possible_next_bool):
                    all_possible_next = all_possible_next + extendAllPossibleNext(tr, [(tr,T_INIT_S,1.0,[])],tr_possible_next,tr_list, type=type, time_window=time_window, max_n=max_n1, also_overlapping=also_overlapping, drop_redundant_after_n=drop_after_n)

            if(not all_possible_next): break
            # 2. select and continue
            Q = buildQBPMatrixX(all_possible_next, type=2)
            v = solveQBP2(Q)
            next_merged = getSelected(v[0], all_possible_next)
            print("next merged: ")
            printTrListWithStatsOrdered(next_merged)
        return next_merged

            
    
# function generates and returns all possible connections with other trajectories (similar to extend, but it works with whole trajectories now)
# just a loop version of ^
def getMergedHypotheses(tr_list: list[Trajectory], time_window=20, type=1, also_overlapping=False, also_drop_redundant=False):
    # init memo:
    global trToPossibleNext
    trToPossibleNext = {}
    mtr_hypotheses = []
    for tr in tr_list:
        all_possible_next = getMergedHypothesis(tr=tr, tr_list=tr_list, time_window=time_window, type=type, also_overlapping=also_overlapping, reset_memo=False, also_drop_redundant=also_drop_redundant)
        mtr_hypotheses = mtr_hypotheses + all_possible_next
    return mtr_hypotheses

# metod extends trajectories
# t: current trajectory
# collected: to be merged
# possible next: possible next trajectories that current can 'see'
# all_trs: all trajectories (because we do not have it in dspace)
CONTINUE_REC = True # for debugging purposes
def extendAllPossibleNext(t:Trajectory, collected:list[tuple[Trajectory, float]], possible_next: list[tuple[Trajectory, float]], all_trs: list[Trajectory], type=1, time_window=20, max_n=MAX_N_TR_MERGE_ONCE, also_overlapping=False, drop_redundant_after_n=-1):
    global CONTINUE_REC
    # 1. check if there are no more possible next (or maximum is reached)
    if(not possible_next or max_n == 0):
        # merge 'em
        # print(collected)
        t_merged = mergeTrajectories2(collected)
        # print(collected[0])
        skipped_time = 0
        time_l = collected[0][0].X[0].t
        time_h = time_l
        
        collected_ids = [x[0].not_so_much_unique_id for x in collected]
        print(collected_ids)
        
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
    if(not CONTINUE_REC):
        return mtr_hypotheses
    try:
        # 2. possible_next_from_this = extendAllPossibleNext()
        # print(possible_next)
        for p in possible_next:
            # tr_possible_next = tr.getPossibleNext(all_trs, max_space_diff=50)
            tr = p[0]

            # check if p is already contained in current path. If it is, continue (to avoid cycles)
            already_in = False
            for c in collected:
                tr2 = c[0]
                if(tr == tr2):
                    already_in = True
                    break
            if(already_in): continue

            tr_possible_next = getPossibleNextWithMemo(tr, all_trs, time_window, type, also_overlapping=also_overlapping)
            len_tr_possible_next = len(tr_possible_next)
            print("[extendAllPossibleNext] possible next: %d"%len_tr_possible_next)
            collected.append( p ) # add it
            possible_next_trs = extendAllPossibleNext(tr, collected, tr_possible_next, all_trs, type=type, time_window=time_window, max_n=max_n-1, also_overlapping=also_overlapping, drop_redundant_after_n=drop_redundant_after_n)
            collected.pop() # remove it

            
            mtr_hypotheses = mtr_hypotheses + possible_next_trs
    except KeyboardInterrupt:
        CONTINUE_REC = False
    # 3. return all collected trajectories
    mtr_len = len(mtr_hypotheses)
    if((drop_redundant_after_n > 0) and (mtr_len > drop_redundant_after_n)):
        print("DROPING REDUNDANT: ----------------------------------------------------------------------------------------- IOIOIOIOIOOIOIOIOIOIOIO")
        mtr_hypotheses = dropRedundant(mtr_hypotheses, red_level=2)
        # print("length (previous, new): %d, %d"%(mtr_len, len(mtr_hypotheses)))
    print(mtr_len)
    return mtr_hypotheses

# function to write trajectory to file
def tr2File(selected_t, pad, path):
    lines_to_file = selected_t.tr2bbStr(pad).split('\n')
    lines_to_file[0] = "1"
    str_to_file = '\n'.join(lines_to_file)
    with open(path+"results.txt", 'w') as fp: # fp: file pointer?
        fp.write(str_to_file)


# visual-features-related functions:
# function connects trajectory using visual information
def connectTrs(tr1:Trajectory, tr_all:list[Trajectory], max_time=100):
    if(tr1 is None):
        return []
    # print("t%d:"%(tr1.not_so_much_unique_id))

    end_time = tr1.X[-1].t
    max_score = 0
    tr_add = None
    for t in tr_all:
        visual_score = getTrVisualSimilarity(tr1, t)
        # print("t%d - t%d: %16.14f"%(tr1.not_so_much_unique_id, t.not_so_much_unique_id, visual_score))
        time_diff = t.origin.t - end_time

        if(time_diff >= 0 and time_diff <= max_time):
            visual_score = getTrVisualSimilarity(tr1, t)
            score = visual_score*(0.5*(1+(time_diff/max_time)))
            # print("t%d - t%d: %6.2f"%(tr1.not_so_much_unique_id, t.not_so_much_unique_id, visual_score))
            # print("t%d - t%d: %6.2f"%(tr1.not_so_much_unique_id, t.not_so_much_unique_id, score))
            if(visual_score > max_score):
                max_score = visual_score
                tr_add = t
    # print()
    if(tr_add is None):
        return []
    else:
        return [tr_add] + connectTrs(tr_add, tr_all)

# function returns init variables from options file
def getSequenceConsts(config_name, init_file: str = 'sequences.ini'):
    init_data = ConfigParser()
    init_data.read(init_file)
    
    frames_path = init_data.get(config_name,'frames_path')
    precomputed_flow = init_data.get(config_name,'precomputed_flow')
    gt_path = frames_path+"../groundtruth.txt"
    dets_path = init_data.get(config_name,'dets_path')
    t_offset = init_data.get(config_name,'t_offset')
    precomputed_feat = init_data.get(config_name, 'precomputed_feat')
    return int(t_offset), frames_path, dets_path, gt_path, precomputed_flow, precomputed_feat

def read_tr_file(filename='trs.p'):
    PATH_TO_DET_FILE = 'tracker/'+filename
    return readTrajectoryFile(PATH_TO_DET_FILE)

# main:
def main():
    # optional
    global OPTIONS
    global PRECOMPUTED_FEAT_PATH
    global PRECOMPUTED_FLOW_PATH
    
    # set default to tracker ws (./tracker/...)
    global COMPUTED_FLOW
    global COMPUTED_FEAT
    global RESULT_PATH
    global USE_PRECOMPUTED
    global USE_PRECOMPUTED_FEAT
    global USE_PRECOMPUTED_FLOW
    
    global SAVE_TRAJECTORIES
    SAVE_FEATURES = OPTIONS['save_features']
    RUN_ALL = OPTIONS['run_all']

    # some defaults:
    T_OFFSET = int(OPTIONS['t_offset'])

    # ARGUMENTS
    parser = argparse.ArgumentParser()
    # model arguments:
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    # dataset-specific paths
    parser.add_argument('--sequence', help="path to sequence")
    parser.add_argument('--dets', help="path to detections")
    parser.add_argument('--track', help="path to tracker directory")
    parser.add_argument('--flow', help="path to precalculated flow")
    parser.add_argument('--feat', help="path to precalculated features")
    args = parser.parse_args()

    # SEQUENCES:
    USE_INIT_FILE = OPTIONS['use_init_file'] # use options.ini
    if(USE_INIT_FILE):
        sequence_str = OPTIONS['sequence']
        T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getSequenceConsts(sequence_str)
        # T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getSequenceConsts('got10k14') # got10k14
        # T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getSequenceConsts('coin18') # coin18
        # T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getSequenceConsts('bird2') # lasot bird2
        # T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getSequenceConsts('chameleon20') # lasot chameleon20
    else: # do not use init file
        if(args.sequence is not None):
            FRAMES_PATH = args.sequence+"/color/"
            GT_PATH = args.sequence+"/groundtruth.txt"
        else:
            if(not USE_INIT_FILE):
                print("[frames] please set path to sequence with --sequence=<path to sequence>")
                exit(1)

        if(args.dets is not None):
            DETS_FILE = args.dets
        else:
            if(not USE_INIT_FILE):
                print("[dets] please set path to detections with --dets=<path to dets file>")
                exit(1)

        # set computed flow and features path, results (tracker workspace)
        if(args.track is not None):
            COMPUTED_FLOW = args.track+"flow/"
            COMPUTED_FEAT = args.track+"feat/"
            RESULT_PATH = args.track
        # else: use default
        
        # if precomputed, then read directly from precomputed path
        if(USE_PRECOMPUTED_FEAT):
            print("[precomputed] using precomputed features")
            if(args.feat is not None):
                PRECOMPUTED_FEAT_PATH = args.feat
            else:
                print("[warn] please set path to precomputed features dir with --feat=<path to precomputed features>")
                exit(1)

        if(USE_PRECOMPUTED_FLOW):
            print("[precomputed] using precomputed flow")
            if(args.flow is not None):
                PRECOMPUTED_FLOW_PATH = args.flow
            else:
                print("[warn] please set path to precomputed flow dir with --flow=<path to precomputed features>")
                exit(1)
            
    
            

    # print(args.sequence)
    print("frames: ",FRAMES_PATH)
    print("dets:   ",DETS_FILE)
    print("flow:   ",COMPUTED_FLOW)
    print("feat:   ",COMPUTED_FEAT)
    print("results:",RESULT_PATH)
    print("preflow:",PRECOMPUTED_FLOW_PATH)
    print("prefeat:",PRECOMPUTED_FEAT_PATH)
    # ----------------------------------------------------------------------
    # return
    # INIT
    D13 = readDetFile2(DETS_FILE) # read detections from file
    D13: list[list[Detection]] = [[]]+D13 # offset it (because time starts at 1, there is nothing on 0)
    GT13: list[list[Detection]] = [[]]+readDetFile2(GT_PATH,T_OFFSET) # ground truth: for testing purposes only
    t_global = T_OFFSET # current time (frame)
    # print(t_global, D13[T_OFFSET: T_OFFSET+10])

    # actual offset for detections list is: T_OFFSET - n (because there are some detections before global offset (initial trajectories))
    n = 0 # number of previous frames for trajectory init (not needed)
    n_res = len(D13) - n - T_OFFSET
    n_all = n_res
    tr: list[Trajectory] = []
    tr_fin: list[Trajectory] = []
    idTr_map: dict[int, Trajectory] = {} # id -> tr mapping: when merging, merged trajectory gets least id - this is used in the end (it contains all id-s that had been tracked)
    # n_stage_II_III = 120 # 5 # 20 # 'time window' - # of frames between trajectory selections
    n_stage_II_III = 30
    # stage_II_III_timew = 20 # time window for merging of trajectories
    stage_II_III_timew = 10

    # create tracker cache dir:
    # Path("tracker/flow/").mkdir(parents=True, exist_ok=True)
    # ------------------------------

    # INIT DSPACE
    frame = getFrameAtI(t_global,FRAMES_PATH)
    h,w,_ = np.shape(frame)
    # dspace = DetectionSpace(h,w, time_offset=(T_OFFSET-n), show_flow=False, disable_vis=RUN_ALL) # also pass time offset for using correct indices | no visualization on run all
    # dspace = DetectionSpace(h,w, time_offset=(T_OFFSET-n), show_flow=True, disable_vis=RUN_ALL) # also pass time offset for using correct indices | no visualization on run all
    dspace = DetectionSpace(h,w, time_offset=(T_OFFSET-n), show_flow=False, disable_vis=RUN_ALL) # also pass time offset for using correct indices | no visualization on run all

    # set last frame
    dspace.lastFrame = frame.copy()
    dspace.map = frame

        # INIT RAFT MODEL FOR OPTICAL FLOW ESTIMATION, IF NEEDED (if USE_PRECOMPUTED is set to False)
    of: OpticalFlow = None
    if(not USE_PRECOMPUTED_FLOW):
        # of:OpticalFlow = OpticalFlow(args,frames_path=FRAMES_PATH, save_path=FLOW_SAVE_PATH)
        of = OpticalFlow(args,frames_path=FRAMES_PATH, save_path=COMPUTED_FLOW) # save_flow = True
        # END INIT RAFT
    
        # INIT FEATURE EXTRACTOR
    featureExt: FeatureExtractor = None
    if(not USE_PRECOMPUTED_FEAT):
        featureExt: FeatureExtractor = FeatureExtractor(model_size='base')
    # featureExt = FeatureExtractor(model_size='small')
    frame_feat_padding: tuple[int, int, int, int] = getImagePadding(np.shape(frame)) # get feature image offsets
    frame_features = computeOrGetPCAFeaturesAtI(t_global, frame, featureExt, save_feat=SAVE_FEATURES) # get features, init trs
        # END INIT FEATURES
    
    
    # set flow map
    # flow = getFlowAtI(t_global, FLOWS_PATH)
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

        # patches = getFeaturesFromFeatureMapAndPadding(d, feature_map_and_featExt=(frame_features, featureExt))
        patches = getFeaturesFromFeatureMapAndPadding3(d, feature_map_and_padding=(frame_features, frame_feat_padding))
        addToAvgTr(patches, t)

        tr.append(t)
        # idTr_map[t.not_so_much_unique_id] = t # add it to map
    
    print("init trs:")
    print(tr)
    # printTrListWithStatsOrdered(tr)
    # END INIT TRAJECTORIES

    # INIT TARGET (visually)
    target_d = TargetDet(GT13[t_global][0]) # first frame: use groundtruth as target
    target_d.tr = target_d.getNearestTr(tr) # init it
    target_d.setLastTDetProps() # set properties of last tdet in trajectory + visuals

    print("%s"%('this is target:'))
    print(target_d)

    print("nearest traj:")
    printTrWithStats(target_d.getNearestTr(tr))
    # END INIT TARGET


    # quick visualization
    if(not RUN_ALL):
        for t in tr:
            t.drawToSpace()
        showInNamed("target", target_d.visual_avg.astype(np.uint8))
        showInNamed("features", frame_features)
        dspace.showSpace(draw_dets=DRAW_DETS)


    # EXTEND
    skip_n = 0
    # skip_n = 27
    # skip_n = 40
    # skip_n = 75
    # skip_n = 80
    # skip_n = 119
    # skip_n = 200
    # skip_n = 285
    # skip_n = 298
    # skip_n = 316
    # skip_n = 340
    # skip_n = 555
    # skip_n = 650
    # skip_n = 2009
    # skip_n = 1108
    # skip_n = 4110 # all
    
    # for visualization (feature maps)
    showing_prev = set() # set showing windows
    showing_next = set()
    # ------------------
    try:
        sslm = 0 # steps since last merge
        print("[EXTEND]")
        # main extend loop
        for i in range(n_res):
            # print("[@"+str(t_global)+"]")
            print("[@ %d / %d (%d%%)]"%(t_global, n_all, int((float(t_global)/float(n_all))*100) ))
            
            # 1. init new frame (also compute flow, if needed, same for visual features):
            next_dets=D13[t_global]
            # print("next dets: ",next_dets)
            dspace.D.append(next_dets)
            frame = setFrameFlowAtI(dspace,t_global, of, frames_path=FRAMES_PATH)

            # get dino features for current frame
            frame_features = computeOrGetPCAFeaturesAtI(t_global, frame, featureExt, save_feat=SAVE_FEATURES)

            # add features to detections:
            for d_i in next_dets:
                # get patches: getPatchesInBB + roiPool (in helper func)
                patches = getFeaturesFromFeatureMapAndPadding3(d_i, feature_map_and_padding=(frame_features, frame_feat_padding))
                d_i.visual_feat = patches
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
                    used_dets_current, forked_tr = t.extend4(feature_map_and_padding=(frame_features, frame_feat_padding)) # need used dets to start new trajectories from unused ones
                    tr_next_from_same = tr_next_from_same + forked_tr
                    
                    # 2.4. update set of used dets (obtained from extend method)
                    used_dets.update(used_dets_current)
                else:
                    # store terminated trajectories in separate list
                    # if last point is too far away from current time, do not include it
                    # if(t.X[-1].t >= (t_global-20)):
                    t.term = True # terminate anyway, ...
                    if(t.X[-1].t > (t_global-stage_II_III_timew)):
                        tr_fin.append(t) # ... add to fin only if they are not too far away from current time (to make space for new trajectories)
                
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
            print("[starting new]")
            # 3. start new trajectories from unused dets
            unused_dets = set(next_dets) - used_dets # difference of sets
            new_tr = []
            for d in unused_dets:
                t_new = Trajectory(d, dspace)
                t_new.build2()
                # also get features (roiPooled)
                # patches = getFeaturesFromFeatureMapAndPadding(d, feature_map_and_featExt=(frame_features, featureExt))
                patches = getFeaturesFromFeatureMapAndPadding3(d, feature_map_and_padding=(frame_features, frame_feat_padding))
                addToAvgTr(patches, t_new)
                new_tr.append(t_new)
            
            # (if there are no more that say, 20 trs already)
            # if(len(new_tr) + len(tr) < 35):
            if(len(new_tr) + len(tr) < 20):
                tr = tr+new_tr
            # tr = tr+new_tr
            # ---


            # 4. update target:
            print("[UPDATING TARGET]")
            if(target_d.tr.term):
                # we need to select new target, because this one is fin
                print("target lost, searching for new based on visual similarity:")
                # first: look in idtrmap:
                ttnsmuid = target_d.tr.not_so_much_unique_id # ttnsmuid: target tr not so much unique id
                # this is best case scenario
                select_new_target = True
                if(ttnsmuid in idTr_map):
                    new_target_tr = idTr_map[ttnsmuid]
                    if(not new_target_tr.term):
                        print("changed previous trajectory with new with same nsmuid")
                        target_d.tr = idTr_map[ttnsmuid]
                        select_new_target = False
                    else:
                        print("trajectory with same nsmuid exists, but is fin. Need to find another")
                if(select_new_target):
                    print("selecting new target, because old one is fin")
                    # get all that are not fin:
                    not_fin: list[Trajectory] = []
                    for t_i in tr:
                        if(not t_i.term):
                            not_fin.append(t_i)
                    if(not_fin):
                        target_d.tr = target_d.getNearestTrVisual(not_fin)
                    else:
                        print("no next trajectories found, keeping last one as is")
            target_d.setLastTDetProps()
            
            if(skip_n <= 0 and not RUN_ALL):
                showInNamed("target", target_d.visual_avg.astype(np.uint8))
            printTrWithStats(target_d.tr)
            print("[END UPDATING TARGET]")
            # --------------



            # 5 pruning step: discard trajectories, that are too far away from target (testing: target is gt)
            print("[pruning]")
            n_prune = 5 # frames
            max_dist = 150
            tr = getPruned(tr, [target_d], n_prune, max_dist)
            
            # -----------------
            
            # 6. hypothesis selection:
            
            # 6.1 analyze trs with Q matrix: get pure, drop redundant
            pure_trs, not_pure = analyzeTrsWithQ(tr) # analyze, drop out pure and copies of same (add pure, do qbp with all else)
            
            # 6.2 build Q with trs, that are not pure:
            not_pure.sort(key=lambda x: x.getScore2(), reverse=True) # sort to minimize chance of getting stuck in some local minimum (there ARE issues with qbp-solver)
            Q = buildQBPMatrixX(not_pure, type=1)

            
            # print("pure:")
            # printTrListWithStatsOrdered(pure_trs)
            # print("not pure:")
            # printTrListWithStatsOrdered(not_pure)

            # 6.3 solve QBP
            # np.savetxt("Q2.txt",Q, fmt="%7.3f")
            print(Q)
            print("solving Q ...")
            v = solveQBP2(Q)
            print("done solving, v:")
            print(v)


            # 6.4 keep only selected trajectories for next frame (all else are term):
            tr_next2: list[Trajectory] = []
            # print("selected: ", end="")
            for ii in range(len(v[0])):
                tr_i = not_pure[ii]
                if(v[0][ii] == 1):
                    # tr_i = tr[ii]
                    tr_next2.append(tr_i)
                    # print("t"+str(tr_i.id),end=" ", flush=True)
                else:
                    tr_i.term = True # terminate ones that are not selected, because they will not make it in next iteration thus won't be updated
            tr = tr_next2 + pure_trs
            # print()
            # ---

            # 6.5 visualization
            if(not RUN_ALL):
                showing_next = set()
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
                    t.drawToSpace(at_t=t_global)
                    showing_next.add(t)
                    t_i = t_i+1
            # ---
            
            # 7. on every n_stage_II_III-th frame, merge trajectories collected so far:
            # order=[2,1]
            order=[1,2]
            sslm = sslm+1
            # print("[!]",sslm)
            if(n_stage_II_III == sslm or len(tr_fin) >= 25):
            # if(n_stage_II_III == sslm):
                if(not RUN_ALL):
                    print("[MERGE TRAJECTORIES]")
                merged_merged:list[Trajectory] = []
                tr_all = tr_fin+tr

                print("tr_all (%d):"%(len(tr_all)))
                printTrListWithStatsOrdered(tr_all)
                
                print("idtr (%d):"%(len(idTr_map.keys())))
                printTrListWithStatsOrdered(idTr_map.values())

                # DEBUG: SAVE TRAJECTORIES, THEN RETURN
                # with open(RESULT_PATH+'trsdeb2.p', 'wb') as fp: # fp: file pointer?
                #     abc = [T_OFFSET, t_global, copy.deepcopy(tr_all)] # int, int, list[Trajectory]
                #     pickle.dump(abc, fp)
                # # save Q:
                # # np.savetxt("Qmat.txt",Q, fmt="%7.3f")
                # return

                # 7.1. stage II (bridged) (is faster)
                print("get merged 1:")
                merged_trs1 = getMergedHypotheses(tr_all, type=order[0], time_window=stage_II_III_timew, also_overlapping=True)
                print("done (1)")
                

                # 7.2. get unused trajectories
                tr_left = set(tr_all)
                t_i = 0
                for t in merged_trs1:
                    key_set = set(t.T2.keys())
                    if(len(key_set) > 1):
                        tr_left = tr_left - key_set
                    t_i = t_i+1
                tr_left = list(tr_left)

                # 7.3. stage III (flow) with unused trajectories (is slower)
                print("get merged 2")
                merged_trs2 = getMergedHypotheses(tr_left, type=order[1], time_window=stage_II_III_timew, also_overlapping=True)
                print("done (2)")

                # set all previous to fin, as they are replaced by new ones:
                for t in tr:
                    t.term = True
                
                
                # 7.4. select best trajectories to be used in new round
                merged_merged = merged_trs1+merged_trs2

                # also prune them before qbp (but only those that are fin):
                # prune ones that are too far before building Q:
                # get only those that are not fin
                print("before pruning")
                to_prune: list[Trajectory] = []
                all_else: list[Trajectory] = []
                for tr in merged_merged:
                    if(len(tr.X) > 1): # also keep only ones that are longer than 1
                        if(tr.term):
                            all_else.append(tr)
                        else:
                            to_prune.append(tr)
                print("[pruning merged]")
                gt_at_ti = GT13[t_global]
                max_dist = 150
                tr = getPruned(to_prune, gt_at_ti, -1, max_dist) # -1? to prune without length restriction
                merged_merged = to_prune + all_else


                merged_merged = dropRedundant(merged_merged, red_level=2) # redundancy level: 2 ('strict' mode: preserve only one end point and number of merged)
                # merged_merged = [t for t in merged_merged if len(t.T2.keys()) > 1] # do not include single 

                # merged_merged.sort(key=lambda x: x.getScoreII(), reverse=True) # sort to minimize chance of getting stuck in some local minimum (there ARE issues with qbp-solver)
                # # some idea to try and modify matrix to better solve it with solver
                # n_single =0
                # single_score2_max = 0.000000000000001
                # single_indices=[]
                # another_i = 0
                # for t in merged_merged:
                #     if(len(t.T2.keys()) == 1):
                #         n_single = n_single+1
                #         single_indices.append(another_i)
                #         t_score = t.getScore2()
                #         if(t_score > single_score2_max):
                #             single_score2_max = t_score
                #     another_i = another_i+1

                # print("st trajektorij: ",len(merged_merged), "st single: ", n_single)
                
                # # merged_merged = merged_trs2
                # print("building Q")
                # Q = dspace.buildQBPMatrixX(merged_merged, type=2)
                # # np.savetxt("Q.txt",Q, fmt="%7.3f")
                # print(Q)

                # if(n_single > 15):
                #     for another_i in single_indices:
                #         t = merged_merged[another_i]
                #         t_current_s = Q[another_i,another_i]
                #         Q[another_i,another_i] = (((t.getScore2()/single_score2_max)*T_INIT_S) + t_current_s)/2

                # q-matrix analysis (type 2 this time)
                simple_tr,all_else = analyzeTrsWithQ(merged_merged, type=2)
                all_else.sort(key=lambda x: x.getScoreII(), reverse=True) # sort to minimize chance of getting stuck in some local minimum
                Q = buildQBPMatrixX(all_else, type=2) # build q with all trs that have not been filtered by analysis

                # SOLVE QBP:
                # print(Q)
                print("solving (merge) ...")
                res = solveQBP2(Q)
                v = res[0]
                # res = solveQBP(Q)
                print("done solving, v:")
                print(res)
                
                # SHOW SELECTED:
                selected_merged:list[Trajectory] = simple_tr # init it with those, that do not need to be selected by qbp
                # dspace.clearSpace()
                # iter over all_else and not over all trs
                for t_i in range(len(v)):
                    if(v[t_i] == 1):
                        merged_at_i = all_else[t_i]
                        if(not (merged_at_i.term and len(merged_at_i.T2.keys()) == 1)):
                            selected_merged.append(merged_at_i)
                    else:
                        all_else[t_i].term = True # terminate ones that are not selected, because they will not make it in next iteration thus won't be updated
                        # printTrWithStats(all_else[t_i], add_to_end=" [x] ")
                
                # add selected merged to idTr_map:
                print("adding selected merged to idTr_map")
                for t_i in selected_merged:
                    idTr_map[t_i.not_so_much_unique_id] = t_i # update trajectories holding/representing that id
                print("added")
                
                # update idtrmap fin/not fin:
                updateFinInIdTR(idTr_map, t_global)
                # print()

                if(not RUN_ALL):
                    print("selected merged: ")
                    printTrListWithStatsOrdered(selected_merged)
                    
                    print("idTr_map:")
                    printTrListWithStatsOrdered(idTr_map.values())

                # use new trajectories in next loop cycle
                # tr = tr+selected_merged # ?? why add? how 'bout id switch?

                tr = selected_merged
                tr_fin = []


                # print("------------------- --------------------------------------------------------------------------------")
                if(skip_n <= 0 and not RUN_ALL):
                    d2_winname = "merged"
                    cv.namedWindow(d2_winname, cv.WINDOW_NORMAL)
                    # for t in selected_merged:
                    #     t.drawToSpace()
                    #     print("t"+str(t.id), end=" ", flush=True)
                    #     dspace.showSpace(draw_dets=DRAW_DETS, draw_last_dets_bb=False, dspace_winname=d2_winname)
                    #     dspace.clearSpace()
                    # print("merged_merged:")
                    # for t in merged_merged:
                    #     t.drawToSpace()
                    #     print("t"+str(t.id), end=" ", flush=True)
                    #     printTrWithStats(t)
                    #     dspace.showSpace(draw_dets=DRAW_DETS, draw_last_dets_bb=False, dspace_winname=d2_winname)
                    #     dspace.clearSpace()
                    # print("selected_merged:")
                    dspace.clearSpace()
                    for t in selected_merged:
                        t.drawToSpace()
                    print()
                    dspace.showSpace(draw_dets=DRAW_DETS, draw_last_dets_bb=False, dspace_winname=d2_winname)
                    # dspace.clearSpace()
                sslm = 0

                # DEBUG: DROP REDUNDANT
                # if(t_global == 1111):
                #     # save merged, return
                #     with open(RESULT_PATH+'trs1111.p', 'wb') as fp: # fp: file pointer?
                #         # abc = [T_OFFSET, t_global, copy.deepcopy(idTr_map), copy.deepcopy(merged_merged)]
                #         abc = [T_OFFSET, t_global, copy.deepcopy(idTr_map), copy.deepcopy(merged_merged)]
                #         pickle.dump(abc, fp)
                #     return

                # DEBUG: SAVE ONES TO MERGE, SAVE MERGED, SAVE Q
                # save merged, return
                # with open(RESULT_PATH+'trsdeb.p', 'wb') as fp: # fp: file pointer?
                #     # abc = [T_OFFSET, t_global, copy.deepcopy(idTr_map), copy.deepcopy(merged_merged)]
                #     abc = [T_OFFSET, t_global, copy.deepcopy(tr_all), copy.deepcopy(merged_merged)] # int, int, list[Trajectory], list[Trajectory]
                #     pickle.dump(abc, fp)
                # # save Q:
                # np.savetxt("Qmat.txt",Q, fmt="%7.3f")
                # return
            # ---

            # some loop-related technical stuff
            # if skip_n is set (non zero), algorithm runs for skip_n frames without visualization
            if(not RUN_ALL):
                if(skip_n <= 0):
                    showing_diff = showing_prev - showing_next
                    if(False):
                        for t in showing_next:
                            print("showing: %d"%(t.not_so_much_unique_id))
                            showInNamed("%d"%(t.not_so_much_unique_id), t.visual_avg.astype(np.uint8))
                    showInNamed("features", frame_features)
                    
                    for t in showing_diff:
                        cv.destroyWindow("%d"%(t.not_so_much_unique_id))
                    showing_prev = showing_next

                    # draw gt bb, draw target bb:
                    # if(gt_at_ti):
                        # drawBoundingBox(dspace.map, gt_at_ti[0].bb, [0,0,255])
                    
                    ext_bb_off = 1
                    ext_bb = list(np.array(target_d.bb[0:2])+ext_bb_off)+list(np.array(target_d.bb[2:])-(ext_bb_off*2))
                    # ext_bb = bbResize(target_d.bb, ext_bb_off=ext_bb_off)
                    drawBoundingBox(dspace.map, ext_bb, [100,0,255])

                    dspace.showSpace(draw_dets=DRAW_DETS)
                    dspace.clearSpace()
                else:
                    skip_n = skip_n-1
            
            # increase time on each iteration (most important detail)
            t_global = t_global + 1
    except KeyboardInterrupt:
        # save tr, tr_fin and T_OFFSET
        print("fin")
    print("[END EXTEND]")
    # END EXTEND

    print("[FINAL PATH CREATION]")
    SAVE_PATH_TR = True
    path_creation_type=1
    path: list[Trajectory] = []
    print("path method: %d"%path_creation_type)
    if(path_creation_type == 1):
        trajectories: list[Trajectory] = list(idTr_map.values()) # final trajectories
        if(trajectories):
            t0: Trajectory = selectBestTrAroundDetAtTime(trajectories, GT13[T_OFFSET][0].bb)
        
            next_trs = extendToEnd2(t0, trajectories, also_drop_redundant=True)
            path=next_trs
            print("final best trs:")
            printTrListWithStatsOrdered(next_trs)
            # all_trs = [next_trs[1]]
            # all_trs = [tr1]
            next_tr = getBestFromList(next_trs)
            print("final best:")
            printTrWithStats(next_tr)
            
            # write trajectory to file
            path2File2([next_tr], n_all, RESULT_PATH)

    # 2. group them 'by hand':
    if(path_creation_type == 2):
        trajectories: list[Trajectory] = list(idTr_map.values())
        if(trajectories):
            t0 = selectBestTrAroundDetAtTime(trajectories, GT13[T_OFFSET][0].bb)
            path = [t0]+connectTrs(t0, trajectories)
            print(path)
            # end 2.

            # write trajectory to file
            # path2File(path, n_all, RESULT_PATH)
            path2File2(path, n_all, RESULT_PATH)
    if(SAVE_PATH_TR):
        with open(RESULT_PATH+'path.p', 'wb') as fp:
            abc = [T_OFFSET, n_all, copy.deepcopy(path)]
            pickle.dump(abc, fp)

    if(SAVE_TRAJECTORIES):
        print("[END] saving trs stored in idtr_map")
        with open(RESULT_PATH+'trsa.p', 'wb') as fp:
            # abc = [T_OFFSET, copy.deepcopy(idTr_map)]
            abc = [T_OFFSET, n_all, copy.deepcopy(idTr_map)]
            pickle.dump(abc, fp)
    if(not USE_PRECOMPUTED_FLOW):
        deletePrecomputed(COMPUTED_FLOW, confirm=ASK_BEFORE_FLOW_DELETE)
    if(not USE_PRECOMPUTED_FEAT):
        deletePrecomputed(COMPUTED_FEAT, confirm=ASK_BEFORE_FLOW_DELETE)
    print("[END] results saved.")
    
    # end main

# debug mains:
def main_d1():
    # optional
    global PRECOMPUTED_FEAT_PATH
    global PRECOMPUTED_FLOW_PATH
    
    # set default to tracker ws (./tracker/...)
    global COMPUTED_FLOW
    global COMPUTED_FEAT
    global RESULT_PATH
    global USE_PRECOMPUTED
    
    global SAVE_TRAJECTORIES
    RUN_ALL = False

    # SEQUENCES:
    # T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getSequenceConsts('bird2') # lasot bird2
    T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getSequenceConsts('got10k14') # lasot bird2
    if(USE_PRECOMPUTED):
        COMPUTED_FLOW = PRECOMPUTED_FLOW_PATH
        COMPUTED_FEAT = PRECOMPUTED_FEAT_PATH

    # INIT
    D13: list[list[Detection]] = readDetFile2(DETS_FILE) # read detections from file
    D13 = [[]]+D13 # offset it (because time starts at 1, there is nothing on 0)
    GT13: list[list[Detection]] = [[]]+readDetFile2(GT_PATH,T_OFFSET) # ground truth: for testing purposes only
    t_global = T_OFFSET # current time (frame)
    to_destroy_w = [] # windows to destroy
    # print(t_global, D13[T_OFFSET: T_OFFSET+10])

    # actual offset for detections list is: T_OFFSET - n (because there are some detections before global offset (initial trajectories))
    n_res = len(D13) - T_OFFSET
    n_all = n_res
    tr: list[Trajectory] = []
    tr_fin: list[Trajectory] = []
    
    # INIT DSPACE
    frame = getFrameAtI(t_global,FRAMES_PATH)
    h,w,_ = np.shape(frame)
    dspace = DetectionSpace(h,w, time_offset=T_OFFSET, show_flow=False, disable_vis=RUN_ALL) # also pass time offset for using correct indices | no visualization on run all

    # set last frame
    dspace.lastFrame = frame.copy()
    dspace.map = frame

    frame_features = computeOrGetPCAFeaturesAtI(t_global, frame, None, save_feat=False) # get features, init trs
    frame_feat_padding: tuple[int, int, int, int] = getImagePadding(np.shape(frame))

    # set flow map
    # flow = getFlowAtI(t_global, FLOWS_PATH)
    flow = computeOrGetFlowAtI(t_global, None)
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

        patches = getFeaturesFromFeatureMapAndPadding3(d, (frame_features, frame_feat_padding))
        addToAvgTr(patches, t)
        winname = "det_r%d"%(d.id)
        showInNamed(winname, patches.astype(np.uint8))
        to_destroy_w.append(winname)

        showInNamed("%d"%(t.not_so_much_unique_id), t.visual_avg.astype(np.uint8))
        

        tr.append(t)
        # idTr_map[t.not_so_much_unique_id] = t # add it to map
    
    print("init trs:")
    print(tr)
    # printTrListWithStatsOrdered(tr)
    # END INIT TRAJECTORIES

    

    # quick visualization
    if(not RUN_ALL):
        for t in tr:
            t.drawToSpace()
        showInNamed("features", frame_features)
        dspace.showSpace(draw_dets=DRAW_DETS)
    

    for i in range(n_res):
        print("[@"+str(t_global)+"]")
        
        # 1. init new frame (also compute flow, if needed, same for visual features):
        next_dets=D13[t_global]
        print("this frame's dets:")
        for di in next_dets:
            print(di)
        # print("next dets: ",next_dets)
        dspace.D.append(next_dets)
        frame = setFrameFlowAtI(dspace,t_global, None, frames_path=FRAMES_PATH)

        # get dino features for current frame
        frame_features = computeOrGetPCAFeaturesAtI(t_global, frame, None, save_feat=False)
        # frame_feat_padding: tuple[int, int, int, int] = getImagePadding(np.shape(frame)) # not needed, suppose that frames are of same size throuought the sequence
        
        # compute (extract) visual features for every detection
        
        for d_i in next_dets:
            # get patches: getPatchesInBB + roiPool (in helper func)
            patches = getFeaturesFromFeatureMapAndPadding3(d_i, feature_map_and_padding=(frame_features, frame_feat_padding))
            d_i.visual_feat = patches
            
            # patches = getFeaturesFromFeatureMapAndPadding3(d_i, feature_map_and_padding=(frame_features, frame_feat_padding), pooled=False)
            # patches_rp = roiPool(patches, use_2d=False)
            # patches2 = patches_rp.astype(np.float64)
            # d_i.visual_feat = patches2
            # winname = "det_r%d"%(d_i.id)
            # showInNamed(winname, patches.astype(np.uint8))

            # ip = ImagePadder(np.shape(patches)[0:2], patch_size=3)
            # mat_p = ip.im_pad(patches,mode='edge')
            # winname = "det_rp%d"%(d_i.id)
            # showInNamed(winname, mat_p.astype(np.uint8))

            # to_destroy_w.append(winname)
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
                # print("current tr: t%d (t%d)"%(t.id, t.not_so_much_unique_id))
                print("current tr: ", end="", flush=True)
                printTrWithStats(t)
                # 2.1. add current (note: this one gets extended)
                tr_next_from_same.append(t)
                
                # 2.2. also include current trajectory (not extended), add it to hypothesis selection
                t_prev = t.getCopy(deep=False)
                t_prev.term = True # terminate it, so it does not extend
                tr_next_from_same.append(t_prev)

                # 2.3. extend current, also add forks, if they occur
                used_dets_current, forked_tr = t.extend4(feature_map_and_padding=(frame_features, frame_feat_padding)) # need used dets to start new trajectories from unused ones
                tr_next_from_same = tr_next_from_same + forked_tr
                
                # 2.4. update set of used dets (obtained from extend method)
                used_dets.update(used_dets_current)
            else:
                # store terminated trajectories in separate list
                t.term = True # terminate anyway, ...
                tr_fin.append(t) # ... add to fin
            
            # 2.5. mark next from same (current + all forks) as exited, if enter exit zone
            for tr_nxt in tr_next_from_same:
                if(dspace.isInExitZone(tr_nxt.X[-1])):
                    tr_nxt.exited = True
                    tr_nxt.term = True

            # 2.6. update next
            tr_next = tr_next + tr_next_from_same

            # if(not RUN_ALL):
            #     # some debug prints
            #     first = True
            #     for tt in tr_next_from_same:
            #         if(first):
            #             first = False
            #             # print(">>> t"+str(tt.id)+""+" - origin:",tt.origin,"score:",tt.getScore2(), "color:",tt.color, tt.X[-2:], "len:",len(tt.X))
            #             # print(">>> t%d (%d) - origin:%s score:%6.2f color:%s len:%d"%(tt.id,tt.not_so_much_unique_id,str(tt.origin),tt.getScore2(), str(tt.color),str(tt.X[-2:]),len(tt.X)))
            #             print(">>> t%d (%d) - origin:%s score:%6.2f color:%s %s len:%d"%(tt.id,tt.not_so_much_unique_id,str(tt.origin),tt.getScore2(), str(tt.color),str(tt.X[-2:]),len(tt.X)))
            #         else:
            #             print("|-> t%d (%d) - origin:%s score:%6.2f color:%s %s len:%d"%(tt.id,tt.not_so_much_unique_id,str(tt.origin),tt.getScore2(), str(tt.color),str(tt.X[-2:]),len(tt.X)))
            # ---
        if(not RUN_ALL):
            print("[DONE EXTENDING]")
        tr = tr_next

        # hypothesis selection:
        pure_trs, not_pure = analyzeTrsWithQ(tr) # analyze, drop out pure and copies of same (add pure, do qbp with all else)
        
        # 6.2 build Q with trs, that are not pure:
        not_pure.sort(key=lambda x: x.getScore2(), reverse=True) # sort to minimize chance of getting stuck in some local minimum (there ARE issues with qbp-solver)
        Q = buildQBPMatrixX(not_pure, type=1)

        # print(Q)
        # print("solving Q ...")
        v = solveQBP2(Q)
        print("done solving, v: ", end="")
        print(v)

        tr_next2: list[Trajectory] = []
        for ii in range(len(v[0])):
            tr_i = not_pure[ii]
            if(v[0][ii] == 1):
                # tr_i = tr[ii]
                tr_next2.append(tr_i)
                # print("t"+str(tr_i.id),end=" ", flush=True)
            else:
                tr_i.term = True # terminate ones that are not selected, because they will not make it in next iteration thus won't be updated
        tr = tr_next2 + pure_trs


        # to_destroy_w = []
        if(not RUN_ALL):
            for t in tr:
                t.drawToSpace()
                print("showing: t%d"%(t.not_so_much_unique_id))
                showInNamed("%d"%(t.not_so_much_unique_id), t.visual_avg.astype(np.uint8))
            showInNamed("features", frame_features.astype(np.uint8))

            for d in next_dets:
                winname = "det%d"%(d.id)
                showInNamed(winname, d.visual_feat.astype(np.uint8))
                to_destroy_w.append(winname)
            

            # draw gt bb, draw target bb:
            if(GT13[t_global]):
                drawBoundingBox(dspace.map, GT13[t_global][0].bb, [0,0,255])

            dspace.showSpace(draw_dets=DRAW_DETS)
            dspace.clearSpace()

            for w in to_destroy_w:
                cv.destroyWindow(w)
            to_destroy_w = []
        
        
        
        t_global = t_global+1


        # ---
        # end main_d1

def visualizeTrs():
    global PRECOMPUTED_FEAT_PATH
    global PRECOMPUTED_FLOW_PATH
    
    # set default to tracker ws (./tracker/...)
    global COMPUTED_FLOW
    global COMPUTED_FEAT
    global RESULT_PATH
    global USE_PRECOMPUTED
    # SEQUENCES:
    T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getSequenceConsts('bird2') # lasot bird2
    # T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getSequenceConsts('got10k14') # lasot bird2
    if(USE_PRECOMPUTED):
        COMPUTED_FLOW = PRECOMPUTED_FLOW_PATH
        COMPUTED_FEAT = PRECOMPUTED_FEAT_PATH

    # INIT
    D13: list[list[Detection]] = readDetFile2(DETS_FILE) # read detections from file
    D13 = [[]]+D13 # offset it (because time starts at 1, there is nothing on 0)
    GT13: list[list[Detection]] = [[]]+readDetFile2(GT_PATH,T_OFFSET) # ground truth: for testing purposes only
    t_global = T_OFFSET # current time (frame)
    to_destroy_w = [] # windows to destroy
    # print(t_global, D13[T_OFFSET: T_OFFSET+10])

    # actual offset for detections list is: T_OFFSET - n (because there are some detections before global offset (initial trajectories))
    n_res = len(D13) - T_OFFSET
    n_all = n_res
    
    
    # INIT DSPACE
    frame = getFrameAtI(t_global,FRAMES_PATH)
    h,w,_ = np.shape(frame)
    dspace = DetectionSpace(h,w, time_offset=T_OFFSET, show_flow=False, disable_vis=False) # also pass time offset for using correct indices | no visualization on run all

    # set last frame
    dspace.lastFrame = frame.copy()
    dspace.map = frame

    print("init dspace: ")
    D13_init = D13[t_global] # init with those from current time step
    # dspace.D.append(D13_init)
    dspace.D.append(D13[:])
    t_global = t_global+1
    n_res = n_res-1 # there is one frame less - [FIX]
    # END INIT DSPACE


    # INIT TRAJECTORIES
    trs: list[Trajectory]
    off, n_all, trs = read_tr_file('trsa.p')
    off, n_all, path = read_tr_file('trs.p')
    print("path: ",path)

    all_trs = trs.values()
    for t in all_trs:
        t.detectionSpace = dspace # so we can draw it in it (set valid pointer)
    for t in path:
        t.detectionSpace = dspace # so we can draw it in it (set valid pointer)
    
    # all_trs = dropRedundant(list(all_trs),red_level=2)
    # print(all_trs)

    t_global = T_OFFSET
    t_global = 300
    # t_global = 1200

    while t_global <= n_all:
        print("[@"+str(t_global)+"]")
        
        # 1. init new frame (also compute flow, if needed, same for visual features):
        next_dets=D13[t_global]
        print("this frame's dets:")
        # print("next dets: ",next_dets)
        dspace.D.append(next_dets)
        frame = setFrameFlowAtI(dspace,t_global, None, frames_path=FRAMES_PATH)

        # for t in all_trs:
        #     first_t = t.X[0].t
        #     if(first_t <= t_global):
        #         color = [150,150,150] # gray
        #         if(t.X[-1].t >= t_global):
        #             color = t.color
        #             # if(t in path):
        #             #     color = [0,0,255]
        #             if((t_global-first_t) < len(t.X)):
        #                 det = t.X[t_global - first_t]
        #                 drawBoundingBox(dspace.map, bbResize(det.bb, 1), t.color)
        #                 drawX(dspace.map, det.x)
        #             else:
        #                 print("[WARN] t_global is %d, %s"%(t_global, str(t.X[-5:])))
        #         t.drawToSpace(color)
        for t in path:
            first_t = t.X[0].t
            if(first_t <= t_global):
                color = [150,150,255] # gray
                if(t.X[-1].t >= t_global):
                    color = [0,0,255]
                    # if(t in path):
                    #     color = [0,0,255]
                    if((t_global-first_t) < len(t.X)):
                        det = t.X[t_global - first_t]
                        drawBoundingBox(dspace.map, bbResize(det.bb, 3),color)
                        drawX(dspace.map, det.x)
                    else:
                        print("[WARN] t_global is %d, %s"%(t_global, str(t.X[-5:])))
                t.drawToSpace(color)
        # draw gt:
        gtdet = GT13[t_global]
        # print(gtdet)
        if(gtdet):
            drawBoundingBox(dspace.map, bbResize(gtdet[0].bb,-1), [0,0,255])
        dspace.showSpace(draw_dets=False, draw_last_dets_bb=True)
        dspace.clearSpace()




        t_global = t_global + 1



    # printTrListWithStatsOrdered(trs.values())
    # merged = getMergedHypotheses(trs.values(), time_window=100)
    # printTrListWithStatsOrdered(merged)
    # simple, all_else = analyzeTrsWithQ(merged)
    # print("simple:")
    # printTrListWithStatsOrdered(simple)
    # print("not simple:")
    # printTrListWithStatsOrdered(all_else)

    # Q = buildQBPMatrixX(all_else, 2)
    # v = solveQBP2(Q)
    # print(v)

    # selected_trs = getSelected(v[0], all_else)
    # printTrListWithStatsOrdered(selected_trs)

def getAtIndices(tr_list: list[Trajectory], indices:list[int]):
    vrni =[]
    for i in range(len(tr_list)):
        if(i in indices):
            vrni.append(tr_list[i])
    return vrni

def extendToEnd(tr_: Trajectory, tr_list: list[Trajectory], time_window=100):
    print("this is extendToEnd")
    # 1. extend given tr as much as possible
    all_possible_next:list[Trajectory] = getMergedHypothesis(tr_,tr_list=tr_list, time_window=time_window, type=1, also_overlapping=True, reset_memo=True) # 2. extend all possible from this one

    # remove redundant:
    next_wo_redundant:list[Trajectory] = dropRedundant(all_possible_next, red_level=2)
    print("all possible next of t%d"%tr_.not_so_much_unique_id)
    printTrListWithStatsOrdered(next_wo_redundant)

    # 3. if it is not at the end already, then find visually best matches to continue:
    next_n_trs = 30
    n_steps = 5 # maximum number of restarts
    next_trs:list[Trajectory] = []
    for tr in next_wo_redundant:
        # 1.1 ending time (search only among those that are from then on)
        print("continuing trajectory:")
        printTrWithStats(tr)
        tr_end_time = tr.X[-1].t
        print("possible next based on time:")
        possible_next = []
        i = 0
        max_next_visual_score = 0
        next_tr_with_max_score = None
        for a in tr_list:
            if a.X[0].t > tr_end_time:
                possible_next.append(a)
                # find visual similarity:
                score = getProbIVBF([0,0,0,0],tr.visual_avg, [0,0,0,0], a.visual_avg, a=0) # consider only visual score (a=0)
                if(max_next_visual_score < score):
                    max_next_visual_score = score
                    next_tr_with_max_score = a
                printTrWithStats(a, i_t=i, add_to_end="%.2f"%score)
                i = i+1
            if(i > next_n_trs):
                break
        print("next tr with max score:")
        if(next_tr_with_max_score is not None):
            printTrWithStats(next_tr_with_max_score)
            # merge it
            time_diff = next_tr_with_max_score.X[0].t - tr.X[-1].t
            ext_tdet = tr.lastNtimesCopy(time_diff-1)
            next_trs.append(mergeTrajectories2([(tr, 0.5,1,[]),(next_tr_with_max_score, max_next_visual_score,1,ext_tdet)]))
        else:
            print("there is no best next")

        # printTrListWithStatsOrdered(possible_next)
        print()
    # 4. drop redundant again
    next_trs = dropRedundant(next_trs)
    # 5. set new target, repeat extend
    printTrListWithStatsOrdered(next_trs)
    # solveqbp
    Q=buildQBPMatrixX(next_trs, type=2)
    print(Q)
    v = solveQBP2(Q)
    next_trs = getSelected(v[0], next_trs)
    print("next trs after qbp")
    printTrListWithStatsOrdered(next_trs)


    next_trs2: list[Trajectory] = []
    for tr in next_trs:
        next_merged = getMergedHypothesis(tr,tr_list=tr_list, time_window=time_window, type=1, also_overlapping=True, reset_memo=False) # 2. extend all possible from this one
        next_trs2 = next_trs2 + next_merged
    next_trs2 = dropRedundant(next_trs2)
    print("next round:")
    printTrListWithStatsOrdered(next_trs2)

    Q=buildQBPMatrixX(next_trs2, type=2)
    print(Q)
    v = solveQBP2(Q)
    next_trs2 = getSelected(v[0], next_trs2)
    print("next trs after qbp")
    printTrListWithStatsOrdered(next_trs2)


    # return next_trs2

def extendToEnd2(tr_: Trajectory, tr_list: list[Trajectory], time_window=100, also_drop_redundant=False):
    print("this is extendToEnd2")
    # 0. INIT LIST
    all_trs: list[Trajectory] = [tr_]
    other_trs: list[Trajectory] = []
    continue_flag = True
    n_steps = 5 # maximum number of restarts
    max_n = 5 # maximum recursion depth of extendAllPossibleNext in getMergedHypothesis
    with_qbp = True

    ii = 0
    while (ii < n_steps and continue_flag):
        # 2. extend all from list
        all_possible_next:list[Trajectory] = []
        for tr in all_trs:
            all_possible_next = all_possible_next + getMergedHypothesis(tr,tr_list=tr_list, time_window=time_window, type=1, also_overlapping=True, reset_memo=True, also_drop_redundant=also_drop_redundant,max_n = max_n, with_qbp=with_qbp) # 2. extend all possible from this one
        print("[extend2end2] all possible next (n): %d"%len(all_possible_next))
        # 1st solve qbp to remove unnecessary ones
        Q = buildQBPMatrixX(all_possible_next, type=2)
        v = solveQBP2(Q)
        all_possible_next = getSelected(v[0], all_possible_next)
        all_trs = all_possible_next

        # 3. if it is not at the end already, then find visually best matches to continue:
        next_n_trs = 30
        next_trs:list[Trajectory] = []
        can_any_continue = False # flag to indicate whether trajectories can continue or not (based on time; if no new trajectory could be added, this stays false and extend does not need to continue)
        for tr in all_trs:
            # 1.1 ending time (search only among those that are from then on)
            # print("continuing trajectory:")
            # printTrWithStats(tr)
            tr_end_time = tr.X[-1].t
            # print("possible next based on time:")
            possible_next = []
            i = 0
            max_next_visual_score = 0
            next_tr_with_max_score = None
            for a in tr_list:
                if a.X[0].t > tr_end_time:
                    possible_next.append(a)
                    # find visual similarity:
                    score = getProbIVBF([0,0,0,0],tr.visual_avg, [0,0,0,0], a.visual_avg, a=0) # consider only visual score (a=0)
                    if(max_next_visual_score < score):
                        max_next_visual_score = score
                        next_tr_with_max_score = a
                    # printTrWithStats(a, i_t=i, add_to_end="%.2f"%score)
                    can_any_continue = True
                    i = i+1
                if(i > next_n_trs):
                    break
            print("next tr with max score:")
            if(next_tr_with_max_score is not None):
                printTrWithStats(next_tr_with_max_score)
                # merge it
                time_diff = next_tr_with_max_score.X[0].t - tr.X[-1].t
                ext_tdet = tr.lastNtimesCopy(time_diff-1)
                next_trs.append(mergeTrajectories2([(tr, 0.5,1,[]),(next_tr_with_max_score, max_next_visual_score,1,ext_tdet)]))
            else:
                print("there is no best next")

            # printTrListWithStatsOrdered(possible_next)
            print()
        if(can_any_continue):
            all_trs = next_trs

        # 2nd qbp
        Q=buildQBPMatrixX(all_trs, type=2)
        v = solveQBP2(Q)
        all_trs = getSelected(v[0], all_trs)
        ii = ii+1
        continue_flag = can_any_continue # if there are hypotheses, that can be extended, then continue, otherwise do not
    return all_trs

# function gets best trajectory based on score (higher is better) and number of merged (lower is better)
def getBestFromList(tr_list: list[Trajectory]):
    def scoreNmergedComparator(tr1: Trajectory, tr2: Trajectory):
        s1 = tr1.getScore2()
        s2 = tr2.getScore2()
        if(math.isclose(s1,s2)):
            nMerged1 = len(tr1.T2.keys())
            nMerged2 = len(tr2.T2.keys())
            if(nMerged1 == nMerged2): return 0
            elif(nMerged1 < nMerged2): return 1
            else: return -1
        elif(s1 > s2): return 1
        else: return -1
    
    # bestTr: Trajectory = tr_list[0]
    bestTr: Trajectory = None
    for t in tr_list:
        if(bestTr is None): bestTr = t # first one
        if(t == bestTr): continue # get past first one
        if(scoreNmergedComparator(t, bestTr) == 1):
            bestTr = t
    return bestTr


def visualizeTrs2():
    global OPTIONS
    global PRECOMPUTED_FEAT_PATH
    global PRECOMPUTED_FLOW_PATH
    
    # set default to tracker ws (./tracker/...)
    global COMPUTED_FLOW
    global COMPUTED_FEAT
    global RESULT_PATH
    global USE_PRECOMPUTED
    # SEQUENCES:
    
    sequence_str = OPTIONS['sequence']
    T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getSequenceConsts(sequence_str)
    # T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getSequenceConsts('bird2') # lasot bird2
    # T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getSequenceConsts('got10k14') # got10k14
    # T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getSequenceConsts('coin18') # lasot bird2
    # T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getSequenceConsts('chameleon20') # lasot chameleon20
    if(USE_PRECOMPUTED):
        COMPUTED_FLOW = PRECOMPUTED_FLOW_PATH
        COMPUTED_FEAT = PRECOMPUTED_FEAT_PATH

    # INIT
    D13: list[list[Detection]] = readDetFile2(DETS_FILE) # read detections from file
    D13 = [[]]+D13 # offset it (because time starts at 1, there is nothing on 0)
    GT13: list[list[Detection]] = [[]]+readDetFile2(GT_PATH,T_OFFSET) # ground truth: for testing purposes only
    t_global = T_OFFSET # current time (frame)
    to_destroy_w = [] # windows to destroy
    # print(t_global, D13[T_OFFSET: T_OFFSET+10])

    # actual offset for detections list is: T_OFFSET - n (because there are some detections before global offset (initial trajectories))
    n_res = len(D13) - T_OFFSET
    n_all = n_res
    
    
    # INIT DSPACE
    frame = getFrameAtI(t_global,FRAMES_PATH)
    h,w,_ = np.shape(frame)
    dspace = DetectionSpace(h,w, time_offset=T_OFFSET, show_flow=False, disable_vis=False) # also pass time offset for using correct indices | no visualization on run all

    # set last frame
    dspace.lastFrame = frame.copy()
    dspace.map = frame

    # return
    print("init dspace: ")
    D13_init = D13[t_global] # init with those from current time step
    # dspace.D.append(D13_init)
    dspace.D.append(D13[:])
    t_global = t_global+1
    n_res = n_res-1 # there is one frame less - [FIX]
    # END INIT DSPACE


    # INIT TRAJECTORIES
    trs: list[Trajectory]
    off, n_all, trs = read_tr_file('trsa.p')
    off, n_all, path = read_tr_file('path.p')
    # off, n_all, path = read_tr_file('trs.p')
    # print("path: ",path)
    # printTrWithStats(path[0])

    all_trs:list[Trajectory] = list(trs.values())
    # all_trs:list[Trajectory] = trs
    for t in all_trs:
        t.detectionSpace = dspace # so we can draw it in it (set valid pointer)
    
    # all trs:
    print("all trs:")
    printTrListWithStatsOrdered(all_trs)
    # return
    # tr1:Trajectory = all_trs[0] # 1. choose one closest to gt in first frame (to start)
    tr1:Trajectory = all_trs[2] # bird2
    # all_trs=path
    # path[0].detectionSpace=dspace
    # # return
    
    # # merge it
    # next_trs = extendToEnd2(tr1, all_trs, also_drop_redundant=True)
    # print(next_trs)
    # printTrListWithStatsOrdered(next_trs)
    # next_tr = getBestFromList(next_trs)
    # next_trs = [next_tr]
    path[0].detectionSpace = dspace
    next_trs = [path[0]]
    
    
    # all_next = []
    # tr11 = next_trs[1]
    
    # # next_tr11 = tr11.getPossibleNext2(all_trs, time_window=100)
    # # next_tr11 = getPossibleNextWithMemo(tr11,all_trs,time_window=100,type=1)
    # # printTrListWithStatsOrdered([a[0] for a in next_tr11])
    # # print(next_tr11)
    # # printTrListWithStatsOrdered(next_tr11)
    # next_merged = getMergedHypothesis(next_trs[1],tr_list=all_trs, time_window=100, type=1, also_overlapping=True, reset_memo=True, also_drop_redundant=True, max_n =5, with_qbp=True) # 2. extend all possible from this onegetMergedHypothesis(tr1,tr_list=all_trs, time_window=100, type=1, also_overlapping=True, reset_memo=True, also_drop_redundant=True) # 2. extend all possible from this one
    # print("next trs:")
    # printTrListWithStatsOrdered(next_merged)

    # return
    
    # for t in next_trs:
    #     printTrWithStats(t)
    #     # all_next = all_next + getMergedHypothesis(t,tr_list=all_trs, time_window=100, type=1, also_overlapping=True, reset_memo=True, also_drop_redundant=True) # 2. extend all possible from this onegetMergedHypothesis(tr1,tr_list=all_trs, time_window=100, type=1, also_overlapping=True, reset_memo=True, also_drop_redundant=True) # 2. extend all possible from this one
    # printTrListWithStatsOrdered(next_trs)
    
    # return
    # print()
    # # all_trs = [next_trs[1]]
    # # all_trs = [tr1]
    # next_tr = getBestFromList(next_trs)
    # printTrWithStats(next_tr)
    # all_trs = [next_tr]
    
    # t0: Trajectory = selectBestTrAroundDetAtTime(trajectories, GT13[T_OFFSET][0].bb)
    # next_trs = extendToEnd2(t0, trajectories)
    # path=next_trs
    # print("final best trs:")
    # printTrListWithStatsOrdered(next_trs)
    # # all_trs = [next_trs[1]]
    # # all_trs = [tr1]
    # next_tr = getBestFromList(next_trs)
    # print("final best:")
    # printTrWithStats(next_tr)
    
    # # write trajectory to file
    # path2File2([next_tr], n_all, RESULT_PATH)

    
    # best_tr = getBestFromList(path)
    # if(best_tr is not None):
    #     best_tr.detectionSpace = dspace
    #     all_trs = [best_tr]
    # else:
    #     all_trs = [tr1]
    # all_trs = [next_merged[0]]
    # all_trs = next_merged
    all_trs = next_trs
    

    t_global = T_OFFSET
    # t_global = 700
    # t_global = 300
    # t_global = 1200
    # all_trs = refined
    # all_trs = tr_list
    # all_trs = merged_selected
    # all_trs = refined
    # t_global = 3200
    while t_global <= n_all:
        print("[@ %d / %d (%d%%)]"%(t_global, n_all, int((float(t_global)/float(n_all))*100) ))
        
        # 1. init new frame (also compute flow, if needed, same for visual features):
        next_dets=D13[t_global]
        # print("this frame's dets:")
        # print("next dets: ",next_dets)
        dspace.D.append(next_dets)
        frame = setFrameFlowAtI(dspace,t_global, None, frames_path=FRAMES_PATH)
        # print("current trs:")
        for t in all_trs:
            first_t = t.X[0].t
            if(first_t <= t_global):
                color = [150,150,150] # gray
                if(t.X[-1].t >= t_global):
                    # printTrWithStats(t)
                    color = t.color
                    # if(t in path):
                    #     color = [0,0,255]
                    t.drawToSpace(color, at_t=t_global, max_len=20)
                    if((t_global-first_t) < len(t.X)):
                        det = t.X[t_global - first_t]
                        drawBoundingBox(dspace.map, bbResize(det.bb, 1), t.color)
                        drawX(dspace.map, det.x)
                    else:
                        print("[WARN] t_global is %d, %s"%(t_global, str(t.X[-5:])))
                else:
                    t.drawToSpace(color, at_t=t_global, max_len=20)
        
        # draw gt:
        gtdet = GT13[t_global]
        # print(gtdet)
        if(gtdet):
            drawBoundingBox(dspace.map, bbResize(gtdet[0].bb,-1), [0,0,255])
        dspace.showSpace(draw_dets=False, draw_last_dets_bb=False)
        dspace.clearSpace()
        t_global = t_global + 1




# merge overlaping
def mergeOverlaping(tr1: Trajectory, tr2: Trajectory):
    overlap = tr1.X[-1].t - tr2.X[0].t
    vrni = []
    if(overlap > 0):
        print("we have overlap")
        a1, b1 = tr1.X[0].t, tr1.X[-1].t
        a2, b2 = tr2.X[0].t, tr2.X[-1].t
        if(a2 > a1 and a2 < b1 and b2 > b1):
            print("merging the 'long' way")
            print("overlap: %d"%(overlap))

            # fix x:
            print(tr1.X)
            print(tr2.X)

            # 1. find first that overlaps
            i = len(tr1.X)-1
            times_of_overlap = set()
            while(tr1.X[i].t >= tr2.X[0].t):
                times_of_overlap.add(tr1.X[i].t)
                i = i-1
            new_x = tr1.X[0:i+1] + tr2.X
            new_x[i+1].color = [10,10,150]

            # 2. get dets from D2 that have that time
            dets_overlap = set()
            for d in tr1.D2.keys():
                if(d.t in times_of_overlap):
                    dets_overlap.add(d)
            
            # 3. new D2:
            new_d2 = {**tr1.D2, **tr2.D2}
            for d in dets_overlap:
                new_d2[d] = 0.0 # remove score from overlaping
            
            # 3. new T2:
            new_t2 = {**tr1.T2, **tr2.T2}


            print(tr1.X[i])
            print(tr2.X[0])

            # get score of that overlap:
            bb1 = tr1.X[i].bb
            bb2 = tr2.X[0].bb
            vf1 = tr1.visual_avg
            vf2 = tr2.visual_avg
            
            t2_score = getProbIVBF(bb1,vf1,bb2,vf2, 0.9)
            new_t2[tr2] = [t2_score,0.5]

            merged_tr = tr1.getCopy()
            merged_tr.X = new_x
            merged_tr.D2 = new_d2
            merged_tr.T2 = new_t2
            merged_tr.visual_avg, merged_tr.visual_n = add2AvgTr(tr1, tr2)
            vrni.append(merged_tr)

            print("tr1 d2:")
            print(tr1.D2)
            print("tr2 d2:")
            print(tr2.D2)
            print("new d2:")
            # print(new_d2)
            new_d2l = list(new_d2.keys())
            for i in range(len(new_d2)):
                d = new_d2l[i]
                print("%d: %s:%f"%(i,str(d), new_d2[d]))





        else:
            print("invalid case, skipping")
    return vrni

# function gets trajectories that are overlaping
def getOverlapingTrs(tr_list: list[Trajectory], max_overlap=20):
    printTrListWithStatsOrdered(tr_list)
    print("this is getoverlapingtrs")

    for tr in tr_list:
        for tr2 in tr_list:
            if(tr == tr2):
                continue
            a = tr.X[0].t
            b = tr.X[-1].t
            a2 = tr2.X[0].t
            b2 = tr2.X[-1].t


            overlap = b - a2
            if(a2 > a and a2 < b and b2 > b and overlap <= max_overlap):
                print("t%d: %d - %d; t%d: %d - %d, overlap: %d"%(tr.not_so_much_unique_id, a, b, tr2.not_so_much_unique_id, a2, b2, overlap))
                mergeOverlaping(tr, tr2)

            else:
                print("(not valid case) t%d: %d - %d; t%d: %d - %d, overlap: %d"%(tr.not_so_much_unique_id, a, b, tr2.not_so_much_unique_id, a2, b2, overlap))




def main_d2():
    global USE_PRECOMPUTED
    global COMPUTED_FEAT
    global COMPUTED_FLOW

    print("this is main d2")
    T_OFFSET = 1
    D13 = [[]] + readDetFile2('testd2.txt', T_OFFSET)

    # init dspace
    hw = (128,128,3)
    dspace = DetectionSpace(hw[0], hw[1], None, T_OFFSET, show_flow=False)
    frame_feat_padding: tuple[int, int, int, int] = getImagePadding(hw) # get feature image offsets
    # dspace.D.append(D13[T_OFFSET])
    # frame = np.zeros((hw[0], hw[1], 3)).astype(np.uint8)
    
    print("[precomputed] using precomputed flow and path")
    FRAMES_PATH = "/home/gasper/Desktop/synthetic/frames/"
    COMPUTED_FLOW = "%s../synthetic_pr/synthetic_flow/"%FRAMES_PATH
    COMPUTED_FEAT = "%s../synthetic_pr/synthetic_feat/"%FRAMES_PATH

    # frame = getFrameAtI(T_OFFSET, fpath)
    # print(frame)
    # dspace.map = frame
    # dspace.lastFrame = frame.copy()

    # dspace.showSpace(at_time=2)
    # dspace.clearSpace()

    # ---
    t_global = T_OFFSET
    while True:
        frame = setFrameFlowAtI(dspace, t_global, frames_path=FRAMES_PATH)
        features = computeOrGetPCAFeaturesAtI(t_global, frame,None)
        
        # 1. init new frame (also compute flow, if needed, same for visual features):
        next_dets=D13[t_global]
        frame = setFrameFlowAtI(dspace,t_global, None, frames_path=FRAMES_PATH)

        # get dino features for current frame
        frame_features = computeOrGetPCAFeaturesAtI(t_global, frame, None)

        # add features to detections:
        for d_i in next_dets:
            # get patches: getPatchesInBB + roiPool (in helper func)
            patches = getFeaturesFromFeatureMapAndPadding3(d_i, feature_map_and_padding=(frame_features, frame_feat_padding))
            d_i.visual_feat = patches

        t_global = t_global + 1
        if(t_global == len(D13)):
            break
    # ---



    # init trajectories
    tr = []

    t1_d = ([dl[0] for dl in D13[1:21]],[100,250,250])
    t2_d = ([dl[-1] for dl in D13[15:]],[70,150,255])
    t3_d = ([dl[0] for dl in D13[24:30]],[20,255,140])
    tr_x_list = [t1_d, t2_d, t3_d]
    

    for tr_xl in tr_x_list:
        ti = Trajectory(tr_xl[0][0], dspace)
        ti.build2()
        ti_x = []
        iii = 0
        for d in tr_xl[0]:
            ti_x.append(TDet_from_Detection(d))
            addToAvgTr(d.visual_feat, ti)
            if(iii != 0):
                ti.D2[d] = 1.0
            iii = iii+1
        ti.X = ti_x
        ti.color = tr_xl[1]
        tr.append(ti)
    

    # dspace.showSpace(draw_dets=False, det_center_shape='.')
    # dspace.clearSpace()

    t_global = T_OFFSET
    while True:
        # print("at time: %d"%t_global)

        frame = setFrameFlowAtI(dspace, t_global, frames_path=FRAMES_PATH)
        features = computeOrGetPCAFeaturesAtI(t_global, frame,None)
        # showInNamed("window", features)
        # frame = getFrameAtI(i, fpath)
        # dspace.map = frame
        # dspace.lastFrame = frame.copy()

        # 1. init new frame (also compute flow, if needed, same for visual features):
        next_dets: list[Detection]=D13[t_global]
        dspace.D.append(next_dets)
        frame = setFrameFlowAtI(dspace,t_global, None, frames_path=FRAMES_PATH)

        # get dino features for current frame
        frame_features = computeOrGetPCAFeaturesAtI(t_global, frame, None)

        # add features to detections:
        for d_i in next_dets:
            # get patches: getPatchesInBB + roiPool (in helper func)
            patches = getFeaturesFromFeatureMapAndPadding3(d_i, feature_map_and_padding=(frame_features, frame_feat_padding))
            d_i.visual_feat = patches
            # showInNamed("feat1", patches.astype(np.uint8))

        # for t in tr:
        #     t.drawToSpace()
        
        # dspace.showSpace(det_center_shape='.', draw_dets=False)
        # dspace.clearSpace()
        # t_global = (t_global % len(D13)) + 1
        t_global = t_global + 1
        if(t_global == len(D13)):
            break
    
    dspace.clearSpace()
    for t in tr:
        t.drawToSpace()
    dspace.showSpace(draw_dets=False)
    dspace.clearSpace()
    # show avg
    # i = 0
    # while True:
    #     showInNamed("t%d avg"%i, tr[i].visual_avg.astype(np.uint8))
    #     cv.waitKey(0)
    #     i = i+1
    #     if(i == len(tr_x_list)):
    #         break
    # merged_tr: list[Trajectory] = mergeOverlaping(tr[0],tr[1])

    # print(getPossibleNextWithMemo(tr[0], tr, type=3))

    

    merged_tr: list[Trajectory] = getMergedHypotheses(tr, type=1, also_overlapping=True)
    Q=buildQBPMatrixX(merged_tr, type=2)
    print(Q)
    v=solveQBP2(Q)
    print(v)

    print(merged_tr)
    i = 0
    # for mt in merged_tr:
    while True:
        mt = merged_tr[i]
        printTrWithStats(mt)
        # print(mt.X)
        mt.drawToSpace()
        dspace.showSpace(draw_dets=False)
        dspace.clearSpace()
        i = (i+1)%len(merged_tr)
# ---
    


# MAIN_N=0
# MAIN_N=0
# MAIN_N=4
if __name__ == "__main__":
    # read_tr_file()
    
    if(MAIN_N == 1):
        main_d1()
    elif(MAIN_N == 2):
        visualizeTrs()
    elif(MAIN_N == 4):
        visualizeTrs2()
    elif(MAIN_N == 3):
        main_d2()
    else:
        main()

