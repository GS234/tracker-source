from helper_func import *
from DetectionSpace import DetectionSpace
from Trajectory import Trajectory
from OpticalFlow import OpticalFlow
from FeatureExtractor import FeatureExtractor, roiPool, patchesInBB, getFeaturesPca
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
# CUDA_VISIBLE_DEVICES=2 python3 main2.py --sequence='/home/gasper/tracker_ws/workspace/sequences/GOT-10k_GOT-10k_Val_000014' --dets='/home/gasper/tracker_ws/dets/GOT-10k_GOT-10k_Val_000014.txt'



# tracker cache: tracker/flow -> flow, tracker -> result, mogoce tut trajektorije
RESULT_PATH = "./tracker/"
COMPUTED_FLOW = RESULT_PATH+"flow/"
COMPUTED_FEAT = RESULT_PATH+"feat/"
ASK_BEFORE_FLOW_DELETE = not True # the '-y' kinda flag
# --------


DRAW_DETS=False
MAIN_DEB= not True
MAIN_QD = not not True
SAVE_TRAJECTORIES= True # switch to save trajectories on every selection step for vizualization/debug purposes
# EXT_THR=10 # maximum number of extrapolation of trajectories (# of consecutive frames without detections for that trajectory (number of relative holes, essentially))
EXT_THR=5 
T_INIT_S = 0.5
MAX_N_TR_MERGE_ONCE = 100 # maximum number of merged trajectories at once (to limit recursion depth)


# some main-specific functions and constants:
USE_PRECOMPUTED = True # use precomputed optical flow (set to False to calculate it on the go) (flows-path must be set)
# function is used when new flow is computed (on every new frame, for stage III, getFlow is still used, but on files that this thing generated previously)
PRECOMPUTED_FLOW_PATH = "./tracker/flow/" # should be set in options.ini
PRECOMPUTED_FEAT_PATH = "./tracker/feat/"


# flow, features:
COMPUTE_OR_GET_TEST = True
def computeOrGetFlowAtI(i:int, optical_flow_gen: OpticalFlow=None):
    if(USE_PRECOMPUTED): return getFlowAtI(i, COMPUTED_FLOW)
    if(COMPUTE_OR_GET_TEST): return computeFlowAtITest(i, save_flow=True) # WARN: this is to test only, on real use cases, use ^
    return optical_flow_gen.computeFlowAtI(i)

def computeOrGetPCAFeaturesAtI(i:int, image:np.ndarray, feature_ext:FeatureExtractor, save_feat=False):
    if(USE_PRECOMPUTED): return getFeaturesAtI(i, COMPUTED_FEAT)
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
    if(i >= 1):
        flow_to = getFlowAtI(i-1, COMPUTED_FLOW)
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
            t_merged = t.getCopy() # origin is set here, and is correct
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

            # also merge looks
            # add2Avg()
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
def getPossibleNextWithMemo(tr: Trajectory, tr_list:list[Trajectory], time_window=20, type=1):
    global trToPossibleNext
    # print(tr.id)
    if(tr in trToPossibleNext):
        print("is in memo, returning")
        return trToPossibleNext[tr]
    else:
        # print("is not in memo")
        if(type == 1):
            vrni = tr.getPossibleNext2(tr_list, time_window=time_window)
            trToPossibleNext[tr] = vrni
            # print(vrni)
            return vrni
        else:
            flows_path = COMPUTED_FLOW
            # return tr.getPossibleNext3(tr_list, flows_path, time_window=time_window)
        
            vrni = tr.getPossibleNext3(tr_list, flows_path, time_window=time_window)
            trToPossibleNext[tr] = vrni
            # print(vrni)
            return vrni
    

# function generates and returns all possible connections with other trajectories (similar to extend, but it works with whole trajectories now)
def getMergedHypotheses(tr_list: list[Trajectory], time_window=20, max_space_diff=100.0, type=1):
    # init memo:
    global trToPossibleNext
    trToPossibleNext = {}
    # print("this is getMergedHypotheses: ")
    mtr_hypotheses = []
    for tr in tr_list:
        # tr_possible_next = []
        # if(type==1):
        #     tr_possible_next = tr.getPossibleNext2(tr_list, time_window=time_window)
        # if(type == 2):
        #     flows_path = COMPUTED_FLOW
        #     tr_possible_next = tr.getPossibleNext3(tr_list, flows_path, time_window=time_window)
        tr_possible_next = getPossibleNextWithMemo(tr, tr_list, time_window, type)
        all_possible_next = extendAllPossibleNext(tr, [(tr,T_INIT_S,1.0,[])],tr_possible_next,tr_list, type=type, max_n=MAX_N_TR_MERGE_ONCE)
        mtr_hypotheses = mtr_hypotheses + all_possible_next
    return mtr_hypotheses

# metod extends trajectories
# t: current trajectory
# collected: to be merged
# possible next: possible next trajectories that current can 'see'
# all_trs: all trajectories (because we do not have it in dspace)
def extendAllPossibleNext(t:Trajectory, collected:list[tuple[Trajectory, float]], possible_next: list[tuple[Trajectory, float]], all_trs: list[Trajectory], type=1, time_window=20, max_n=MAX_N_TR_MERGE_ONCE):
    # 1. check if there are no more possible next (or maximum is reached)
    if(not possible_next or max_n == 0):
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
        # tr_possible_next = []
        # if(type==1):
        #     tr_possible_next = tr.getPossibleNext2(all_trs, time_window=time_window)
        # if(type==2):
        #     flows_path = COMPUTED_FLOW
        #     tr_possible_next = tr.getPossibleNext3(all_trs, flows_path, time_window=time_window)
        tr_possible_next = getPossibleNextWithMemo(tr, all_trs, time_window, type)
        collected.append( p ) # add it
        possible_next_trs = extendAllPossibleNext(tr, collected, tr_possible_next, all_trs, type=type, time_window=time_window, max_n=max_n-1)
        collected.pop() # remove it
        mtr_hypotheses = mtr_hypotheses + possible_next_trs
    
    # 3. return all collected trajectories
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


# function prints path (list of trajectories) to file
# n: total number of frames
def path2File(path: list[Trajectory], n:int, results_path):
    with open(results_path+"results.txt", 'w') as fp: # fp: file pointer?
        bb_total = 0
        for i in range(len(path)):
            if(i != 0):
                t_end = path[i-1].X[-1].t
                t_begin = path[i].origin.t
                t_diff = t_begin - t_end
                t_pad = t_diff-1
                
                if(i == 0):
                    tr_string, n_bb = path[i-1].tr2bbStr2(t_pad, different_first_line=True)
                else:
                    tr_string, n_bb = path[i-1].tr2bbStr2(t_pad)
                fp.write(tr_string)
               
                bb_total = bb_total + n_bb

            # also add last one
            if(i == len(path)-1):
                # pad last one to end of sequence:
                if(i == 0): # in case this was the only tr in path
                    tr_string, n_bb = path[i].tr2bbStr2(0, different_first_line=True)
                else:
                    tr_string, n_bb = path[i].tr2bbStr2(0)
                fp.write(tr_string)
                bb_total = bb_total + n_bb
                t_pad = n - bb_total
                tr_string = path[i].lastNtimes(t_pad)
                fp.write(tr_string)
                bb_total = bb_total + t_pad

# -------------------------

# function returns init variables from options file
def getConfigConsts(config_name, init_file: str = 'options.ini'):
    init_data = ConfigParser()
    init_data.read(init_file)
    
    frames_path = init_data.get(config_name,'frames_path')
    precomputed_flow = init_data.get(config_name,'precomputed_flow')
    gt_path = frames_path+"../groundtruth.txt"
    dets_path = init_data.get(config_name,'dets_path')
    t_offset = init_data.get(config_name,'t_offset')
    precomputed_feat = init_data.get(config_name, 'precomputed_feat')
    return int(t_offset), frames_path, dets_path, gt_path, precomputed_flow, precomputed_feat

# main:
def main():
    # optional
    global PRECOMPUTED_FEAT_PATH
    global PRECOMPUTED_FLOW_PATH
    
    # set default to tracker ws (./tracker/...)
    global COMPUTED_FLOW
    global COMPUTED_FEAT
    global RESULT_PATH
    global USE_PRECOMPUTED
    
    global SAVE_TRAJECTORIES
    RUN_ALL = not False

    # some defaults:
    T_OFFSET = 1

    # SEQUENCES:
    USE_INIT_FILE = True # use options.ini
    if(USE_INIT_FILE):
        # T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getConfigConsts('got10k14') # got10k14
        # T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getConfigConsts('coin18') # got10k14
        T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getConfigConsts('bird2') # lasot bird2
    
    
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
    args = parser.parse_args()

    if(args.sequence is not None):
        FRAMES_PATH = args.sequence+"/color/"
        GT_PATH = args.sequence+"/groundtruth.txt"

        PRECOMPUTED_FEAT_PATH = args.sequence+'_feat/'
        PRECOMPUTED_FLOW_PATH = args.sequence+'_flow/'
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
    if(args.track is not None):
        COMPUTED_FLOW = args.track+"flow/"
        COMPUTED_FEAT = args.track+"feat/"
        RESULT_PATH = args.track
    
    # if precomputed, then read directly from precomputed path (computed_flow = precomputed_flow_path)
    if(USE_PRECOMPUTED):
        print("[precomputed] using precomputed flow and path")
        COMPUTED_FLOW = PRECOMPUTED_FLOW_PATH
        COMPUTED_FEAT = PRECOMPUTED_FEAT_PATH

    # print(args.sequence)
    print("frames: ",FRAMES_PATH)
    print("dets:   ",DETS_FILE)
    print("flow:   ",COMPUTED_FLOW)
    print("feat:   ",COMPUTED_FEAT)
    print("results:",RESULT_PATH)
    print("preflow:",PRECOMPUTED_FLOW_PATH)
    print("prefeat:",PRECOMPUTED_FEAT_PATH)
    # ----------------------------------------------------------------------

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
    if(not USE_PRECOMPUTED):
        # of:OpticalFlow = OpticalFlow(args,frames_path=FRAMES_PATH, save_path=FLOW_SAVE_PATH)
        of = OpticalFlow(args,frames_path=FRAMES_PATH, save_path=COMPUTED_FLOW) # save_flow = True
        # END INIT RAFT
    
        # INIT FEATURE EXTRACTOR
    featureExt = FeatureExtractor(model_size='base')
    featureExt.getFeatures(frame) # sets necessary offsets by computing first frame (could also do that differently)
    frame_features = computeOrGetPCAFeaturesAtI(t_global, frame, featureExt, save_feat=True) # get features, init trs
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

        patches = getFeaturesFromFeatureMapAndPadding(d, feature_map_and_featExt=(frame_features, featureExt))
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
            print("[@"+str(t_global)+"]")
            
            # 1. init new frame (also compute flow, if needed, same for visual features):
            next_dets=D13[t_global]
            # print("next dets: ",next_dets)
            dspace.D.append(next_dets)
            frame = setFrameFlowAtI(dspace,t_global, of, frames_path=FRAMES_PATH)

            # get dino features for current frame
            frame_features = computeOrGetPCAFeaturesAtI(t_global, frame, featureExt, save_feat=True)
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
                    used_dets_current, forked_tr = t.extend4(feature_map_and_featExt=(frame_features, featureExt)) # need used dets to start new trajectories from unused ones
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
                patches = getFeaturesFromFeatureMapAndPadding(d, feature_map_and_featExt=(frame_features, featureExt))
                addToAvgTr(patches, t_new)
                new_tr.append(t_new)
            
            # (if there are no more that say, 20 trs already)
            # if(len(new_tr) + len(tr) < 35):
            if(len(new_tr) + len(tr) < 20):
                tr = tr+new_tr
            # tr = tr+new_tr
            # ---


            # update target:
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



            # 3.1 pruning step: discard trajectories, that are too far away from target (testing: target is gt)
            # 3.1.1: get current gt:
            print("[pruning]")
            gt_at_ti = GT13[t_global]
            n_prune = 5 # frames
            max_dist = 150
            # tr = getPruned(tr, gt_at_ti, n_prune, max_dist)
            tr = getPruned(tr, [target_d], n_prune, max_dist)
            
            # -----------------
            
            # 4. hypothesis selection:
            
            # 4.1 analyze trs with Q matrix: get pure, drop redundant
            pure_trs, not_pure = analyzeTrsWithQ(tr) # analyze, drop out pure and copies of same (add pure, do qbp with all else)
            
            # 4.2 build Q with trs, that are not pure:
            # tr.sort(key=lambda x: x.getScore2(), reverse=True) # sort to minimize chance of getting stuck in some local minimum (there ARE issues with qbp-solver)
            Q = dspace.buildQBPMatrix3(not_pure)
            
            print("pure:")
            printTrListWithStatsOrdered(pure_trs)
            print("not pure:")
            printTrListWithStatsOrdered(not_pure)
            print(Q)

            # 4.3 solve QBP
            # np.savetxt("Q2.txt",Q, fmt="%7.3f")
            print("solving:")
            v = dspace.solveQBP2(Q)
            print(v)


            # 5. keep only selected trajectories for next frame (all else are term):
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

            # 6. visualization
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
                    t.drawToSpace()
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
                print("before get merged 1")
                merged_trs1 = getMergedHypotheses(tr_all, type=order[0], time_window=stage_II_III_timew)
                print("after get merged 1")
                

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
                print("before get merged 2")
                merged_trs2 = getMergedHypotheses(tr_left, type=order[1], time_window=stage_II_III_timew)
                print("after get merged 2")

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
                Q = buildQBPMatrixX(all_else, type=2) # build q with all trs that have not been filtered by analysis

                # SOLVE QBP:
                # print(Q)
                print("solving Q")
                res = dspace.solveQBP2(Q)
                # res = dspace.solveQBP(Q)
                print(res)
                v = res[0]
                
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
                        printTrWithStats(all_else[t_i], add_to_end=" [x] ")
                
                # add selected merged to idTr_map:
                print("adding selected merged to idTr_map")
                for t_i in selected_merged:
                    idTr_map[t_i.not_so_much_unique_id] = t_i # update trajectories holding/representing that id
                
                # update idtrmap fin/not fin:
                updateFinInIdTR(idTr_map, t_global)
                # print()

                if(not RUN_ALL):
                    print("selected merged: ")
                    printTrListWithStatsOrdered(selected_merged)
                    
                    print("idTr_map:")
                    printTrListWithStatsOrdered(idTr_map.values())

                # 7.5. use new trajectories in next round
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
                    if(gt_at_ti):
                        drawBoundingBox(dspace.map, gt_at_ti[0].bb, [0,0,255])
                    
                    ext_bb_off = 1
                    ext_bb = list(np.array(target_d.bb[0:2])+ext_bb_off)+list(np.array(target_d.bb[2:])-(ext_bb_off*2))
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
    path_creation_type=2
    
    if(path_creation_type == 1):
        # 1. with merging
        # print("merge fin: ")
        merged_merged:list[Trajectory] = []
        merge_fin = list(idTr_map.values())
        if(not RUN_ALL):
            printTrListWithStatsOrdered(merge_fin)
        
        # 7.1. stage II (bridged) (is faster)
        merge_fin = dropRedundant(merge_fin)
        merged_trs1 = getMergedHypotheses(merge_fin, type=order[0], time_window=60) 
        
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
        merged_trs2 = getMergedHypotheses(tr_left, type=order[1], time_window=60)
        # [INFO] flow is no longer needed from here on, so delete possibly precomputed to save space

        merged_merged = merged_trs1+merged_trs2
        merged_merged = dropRedundant(merged_merged, red_level=2)
        merged_merged.sort(key=lambda x: x.getScoreII(), reverse=True) # sort to minimize chance of getting stuck in some local minimum (there ARE issues with qbp-solver)
        # merged_merged = merged_trs2

        early_traj = [] # ones that start somwhere in the begining
        for t in merged_merged:
            if(t.X[0].t <= 10):
                early_traj.append(t)
                printTrWithStats(t, t.not_so_much_unique_id)

    

        # merged_merged = merged_trs2
        merged_merged = early_traj
        print("building Q")
        Q = dspace.buildQBPMatrixX(merged_merged, type=2)
        print(Q)

        
        # SOLVE QBP:
        # print(Q)
        np.savetxt("Q.txt",Q, fmt="%7.3f")
        res = dspace.solveQBP2(Q)
        # res = dspace.solveQBP(Q)
        # print(res)
        v = res[0]

        selected_merged = []
        for i in range(len(merged_merged)):
            if(v[i] == 1):
                selected_merged.append(merged_merged[i])
        merged_merged = selected_merged

        # SHOW RESULTS:
        if(not RUN_ALL):
            dspace.clearSpace()
            i_t = 0
            
            # for tr in idTr_map:
            #     trr = idTr_map[tr]
            for trr in merged_merged:
                printTrWithStats(trr, i_t=i_t)
                trr.drawToSpace()
                dspace.showSpace(draw_dets=DRAW_DETS, draw_last_dets_bb=False, dspace_winname="merged")
                dspace.clearSpace()
                i_t = i_t+1
            
            # for tr in idTr_map:
                # trr = idTr_map[tr]
            for trr in merged_merged:
                trr.drawToSpace()
                i = i+1
            
            dspace.showSpace(draw_dets=DRAW_DETS, draw_last_dets_bb=False, dspace_winname="merged")
            dspace.clearSpace()
        
        # select object to track (from ground truth)
        # get first frame
        selected_region = []
        with open(GT_PATH) as fd:
            first_l = fd.readline()
            det = first_l[0:-1].split(",")
            selected_region = [float(i) for i in det]
        # print(selected_region)

        selected_t = selectBestTrAroundDetAtTime(merged_merged, selected_region)
        
        if(not RUN_ALL):
            print("selected: ")
            printTrWithStats(selected_t)
            selected_t.drawToSpace()
            
            dspace.showSpace(draw_dets=DRAW_DETS, draw_last_dets_bb=False, dspace_winname="merged")
            dspace.clearSpace()
        
        # write trajectory to file
        tr2File(selected_t, n_res, RESULT_PATH)

    # 2. group them by hand:
    if(path_creation_type == 2):
        SAVE_PATH_TR = False
        trajectories: list[Trajectory] = list(idTr_map.values())
        if(trajectories):
            t0 = selectBestTrAroundDetAtTime(trajectories, GT13[T_OFFSET][0].bb)
            path = [t0]+connectTrs(t0, trajectories)
            print(path)
            # end 2.

            # write trajectory to file
            path2File(path, n_all, RESULT_PATH)
            if(SAVE_PATH_TR):
                with open(RESULT_PATH+'trs_path.p', 'wb') as fp:
                    abc = [T_OFFSET, copy.deepcopy(path)]
                    pickle.dump(abc, fp)

    if(SAVE_TRAJECTORIES):
        print("[END] saving trs stored in idtr_map")
        with open(RESULT_PATH+'trs.p', 'wb') as fp:
            # abc = [T_OFFSET, copy.deepcopy(idTr_map)]
            abc = [T_OFFSET, copy.deepcopy(idTr_map)]
            pickle.dump(abc, fp)
    if(not USE_PRECOMPUTED):
        deletePrecomputed(COMPUTED_FLOW, confirm=ASK_BEFORE_FLOW_DELETE)
        deletePrecomputed(COMPUTED_FEAT, confirm=ASK_BEFORE_FLOW_DELETE)
    print("[END] results saved.")
    
    # end main

# debug main:
def main_d():
    print("[INFO] This is main_d. To run main, set MAIN_DEB to False.")

    FRAMES_PATH = "frames/LaSOT_bird-2/color/"
    # DETS_FILE = "testing_data/LaSOT_bird-2_001.txt"
    DETS_FILE = "../src/results.txt"
    GT_PATH = "frames/LaSOT_bird-2/groundtruth.txt"
    T_OFFSET = 1
    n = 0

    # ----------------------------------------------------------------------

    D13 = [[]]+readDetFile2(DETS_FILE, T_OFFSET) # read detections from file
    D14 = [[]]+readDetFile2(GT_PATH, T_OFFSET) # read ground truth
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
    with open(GT_PATH) as fd:
        first_l = fd.readline()
        det = first_l[0:-1].split(",")
        target_select = [float(i) for i in det]
    print("selected det.:", target_select)
    
    
    # INIT DSPACE
    frame = getFrameAtI(t_global,FRAMES_PATH)
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
        frame = getFrameAtI(t_global,FRAMES_PATH)
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
    global COMPUTED_FLOW
    global COMPUTED_FEAT

    # T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, _, _ = getConfigConsts('bird2') # lasot bird2 sequence
    # T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, _, _ = getConfigConsts('got10k14') # got10k14 sequence
    T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, COMPUTED_FLOW, COMPUTED_FEAT = getConfigConsts('coin18') # coin18 sequence
    # T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, COMPUTED_FLOW, COMPUTED_FEAT = getConfigConsts('bird2') # bird2 sequence
    t_global = T_OFFSET
    # init dspace
    
    # get frame:
    frame0 = getFrameAtI(t_global,FRAMES_PATH)
    # print(frame0)
    h, w, _ = np.shape(frame0)
    dspace = DetectionSpace(h,w,time_offset=T_OFFSET, show_flow=False, disable_vis=False)
    dspace.map = frame0
    dspace.lastFrame = frame0.copy()
    D13 = [[]]+readDetFile2(DETS_FILE, T_OFFSET)
    GT13: list[list[Detection]] = [[]]+readDetFile2(GT_PATH,T_OFFSET)
    n = len(D13)-1 # all frames
    print(n)
    dspace.D = D13[t_global:]
    # dspace.showSpace()
    # end init dspace

    # init feature extractor
    featureExt = FeatureExtractor(model_size='base')
    featureExt.getFeatures(frame0) # sets necessary offsets by computing features for first frame
    # end init feature ext

    # get trajectories:
    # T_OFFSET, t_global, tr_all = read_tr_file('trsdeb2.p')
    file = read_tr_file('results_seq/coin18/trs.p')
    T_OFFSET, idTr_map = file
    print(file)

    print("tr all:")
    
    printTrListWithStatsOrdered(idTr_map.values())

    # ------------
    # # 7.1. stage II (bridged) (is faster)
    # # order=[2,1]
    # order=[1,2] # maybe this, flow is slow
    # stage_II_III_timew = 10
    # print("before get merged 1")
    # merged_trs1 = getMergedHypotheses(tr_all, type=order[0], time_window=stage_II_III_timew)
    # print("after get merged 1")
    

    # # 7.2. get unused trajectories
    # tr_left = set(tr_all)
    # t_i = 0
    # for t in merged_trs1:
    #     key_set = set(t.T2.keys())
    #     if(len(key_set) > 1):
    #         tr_left = tr_left - key_set
    #     t_i = t_i+1
    # tr_left = list(tr_left)

    # # 7.3. stage III (flow) with unused trajectories (is slower)
    # print("before get merged 2")
    # merged_trs2 = getMergedHypotheses(tr_left, type=order[1], time_window=stage_II_III_timew)
    # print("after get merged 2")

    # # set all previous to fin, as they are replaced by new ones:
    # # for t in tr:
    # #     t.term = True
    
    
    # # 7.4. select best trajectories to be used in new round
    # merged_merged = merged_trs1+merged_trs2

    # # also prune them before qbp (but only those that are fin):
    # # prune ones that are too far before building Q:
    # # get only those that are not fin
    # print("before pruning")
    # to_prune: list[Trajectory] = []
    # all_else: list[Trajectory] = []
    # for tr in merged_merged:
    #     if(len(tr.X) > 1): # also keep only ones that are longer than 1
    #         if(tr.term):
    #             all_else.append(tr)
    #         else:
    #             to_prune.append(tr)
    # print("[pruning merged]")
    # gt_at_ti = GT13[t_global]
    # max_dist = 150
    # tr = getPruned(to_prune, gt_at_ti, -1, max_dist) # -1? to prune without length restriction
    # merged_merged = to_prune + all_else


    # merged_merged = dropRedundant(merged_merged, red_level=2) # redundancy level: 2 ('strict' mode: preserve only one end point and number of merged)
    # # q-matrix analysis (type 2 this time)
    # simple_tr,all_else = analyzeTrsWithQ(merged_merged, type=2)
    # Q = buildQBPMatrixX(all_else, type=2) # build q with all trs that have not been filtered by analysis

    # # SOLVE QBP:
    # # print(Q)
    # print("solving Q")
    # res = dspace.solveQBP2(Q)
    # # res = dspace.solveQBP(Q)
    # print(res)
    # v = res[0]
    # selected_trs = simple_tr + [x[0] for x in zip(all_else, v) if x[1] == 1]

    # print("selected:")
    # printTrListWithStatsOrdered(selected_trs)
    # ------------

    
    # print("merged merged:")
    # printTrListWithStatsOrdered(merged_merged)

    
    # Q = buildQBPMatrixX(merged_merged, type=2) # add this, and also make memoization
    # print(Q)
    # print(dspace.solveQBP2(Q))
    
    # a,b = analyzeTrsWithQ(merged_merged, type=2)
    # print("this is a:")
    # printTrListWithStatsOrdered(a)
    # print("this is b:")
    # printTrListWithStatsOrdered(b)
    # return


    # merged_merged:list[Trajectory] = []
    # t_off, t_2, selected_t, idTr_map, merged_merged = read_tr_file('trs.p')
    # _, idTr_map = read_tr_file('trsdeb.p')

    
    # i = 0
    # print("idTr_map (all):")
    # for id_tr in idTr_map:
    #     tr = idTr_map[id_tr]
    #     printTrWithStats(tr, i)
    #     tr.detectionSpace = dspace
    #     i+=1


    # show trajectories in current time (and image features) ---
    # print i-th tr
    # trajectories = merged_merged
    trajectories: list[Trajectory] = list(idTr_map.values())
    t0 = selectBestTrAroundDetAtTime(trajectories, GT13[T_OFFSET][0].bb)
    path = [t0]+connectTrs(t0, trajectories)
    
    path_trs = []
    for tr in path:
        tr.color = [0,0,255]
        path_trs.append(tr)
    # path2File(path, n, RESULT_PATH+'2')
    # trajectories = trajectories + path
    t_global = 1
    # t_global = 570
    # t_global = 800
    # t_global = 805
    # t_global = 3690
    for t in trajectories:
        t.detectionSpace = dspace
    trajectories = path_trs

    SAVE_FEAT = not True
    SHOW_IMG = True
    while t_global <= n:
        print("-> ",t_global) # print time
        frame = getFrameAtI(t_global, FRAMES_PATH)
        dspace.map = frame
        dspace.lastFrame = frame.copy()

        # patches:
        pca_feat = computeOrGetPCAFeaturesAtI(t_global, frame, featureExt, save_feat=SAVE_FEAT)
        pad_l, pad_u = featureExt.last_padder.pad_l, featureExt.last_padder.pad_u # used to align bb properly
        if(SHOW_IMG):
            showInNamed("image - features", pca_feat)

        # get all trajectories, that live in this time instant:
        # trajectories_that_live_at_this_time = []
        print("trajectories at this time: ")
        for tr in trajectories:
            # printTrWithStats(tr)
            # if(tr.X[0].t <= t_global and t_global <= tr.X[-1].t):
            if(tr.X[0].t <= t_global and t_global <= tr.X[0].t + len(tr.X)-1):
                # trajectories_that_live_at_this_time.append(tr)
                # current tdet:
                t = t_global-tr.X[0].t # offset time
                tr_det_t:TDet = tr.X[t]
                printTrWithStats(tr)
                
                # get features:
                patches = patchesInBB(pca_feat, tr_det_t.bb, xy_off=[pad_l, pad_u])
                patches_rp = roiPool(patches, use_2d=False)
                
                # draw things
                if(SHOW_IMG):
                    tr.drawToSpace() # draw traj
                    drawBoundingBox(dspace.map, tr_det_t.bb, tr.color) # draw bb
                    drawX(dspace.map, tr_det_t.x)
                    # showInNamed("%d"%(tr.not_so_much_unique_id), patches_rp) # draw features

        # draw gt:
        if(SHOW_IMG):
            if(GT13[t_global]):
                drawBoundingBox(dspace.map, GT13[t_global][0].bb, [0,50,205])
            dspace.showSpace(draw_dets=False, draw_last_dets_bb=True, at_time=t_global, bb_color=[0,255,200])
            dspace.clearSpace()
        t_global = t_global +1
    # ---------------------------------
    
    # printTrWithStats(selected_t)
    # # print(idTr_map)
    # # print(merged_merged)
    # # def sort_fun1(x):
    # #     return len(x.T2.keys())
    # # def sort_fun2(x):
    # #     return len(x.T2.keys())
    # def len_nmerged_comparator(x,y):
    #     len_merged_diff = len(x.T2.keys()) - len(y.T2.keys())
    #     len_diff = len(x.X) - len(y.X)
    #     if(len_merged_diff == 0):
    #         return (len_diff)
    #     else:
    #         return len_merged_diff
        
    # from functools import cmp_to_key
    # # merged_merged.sort(key=lambda x: len(x.T2.keys()), reverse=True)
    # merged_merged.sort(key=cmp_to_key(len_nmerged_comparator), reverse=True)
    # # merged_merged_selected_by_hand = []
    # # for i in range(len(merged_merged)):
    # #     if(i in [0,9,51,105,130,140,144]):
    # #         merged_merged_selected_by_hand.append(merged_merged[i])
    # merged_merged = dropRedundant(merged_merged, red_level=2)
    # merged_merged = merged_merged_selected_by_hand
    # printTrListWithStatsOrdered(merged_merged)
    # merged_merged_temp = merged_merged
    # merged_merged = merged_merged[8:43+1]
    # merged_merged.sort(key=lambda x: x)

    # Q = dspace.buildQBPMatrixX(merged_merged, type=2)
    # selected = dspace.solveQBP2(Q)
    # v= selected[0]
    # sel_tr = zip(merged_merged, v)
    # selected_trs = [x[0] for x in sel_tr if x[1] == 1]
    
    # print(selected)
    # printTrListWithStatsOrdered(selected_trs)
    # print(sel_tr)

    # i = 0
    # print("merged:")
    # for i in range(len(merged_merged)):
    #     tr = merged_merged[i]
    # # for tr in merged_merged:
    # # for tr in selected_trs:
    #     prob_same = False
    #     for j in range(i):
    #         tr_j = merged_merged[j]
    #         tr_test = trDuckTest(tr, tr_j, also_check_origin=False)
    #         if(tr_test):
    #             prob_same = True
    #             break

    #     printTrWithStats(tr, i, add_to_end="keys: %s, dt=%s"%(str( sorted([x.not_so_much_unique_id for x in tr.T2.keys()]) ), str(prob_same) ))
    #     # printTrWithStats(tr, i, add_to_end="keys: %s"%(str( sorted([x.id for x in tr.T2.keys()]) ) ))
    #     tr.detectionSpace = dspace
    #     tr.drawToSpace()
    #     dspace.showSpace(draw_dets=False, draw_last_dets_bb=False)
    #     dspace.clearSpace()

        # # print(tr.T2.keys())
        # for t in tr.T2.keys():
        #     print(t.X[0], end=" ", flush=True)
        # print()
        # i+=1

    # one specific trajectory
    # printTrWithStats(idTr_map[1])
    # for tr in idTr_map[1].T2.keys():
    #     printTrWithStats(tr)


    # build qbp:
    # idx = [7,17,12,10]
    # idx = [7,26]
    # merged_selected:list[Trajectory] = []
    # for i in range(len(merged_merged)):
    #     if(i in idx):
    #         merged_selected.append(merged_merged[i])
    # Q = buildQBPFromTrs(merged_selected,dspace) # method simply called build method from dspace, it can now be done directly
    # print(Q)
    # v = dspace.solveQBP2(Q)
    # print(v)

    # check something:
    # t_at_i = 25
    # print("this is t at ",t_at_i,":")
    # tr19 = merged_merged[t_at_i]
    # printTrWithStats(tr19)
    # print(list(list(tr19.T2.keys())[0].T2)[-1].origin)
    # print(tr19.X)
    # for tr in tr19.T2.keys():
    #     printTrWithStats(tr)
    # print("---")
    
    # trs = list(list(tr19.T2.keys())[0].T2)
    # dspace.map = getFrameAtI(781, FRAMES_PATH)
    # dspace.lastFrame = dspace.map.copy()
    # for tr in trs:
    #     printTrWithStats(tr)
    #     tr.detectionSpace = dspace
    #     tr.drawToSpace()

    # dspace.showSpace(draw_dets=False, draw_last_dets_bb=False)
    # dspace.showSpace(draw_dets=False, draw_last_dets_bb=False)
    

    # merged_trs = getMergedHypotheses(trs,30,type=2)
    # # print(merged_trs[0].X[-5:])
    # # print(merged_trs[1].X[-5:])
    # # print(merged_trs[2].X[0])
    # print(merged_trs)
    # for m in merged_trs:
    #     printTrWithStats(m)
    
    
    # return
        
    # show trajectories (selected only)
    # while True:
    # # for ii in i:
    #     ii = i[j]
    #     tr = merged_merged[ii]
    #     start_time = tr.X[0].t
    #     end_time = tr.X[-1].t
    #     print(ii)
    #     for t in range(end_time-start_time):
    #         t_global = start_time+t
    #         # set frame
    #         frame = getFrameAtI(t_global, FRAMES_PATH)
    #         dspace.map = frame
    #         dspace.lastFrame = frame.copy()
            

    #         # current tdet:
    #         tr_det_t:TDet = tr.X[t]
    #         merged_merged[ii].drawToSpace() # draw traj
    #         drawBoundingBox(dspace.map, tr_det_t.bb) # draw bb
    #         drawX(dspace.map, tr_det_t.x)
    #         dspace.showSpace(draw_dets=False, draw_last_dets_bb=False)
    #         dspace.clearSpace()
    #     j = (j + 1)%len(i)
    
    # t_global = t_global+1
    # while True:
    #     frame = getFrameAtI(t_global, FRAMES_PATH)
    #     dspace.map = frame
    #     dspace.lastFrame = frame.copy()

    #     dspace.showSpace()
    #     dspace.clearSpace()
    #     t_global = t_global + 1

    

def read_tr_file(filename='trs.p'):
    PATH_TO_DET_FILE = 'tracker/'+filename
    tr = None
    with open(PATH_TO_DET_FILE, 'rb') as fd:
        tr = pickle.load(fd)
    return tr
    # for t in tr:
    #     printTrWithStats(t, t.not_so_much_unique_id)
    


if __name__ == "__main__":
    # read_tr_file()
    
    if(MAIN_DEB):
        main_d()
    elif(MAIN_QD):
        main_qd()
    else:
        main()

