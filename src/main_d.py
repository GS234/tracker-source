from helper_func import *
from DetectionSpace import DetectionSpace
from Trajectory import Trajectory
from OpticalFlow import OpticalFlow
from FeatureExtractor import FeatureExtractor, roiPool, patchesInBB, getFeaturesPca
from Detection import Detection, TDet, TDet_from_Detection
import cv2 as cv
import numpy as np
import copy # for deepcopy (visualization purposes)
import pickle
import argparse # for raft model argument parser
from pathlib import Path
from configparser import ConfigParser
np.set_printoptions(suppress=True, precision=3, linewidth=1000)

SEQUENCE = "LaSOT_bird-15"
DATA_ROOT = "../data/"
FRAMES_PATH = DATA_ROOT+"frames/"+SEQUENCE+"/color/"

# tracker cache: tracker/flow -> flow, tracker -> result, mogoce tut trajektorije
COMPUTED_FLOW = "./tracker/flow/"
COMPUTED_FEAT = "./tracker/feat/"
RESULT_PATH = "./tracker/"
# --------

USE_PRECOMPUTED = not True # use precomputed optical flow (set to False to calculate it on the go) (flows-path must be set)
ASK_BEFORE_FLOW_DELETE = not not True # the '-y' kinda flag
FLOWS_PATH = DATA_ROOT + "flow_est/"+SEQUENCE+"/"
if(not USE_PRECOMPUTED):
    FLOWS_PATH = DATA_ROOT + "flow_est/"+SEQUENCE+"/computed/" # this is here to not overwrite existing data, in normal cases it would be same as ^ (FLOWS_PATH)
else:
    COMPUTED_FLOW = FLOWS_PATH

DRAW_DETS=False
SAVE_TRAJECTORIES= not True # switch to save trajectories on every selection step for vizualization/debug purposes
# EXT_THR=10 # maximum number of extrapolation of trajectories (# of consecutive frames without detections for that trajectory (number of relative holes, essentially))
EXT_THR=5 
PO_THR=7 # possibly occluded threshold: number of holes before trajectory is marked as possibly occluded
UNSELECTED_STRIKE_MAX=5 # maximum number of times that trajectory is not selected but included in set
T_INIT_S = 0.5
MAX_N_TR_MERGE_ONCE = 100 # maximum number of merged trajectories at once (to limit recursion depth)


# some main-specific functions:
def buildQBPFromTrs(trs:list[Trajectory], dspace:DetectionSpace):
    Q = dspace.buildQBPMatrixX(trs,type=2)
    # print(Q)
    return Q

def read_tr_file(filename='trs.p'):
    PATH_TO_DET_FILE = 'tracker/'+filename
    tr = None
    with open(PATH_TO_DET_FILE, 'rb') as fd:
        tr = pickle.load(fd)
    return tr
    # for t in tr:
    #     printTrWithStats(t, t.not_so_much_unique_id)

# function is used when new flow is computed (on every new frame, for stage III, getFlow is still used, but on files that this thing generated previously)
def computeOrGetFlowAtI(i:int, optical_flow_gen: OpticalFlow=None):
    if(not USE_PRECOMPUTED):
        # return optical_flow_gen.computeFlowAtI(i)
        return optical_flow_gen.computeFlowAtITest(i) # WARN: this is to test only, on real use cases, use ^
    return getFlowAtI(i, COMPUTED_FLOW)

def setFrameFlowAtI(dspace: DetectionSpace, i: int, optical_flow_gen: OpticalFlow = None):
    # set new frame
    frame = getFrameAtI(i,FRAMES_PATH)
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

# function drops redundant trajectories from list (those that are same (duplicated))
def dropRedundant(tr_list:list[Trajectory]):
    new_l = []
    red_count = 0
    for i in range(len(tr_list)):
        already_contained = False
        for j in range(i):
            if(i == j):
                continue
            if(tr_list[i].T2.keys() == tr_list[j].T2.keys()):
                already_contained = True
                red_count = red_count+1
                break
        if(not already_contained):
            new_l.append(tr_list[i])
    print("[dropRed] dropped %d redundant trajectories."%red_count)
    return new_l

# get number of pure trajectories: that only have merit term
def getPureTrN(Q):
    n, n = np.shape(Q)
    num = 0
    for i in range(n):
        if(isPure(Q,i)):
            num = num+1
        
    # print(num)
    return num

# function checks if trajectory in Q at i is pure
def isPure(Q,i):
    vrni = False
    x = Q[:, i] # column
    temp = x[i]
    x[i] = 0
    if(math.isclose(np.sum(x), 0)):
        vrni=True
    x[i] = temp
    return vrni

def updateFinInIdTR(idTR_map: dict[int, Trajectory], t_current:int):
    for k in idTR_map.keys():
        tr = idTR_map[k]
        if(tr.X[-1].t < t_current):
            tr.term=True

def printTrListWithStatsOrdered(tr_list: list[Trajectory]):
    t_i = 0
    for t in tr_list:
        printTrWithStats(t, t_i)
        t_i = t_i+1

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
        tr_possible_next = []
        if(type==1):
            tr_possible_next = tr.getPossibleNext2(all_trs, time_window=time_window)
        if(type==2):
            flows_path = COMPUTED_FLOW
            tr_possible_next = tr.getPossibleNext3(all_trs, flows_path, time_window=time_window)
        collected.append( p ) # add it
        possible_next_trs = extendAllPossibleNext(tr, collected, tr_possible_next, all_trs, time_window=time_window, max_n=max_n-1)
        collected.pop() # remove it
        mtr_hypotheses = mtr_hypotheses + possible_next_trs
    
    # 3. return all collected trajectories
    return mtr_hypotheses

def printTrWithStats(t: Trajectory, i_t=None, also_T2k=False):
    s = "t%-5s (t%-5s) o=[%-3d,%-3d], c=%-3d, len=%-4d, score=%9.4f %5d - %-5d, n_merged=%3d (fin:%s)" % \
          (t.not_so_much_unique_id,t.id, t.origin.x[0],t.origin.x[1], t.color[2], len(t.X), t.getScore2(), t.X[0].t, t.X[-1].t, len(t.T2.keys()), str(t.term))
    if(i_t is not None):
        s = "%3d. %s"%(i_t,s)
    
    if(also_T2k):
        joined = ""
        for k in t.T2.keys():
            joined = "%s %s"%(joined,k.id)
        s = "%s [%s]"%(s,joined)
    print(s)

def tr2File(selected_t, pad, path):
    lines_to_file = selected_t.tr2bbStr(pad).split('\n')
    lines_to_file[0] = "1"
    str_to_file = '\n'.join(lines_to_file)
    with open(path+"results.txt", 'w') as fp: # fp: file pointer?
        fp.write(str_to_file)


# debug main:
def main_d1():
    print("[INFO] This is main_d. To run main, set MAIN_DEB to False.")

    FRAMES_PATH = '/home/gasper/disk/Nedokumenti/Faks/didi_sequences/LaSOT_bird-2/color/'
    FLOW_PATH = '/home/gasper/disk/Nedokumenti/Faks/didi_sequences/LaSOT_bird-2_1/'
    COMPUTED_FLOW = '/home/gasper/disk/Nedokumenti/Faks/didi_sequences/LaSOT_bird-2_1/'
    # DETS_PATH = '/home/gasper/Faks/3_letnik/diplomska/koda/data/detections/LaSOT_bird-2.txt'
    DETS_PATH = '/home/gasper/Faks/3_letnik/diplomska/koda/src/results.txt'
    GT_PATH = '/home/gasper/disk/Nedokumenti/Faks/didi_sequences/LaSOT_bird-2/groundtruth.txt' # ground truth

    # FRAMES_PATH = "frames/LaSOT_bird-2/color/"
    # FLOWS_PATH = "flow_est/LaSOT_bird-2_1/"
    # # DETS_FILE = "testing_data/LaSOT_bird-2_001.txt"
    # DETS_PATH = "../src/results.txt"
    # GT_PATH = "frames/LaSOT_bird-2/groundtruth.txt"
    T_OFFSET = 1
    n = 0

    # ----------------------------------------------------------------------

    D13 = [[]]+readDetFile2(DETS_PATH, T_OFFSET) # read detections from file
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


# function connects trajectory using visual information
def connectTrs(tr1:Trajectory, tr_all:list[Trajectory], max_time=100):
    if(tr1 is None):
        return []
    # print("t%d:"%(tr1.not_so_much_unique_id))

    end_time = tr1.X[-1].t
    max_score = 0
    tr_add = None
    for t in tr_all:
        visual_score = getVisualSimilarity(tr1, t)
        # print("t%d - t%d: %16.14f"%(tr1.not_so_much_unique_id, t.not_so_much_unique_id, visual_score))
        time_diff = t.origin.t - end_time

        if(time_diff >= 0 and time_diff <= max_time):
            visual_score = getVisualSimilarity(tr1, t)
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



# function calculates visual distance:
def getVisualSimilarity(tr1:Trajectory, tr2:Trajectory):
    k = 100
    v1 = tr1.possibly_occluded.reshape(-1)
    v2 = tr2.possibly_occluded.reshape(-1)
    # print(v1)
    diff = v2-v1
    dist = np.sqrt(np.dot(diff, diff))
    dist_k = dist/k
    return np.exp(-dist_k)
    # return dist


def getOrComputePCAFeaturesAtI(i:int, image:np.ndarray, feature_ext:FeatureExtractor, save_feat=False):
    try:
        return getFeaturesAtI(i, COMPUTED_FEAT)
    except:
        frame_feat = feature_ext.getFeatures(image)['x_norm_patchtokens'].to('cpu').numpy() # get frame features (patchtokens)
        # print("frame_feat:")
        # print(np.shape(frame_feat))
        pca_feat = getFeaturesPca(frame_feat, (feature_ext.last_padder.patch_h, feature_ext.last_padder.patch_w))
        if(save_feat):
            np.save(COMPUTED_FEAT+i2frameI(i)+".npy",pca_feat)
        return pca_feat

# addToAvg wrapper: calls it and modifies trajectory properties
# def addToAvgTr(a_i, tr:Trajectory):
#     if(tr.possibly_occluded is None):
#         tr.possibly_occluded = a_i
#     else:
#         new_a = addToAvg(tr.possibly_occluded, a_i, tr.possible_next)
#         tr.possibly_occluded = new_a
#    tr.possible_next = tr.possible_next+1

# def tr2File(selected_t, pad, path):
#     lines_to_file = selected_t.tr2bbStr(pad).split('\n')
#     lines_to_file[0] = "1"
#     str_to_file = '\n'.join(lines_to_file)
#     with open(path+"results.txt", 'w') as fp: # fp: file pointer?
#         fp.write(str_to_file)


def path2File(path: list[Trajectory], n:int, results_path):
    with open(results_path+"results.txt", 'w') as fp: # fp: file pointer?
        bb_total = 0
        for i in range(len(path)):
            if(i != 0):
                t_end = path[i-1].X[-1].t
                t_begin = path[i].origin.t
                t_diff = t_begin - t_end
                t_pad = t_diff-1
                # print("%d - %d, should add %d frames (range %d - %d) "%(t_end, t_begin, t_pad, t_end+1, t_end+t_pad), end="")
                # print("(", end="", flush=True)
                tr_string, n_bb = path[i-1].tr2bbStr2(t_pad)
                fp.write(tr_string)
                # print(" (added %d bbs)"%(n_bb), end="")
                bb_total = bb_total + n_bb

                # for j in range(t_add):
                #     if(j == 0 or j+1 == t_add):
                #         print(" %d"%(t_beginAdd+j), end=" ")
            #     print(")", end=" ", flush=True)
            # print(len(path[i].X))

            # also add last one
            if(i == len(path)-1):
                # pad last one to end of sequence:
                tr_string, n_bb = path[i].tr2bbStr2(0)
                fp.write(tr_string)
                bb_total = bb_total + n_bb
                t_pad = n - bb_total
                tr_string = path[i].lastNtimes(t_pad)
                fp.write(tr_string)
                bb_total = bb_total + t_pad
    #         print(" (added %d bbs)"%(n_bb))
    # print("bb total: %d"%(bb_total))
    # for tdet in path[-4].X:
    #     print(tdet)

def getConfigConsts(config_name, init_file: str = 'options.ini'):
    init_data = ConfigParser()
    init_data.read('options.ini')
    
    frames_path = init_data.get(config_name,'frames_path')
    flow_path = init_data.get(config_name,'flow_path')
    computed_flow = init_data.get(config_name,'computed_flow')
    gt_path = frames_path+"../groundtruth.txt"
    dets_path = init_data.get(config_name,'dets_path')
    t_offset = init_data.get(config_name,'t_offset')
    return frames_path, dets_path, int(t_offset), flow_path, computed_flow, gt_path


# quick debug main:
def main_d2():
    global COMPUTED_FLOW
    print("[INFO] This is main_qd. To run main, set MAIN_DEB to False. To run main_deb, set MAIN_QD to False.")


    # init vars:
    # FRAMES_PATH = '/home/gasper/disk/Nedokumenti/Faks/didi_sequences/LaSOT_bird-2/color/'
    # FLOW_PATH = '/home/gasper/disk/Nedokumenti/Faks/didi_sequences/LaSOT_bird-2_1/'
    # COMPUTED_FLOW = '/home/gasper/disk/Nedokumenti/Faks/didi_sequences/LaSOT_bird-2_1/'
    # GT_PATH = '/home/gasper/disk/Nedokumenti/Faks/didi_sequences/LaSOT_bird-2/groundtruth.txt' # ground truth
    # DETS_PATH = '/home/gasper/Faks/3_letnik/diplomska/koda/data/detections/LaSOT_bird-2.txt'
    # T_OFFSET = 1
    FRAMES_PATH, DETS_PATH, T_OFFSET, FLOW_PATH, COMPUTED_FLOW, GT_PATH = getConfigConsts('bird2')
    t_global = T_OFFSET
    # init dspace
    
    # init feature extractor
    featureExt = FeatureExtractor(model_size='base')
    # end init feature ext

    # get frame:
    frame0 = getFrameAtI(t_global,FRAMES_PATH)
    # frame0 = np.zeros((10,10,3))
    # print(frame0)
    h, w, _ = np.shape(frame0)
    dspace = DetectionSpace(h,w,time_offset=T_OFFSET, show_flow=False)
    dspace.map = frame0
    dspace.lastFrame = frame0.copy()
    D13: list[list[Detection]] = [[]]+readDetFile2(DETS_PATH, T_OFFSET)
    GT13: list[list[Detection]] = [[]]+readDetFile2(GT_PATH,T_OFFSET)
    # print(GT13[3693])
    # print(GT13[3694])
    n = len(D13)-1 # all frames
    print(n)
    dspace.D = D13[t_global:]
    # dspace.showSpace()
    # end init dspace

    # compute features for first frame (to set ImagePadder)
    featureExt.getFeatures(frame0) # sets necessary offsets



    # get trajectories:
    merged_merged:list[Trajectory] = []
    # t_off, t_2, idTr_map, merged_merged = read_tr_file('trs.p')
    t_off, t_2, idTr_map, merged_merged = read_tr_file('trs_w_feat.p')
    # print(idTr_map)
    # print(merged_merged)
    
    i = 0
    print("merged:")
    for tr in merged_merged:
        printTrWithStats(tr, i)
        tr.detectionSpace = dspace
        
        # also init features:
        # tr.possibly_occluded = None
        # tr.possible_next = 0


        i = i+1
    
    i = 0
    print("idTr_map (all):")
    for id_tr in idTr_map:
        tr = idTr_map[id_tr]
        printTrWithStats(tr, i)
        tr.detectionSpace = dspace
        # also init features:
        # tr.possibly_occluded = None
        # tr.possible_next = 0
        i = i+1

    # one specific trajectory

    # show trajectories in current time (and image features)
    # print i-th tr
    # trajectories = merged_merged
    trajectories: list[Trajectory] = idTr_map.values()
    t_global = 1
    # t_global = 805

    t0:Trajectory = idTr_map[0]
    # print(t0, t0.not_so_much_unique_id)
    path = [t0]+connectTrs(t0, trajectories)
    # print(trajectories)
    # for t in path:
    #     print(t)
    print("path:")
    printTrListWithStatsOrdered(path)

    # path2File(path, n, './')
    # return
    

    SAVE_FEAT = not True
    SHOW_IMG =  True
    showing_prev = set() # set showing windows
    showing_next = set()
    t_global = 3600
    while t_global <= n:
        print("-> ",t_global) # print time
        frame = getFrameAtI(t_global, FRAMES_PATH)
        dspace.map = frame
        dspace.lastFrame = frame.copy()

        # # patches:
        # pca_feat = getOrComputePCAFeaturesAtI(t_global, frame, featureExt, save_feat=SAVE_FEAT)
        # pad_l, pad_u = featureExt.last_padder.pad_l, featureExt.last_padder.pad_u # used to align bb properly
        # if(SHOW_IMG):
        #     showInNamed("image - features", pca_feat)

        # get all trajectories, that live in this time instant:
        # trajectories_that_live_at_this_time = []
        print("trajectories at this time: ")
        showing_next = set()
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
                # patches = patchesInBB(pca_feat, tr_det_t.bb, xy_off=[pad_l, pad_u])
                # patches_rp = roiPool(patches, use_2d=False)

                # # update look with new features:
                # addToAvgTr(patches_rp.astype(np.float64), tr)


                # if(tr.not_so_much_unique_id == 0):
                #     showInNamed("t0 features", patches_rp)
                #     addToAvgTr(patches_rp.astype(np.float64), tr)
                #     t0_avg_feat = tr.possibly_occluded
                #     # update avg look:
                #     # if(t0_avg_feat is None):
                #     #     t0_avg_feat = patches_rp.astype(np.float64)
                #     # else:
                #     #     t0_avg_feat = addToAvg(t0_avg_feat, patches_rp.astype(np.float64), t0_i)
                #     # t0_i = t0_i+1
                
                # draw things
                if(SHOW_IMG):
                    color = [150,150,150] # gray
                    if(tr in path):
                        color = [0,180,255] # somewhat red
                    tr.drawToSpace(color=color) # draw traj
                    drawBoundingBox(dspace.map, tr_det_t.bb, tr.color) # draw bb
                    drawX(dspace.map, tr_det_t.x)
                    if(True):
                        showInNamed("%d"%(tr.not_so_much_unique_id), tr.possibly_occluded.astype(np.uint8)) # draw features (avg)
                        showing_next.add(tr)


        # destroy all windows which do not have their trajectory showing anymore
        # showing_prev = set() # set showing windows
        # showing_next = set()
        # showing_next = set()
        showing_diff = showing_prev - showing_next
        for t in showing_diff:
            
            cv.destroyWindow("%d"%(t.not_so_much_unique_id))
        showing_prev = showing_next

        # draw gt:
        if(SHOW_IMG):
            if(GT13[t_global]): # if it has bb
                drawBoundingBox(dspace.map, GT13[t_global][0].bb, [0,50,205])
            # showInNamed("t0 avg:", t0_avg_feat.astype(np.uint8))
            dspace.showSpace(draw_dets=False, draw_last_dets_bb=True, at_time=t_global, bb_color=[0,255,200])
            dspace.clearSpace()
        t_global = t_global +1


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



def get_avg_tr_look():    
    print("[INFO] this is to compute trajectories' average look")

    # init vars:
    FRAMES_PATH = '/home/gasper/disk/Nedokumenti/Faks/didi_sequences/LaSOT_bird-2/color/'
    T_OFFSET = 1
    t_global = T_OFFSET
    n = 4101
    
    # init feature extractor
    featureExt = FeatureExtractor(model_size='base')
    

    # get frame:
    frame0 = getFrameAtI(t_global,FRAMES_PATH)
    # compute features for first frame (to set ImagePadder)
    featureExt.getFeatures(frame0) # sets necessary offsets

    # get trajectories:
    merged_merged:list[Trajectory] = []
    t_off, t_2, idTr_map, merged_merged = read_tr_file('trs.p')
    print(idTr_map)
    print(merged_merged)
    
    i = 0
    print("merged:")
    for tr in merged_merged:
        printTrWithStats(tr, i)
        # also init features:
        tr.possibly_occluded = None
        tr.possible_next = 0
        i = i+1

    i = 0
    print("idTr_map (all):")
    for id_tr in idTr_map:
        tr = idTr_map[id_tr]
        printTrWithStats(tr, i)
        # also init features:
        tr.possibly_occluded = None
        tr.possible_next = 0
        i = i+1

    trajectories: list[Trajectory] = idTr_map.values()
    t_global = 1
    # t_global = 805

    SAVE_FEAT = not True
    SAVE_TRAJECTORIES = False
    while t_global <= n:
        print("-> ",t_global) # print time
        # frame = getFrameAtI(t_global, FRAMES_PATH)

        # patches:
        # pca_feat = getOrComputePCAFeaturesAtI(t_global, frame, featureExt, save_feat=SAVE_FEAT)
        pca_feat = getOrComputePCAFeaturesAtI(t_global, None, featureExt, save_feat=SAVE_FEAT)
        pad_l, pad_u = featureExt.last_padder.pad_l, featureExt.last_padder.pad_u # used to align bb properly

        # get all trajectories, that live in this time instant:
        for tr in trajectories:
            if(tr.X[0].t <= t_global and t_global <= tr.X[0].t + len(tr.X)-1):
                # trajectories_that_live_at_this_time.append(tr)
                # current tdet:
                t = t_global-tr.X[0].t # offset time
                tr_det_t:TDet = tr.X[t]
                
                # get features:
                patches = patchesInBB(pca_feat, tr_det_t.bb, xy_off=[pad_l, pad_u])
                patches_rp = roiPool(patches, use_2d=False)

                # update look with new features:
                addToAvgTr(patches_rp.astype(np.float64), tr)
                # draw things
        t_global = t_global +1

    if(SAVE_TRAJECTORIES):
        with open(RESULT_PATH+'trs_w_feat.p', 'wb') as fp:
            abc = [T_OFFSET, t_global-1, copy.deepcopy(idTr_map), np.array([])]
            pickle.dump(abc, fp)


def compute_feat_for_sequence():    
    print("[INFO] this is compute_feat_for_sequence to (pre)compute sequence features")

    # init vars:
    FRAMES_PATH, _, T_OFFSET, _, _, _ = getConfigConsts('got10k14')
    t_global = T_OFFSET
    n = 100    
    # init feature extractor
    featureExt = FeatureExtractor(model_size='base')


    while t_global <= n:
        print("-> ",t_global) # print time
        frame = getFrameAtI(t_global, FRAMES_PATH)
        pca_feat = getOrComputePCAFeaturesAtI(t_global, frame, featureExt, save_feat=True)
        showInNamed("frame",frame)
        showInNamed("pca",pca_feat)
        cv.waitKey(0)
       
        t_global = t_global +1



    
main_di = 4
if __name__ == "__main__":
    # read_tr_file()
    if(main_di == 1):
        main_d1()
    if(main_di == 2):
        main_d2()
    if(main_di == 3):
        get_avg_tr_look()
    if(main_di == 4):
        compute_feat_for_sequence()


