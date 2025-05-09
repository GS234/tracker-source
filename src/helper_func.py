from __future__ import annotations # da delajo tut type hint-i znotraj istega class-a
import numpy as np
import cv2 as cv
import random
import pickle
# from matplotlib import pyplot as plt
import math
import traceback
import os
import glob
from FeatureExtractor import roiPool, patchesInBB, getImagePadding
from Detection import Detection, TDet
from collections import deque # for queue (de - double ended) (solveQBP2)
from configparser import ConfigParser
from functools import cmp_to_key

import sys
sys.path.append('../RAFT/core') # raft stuff
from utils import flow_viz # type: ignore

# to avoid cyclic import; use it only in type checking
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Trajectory import Trajectory

# to read config file:
def simpleNotTrueFalseParser(s:str):
    def simpleNotTrueFalseParserList(s:list[str]):
        if(len(s) == 0): return False
        if(s[0] == 'True'): return True
        if(s[0] == 'False'): return False
        if(s[0] == 'not'): return not simpleNotTrueFalseParserList(s[1:])
        return False
    return simpleNotTrueFalseParserList(s.split(' '))

def getOptionNamespace(config_name, init_file: str = 'options.ini'):
    # print("this is getoptionnamespace")
    init_data = ConfigParser()
    init_data.read(init_file)
    # config_name = 'main2'
    # items = init_data.items(config_name)
    items = {}
    try:
        items = dict(init_data.items(config_name))
        for k in items:
            value = items[k]
            values_split = value.split(' ')
            if('True' in values_split or 'False' in values_split):
                items[k] = simpleNotTrueFalseParser(value)
        # print(items)
    except:
        print("[getOptionNamespace] error while reading file '%s'"%init_file)
    return items

def getOptionNamespaceWdefaultInit(config_name, init_file: str='options.ini'):
    # get default namespace:
    default_ns = getOptionNamespace(config_name=config_name, init_file='defaults.ini')

    # get this namespace:
    this_ns = getOptionNamespace(config_name, init_file)
    for k in this_ns:
        default_ns[k] = this_ns[k]
    return default_ns
# OPTIONS_TR = getOptionNamespaceWdefaultInit('tr', 'options.ini')
# A1 = float(OPTIONS_TR['a1']) # to set default A1 value for getprobiv
A1 = 1.0


def printTrWithStats(t: Trajectory, i_t=None, also_T2k=False, add_to_end = ""):
    s = "t%-5s (t%-5s) o=[%-3d,%-3d], c=%-3d, len=%-4d, score=%9.4f %5d - %-5d, n_merged=%3d (fin:%s)" % \
          (t.not_so_much_unique_id,t.id, t.origin.x[0],t.origin.x[1], t.color[2], len(t.X), t.getScore2(), t.X[0].t, t.X[-1].t, len(t.T2.keys()), str(t.term))
    if(i_t is not None):
        s = "%3d. %s"%(i_t,s)
    
    if(also_T2k):
        joined = ""
        for k in t.T2.keys():
            joined = "%s %s"%(joined,k.id)
        s = "%s [%s]"%(s,joined)
    
    s = "%s %s"%(s,str(add_to_end))
    print(s)

def printTrListWithStatsOrdered(tr_list: list[Trajectory]):
    t_i = 0
    for t in tr_list:
        printTrWithStats(t, t_i)
        t_i = t_i+1


# takes list of tuples, returns list of detections
def coords2detect(X):
    return [Detection(x) for x in X]

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

def detections2map(D, map, color: list = [255,255,255]): # draws detections
    h,w,_ = np.shape(map)
    t_now = len(D) # which time instant is it
    for d in D:
        # x, y = d.x
        y, x = d.x # dimensions are switched! (because of row major order; it is messy [POSSIBLE TODO: think about it])
        # out of bounds is possible, this is the easiest quick fix, should probably be made different
        y = int(y)
        x = int(x)
        x = x%h
        y = y%w
        
        # map[x,y]= 1.0 - (1.0 / (1.0+(d.t/20.0)))*0.9 # more correct, as more recent detections should be brighter
        # map[x,y]= (1.0 / (1.0+(d.t/8))) # I like this more, but is not correct because of ^
        map[x,y]= color

def coords2map(X, map, color: list = [255,255,255], overwrite=True):
    h,w,_ = np.shape(map)
    # print("X: ",X)
    for point in X:
        x, y = point
        x = x%h
        y = y%w
        
        
        if(overwrite or np.sum(map[x,y]) == 0):
            # print(x,y)
            map[x,y] = color



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
        i_i = rev_i*(n-1-(i<<1))+i # branchless, optimized :)

        vec[i_i] = vec[i_i] + c
        if(vec[i_i] == 2):
            vec[i_i] = 0
            c = 1
        else:
            break

# function creates array that has binary representation of integer
def binArrFromInt(n: int, l: int):
    i = 0
    v = np.zeros(l).astype(np.int8)
    while (n != 0 and i < l):
        v[i] = n%2
        n = n>>1
        i = i+1
    return v


# read detections from file and write it to list
def readDetFile(filename: str):
    detections = []
    with open(filename) as fd:
        for line in fd:
            dets = line[0:-1].split(";")
            a = []
            for d in dets:
                d_split = d.split(',')
                if(d_split[0]):
                    # print(d_split)
                    bb = [int(float(i)) for i in d.split(',')]
                # a.append(tuple(d.split(',')))
                a.append(tuple(bb))
            detections.append(a)
            # print()
    return detections

# like ^ (readDetFile), this one returns detections instead of tuples (might need to offset it with +1 -> yes)
def readDetFile2(filename: str, time_off=1):
    detections = []
    with open(filename) as fd:
        frame_i = 0 + time_off
        for line in fd: # line represents frame at i (frame_i) - STARTS WITH 0
            dets = line[0:-1].split(";")
            a = []
            for d in dets:
                d_split = d.split(',')
                # if(d_split[0]):
                if(len(d_split) == 4): # if length is divisible
                    # print(d_split)
                    # bb = [int(float(i)) for i in d.split(',')]
                    # bb = [float(i) for i in d.split(',')] # now it is float
                    bb = [float(i) for i in d_split] # now it is float
                    detection = bbDet2Det(bb, frame_i)
                    a.append(detection)
            detections.append(a)
            # print()
            frame_i = frame_i + 1
    return detections

# merges two detection lists together (assuming same offset ( t(dl1[0]) == t(dl2[0]) ))
def mergeDetLists(dl1: list[list[Detection]],dl2: list[list[Detection]]):
    # assume same offset (both start at same time)
    dmerged = dl1
    other = dl2
    if(len(dl2) > len(dl1)):
        dmerged = dl2
        other = dl1
    i = 0
    while (i < len(dmerged)):
        if(i < len(other)):
            dmerged[i] = dmerged[i] + other[i]
        else: # other list ended, no need to continue
            break
        i = i+1
    return dmerged

def getFileAtI(i, path, ending='.npy'):
    frame_i = f'{i:08}'
    return np.load(path+frame_i+ending)


# OPTICAL FLOW HELPER FUNCTIONS:
def readFlowFile(filename: str):
    with open(filename, 'rb') as fp:
        flow = pickle.load(fp)
        return flow

def flow2img(flow):
    flo = flow_viz.flow_to_image(flow)
    return flo

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

# function is used to get vector's magnitude and angle (in terms of detections: velocity, theta)
def getVecMagAng(vec):
    # principal_ax = np.array([1, 0]) # calculate relative to this (base unit vector, see DetectionSpace.estimateNext2 - same thing)
    mag = np.sqrt(np.dot(vec, vec))
    ang = 0.0
    if(mag != 0):
        cos_theta = vec[0] / mag  # this is it, just trust me bro (it is simplification of --> vec*[1,0] / |[1,0]*|vec||) (calculation relative to principal ax. vector: [1,0])
        ang = np.arccos( cos_theta )
    return mag, ang

# function takes region of optical flow map and calculates displacement vector (median of all pixels within bb)
def getMotionVec(flo_region, mean=False):
    # print(flo_region)
    flo_reg_resh = flo_region.reshape((-1,2))
    vec=[0,0]
    if(not flo_region.any()):
        print("[WARN] getMotionVec: flow region empty, returning None")
        # return np.array(vec)
        return None
    
    if(mean):
        vec = np.mean(flo_reg_resh,axis=0)
        # print("mean: ",vec)
    else:
        vec = np.median(flo_reg_resh,axis=0)
        # print("median: ",vec)
    # max = np.amax(flo_reg_resh, axis=0)
    # print("max: ", max)
    return vec

def getFlowAtI(i, path):
    return getFileAtI(i, path)

def getFlowToI(i, path):
    try:
        i_1 = i-1
        flow_to = getFlowAtI(i_1, path)
        return flow_to
    except FileNotFoundError:
        print("[WARN] File not found. Optical flow might not exist at i: %d. Returning blank flow."%(i_1))
        return np.zeros(np.shape(getFlowAtI(i,path)))
    except Exception:
        print("[WARN] Flow to frame at i: ",i," could not be determined due to unknown reason. More info:")
        print(traceback.format_exc())
        return np.zeros(np.shape(getFlowAtI(i,path)))
    
# function returns flow into and outta current frame
def getFlowToFromAtI(i, path):
    flow_from = getFlowAtI(i, path)
    flow_to = getFlowToI(i, path) # return empty if next part throws error
    # flow_to = np.zeros(np.shape(flow_from)) # return empty if next part throws error
    # try:
    #     flow_to = getFlowAtI(i-1, path)
    # except FileNotFoundError:
    #     print("[WARN] File not found. Optical flow might not exist at i: ", i)
    # except Exception:
    #     print("[WARN] Flow to frame at i: ",i," could not be determined due to unknown reason. More info:")
    #     print(traceback.format_exc())
    return flow_to, flow_from



# function crops region defined by detection's bb and calculates motion vector for it
def getDetMotionVector(d: Detection, flow_map):
    flow_region = getRect(d, flow_map)
    motion_vec = getMotionVec(flow_region)
    return motion_vec

def flowVec2Det(dets: list[list[Detection]], path: str, begin:int = 0, end: int = -1, n_bins=8):
    if(end < begin):
        begin=0
        end=-1
    if(end == -1):
        end = len(dets)
    
    i = begin
    while(i < end):
        # get frame at i
        flow = getFlowAtI(i, path)
        # for each detection calculate color hist
        for d in dets[i]:
            motion_vec = getDetMotionVector(d=d, flow_map=flow)        
            d.flow_vector=motion_vec
            d.has_flow_vector=True
        i = i+1
    
def updateDetMotionVecFromFlowMap(det: Detection, flow_map):
    flow_bb = getRect(det, flow_map)
    det.flow_vector = getMotionVec(flow_bb)
    det.has_flow_vector = True

# function deletes flow files (.npy - that of numpy.save) from directory (WARNING: USE WISELY)
def deletePrecomputed(path: str, confirm=True):
    if(confirm):
        answer = input("[WARN] you are about to delete precomputed files from '%s'. Do you want to continue? Y - yes, (everything else) - no: "%(path))
        if(answer != 'Y'):
            print("[INFO] abort delete")
            return

    files = glob.glob(path+"*.npy")
    for f in files:
        os.remove(f)

# ------------------------------------

# feature-related functions:
def getFeaturesAtI(i:int, path:str):
    return getFileAtI(i, path)
# --------------------------

def getDistBetweenDets(d1:Detection, d2:Detection):
    diff = d1.x - d2.x
    return np.sqrt(np.dot(diff, diff))


# function returns rectangular region around detection (crop bounding box) (wrapper function)
def getRect(det:Detection, arr):
    return getRectBb(det.bb, arr)

# in some places we only have bounding box
def getRectBb(bb: tuple, arr):
    y, x, h, w = bb
    y,x,h,w = int(y), int(x), int(h), int(w) # we need int to index array
    bb_area = arr[x:x+w+1,y:y+h+1] # bounding box image
    return bb_area
    

# redundant: throw out
# function adds color histograms to detections (separate function for convenience (frames have different path))
def colorHist2Det(dets: list[list[Detection]], path: str, begin:int = 0, end: int = -1, n_bins=8):
    if(end < begin):
        begin=0
        end=-1
    if(end == -1):
        end = len(dets)
    
    i = begin
    while(i < end):
        # get frame at i
        # frame = getFrameAtI(i+1, path) # tezave: +1
        frame = getFrameAtI(i, path)
        # cv.imshow("debug", frame)
        # for each detection calculate color hist
        for d in dets[i]:
            updateDetColorHistFromFrame(d, frame, n_bins=n_bins)
            # cv.imshow("c", frame_bb)
            # cv.waitKey(0)
            # showHists([d.color_hist])
        # cv.waitKey(0)
        i = i+1

# function: crop detection from frame, calculate color histogram
# this is separated from colorHist2Det because is also used elsewhere (in main, might also in some place else)
def updateDetColorHistFromFrame(det: Detection, frame, n_bins=8):
    frame_bb = getRect(det, frame)
    det.color_hist = getColorHist(frame_bb,n_bins=n_bins)
    det.hasHist = True


# function calculates color histogram from current frame's detecitons
def getColorHist(frame_bb, n_bins=8):
    color_hist = np.zeros((n_bins, n_bins, n_bins))
    # 1. put each pixel to its corresponding bin
    for row in frame_bb: # row-major? should be
        for pixel in row:
            # 1.1 pixel has rgb; calculate appropriate bins (for each color) and increase cell value
            px = np.array(pixel)
            px = (px / 255) * (n_bins-1)
            px = px.astype(np.uint8)
            # print(pixel, ", ",px)
            c1, c2, c3 = px[0],px[1],px[2] # not that important which color is which
            # 1.2 increase value of correct bin
            color_hist[c1][c2][c3] += 1
    # 2. normalize histogram
    color_hist = color_hist / np.sum(color_hist)
    return color_hist

# function calculates bhattacharyya coefficient for 2 histograms
def compareHists(a, A):
    return np.sum(np.sqrt(a*A)) # bhattacharyya distance
# ------------------------------------------------


# funciton reads trajectories from file and returns array of trajectories
def readTrajectoryFile(filename: str):
    tr_list = []
    with open(filename, 'rb') as fp: # fp: file pointer?
        tr_list = pickle.load(fp)
    
    # need to fix detections from some files:
    # t0 = tr_list[0:4]
    # tr_list = tr_list[4:]
    # tr_list.insert(0, t0)
    return tr_list


# duck test: if it looks like a duck and if it quacks like a duck, then it is probably a duck
def trDuckTest(tr1:Trajectory, tr2:Trajectory, also_check_origin=True):
    i_ending_x = list(tr1.X[-1].x)
    j_ending_x = list(tr2.X[-1].x)

    i_nmerged = len(tr1.T2.keys())
    j_nmerged = len(tr2.T2.keys())

    if(also_check_origin):
        i_origin_x = list(tr1.origin.x)
        j_origin_x = list(tr2.origin.x)

        return (i_origin_x == j_origin_x and i_ending_x == j_ending_x and i_nmerged == j_nmerged)
    
    return (i_ending_x == j_ending_x and i_nmerged == j_nmerged)



# function drops redundant trajectories from list (those that are same (duplicated))
# use_duck_test: removes even more trajectories
# duck test (variation): if starts in same point, if it has same amount of connected trajectories and ends in same point, then it is probably same tr :)
def dropRedundant(tr_list:list[Trajectory], red_level=0):
    also_check_origin=True
    use_duck_test=False
    if(red_level >= 1):
        use_duck_test=True
    if(red_level >= 2):
        also_check_origin=False
    new_l = []
    red_count = 0
    for i in range(len(tr_list)):
        tr_i = tr_list[i]
        already_contained = False
        for j in range(i):
            tr_j = tr_list[j]
            if(i == j):
                continue
            if(tr_i.T2.keys() == tr_j.T2.keys()):
                already_contained = True
                red_count = red_count+1
                break
            if(use_duck_test):
                # check if starts with same tr, ends with same tr, has same number of trs
                if(trDuckTest(tr_i, tr_j, also_check_origin=also_check_origin)):
                    print(tr_i, " is 'same' as ",tr_j)
                    red_count = red_count+1
                    already_contained = True
                    break
                
        if(not already_contained):
            new_l.append(tr_list[i])
    print("[dropRed] dropped %d redundant trajectories."%red_count)
    return new_l

def isRedundant(tr_i:Trajectory, tr_list:list[Trajectory], red_level=0):
    also_check_origin=True
    use_duck_test=False
    if(red_level >= 1):
        use_duck_test=True
    if(red_level >= 2):
        also_check_origin=False
    
    already_contained = False
    for tr_j in tr_list:
        if(tr_i.T2.keys() == tr_j.T2.keys()):
            already_contained = True
            red_count = red_count+1
            break
        if(use_duck_test):
            # check if starts with same tr, ends with same tr, has same number of trs
            if(trDuckTest(tr_i, tr_j, also_check_origin=also_check_origin)):
                print(tr_i, " is 'same' as ",tr_j)
                red_count = red_count+1
                already_contained = True
                break
    return already_contained

# bounding box deteciton to Detection - generates Detection object with coordinates of center of a bounding box
def bbDet2Det(bb: list[float], t: int = 0):
    # x, y = bb[0]+bb[2]/2,bb[1]+bb[3]/2
    x,y = getBBCenter(bb)
    return Detection([x,y], t=t, bb=bb)

# function reads i-th frame
def getFrameAtI(i: int, path: str, toBGR=False):
    frame = cv.imread(path+str(i2frameI(i))+".jpg")
    if(toBGR):
        return cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    else:
        return frame

# returns i to frame i format (i -> 0000000i)
def i2frameI(i:int):
    return f'{i:08}'

# def showHists(hists:list, c=1):
#     color = ['b','g','r']
#     # show only in one dimension
#     _,ax = plt.subplots(1,len(hists))
#     # print(ax)
#     for j in range(len(hists)):
#         hist = hists[j]
#         arr = np.zeros((np.shape(hist)[0],3))
        
        
#         for i in range(len(arr)):
#             arr[i,0] = np.sum(hist[i,:,:]) # c1
#             arr[i,1] = np.sum(hist[:,i,:]) # c2
#             arr[i,2] = np.sum(hist[:,:,i]) # c3
#         x = np.arange(len(arr))
        
        
#         if(len(hists) == 1):
#             ax.bar(x,arr[:,c], color=color[c%3])
#         else:
#             ax[j].bar(x,arr[:,c], color=color[c%3])
#     # print(arr)
    
#     # plt.bar(x,arr)
#     plt.show()


def getTrColor():
    return tr_colors[int(random.random()*len(tr_colors))]

# drawing functions (previously in DetectionSpace):

# method draws x instead of .
def drawX(image, c:list, color=[255,255,255]) -> None:
    drawShape(shapes['x'], image,c,color )

def drawO(image, c:list, color=[255,255,255]) -> None:
    drawShape(shapes['o'], image,c,color )

def drawDot(image, c:list, color=[255,255,255]) -> None:
    drawShape(shapes['.'],image,c,color)
    

def drawShape(shape, image, c:list, color=[255,255,255]) -> None:
    shape = shape + np.array(c).astype(np.int32) # add origin
    p_list = [(x[1], x[0]) for x in shape]
    coords2map(p_list, image, color=color)


def drawLine(image, x1, x2, color=[0,0,255]):
    x1 = np.array(x1)
    x2 = np.array(x2)
    n = (x2 - x1).astype(np.int32) # normal from x1 to x2
    # print(n.reshape( (2,1) ))
    n_len = np.sqrt(np.dot(n.T,n))
    if(n_len == 0): # if the same point, no need to draw :)
        # coords2map([(int(x1[1]),int(x1[0]))], image, color=color, overwrite=True)
        drawDot(image, x1, color=color)
        return
    n = n/n_len
    n = n.reshape((2,1))
    
    values = np.arange(0, int(n_len), 0.1)
    values = values.reshape((1,len(values)))
    
    points = np.dot(n, values) + x1.reshape((2,1))
    # points = points.astype(np.int32)
    p_list = [(int(x[1]), int(x[0])) for x in points.T]
    coords2map(p_list, image, color=color, overwrite=True)

def drawLine2(image, x1, x2, color=[0,0,255], shape='.'):
    shape_l = getShape(shape)
    x1 = np.array(x1)
    x2 = np.array(x2)
    n = (x2 - x1).astype(np.int32) # normal from x1 to x2
    # print(n.reshape( (2,1) ))
    n_len = np.sqrt(np.dot(n.T,n))
    if(n_len == 0): # if the same point, no need to draw :)
        # coords2map([(int(x1[1]),int(x1[0]))], image, color=color, overwrite=True)
        drawDot(image, x1, color=color)
        return
    n = n/n_len
    n = n.reshape((2,1))
    
    values = np.arange(0, int(n_len), 0.1)
    values = values.reshape((1,len(values)))
    
    points = np.dot(n, values) + x1.reshape((2,1))
    # points = points.astype(np.int32)
    
    for x in points.T:
        # c = (int(x[1]), int(x[0]))
        drawShape(shape=shape_l, image=image, c=x, color=color)

# method draws bounding box in detection space
def drawBoundingBox(image, bb: tuple, color: list = [0,255,0]) -> None:
    # print(W, H)
    # 1. starting coordinate:
    y, x, h, w = bb
    y, x, h, w = int(y), int(x), int(h), int(w)
    points = []

    # 2. draw horizontally:
    for i in range(w):
        # x_i, y1_i, y2_i = x+i, y, y+h

        points.append((x+i, y))
        points.append((x+i+1, y+h))

    # 3. draw vertically:
    for i in range(h):
        points.append((x, y+i+1))
        points.append((x+w, y+i))
    coords2map(points, image, color=color, overwrite=True)

# function shows image in named window with name name
def showInNamed(name, image):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, image)

# function adds element to average, if it is 
def addToAvg(a_avg, a_i, i) -> float:
    return ( a_avg + (a_i/i) )*( i/(i+1) )

# more general version of ^ (can be used to add 2 averages and get average as if it were whole)
def add2Avg(a1, a2, a_i1, a_i2):
    return (a1 + (a2*a_i2)/a_i1) * (a_i1 / (a_i1+a_i2))

# addToAvg wrapper: calls it and modifies trajectory properties
def addToAvgTr(a_i, tr:Trajectory):
    if(tr.visual_avg is None):
        tr.visual_avg = a_i
    else:
        new_a = addToAvg(tr.visual_avg, a_i, tr.visual_n)
        tr.visual_avg = new_a
    tr.visual_n = tr.visual_n+1

# add2Avg wrapper: calls it and modifies trajectory properties
def add2AvgTr(tr1:Trajectory, tr2:Trajectory):
    if(tr1.visual_avg is None):
        return (tr2.visual_avg, tr2.visual_n)
    if(tr2.visual_avg is None):
        return (tr1.visual_avg, tr1.visual_n)
    
    new_a = add2Avg(tr1.visual_avg, tr2.visual_avg, tr1.visual_n, tr2.visual_n)
    new_ai = tr1.visual_n + tr2.visual_n
    return (new_a, new_ai)

# function calculates visual distance:
def getTrVisualSimilarity(tr1:Trajectory, tr2:Trajectory):
    return getVisualSimilarity2(tr1.visual_avg, tr2.visual_avg)

def getVisualSimilarity(features1, features2):
    k = 100
    v1 = features1.reshape(-1)
    v2 = features2.reshape(-1)
    # print(v1)
    dist = getVecDist(v1,v2)
    dist_k = dist/k
    return np.exp(-dist_k)

# similar to ^, but compares as colors (avg feature values)
def getVisualSimilarity2(features1, features2):
    # print("<getVisualSimilarity2>")
    k = 100
    # print("f1, 2: ")
    # print(features1)
    # print(np.mean(features1, axis=2))
    # print(features2)
    # v1 = features1.reshape(-1)
    # v2 = features2.reshape(-1)
    v1 = np.mean(features1, axis=2).reshape(-1)
    v2 = np.mean(features2, axis=2).reshape(-1)
    # print(v1, v2)
    # print(v1)
    dist = getVecDist(v1,v2)
    dist_k = dist/k
    return np.exp(-dist_k)

def getVecDist(v1, v2):
    diff = v2-v1
    dist = np.sqrt(np.dot(diff, diff))
    return dist

# function calculates score of next detection (d) based on current estimated detection (td)
# getProbIV -> Iou + Visual
# prob = a*IoU + (1-a)*feat_sim
def getProbIV(d:Detection, td:Detection, a=A1) -> float:
    return getProbIVBF(d.bb, d.visual_feat, td.bb, td.visual_feat, a=a)

def getProbIVBF(bb1,f1,bb2,f2, a=A1) -> float:
    # print("a is: %.2f"%a)
    iou_prob = IoUbb(bb1, bb2)
    visual_prob = 1

    if(f1 is not None and f2 is not None):
        visual_prob = getVisualSimilarity2(f1, f2)
        # visual_prob = getVisualSimilarity(f1, f2)
        # print("[getprobIV] (%.2f,%.2f) - (%.2f,%.2f) visual similarity score: %6.2f"%(bb1[0],bb1[1], bb2[0],bb2[1], visual_prob))
    elif(not math.isclose(1,a)):
        print("[warn] visual score could not be calculated. Assuming 1 (totally similar)")
    # visual_prob = getVisualSimilarity(feat1, feat2)
    ret_val = a*iou_prob + (1-a)*visual_prob
    return ret_val

# function is actually very extend4 specific: needs to have feature_map_and_padding with as tuple -> (feature_map, pad_l, pad_u)
def getFeaturesFromFeatureMapAndPadding(det:Detection, feature_map_and_featExt):
    pca_feat, featureExt = feature_map_and_featExt
    pad_l, pad_u = featureExt.last_padder.pad_l, featureExt.last_padder.pad_u # used to align bb properly
    # get features:
    patches = patchesInBB(pca_feat, det.bb, xy_off=[pad_l, pad_u])
    patches_rp = roiPool(patches, use_2d=False)
    return patches_rp.astype(np.float64)

def getFeaturesFromFeatureMapAndPadding2(det:Detection, feature_map_and_featExt):
    pca_feat, featureExt = feature_map_and_featExt
    pad_l, pad_u = featureExt.last_padder.pad_l, featureExt.last_padder.pad_u # used to align bb properly
    # get features:
    patches = patchesInBB(pca_feat, det.bb, xy_off=[pad_l, pad_u])
    return patches

def getFeaturesFromFeatureMapAndPadding3(det:Detection, feature_map_and_padding, patch_size=14, pooled=True):
    pca_feat = feature_map_and_padding[0]
    pad_l, _, pad_u, _ = feature_map_and_padding[1]
    # get features:
    patches = patchesInBB(pca_feat, det.bb, xy_off=[pad_l, pad_u], patch_size=patch_size)
    if(pooled):
        patches_rp = roiPool(patches, use_2d=False)
        return patches_rp.astype(np.float64)
    return patches

def bb2str(bb):
    return "%6.2f,%6.2f,%6.2f,%6.2f"%(bb[0],bb[1],bb[2],bb[3])

# function resizes bounding box (if off=1, then new bb is outline of old one)
def bbResize(bb: list[int], ext_bb_off: int = 1):
    ext_bb = list(np.array(bb[0:2])+ext_bb_off)+list(np.array(bb[2:])-(ext_bb_off*2))
    return ext_bb

def getBBCenter(bb: list[int]):
    bb = np.array(bb)
    return bb[0:2] + bb[2:]/2


# gets intersection bounds of bounding boxes
def getBBIntersectionBounds(d1: Detection, d2: Detection):
    return getBBIntersectionBoundsBB(d1.bb, d2.bb)

def getBBIntersectionBoundsBB(bb1, bb2):
    tl_a = np.array(bb1[0:2])
    tl_b = np.array(bb2[0:2])

    br_a = tl_a + np.array(bb1[2:])
    br_b = tl_b + np.array(bb2[2:])

    xA = max(tl_a[0], tl_b[0])
    yA = max(tl_a[1], tl_b[1])
    xB = min(br_a[0], br_b[0])
    yB = min(br_a[1], br_b[1])
    
    I_x = xB - xA
    I_y = yB - yA
    return (I_x, I_y)

# determines if trajectories are overlapping (order of trs is important)
def isOverlapValid(first: Trajectory, second:Trajectory, overlap_offset=-1):
    a1, b1 = first.X[0].t, first.X[-1].t
    a2, b2 = second.X[0].t, second.X[-1].t
    return (a2 > a1 and a2 < b1 and b2 > b1 and (overlap_offset < 0 or (b1-a2 <= overlap_offset)))



# calculates iou
def IoU(d1:Detection, d2: Detection):
    return IoUbb(d1.bb, d2.bb)

def IoUbb(bb1, bb2):
    # determine the (x, y)-coordinates of the intersection rectangle
    I_x, I_y = getBBIntersectionBoundsBB(bb1,bb2)    
    
    # compute the area of intersection rectangle
    iou = 0.0
    if((I_x > 0) and (I_y > 0)):
        I = I_x*I_y
        U = np.prod(bb1[2:]) + np.prod(bb2[2:]) - I
        iou =  I/U

    return iou


# some kind of iou, but for vectors
def vecScore(a,b, l=0.01):
    score = 1.0
    a = np.array(a)
    b = np.array(b)
    c = -b+a
    # get distance of vectors:
    a_d = np.sqrt(np.dot(a,a))
    b_d = np.sqrt(np.dot(b,b))
    c_d = np.sqrt(np.dot(c,c))
    # score = score - (c_d / (a_d+b_d))
    # print(a_d,b_d)
    return np.exp(-l*(a_d+b_d))

def getSelected(v: list[int], trs: list[Trajectory]):
    return [t[1] for t in zip(v, trs) if t[0] == 1]

# QBP stuff: build matrix (previously in dspace)
# method builds trajectory interatction matrix
# type 1: detection-wise, type 2: trajectory-wise
def buildQBPMatrixX(tr_list: list[Trajectory], type=1):
    # 1. calculate q_ii terms ("merit terms")
    Q_ii = [] # list of q_ii (trajectory scores, "merit terms")
    for tr in tr_list:
        # q_ii = tr.getScore2()
        q_ii = tr.getScoreX(type=type)
        Q_ii.append(q_ii)
    
    Q = np.diag(Q_ii) # make diagonal matrix
    
    # 2. calculate q_ij terms (interaction terms (similar to q_ii, but only consider intersecting trajectory points))
    n_tr = len(Q_ii)
    m = 0 # row index
    n = 0 # column index

    # I miss good old for loops from java so much ...
    iii = 0
    while( m <= (n_tr-1)):
        n = m+1 # calculate only terms above diagonal, because Q is symmetric (Q[i,j] = Q[j,i])
        while( n <= (n_tr -1)):
            # 1. calculate interaction cost (points that are in intersection of both hypotheses)
            # q_ij = tr_list[m].getInteractionCost2(tr_list[n])
            q_ij = tr_list[m].getInteractionCostX(tr_list[n], type=type)
            # print("t%-5d - t%-5d: %-6.2f" % (tr_list[m].id, tr_list[n].id, q_ij))
            # 2. set q_ij term (q_ij, q_ji)
            Q[m,n] = q_ij
            Q[n,m] = q_ij
            n = n+1
            iii+=1
        m = m+1
    return Q

# qbp solvers:
# method solves qbp (returns list of selected hypotheses)
def solveQBP(Q):
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
def solveQBP2(Q: np.array, debug=False):
    h,w = np.shape(Q)
    print("solving QBP (%dx%d)"%(h,w))
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
    
    if(debug):
        print("n_iter: " + str(n_iter), " n_combinations: ", (1 << n_el)) # some stats
    # return (v_max, D_max)
    return (local_max_v, local_max_d)

# solveQBP3
def solveQBP3(Q):
    n,_ = np.shape(Q)
    # print(Q)

    # start without any motions
    m = 0
    m_max = 0
    V = np.zeros((1,n)).astype(np.uint8)
    max_v = V[0]
    
    while(V.size != 0):
        V = allNextCombsFromPrev(V)
        res = np.dot(np.dot(V, Q), V.T)
        results = np.diag(res)

        mask = np.where(results <= m, False, True)

        results=results[mask]
        V = V[mask]
        if(V.size == 0):
            break

        min_res = np.min(results)
        m = min_res # update previous min
        max_idx = np.argmax(results)
        max_res = results[max_idx]
        if(max_res < m_max):
            break
        m_max = max_res
        max_v = V[max_idx]
    # print(max_v)
    return (max_v.astype(np.uint8), m_max)

def allNextCombsFromPrev(prev):
    first = True
    next_r = []
    for arr in prev:
        ones = np.sum(arr)
        next_combs = allNextCombs(arr)
        if(first):
            next_r = next_combs
            first = False
        else:
            mask = np.dot(next_r, next_combs.T)
            mask = np.where(mask == ones, 0, 1)
            mask = np.prod(mask, axis=0)
            mask = mask.astype(np.bool_)
            next_r = np.vstack((next_r, next_combs[mask]))
    return next_r

def allNextCombs(bin_arr):
    n = len(bin_arr)
    ones = np.sum(bin_arr) # get number of ones
    zeros = np.uint64(n-ones)

    # create matrix
    ret_m = np.zeros((zeros, n))
    # print(ret_m)
    j = 0
    for i in range(n):
        if(bin_arr[i] == 1):
            ret_m[:,i] = 1
        else:
            ret_m[j, i] = 1
            j = j+1
    return ret_m
# ---------

# Q matrix analyze (moved from main2)
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
def analyzeTrsWithQ(tr_list: list[Trajectory], type=1, max_n=5):
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


class TrGroup:
    def __init__(self, tr_set: set):
        self.trs: set = tr_set
        self.root: TrGroup = None # has no root
        
    
    def isInGroup(self, tr):
        return tr in self.trs
    
    def getRoot(self) -> TrGroup:
        a = self
        i = 0
        while a.root is not None:
            # print(a.root)
            a = a.root
            i = i+1
            # if(i > 20):
            #     break
        return a
    
    def addSet(self, to_add):
        self.trs.update(to_add)
    
    def __str__(self):
        return str(self.trs)
    
    def __repr__(self):
        return self.__str__()
    
    def toList(self):
        return list(self.trs)

        
# function finds independent subsets (groups) of trajectories
def getGroupsFromQ(Q):
    # print("this is get groups from Q:")
    # print(Q)
    n, _ = np.shape(Q)
    Q2 = np.where(np.isclose(Q, 0.0), 0, 1).astype(np.uint8)
    # print(n)
    trs: list[int] = list(range(0,n))
    nodes: list[TrGroup] = []
    for i in range(n):
        tr = trs[i]
        new_node = TrGroup({tr})
        nodes.append(new_node)
    # for nod in nodes:
    #     print(nod, nod.getRoot())
    # return
    for i in range(n):
        for j in range(i+1):
            if(i==j):
                continue
            else:
                connection_val = Q2[i,j]
                if(connection_val == 1): # we have connection, see which tr is connected to
                    tr_from = nodes[i]
                    tr_to = nodes[j]

                    if(tr_from.getRoot() == tr_to.getRoot()):
                        pass
                        # print("same root, skipping")
                        # tr_from.root = tr_from.getRoot()
                        # tr_to.root = tr_to.getRoot()
                    else:
                        # create new group
                        new_group = TrGroup(tr_from.getRoot().trs.copy())
                        new_group.addSet(tr_to.getRoot().trs)
                        # print("%s (%s) + %s (%s) = %s"%(tr_from, tr_from.getRoot(), tr_to, tr_to.getRoot(), new_group))
                        tr_from.getRoot().root = new_group
                        tr_to.getRoot().root = new_group

    ret_val = []
    all_roots:set[TrGroup] = set({})
    # maybe (most likely) there is better way of doing this
    for g in nodes:
        all_roots.add(g.getRoot())
    for r in all_roots:
        ret_val.append(r.toList())
    
    return ret_val

# function does multiple things:
# 1. builds qbp
# 2. finds closed groups (independent from one another - there are no interactions between them)
# 3. solves qbp for each group (sorts it first)
# 4. returns result in same format as qbp solver (v, score)
def buildGroupSolve(tr_list: list[Trajectory], type=1):
    print("[buildGroupSolve] solving with groups")
    if(not tr_list): return (np.array([]).astype(np.uint8), 0.0)
    Q = buildQBPMatrixX(tr_list, type=type)
    n, _ = np.shape(Q)
    print("[buildGroupSolve] og Q (%dx%d):"%(n,n))
    print(Q)
    groups = getGroupsFromQ(Q)
    return_v: list[int] = np.zeros(len(tr_list)).astype(np.uint8)
    return_score: float = 0.0
    
    # deli in vladaj
    g_i = 1
    for g in groups:
        print("group %d %s:"%(g_i, g))
        selected_trs: list[Trajectory] = []
        for tr_i in g:
            selected_trs.append(tr_list[tr_i])
        
        g_trs = list(zip(g, selected_trs))

        # NEED TO SORT G ALSO (with same permutation)
        if(type == 1):
            g_trs.sort(key=lambda x: x[1].getScore2(), reverse=True)
        elif(type == 2):
            g_trs.sort(key=lambda x: x[1].getScoreII(), reverse=True)
        g, selected_trs = zip(*g_trs)
        g = list(g)
        selected_trs = list(selected_trs)
        
        Q = buildQBPMatrixX(selected_trs, type=type)
        print(Q)
        print("solving Q ...")
        v = solveQBP2(Q)
        return_score = return_score+v[1]
        print(v)
        selected = getSelected(v[0], g)
        for s_i in selected:
            return_v[s_i] = 1
        # print(selected)
        g_i = g_i + 1
    return (return_v, return_score)


# --------------------

    

tr_colors=[
    (158,98,64),
    (222,164,126),
    (205,70,49),
    (248,242,220),
    (129,173,200),
    (129,173,200),
    (164,175,105),
    (211,82,105),
    (242,255,73),
    (255,66,66),
    (251,98,246),
    (100,93,215),
    (179,255,252),
    (138,162,158),
    (61,84,103),
    (104,105,99),
    (128,207,169),
    (15,3,38),
    (181,217,156),
    (177,204,116)
]

shapes = {
    'o': np.array([
        [-1,-2],
        [0,-2],
        [1,-2],
        
        
        [-1,2],
        [0,2],
        [1,2],
        
        
        [-2,-1],
        [2,-1],
        [-2,0],
        [2,0],
        [-2,1],
        [2,1],
    ]),
    'x': np.array([
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
    ]),
    '.': np.array([
        [0,0]
    ]),
    'full_o': np.array([
        [-1,-2],
        [0,-2],
        [1,-2],
        [-1,-1],
        [0,-1],
        [1,-1],
        
        
        [-1,2],
        [0,2],
        [1,2],
        [-1,1],
        [0,1],
        [1,1],
        
        
        [-2,-1],
        [-2,0],
        [2,-1],
        [2,0],
        [-2,1],
        [2,1],
        [-1,0],
        [1,0],
    ])
}

# can be used with drawShape
def getShape(s):
    return shapes[s]



def mergeLists(l1: list[TDet], l2: list[TDet]):
    list_merged: list[TDet] = []
    if(l1 and l2):
        l1_l = len(l1)
        l2_l = len(l2)
        l1_i, l2_i = 0,0
        while(l1_i < l1_l or l2_i < l2_l):
            if(l1_i < l1_l and l2_i<l2_l):
                a = l1[l1_i]
                b = l2[l2_i]

                if(b.t <= a.t):
                    list_merged.append(b)
                    l2_i = l2_i+1
                    l1_i = l1_i+1
                else:
                    list_merged.append(a)
                    l1_i = l1_i+1
            elif(l1_i == l1_l):
                b = l2[l2_i]
                list_merged.append(b)
                l2_i = l2_i+1
            elif(l2_i == l2_l):
                a = l1[l1_i]
                list_merged.append(a)
                l1_i = l1_i+1
    return list_merged

# file output:
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
                
                if(i-1 == 0): # if we are writing first one
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

# function creates results.txt file from path (path is list of trajectories) - writes bounding boxes into file
def path2File2(path: list[Trajectory], n:int, results_path="./", filename="results.txt"):
    t_off = 1 # assume T_OFFSET as 1
    path_list = path2list(path, n, t_off)
    if(path_list):
        # print(path[0].X)
        with open(results_path+filename, 'w') as fp: # fp: file pointer?
            for i in range(len(path_list)):
                td_at_i = path_list[i]
                if(i == 0):
                    fp.write("1\n")
                else:
                    if(td_at_i.bb[3]==0):
                        print(td_at_i.bb)
                    fp.write(bb2str(td_at_i.bb)+"\n")
    else:
        print("[WARN] <path2File2> path list is empty, file not created")

# function merges list of trajectories to list of detections, which can then be written to file
def path2list(tr_list: list[Trajectory], n_all:int, t_offset:int):
    id_idx = {} # map: tr id -> X index
    path_x = {} # also map: better idea
    t_global = t_offset
    path_x_id = {} # debug map: t_global -> tr_id
    while (t_global <= n_all):
        path_x[t_global] = None # init it
        path_x_id[t_global] = None # init it
        # print("t: ", t_global)
        # print("trajectories at this time: ")

        for tr in tr_list:
            # if(tr.X[0].t <= t_global and t_global <= tr.X[0].t + len(tr.X)-1):
            if(tr.X[0].t <= t_global and t_global <= tr.X[-1].t):
                # print(tr)
                # every trajectory has its own index (at which point it is currently)
                tr_id_idx = 0
                if(not tr.id in id_idx):
                    id_idx[tr.id] = 0
                else:
                    tr_id_idx = id_idx[tr.id]
                
                if(tr.X[tr_id_idx].t < t_global):
                    while(tr.X[tr_id_idx].t < t_global):
                        tr_id_idx = tr_id_idx+1
                    # update possibly changed tr_id_idx
                    id_idx[tr.id] = tr_id_idx
                tr_td = tr.X[tr_id_idx]
                if(tr_td.t == t_global):
                    # we have point that is at this time, add it:
                    path_x[t_global] = tr.X[tr_id_idx]
                    path_x_id[t_global] = tr.id
                    # print("%d: adding point from %d"%(t_global, tr.id))

        # if there is no trajectory at this time, then add previous detection
        t_global = t_global + 1
        # input("continue?")
    # print(path_x_id)

    first_non_null_encounter = False
    last_non_null_td = None
    for i in range(len(path_x)):
        t_global = t_offset+i

        td_at_i = path_x[t_global]
        if(td_at_i is not None):
            if(not first_non_null_encounter):
                first_non_null_encounter = True
                # we found first non null element, fill back:
                for j in range(i):
                    t_local = t_offset+j
                    path_x[t_local] = TDet(td_at_i.x, t_local)
                    path_x[t_local].bb = td_at_i.bb
            last_non_null_td = td_at_i
        else:
            if(last_non_null_td is not None):
                # copy it, increase time
                path_x[t_global] = TDet(last_non_null_td.x,t_global)
                path_x[t_global].bb = last_non_null_td.bb

    # print("path:")
    path_list = []
    if(first_non_null_encounter):
        for i in path_x:
            path_list.append(path_x[i])
    return path_list
# -------------------------

# function breaks single list of trajectories into multiple smaller ones
def groupTrs(tr: list[Trajectory], timew=200):
    print("this is groupTrs:")
    gropus: list[list[Trajectory]] = []

    # comparator
    def endStartTimeComparator(tr1: Trajectory, tr2: Trajectory):
        t1_start = tr1.X[0].t
        t1_end = tr1.X[-1].t
        
        t2_start = tr2.X[0].t
        t2_end = tr2.X[-1].t
        
        t12e_d =  t1_end - t2_end
        t12s_d =  t1_start - t2_start
        # if(t12s_d == 0):
            # return t12e_d
        # return t12e_d
        if(t12s_d == 0):
            return t12e_d
        return t12s_d

    tr_sorted = sorted(tr, key=cmp_to_key(endStartTimeComparator))
    # print("sorted:")
    # printTrListWithStatsOrdered(tr)
    end_t = tr_sorted[-1].X[-1].t
    time_w = timew
    edge = 0
    used_trs: set[Trajectory] = set()
    i = 0
    while(edge < end_t):
        # print("group %d (edge: %d):"%(i, edge+time_w))
        group = []
        for t in tr_sorted:
            if(t.X[-1].t <= (edge+time_w) and t not in used_trs):
                # printTrWithStats(t)
                group.append(t)
                used_trs.add(t)
        gropus.append(group)
        # print()
        i = i+1
        
        
        edge = edge + time_w
    return gropus
    

if __name__ == '__main__':
    pass
    

