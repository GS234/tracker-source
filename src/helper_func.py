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
from FeatureExtractor import roiPool, patchesInBB
from Detection import Detection

import sys
sys.path.append('../RAFT/core') # raft stuff
from utils import flow_viz

# to avoid cyclic import; use it only in type checking
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Trajectory import Trajectory


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

# bounding box deteciton to Detection - generates Detection object with coordinates of center of a bounding box
def bbDet2Det(bb: tuple, t: int = 0):
    x, y = bb[0]+bb[2]/2,bb[1]+bb[3]/2
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
        coords2map([(int(x1[1]),int(x1[0]))], image, color=color, overwrite=True)
        return
    n = n/n_len
    n = n.reshape((2,1))
    
    values = np.arange(0, int(n_len), 0.1)
    values = values.reshape((1,len(values)))
    
    points = np.dot(n, values) + x1.reshape((2,1))
    # points = points.astype(np.int32)
    p_list = [(int(x[1]), int(x[0])) for x in points.T]
    coords2map(p_list, image, color=color, overwrite=True)

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
    return getVisualSimilarity(tr1.visual_avg, tr2.visual_avg)

def getVisualSimilarity(features1, features2):
    k = 100
    v1 = features1.reshape(-1)
    v2 = features2.reshape(-1)
    # print(v1)
    dist = getVecDist(v1,v2)
    dist_k = dist/k
    return np.exp(-dist_k)

def getVecDist(v1, v2):
    diff = v2-v1
    dist = np.sqrt(np.dot(diff, diff))
    return dist



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

def bb2str(bb):
    return "%6.2f,%6.2f,%6.2f,%6.2f"%(bb[0],bb[1],bb[2],bb[3])

# gets intersection bounds of bounding boxes
def getBBIntersectionBounds(d1: Detection, d2: Detection):
    tl_a = np.array(d1.bb[0:2])
    tl_b = np.array(d2.bb[0:2])

    br_a = tl_a + np.array(d1.bb[2:])
    br_b = tl_b + np.array(d2.bb[2:])

    xA = max(tl_a[0], tl_b[0])
    yA = max(tl_a[1], tl_b[1])
    xB = min(br_a[0], br_b[0])
    yB = min(br_a[1], br_b[1])
    
    I_x = xB - xA
    I_y = yB - yA
    return (I_x, I_y)

# calculates iou
def IoU(d1:Detection, d2: Detection):
    # determine the (x, y)-coordinates of the intersection rectangle
    I_x, I_y = getBBIntersectionBounds(d1,d2)    
    
    # compute the area of intersection rectangle
    iou = 0.0
    if((I_x > 0) and (I_y > 0)):
        I = I_x*I_y
        U = np.prod(d1.bb[2:]) + np.prod(d2.bb[2:]) - I
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
    ])
}



# if __name__ == "__main__":
#     # filename = "../data/detections/LaSOT_car-17.txt"
#     # readDetFile(filename=filename)
#     print(shapes['o'])
#     print(shapes['x'])

#     n=11
#     map = np.zeros((n,n, 3)).astype(np.uint8)

#     drawO(map, [5,5], [0,255,255])
#     # drawX(map, [5,5], [250,200,125])
    


#     cv.namedWindow("map", cv.WINDOW_NORMAL)
#     cv.resizeWindow("map", 500,500)
#     cv.imshow("map",map)
#     cv.waitKey(0)

