from Detection import Detection
import numpy as np
import cv2 as cv
import random
import pickle
from matplotlib import pyplot as plt

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

# like ^ (readDetFile), this one returns detections instead of tuples (might need to offset it with +1)
def readDetFile2(filename: str):
    detections = []
    with open(filename) as fd:
        frame_i = 0
        for line in fd: # line represents frame at i (frame_i) - STARTS WITH 0
            dets = line[0:-1].split(";")
            a = []
            for d in dets:
                d_split = d.split(',')
                if(d_split[0]):
                    # print(d_split)
                    bb = [int(float(i)) for i in d.split(',')]
                    detection = bbDet2Det(bb, frame_i)
                    a.append(detection)
            detections.append(a)
            # print()
            frame_i = frame_i + 1
    return detections

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
    y, x, h, w = det.bb
    frame_bb = frame[x:x+w+1,y:y+h+1] # bounding box image
    det.color_hist = getColorHist(frame_bb,n_bins=n_bins)
    det.hasHist = True

            

# method calculates color histogram from current frame's detecitons
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


# bounding box deteciton to Detection - generates Detection object with coordinates of center of a bounding box
def bbDet2Det(bb: tuple, t: int = 0):
    x, y = bb[0]+bb[2]//2,bb[1]+bb[3]//2 # (x, y swapped, because of coordinate system)
    return Detection([x,y], t=t, bb=bb)

# function reads i-th frame
def getFrameAtI(i: int, path: str, toBGR=False):
    frame_i = f'{i:08}'
    frame = cv.imread(path+str(frame_i)+".jpg")
    if(toBGR):
        return cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    else:
        return frame

def showHists(hists:list, c=1):
    color = ['b','g','r']
    # show only in one dimension
    _,ax = plt.subplots(1,len(hists))
    # print(ax)
    for j in range(len(hists)):
        hist = hists[j]
        arr = np.zeros((np.shape(hist)[0],3))
        
        
        for i in range(len(arr)):
            arr[i,0] = np.sum(hist[i,:,:]) # c1
            arr[i,1] = np.sum(hist[:,i,:]) # c2
            arr[i,2] = np.sum(hist[:,:,i]) # c3
        x = np.arange(len(arr))
        
        
        if(len(hists) == 1):
            ax.bar(x,arr[:,c], color=color[c%3])
        else:
            ax[j].bar(x,arr[:,c], color=color[c%3])
    # print(arr)
    
    # plt.bar(x,arr)
    plt.show()

if __name__ == "__main__":
    filename = "../data/detections/LaSOT_car-17.txt"
    readDetFile(filename=filename)

def getTrColor():
    return tr_colors[int(random.random()*len(tr_colors))]


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



