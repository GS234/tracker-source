from Detection import Detection
import numpy as np
import random
import pickle

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
    for point in X:
        x, y = point
        x = x%h
        y = y%w
        
        
        if(overwrite or np.sum(map[x,y]) == 0):
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



