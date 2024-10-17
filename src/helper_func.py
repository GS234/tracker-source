from Detection import Detection
import numpy as np
import random


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

def detections2map(D, map):
    n = np.shape(map)[0]
    t_now = len(D) # which time instant is it
    for d in D:
        x, y = d.x
        # out of bounds is possible, this is the easiest quick fix, should probably be made different
        x = x%n
        y = y%n
        
        # map[x,y]= 1.0 - (1.0 / (1.0+(d.t/20.0)))*0.9 # more correct, as more recent detections should be brighter
        map[x,y]= 1.0
        # map[x,y]= (1.0 / (1.0+(d.t/8))) # I like this more, but is not correct because of ^

def coords2map(X, map, brightness=0.5, overwrite=True):
    n = np.shape(map)[0]
    for point in X:
        x, y = point
        x = x%n
        y = y%n
        if(overwrite or map[x,y] == 0):
            map[x,y]= brightness



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


# read detections from file and write it to list
# [TODO] - convert tuples to 'some kind' of detections
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

# bounding box deteciton to Detection - generates Detection object with coordinates of center of a bounding box
# [TODO]
def bbDet2TDet(bb: tuple):
    x, y = bb[1],bb[0] # (x, y swapped, because of coordinate system)
    return Detection()
    pass



if __name__ == "__main__":
    filename = "../data/detections/LaSOT_car-17.txt"
    readDetFile(filename=filename)

