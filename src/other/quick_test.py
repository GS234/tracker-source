import pickle
import numpy as np
import cv2 as cv

import sys
sys.path.append('../')
sys.path.append('../../RAFT/core') # raft stuff
from helper_func import flow2img

# this file was used to convert one big optical flow estimation file into separate files with optical flow estimation for each frame

def open_it(filename):
    with open(filename, 'rb') as fp:
            flow = pickle.load(fp)
            return flow

DATA_ROOT = "../../data/"
FLOW_PATH = "flow_est/LaSOT_bird-2_1/"

def getFlowAtI(i):
    frame_i = f'{i:08}'
    return np.load(DATA_ROOT+FLOW_PATH+frame_i+".npy")
    

def convert_it(filename):
    a = open_it(DATA_ROOT + FLOW_PATH + filename)

    i = 1000
    n = 2000

    while(i < n):
        frame_i = f'{i:08}'
        np.save("./flow_est/"+frame_i, a[i])
        i = i+1


if __name__ == "__main__":
    # convert_it("flowEst.p")
    for i in range(1000):
        flow = getFlowAtI(i)
        flow_im = flow2img(flow)
        
        cv.imshow("flow img",flow_im)
        cv.waitKey(0)
    
            
