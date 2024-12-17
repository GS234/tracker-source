import pickle
import numpy as np
import cv2 as cv

import sys
sys.path.append('../')
sys.path.append('../../RAFT/core') # raft stuff
from helper_func import flow2img, getFlowToFromAtI

# this file was used to convert one big optical flow estimation file into separate files with optical flow estimation for each frame (in each frame for the next frame (where pixels will move))

def open_it(filename):
    with open(filename, 'rb') as fp:
            flow = pickle.load(fp)
            return flow

DATA_ROOT = "../../data/"
# FLOW_PATH = "flow_est/LaSOT_bird-2_1/"
FLOW_PATH = "testing/flows/"
FLOW_FILE = "squares_flo_est.p"
CONVERT_IT = not True

def getFlowAtI(i):
    frame_i = f'{i:08}'
    return np.load(DATA_ROOT+FLOW_PATH+frame_i+".npy")
    

def convert_it(filename):
    a = open_it(DATA_ROOT + FLOW_PATH + filename)

    i = 0
    n = 6

    while(i < n):
        frame_i = f'{i:08}'
        # np.save("./flow_est/"+frame_i, a[i])
        np.save(DATA_ROOT+FLOW_PATH+"flows/"+frame_i, a[i])
        i = i+1


def main():
    if(CONVERT_IT):
        convert_it(FLOW_FILE)
    else:
        first = True
        for i in range(100):
            i = i%6
            # flow = getFlowAtI(i)
            flow_in, flow_out = getFlowToFromAtI(i, DATA_ROOT+FLOW_PATH)
            flow_in_im = flow2img(flow_in)
            flow_out_im = flow2img(flow_out)
            
            
            flow_in_n = "flow in"
            flow_out_n = "flow out"
            
            
            cv.imshow(flow_in_n,flow_in_im)
            cv.imshow(flow_out_n,flow_out_im)
            cv.waitKey(0)



if __name__ == "__main__":
    main()
    
            
