import pickle
import numpy as np
import cv2 as cv

import sys
sys.path.append('../')
sys.path.append('../../RAFT/core') # raft stuff
from helper_func import flow2img, getFlowToFromAtI, getFlowAtI, getFlowToI, getFrameAtI

# this file was/is used to convert one big optical flow estimation file into separate files with optical flow estimation for each frame
# (in each frame for the next frame (where pixels will move))

def open_it(filename):
    with open(filename, 'rb') as fp:
            flow = pickle.load(fp)
            print(len(flow))
            return flow

DATA_ROOT = "../../data/"
# FLOW_PATH = "flow_est/LaSOT_bird-2_1/"
FLOW_PATH = "testing/flows/" 
FLOW_FILE = "squares_flo_est.p"

FLOW_FILE = "/media/gasper/Kevdr/Nedokumenti/flowEst.p"
save_loc = "../../data/flow_est/LaSOT_bird-2_1/tt/"
FLOW_PATH = "../../data/flow_est/LaSOT_bird-2_1/"
FRAMES_PATH = "../../data/frames/LaSOT_bird-2/color/"

CONVERT_IT = not True

def convert_it(filename, save_loc=DATA_ROOT+FLOW_PATH+"flows/"):
    a=open_it(filename)
        
    i = 4000
    n = 5000
    
    try:
        while(i < n):
            frame_i = f'{i:08}'
            np.save(save_loc+frame_i, a[i]) # save as file.npy (numpy matrix)
            i = i+1
    except:
    	print("ni vec")


def main():
    if(CONVERT_IT):
        convert_it(FLOW_FILE, save_loc=save_loc)
    else:
        first = True
        for i in range(2950,3200):
            if(i%100 == 0):
                print("i:",i)
            # flow = getFlowAtI(i)
            frame = getFrameAtI(i+1,FRAMES_PATH)
            flow_in, flow_out = getFlowToFromAtI(i, FLOW_PATH)
            flow_in_im = flow2img(flow_in)
            flow_out_im = flow2img(flow_out)
            
            frame_n = "frame"
            flow_in_n = "flow in"
            flow_out_n = "flow out"
            
            cv.imshow(flow_in_n,flow_in_im)
            cv.imshow(frame_n, frame)
            cv.imshow(flow_out_n,flow_out_im)
            cv.waitKey(0)
            



if __name__ == "__main__":
    main()
    
            
