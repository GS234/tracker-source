import sys
sys.path.append('../RAFT/core')
import argparse
import cv2 as cv
import numpy as np
import torch
from PIL import Image
import time
from helper_func import *


from raft import RAFT
from utils.utils import InputPadder

# run as: python3 flow_est.py --model=../../RAFT/models/raft-things.pth --path=../../data/sample/
# run as: python3 flow_est.py --model=../../RAFT/models/raft-things.pth

# defaults:
# DATA_ROOT = '../../data/'
DATA_ROOT = '../data/'
ESTIMATES_ROOT = 'estimates/'
SEQUENCE = "LaSOT_bird-15"
FRAMES_PATH = '../data/frames/'+SEQUENCE+'/color/'
DEVICE = 'cuda'
CALCULATE_FLOW = not True
MODEL = "../RAFT/models/raft-things.pth"
BLANK_FLOW = True


class OpticalFlow:
    def __init__(self, model_args, frames_path=None, save_path=None, no_print=False):
        # init model:
        print("[FLOW] init model")
        self.calculate_blank = True
        if(model_args.model is None):
            print("[INFO] no specified model, using default: " + MODEL)
            model_args.model = '../RAFT/models/raft-things.pth'
        if(frames_path is None):
            print("[INFO] path is not specified, using default: " + FRAMES_PATH)
            frames_path = FRAMES_PATH
        self.save_flow = True
        if(save_path is None):
            print("[INFO] no save path specified, flow will not be saved")
            self.save_flow=False
        else:
            self.save_path = save_path

        print(model_args)

        self.model = torch.nn.DataParallel(RAFT(model_args))
        self.model.load_state_dict(torch.load(model_args.model))

        self.model = self.model.module
        self.model.to(DEVICE)
        self.model.eval()
        self.frames_path = frames_path
        print("[FLOW] init done") #, carry on
        self.no_print = no_print

    # method computes optical flow between image at i and i+1 (from current to next)
    def computeFlowAtI(self, i):
        with torch.no_grad():
            # get images
            path = self.frames_path
            image_i = i2frameI(i) # current
            image_ii = i2frameI(i+1) # next
            # images = [cv.imread(path + image_i+'.jpg',cv.COLOR_RGB2BGR), cv.imread(path + image_ii+'.jpg',cv.COLOR_RGB2BGR)]
            
            image1 = load_image(path+image_i+'.jpg')
            try:
                image2 = load_image(path+image_ii+'.jpg')
            except FileNotFoundError:
                print("[WARN] could not load second frame (at i %d). Possibly end of sequence reached. Returning blank flow."%(i+1))
                image_i1 = i2frameI(i)
                shape = tuple(np.shape(image1)[-2:])
                shape =(*shape,2)
                flow = np.zeros(shape)
                if(self.save_flow):
                    np.save(self.save_path+image_i1+'.npy', flow) # save as nnnnnnnn.npy (numpy matrix)
                return flow
            
            padder = InputPadder(image1.shape) # pad image to dimensions required by raft model
            image1, image2 = padder.pad(image1, image2)

            # calls forward
            if(not self.no_print):
                print("estimating flow ... ", end="", flush=True)
                start = time.time()
            flow_low, flow_up = self.model(image1, image2, iters=20, test_mode=True)
            if(not self.no_print):
                stop = time.time()
                print("flow computed, time: ", stop-start, " s")
            
            flow = padder.unpad(flow_up)[0].permute(1,2,0).cpu().numpy()
            if(self.save_flow):
                image_i1 = i2frameI(i)
                np.save(self.save_path+image_i1+'.npy', flow) # save as nnnnnnnn.npy (numpy matrix)
            return flow
    
    # test method, to simulate ^ (but it returns precalculated flow, when on titanX, upper method should be used)
    # def computeFlowAtITest(self, i, name_off=0):
    #     # flow_path = "../data/flow_est/"+SEQUENCE+"/"
    #     # flow_path = "/home/gasper/disk/Nedokumenti/Faks/didi_sequences/LaSOT_bird-2_flow/"
    #     # flow_path = "/home/gasper/disk/Nedokumenti/Faks/didi_sequences/LaSOT_bird-2_flow/"
    #     flow_path = "/home/gasper/Faks/3_letnik/diplomska/koda/data/flow_est/got10k/"
    #     frame_i = f'{i:08}'
    #     flow = np.load(flow_path+frame_i+".npy")
        
    #     # image_i = f'{i:08}.jpg'
    #     # cv.imread(FRAMES_PATH+image_i)
    #     # image_i = f'{i+1:08}.jpg'
    #     # cv.imread(FRAMES_PATH+image_i) # this could be problematic
        
    #     if(self.save_flow):
    #         image_i1 = f'{i+name_off:08}' 
    #         np.save(self.save_path+image_i1+'.npy', flow) # save as nnnnnnnn.npy (numpy matrix)
    #     return flow


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def numpy_2_torch(img: np.array):
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


# quick test & debug
if __name__ == '__main__':
    # prepare model arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    
    of:OpticalFlow = OpticalFlow(args, save_path="./abcde/")
    n = 7
    off = 4095
    if(CALCULATE_FLOW):
        for i in range(off,n+off):
            # of.computeFlowAtITest(i)
            of.computeFlowAtI(i)
            
    
    # vizualize:
    FLOW_PATH2 = "../data/flow_est/LaSOT_bird-2_1/"
    FLOW_PATH3 = "./abcde/"
    for i in range(off,n+off):
        # flow = getFlowAtI(i)
        frame = getFrameAtI(i,FRAMES_PATH)
        flow_in1, flow_out1 = getFlowToFromAtI(i, FLOW_PATH2)
        flow_in2, flow_out2 = getFlowToFromAtI(i, FLOW_PATH3)
        flow_in_im1 = flow2img(flow_in1)
        flow_out_im1 = flow2img(flow_out1)
        
        flow_in_im2 = flow2img(flow_in2)
        flow_out_im2 = flow2img(flow_out2)
        
        frame_n = "frame"
        flow_in_n1 = "flow in1"
        flow_out_n1 = "flow out1"
        flow_in_n2 = "flow in2"
        flow_out_n2 = "flow out2"
        
        cv.imshow(flow_in_n1,flow_in_im1)
        cv.imshow(flow_in_n2,flow_in_im2)
        cv.imshow(frame_n, frame)
        cv.imshow(flow_out_n1,flow_out_im1)
        cv.imshow(flow_out_n2,flow_out_im2)
        cv.waitKey(0)
   
