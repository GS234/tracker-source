import sys
sys.path.append('../../RAFT/core')
sys.path.append('../')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import time
import pickle
from helper_func import *


from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

# run as: python3 flow_est.py --model=../../RAFT/models/raft-things.pth --path=../../data/sample/
# run as: python3 flow_est.py --model=../../RAFT/models/raft-things.pth


# DATA_ROOT = '../../data/'
DATA_ROOT = '../../data/'
ESTIMATES_ROOT = 'estimates/'
DEVICE = 'cuda'
CALCULATE_FLOW = False

# ESTIMATES_FILE = 'estimates.p'
# kvadrati
# ESTIMATES_FILE = 'squares_flo_est.p'
# FRAMES_PATH = 'testing/sequence/'
# DETS_FILE = 'testing/sequence/dets.txt'

# pingvini5
ESTIMATES_FILE = 'sample_flo_2.p'
DETS_FILE = 'sample/dets.txt'
FRAMES_PATH = 'sample/'

def getX(c:list) -> None:
    x_shape = np.array([
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
    ])
    x_shape = x_shape + c # add origin
    p_list = [(x[1], x[0]) for x in x_shape]
    return p_list

def getLinePoints(x1, x2, color=[0,0,255]):
    x1 = np.array(x1)
    x2 = np.array(x2)
    n = (x2 - x1) # normal from x1 to x2
    # print(n.reshape( (2,1) ))
    n_len = np.sqrt(np.dot(n.T,n))
    if(n_len == 0): # if the same point, no need to draw :)
        return [x1]
    n = n/n_len
    n = n.reshape((2,1))
    
    values = np.arange(0, int(n_len), 0.1)
    values = values.reshape((1,len(values)))
    
    points = np.dot(n, values) + x1.reshape((2,1))
    p_list = [(int(x[1]), int(x[0])) for x in points.T]
    return p_list

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def numpy_2_torch(img: np.array):
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def getRectBounds(ul, dr):
    ret = []
    wh = dr-ul
    w = np.arange(0,wh[0]) # width
    h = np.arange(0,wh[1]) # height

    # --
    x1 = w+ul[0]
    y1 = np.ones(wh[0])*ul[1]
    xy1 = np.array([y1,x1+1]).T.astype(np.int32)

    # |
    x2 = np.ones(wh[1])*ul[0]
    y2 = h+ul[1]
    xy2 = np.array([y2,x2]).T.astype(np.int32)

    # --
    # x3 = w+ul[0]
    y3 = np.ones(wh[0])*ul[1]+wh[1]
    xy3 = np.array([y3,x1]).T.astype(np.int32)
    
    #   |
    # x4 = h+ul[1]
    x4 = np.ones(wh[1])*ul[0]+wh[0]
    xy4 = np.array([y2+1,x4]).T.astype(np.int32)


    # print("xy3: ",xy3)
    return np.vstack((xy1, xy2, xy3, xy4))

def getRect(img, ul, dr):
    wh = dr-ul # dx, dy
    return img[ul[1]:ul[1]+wh[1]+1, ul[0]:ul[0]+wh[0]+1]

# function calculates vector of direction in which region moved
# function returns median value, unless mean arg is set to True
def getMotionVec(flo_region, mean=False):
    # n = np.sum(np.shape(flo_region)) # number of pixels
    # print(flo_region, np.shape(flo_region))
    # print(flo_region)
    flo_reg_resh = flo_region.reshape((-1,2))
    # print(flo_reg_resh)
    vec=[0,0]
    if(mean):
        vec = np.mean(flo_reg_resh,axis=0)
        print("mean: ",vec)
    else:
        # MEDIAN JE BOLS, KER IMAMO PRI DETEKCIJAH SE BG; MEAN JE SKOR ENAK, CE IMAMO SAMO REGION Z GIBANJEM
        vec = np.median(flo_reg_resh,axis=0)
        print("median: ",vec)
    max = np.amax(flo_reg_resh, axis=0)
    
    print("max: ", max)

    return vec
    



def viz(img, flo, dets: list[Detection]=[]):
    # img = img[0].permute(1,2,0).cpu().numpy()
    # flo_og = flo[0].permute(1,2,0).cpu().numpy()
    # print(img, flo)
    flo_og = flo.copy()
    # img = np.zeros( (np.shape(flo_og)[0], np.shape(flo_og)[1], 3))
    
    # print(flo)
    # map flow to rgb image
    print(np.shape(flo_og))
    flo = flow_viz.flow_to_image(flo_og)
    # img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    x_n = 6
    x = [[62,51], # 0
         [251,254],
         
         [357, 156], # 2
         [467, 260],

         [307, 69], # 4
         [446, 198],

         [299,310], # 6
         [416,430]

         ]
    ul = np.array(x[x_n]).astype(np.int32)
    dr = np.array(x[x_n+1]).astype(np.int32)
    wh = dr-ul
    # print(ul)
    # coords2map([ul,dr], flo, [0,0,0])
    
    # flo[10,15] = [0,0,0]
    
    if(len(dets) != 0):
        ii=0
        for d in dets:
            ul = np.array([d.bb[0],d.bb[1]])
            dr = np.array([d.bb[0]+d.bb[2],d.bb[1]+d.bb[3]])
            center = np.array([d.bb[0]+d.bb[2]//2,d.bb[1]+d.bb[3]//2]).astype(np.int32)
            print(dr)
            img_2 = getRect(flo, ul,dr)
            flo_2 = getRect(flo_og, ul, dr)
            res = getMotionVec(flo_2, mean=False) # uses median
            # res = getMotionVec(flo_2, mean=True)
            print(str(ii)+": ",res)
            
            coords2map(getLinePoints(center, (center+(res*10).astype(np.int32))), flo)
            coords2map(getX(center),flo, [0,0,0])
            # cv2.imshow('rect '+str(ii), img_2)
            ii = ii + 1
        for d in dets:
            ul = np.array([d.bb[0],d.bb[1]])
            dr = np.array([d.bb[0]+d.bb[2],d.bb[1]+d.bb[3]])
            coords2map(getRectBounds(ul,dr), img, [150,150,150])
            coords2map(getRectBounds(ul,dr), flo, [0,0,0])
    
    cv2.imshow('image', (img[:, :, [2,1,0]]).astype(np.uint8))
    cv2.imshow('flow', (flo[:, :]).astype(np.uint8))
    cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        # images = glob.glob(os.path.join(args.path, '*.png')) + \
        #          glob.glob(os.path.join(args.path, '*.jpg'))
        print(args.path)
        # path = DATA_ROOT+"testing/sequence/"
        path = DATA_ROOT+FRAMES_PATH
        images = glob.glob(os.path.join(path, '*.png')) + \
                 glob.glob(os.path.join(path, '*.jpg'))
        
        images = sorted(images)
        
        # calculate first ...
        # imgs = []
        flows = [] # we do not have flow estimation from '0th' frame to 1st frame, so pad
        
        last_im = None
        imzip = zip(images[:-1], images[1:])
        for imfile1, imfile2 in imzip:
            
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            # print(image1[0].permute(1,2,0).cpu().numpy())
            # image3 = image1.cpu()
            # print(image3)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # calls forward
            print("forward is called ... ")
            start = time.time()
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            stop = time.time()
            print("forward done, time: ", stop-start, " s")
            
            # imgs.append(padder.unpad(image1)[0].permute(1,2,0).cpu().numpy())
            flows.append(padder.unpad(flow_up)[0].permute(1,2,0).cpu().numpy())
            # last_im = image2
        
        #  ... show next:
        # first: save optical flow estimation and images to pickle:
        # last_im_numpy = padder.unpad(last_im)[0].permute(1,2,0).cpu().numpy()
        # imgs.append(last_im_numpy) # add last
        flows.append(flows[-1]*0) # from last to next to last? I think not!
        # im_fl = zip(imgs, flows)
        with open(ESTIMATES_ROOT+ESTIMATES_FILE, 'wb') as fp: # fp: file pointer?
            # pickle.dump(im_fl, fp)
            pickle.dump(flows, fp)

        # im_fl_read = Non¸e
        # with open(ESTIMATES_ROOT+'estimates.p', 'rb') as fp:
        #     im_fl_read = pickle.load(fp)

        # while(True):
        #     # for im, fl in zip(imgs, flows):
        #     for im, fl in im_fl:
        #         viz(im, fl)
        
if __name__ == '__main__':
    # prepare model arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    # print(args)

    im_fl_read = None
    if(CALCULATE_FLOW):
        demo(args)
    
    with open(ESTIMATES_ROOT+ESTIMATES_FILE, 'rb') as fp:
        im_fl_read = pickle.load(fp)
        print("im_fl_read is oftype(",type(im_fl_read),")")

    dets = readDetFile2(DATA_ROOT+DETS_FILE)
    # dets = [[],[]]
    # print("detecitons: ", dets)


    # print(im_fl_read)
    im_fl_read = list(im_fl_read)
    # print("im fl read, shape: ", np.shape(im_fl_read))
    i = 0
    det_ind = 0
    while(True):
        # for im, fl in zip(imgs, flows):
        
        if(i >= len(im_fl_read)):
            i = 0
        fl = im_fl_read[i]
        img = getFrameAtI(i, DATA_ROOT+FRAMES_PATH)
        viz(img, fl, dets[det_ind])
        i = i+1
        det_ind = (det_ind + 1) % (len(dets))