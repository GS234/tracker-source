import sys
sys.path.append('../../RAFT/core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import time
import pickle

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

# run as: python3 flow_est.py --model=../../RAFT/models/raft-things.pth --path=../../data/sample/


DATA_ROOT = '../../data/'
DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        # calculate first ...
        imgs = []
        flows = []

        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # calls forward
            print("forward is called ... ")
            start = time.time()
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            stop = time.time()
            print("forward done, time: ", stop-start, " s")
            imgs.append(image1)
            flows.append(flow_up)
            # viz(image1, flow_up)
        
        #  ... show next:
        # first: save optical flow estimation and images to pickle:
        im_fl = zip(imgs, flows)
        with open('estimates.p', 'wb') as fp: # fp: file pointer?
            pickle.dump(im_fl, fp)

        # im_fl_read = None
        # with open ('estimates.p', 'rb') as fp:
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

    # demo(args)

    im_fl_read = None
    with open ('estimates.p', 'rb') as fp:
        im_fl_read = pickle.load(fp)

    # print(im_fl_read)
    im_fl_read = list(im_fl_read)
    i = 0
    while(True):
        # for im, fl in zip(imgs, flows):
        
        if(i >= len(im_fl_read)):
            i = 0
        im, fl = im_fl_read[i]
        viz(im, fl)
        i = i+1