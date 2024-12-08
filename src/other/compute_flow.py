import sys
sys.path.append('../../RAFT/core') # add raft to path

import argparse
import os
import glob
import numpy as np
import torch
from PIL import Image
import time
import pickle

from raft import RAFT
from utils.utils import InputPadder

# run as: python3 compute_flow.py --model=../../RAFT/models/raft-things.pth --path=../../data/sample/
# run as: python3 flow_est.py --model=../../RAFT/models/raft-things.pth

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def computeFlow(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        path = args.path
        images = glob.glob(os.path.join(path, '*.png')) + \
                 glob.glob(os.path.join(path, '*.jpg'))
        images = sorted(images)
        
        # calculate first ...
        flows = []
        imzip = zip(images[:-1], images[1:])
        for imfile1, imfile2 in imzip:
            
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
            flows.append(padder.unpad(flow_up)[0].permute(1,2,0).cpu().numpy())
        print("padding flow array with empty flow estimation")
        flows.append(flows[-1]*0) # from last to next to last? I think not! (pad last)
        
        # save optical flow estimation and images to pickle:
        with open('./flowEst.p', 'wb') as fp:
            pickle.dump(flows, fp)
        
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

    computeFlow(args)
    