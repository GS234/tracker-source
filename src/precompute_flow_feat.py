from helper_func import *
from FeatureExtractor import FeatureExtractor, roiPool, patchesInBB, getFeaturesPca
from OpticalFlow import OpticalFlow
import cv2 as cv
import argparse
import os, os.path
from pathlib import Path

import warnings
warnings.simplefilter('ignore')

# CUDA_VISIBLE_DEVICES=2 python3 precompute_flow_feat.py --sequence 'LaSOT_bird-15' --seq_path '/home/gasper/Faks/3_letnik/diplomska/koda/data/frames/' --save_path '/home/gasper/Faks/3_letnik/diplomska/koda/data/' --feat
# CUDA_VISIBLE_DEVICES=2 python3 precompute_flow_feat.py --sequence='LaSOT_bird-2' --save_path='/home/gasper/disk/Nedokumenti/Faks/precomputed/'
# CUDA_VISIBLE_DEVICES=2 python3 precompute_flow_feat.py --sequence='LaSOT_bird-2'

# flow, feat getters:
COMPUTE_OR_GET_TEST = not True
PRECOMPUTED_FLOW_PATH = None
PRECOMPUTED_FEAT_PATH = None
COMPUTED_FLOW = None
COMPUTED_FEAT = None

def computeOrGetFlowAtI(i:int, optical_flow_gen: OpticalFlow=None):
    if(COMPUTE_OR_GET_TEST): return computeFlowAtITest(i, save_flow=True) # WARN: this is to test only, on real use cases, use ^
    return optical_flow_gen.computeFlowAtI(i)

def computeOrGetPCAFeaturesAtI(i:int, image:np.ndarray, feature_ext:FeatureExtractor, save_feat=False):
    if(COMPUTE_OR_GET_TEST): return computePCAFeaturesAtItest(i, save_feat=True)
    print("[getOrComputePCAFeatures] computing features")
    frame_feat = feature_ext.getFeatures(image)['x_norm_patchtokens'].to('cpu').numpy() # get frame features (patchtokens)
    pca_feat = getFeaturesPca(frame_feat, (feature_ext.last_padder.patch_h, feature_ext.last_padder.patch_w))
    print("[getOrComputePCAFeatures] features computed")
    if(save_feat):
        np.save(COMPUTED_FEAT+i2frameI(i)+".npy",pca_feat)
    return pca_feat

# test methods: are used if flag is set
def computeFlowAtITest(i:int, save_flow=False):
    flow = getFlowAtI(i, PRECOMPUTED_FLOW_PATH)
    if(save_flow):
        np.save(COMPUTED_FLOW+i2frameI(i)+".npy",flow) # save as nnnnnnnn.npy (numpy matrix)
    return flow

def computePCAFeaturesAtItest(i:int, save_feat=False):
    pca_feat = getFeaturesAtI(i, PRECOMPUTED_FEAT_PATH)
    if(save_feat):
        np.save(COMPUTED_FEAT+i2frameI(i)+".npy",pca_feat)
    return pca_feat
# -----------------


def main():
    global COMPUTED_FLOW
    global COMPUTED_FEAT

    global PRECOMPUTED_FEAT_PATH
    global PRECOMPUTED_FLOW_PATH

    # ARGUMENTS
    sequence = ''
    # seq_path = '/home/gasper/tracker_ws/workspace/sequences/'
    seq_path = '/home/gasper/disk/Nedokumenti/Faks/didi_sequences/'
    save_path = './'

    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', help='sequence name')
    parser.add_argument('--seq_path', help='path to sequences')
    parser.add_argument('--save_path', help='path to sequences')
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    
    parser.add_argument('--flow', help='calculate flow (note: if none, both are calculated)', action='store_true', default=False)
    parser.add_argument('--feat', help='calculate features (note: if none, both are calculated)', action='store_true', default=False)
    args = parser.parse_args()
    
    print(args)
    calculate_mode = 0 # 0 - both, 1: flow-only, 2: frames-only
    if(not (args.flow or args.feat) or (args.flow and args.feat)):
        print("calculating flow and features")
        calculate_mode = 0
    elif(args.flow):
        print("calculating flow only")
        calculate_mode = 1
    else:
        print('calculating features only')
        calculate_mode = 2

    if(args.sequence is None or args.sequence==''):
        print("[warn] please specify sequence with --sequence '<sequence>'")
        return
    else: sequence = args.sequence
    if(args.seq_path is not None): seq_path = args.seq_path
    if(args.save_path is not None): save_path = args.save_path
    

    # print(args.sequence)
    print("using sequence: ",sequence)
    print("sequence path : ",seq_path)
    # ------------
    FRAMES_PATH = "%s%s/color/"%(seq_path,sequence)
    t_global = 1

    # make appropriate directories to store flow:
    COMPUTED_FLOW = "%s%s_pr/%s_flow/"%(save_path, sequence, sequence)
    COMPUTED_FEAT = "%s%s_pr/%s_feat/"%(save_path, sequence, sequence)

    # this is for testing only
    # PRECOMPUTED_FLOW_PATH = "%s%s_flow/"%(seq_path, sequence)
    # PRECOMPUTED_FEAT_PATH = "%s%s_feat/"%(seq_path, sequence)
    # ------

    # init raft, dinov2:
    n = len([name for name in os.listdir(FRAMES_PATH) if os.path.isfile(os.path.join(FRAMES_PATH, name))]) # number of files in path

    # INIT OPTICAL FLOW
    if(calculate_mode == 0 or calculate_mode == 1):
        # of = OpticalFlow(args,frames_path=FRAMES_PATH, save_path=COMPUTED_FLOW) # save_flow = True
        Path(COMPUTED_FLOW).mkdir(parents=True, exist_ok=True)
        of = OpticalFlow(args,frames_path=FRAMES_PATH, save_path=COMPUTED_FLOW, no_print=True) # save_flow = True (because save_path is not none)

    # INIT DINOV2
    if(calculate_mode == 0 or calculate_mode == 2):
        frame = getFrameAtI(t_global,FRAMES_PATH)
        Path(COMPUTED_FEAT).mkdir(parents=True, exist_ok=True)
        featureExt = FeatureExtractor(model_size='base')
        featureExt.getFeatures(frame) # sets necessary offsets by computing first frame (could also do that differently)

    VISUALIZE = False
    nn = n
    try:
        for i in range(nn):
            if(calculate_mode == 0 or calculate_mode == 1):
                flow=computeOrGetFlowAtI(t_global, of)
            
            if(calculate_mode == 0 or calculate_mode == 2):
                frame = getFrameAtI(t_global, FRAMES_PATH)
                feat=computeOrGetPCAFeaturesAtI(t_global, frame, featureExt, True)

            # visualize, if:
            if(VISUALIZE):
                print("t_global: %d"%(t_global))
                showInNamed("frame", frame)
                flow_img = flow2img(flow)
                showInNamed("flow", flow_img)
                showInNamed("feat", feat)
                cv.waitKey(0)
            
            print("frames: %d/%d (%.0f%%)"%(t_global, nn, (float(t_global)/float(nn))*100))
            
            t_global = t_global + 1
    except KeyboardInterrupt:
        print('da da da')
    print()
        

if __name__ == '__main__':
    main()

