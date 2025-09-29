from helper_func import *
from FeatureExtractor import FeatureExtractor, getFeaturesPca
from OpticalFlow import OpticalFlow
import argparse
import cv2 as cv
import os, os.path
from pathlib import Path
import time
from torch import OutOfMemoryError

import warnings
warnings.simplefilter('ignore')

# CUDA_VISIBLE_DEVICES=2 python3 precompute_flow_feat.py --sequence 'LaSOT_bird-15' --seq_path '/home/gasper/Faks/3_letnik/diplomska/koda/data/frames/' --save_path '/home/gasper/Faks/3_letnik/diplomska/koda/data/' --feat
# CUDA_VISIBLE_DEVICES=2 python3 precompute_flow_feat.py --sequence='LaSOT_bird-2' --save_path='/home/gasper/disk/Nedokumenti/Faks/precomputed/'
# CUDA_VISIBLE_DEVICES=2 python3 precompute_flow_feat.py --sequence='LaSOT_bird-2'
# CUDA_VISIBLE_DEVICES=2 python3 precompute_flow_feat.py --sequence LaSOT_train-7 --seq_path /home/gasper/tracker_ws/workspace/sequences/ --save_path '/home/gasper/precomputed/' --flow
# CUDA_VISIBLE_DEVICES=3 python3 precompute_flow_feat.py --sequence LaSOT_train-7 --seq_path /home/gasper/tracker_ws/workspace/sequences/ --save_path '/home/gasper/precomputed/' --flow

# flow, feat getters:
OPTIONS = getOptionNamespaceWdefaultInit('detan', 'options.ini')

def computeOrGetFlowAtI(i:int, optical_flow_gen: OpticalFlow=None):
    return optical_flow_gen.computeFlowAtI(i)

def computeOrGetFeaturesAtI(i:int, image:np.ndarray, feature_ext:FeatureExtractor):
    # print("[getOrComputePCAFeatures] computing features")
    frame_feat = feature_ext.getFeatures(image)['x_norm_patchtokens'].to('cpu').numpy() # get frame features (patchtokens)
    return frame_feat


# -----------------

def get_sequence_list():
    detsd = OPTIONS["dets_dir"]
    print("this is get_sequence_list")
    
    # det files:
    det_files = os.listdir(detsd)
    filenames: list[str] = []
    for d in det_files:
        # if(os.path.isdir("%s%s"%(detsd, d))):
        if(d.endswith(".txt")):
            # print(d)
            filenames.append(d.split(".")[0])
    filenames.sort()
    # print(filenames)
    return filenames




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    # max_n = 20 # max frame-ov
    max_n = 3 # -1: use all
    # print("main je tole")
    # calculate_mode = 0
    calculate_mode = 1
    sequences_dir = OPTIONS['sequences_dir']
    dets_dir = OPTIONS['dets_dir']
    # get all sequences
    print(sequences_dir)
    print(dets_dir)
    # sequences_list = get_sequence_list()
    sequences_list = ["GOT-10k_GOT-10k_Val_000014", "LaSOT_bird-2","LaSOT_chameleon-20", "LaSOT_coin-18", "LaSOT_train-7"]
    # print(sequences_list)

    # init raft, dinov2:
    # INIT OPTICAL FLOW
    if(calculate_mode == 0 or calculate_mode == 1):
        # of = OpticalFlow(args,frames_path=FRAMES_PATH, save_path=COMPUTED_FLOW) # save_flow = True
        of = OpticalFlow(args,frames_path="", save_path=None, no_print=True) # save_flow = False (because save_path is none)

    # INIT DINOV2
    if(calculate_mode == 0 or calculate_mode == 2):
        # frame = getFrameAtI(t_global,FRAMES_PATH)
        featureExt = FeatureExtractor(model_size='base')

    output_results: str = ""

    flow_times:list[float] = []
    feat_times:list[float] = []
    # sequence-specific init (for every sequence)
    # for sequence in sequences_list:
    for i in range(len(sequences_list)):
        flow_times_seq:list[float] = []
        feat_times_seq:list[float] = []

        sequence = sequences_list[i]
        FRAMES_PATH = "%s%s/color/"%(sequences_dir,sequence)
        if(calculate_mode == 0 or calculate_mode == 1):
            of.frames_path = FRAMES_PATH
        # print(FRAMES_PATH)

        t_global = 1
        n = len([name for name in os.listdir(FRAMES_PATH) if os.path.isfile(os.path.join(FRAMES_PATH, name))]) # number of files in path
        # print(n)
        frame = getFrameAtI(t_global,FRAMES_PATH)
        h, w, _ = np.shape(frame) # get frame dimensions
        print("frame size: %dx%d"%(w, h))
        print("using sequence %s (%dx%d, %d frames)"%(sequence, w, h, n))

        if(calculate_mode == 0 or calculate_mode == 2):
            featureExt.getFeatures(frame) # sets necessary offsets by computing first frame (could also do that differently)
        
        nn = n
        print_results = True
        try:
            for i in range(nn):
                if(max_n != -1 and i >= max_n):
                    break
                result_str = ""
                if(calculate_mode == 0 or calculate_mode == 1):
                    start_t = time.time()
                    flow=computeOrGetFlowAtI(t_global, of)
                    end_t = time.time()
                    flow_time_i = (end_t-start_t)
                    flow_times.append(flow_time_i)
                    flow_times_seq.append(flow_time_i)
                    avg_flow = "avg flow: %.2f"%np.mean(flow_times_seq)
                    result_str = "%s, %s"%(result_str, avg_flow)
                
                if(calculate_mode == 0 or calculate_mode == 2):
                    frame = getFrameAtI(t_global, FRAMES_PATH)

                    start_t = time.time()
                    feat=computeOrGetFeaturesAtI(t_global, frame, featureExt)
                    end_t = time.time()
                    feat_time_i = (end_t-start_t)
                    # print("feat time: %f"%feat_time_i)
                    feat_times.append(feat_time_i)
                    feat_times_seq.append(feat_time_i)
                    result_str = "%s%s"%(result_str, ", avg feat: %.2f"%np.mean(feat_times_seq))

                print("frames: %d/%d (%.0f%%)%s"%(t_global, nn, (float(t_global)/float(nn))*100, result_str))
                
                t_global = t_global + 1
        except KeyboardInterrupt:
            print('abort')
            break
        except OutOfMemoryError:
            print("cuda out of memory, ignoring results")
            print_results = False
        if(print_results):
            result_str = ""
            if(calculate_mode == 0 or calculate_mode == 1):
                avg_flow = "flow: %.2f"%np.mean(flow_times_seq)
                result_str = "%s %s"%(result_str, avg_flow)
            if(calculate_mode == 0 or calculate_mode == 2):
                avg_feat = "feat: %.2f"%np.mean(feat_times_seq)
                result_str = "%s %s"%(result_str, avg_feat)
            # print("%s (%dx%d, %d frames): avg flow: %.2f, avg feat: %.2f"%(sequence, w, h, n, np.mean(flow_times_seq), np.mean(feat_times_seq))) # print to file "%(sequence, w, h, n)
            output_result = "%s (%dx%d, %d frames): %s"%(sequence, w, h, n, result_str)
            print(output_result) # print to file "%(sequence, w, h, n)
            output_results = "%s%s\n"%(output_results, output_result)
        print()
    
    result_str = ""
    if(calculate_mode == 0 or calculate_mode == 1):
        avg_flow = "flow: %.2f"%np.mean(flow_times)
        result_str = "%s %s"%(result_str, avg_flow)
    if(calculate_mode == 0 or calculate_mode == 2):
        avg_feat = "feat: %.2f"%np.mean(feat_times)
        result_str = "%s %s"%(result_str, avg_feat)

    output_result = "global avg times: %s"%(result_str)
    print(output_result) # print to file
    output_results = "%s%s\n"%(output_results, output_result)

    print()
    print(output_results)
    with open("speed.txt", "a") as f:
        print(output_results, file=f)
        
        

if __name__ == '__main__':
    main()

