import cv2 as cv
import numpy as np
from helper_func import *
import DetectionSpace

# load dets, load gt
# load frame
OPTIONS = getOptionNamespaceWdefaultInit("detan", 'options.ini')
print(OPTIONS)

def main():
    sequence_str = OPTIONS['sequence']
    print(sequence_str)
    T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getSequenceConsts(sequence_str)
    
    # INIT
    D13: list[list[Detection]] = readDetFile2(DETS_FILE) # read detections from file
    D13 = [[]]+D13 # offset it (because time starts at 1, there is nothing on 0)
    GT13: list[list[Detection]] = [[]]+readDetFile2(GT_PATH,T_OFFSET) # ground truth: for testing purposes only
    t_global = T_OFFSET # current time (frame)
    # actual offset for detections list is: T_OFFSET - n (because there are some detections before global offset (initial trajectories))
    n_res = len(D13) - T_OFFSET
    n_all = n_res

    N = float(OPTIONS['n'])
    RUN_ALL = OPTIONS['run_all']
    n_tp = 0 # true positive
    n_fp = 0 # false positive
    n_total = 0 # all
    
    while(t_global <= n_all):
        if(not RUN_ALL): print("[@ %d / %d (%d%%)]"%(t_global, n_all, int((float(t_global)/float(n_all))*100) ))
        if(not RUN_ALL): frame = getFrameAtI(t_global, FRAMES_PATH)
        current_dets = D13[t_global]
        current_gt = GT13[t_global]
        n_det = len(current_dets)
        n_total = n_total + n_det
        gt = None
        if(current_gt):
            gt = current_gt[0]
            if(not RUN_ALL): drawBoundingBox(frame, bbResize(gt.bb,1), color=[0,100,255])
        for cd in current_dets:
            if(not RUN_ALL): drawBoundingBox(frame, cd.bb)
            if(gt is not None):
                iou = IoU(gt, cd)
                contributes = ""
                if(iou > 0):
                    if(iou >= N):
                        contributes = " <- "
                        n_tp = n_tp + 1
                    else:
                        n_fp = n_fp + 1
                if(not RUN_ALL): print("gt-cd%d:%6.2f%s"%(cd.id, iou, contributes))

        if(not RUN_ALL): 
            showInNamed("image", frame)
            cv.waitKey(0)
        t_global = t_global+1
    
    print("all: %d, true positive: %d, false positive: %d"%(n_total, n_tp, n_fp))
    

if __name__ == '__main__':
    main()