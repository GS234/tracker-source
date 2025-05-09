import cv2 as cv
import numpy as np
from helper_func import *
import os


# class to store results of deteciton analyze
class DetAnResult:
    def __init__(self, seq_name, n, tp, fp):
        self.name = seq_name
        self.n = n
        self.tp = tp
        self.fp = fp
        
        # kinda don't have these
        self.tn = 0.0
        self.fn = 0.0
    
    def getName(self):
        return self.name
    def getTp(self):
        return self.tp
    
    def getFp(self):
        return self.fp
    
    def getN(self):
        return self.n
    
    
    def __str__(self):
        return "%s: all: %d, tp: %d, fp: %d"%(self.name, self.n, self.tp, self.fp)
    
    def __repr__(self):
        return "{%s,%d,%d,%d}"%(self.name, self.n, self.tp, self.fp)
        # return self.__str__()
    
    # some other calculations:
    def getPrecision(self):
        return self.tp / (self.tp + self.fp)
    
    def getRecall(self):
        return self.tp / (self.tp+self.fn)
    
    def getFPR(self):
        return self.fp / (self.fp+self.tn)
    
    def getAccuracy(self):
        return (self.tp + self.tn) / (self.tp+self.tn+self.fp+self.fn)



# load dets, load gt
# load frame
OPTIONS = getOptionNamespaceWdefaultInit("detan", 'options.ini')
OPTINOS_MAIN = getOptionNamespaceWdefaultInit("main2", 'options.ini') # for workspace dir
RESULT_PATH=OPTINOS_MAIN['result_path']
DETS_DIR = OPTIONS['dets_dir']
SEQUENCES_DIR = OPTIONS['sequences_dir']
# print(OPTIONS)

def getSequences():
    dir_list = os.listdir(DETS_DIR)
    aa = []
    print(dir_list)
    for f in dir_list:
        if(f.endswith('.txt')):
            sequence = f[0:-4]
            aa.append(sequence)
            # print(f[0:-4])
    aa.sort()
    # print(aa)
    return aa

def main():
    # 1. get sequences:
    sequences = getSequences() # from detections
    all_sequences = os.listdir(SEQUENCES_DIR)
    THRESHOLD = float(OPTIONS['n'])
    # print(all_sequences)

    # aaa = "LaSOT_bird-2_"
    # print(aaa in all_sequences)

    # return

    # 2. for each sequence calculate score
    results = []
    for sequence in sequences:
        if(sequence in all_sequences):
            print("using %s:"%sequence)
            
            # INIT
            D13: list[list[Detection]] = readDetFile2("%s%s.txt"%(DETS_DIR, sequence)) # read detections from file
            T_OFFSET = 1
            D13 = [[]]+D13 # offset it (because time starts at 1, there is nothing on 0)
            GT13: list[list[Detection]] = [[]]+readDetFile2("%s%s/groundtruth.txt"%(SEQUENCES_DIR, sequence),T_OFFSET) # ground truth: for testing purposes only
            t_global = T_OFFSET # current time (frame)
            # actual offset for detections list is: T_OFFSET - n (because there are some detections before global offset (initial trajectories))
            n_res = len(D13) - T_OFFSET
            n_all = n_res

            RUN_ALL = OPTIONS['run_all']
            n_tp = 0 # true positive
            n_fp = 0 # false positive
            n_total = 0 # all
            
            while(t_global <= n_all):
                # if(not RUN_ALL): print("[@ %d / %d (%d%%)]"%(t_global, n_all, int((float(t_global)/float(n_all))*100) ))
                # if(not RUN_ALL): frame = getFrameAtI(t_global, FRAMES_PATH)
                current_dets = D13[t_global]
                current_gt = GT13[t_global]
                n_det = len(current_dets)
                n_total = n_total + n_det
                gt = None
                if(current_gt):
                    gt = current_gt[0]
                    # if(not RUN_ALL): drawBoundingBox(frame, bbResize(gt.bb,1), color=[0,100,255])
                for cd in current_dets:
                    # if(not RUN_ALL): drawBoundingBox(frame, cd.bb)
                    if(gt is not None):
                        iou = IoU(gt, cd)
                        contributes = ""
                        if(iou > 0):
                            if(iou >= THRESHOLD):
                                contributes = " <- "
                                n_tp = n_tp + 1
                            else:
                                n_fp = n_fp + 1
                        if(not RUN_ALL): print("gt-cd%d:%6.2f%s"%(cd.id, iou, contributes))

                # if(not RUN_ALL): 
                #     showInNamed("image", frame)
                #     cv.waitKey(0)
                t_global = t_global+1
            
            seq_result = DetAnResult(sequence, n_total, n_tp, n_fp)
            print(seq_result)
            results.append(seq_result)
            # print("all: %d, true positive: %d, false positive: %d"%(n_total, n_tp, n_fp))
    print(results)
    with open(RESULT_PATH+'detAn.p', 'wb') as fp:
        abc = [THRESHOLD, results]
        pickle.dump(abc, fp)

#         [main2]
# ; path to results (some kind of tracker workspace)
# RESULT_PATH=./tracker/

def read_detAn(filename: str):
    with open(filename, 'rb') as fp: # fp: file pointer?
        thr, results = pickle.load(fp)
    
    print("threshold: %f"%thr)
    print("results: ")
    for r in results:
        print(r)

def test_main():
    print("this is test_main:")
    resultsd = OPTIONS["results_dir"]
    gt_pathd = OPTIONS["gt_dir"]
    THRESHOLD = float(OPTIONS['n'])
    # print("res dir: %s\ngt dir: %s"%(resultsd, gt_pathd))
    # sequences = getSequences() # from detections
    # print(sequences)

    results_dirs = os.listdir(resultsd)
    results_dirs2: list[str] = []
    for d in results_dirs:
        if(os.path.isdir("%s%s"%(resultsd, d))):
            # print(d)
            results_dirs2.append(d)
    results_dirs = results_dirs2
    results_dirs.sort()
    
    results_dirs = [results_dirs[0]]
    result_stats = []
    # print(results_dirs)
    for d in results_dirs:
        # get trajectory file
        # get gt file
        result_dets = readDetFile2("%s%s/%s_001.txt"%(resultsd, d, d))
        gt_dets = readDetFile2("%s%s_gt.txt"%(gt_pathd, d))
        
        result_dets_len = len(result_dets)
        gt_dets_len = len(gt_dets)
        if(result_dets_len == gt_dets_len):
            n_all = gt_dets_len
            t_global = 1
            print(gt_dets)
            n = 0
            n_tp = 0
            n_fp = 0
            while(t_global < n_all):
                
                current_det = result_dets[t_global]
                current_gt = gt_dets[t_global]
                if(current_det and current_gt):
                    current_det = current_det[0]
                    current_gt = current_gt[0]
                
                    
                    iou = IoU(current_gt, current_det)
                    if(iou > 0):
                        if(iou >= THRESHOLD):
                            n_tp = n_tp + 1
                        else:
                            n_fp = n_fp + 1
                    n = n+1
                t_global = t_global+1
                
            seq_result = DetAnResult(d, n,n_tp, n_fp)
            result_stats.append(seq_result)
    
    # TODO: different thresholds, ..., plot
    print(result_stats)


if __name__ == '__main__':
    mode = int(OPTIONS["mode"])
    if(mode == 0):
        # analyze:
        main()
    elif(mode == 1):
        # show results:
        read_detAn(RESULT_PATH+"detAn.p")
    elif(mode == 2):
        test_main()