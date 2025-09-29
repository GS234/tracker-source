import cv2 as cv
import numpy as np
from helper_func import *
import os
import random


# class to store results of deteciton analysis
class ResultNode2:
    def __init__(self, n, threshold, tp, fp, fn, fp2=0, fn2=0):
        self.threshold = threshold
        self.n = n
        self.tp = tp
        self.fp = fp
        self.fn = fn
        
        # object-specific: consider only those that overlap with groundtruth (or not)
        self.fp2 = fp2 # overlaping detections that are lower than threshold
        self.fn2 = fn2 # number of frames without overlap (iou = 0)

        
        
        
        # kinda don't have these
        self.tn = 0.0
    
    def getTp(self):
        return self.tp
    
    def getFp(self):
        return self.fp
    
    def getFn(self):
        return self.fn
    
    def getFp2(self):
        return self.fp2
    
    def getFn2(self):
        return self.fn2
    
    def getN(self):
        return self.n
    
    def __str__(self):
        return "all: %d, tp: %d, fp: %d"%(self.n, self.tp, self.fp)
    
    def __repr__(self):
        return "{%d,%d,%d}"%(self.n, self.tp, self.fp)
        # return self.__str__()
    
    # some other calculations:
    def getPrecision(self, use_2 = False):
        tp = self.tp
        fp = self.fp
        if(use_2):
            fp = self.fp2

        den = (tp + fp)
        if(den == 0):
            print("ResultNode2.getPrecision: [WARN] division by zero, returning 0")
            return 0
        return tp / (tp + fp)
    
    def getRecall(self, use_2 = False):
        tp = self.tp
        fn = self.fn
        if(use_2):
            fn = self.fn2
        
        den = (tp+fn)
        if(den == 0):
            print("ResultNode2.getRecall: [WARN] division by zero, returning 0")
            return 0
        return tp / (tp+fn)
    
    def getFPR(self):
        return self.fp / (self.fp+self.tn)
    
    def getAccuracy(self):
        return (self.tp + self.tn) / (self.tp+self.tn+self.fp+self.fn)

    def getPercentPr(self):
        return self.tp / self.n
        
class DetAnResult:
    def __init__(self, seq_name):
        self.name = seq_name
        self.results: dict[int, ResultNode2] = {}
        self.emptyResult = ResultNode2(0,0,0,0)
    
    def addResults(self, threshold: int, results: ResultNode2):
        self.results[threshold] = results
    
    def getResults(self, threshold):
        if(threshold in self.results):
            return self.results[threshold]
        else: return self.emptyResult
    
    def __str__(self):
        return "%s: results: %s"%(self.name, str(self.results))
    def __repr__(self):
        return self.__str__()

class DetAnResult2:
    def __init__(self, seq_name: str):
        self.sequence_dets = SequenceDets(seq_name)
        pass

class SequenceDets:
    def __init__(self, seq_name: str, time_off=1):
        # load detections and groundtruth from file
        detsp = DETS_DIR
        gtp = OPTIONS["gt_dir"]
        # print(seq_name)
        det_filename = "%s%s.txt"%(detsp,seq_name)
        gt_filename = "%s%s_gt.txt"%(gtp,seq_name)

        # print(det_filename)
        # print(gt_filename)
        self.time_off = time_off
        self.D13 = readDetFile2(det_filename, time_off=time_off)
        self.GT13 = readDetFile2(gt_filename, time_off=time_off)
        self.valid = True
        if(len(self.GT13) != len(self.D13)):
            print("SequenceDets: [WARN] length of detections not equal to that of gt")
            self.valid = False
        self.seq_len = len(self.GT13)
    
    def getDetsAtI(self, i:int):
        return self.D13[i]
    
    def getGtAtI(self, i:int):
        return self.GT13[i]
    
    def getDetGtAtI(self, i):
        return self.getDetsAtI(i), self.getGtAtI(i)

def getResAvg(results: list[DetAnResult],threshold: float):
    threshold = int(threshold*100)
    avg_percentage = 0
    n_res = 0
    for r in results:
        if(threshold in r.results):
            n_res = n_res + 1
            result_thr =r.results[threshold]
            percentage = result_thr.getPercentPr()
            # print(result_thr, percentage)
            avg_percentage = avg_percentage + percentage
    if(n_res > 0):
        avg_percentage = avg_percentage/n_res
    return avg_percentage

def getThrRange(step):
    thresholds = np.arange(0,1+step,step)
    return thresholds

# load dets, load gt
# load frame
OPTIONS = getOptionNamespaceWdefaultInit("detan", 'options.ini')
OPTINOS_MAIN = getOptionNamespaceWdefaultInit("main2", 'options.ini') # for workspace dir
RESULT_PATH=OPTINOS_MAIN['result_path']
DETS_DIR = OPTIONS['dets_dir']
RESULTS_DIR = OPTIONS['results_dir']
SEQUENCES_DIR = OPTIONS['sequences_dir']
SAVE_RESULTS=OPTIONS['save_results']

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

SEQUENCE_NAMES_LIST = get_sequence_list()

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

def getMinMaxAvgIoU2(dets, gt):
    ret_iou = (0,0,0,0)
    if(dets):
        min_iou = 1
        iou_changed = False
        max_iou = 0
        avg_iou = 0
        rand_iou = 0
        n = 0

        rand_int = random.randrange(len(dets))
        i = 0
        for d in dets:
            current_iou = IoU(d, gt)
            if(current_iou > 0): # those that are inside
                if(current_iou < min_iou ):
                    # min_det = d
                    min_iou = current_iou
                    iou_changed = True
                if(current_iou > max_iou ):
                    # min_det = d
                    max_iou = current_iou
                avg_iou = avg_iou + current_iou
                n = n+1
            if(i == rand_int):
                rand_iou = current_iou
            i = i + 1
        
        if(n != 0):
            avg_iou = avg_iou/n
        else: avg_iou = 0.0
        if(not iou_changed):
            min_iou = 0
        ret_iou = (min_iou, max_iou, avg_iou, rand_iou)
    return ret_iou

def read_detAn(filename: str):
    with open(filename, 'rb') as fp:
        thr, results = pickle.load(fp)
    
    print("threshold: %f"%thr)
    print("results: ")
    for r in results:
        print(r)

def read_list(filename: str):
    lines: list[str] = []
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i][:-1]
    return lines

def getResultsFromFile(filename:str):
    with open(filename, 'rb') as fp:
        results = pickle.load(fp)
        return results

def ourAvg(list):
    return np.mean(list)

def pr_seq(seq_name:str):
    pass

# returns true if tp, false if fp (og)
def isTp(det: Detection, gt: Detection, threshold:float):
    iou = 0
    if(gt is not None):
        iou = IoU(det, gt)
    # else: return None # to handle 'object-specific' case
    if(iou >= threshold):
        return True
    return False

# like ^, also return IoU
def isTp2(det: Detection, gt: Detection, threshold:float):
    iou = 0
    if(gt is not None):
        iou = IoU(det, gt)
    # else: return None # to handle 'object-specific' case
    if(iou >= threshold):
        return True, iou
    return False, iou

def getTpFpFnCount(det_l:list[Detection] = [], gt:Detection = None, thr:float = 0.5):
    n = len(det_l)
    fp = 0
    fn = 0
    tp = 0

    if(n > 0): # we have dets, they are either tp or fp
        for d in det_l:
            tp_check = isTp(d, gt,thr) # og way
            if(tp_check): tp = tp + 1
            else: 
                fp = fp + 1
        

    else: # we do not have dets, this is potentially fn
        if(gt is not None):
            fn = fn + 1
    return tp, fp, fn, n

def getTpFpFnCount2(det_l:list[Detection] = [], gt:Detection = None, thr:float = 0.5):
    n = len(det_l)
    fp = 0
    fn = 0
    tp = 0

    fn2 = 0 # not tp, if iou == 0: +1
    fp2 = 0 

    if(n > 0): # we have dets, they are either tp or fp
        inc_fn2 = True
        for d in det_l:
            iou = 0
            tp_check, iou = isTp2(d, gt,thr)
            if(tp_check): tp = tp + 1
            else: 
                fp = fp + 1
                if(not np.isclose(iou, 0)): # kukr hitr mamo eno k ma iou > 0 ni vec to fn + inc fp
                    inc_fn2 = False
                    fp2 = fp2 + 1
        if(inc_fn2):
            fn2 = fn2+1

    else: # we do not have dets, this is potentially fn
        if(gt is not None):
            fn = fn + 1
            fn2 = fn2 + 1
    return tp, fp, fn, n, fp2, fn2



def pr():
    global SAVE_RESULTS
    print("this is pr:")
    detsd = OPTIONS["dets_dir"]
    gt_pathd = OPTIONS["gt_dir"]
    thr = 0.75 # avg pr, r, r2: 0.032, 0.941, 0.360
    # thr = 0.5 # avg pr, r, r2: 0.087, 0.978, 0.486
    # thr = 0.25 # avg pr, r, r2: 0.120, 0.987, 0.548
    print("threshold: %f"%thr)

    # for seq_name in [SEQUENCE_NAMES_LIST[0]]:
    avg_p, avg_r, avg_r2 = 0, 0, 0
    n_avg = 0
    result_list = []
    for seq_name in SEQUENCE_NAMES_LIST:
        # seq_name = SEQUENCE_NAMES_LIST[0]
        seq1 = SequenceDets(seq_name)
        
        seq1_n_dets = 0
        seq1_tp = 0
        seq1_fp = 0
        seq1_fn = 0
        
        seq1_fp2 = 0
        seq1_fn2 = 0

        if(seq1.valid):
            for i in range(seq1.seq_len):
                dets_i, gt_i = seq1.getDetGtAtI(i)
                gt = None
                if(gt_i):
                    gt = gt_i[0]

                # tp,fp,fn,n = getTpFpFnCount(dets_i, gt, thr)
                tp,fp,fn,n, fp2, fn2 = getTpFpFnCount2(dets_i, gt, thr)
                seq1_n_dets += n
                seq1_fp += fp
                seq1_fn += fn
                seq1_tp += tp

                seq1_fp2 += fp2
                seq1_fn2 += fn2
                
            seq1_result = ResultNode2(seq1_n_dets, int(thr*10), seq1_tp, seq1_fp, seq1_fn, seq1_fp2, seq1_fn2)
            # print(seq1_result)

            # print("fn: %d"%seq1_result.getFn())
            pr = seq1_result.getPrecision()
            r = seq1_result.getRecall()
            r2 = seq1_result.getRecall(use_2=True)
            if(np.isclose(pr, 0) and np.isclose(r, 0)):
                pass
            else:
                avg_p += pr
                avg_r += r
                avg_r2 += r2
                n_avg += 1
                print("%s: p: %f, r: %f, r2: %f"%(seq_name, pr,r, r2))
                result_list.append(seq1_result)
            
            # print("precision: %f"%seq1_result.getPrecision())
            # print("recall: %f"%seq1_result.getRecall())
        else:
            print("%s: no valid results"%seq_name)
    print("avg. pr, r, r2: %f, %f, %f"%(avg_p/n_avg, avg_r/n_avg, avg_r2/n_avg))
    if(SAVE_RESULTS):
        print("saving results")
        with open(RESULT_PATH+'det_precision_recall.p', 'wb') as fp:
            abc = [avg_p, avg_r, n_avg, result_list]
            pickle.dump(abc, fp)
    # save results:




def main():
    global SAVE_RESULTS
    print("this is test_main:")
    resultsd = OPTIONS["results_dir"]
    gt_pathd = OPTIONS["gt_dir"]
    thr_step = 0.1
    thresholds = getThrRange(thr_step)
    # print(thresholds)
    # return
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
    
    # results_dirs = [results_dirs[0]]
    sequence_results: list[DetAnResult] = []
    # print(results_dirs)
    # for d in results_dirs:
    all_res_dirs = len(results_dirs)
    used_sequences = []
    for i in range(all_res_dirs):
        d = results_dirs[i]
        print("using sequence %s [%d / %d (%2.0f%%)]:"%(d, i, all_res_dirs, (float(i)/float(all_res_dirs))*100))
        # get trajectory file
        # get gt file
        result_dets = readDetFile2("%s%s/%s_001.txt"%(resultsd, d, d))
        gt_dets = readDetFile2("%s%s_gt.txt"%(gt_pathd, d))
        
        result_dets_len = len(result_dets)
        gt_dets_len = len(gt_dets)
        if(result_dets_len == gt_dets_len):
            used_sequences.append(d)
            sequence_result = DetAnResult(d)
            n_all = gt_dets_len
            for thr in thresholds:
                t_global = 1
                # print(gt_dets)
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
                            if(iou >= thr):
                                n_tp = n_tp + 1
                            else:
                                n_fp = n_fp + 1
                        n = n+1
                    t_global = t_global+1
                    
                seq_result = ResultNode2(n, thr,n_tp, n_fp)
                sequence_result.addResults(int(thr*100), seq_result)
            sequence_results.append(sequence_result)
        print("done with %s."%(d))
    for sr in sequence_results:
        print(sr)
    
    if(SAVE_RESULTS):
        print("saving results")
        with open(RESULT_PATH+'sequence_results.p', 'wb') as fp:
            abc = sequence_results
            pickle.dump(abc, fp)
    print("used sequences (%d):"%(len(used_sequences)))
    for s in used_sequences:
        print(s)

def main_use_dets():
    global SAVE_RESULTS
    print("this is main_use_dets:")
    detsd = OPTIONS["dets_dir"]
    gt_pathd = OPTIONS["gt_dir"]
    thr_step = 0.1
    thresholds = getThrRange(thr_step)

    # det files:
    det_files = os.listdir(detsd)
    det_files2 = []
    filenames: list[str] = []
    for d in det_files:
        # if(os.path.isdir("%s%s"%(detsd, d))):
        if(d.endswith(".txt")):
            # print(d)
            filenames.append(d)
    
    sequences = read_list(RESULTS_DIR+'list.txt')
    # print(sequences)
    filenames.sort()
    filenames2 = []
    # filenames = [filenames[0]]
    for r in filenames:
        r_wo_txt = r.split('.')[0]
        if(r_wo_txt in sequences):
            det_files2.append(r_wo_txt)
            filenames2.append(r)
            # print(r)
    filenames = filenames2
    

    n_th = 0
    # filenames = [filenames[n_th]]
    # results_dirs2 = [results_dirs2[n_th]]
    score_types = [0,1,2,3]
    sequence_results: list[DetAnResult] = []
    for s in score_types:
        sequence_results.append([])

    # print(results_dirs)
    # for d in results_dirs:
    # all_res_dirs = len(filenames)
    all_res_dirs = len(filenames)
    for i in range(all_res_dirs):
        d = filenames[i]
        d2 = det_files2[i]
        print("using sequence %s [%d / %d (%2.0f%%)]:"%(d2, i, all_res_dirs, (float(i)/float(all_res_dirs))*100))
        # get trajectory file
        # get gt file
        result_dets = readDetFile2("%s%s"%(detsd, d))
        # result_dets = readDetFile2("%s%s/%s_001.txt"%(resultsd, d2, d2))
        gt_dets = readDetFile2("%s%s_gt.txt"%(gt_pathd, d2))
        
        result_dets_len = len(result_dets)
        gt_dets_len = len(gt_dets)
        if(result_dets_len == gt_dets_len):
            sequence_result_i: list[DetAnResult] = []
            for s in score_types:
                # sequence_result_i.append(DetAnResult("%s_%d"%(d2, s)))
                sequence_result_i.append(DetAnResult("%s"%(d2)))
            n_all = gt_dets_len
            for thr in thresholds:
                t_global = 1
                # print(gt_dets)
                results_types=[]
                # init
                for s in score_types:
                    # [n,n_tp,n_fp]
                    results_types.append([0,0,0])
                # n = 0
                # n_tp = 0
                # n_fp = 0
                while(t_global < n_all):
                    
                    current_dets = result_dets[t_global]
                    current_gt = gt_dets[t_global]
                    if(current_dets and current_gt):
                        current_gt = current_gt[0]
                        min_max_avg_iou = getMinMaxAvgIoU2(current_dets, current_gt)
                        for s in score_types:
                            iou = min_max_avg_iou[s]
                            if(iou > 0):
                                if(iou >= thr):
                                    results_types[s][0] = results_types[s][0]+1
                                    # n_tp = n_tp + 1
                                else:
                                    results_types[s][1] = results_types[s][1]+1
                                    # n_fp = n_fp + 1
                            results_types[s][2] = results_types[s][2]+1 # n++
                            # results_types[s][2] = results_types[s][2]+1
                            # n = n+1
                    t_global = t_global+1
                
                for s in score_types:
                    seq_result_i = ResultNode2(results_types[s][2], thr,results_types[s][0], results_types[s][1])
                    
                    sequence_result_i[s].addResults(int(thr*100), seq_result_i)
            # sequence_results.append(sequence_result)
            for s in score_types:
                sequence_results[s].append(sequence_result_i[s])

        print("done with %s."%(d2))
    for sr in sequence_results:
        print(sr)
    
    if(SAVE_RESULTS):
        print("saving results")
        with open(RESULT_PATH+'dets_results_all.p', 'wb') as fp:
            abc = sequence_results
            pickle.dump(abc, fp)

if __name__ == '__main__':
    mode = int(OPTIONS["mode"])
    mode = 1
    if(mode == 0):
        main()
    if(mode == 1):
        pr()
