import cv2 as cv
import numpy as np
from helper_func import *
import os
import random


# class to store results of deteciton analysis
class ResultNode:
    def __init__(self, n, threshold, tp, fp):
        self.threshold = threshold
        self.n = n
        self.tp = tp
        self.fp = fp
        
        # kinda don't have these
        self.tn = 0.0
        self.fn = 0.0
    
    def getTp(self):
        return self.tp
    
    def getFp(self):
        return self.fp
    
    def getN(self):
        return self.n
    
    def __str__(self):
        return "all: %d, tp: %d, fp: %d"%(self.n, self.tp, self.fp)
    
    def __repr__(self):
        return "{%d,%d,%d}"%(self.n, self.tp, self.fp)
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

    def getPercentPr(self):
        return self.tp / self.n
        
class DetAnResult:
    def __init__(self, seq_name):
        self.name = seq_name
        self.results: dict[int, ResultNode] = {}
        self.emptyResult = ResultNode(0,0,0,0)
    
    def addResults(self, threshold: int, results: ResultNode):
        self.results[threshold] = results
    
    def getResults(self, threshold):
        if(threshold in self.results):
            return self.results[threshold]
        else: return self.emptyResult
    
    def __str__(self):
        return "%s: results: %s"%(self.name, str(self.results))
    def __repr__(self):
        return self.__str__()

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

def getMinMaxAvgIoU(dets, gt, iou_type=0):
    # iou_types: 0 - min, 1 - max, 2 - avg
    ret_iou = 0
    if(dets):
        if(iou_type == 0 or iou_type == 1):
            # min_det = dets[0]
            min_iou = 1-iou_type
            iou_changed = False
            for d in dets:
                current_iou = IoU(d, gt)
                if(current_iou > 0 and ((iou_type == 0 and current_iou < min_iou) or (iou_type == 1 and current_iou > min_iou)) ):
                    # min_det = d
                    min_iou = current_iou
                    iou_changed = True
            if(iou_changed):
                ret_iou = min_iou
        if(iou_type == 2):
            avg_iou = 0
            n = 0
            for d in dets:
                current_iou = IoU(d, gt)
                if(current_iou > 0):
                    print(current_iou, end = " ")
                    avg_iou = avg_iou + current_iou
                    n = n+1
            if(n != 0):
                avg_iou = avg_iou/n
                ret_iou = avg_iou
                print(ret_iou)
            print(ret_iou)
    return ret_iou

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


def main2():
    return

#         [main2]
# ; path to results (some kind of tracker workspace)
# RESULT_PATH=./tracker/

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
                    
                seq_result = ResultNode(n, thr,n_tp, n_fp)
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
        result_dets = readDetFile2("%s%s"%(detsd, d))
        # result_dets = readDetFile2("%s%s/%s_001.txt"%(resultsd, d2, d2))
        # get gt file
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
                                if(iou >= thr): # tp
                                    results_types[s][0] = results_types[s][0]+1
                                    # n_tp = n_tp + 1
                                else: # fp
                                    results_types[s][1] = results_types[s][1]+1
                                    # n_fp = n_fp + 1
                            # fn?
                            results_types[s][2] = results_types[s][2]+1 # n++ -> all
                            # results_types[s][2] = results_types[s][2]+1
                            # n = n+1
                    t_global = t_global+1
                
                for s in score_types:
                    seq_result_i = ResultNode(results_types[s][2], thr,results_types[s][0], results_types[s][1])
                    
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

def getResultsFromFile(filename:str):
    with open(filename, 'rb') as fp:
        results = pickle.load(fp)
        return results

def showResults():
    print("this is show results")

    # graph legend:
    g_legend = []
    # get results:
    results: list[list[DetAnResult]] = None
    
    # load results of detections:
    results = getResultsFromFile(RESULT_PATH+'dets_results_all_rand_gt.p')
    print(results[0][0])
    g_legend = g_legend + ['dmin','dmax', 'davg', 'drand']

    # load results of results:
    results2 = getResultsFromFile(RESULT_PATH+'sequence_results_09.p')
    results.append(results2) # add it to all results
    g_legend.append('IoU + feat')
    
    # load results of results:
    results2 = getResultsFromFile(RESULT_PATH+'sequence_results_09_gt.p')
    results.append(results2) # add it to all results
    g_legend.append('IoU + feat + gt')
    
    # load results of results:
    results2 = getResultsFromFile(RESULT_PATH+'sequence_results_10.p')
    results.append(results2) # add it to all results
    g_legend.append('IoU-only')
    
    # load results of results:
    results2 = getResultsFromFile(RESULT_PATH+'sequence_results_00.p')
    results.append(results2) # add it to all results
    g_legend.append('feat-only')

    # load results of results:
    results2 = getResultsFromFile(RESULT_PATH+'sequence_results_seqTrack.p')
    results.append(results2) # add it to all results
    g_legend.append('seqTrack')
    
    
    
    # results = [results]
    sequences = read_list(RESULTS_DIR+'list.txt')
    results22: list[list[DetAnResult]] = []
    print(sequences)
    # print(results)
    for r in results:
        results2: list[DetAnResult] = []
        for rr in r:
            if(rr.name in sequences):
                results2.append(rr)
        results22.append(results2)
    # print(results22)
    results = results22
    
    # print(results)
    dx = 0.1
    thresholds = getThrRange(dx)
    all_avgs = []
    for thr in thresholds:
        avgs = []
        for r in results:
            thr_avg = getResAvg(r, thr)
            avgs.append(thr_avg)
            print("%4.2f"%thr_avg, end=" ", flush=True)
        print()
        all_avgs.append(avgs)
    
    all_avgs = np.array(all_avgs)
    print(all_avgs[:,0])
    m,n = np.shape(all_avgs)
    
    VISUAL = OPTIONS['visual']
    if(VISUAL):
        from matplotlib import pyplot as plt
        # fixed points:
        x = [0,0,1]
        y = [1,0,0]
        
        # also show graph
        fig, ax = plt.subplots(1,1)
        ax.axis('equal')
        ax.set_xlabel('threshold')
        ax.set_ylabel('success')
        
        for i in range(n):
            alpha = 1.0
            opts = ''
            if(i in [1,2,3]):
                alpha = 0.50
            if(i in [0,1,2,3]):
                opts = '--'
            ax.plot(thresholds, all_avgs[:,i], opts, alpha=alpha, linewidth=1.0)
            # print(np.trapz(all_avgs[:,i], thresholds, dx))
            print("sequence: %s, quality: %.4f"%(g_legend[i],np.trapz(all_avgs[:,i], thresholds, dx)))
        ax.legend(g_legend)

        # draw axes
        ax.plot(x, y, 'k--', alpha=0.2)

        plt.show()
        fig.savefig("success-threshold.svg")

def ourAvg(list):
    # return np.mean(list) * (1- 0.1*0.67327)
    # return np.mean(list) * (1+ 0.033)
    return np.mean(list)
    # return np.sum(list) / len(list)

def showResults2():
    print("this is show results")

    # graph legend:
    g_legend = []
    # get results:
    results: list[list[DetAnResult]] = None
    
    # load results of detections: 0, 1, 2, 3
    results = getResultsFromFile(RESULT_PATH+'dets_results_all_rand_gt.p')
    print(results[0][0])
    g_legend = g_legend + ['HypMin','HypMax', 'HypAvg', 'HypRand']

    # load results of results: 4
    results2 = getResultsFromFile(RESULT_PATH+'sequence_results_09.p')
    results.append(results2) # add it to all results
    # g_legend.append('IoU + feat')
    g_legend.append('GMH&SB')
    
    # load results of results: 5
    # results2 = getResultsFromFile(RESULT_PATH+'sequence_results_09_gt.p')
    # results.append(results2) # add it to all results
    # g_legend.append('IoU + znač. + gt')
    
    # load results of results: 6
    results2 = getResultsFromFile(RESULT_PATH+'sequence_results_10.p')
    results.append(results2) # add it to all results
    g_legend.append('GMH&SB - IoU')
    
    # load results of results: 7
    results2 = getResultsFromFile(RESULT_PATH+'sequence_results_00.p')
    results.append(results2) # add it to all results
    g_legend.append('GMH&SB - viz')

    # load results of results: 8
    # results2 = getResultsFromFile(RESULT_PATH+'sequence_results_seqTrack.p')
    # results.append(results2) # add it to all results
    # g_legend.append('SeqTrack')

    # results:
    # 0 hypmin
    # 1 hypmax
    # 2 hypavg
    # 3 hyprand
    # 4 gmhnsb (0.9)
    # 5 gmhnsb (1.0) (iou)
    # 6 gmhnsb (0.0) (viz)
    
    
    
    # results = [results]
    sequences = read_list(RESULTS_DIR+'list.txt')
    results22: list[list[DetAnResult]] = []
    # print(sequences)
    # print(results)
    for r in results:
        results2: list[DetAnResult] = []
        for rr in r:
            if(rr.name in sequences):
                results2.append(rr)
        results22.append(results2)
    # print(results22)
    results = results22
    # print(results)
    
    # print(results)
    dx = 0.1
    thresholds = getThrRange(dx)
    all_avgs = []
    for thr in thresholds:
        avgs = []
        for r in results:
            thr_avg = getResAvg(r, thr)
            avgs.append(thr_avg)
            print("%4.2f"%thr_avg, end=" ", flush=True)
        print()
        all_avgs.append(avgs)
    
    all_avgs = np.array(all_avgs)
    print(all_avgs[:,0])
    m,n = np.shape(all_avgs)
    
    VISUAL = OPTIONS['visual']
    if(VISUAL):
        from matplotlib import pyplot as plt
        # fixed points:
        x = [0,0,1]
        y = [1,0,0]
        
        # also show graph
        fig1, ax1 = plt.subplots(1,1)
        fig2, ax2 = plt.subplots(1,1)
        fig3, ax3 = plt.subplots(1,1)
        fig4, ax4 = plt.subplots(1,1)
        for ax in [ax1, ax2, ax3, ax4]:
            ax.axis('equal')
            ax.set_xlabel('τ')
            ax.set_ylabel('SR')
            ax.margins(x=0, y=0)
            ax.set_box_aspect(1)
        
        ax1_legend = []
        ax2_legend = []
        ax3_legend = []
        ax4_legend = []



        for i in range(n):
            alpha = 1.0
            opts = ''
            if(i in [1,2,3]):
                # alpha = 0.50
                alpha = 1.0
            if(i in [0,1,2,3]):
                opts = '--'
            
            # if(i <= 4):
            # figure 1
            if(i <= 4 and i in [1,3,4]):
            # if(i <= 4 and i in [1,4]):
            # if(i <= 4 and i in [1,3,4]):
                ax1.set_title("Graf SR(τ)")
                ax1.plot(thresholds, all_avgs[:,i], opts, alpha=alpha, linewidth=1.0)
                ax1_legend.append(g_legend[i])
                # print(np.trapz(all_avgs[:,i], thresholds, dx))
                # print("sequence: %s, quality: %.4f, %.4f"%(g_legend[i],np.trapz(all_avgs[:,i], thresholds), np.mean(all_avgs[:,i])))
                print("sequence: %s, quality: %.4f, %.4f"%(g_legend[i],np.trapz(all_avgs[:,i], thresholds), ourAvg(all_avgs[:,i])))
            
            # figure 4
            if(i <= 4 and i in [3,4]):
                ax4.set_title("Graf SR(τ)")
                ax4.plot(thresholds, all_avgs[:,i], opts, alpha=alpha, linewidth=1.0)
                ax4_legend.append(g_legend[i])
                # print(np.trapz(all_avgs[:,i], thresholds, dx))
                # print("sequence: %s, quality: %.4f, %.4f"%(g_legend[i],np.trapz(all_avgs[:,i], thresholds), np.mean(all_avgs[:,i])))
                print("sequence: %s, quality: %.4f, %.4f"%(g_legend[i],np.trapz(all_avgs[:,i], thresholds), ourAvg(all_avgs[:,i])))
            
            # figure 2
            if(i in [4, 8]):
                ax2.set_title("Graf SR(τ)")
                ax2.plot(thresholds, all_avgs[:,i], opts, alpha=alpha, linewidth=1.0)
                ax2_legend.append(g_legend[i])
                # print(np.trapz(all_avgs[:,i], thresholds, dx))
                print("sequence: %s, quality: %.4f, %.4f"%(g_legend[i],np.trapz(all_avgs[:,i], thresholds), ourAvg(all_avgs[:,i])))
            
            # figure 3
            if(i in [4,5,6,7]):
                ax3.set_title("Graf SR(τ)")
                ax3.plot(thresholds, all_avgs[:,i], opts, alpha=alpha, linewidth=1.0)
                legend_entry = g_legend[i]
                # if(i == 4):
                #     legend_entry = "IoU + znač."
                ax3_legend.append(legend_entry)
                # print(np.trapz(all_avgs[:,i], thresholds, dx))
                print("%s - quality: %.4f, %.4f"%(legend_entry,np.trapz(all_avgs[:,i], thresholds), ourAvg(all_avgs[:,i])))
        ax1.legend(ax1_legend)
        ax2.legend(ax2_legend)
        ax3.legend(ax3_legend)
        ax4.legend(ax4_legend)

        # draw axes
        ax1.plot(x, y, 'k--', alpha=0.0)
        ax2.plot(x, y, 'k--', alpha=0.0)
        ax3.plot(x, y, 'k--', alpha=0.0)
        ax4.plot(x, y, 'k--', alpha=0.0)

        plt.show()
        # fig.savefig("success-threshold.svg")
        if(True):
            fig1.savefig("dets_alg_graph.svg")
            fig2.savefig("alg_seqtrack.svg")
            fig3.savefig("alg_different_a.svg")
            fig4.savefig("dets_alg_graph_2.svg")

        

        


if __name__ == '__main__':
    mode = int(OPTIONS["mode"])
    if(mode == 0):
        # analyze:
        main()
    elif(mode == 1):
        # show results:
        read_detAn(RESULT_PATH+"detAn.p")
    elif(mode == 2):
        main2()
    elif(mode == 3):
        showResults()
    elif(mode == 4):
        # analyze:
        main_use_dets()
    elif(mode == 5):
        # analyze:
        showResults2()
