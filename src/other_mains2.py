import numpy as np
from helper_func import *
from DetectionSpace import DetectionSpace
from Detection import Detection
from Trajectory import Trajectory
from kf import Kalman
import copy
# from main2 import getMergedHypotheses, computeOrGetFlowAtI
# from main2 import USE_PRECOMPUTED_FLOW as UPF, PRECOMPUTED_FLOW_PATH as PFP
import main2

def mainInitSelect(n: int, options: dict):
    global OPTIONS
    OPTIONS = options
    if(n == 6):
        main6()
    if(n == 7):
        main7()
    if(n == 8):
        main8()
    if(n == 9):
        main9()
    if(n == 10):
        mainX()
    if(n == 11):
        mainXI()

def setFrameFlowAtI(dspace: DetectionSpace, i: int, flow_path="", frames_path=""):
    # print("fpath: %s"%frames_path)
    # set new frame
    frame = getFrameAtI(i,frames_path)
    dspace.lastFrame = frame.copy()
    dspace.map = frame
    
    # set new flow
    # flow_from = getFlowAtI(i, DATA_ROOT+FLOWS_PATH)
    flow_from = getFlowAtI(i, flow_path)
    dspace.flow_map = flow_from
    flow_img = flow2img(flow_from)
    dspace.last_flow_img = flow_img.copy()
    dspace.flow_img = flow_img
    
    #  also set flow to this image
    if(i > 1):
        flow_to = getFlowAtI(i-1, flow_path)
        dspace.flow_to = flow_to
    else: # there is no flow into first frame
        h,w = dspace.hw
        dspace.flow_map=np.zeros((h,w,2)).astype(np.uint8) # optical flow (outta this frame)
    return frame

def getTrFromResultFile(dspace, filename, time_off = 1, color = [0,0,255]):
    tr_dets = readDetFile2(filename, time_off)
    if(not tr_dets[0]):
        second_det = tr_dets[1][0]
        zero_det = Detection(second_det.x,time_off,second_det.bb)
        tr_dets[0].append(zero_det)
    
    print(tr_dets[0:10])
    tr1 = Trajectory(tr_dets[0][0], dspace, color=color)
    tr1.build2()
    tr1.X = [x[0] for x in tr_dets]
    return tr1


def mainXI():
    # global USE_PRECOMPUTED_FLOW
    # global PRECOMPUTED_FLOW_PATH
    print("this is mainXI")
    print("options:")
    # print(OPTIONS)
    sequence_str = OPTIONS['sequence']
    T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getSequenceConsts(sequence_str)
    print(FRAMES_PATH)
    
    
    main2.PRECOMPUTED_FLOW_PATH = PRECOMPUTED_FLOW_PATH
    main2.PRECOMPUTED_FEAT_PATH = PRECOMPUTED_FEAT_PATH
    main2.USE_PRECOMPUTED = True

    # INIT
    # D13 = readDetFile2(DETS_FILE) # read detections from file
    G13 = readDetFile2(GT_PATH)
    D0 = readDetFile2(DETS_FILE)
    # for d_l in G13:
    #     if(d_l):
    #         d_l[0].color=[0,0,255]
    # D13: list[list[Detection]] = [[]]+D13 # offset it (because time starts at 1, there is nothing on 0)
    D13: list[list[Detection]] = [[]]+G13 # offset it (because time starts at 1, there is nothing on 0)
    D0: list[list[Detection]] = [[]]+D0 # offset it (because time starts at 1, there is nothing on 0)
    skip_n=0
    # skip_n=1108
    T_OFFSET = T_OFFSET + skip_n
    t_global = T_OFFSET # current time (frame)
    
    
    
    n = 0 # number of previous frames for trajectory init (not needed)
    n_res = len(D13) - n - T_OFFSET
    # n_all = n_res
    n_all = len(D13)
    print(n_all)


    # INIT DSPACE
    frame = getFrameAtI(t_global,FRAMES_PATH)
    frame_feat_padding: tuple[int, int, int, int] = getImagePadding(np.shape(frame)) # get feature image offsets
    h,w,_ = np.shape(frame)
    
    dspace: DetectionSpace = DetectionSpace(h,w, time_offset=(T_OFFSET-n), show_flow=False, disable_vis=False)
    dspace2: DetectionSpace = DetectionSpace(h,w, time_offset=(T_OFFSET-n), show_flow=False, disable_vis=False)
    
    # set last frame
    dspace.lastFrame = frame.copy()
    dspace.map = frame
    
    dspace2.lastFrame = frame.copy()
    dspace2.map = frame

    

    # print(D13[0:10])
    # print()
    # print(D13[t_global])
    d_at_t = D13[t_global]

    dspace.D.append(d_at_t)
    setFrameFlowAtI(dspace, t_global, PRECOMPUTED_FLOW_PATH, FRAMES_PATH)
    
    dspace2.D.append(d_at_t)
    setFrameFlowAtI(dspace2, t_global, PRECOMPUTED_FLOW_PATH, FRAMES_PATH)
    dspace.showSpace(draw_dets=False, draw_exit_zone=False, draw_last_dets_bb=True, bb_color=[0,0,255], use_waitkey=False)
    dspace2.showSpace(draw_dets=False, draw_exit_zone=False, draw_last_dets_bb=True, bb_color=[0,0,255])
    t_global = t_global +1

    #tracker1:
    # tr1_file = "/home/gasper/Faks/3_letnik/diplomska/koda/src/tracker/results_seq/bird2_latest/results.txt"
    # tr1 = getTrFromResultFile(dspace, tr1_file, T_OFFSET, color=[0,255,255])
    # start_t = tr1.X[0].t
    # print(tr1.X)
    # printTrWithStats(tr1)

    # tr_file = "/home/gasper/Faks/3_letnik/diplomska/koda/src/tracker/trs_bird2.p" # bird
    # tr_file = "/home/gasper/Faks/3_letnik/diplomska/koda/src/tracker/trs_gecko.p" # chameleon20
    # tr_file = "/home/gasper/Faks/3_letnik/diplomska/koda/src/tracker/trs_coin18.p" # coin18
    tr_file = "/home/gasper/Faks/3_letnik/diplomska/koda/src/tracker/trs_got10k.p"
    tr_n = readTrajectoryFile(tr_file)
    tr1: Trajectory = tr_n[2][0]
    tr1.detectionSpace = dspace
    # print(tr_n)
    start_t1 = tr1.X[0].t

    #tracker2:
    # tr1_file = "/media/gasper/Seagate Basic/Nedokumenti/Faks/results/SeqTrack-didi-results/SeqTrack/baseline/LaSOT_bird-2/LaSOT_bird-2_001.txt" #bird2
    # tr1_file = "/media/gasper/Seagate Basic/Nedokumenti/Faks/results/SeqTrack-didi-results/SeqTrack/baseline/LaSOT_chameleon-20/LaSOT_chameleon-20_001.txt" #chameleon20
    # tr1_file = "/media/gasper/Seagate Basic/Nedokumenti/Faks/results/SeqTrack-didi-results/SeqTrack/baseline/LaSOT_coin-18/LaSOT_coin-18_001.txt" # coin18
    tr1_file = "/media/gasper/Seagate Basic/Nedokumenti/Faks/results/SeqTrack-didi-results/SeqTrack/baseline/GOT-10k_GOT-10k_Val_000014/GOT-10k_GOT-10k_Val_000014_001.txt"

    tr2 = getTrFromResultFile(dspace2, tr1_file, T_OFFSET, color=[255,255,0])
    start_t2 = tr2.X[0].t

    # t_global_stops = [0,530,2236,2880]
    # t_i = 1
    n_stop = n_all
    # t_global = 855
    # t_global = 900
    t_global = 80
    only_second = not True
    while(t_global <= n_stop):
        print("at time: %d"%t_global)
        if(not only_second): setFrameFlowAtI(dspace, t_global, PRECOMPUTED_FLOW_PATH, FRAMES_PATH)
        frame = setFrameFlowAtI(dspace2, t_global, PRECOMPUTED_FLOW_PATH, FRAMES_PATH)
        gt_at_t = D13[t_global]
        if(not only_second): dspace.D.append(gt_at_t)
        dspace2.D.append(gt_at_t)
        dets_at_t = D0[t_global]
        if(not only_second):
            #draw dets:
            for d in dets_at_t:
                drawBoundingBox(dspace.map, d.bb, opacity=0.4)
            # t1
            tr1.drawToSpace(at_t=t_global, draw_ends=False)
            drawBoundingBox(dspace.map, tr1.X[t_global - start_t1].bb, color=[0,255,255])
            drawX(dspace.map, tr1.X[t_global - start_t1].x)
        # t2
        tr2.drawToSpace(at_t=t_global, draw_ends=False)
        drawBoundingBox(dspace2.map, tr2.X[t_global - start_t2].bb, color=[255,255,0])
        drawX(dspace2.map, tr2.X[t_global - start_t2].x)
        if(not only_second): dspace.showSpace(draw_dets=False, draw_last_dets_bb=True, draw_exit_zone=False, bb_color=[0,0,255], use_waitkey=False) #use_waitkey is false, so it does not block this one (waits after other dspace finishes drawing)
        dspace2.showSpace(draw_dets=False, draw_last_dets_bb=True, draw_exit_zone=False, bb_color=[0,0,255])
        if(not only_second): dspace.clearSpace()
        dspace2.clearSpace()
        t_global = t_global +1


def mainX():
    # global USE_PRECOMPUTED_FLOW
    # global PRECOMPUTED_FLOW_PATH
    print("this is mainX")
    print("options:")
    # print(OPTIONS)
    sequence_str = OPTIONS['sequence']
    T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getSequenceConsts(sequence_str)
    print(FRAMES_PATH)
    
    
    main2.PRECOMPUTED_FLOW_PATH = PRECOMPUTED_FLOW_PATH
    main2.PRECOMPUTED_FEAT_PATH = PRECOMPUTED_FEAT_PATH
    main2.USE_PRECOMPUTED = True

    # INIT
    D13 = readDetFile2(DETS_FILE) # read detections from file
    D13: list[list[Detection]] = [[]]+D13 # offset it (because time starts at 1, there is nothing on 0)
    skip_n=0
    skip_n=1108
    T_OFFSET = T_OFFSET + skip_n
    t_global = T_OFFSET # current time (frame)
    
    
    
    n = 0 # number of previous frames for trajectory init (not needed)
    n_res = len(D13) - n - T_OFFSET
    # n_all = n_res
    n_all = len(D13)
    print(n_all)


    # INIT DSPACE
    frame = getFrameAtI(t_global,FRAMES_PATH)
    frame_feat_padding: tuple[int, int, int, int] = getImagePadding(np.shape(frame)) # get feature image offsets
    h,w,_ = np.shape(frame)
    
    dspace: DetectionSpace = DetectionSpace(h,w, time_offset=(T_OFFSET-n), show_flow=False, disable_vis=False)
    
    # set last frame
    dspace.lastFrame = frame.copy()
    dspace.map = frame

    # print(D13[0:10])
    # print()
    # print(D13[t_global])
    d_at_t = D13[t_global]

    dspace.D.append(d_at_t)
    setFrameFlowAtI(dspace, t_global, PRECOMPUTED_FLOW_PATH, FRAMES_PATH)
    dspace.showSpace(draw_dets=False, draw_exit_zone=False)
    t_global = t_global +1

    n_stop = n_all
    while(t_global <= n_stop):
        print("at time: %d"%t_global)
        frame = setFrameFlowAtI(dspace, t_global, PRECOMPUTED_FLOW_PATH, FRAMES_PATH)
        d_at_t = D13[t_global]

        # if(t_global == 126 or t_global == 127):
        #     d_at_t = [d_at_t[2]]
        dspace.D.append(d_at_t)

        
        frame_features = main2.computeOrGetPCAFeaturesAtI(t_global, frame, None, save_feat=False)
        # add features to detections:
        for d_i in d_at_t:
            patches = getFeaturesFromFeatureMapAndPadding3(d_i, feature_map_and_padding=(frame_features, frame_feat_padding))
            showInNamed("d%d"%(d_i.id), patches.astype(np.uint8))
            d_i.visual_feat = patches

        
            
        dspace.showSpace(draw_dets=False, draw_last_dets_bb=True, draw_exit_zone=False)
        t_global = t_global +1



def main9():
    # global USE_PRECOMPUTED_FLOW
    # global PRECOMPUTED_FLOW_PATH
    print("this is main9")
    print("options:")
    # print(OPTIONS)
    sequence_str = OPTIONS['sequence']
    T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getSequenceConsts(sequence_str)
    print(FRAMES_PATH)
    
    
    main2.PRECOMPUTED_FLOW_PATH = PRECOMPUTED_FLOW_PATH
    main2.USE_PRECOMPUTED = True


    
    
    
    # INIT
    D13 = readDetFile2(DETS_FILE) # read detections from file
    D13: list[list[Detection]] = [[]]+D13 # offset it (because time starts at 1, there is nothing on 0)
    skip_n=0
    skip_n=2064
    T_OFFSET = T_OFFSET + skip_n
    t_global = T_OFFSET # current time (frame)
    
    n = 0 # number of previous frames for trajectory init (not needed)
    n_res = len(D13) - n - T_OFFSET
    # n_all = n_res
    n_all = len(D13)
    print(n_all)


    # INIT DSPACE
    frame = getFrameAtI(t_global,FRAMES_PATH)
    h,w,_ = np.shape(frame)
    
    dspace: DetectionSpace = DetectionSpace(h,w, time_offset=(T_OFFSET-n), show_flow=False, disable_vis=False)
    
    # set last frame
    dspace.lastFrame = frame.copy()
    dspace.map = frame

    # print(D13[0:10])
    # print()
    # print(D13[t_global])
    d_at_t = D13[t_global]
    print(d_at_t)
    d_sel = []
    for d in d_at_t:
        if(d.id in {4394,4396}):
            d_sel.append(d)
    d_at_t = d_sel


    dspace.D.append(d_at_t)
    setFrameFlowAtI(dspace, t_global, PRECOMPUTED_FLOW_PATH, FRAMES_PATH)
    dspace.showSpace(draw_dets=False)
    t_global = t_global +1

    

    # make trajectories from dets at this time
    all_trs: list[Trajectory] = []
    d0 = d_at_t[0] # levi
    d1 = d_at_t[1] # desni
    

    tr0 = Trajectory(d0, dspace)
    tr0.build2()
    all_trs.append(tr0)
    
    tr01 = Trajectory(d0, dspace)
    tr01.build2()
    all_trs.append(tr01)
    
    tr1 = Trajectory(d1, dspace)
    tr1.build2()
    all_trs.append(tr1)
    
    tr11 = Trajectory(d1, dspace)
    tr11.build2()
    all_trs.append(tr11)

    
    n_stop = n_all
    n_stop = 2170
    
    while(t_global <= n_stop):
        print("at time: %d"%t_global)
        setFrameFlowAtI(dspace, t_global, PRECOMPUTED_FLOW_PATH, FRAMES_PATH)
        # d_at_t = D13[t_global]
        # dspace.D.append(d_at_t)

        # used_dets: set = {}
        
        if(t_global == 2080):
            tr11.term = True
        if(t_global == 2095):
            tr01.term = True
            # start new:
            
            td = copy.copy(tr1.X[-1])
            displacement = np.array([5,7])
            td.x = np.array(td.x) + displacement
            print(td.bb)
            td.bb[0:2] += displacement
            print(td.bb)
            # tr21 = Trajectory(tr1.X[-1], dspace)
            tr21 = Trajectory(td, dspace)
            tr21.build2()
            all_trs.append(tr21)


        for tr in all_trs:
            tr_color = tr.color
            if(not tr.term):
                used_dets, forks = tr.extend4()
            else:
                tr_color = [100,100,100]
            if(tr not in {tr1, tr0}):
                if(not tr.term):
                    drawBoundingBox(dspace.map, tr.nextStateEstimate.bb, [200,200,0])
                tr.drawToSpace(tr_color)
        
            
        dspace.showSpace(draw_dets=False, draw_last_dets_bb=False)
        t_global = t_global +1
    

    # time travel:
    t_global = 2095
    all_trs = [tr11, tr01, tr21]
    setFrameFlowAtI(dspace, t_global, PRECOMPUTED_FLOW_PATH, FRAMES_PATH)
    for tr in all_trs:
        tr.drawToSpace()
    drawBoundingBox(dspace.map, tr21.origin.bb)
    drawX(dspace.map, tr21.origin.x)
    dspace.showSpace(draw_dets=False, draw_last_dets_bb=False)
    dspace.clearSpace()


    # merge trs
    for tr in all_trs:
        tr.drawToSpace()
    dspace.showSpace(draw_dets=False, draw_last_dets_bb=False)
    dspace.clearSpace()
    a: list[Trajectory]  = main2.getMergedHypotheses(all_trs, type=2)
    printTrListWithStatsOrdered(a)
    
    for tr in a:
        if(tr.not_so_much_unique_id == 4):
            continue
        tr.drawToSpace()
    # drawBoundingBox(dspace.map, a[1].origin.bb)
    
    dspace.showSpace(draw_dets=False, draw_last_dets_bb=False, draw_exit_zone=False)



# bridging technique example
def main8():
    print("this is main8")
    print("options:")
    # print(OPTIONS)
    
    sequence_str = OPTIONS['sequence']
    T_OFFSET, FRAMES_PATH, DETS_FILE, GT_PATH, PRECOMPUTED_FLOW_PATH, PRECOMPUTED_FEAT_PATH = getSequenceConsts(sequence_str)
    print(FRAMES_PATH)
    
    # INIT
    D13 = readDetFile2(DETS_FILE) # read detections from file
    D13: list[list[Detection]] = [[]]+D13 # offset it (because time starts at 1, there is nothing on 0)
    skip_n=1148
    T_OFFSET = T_OFFSET + skip_n
    t_global = T_OFFSET # current time (frame)
    
    n = 0 # number of previous frames for trajectory init (not needed)
    n_res = len(D13) - n - T_OFFSET
    n_all = n_res


    # INIT DSPACE
    frame = getFrameAtI(t_global,FRAMES_PATH)
    h,w,_ = np.shape(frame)
    
    dspace: DetectionSpace = DetectionSpace(h,w, time_offset=(T_OFFSET-n), show_flow=False, disable_vis=False)
    
    # set last frame
    dspace.lastFrame = frame.copy()
    dspace.map = frame

    # print(D13[0:10])
    # print()
    # print(D13[t_global])
    d_at_t = D13[t_global]
    dspace.D.append(D13[t_global])

    dspace.showSpace(draw_dets=False)
    t_global = t_global +1

    # make trajectories from dets at this time
    all_trs: list[Trajectory] = []
    for d in d_at_t:
        tr = Trajectory(d, dspace)
        tr.build2()
        all_trs.append(tr)
    print(d_at_t)
    print(all_trs)
    
    for tr in all_trs:
        tr.drawToSpace()
    dspace.showSpace(draw_dets=False)
    dspace.clearSpace()

    n_stop = n_all
    n_stop = 1247

    preserve = []
    while(t_global <= n_stop):
        print("at time: %d"%t_global)
        frame = getFrameAtI(t_global,FRAMES_PATH)
        dspace.lastFrame = frame.copy()
        dspace.map = frame
        d_at_t = D13[t_global]
        dspace.D.append(d_at_t)

        used_dets: set = {}
        for tr in all_trs:
            tr_color = tr.color
            if(not tr.term):
                used_dets, forks = tr.extend4()
                if(tr.holes_ref >= 5):
                    tr.term = True
            else:
                tr_color = [100,100,100]
            if(not tr.term):
                drawBoundingBox(dspace.map, tr.nextStateEstimate.bb, [200,200,0])
            tr.drawToSpace(tr_color)
        if(t_global in {1221,1203,1225, 1173, 1163, 1199}):
            dets_to_tr = set(d_at_t) - used_dets
            for d in dets_to_tr:
                tr = Trajectory(d, dspace)
                tr.build2()
                all_trs.append(tr)
                if(d.id in {2608, 2510}):
                    preserve.append(tr.id)
                print("adding t%d (det id: %d)"%(tr.id, d.id))
                printTrWithStats(tr)
            
        dspace.showSpace(draw_dets=False, draw_last_dets_bb=False)
        t_global = t_global +1
    
    # merge trs
    all_trs2: list[Trajectory] = []
    for t in all_trs:
        if(t.id in preserve):
            all_trs2.append(t)
    all_trs = all_trs2

    dspace.clearSpace()
    for tr in all_trs:
        tr.drawToSpace()
    dspace.showSpace(draw_dets=False, draw_last_dets_bb=False)
    dspace.clearSpace()
    a: list[Trajectory]  = main2.getMergedHypotheses(all_trs)
    printTrListWithStatsOrdered(a)
    # a[0].color = [0,200,200]
    # a[1].color = [200,90,200]
    for tr in a:
        tr.drawToSpace()
    
    
    dspace.showSpace(draw_dets=False, draw_last_dets_bb=False)
    
    




def main7():
    print("this is main7 - kalman filter test")
    n = 150
    blank_map = np.zeros((n,n,3)).astype(np.uint8)+40
    zero_const = np.array([[0]])
    dspace = DetectionSpace(n,n, show_flow=False)
    dspace.map = blank_map
    # dspace.showSpace()


    # detekcije:

    # ground truth:
    bb_size = 20
    start_x  = np.array([20,20])
    nn = 20
    # x = []
    D13 = []
    for i in range(nn):
        i2 = 4*i
        x_i = start_x + [i2,i2]
        bb = [x_i[0],x_i[1],bb_size, bb_size]
        d = bbDet2Det(bb, i)
        d.color = [30,70,255]
        D13.append([d])
        # x.append(d)
    dspace.D = D13
    # dspace.showSpace(det_color=[100,130,250])

    # measurements:
    mean = 0.0
    dev = 3.8


    bb_size = 20
    start_x  = np.array([20,20])
    nn = 20
    
    x_mes:list[Detection] = []
    for i in range(nn):
        i2 = 4*i
        x_y_dev = np.array([np.random.normal(mean, dev), np.random.normal(mean, dev)])
        x_i = start_x + [i2,i2] + x_y_dev
        bb = [x_i[0],x_i[1],bb_size, bb_size]
        d = bbDet2Det(bb, i)
        d.color = [0,255,0]
        x_mes.append(d)
        D13[i].append(d)
        # x.append(d)

    dspace.D = D13
    dspace.showSpace(draw_last_dets_bb=False)

    # kf: build trajectory
    # state vec: x = [x, y, x_dt, y_dt]
    # we have random constant acceleration (modelled as uncontrolled input)
    q_cov = np.array([[1.0,0,0,0],[0,1.0,0,0],[0,0,1,0],[0,0,0,1]])
    r_cov = np.array([[dev, 0],[0,dev]])

    # model:
    dt = 1
    f_mat = np.array([[1,0,dt, 0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]])
    h_mat = np.array([[1,0,0,0],[0,1,0,0]])

    kf = Kalman(h_mat, f_mat, q_cov, r_cov)
    print(kf)

    g_mat = np.array([[0]])
    u_ = np.array([[0]])

    # init conditions:
    x_pos_init = D13[0][0].bb[0:2]
    x_ = np.array([[x_pos_init[0], x_pos_init[1],0,0]]).T
    p_ = np.array([[5,0,0,0],[0,5,0,0],[0,0,1,0],[0,0,0,1]])

    # 1st predict (estimate)
    x_1 = kf.predict(x_, g_mat, u_)
    p_1 = kf.covPredict(p_)
    

    # main loop: kalman filtering
    i_fin = len(x_mes)
    i = 0
    while(i < i_fin):
        z_det = x_mes[i]
        
        z_det_xy = z_det.bb[0:2]
        z_1 = np.array([[z_det_xy[0], z_det_xy[1]]]).T
        
        K = kf.getK(p_1)
        x_ = kf.stateUpdate(x_1, z_1, K)
        p_ = kf.covUpdate(p_1, K)
        
        # predict:
        x_1 = kf.predict(x_, g_mat, u_)
        p_1 = kf.covPredict(p_)
        
        # print("[%d] current state: x: %s, p: %s\npredict: x: %s, p: %s"%((i+1),str(x_), str(p_), str(x_1), str(p_1)))
        
        i = i+1


    





def main6():
    print("this is main6 from other_mains2")
    print("options: ")
    print(OPTIONS)

    # _, _, all_trs = readTrajectoryFile("./tracker/all_trs.p")
    # all_trs = list(all_trs.values())
    # printTrListWithStatsOrdered(all_trs)
    # print()

    # trs = readTrajectoryFile('../tr.p')
    # trs_selected = buildGroupSolve([])
    # trs_selected = buildGroupSolve(trs)
    # print(trs_selected)
    # aaa = []
    # for aa in a:
    #     aaa = aaa+aa
    # aaa.sort()
    # print(aaa)

    # pure, not_pure = analyzeTrsWithQ(trs)
    # # print(pure)
    # print("pure: ")
    # printTrListWithStatsOrdered(pure)
    # print("not pure: ")
    # printTrListWithStatsOrdered(not_pure)
    # Q = buildQBPMatrixX(not_pure, type=1)
    # np.savetxt("q.txt",Q, fmt="%7.2f")
    # print(Q)

    # Q = np.array([
    #     #1 2 3 4 5 6 7 8 9 10
    #     [1,1,0,0,0,0,0,0,0,0], # 1
    #     [1,1,1,0,0,0,0,0,0,0], # 2
    #     [0,1,1,0,0,0,0,0,0,0], # 3
    #     [0,0,0,1,0,0,0,0,1,0], # 4
    #     [0,0,0,0,1,1,0,0,0,1], # 5
    #     [0,0,0,0,1,1,0,0,0,1], # 6
    #     [0,0,0,0,0,0,1,0,1,0], # 7
    #     [0,0,0,0,0,0,0,1,0,1], # 8
    #     [0,0,0,1,0,0,1,0,1,0], # 9
    #     [0,0,0,0,1,1,0,1,0,1], # 10
    #     ])
    # Q = np.array(
    #     [[ 4.557, -0.   , -4.347, -0.   , -3.446, -0.   , -2.385, -2.01 , -0.   , -1.554, -0. ,   -0.6  ],
    #      [-0.   ,  3.711, -0.   , -3.526, -0.   , -2.543, -0.   , -0.   , -1.593, -0.   , -0.6,   -0.   ],
    #      [-4.347, -0.   ,  3.623, -0.   , -3.446, -0.   , -2.385, -2.01 , -0.   , -1.554, -0. ,   -0.6  ],
    #      [-0.   , -3.526, -0.   ,  2.938, -0.   , -2.543, -0.   , -0.   , -1.593, -0.   , -0.6,   -0.   ],
    #      [-3.446, -0.   , -3.446, -0.   ,  2.872, -0.   , -2.385, -2.01 , -0.   , -1.554, -0. ,   -0.6  ],
    #      [-0.   , -2.543, -0.   , -2.543, -0.   ,  2.119, -0.   , -0.   , -1.593, -0.   , -0.6,   -0.   ],
    #      [-2.385, -0.   , -2.385, -0.   , -2.385, -0.   ,  1.987, -2.01 , -0.   , -1.554, -0. ,   -0.6  ],
    #      [-2.01 , -0.   , -2.01 , -0.   , -2.01 , -0.   , -2.01 ,  1.675, -0.   , -1.554, -0. ,   -0.6  ],
    #      [-0.   , -1.593, -0.   , -1.593, -0.   , -1.593, -0.   , -0.   ,  1.328, -0.   , -0.6,   -0.   ],
    #      [-1.554, -0.   , -1.554, -0.   , -1.554, -0.   , -1.554, -1.554, -0.   ,  1.295, -0. ,   -0.6  ],
    #      [-0.   , -0.6  , -0.   , -0.6  , -0.   , -0.6  , -0.   , -0.   , -0.6  , -0.   ,  0.5,   -0.   ],
    #      [-0.6  , -0.   , -0.6  , -0.   , -0.6  , -0.   , -0.6  , -0.6  , -0.   , -0.6  , -0. ,    0.5  ]])
    Q = np.array([
        [ 3.419, -3.134, -2.348, -1.698, -0.959, -0.6,  ],
        [-3.134,  2.612, -2.348, -1.698, -0.959, -0.6,  ],
        [-2.348, -2.348,  1.957, -1.698, -0.959, -0.6,  ],
        [-1.698, -1.698, -1.698,  1.415, -0.959, -0.6,  ],
        [-0.959, -0.959, -0.959, -0.959,  0.799, -0.6,  ],
        [-0.6  , -0.6,   -0.6,   -0.6,   -0.6,    0.5,  ]])
    print(Q)
    # a = getGroupsFromQ(Q)
    a = buildGroupSolve([1],1)
    print("without groups:")
    res = solveQBP2(Q)
    print(res)
    print("solving with groups:")
    print(a)


# solving (merge) ...
# [[ 3.419 -3.134 -2.348 -1.698 -0.959 -0.6  ]
#  [-3.134  2.612 -2.348 -1.698 -0.959 -0.6  ]
#  [-2.348 -2.348  1.957 -1.698 -0.959 -0.6  ]
#  [-1.698 -1.698 -1.698  1.415 -0.959 -0.6  ]
#  [-0.959 -0.959 -0.959 -0.959  0.799 -0.6  ]
#  [-0.6   -0.6   -0.6   -0.6   -0.6    0.5  ]]
# solving QBP (6x6)
# [buildGroupSolve] solving with groups
# group 1 [0, 1, 2, 3, 4]:
# [[ 3.419 -3.134 -2.348 -1.698 -0.959]
#  [-3.134  2.612 -2.348 -1.698 -0.959]
#  [-2.348 -2.348  1.957 -1.698 -0.959]
#  [-1.698 -1.698 -1.698  1.415 -0.959]
#  [-0.959 -0.959 -0.959 -0.959  0.799]]
# solving Q ...
# solving QBP (5x5)
# (array([1, 0, 0, 0, 0], dtype=uint8), 3.418967849300367)
# [0]
# group 2 [5]:
# [[0.5]]
# solving Q ...
# solving QBP (1x1)
# (array([1], dtype=uint8), 0.5)
# [5]
# done solving, v:
# (array([1, 0, 0, 0, 0, 0], dtype=uint8), 3.418967849300367)
# v2:
# (array([1, 0, 0, 0, 0, 1], dtype=uint8), 3.918967849300367)
# adding selected merged to idTr_map
# added
# [@ 122 / 4101 (2%)]
# [starting new]
# [UPDATING TARGET]
# target lost, searching for new based on visual similarity:
