import sys
sys.path.append("../")
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from helper_func import *

DATA_ROOT = "../../data/"
FRAMES_PATH = "frames/LaSOT_bird-2/color/"

def main():
    t_i = 10
    frame = getFrameAtI(t_i,DATA_ROOT+FRAMES_PATH)
    D13 = readDetFile2(DATA_ROOT+"detections/LaSOT_bird-2.txt") # read detections from file

    dets = D13[t_i]
    hists = []
    for d in dets:
        bb_frame = getBoundingBoxPx(frame, d.bb)
        cv.imshow("def"+str(t_i), bb_frame)
        hist = getColorHist(bb_frame, 10)
        hists.append(hist.copy())
        print(hist)
        # showHist(hist)
        cv.waitKey(0)
    for d in dets:
        drawBoundingBox(frame, d.bb)
    cv.imshow("abc", frame)
    cv.waitKey(0)

    showHists(hists)
    hists_avg = (0.5*hists[0]+0.5*hists[2])
    hists.append(hists_avg)
    showHists(hists)

    # comparison:
    n = len(dets)
    print(dets)
    print(dets[0])
    print("comparing hists: ")
    for i in range(n):
        a_bb_frame = getBoundingBoxPx(frame, dets[i].bb)
        a = getColorHist(a_bb_frame)
        print(i, ": ", end="", flush=True)
        for j in range(n):
            b_bb_frame = getBoundingBoxPx(frame, dets[j].bb)
            b = getColorHist(b_bb_frame)
            print(compareHists(a,b), end=" ", flush=True)
        print()



    # print(D13)

def getColorHist(frame_bb, n_bins=10):
    color_hist = np.zeros((n_bins, n_bins, n_bins))
    # 1. put each pixel to its corresponding bin
    for row in frame_bb: # row-major? should be
        for pixel in row:
            # 1.1 pixel has rgb; calculate appropriate bins (for each color) and increase cell value
            # c1, c2, c3 # not that important which color is which (cv uses bgr, rgb), as long as everything is handled equally
            px = np.array(pixel)
            px = (px / 255) * (n_bins-1)
            px = px.astype(np.uint8)
            c1, c2, c3 = px[0],px[1],px[2]
            # 1.2 increase value of correct bin
            color_hist[c1][c2][c3] += 1
    # 2. normalize histogram
    color_hist = color_hist / np.sum(color_hist)
    return color_hist

def showHist(hist):
    # show only in one dimension
    arr = np.zeros(np.shape(hist)[0])
    for i in range(len(arr)):
        arr[i] = np.sum(hist[i])
    
    x = np.arange(len(arr))
    # print(arr)
    
    plt.bar(x,arr)
    plt.show()
        
    pass

def showHists(hists:list):
    # show only in one dimension
    _,ax = plt.subplots(1,len(hists))
    print(ax)
    for j in range(len(hists)):
        hist = hists[j]
        arr = np.zeros(np.shape(hist)[0])
        for i in range(len(arr)):
            arr[i] = np.sum(hist[i])
        x = np.arange(len(arr))
        if(len(hists) == 1):
            ax.bar(x,arr)
        else:
            ax[j].bar(x,arr)
    # print(arr)
    
    # plt.bar(x,arr)
    plt.show()
        

def getBoundingBoxPx(frame, bb: tuple):
    y, x, h, w = bb
    return frame[x:x+w+1,y:y+h+1]

def compareHists(a, A):
    return np.sum(np.sqrt(a*A)) # bhattacharyya distance


def drawBoundingBox(frame, bb: tuple, color: list = [0,255,0]) -> None:
    # print(W, H)
    # 1. starting coordinate:
    y, x, h, w = bb
    # print(bb)

    points = []

    # print(image)
    # 2. draw horizontally:
    for i in range(w):
        x_i, y1_i, y2_i = x+i, y, y+h

        points.append((x+i, y))
        points.append((x+i, y+h))

    # 3. draw vertically:
    for i in range(h):
        points.append((x, y+i))
        points.append((x+w, y+i))
    coords2map(points, frame, color=color, overwrite=True)

    

if __name__ == "__main__":
    main()
