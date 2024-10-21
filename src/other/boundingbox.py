import sys
sys.path.append("../")
import numpy as np
from helper_func import *
import cv2 as cv

DATA_PATH = "../../data/"
X_FFFFFF = [255,255,255]

def writeBoundingBox(image: np.array, bb: tuple, color: list = X_FFFFFF) -> None:
    H, W, _ = np.shape(image)
    # print(W, H)
    # 1. starting coordinate:
    y, x, h, w = bb
    # print(bb)

    # print(image)
    # 2. draw horizontally:
    for i in range(w):
        x_i, y1_i, y2_i = x+i, y, y+h

        image[x+i, y] = color
        image[x+i, y+h] = color

    # 3. draw vertically:
    for i in range(h):
        image[x, y+i] = color
        image[x+w, y+i] = color

def drawX(image: np.array, X) -> None:
    x, y = X[1],X[0]
    
    
    image[x-2,y-2] = X_FFFFFF
    image[x-1,y-1] = X_FFFFFF
    image[x,y]     = X_FFFFFF
    image[x+1,y+1] = X_FFFFFF
    image[x+2,y+2] = X_FFFFFF

    image[x-2,y+2] = X_FFFFFF
    image[x-1,y+1] = X_FFFFFF
    image[x+1,y-1] = X_FFFFFF
    image[x+2,y-2] = X_FFFFFF



def main():
    # a = np.zeros(10)
    # print(a)
    # for i in range(10):
    #     incIndVec(a)
    #     print(a)

    groundt = readDetFile(DATA_PATH+"frames/LaSOT_bird-2/groundtruth.txt")
    dets = readDetFile(DATA_PATH+"detections/LaSOT_bird-2.txt")
    # print(dets)
    
    # print(a)

    n = 2000

    for i in range(1,n+1):
        frame_i = f'{i:08}'
        frame = cv.imread(DATA_PATH+"frames/LaSOT_bird-2/color/"+str(frame_i)+".jpg")
        bb = groundt[i][0]
        writeBoundingBox(frame, bb, [0,255,0])

        # drawX(frame, [bb[1],bb[0]])
        drawX(frame, [bb[0]+bb[2]//2, bb[1]+bb[3]//2])
        


        for det_bb in dets[i]:
            writeBoundingBox(frame, det_bb, [255,255,255])
            drawX(frame, [det_bb[0]+det_bb[2]//2, det_bb[1]+det_bb[3]//2])
        



        print("frame: ", i)
        cv.imshow("abc", frame)
        cv.waitKey(0)


    




if __name__ == '__main__':
    main()