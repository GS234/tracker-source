from helper_func import *
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', help="path to file")
    parser.add_argument('-n', help="number of frames")
    parser.add_argument('-d', help="path to dets file")
    parser.add_argument('-r', help="path where to save")
    args = parser.parse_args()

    FILENAME = "./trs.p"
    RESULTS = "./"
    n = 100
    if(args.n is None):
        if(args.d is None):
            print("[WARN] enter sequence length: -n=<num> or path to detections file")
            return
        else:
            # open dets, get length, save
            n = len(readDetFile2(args.d))
    else:
        n = int(args.n)
    if(args.r is not None):
        RESULTS=args.r

    if(args.p is not None):
        print("[INFO] using file %s"%(args.p))
        FILENAME = args.p
    
    # open trs file:
    # this could take any of following shapes:
    # (T_OFFSET, list[Trajectories]), (most common)
    # (T_OFFSET, n_all, list[Trajectories]), (new format, it also saves number of frames so we do not need to find it ourselves)
    # (T_OFFSET, list[Trajectories], list[Trajectories])
    # ... or something else (if not sure, check it)
    _,path = readTrajectoryFile(FILENAME)
    print(path)
    path2File2(path, n, RESULTS, filename="")


if __name__ == '__main__':
    main()