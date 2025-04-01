#!/bin/bash

root_dir=$(dirname -- "$( readlink -f -- "$0"; )") # this is root directory (where this script is)

# sequences_path="$root_dir/workspace/sequences/" # set accordingly to actual directory tree
sequences_path="$root_dir/../../data/frames/" # to test locally
# save_path="$root_dir/results/" # further divided into bb, tracks
save_path="$root_dir/../../data/precomputed/"
# main_path="$root_dir/../python/src/"
main_path="$root_dir/../"

run_python(){
    echo "using sequence ${1}:"
    # echo "frames path: ${2}"
    # echo "dets path: ${3}"

    # here, python gets called, and it creates 2 files inside track: result.txt and trs.p
    # when python is done, those new files should get copied to their locations in results
    #echo "python3 main2.py --sequence='${sequences_path}${1}' --dets='${dets_path}${1}.txt' --track='${track_path}'"
    cd "${main_path}" # go to python
    # python3 main2.py --sequence="${sequences_path}${1}" --dets="${dets_path}${1}.txt" --track="${track_path}"
    (python3 precompute_flow_feat.py --sequence "${1}" --seq_path "${sequences_path}" --save_path "${save_path}" --feat)
    echo $(pwd)
    cd "${root_dir}" # go back
    echo $(pwd)
}


# read from list

for i in $(cat "${root_dir}/../../data/precomputed/list.txt"); do
    # echo $i
    echo running python:
    # echo "python3 precompute_flow_feat.py --sequence ${i} --seq_path ${sequences_path} --save_path ${save_path} --feat"
    run_python "${i}"
    echo done running
done



# python3 precompute_flow_feat.py --sequence 'LaSOT_bird-15' --seq_path '/home/gasper/Faks/3_letnik/diplomska/koda/data/frames/' --save_path '/home/gasper/Faks/3_letnik/diplomska/koda/data/' --feat