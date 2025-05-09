#!/usr/bin/bash

run_python(){
        echo "using sequence ${1}:"
        # echo "frames path: ${2}"
        # echo "dets path: ${3}"

        # here, python gets called, and it creates 2 files inside track: result.txt and trs.p
        # when python is done, those new files should get copied to their locations in results
        #echo "python3 main2.py --sequence='${sequences_path}${1}' --dets='${dets_path}${1}.txt' --track='${track_path}'"
        cd "${main_path}" # go to python
        python3 main2.py --sequence="${sequences_path}${1}" --dets="${dets_path}${1}.txt" --track="${track_path}" --feat="${precomputed_path}${1}_pr/${1}_feat/"
        cd "${root_dir}" # go back



        # 1. mkdir
        mkdir "${results_path}bb/${1}/"
        mkdir "${results_path}tracks/${1}/"

        # 2. cp 
        cp "${track_path}results.txt" "${results_path}bb/${1}/${1}_001.txt"
        cp "${track_path}trs.p" "${results_path}tracks/${1}/${1}_trs.p"
        echo "done with sequence ${1}"
}

#first path must be path to detections files
get_other(){
        path_d="${1}"
        path_r="${2}"

#       seq_1=$(cd $path1 && ls -d */)
#       seq_1=${seq_1//\// }

        seq_1=$(cd $path_d && ls | grep .txt)
        seq_1=${seq_1//.txt/ } # remove .txt

        seq_2=$(cd $path_r && ls -d */)
        seq_2=${seq_2//\// }

        res=$seq_1
        for s in $(echo "${seq_2}"); do
        {
                #echo $s
                res=${res//$s / }
        }; done
        echo $res
}

get_other2(){
        seq_2=$(cd $results_path && ls -d */)
        seq_2=${seq_2//\// }

        for s in $(echo "${seq_2}"); do
        {
                #echo $s
                res=${res//$s / }
        }; done
        echo $res
}

#first path must be path to list of all sequences
get_other3(){
        path_d="${1}"
        path_r="${2}"

        # seq_1=$(cd $path_d && ls | grep .txt)
        # seq_1=${seq_1//.txt/ } # remove .txt
        seq_1=$(cd $path_d && $(cat "list.txt"))
        

        seq_2=$(cd $path_r && ls -d */)
        seq_2=${seq_2//\// }

        res=$seq_1
        for s in $(echo "${seq_2}"); do
        {
                #echo $s
                res=${res//$s / }
        }; done
        echo $res
}


root_dir=$(dirname -- "$( readlink -f -- "$0"; )") # this is root directory (where this script is)
#echo $root_dir


sequences_path="$root_dir/workspace/sequences/" # set accordingly to actual directory tree
track_path="$root_dir/tracker/" # this does not change
dets_path="$root_dir/dets/" # where detections are
results_path="$root_dir/results/" # further divided into bb, tracks
main_path="$root_dir/../python/src/"
precomputed_path="$root_dir/../precomputed/"


# 1.a get sequences (from sequences):

#sequences=$(cd $sequences_path && ls -d */)
#sequences=${sequences//\// } # remove trailing slashes

# 1.b get sequences (from detectons):

sequences=$(cd $dets_path && ls | grep .txt)
sequences=${sequences//.txt/ } # remove .txt

#echo $sequences
# 2. for each sequence (that is not already in results):

seq_rem=$(get_other "${dets_path}" "${results_path}/bb/")
#echo $seq_rem

#for sequence in $sequences
for sequence in $seq_rem
do
        echo "${sequence}"
#       run_python "${sequence}"
done

# run_python "LaSOT_bird-15"
