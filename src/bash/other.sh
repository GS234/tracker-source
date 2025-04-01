#!/bin/bash


get_other(){
    path1="${1}"
    path2="${2}"

    seq_1=$(cd $path1 && ls -d */)
    seq_1=${seq_1//\// }
    
    seq_2=$(cd $path2 && ls -d */)
    seq_2=${seq_2//\// }

    res=$seq_1
    for s in "${seq_2}"; do
    {
        res=${res//$s/ }
    }; done
    echo $res
}

# get_other "path1" "path2"


list=$(ls path1)
n_skip=0
for l in $list; do
    # echo $l
    $(cd path2; [ ! -e "${l}" ] && ln -s ../path1/$l $l || exit 1)
    [ $? -eq 1 ] && { echo "file ${l} exists, skipping"; n_skip=$(( $n_skip+1 )); }
    # echo $?
done;
echo "skipped ${n_skip} files"

