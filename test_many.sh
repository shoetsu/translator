#!/bin/bash
usage() {
    echo "Usage:$0 root_dir"
    exit 1
}

if [ $# -lt 1 ];then
    usage;
fi
root_dir=$1
opt=""
#opt='--target_attribute=Weight --test_data_path=dataset/test.weight.csv'
opt='--test_data_path=dataset/test.price.empty.csv'
#opt='--test_data_path=dataset/test.annotated.csv'
for checkpoint in $(ls -d $root_dir/*); do
    nohup ./run.sh $checkpoint test $opt &
done;


