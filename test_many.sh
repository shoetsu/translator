#!/bin/bash
usage() {
    echo "Usage:$0 root_dir"
    exit 1
}

if [ $# -lt 1 ];then
    usage;
fi
root_dir=$1
opt=''
#opt='--test_data_path=dataset/test.complicated.csv'
#opt='--test_data_path=dataset/test.annotated.csv'
for checkpoint in $(ls -d $root_dir/*); do
    ./run.sh $checkpoint test $opt &
done;


