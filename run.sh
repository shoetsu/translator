#!/bin/bash

#input from shell
usage() {
    echo "Usage:$0 checkpoint_path mode [config_path]"
    exit 1
}

# optの解析 (argv, argc, opt_names, opt_valuesを設定)
source ./scripts/manually_getopt.sh $@

if [ $argc -lt 2 ];then
    usage;
fi

bin_file=bins/wikiP2D.py
checkpoint_path=${argv[0]}
mode=${argv[1]}
config_path=${argv[2]}
config_file=config
log_file=$checkpoint_path/${mode}.log

if [ "${config_path}" = "" ]; then
    if [ -e ${checkpoint_path}/${config_file} ]; then
	config_path=${checkpoint_path}/${config_file}
    else
	config_path=configs/config
	#echo "specify config file when start training from scratch."
	#exit 1
    fi
# else
#     if [ -e ${checkpoint_path}/${config_file} ]; then
# 	config_path=${checkpoint_path}/${config_file}
#     fi
fi

# 実行時オプションを優先(from manually_optget.sh)
for i in $(seq 0 $(expr ${#opt_names[@]} - 1)); do
    name=${opt_names[$i]}
    value=${opt_values[$i]}
    eval $name=$value
done;

# get params from input args
params_arr=(log_file interactive cleanup test_data_path evaluate_data_path debug batch_size)
params=""


for param in ${params_arr[@]}; do 
    if [ ! ${!param} = "" ]; then
	params="${params} --${param}=${!param}"
    fi
done;

python src/main.py $checkpoint_path $mode $config_path $params
