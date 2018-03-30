#!/bin/bash
#root_path=checkpoints/latest/
root_path=$1
#opt="--cleanup=T --train_data_path=dataset/train.price.rate.csv "
opt=""

#model_types=(0.baseline 1.balanced 4.normalize 5.normalize.balanced)
model_types=(0_unit selective_selective zero_zero all_all)
for mt in ${model_types[@]}; do
    echo "    nohup ./run.sh $root_path/$mt train &"
    nohup ./run.sh $root_path/$mt train $opt &
done;

# #model_types=(2.independent 6.normalize.balanced.independent)
# #model_types=(4C.rate.independent.normalize)
# model_types=()
# column_types=(LB UB Unit Rate)
# for mt in ${model_types[@]}; do
#     for ct in ${column_types[@]};do
# 	echo "nohup ./run.sh checkpoints/latest/independent/$mt/$ct train &"
# 	nohup ./run.sh $root_path/$mt/$ct train $opt &
#     done;
# done;


