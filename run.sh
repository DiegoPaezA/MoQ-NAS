#!/bin/bash
dataset="cifar10"
exp_path_base="experiment_${dataset}_acc_1"
config_file="config_files_cifar"
fitness_metric="best_accuracy"
data_path="${dataset}_data"
log_level="INFO"
network_config="backbone"
backbone_name="mobilenet_v3_large"

dataset_sample_size=10000

configs=("config0.txt")
exps=("exp1")
cuda_devices=("0")

# Loop over the length of the configs array
for ((j=0; j<${#configs[@]}; j++)); do
    config="${configs[$j]}"
    exp="${exps[$j]}"
    cuda_device="${cuda_devices[$j]}"

    echo "Running evolution experiment with $config"

    for ((i=1; i<=1; i++)); do # Change the range to the number of repeats
        exp_path="${exp_path_base}/${exp}_repeat_$i"

        CUDA_VISIBLE_DEVICES="$cuda_device" python run_evolution.py \
            --experiment_path "$exp_path" \
            --config_file "${config_file}/${config}" \
            --data_path "$data_path" \
            --dataset "$dataset" \
            --limit_data_value "$dataset_sample_size" \
            --fitness_metric "$fitness_metric" \
            --early_stopping \
            --network_config "$network_config" \
            --backbone_name "$backbone_type" \
            --en_pop_crossover \
            --log_level "$log_level"
    done
done