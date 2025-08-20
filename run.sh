#!/bin/bash
dataset="cifar10" # Change this to the dataset you want to run: "pathmnist", "octmnist", "tissuemnist", "organamnist"
exp_path_base="experiment_v3_${dataset}"
config_file="config_files_cifar"
fitness_metric="best_accuracy"
data_path="${dataset}_data"
log_level="INFO"
network_config="default"
backbone_name="resnet18"
use_cache=false  # Use cached evaluations to speed up runs
early_stopping=false
en_pop_crossover=true
elite_mode="bootstrap_k" # "single" | "global_k" | "bootstrap_k" | "old"

dataset_sample_size=10000

configs=("config5.txt")
exps=("exp12")
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
            $($early_stopping && echo "--early_stopping")\
            --network_config "$network_config" \
            --backbone_name "$backbone_name" \
            $($en_pop_crossover && echo "--en_pop_crossover")\
            $($use_cache && echo "--use_cache")\
            --elite_mode "$elite_mode" \
            --log_level "$log_level"
    done
done