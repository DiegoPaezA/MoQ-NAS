#!/bin/bash
dataset="cifar10" # Change this to the dataset you want to run: "pathmnist", "octmnist", "tissuemnist", "organamnist"
exp_path_base="experiment_v3_${dataset}"
config_file="config_files_cifar"
fitness_metric="best_accuracy"
data_path="${dataset}_data"
log_level="INFO"
network_config="backbone"
backbone_name="resnet18"

dataset_sample_size=10000

configs=("config4.txt")
exps=("exp4")
cuda_devices=("0")

# Loop over the length of the configs array
for ((j=0; j<${#configs[@]}; j++)); do
    config="${configs[$j]}"
    exp="${exps[$j]}"
    cuda_device="${cuda_devices[$j]}"

    echo "Running evolution experiment with $config"

    for ((i=2; i<=3; i++)); do # Change the range to the number of repeats
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
            --backbone_name "$backbone_name" \
            --en_pop_crossover \
            --log_level "$log_level"
    done
done