#!/bin/bash

# —— Experiment settings —— 
dataset="cifar10"
exp_path_base="experiment_${dataset}_ga"
config_dir="config_files_med"
fitness_metric="best_accuracy"
data_path="${dataset}_data"
log_level="INFO"
network_config="default"
backbone_name="resnet18"

# —— GA hyperparameters —— 
population_size=20
num_generations=300
max_num_nodes=20
crossover_rate=0.4
mutation_rate=0.1
elitism=true
patience=60

# —— sample size & repeats —— 
dataset_sample_size=10000
configs=("config1.txt")     # list your config files here
exps=("exp1")               # corresponding experiment names
cuda_devices=("0")          # GPU IDs for each run

# —— Loop over configs & repeats —— 
for ((j=0; j<${#configs[@]}; j++)); do
    config_file="${config_dir}/${configs[$j]}"
    exp_name="${exps[$j]}"
    cuda_device="${cuda_devices[$j]}"

    echo "=== Running ${dataset} / ${configs[$j]} on GPU ${cuda_device} ==="

    for ((i=1; i<=1; i++)); do  # change 1 to your number of repeats
        exp_path="${exp_path_base}/${exp_name}_repeat_${i}"

        CUDA_VISIBLE_DEVICES="$cuda_device" python run_ga_evolution.py \
            --experiment_path      "$exp_path" \
            --config_file          "$config_file" \
            --data_path            "$data_path" \
            --dataset              "$dataset" \
            --limit_data_value     "$dataset_sample_size" \
            --fitness_metric       "$fitness_metric" \
            --patience             "$patience" \
            --crossover_rate       "$crossover_rate" \
            --mutation_rate        "$mutation_rate" \
            --population_size      "$population_size" \
            --num_generations      "$num_generations" \
            --max_num_nodes        "$max_num_nodes" \
            $( $elitism && echo "--elitism" ) \
            --network_config       "$network_config" \
            --backbone_name        "$backbone_name" \
            --log_level            "$log_level"
    done
done
