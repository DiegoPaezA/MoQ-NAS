#!/bin/bash
dataset="cifar10" # Change to: "pathmnist", "octmnist", "tissuemnist", "organamnist", etc.
exp_path_base="experiment_v3_${dataset}"
config_file="config_files_cifar"
fitness_metric="best_accuracy"
data_path="${dataset}_data"
log_level="INFO"
network_config="default"
backbone_name="resnet18"
dataset_sample_size=10000

# NEW: dataset YAML path (uses the dataset name above)
config_path_dataset="configs/${dataset}.yaml"

# --- Evolution Toggles ---
use_cache=false          # Use cached evaluations to speed up runs
early_stopping=false
en_pop_crossover=true
elite_mode="global_k"    # "single" | "global_k" | "bootstrap_k" | "old"

# --- New Network Architecture Rule Toggles ---
# Enabled by default. Set to 'false' to add the corresponding '--no-...' flag.
truncate_after_noop=false
avoid_consecutive_pool=true
enforce_noop_in_update=true # avoid no-op in the min_active_len (default 5) update

# --- Experiment Setup ---
configs=("config5.txt")
exps=("exp19")
cuda_devices=("0")

# Loop over the length of the configs array
for ((j=0; j<${#configs[@]}; j++)); do
    config="${configs[$j]}"
    exp="${exps[$j]}"
    cuda_device="${cuda_devices[$j]}"

    echo "Running evolution experiment with $config"

    for ((i=1; i<=3; i++)); do # Change the range to the number of repeats
        exp_path="${exp_path_base}/${exp}_repeat_$i"

        CUDA_VISIBLE_DEVICES="$cuda_device" python run_evolution.py \
            --experiment_path "$exp_path" \
            --config_file "${config_file}/${config}" \
            --data_path "$data_path" \
            --dataset "$dataset" \
            --config_path_dataset "$config_path_dataset" \
            --limit_data_value "$dataset_sample_size" \
            --fitness_metric "$fitness_metric" \
            --network_config "$network_config" \
            --backbone_name "$backbone_name" \
            --elite_mode "$elite_mode" \
            $($early_stopping && echo "--early_stopping") \
            $($en_pop_crossover && echo "--en_pop_crossover") \
            $($use_cache && echo "--use_cache") \
            $($truncate_after_noop && echo "--no-truncate-after-noop") \
            $($avoid_consecutive_pool && echo "--no-avoid-consecutive-pool") \
            $($enforce_noop_in_update && echo "--no-enforce-noop-in-update") \
            --log_level "$log_level"
    done
done