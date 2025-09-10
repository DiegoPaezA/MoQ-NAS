#!/bin/bash

# —— Experiment settings —— 
dataset="cifar10"
exp_path_base="experiment_${dataset}_moqnas"
config_dir="config_files_cifar"
fitness_metric="best_accuracy"
data_path="${dataset}_data"
log_level="INFO"
network_config="default"
backbone_name="resnet18"   # only used if network_config="backbone"
continue_path=""           # set to non-empty string to resume from a previous run

# —— MO-QNAS hyperparameters —— 
optimizer="AdamW"
save_checkpoints_epochs=5
# Booleans: set to "true" or "false"
use_cache=false           # Use cached evaluations to speed up runs
data_augmentation=false
early_stopping=false
en_pop_crossover=true
multi_objective=true
elite_mode="bootstrap_k"  # "single" | "global_k" | "bootstrap_k" | "old"

# —— Network Architecture Rules ——
# These are enabled by default. Set to "false" to add the corresponding '--no-...' flag.
truncate_after_noop=false
avoid_consecutive_pool=true
enforce_noop_in_update=true

# —— sample size & repeats —— 
dataset_sample_size=10000
configs=("config5.txt")   # list your MO-QNAS config files here (inside ${config_dir}/)
exps=("exp5")             # corresponding experiment names
cuda_devices=("1")        # GPU IDs for each run

# Number of repeats per configuration
num_repeats=3

# —— Loop over configs & repeats —— 
for ((j=0; j<${#configs[@]}; j++)); do
    config_file="${config_dir}/${configs[$j]}"
    exp_name="${exps[$j]}"
    cuda_device="${cuda_devices[$j]}"

    echo "=== Running MO-QNAS on dataset=${dataset}, config=${configs[$j]}, GPU=${cuda_device} ==="

    for ((i=1; i<=num_repeats; i++)); do
        exp_path="${exp_path_base}/${exp_name}_repeat_${i}"
        echo "   → Repeat $i: saving to ${exp_path}"

        CUDA_VISIBLE_DEVICES="$cuda_device" python run_evolution_moqnas.py \
            --experiment_path       "$exp_path" \
            --data_path             "$data_path" \
            --dataset               "$dataset" \
            --config_file           "$config_file" \
            --continue_path         "$continue_path" \
            --log_level             "$log_level" \
            --optimizer             "$optimizer" \
            --fitness_metric        "$fitness_metric" \
            --save_checkpoints_epochs "$save_checkpoints_epochs" \
            --limit_data_value      "$dataset_sample_size" \
            --backbone_name         "$backbone_name" \
            --network_config        "$network_config" \
            --elite_mode            "$elite_mode" \
            $($use_cache           && echo "--use_cache") \
            $($data_augmentation   && echo "--data_augmentation" ) \
            $($early_stopping      && echo "--early_stopping" ) \
            $($en_pop_crossover    && echo "--en_pop_crossover" ) \
            $($multi_objective     && echo "--multi_objective" ) \
            $($truncate_after_noop && echo "--no-truncate-after-noop") \
            $($avoid_consecutive_pool && echo "--no-avoid-consecutive-pool") \
            $($enforce_noop_in_update && echo "--no-enforce-noop-in-update")
    done
done