#!/bin/bash

#=======================================================================
# Script to run parallel retraining for models from a MoQ-NAS experiment
#=======================================================================

# --- Experiment to Retrain ---
# This should point to the specific experiment run you want to retrain from.
exp_path="experiment_cifar10_moqnas/exp4_repeat_2"
data_path="cifar10_data"
dataset="cifar10"
log_level="INFO"

# --- GPU Configuration ---
# Set the GPU(s) to use for the retraining workers.
# For a single GPU: cuda_devices="0"
# For multiple GPUs: cuda_devices="0,1"
cuda_devices="1"

# --- Retraining Selection Options ---
# Set 'retrain_all=true' to retrain all valid models from the Pareto front.
# If 'false', it will use 'sort_by' and 'top_n'.
retrain_all=true
sort_by="accuracy"  # Metric to sort by if not retraining all ('accuracy', 'params', 'inference_time')
top_n=5             # Number of top models to retrain if not retraining all

# --- Concurrency & Repetitions ---
# How many models to retrain in parallel at once.
# Leave empty to default to the number of available GPUs.
max_parallel_workers=4
num_repetitions=1   # How many times to retrain each selected model.

# --- Training Hyperparameter Overrides ---
# These will override the default settings in retrain_parallel.py
max_epochs=150
epochs_to_eval=150
batch_size=256
optimizer="AdamW"
# IMPORTANT: Keep num_workers at 0 to avoid nested parallelism and crashes.
num_workers=0

patience_retrain=50
# Booleans: set to "true" or "false"
data_augmentation=true
limit_data=false


#=======================================================================
# --- Build and Run the Command ---
#=======================================================================

echo "=== Starting Retraining Script for Experiment: ${exp_path} ==="
echo "Using GPUs: ${cuda_devices}"

# Build the command in an array for robustness against spaces and special characters
CMD=(
    "python" "retrain_parallel.py"
    "--experiment_path"    "$exp_path"
    "--data_path"          "$data_path"
    "--dataset"            "$dataset"
    "--log_level"          "$log_level"
    "--num_repetitions"    "$num_repetitions"
    "--max_epochs"         "$max_epochs"
    "--epochs_to_eval"     "$epochs_to_eval"
    "--batch_size"         "$batch_size"
    "--optimizer"          "$optimizer"
    "--num_workers"        "$num_workers"
    "--patience_retrain"   "$patience_retrain"
)

# Add conditional flags for retraining selection
if [[ "$retrain_all" = true ]]; then
    CMD+=(--retrain_all)
else
    if [[ -n "$sort_by" ]]; then
        CMD+=(--sort_by "$sort_by")
    fi
    if [[ -n "$top_n" ]]; then
        CMD+=(--top_n "$top_n")
    fi
fi

if [[ -n "$max_parallel_workers" ]]; then
    CMD+=(--max_parallel_workers "$max_parallel_workers")
fi

# Add conditional flags for boolean training options
if [[ "$data_augmentation" = true ]]; then
    CMD+=(--data_augmentation)
fi
if [[ "$limit_data" = true ]]; then
    CMD+=(--limit_data)
fi

# Set the CUDA environment variable and execute the command
echo "Running command: CUDA_VISIBLE_DEVICES=\"$cuda_devices\" ${CMD[@]}"
CUDA_VISIBLE_DEVICES="$cuda_devices" "${CMD[@]}"

echo "=== Retraining Finished ==="