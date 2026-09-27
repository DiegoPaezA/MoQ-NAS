#!/bin/bash

# --- 🎮 GPU SELECTION ---
# Case A: One GPU -> Runs SEQUENTIALLY
CUDA_DEVICES=("1") 

# Case B: Two GPUs -> Runs in PARALLEL
# CUDA_DEVICES=("0" "1")

# --- CONFIGURATION ---
ARCHS="resnet18 resnet50 efficientnet_v2_s convnext_tiny mobilenet_v3_large"

# Repetitions per model. Each seed gets its own output subfolder so
# checkpoints/CSVs from different runs never overwrite each other.
SEEDS=(1 2 3)

# Image settings
IMG_SIZE=96
RESIZE_MODE="letterbox"

# Config files
CONFIG_PERSON="dataset_configs/person_bin_96.yaml"

# Data paths
DATA_PERSON="datasets/personbin_data_96"

# ---------------------

# 1. Define Defaults
OPTIONAL_ARGS=""
OUTPUT_DIR="checkpoints/baseline_limit_96"
MODE_DESC="FULL-TRAINING (Pretrained)"

# Default Epochs (Standard Fine-tuning / Head Only)
MAX_EPOCHS=10

# --- 🔄 ARGUMENT PARSING ---
for arg in "$@"; do
    case $arg in
        --head_only)
            OPTIONAL_ARGS="$OPTIONAL_ARGS --freeze_backbone"
            # Update directory logic
            if [[ "$OUTPUT_DIR" == *"scratch"* ]]; then
                OUTPUT_DIR="checkpoints/baseline_scratch_freeze_limit_96"
            else
                OUTPUT_DIR="checkpoints/baseline_freeze_limit_96"
            fi
            MODE_DESC="HEAD-ONLY (Frozen Backbone)"
            
            # Set Epochs for Head Only
            MAX_EPOCHS=10
            ;;
            
        --from_scratch)
            OPTIONAL_ARGS="$OPTIONAL_ARGS --from_scratch"
            # Update directory logic
            if [[ "$OUTPUT_DIR" == *"freeze"* ]]; then
                OUTPUT_DIR="checkpoints/baseline_scratch_freeze_limit_96"
            else
                OUTPUT_DIR="checkpoints/baseline_scratch_limit_96"
            fi
            MODE_DESC="FULL-TRAINING (From Scratch)"
            
            # Set Epochs for From Scratch
            MAX_EPOCHS=50
            ;;
    esac
done

echo "✅ Running Mode: $MODE_DESC"
echo "✅ Max Epochs: $MAX_EPOCHS"
echo "✅ Optional Args: $OPTIONAL_ARGS"

# Create directories
LOG_DIR="logs"
NUM_GPUS=${#CUDA_DEVICES[@]}
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

echo "✅ Log files will be saved in '$LOG_DIR'"
echo "✅ Checkpoints will be saved in '$OUTPUT_DIR'"
echo "✅ GPU Setup: ${NUM_GPUS} device(s): ${CUDA_DEVICES[*]}"
echo "------------------------------------------------------"


GPU_A=${CUDA_DEVICES[0]}
EVAL_GPU=${CUDA_DEVICES[0]}

for seed in "${SEEDS[@]}"; do
    RUN_OUTPUT_DIR="$OUTPUT_DIR/seed_${seed}"
    mkdir -p "$RUN_OUTPUT_DIR"
    echo "🌱 === RUN seed=$seed -> $RUN_OUTPUT_DIR ==="

    for arch in $ARCHS; do
        echo "🚀 Processing ARCHITECTURE: $arch (seed $seed)"

        LOG_FILE_PERSONBIN="$LOG_DIR/personbin_${arch}_seed${seed}.log"

        echo "   --> [GPU $GPU_A] Personbin training..."
        CUDA_VISIBLE_DEVICES="$GPU_A" python scripts/fairness_baseline/train.py \
            --config_file $CONFIG_PERSON \
            --data_path $DATA_PERSON \
            --experiment_path "$RUN_OUTPUT_DIR" \
            --results_csv "$RUN_OUTPUT_DIR/personbin_results_acc.csv" \
            --archs $arch \
            --max_epochs $MAX_EPOCHS \
            --limit_data \
            --limit_data_value 10000 \
            --seed $seed \
            $OPTIONAL_ARGS > "$LOG_FILE_PERSONBIN" 2>&1

        echo "✅ Finished ARCHITECTURE: $arch (seed $seed)"
        echo "------------------------------------------------------"
    done

    # --- Evaluation (per seed) ---
    echo "🚀 Starting evaluation for seed $seed..."
    echo "Evaluating on FACET..."
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" python scripts/fairness_baseline/evaluate.py \
        --ckpt_dir "$RUN_OUTPUT_DIR" \
        --dataset_name facet \
        --csv_path datasets/facet_data/facet_eval.csv \
        --filter person \
        --img_size $IMG_SIZE \
        --resize_mode $RESIZE_MODE \
        --beta 0.2 \
        --cache_dir .cache/facet_crops

    echo "✅ Evaluation complete for seed $seed."
done

echo "✅ All ${#SEEDS[@]} repetitions complete."