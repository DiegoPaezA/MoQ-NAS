#!/bin/bash

# This script trains models for multiple architectures, running only two jobs
# (one for each dataset) in parallel at a time to avoid memory issues.

# Define the architectures to train
ARCHS="resnet18 resnet50 efficientnet_v2_s convnext_tiny mobilenet_v3_large mnasnet1_0"

# --- MODIFICATION START ---
# Define an empty variable for our optional arguments
OPTIONAL_ARGS=""
OUTPUT_DIR="checkpoints/baselines"

# Check if the first command-line argument is "head_only"
if [[ "$1" == "--head_only" ]]; then
    echo "✅ Running in HEAD-ONLY mode: Backbone will be frozen."
    # If it is, populate the variable with the flags you want
    OPTIONAL_ARGS="--freeze_backbone"
    OUTPUT_DIR="checkpoints/baselines_head_only"
else
    echo "✅ Running in FULL-TRAINING mode: The entire model will be trained."
fi
# --- MODIFICATION END ---

# Create a directory to store log files
LOG_DIR="logs"
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR
echo "✅ Log files will be saved in the '$LOG_DIR' directory."
echo "------------------------------------------------------"

# --- Main Training Loop ---
# This loop processes one architecture at a time.
for arch in $ARCHS; do
    echo "🚀 Starting parallel training for ARCHITECTURE: $arch"

    # --- Launch the facebin training job in the background ---
    LOG_FILE_FACEBIN="$LOG_DIR/facebin_${arch}.log"
    echo "--> Starting facebin training... Log: $LOG_FILE_FACEBIN"
    CUDA_VISIBLE_DEVICES=0 python scripts/fairness_baseline/train.py \
        --data_root datasets/facebin_data \
        --archs $arch \
        --out_dir $OUTPUT_DIR \
        --epochs 10 \
        $OPTIONAL_ARGS > "$LOG_FILE_FACEBIN" 2>&1 & # <-- VARIABLE ADDED

    # --- Launch the personbin training job in the background ---
    LOG_FILE_PERSONBIN="$LOG_DIR/personbin_${arch}.log"
    echo "--> Starting personbin training... Log: $LOG_FILE_PERSONBIN"
    CUDA_VISIBLE_DEVICES=1 python scripts/fairness_baseline/train.py \
        --data_root datasets/personbin_data \
        --archs $arch \
        --out_dir $OUTPUT_DIR \
        --epochs 10 \
        $OPTIONAL_ARGS > "$LOG_FILE_PERSONBIN" 2>&1 & # <-- VARIABLE ADDED

    # --- Wait for ONLY the two jobs above to complete ---
    echo "⏳ Waiting for $arch training (facebin & personbin) to complete..."
    wait
    echo "✅ Finished training for ARCHITECTURE: $arch"
    echo "------------------------------------------------------"
done

echo "✅ All architectures have been trained successfully."
echo "------------------------------------------------------"


# --- Proceed with Evaluation sequentially ---
echo "🚀 Starting evaluation..."

# Evaluate for FairFace
echo "Evaluating model on FairFace dataset..."
python scripts/fairness_baseline/evaluate.py \
    --ckpt_dir $OUTPUT_DIR \
    --dataset_name fairface \
    --csv_path datasets/FairFace/0.25/fairface_val.csv \
    --filter facebin

# Evaluate for FACET
echo "Evaluating model on FACET dataset..."
python scripts/fairness_baseline/evaluate.py \
    --ckpt_dir $OUTPUT_DIR \
    --dataset_name facet \
    --csv_path datasets/facet_data/facet_eval.csv \
    --filter personbin \
    --cache_dir .cache/facet_crops

echo "🎉 Evaluation complete. All tasks finished."