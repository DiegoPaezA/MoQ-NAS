#!/bin/bash

# --- 🎮 GPU SELECTION ---
# Define your GPUs here.
# Case A: One GPU -> Runs SEQUENTIALLY (safe for single GPU)
CUDA_DEVICES=("1") 

# # Case B: Two GPUs -> Runs in PARALLEL (faster)
# CUDA_DEVICES=("0" "1")

# --- CONFIGURATION ---
# Define the architectures to train
ARCHS="resnet18 resnet50 efficientnet_v2_s convnext_tiny mobilenet_v3_large mnasnet1_0"

# Image settings (Must match your prepared data)
IMG_SIZE=96
RESIZE_MODE="letterbox" # or "center_crop"

# Config files (Ensure these exist)
CONFIG_PERSON="configs/person_bin_96.yaml"
CONFIG_FACE="configs/face_bin_96.yaml"

# Data paths (Root directory of the specific dataset)
DATA_PERSON="datasets/personbin_data_96"
DATA_FACE="datasets/facebin_data_96"

# ---------------------

# Define an empty variable for our optional arguments
OPTIONAL_ARGS=""
OUTPUT_DIR="checkpoints/baseline_limit_96"

# Check if the first command-line argument is "--head_only"
if [[ "$1" == "--head_only" ]]; then
    echo "✅ Running in HEAD-ONLY mode: Backbone will be frozen."
    OPTIONAL_ARGS="--freeze_backbone"
    OUTPUT_DIR="checkpoints/baseline_freeze_limit_96"
else
    echo "✅ Running in FULL-TRAINING mode: The entire model will be trained."
fi

# Create a directory to store log files
LOG_DIR="logs"
NUM_GPUS=${#CUDA_DEVICES[@]}
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR
echo "✅ Log files will be saved in the '$LOG_DIR' directory."
echo "✅ Checkpoints will be saved in '$OUTPUT_DIR'"
echo "✅ GPU Setup: Defined ${NUM_GPUS} device(s): ${CUDA_DEVICES[*]}"
echo "------------------------------------------------------"


for arch in $ARCHS; do
    echo "🚀 Processing ARCHITECTURE: $arch"
    
    LOG_FILE_FACEBIN="$LOG_DIR/facebin_${arch}.log"
    LOG_FILE_PERSONBIN="$LOG_DIR/personbin_${arch}.log"

    if [ "$NUM_GPUS" -ge 2 ]; then
        # ================= PARALLEL MODE =================
        GPU_A=${CUDA_DEVICES[0]}
        GPU_B=${CUDA_DEVICES[1]}

        echo "⚡ Mode: PARALLEL"
        echo "   --> [GPU $GPU_A] Facebin training..."
        CUDA_VISIBLE_DEVICES="$GPU_A" python scripts/fairness_baseline/train.py \
            --config_file $CONFIG_FACE \
            --data_path $DATA_FACE \
            --experiment_path $OUTPUT_DIR \
            --archs $arch \
            --max_epochs 10 \
            --limit_data \
            --limit_data_value 10000\
            $OPTIONAL_ARGS > "$LOG_FILE_FACEBIN" 2>&1 &
            
        echo "   --> [GPU $GPU_B] Personbin training..."
        CUDA_VISIBLE_DEVICES="$GPU_B" python scripts/fairness_baseline/train.py \
            --config_file $CONFIG_PERSON \
            --data_path $DATA_PERSON \
            --experiment_path $OUTPUT_DIR \
            --archs $arch \
            --max_epochs 10 \
            --limit_data \
            --limit_data_value 10000\
            $OPTIONAL_ARGS > "$LOG_FILE_PERSONBIN" 2>&1 &
            
        wait
    else
        # ================= SEQUENTIAL MODE =================
        GPU_A=${CUDA_DEVICES[0]}
        
        echo "🐢 Mode: SEQUENTIAL (using GPU $GPU_A)"
        
        echo "   --> [1/2] Facebin training..."
        CUDA_VISIBLE_DEVICES="$GPU_A" python scripts/fairness_baseline/train.py \
            --config_file $CONFIG_FACE \
            --data_path $DATA_FACE \
            --experiment_path $OUTPUT_DIR \
            --archs $arch \
            --max_epochs 10 \
            --limit_data \
            --limit_data_value 10000\
            $OPTIONAL_ARGS > "$LOG_FILE_FACEBIN" 2>&1
            
        echo "   --> [2/2] Personbin training..."
        CUDA_VISIBLE_DEVICES="$GPU_A" python scripts/fairness_baseline/train.py \
            --config_file $CONFIG_PERSON \
            --data_path $DATA_PERSON \
            --experiment_path $OUTPUT_DIR \
            --archs $arch \
            --max_epochs 10 \
            --limit_data \
            --limit_data_value 10000\
            $OPTIONAL_ARGS > "$LOG_FILE_PERSONBIN" 2>&1
    fi

    echo "✅ Finished ARCHITECTURE: $arch"
    echo "------------------------------------------------------"
done

# --- Evaluation ---
echo "🚀 Starting evaluation..."
CKPT_DIR="$OUTPUT_DIR/baselines"

# We use the first defined GPU for evaluation
EVAL_GPU=${CUDA_DEVICES[0]}

echo "Evaluating on FairFace..."
CUDA_VISIBLE_DEVICES="$EVAL_GPU" python scripts/fairness_baseline/evaluate.py \
    --ckpt_dir $CKPT_DIR \
    --dataset_name fairface \
    --csv_path datasets/FairFace/0.25/fairface_val.csv \
    --filter face \
    --img_size $IMG_SIZE \
    --resize_mode $RESIZE_MODE \
    --beta 0.2

echo "Evaluating on FACET..."
CUDA_VISIBLE_DEVICES="$EVAL_GPU" python scripts/fairness_baseline/evaluate.py \
    --ckpt_dir $CKPT_DIR \
    --dataset_name facet \
    --csv_path datasets/facet_data/facet_eval.csv \
    --filter person \
    --img_size $IMG_SIZE \
    --resize_mode $RESIZE_MODE \
    --beta 0.2 \
    --cache_dir .cache/facet_crops

echo "✅ Evaluation complete."