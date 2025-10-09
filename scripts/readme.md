# MoQ-NAS: Multi-Objective Quantized Neural Architecture Search

This repository contains the official implementation of **MoQ-NAS**, a framework for finding hardware‑efficient and quantized neural network architectures using multi‑objective evolutionary algorithms.

---

## Fairness Baselines Workflow

This section details the end‑to‑end workflow for training and evaluating baseline models on fairness metrics. The process is broken down into four main stages:

1. **Download Datasets**: Download all required raw datasets.
2. **Prepare Datasets**: Process the raw data into binary classification formats.
3. **Train Baseline Models**: Train standard architectures (e.g., ResNet) on the prepared datasets.
4. **Evaluate Fairness**: Measure fairness of the trained models on the FairFace and FACET benchmarks.

---

### 1) Download Raw Datasets

Run the download scripts to fetch the **COCO**, **WIDER Face**, **Places365**, and **FairFace** datasets. These scripts will place the data in the `data/` directory.

```bash
# Download COCO subset, WIDER Face, and Places365
bash scripts/download_datasets/download_coco_subset.sh
python scripts/download_datasets/download_wider_places.py

# Download the FairFace dataset
python scripts/download_datasets/download_fairface.py
```

---

### 2) Prepare Datasets for Training

Process the raw data into the binary **PERSON vs. NON‑PERSON** and **FACE vs. NON‑FACE** formats required for training. The output will be saved to the `datasets/` directory by default.

```bash
# Build the PERSON vs. NON-PERSON dataset from COCO
python scripts/fairness_baseline/prepare_data.py --build_person

# Build the FACE vs. NON-FACE dataset from WIDER Face and Places365
python scripts/fairness_baseline/prepare_data.py --build_face

# Build FACET evaluation CSV
python scripts/fairness_baseline/prepare_data.py --build_facet
```

---

### 3) Train Baseline Models

Train the baseline models on the prepared datasets using the centralized training script. The script can train multiple architectures at once and saves checkpoints and a results CSV.

> **Tip:** Always run these commands from the project root directory (e.g., `~/MoQ-NAS`).

```bash
# Example: Train ResNet18 and ResNet50 on the 'personbin_data' dataset
python scripts/fairness_baseline/train.py \
    --data_root datasets/personbin_data \
    --archs resnet18,resnet50,efficientnet_v2_s,convnext_tiny,mobilenet_v3_large,mnasnet1_0 \
    --epochs 10
```

```bash

# Example: Train ResNet18 on the 'facebin_data' dataset, training only the head
python scripts/fairness_baseline/train.py \
    --data_root datasets/facebin_data \
    --archs resnet18,resnet50,efficientnet_v2_s,convnext_tiny,mobilenet_v3_large,mnasnet1_0 \
    --epochs 10
```

---

### 4) Evaluate Model Fairness

Evaluate the fairness of your trained models using the centralized evaluation script. This uses a unified `FairnessMetric` to ensure consistent calculations across datasets.

> **Tip:** Run from the project root directory.

```bash
# Example: Evaluate 'personbin' models on the FACET dataset (skin tone)
python scripts/fairness_baseline/evaluate.py \
    --ckpt_dir checkpoints/baselines \
    --dataset_name facet \
    --csv_path datasets/facet_data/facet_eval.csv \
    --filter personbin \
    --cache_dir .cache/facet_crops
```

```bash
# Example: Evaluate 'facebin' models on the FairFace dataset (race)
python scripts/fairness_baseline/evaluate.py \
    --ckpt_dir checkpoints/baselines \
    --dataset_name fairface \
    --csv_path datasets/FairFace/0.25/fairface_val.csv \
    --filter facebin
```

The evaluation script will generate a detailed `fairness_results_*.json` file in your checkpoint directory.

---

## Directory Expectations

- `datasets/` – raw datasets downloaded by the scripts.
- `datasets/` – processed binary datasets for training.
- `checkpoints/baselines/` – trained model weights and result summaries.
- `.cache/` – optional caches used during evaluation.

---

## Reproducibility

For deterministic runs, consider setting random seeds and CUDA flags as appropriate for your environment.
