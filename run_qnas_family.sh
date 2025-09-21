#!/usr/bin/env bash
# ===============================
# run_qnas_family.sh
# QNAS / MO-QNAS runner
# ===============================

# —— Which algorithms to run (choose any of: qnas, moqnas) ——
algos=("qnas" "moqnas")

# —— Experiment settings ——
dataset="cifar10"
data_path="${dataset}_data"
config_dir="config_files_cifar"
exp_root="experiment_${dataset}_qfamily"
log_level="INFO"
network_config="default"
backbone_name="resnet18"           # used only if network_config="backbone"
fitness_metric="best_accuracy"

# NEW: dataset YAML path derived from dataset name
config_path_dataset="configs/${dataset}.yaml"

# —— Common toggles (kept from your scripts) ——
use_cache=false                     # :contentReference[oaicite:14]{index=14}
early_stopping=false                # :contentReference[oaicite:15]{index=15}
en_pop_crossover=true               # :contentReference[oaicite:16]{index=16}

# —— QNAS-specific ——
elite_mode_qnas="global_k"          # "single" | "global_k" | "bootstrap_k" | "old"  :contentReference[oaicite:17]{index=17}

# Architecture rule toggles (enabled by default in your repo).
# Set to false to APPEND the corresponding '--no-...' flag (matching your scripts).
truncate_after_noop=false           # :contentReference[oaicite:18]{index=18}
avoid_consecutive_pool=true         # :contentReference[oaicite:19]{index=19}
enforce_noop_in_update=true         # :contentReference[oaicite:20]{index=20}

# —— MO-QNAS-specific ——
optimizer="AdamW"                   # :contentReference[oaicite:21]{index=21}
save_checkpoints_epochs=5           # :contentReference[oaicite:22]{index=22}
data_augmentation=false             # :contentReference[oaicite:23]{index=23}
elite_mode_moqnas="moead_topk"      # "single"|"global_k"|"bootstrap_k"|"old"|"moead_topk"  :contentReference[oaicite:24]{index=24}
ref_dir_method="das-dennis"         # "das-dennis"|"dirichlet"                             :contentReference[oaicite:25]{index=25}
continue_path=""                    # resume path, keep empty if not resuming             :contentReference[oaicite:26]{index=26}

# —— dataset size & repeats ——
dataset_sample_size=10000           # :contentReference[oaicite:27]{index=27}
configs=("config6.txt")             # QNAS
exps=("exp18")
cuda_devices=("0,1")
num_repeats=3

# --------------- Runner ---------------
for ((j=0; j<${#configs[@]}; j++)); do
  cfg="${config_dir}/${configs[$j]}"
  exp="${exps[$j]}"
  cuda="${cuda_devices[$j]}"

  for algo in "${algos[@]}"; do
    echo "=== ${algo} | ${dataset} | ${configs[$j]} | CUDA=${cuda} ==="
    for ((i=1; i<=num_repeats; i++)); do
      exp_path="${exp_root}/${algo}/${exp}_repeat_${i}"
      mkdir -p "${exp_path}"

      # Common args
      COMMON_ARGS=(
        --experiment_path      "${exp_path}"
        --config_file          "${cfg}"
        --data_path            "${data_path}"
        --dataset              "${dataset}"
        --config_path_dataset  "${config_path_dataset}"
        --limit_data_value     "${dataset_sample_size}"
        --fitness_metric       "${fitness_metric}"
        --network_config       "${network_config}"
        --backbone_name        "${backbone_name}"
        --log_level            "${log_level}"
        $($use_cache && echo --use_cache)
        $($early_stopping && echo --early_stopping)
        $($en_pop_crossover && echo --en_pop_crossover)
        $($truncate_after_noop && echo --no-truncate-after-noop)
        $($avoid_consecutive_pool && echo --no-avoid-consecutive-pool)
        $($enforce_noop_in_update && echo --no-enforce-noop-in-update)
      )

      if [[ "${algo}" == "qnas" ]]; then
        CUDA_VISIBLE_DEVICES="${cuda}" python run_all_evolution.py \
          --algo qnas "${COMMON_ARGS[@]}" \
          --elite_mode "${elite_mode_qnas}"

      elif [[ "${algo}" == "moqnas" ]]; then
        CUDA_VISIBLE_DEVICES="${cuda}" python run_all_evolution.py \
          --algo moqnas "${COMMON_ARGS[@]}" \
          --optimizer "${optimizer}" \
          --save_checkpoints_epochs "${save_checkpoints_epochs}" \
          --elite_mode "${elite_mode_moqnas}" \
          --ref_dir_method "${ref_dir_method}" \
          $( [[ -n "${continue_path}" ]] && echo --continue_path "${continue_path}" ) \
          $( $data_augmentation && echo --data_augmentation )
      fi
    done
  done
done
