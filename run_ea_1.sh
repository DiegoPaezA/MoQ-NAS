#!/usr/bin/env bash
# ================================
# run_ea_family.sh
# GA / NSGA-II / NSGA-III / MOEA-D runner
# ================================

# —— Which algorithms to run (choose any of: ga, nsga2, nsga3, moead) ——
#algos=("ga" "nsga2" "nsga3" "moead")
algos=("ga")

# —— Experiment settings ——
dataset="cifar10"
data_path="datasets/${dataset}_data"
config_dir="experiment_configs/cifar"
exp_root="experiment_${dataset}_ea"
log_level="INFO"
network_config="default"
backbone_name="resnet18"   # used only if network_config="backbone"
fitness_metric="best_accuracy"

# NEW: dataset YAML path derived from dataset name
config_path_dataset="dataset_configs/${dataset}.yaml"

# —— GA/NSGA/MOEA-D shared hyperparameters ——
population_size=20
num_generations=150
max_num_nodes=20
crossover_rate=0.5
mutation_rate=0.2
mutation_strategy="standard" # options: random, standard, swap, block, neighbor
elitism=true
early_stopping=false
use_cache=false
patience=60

# —— NSGA-III & MOEA-D both can use lattice divisions (Das & Dennis) ——
ref_divisions=12           # set "" for auto

# —— MOEA-D specific ——
moead_T=20                 # neighborhood size
moead_scalar="tchebycheff" # or "weighted_sum"
moead_pneighbor=0.9        # prob of mating within neighborhood

# —— dataset size & repeats ——
dataset_sample_size=10000
configs=("config0.yaml")
exps=("exp4")
cuda_devices=("1")
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
        --patience             "${patience}"
        --crossover_rate       "${crossover_rate}"
        --mutation_rate        "${mutation_rate}"
        --mutation_strategy    "${mutation_strategy}"
        --population_size      "${population_size}"
        --num_generations      "${num_generations}"
        --max_num_nodes        "${max_num_nodes}"
        --network_config       "${network_config}"
        --backbone_name        "${backbone_name}"
        --log_level            "${log_level}"
        --gpu_list             "${cuda_devices}"
        $($elitism && echo --elitism)
        $($use_cache && echo --use_cache)
        $($early_stopping && echo --early_stopping)
      )

      if [[ "${algo}" == "ga" ]]; then
        CUDA_VISIBLE_DEVICES="${cuda}" python run_all_evolution.py \
          --algo ga "${COMMON_ARGS[@]}"

      elif [[ "${algo}" == "nsga2" ]]; then
        CUDA_VISIBLE_DEVICES="${cuda}" python run_all_evolution.py \
          --algo nsga2 "${COMMON_ARGS[@]}"

      elif [[ "${algo}" == "nsga3" ]]; then
        CUDA_VISIBLE_DEVICES="${cuda}" python run_all_evolution.py \
          --algo nsga3 "${COMMON_ARGS[@]}" \
          $( [[ -n "${ref_divisions}" ]] && echo --ref_divisions "${ref_divisions}" )

      elif [[ "${algo}" == "moead" ]]; then
        CUDA_VISIBLE_DEVICES="${cuda}" python run_all_evolution.py \
          --algo moead "${COMMON_ARGS[@]}" \
          $( [[ -n "${ref_divisions}" ]] && echo --ref_divisions "${ref_divisions}" ) \
          --moead_T "${moead_T}" \
          --moead_scalar "${moead_scalar}" \
          --moead_pneighbor "${moead_pneighbor}"
      fi
    done
  done
done