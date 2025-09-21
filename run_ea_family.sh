#!/usr/bin/env bash
# ================================
# run_ea_family.sh
# GA / NSGA-II / NSGA-III runner
# ================================

# —— Which algorithms to run (choose any of: ga, nsga2, nsga3) ——
#algos=("ga" "nsga2" "nsga3")
algos=("nsga3")
# —— Experiment settings ——
dataset="cifar10"
data_path="${dataset}_data"
config_dir="config_files_cifar"
exp_root="experiment_${dataset}_ea"
log_level="INFO"
network_config="default"
backbone_name="resnet18"   # used only if network_config="backbone"
fitness_metric="best_accuracy"

# —— GA/NSGA hyperparameters (from your GA/NSGA scripts) ——
population_size=20         
num_generations=150        
max_num_nodes=20           
crossover_rate=0.5         
mutation_rate=0.2          
elitism=true               
early_stopping=false       
use_cache=false            
patience=60                

# —— NSGA-III specific ——
ref_divisions=12           # set "" for auto

# —— dataset size & repeats ——
dataset_sample_size=10000 
configs=("config0.txt")    # same shape as your originals
exps=("exp_1")
cuda_devices=("0")
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
        --limit_data_value     "${dataset_sample_size}"
        --fitness_metric       "${fitness_metric}"
        --patience             "${patience}"
        --crossover_rate       "${crossover_rate}"
        --mutation_rate        "${mutation_rate}"
        --population_size      "${population_size}"
        --num_generations      "${num_generations}"
        --max_num_nodes        "${max_num_nodes}"
        --network_config       "${network_config}"
        --backbone_name        "${backbone_name}"
        --log_level            "${log_level}"
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
      fi
    done
  done
done
