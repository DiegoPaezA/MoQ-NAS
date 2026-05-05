#!/bin/bash

############################################################################
# CONFIGURACIÓN
############################################################################

# ---- ¡AJUSTA ESTA VARIABLE! ----
# Define el número máximo de procesos que se ejecutarán en paralelo.
# Con 2 GPUs, un buen punto de partida es 2, 4 o 6, dependiendo de la memoria
# de tus GPUs y la demanda de cada proceso.
MAX_PARALLEL_PROCESSES=3

# GPUs disponibles (separa los IDs con espacios)
GPUS_AVAILABLE=(1)
NUM_GPUS=${#GPUS_AVAILABLE[@]}

# Variables generales del experimento
dataset="cifar10"
network_config="default"

############################################################################
# FUNCIÓN DE ENTRENAMIENTO
############################################################################

# Esta función encapsula el comando para un solo reentrenamiento
# Argumentos: 1=experimento, 2=repetición, 3=ID de la GPU
run_retrain_task() {
    local exp=$1
    local repeat=$2
    local gpu_id=$3

    echo "▶️  Iniciando $exp | Repetición $repeat | en GPU:$gpu_id"
    local exp_path="experiment_${dataset}_qfamily/qnas/${exp}_repeat_${repeat}"

    # Se ejecuta el comando en segundo plano (&) en la GPU especificada.
    # El script de Python siempre verá la GPU asignada como 'cuda:0' gracias
    # a la variable de entorno CUDA_VISIBLE_DEVICES.
    CUDA_VISIBLE_DEVICES=$gpu_id python retrain_model.py \
        --experiment_path "$exp_path" \
        --data_path "datasets/${dataset}_data" \
        --dataset "$dataset" \
        --retrain_folder retrain \
        --config_code F13 \
        --log_level INFO \
        --max_epochs 300 \
        --epochs_to_eval 300 \
        --patience_retrain 300 \
        --batch_size 256 \
        --eval_batch_size 256 \
        --device "cuda:0" \
        --num_repetitions 3 \
        --lr_scheduler "multistep" \
        --data_augmentation \
        --network_config "$network_config" \
        --optimizer "AdamW" &
}

############################################################################
# ORQUESTADOR PRINCIPAL
############################################################################

# Contador para asignar las GPUs de forma rotativa
tasks_launched=0

# Bucle para los experimentos del 8 al 11
#for exp_num in {8..11}; do
for exp_num in 23; do
    exp="exp${exp_num}"
    # Bucle para las 3 repeticiones de cada experimento
    for repeat in {1..3}; do
        # Si ya hemos lanzado el número máximo de procesos permitidos,
        # esperamos a que uno termine antes de lanzar el siguiente.
        if (( tasks_launched >= MAX_PARALLEL_PROCESSES )); then
            wait -n
        fi

        # Asignamos una GPU de forma cíclica (0, 1, 0, 1, ...)
        gpu_index=$((tasks_launched % NUM_GPUS))
        gpu_id=${GPUS_AVAILABLE[$gpu_index]}

        # Lanzamos la tarea de reentrenamiento en segundo plano
        run_retrain_task "$exp" "$repeat" "$gpu_id"

        # Incrementamos el contador de tareas lanzadas
        ((tasks_launched++))
    done
done

# Espera final: nos aseguramos de que todos los procesos restantes terminen
echo "✅ Todos los procesos han sido lanzados. Esperando a que los últimos finalicen..."
wait
echo "🎉 ¡Todos los reentrenamientos han finalizado con éxito!"