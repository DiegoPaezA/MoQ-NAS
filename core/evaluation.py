# -*- coding: utf-8 -*-
""" Copyright (c) 2023, Diego Páez
    * Licensed under The MIT License [see LICENSE for details]

    - Distribute and Evaluate the population using multiple processes.
"""

import os
import torch
import time
from typing import Dict, Any
import torch.multiprocessing as mp
from .cnn import input, master
from utils.helpers import init_log, setup_dataset_info


class EvalPopulation(object):
    """
    Evaluate a population using multiple processes.

    This class is designed to distribute the evaluation of a population of models
    using multiple processes.
    
    Parameters
    ----------
    params : dict
        A dictionary containing parameters for the evaluation process.
    fn_dict : dict
        A dictionary containing definitions of the functions.
    log_level : str, optional
        The logging level for the internal logger (default is 'INFO').

    Attributes
    ----------
    train_params : dict
        Parameters for the training and evaluation process.
    fn_dict : dict
        Definitions of the functions used in the evaluation.
    timeout : int
        Timeout value for the Dask operations.
    logger : logger
        Internal logger for logging messages.
    gpus : list
        List of GPU devices available for evaluation.
    client : Client
        Dask client for managing the distributed computation.

    Methods
    -------
    __call__(decoded_params, decoded_nets, generation)
        Perform the evaluation of the population.
    
    """
    def __init__(self, params: dict, fn_dict: dict, log_level: str = 'INFO'):
        """
        Initialize the EvalPopulation object.

        Arguments:
        params : dict
            A dictionary containing parameters for the evaluation process.
        fn_dict : dict
            A dictionary containing definitions of the functions.
        log_level : str, optional
            The logging level for the internal logger (default is 'INFO').
        """
        self.fn_dict = fn_dict
        self.logger = init_log(log_level, name=__name__)
        self.gpus = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        self.loader = input.GenericDataLoader(params=params)
        self.train_params = setup_dataset_info(params)
        
        if 'objectives' not in self.train_params:
            raise KeyError("train_params must contain 'objectives' for evaluation.")
        
        multi_obj    = self.train_params.get('multi_objective', False)
        objectives   = self.train_params.get('objectives', [])
        self.train_params['fitness_metric'] = objectives[0] if isinstance(objectives, list) and objectives else 'best_accuracy'
        
        # if not multi_obj:
        #     # Reemplaza únicamente el primer objetivo por la métrica principal
        #     if isinstance(objectives, list) and objectives:
        #         objectives[0] = main_metric
        #     else:
        #         # Si no había lista o estaba vacía, inicializa con la principal
        #         objectives = [main_metric]
        #     self.train_params['objectives'] = objectives
        #     self.logger.info(f"Setting main metric '{main_metric}' as the first objective.")
        self.metric_names = self.train_params['objectives']
        
        # mp.set_start_method('spawn')  # <- Mantener comentado en Linux para alto rendimiento con 'fork'
        self.logger.info(f"Evaluation process initialized with {len(self.gpus)} GPUs")        
        
    def __call__(self, decoded_params: list, decoded_nets: list, generation: int):
        """
        Evaluate the population.

        Parameters
        ----------
        decoded_params : list
            List of dictionaries containing the parameters for each model.
        decoded_nets : list
            List of lists containing the network architectures for each model.
        generation : int
            The generation number for tracking purposes.

        Returns
        -------
        evaluations : dict
            A dictionary containing the evaluations of the population, where each key is a candidate ID
            and the value is a list of evaluation metrics { original_index: { 'metric1': val1, 'metric2': val2, … }, … }.

        Raises
        ------
        TimeoutError
            If the Dask operations exceed the specified timeout.
        """
        pop_size = len(decoded_nets)
        result_queue: mp.Queue = mp.Queue()
        
        # --- THIS IS THE FINAL LOGIC ---
        gpu_semaphores = None
        # Check if FairnessMetric is in the user's configuration
        is_fairness_enabled = any(
            metric.get('name') == 'FairnessMetric' for metric in self.train_params.get('metrics', [])
        )

        if is_fairness_enabled:
            manager = mp.Manager()
            if self.gpus:
                # Create a semaphore for each specific GPU
                gpu_semaphores = {gpu: manager.Semaphore(1) for gpu in self.gpus}
                self.logger.info(f"FairnessMetric enabled. Creating a dedicated semaphore for each of {len(self.gpus)} GPUs.")
            else: # Fallback for CPU
                gpu_semaphores = {"cpu": manager.Semaphore(1)}
        else:
            self.logger.info("FairnessMetric not found in config. Semaphore locking is disabled.")

        # Temporal solution to distribute the individuals in the threads
        selected_thread = 0
        individual_per_thread = []
        for idx in range(pop_size):
            individual_per_thread.append((idx, selected_thread, decoded_nets[idx], decoded_params[idx]))
            selected_thread += 1
            if selected_thread >= self.train_params['threads']:
                selected_thread = selected_thread % self.train_params['threads']
        
        processes = []
        print("\n")
        self.logger.info(f"Starting the Generation {generation} with {pop_size} individuals")
        evol_time_start = time.perf_counter()

        # Create a process for each thread
        for thread_id in range(self.train_params['threads']):
            batch_thread = [x for x in individual_per_thread if x[1] == thread_id]
            if not batch_thread:
                continue
            gpu_device = self.gpus[thread_id % len(self.gpus)]
            p = mp.Process(
                target=self.run_individuals,
                args=(generation, self.train_params, self.fn_dict, batch_thread, gpu_device, result_queue, gpu_semaphores)
            )
            p.start()
            processes.append(p)

        results: Dict[int, Dict[str, Any]] = {}
        for _ in range(pop_size):
            idx, res_dict = result_queue.get()
            # return only the self.metric_names
            res_dict = {k: res_dict[k] for k in self.metric_names if k in res_dict}
            results[idx] = res_dict
        
        for p in processes:
            p.join()

        evol_end_time = time.perf_counter()
        mins, secs = divmod(evol_end_time - evol_time_start, 60)
        self.logger.info(f"Time elapsed for {pop_size} individuals: {int(mins)}m {int(secs)}s")

        return results

    def run_individuals(self, generation, train_params, fn_dict, individuals_thread, gpu_device, queue: mp.Queue, gpu_semaphores=None):
        # --- Cambios mínimos para rendimiento y correctitud por proceso ---
        # 1) Fijar dispositivo CUDA explícito por proceso (evita 'current_device' heredado)
        if isinstance(gpu_device, str) and gpu_device.startswith("cuda"):
            torch.cuda.set_device(int(gpu_device.split(":")[1]))
            torch.backends.cudnn.benchmark = True  # acelera convs con tamaños fijos

        # 2) Evitar sobre-suscripción de hilos BLAS/CPU dentro de cada proceso
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

        # 3) Reutilizar DataLoaders por proceso (ya lo hacías) — una sola construcción
        train_loader, val_loader = self.loader.get_loader(pin_memory_device=gpu_device)

        for original_idx, thread_id, decoded_net, decoded_params in individuals_thread:
            id_str = f"{generation}_{decoded_params.get('candidate_id', original_idx)}"
            try:
                results_dict = master.fitness(id_str,
                            {**train_params, 'device': gpu_device},
                            fn_dict,
                            decoded_net,
                            decoded_params,
                            train_loader,
                            val_loader,
                            gpu_semaphores=gpu_semaphores)
            except RuntimeError as e:
                # Manejo específico de OOM u otros errores de runtime
                self.logger.error(f"RuntimeError training model {id_str}: {e}")
                results_dict = {k: 0.0 for k in self.metric_names}
            except Exception as e:
                # Cualquier otro error
                self.logger.error(f"Error training model {id_str}: {e}")
                results_dict = {k: 0.0 for k in self.metric_names}
            
            queue.put((original_idx, results_dict))

            # Log the results for each candidate
            if not results_dict:
                self.logger.warning(f"Thread {thread_id} – candidate {original_idx}: No results returned.")
                continue
            else:
                metrics_log = ", ".join(f"{k}={results_dict.get(k, 0.0):.2f}" for k in self.metric_names)
                self.logger.info(f"Thread {thread_id} – candidate {original_idx}: {metrics_log}")