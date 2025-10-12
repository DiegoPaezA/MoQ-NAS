# -*- coding: utf-8 -*-
""" Copyright (c) 2023, Diego Páez
    * Licensed under The MIT License [see LICENSE for details]

    - Distribute and Evaluate the population using multiple processes.
"""

import os
import torch
import time
import copy
from typing import Dict, Any
import torch.multiprocessing as mp
from .cnn import input, master
from utils.helpers import init_log, setup_dataset_info

worker_data_loader = None

class EvalPopulation(object):
    """
    Evaluate a population using a two-stage process.
    1. Primary objectives are evaluated in parallel.
    2. Expensive post-processing objectives (e.g., Fairness) are evaluated serially.

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
        self.loader = input.GenericDataLoader(params=params)
        self.train_params = setup_dataset_info(params)
        if 'objectives' not in self.train_params:
            raise KeyError("train_params must contain 'objectives' for evaluation.")
        
        multi_obj    = self.train_params.get('multi_objective', False)
        objectives   = self.train_params.get('objectives', [])
        self.train_params['fitness_metric'] = objectives[0] if isinstance(objectives, list) and objectives else 'best_accuracy'
        
        all_objectives = list(self.train_params['objectives'])
        self.fairness_metric_name = None
        self.primary_metric_names = []
        self.parallel_train_params = copy.deepcopy(self.train_params) # Params for subprocesses

        # Check if FairnessMetric is defined in the detailed metrics configuration
        fairness_metric_config = next((m for m in self.train_params.get('metrics', []) if m['name'] == 'FairnessMetric'), None)

        if fairness_metric_config:
            # Convention: The fairness score is the LAST objective in the 'objectives' list.
            self.fairness_metric_name = all_objectives[-1]
            self.primary_metric_names = all_objectives[:-1]

            # CRITICAL: Create a new 'metrics' list for the parallel trainers
            # that EXCLUDES the FairnessMetric. This prevents them from running it.
            self.parallel_train_params['metrics'] = [
                m for m in self.train_params.get('metrics', []) if m['name'] != 'FairnessMetric'
            ]
            self.logger.info(f"Primary metrics for PARALLEL evaluation: {self.primary_metric_names}")
            self.logger.info(f"Fairness metric for SERIAL evaluation: '{self.fairness_metric_name}'")
        else:
            self.primary_metric_names = all_objectives
            self.logger.info(f"All metrics will be evaluated in PARALLEL: {self.primary_metric_names}")
        
        self.metric_names = self.primary_metric_names

        #self.logger.info(f"Evaluation process initialized with {len(self.gpus)} GPUs")

        
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
            p = mp.Process(
                target=self.run_individuals,
                args=(generation, self.parallel_train_params, self.fn_dict, batch_thread, thread_id, result_queue)
            )
            p.start()
            processes.append(p)

        results: Dict[int, Dict[str, Any]] = {}
        model_specs = [None] * pop_size
        for _ in range(pop_size):
            idx, res_dict, model_path = result_queue.get()
            results[idx] = res_dict
            
            if model_path and os.path.exists(model_path):
                # Save spec for later fairness eval; do NOT build/load model here
                model_specs[idx] = {
                    "model_path": model_path,
                    "decoded_net": decoded_nets[idx],
                    "decoded_params": decoded_params[idx],
                    "info_dir": os.path.dirname(model_path)
                }
            else:
                model_specs[idx] = None
        for p in processes:
            p.join()
        self.logger.info("Parallel evaluation complete.")

        if self.fairness_metric_name:
            # Run on CUDA in a spawned child; parent stays CUDA-free
            fairness_metrics_dict = self.evaluate_fairness_parallel_cuda(model_specs)
            self.logger.info("Merging fairness results...")
            for idx, fairness_scores in fairness_metrics_dict.items():
                if idx in results:
                    results[idx].update(fairness_scores)
                    score = fairness_scores.get(self.fairness_metric_name, fairness_scores.get("fairness_score"))
                    self.logger.info(f"Candidate {idx} - {self.fairness_metric_name}: {score:.4f}")
            self.logger.info("Merging complete.")

        evol_end_time = time.perf_counter()
        mins, secs = divmod(evol_end_time - evol_time_start, 60)
        self.logger.info(f"Total time elapsed for generation {generation}: {int(mins)}m {int(secs)}s")

        return results

    def run_individuals(self, generation, train_params, fn_dict, individuals_thread, thread_id, queue: mp.Queue):
        # --- Cambios mínimos para rendimiento y correctitud por proceso ---
        # 1. Determine the device.
        global worker_data_loader
        gpu_device = 'cpu' # Default to CPU
        try:
            # This is the first and only place CUDA is initialized for this process.
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                if num_gpus > 0:
                    gpu_idx = thread_id % num_gpus
                    torch.cuda.set_device(gpu_idx)
                    gpu_device = f'cuda:{gpu_idx}'
        except Exception as e:
            self.logger.error(f"CUDA initialization failed for thread {thread_id}: {e}. Falling back to CPU.")
            gpu_device = 'cpu'


        # 2) Evitar sobre-suscripción de hilos BLAS/CPU dentro de cada proceso
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

        # if worker_data_loader is None:
        #     worker_data_loader = input.GenericDataLoader(params=train_params)
            #self.train_params = setup_dataset_info(self.train_params)
        # 3) Reutilizar DataLoaders por proceso (ya lo hacías) — una sola construcción
        train_loader, val_loader = self.loader.get_loader(pin_memory_device=gpu_device)

        for original_idx, thread_id, decoded_net, decoded_params in individuals_thread:
            id_str = f"{generation}_{decoded_params.get('candidate_id', original_idx)}"
            model_path = None
            try:
                results_dict, model_path = master.fitness(id_str,
                            {**train_params, 'device': gpu_device},
                            fn_dict,
                            decoded_net,
                            decoded_params,
                            train_loader,
                            val_loader)
            except RuntimeError as e:
                # Manejo específico de OOM u otros errores de runtime
                self.logger.error(f"RuntimeError training model {id_str}: {e}")
                results_dict = {k: 0.0 for k in self.metric_names}
            except Exception as e:
                # Cualquier otro error
                self.logger.error(f"Error training model {id_str}: {e}")
                results_dict = {k: 0.0 for k in self.metric_names}
            filtered_results = {k: results_dict.get(k, 0.0) for k in self.metric_names}
            queue.put((original_idx, filtered_results, model_path))

            # Log the results for each candidate
            if not results_dict:
                self.logger.warning(f"Thread {thread_id} – candidate {original_idx}: No results returned.")
                continue
            else:
                metrics_log = ", ".join(f"{k}={results_dict.get(k, 0.0):.3f}" for k in self.metric_names)
                self.logger.info(f"Thread {thread_id} – candidate {original_idx}: {metrics_log}")

    # inside EvalPopulation
    def evaluate_fairness_in_cuda_spawn(self, model_specs: list, device_idx: int | None = None):
        import torch.multiprocessing as mp
        from core.fairness.fairness_worker import fairness_spawn_runner  # top-level function

        metric_config = next(
            (m for m in self.train_params.get('metrics', []) if m['name'] == 'FairnessMetric'),
            None
        )
        if not metric_config:
            return {i: {self.fairness_metric_name: 0.0} for i in range(len(model_specs))}
        fairness_params = metric_config.get('params', {}) or {}

        shard = [(i, spec) for i, spec in enumerate(model_specs) if spec]

        ctx = mp.get_context("spawn")
        out_q = ctx.Queue()
        p = ctx.Process(
            target=fairness_spawn_runner,
            args=(
                out_q,
                shard,
                self.parallel_train_params,  # must be a plain dict
                self.fn_dict,                # ensure only top-level callables inside
                self.fairness_metric_name,
                fairness_params,
                device_idx,
            ),
        )
        p.start()
        merged = out_q.get()
        p.join()

        for i in range(len(model_specs)):
            merged.setdefault(i, {self.fairness_metric_name: 0.0})
        return merged
    
    def evaluate_fairness_parallel_cuda(self, model_specs: list, processes_per_gpu: int = 10) -> Dict[int, Dict[str, float]]:
        import torch.multiprocessing as mp
        from core.fairness.fairness_worker import (
            device_count_probe_runner,
            fairness_queue_runner,
        )

        # Parent-safe metric config
        metric_config = next((m for m in self.train_params.get('metrics', [])
                            if m['name'] == 'FairnessMetric'), None)
        if not metric_config:
            return {i: {self.fairness_metric_name: 0.0} for i in range(len(model_specs))}
        fairness_params = metric_config.get('params', {}) or {}
        input_shape = self.train_params.get('input_shape', (3, 224, 224))
        fairness_params['img_size'] = input_shape[2]

        # Probe GPU count in a spawned child (parent stays CUDA-free)
        ctx = mp.get_context("spawn")
        probe_q = ctx.Queue()
        probe_p = ctx.Process(target=device_count_probe_runner, args=(probe_q,))
        probe_p.start()
        num_gpus = probe_q.get()
        probe_p.join()

        # Build index list
        indices = [i for i, spec in enumerate(model_specs) if spec]
        if not indices:
            return {i: {self.fairness_metric_name: 0.0} for i in range(len(model_specs))}

        # ---- NEW: fan out to multiple processes per GPU ----
        # total "virtual slots" = num_gpus * processes_per_gpu
        # (If there are 0 GPUs, keep 0 slots; we’ll end with empty procs and fill zeros later.)
        slots = max(0, num_gpus * max(1, int(processes_per_gpu)))
        if slots == 0:
            # No GPU present; fall back to zeros (keeps previous behavior)
            merged = {}
            for i in range(len(model_specs)):
                merged.setdefault(i, {self.fairness_metric_name: 0.0})
            return merged

        # Distribute work round-robin across slots
        shards = [[] for _ in range(slots)]
        for k, idx in enumerate(indices):
            shards[k % slots].append((idx, model_specs[idx]))

        # Launch one spawned worker per slot, binding each to a GPU via device_idx = slot % num_gpus
        res_q = ctx.Queue()
        procs = []
        for slot_id, shard in enumerate(shards):
            if not shard:
                continue
            dev_idx = slot_id % num_gpus
            p = ctx.Process(
                target=fairness_queue_runner,
                args=(
                    res_q,
                    shard,
                    self.parallel_train_params,
                    self.fn_dict,
                    self.fairness_metric_name,
                    fairness_params,
                    dev_idx,  # each slot pinned to a GPU index
                ),
            )
            p.start()
            procs.append(p)

        # Gather results
        merged: Dict[int, Dict[str, float]] = {}
        for _ in range(len(procs)):
            part = res_q.get()
            merged.update(part)

        for p in procs:
            p.join()

        # Fill any missing entries
        for i in range(len(model_specs)):
            merged.setdefault(i, {self.fairness_metric_name: 0.0})

        return merged
