# -*- coding: utf-8 -*-
""" Copyright (c) 2023, Diego Páez
    * Licensed under The MIT License [see LICENSE for details]

    - Distribute and Evaluate the population using multiple processes.
"""

import os
import torch
import time
import copy
from typing import Dict, Any, List
import torch.multiprocessing as mp
from .cnn import input, master
from utils.helpers import init_log, setup_dataset_info
from utils.seeding import seed_candidate

worker_data_loader = None

class EvalPopulation(object):
    """
    Evaluate a population using a two-stage process.
    1. Primary objectives are evaluated in parallel.
    2. Expensive post-processing objectives (e.g., Fairness) are evaluated serially.
    """
    def __init__(self, params: dict, fn_dict: dict, log_level: str = 'INFO'):
        self.fn_dict = fn_dict
        self.logger = init_log(log_level, name=__name__)
        self.global_seed = int(params.get('seed', 42))
        self.loader = input.GenericDataLoader(params=params)
        self.train_params = setup_dataset_info(params)
        if 'objectives' not in self.train_params:
            raise KeyError("train_params must contain 'objectives' for evaluation.")
        
        objectives = self.train_params.get('objectives', [])
        self.train_params['fitness_metric'] = objectives[0] if isinstance(objectives, list) and objectives else 'best_accuracy'
        
        all_objectives = list(self.train_params['objectives'])
        
        # Initialize lists
        self.fairness_metric_names = []
        self.primary_metric_names = []
        self.parallel_train_params = copy.deepcopy(self.train_params) # Params for subprocesses

        # Check if FairnessMetric is defined in the detailed metrics configuration
        fairness_metric_config = next((m for m in self.train_params.get('metrics', []) if m['name'] == 'FairnessMetric'), None)
        self.fairness_key_map = {
                'fairness_spd': 'spd_sum',
                'fairness_mean_tpr': 'mean_tpr',
                'fairness_score': 'fairness_score'
            }
        if fairness_metric_config:
            # 1. DYNAMIC SEPARATION
            # Filter objectives containing 'fairness' into the serial list
            self.fairness_metric_names = [obj for obj in all_objectives if 'fairness' in obj]
            
            # Everything else is a primary metric (run in parallel)
            self.primary_metric_names = [obj for obj in all_objectives if 'fairness' not in obj]

            # 2. CRITICAL: Exclude FairnessMetric from parallel workers
            self.parallel_train_params['metrics'] = [
                m for m in self.train_params.get('metrics', []) if m['name'] != 'FairnessMetric'
            ]
            self.logger.info(f"Primary metrics for PARALLEL evaluation: {self.primary_metric_names}")
            self.logger.info(f"Fairness metrics for SERIAL evaluation: {self.fairness_metric_names}")
        else:
            self.primary_metric_names = all_objectives
            self.logger.info(f"All metrics will be evaluated in PARALLEL: {self.primary_metric_names}")
        
        # Combined list for tracking purposes
        self.metric_names = self.primary_metric_names + self.fairness_metric_names

    def __call__(self, decoded_params: list, decoded_nets: list, generation: int):
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
                # Save spec for later fairness eval
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

        # --- FAIRNESS EVALUATION BLOCK ---
        if self.fairness_metric_names:
            # Run parallel evaluation
            fairness_results = self.evaluate_fairness_parallel_cuda(model_specs)
            self.logger.info("Merging fairness results...")
            
            for idx, f_metrics in fairness_results.items():
                if idx in results:
                    # 1. Update the candidate's result with all raw data (useful for logs/storage)
                    results[idx].update(f_metrics)
                    
                    log_parts = []
                    # 2. Extract specifically the objectives requested in the config
                    for obj_name in self.fairness_metric_names:
                        # Use the map to find the internal key (e.g., 'fairness_spd' -> 'spd_sum')
                        internal_key = self.fairness_key_map.get(obj_name)
                        
                        if not internal_key:
                            internal_key = obj_name

                        val = f_metrics.get(internal_key)
                        
                        if val is not None:
                            results[idx][obj_name] = val 
                            log_parts.append(f"{obj_name}: {val:.4f}")
                        else:
                            self.logger.warning(f"Metric '{internal_key}' not found for candidate {idx}")
                            results[idx][obj_name] = 0.0 # Default fallback
                            log_parts.append(f"{obj_name}: N/A")

                    log_msg = ", ".join(log_parts)
                    self.logger.info(f"Candidate {idx} - {log_msg}")
        else:
            # remove the model files if no fairness evaluation is done
            for spec in model_specs:
                if spec and os.path.exists(spec["model_path"]):
                    try:
                        os.remove(spec["model_path"])
                    except Exception as e:
                        self.logger.warning(f"Could not remove model file {spec['model_path']}: {e}")

        evol_end_time = time.perf_counter()
        mins, secs = divmod(evol_end_time - evol_time_start, 60)
        self.logger.info(f"Total time elapsed for generation {generation}: {int(mins)}m {int(secs)}s")

        return results

    def run_individuals(self, generation, train_params, fn_dict, individuals_thread, thread_id, queue: mp.Queue):
        global worker_data_loader
        gpu_device = 'cpu'
        try:
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                if num_gpus > 0:
                    gpu_idx = thread_id % num_gpus
                    torch.cuda.set_device(gpu_idx)
                    gpu_device = f'cuda:{gpu_idx}'
        except Exception as e:
            self.logger.error(f"CUDA initialization failed for thread {thread_id}: {e}. Falling back to CPU.")
            gpu_device = 'cpu'

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

        train_loader, val_loader = self.loader.get_loader(pin_memory_device=gpu_device)

        for original_idx, thread_id, decoded_net, decoded_params in individuals_thread:
            candidate_id = decoded_params.get('candidate_id', original_idx)
            id_str = f"{generation}_{candidate_id}"
            model_path = None
            # Per-candidate deterministic seeding: accuracy must not depend on
            # which worker trains the candidate or in what order.
            cand_seed_id = candidate_id if isinstance(candidate_id, int) else original_idx
            cand_seed = seed_candidate(self.global_seed, generation, cand_seed_id)
            # The loader is shared by all candidates of this process; reseed its
            # generator so shuffle order/augmentation don't depend on how many
            # candidates were trained before in the same process.
            if getattr(train_loader, 'generator', None) is not None:
                train_loader.generator.manual_seed(cand_seed)
            try:
                results_dict, model_path = master.fitness(id_str,
                            {**train_params, 'device': gpu_device},
                            fn_dict,
                            decoded_net,
                            decoded_params,
                            train_loader,
                            val_loader)
            except RuntimeError as e:
                self.logger.error(f"RuntimeError training model {id_str}: {e}")
                results_dict = {k: 0.0 for k in self.metric_names}
            except Exception as e:
                self.logger.error(f"Error training model {id_str}: {e}")
                results_dict = {k: 0.0 for k in self.metric_names}
            
            # Filter results for current thread (primary metrics only)
            filtered_results = {k: results_dict.get(k, 0.0) for k in self.primary_metric_names}
            queue.put((original_idx, filtered_results, model_path))

            if not results_dict:
                self.logger.warning(f"Thread {thread_id} – candidate {original_idx}: No results returned.")
                continue
            else:
                metrics_log = ", ".join(f"{k}={results_dict.get(k, 0.0):.3f}" for k in self.primary_metric_names)
                self.logger.info(f"Thread {thread_id} – candidate {original_idx}: {metrics_log}")
    
    def evaluate_fairness_parallel_cuda(self, model_specs: list, processes_per_gpu: int = 10) -> Dict[int, Dict[str, float]]:
        import torch.multiprocessing as mp
        from core.fairness.fairness_worker import (
            device_count_probe_runner,
            fairness_queue_runner,
        )

        metric_config = next((m for m in self.train_params.get('metrics', [])
                            if m['name'] == 'FairnessMetric'), None)
        
        # If no config or no metrics, return zeros
        if not metric_config or not self.fairness_metric_names:
            return {i: {name: 0.0 for name in self.fairness_metric_names} for i in range(len(model_specs))}
        
        fairness_params = metric_config.get('params', {}) or {}
        # Fairness evaluation follows the run's precision policy (Area 4);
        # a metric-level 'precision' in the config may still override it.
        fairness_params.setdefault('precision', self.train_params.get('precision', 'fp32'))
        input_shape = self.train_params.get('input_shape', (3, 224, 224))
        fairness_params['img_size'] = input_shape[2]

        # Probe GPU count
        ctx = mp.get_context("spawn")
        probe_q = ctx.Queue()
        probe_p = ctx.Process(target=device_count_probe_runner, args=(probe_q,))
        probe_p.start()
        num_gpus = probe_q.get()
        probe_p.join()

        indices = [i for i, spec in enumerate(model_specs) if spec]
        if not indices:
            return {i: {name: 0.0 for name in self.fairness_metric_names} for i in range(len(model_specs))}

        slots = max(0, num_gpus * max(1, int(processes_per_gpu)))
        if slots == 0:
            merged = {}
            for i in range(len(model_specs)):
                merged.setdefault(i, {name: 0.0 for name in self.fairness_metric_names})
            return merged

        shards = [[] for _ in range(slots)]
        for k, idx in enumerate(indices):
            shards[k % slots].append((idx, model_specs[idx]))

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
                    self.fairness_metric_names, # PASSING LIST HERE
                    fairness_params,
                    dev_idx,
                ),
            )
            p.start()
            procs.append(p)

        merged: Dict[int, Dict[str, float]] = {}
        for _ in range(len(procs)):
            part = res_q.get()
            merged.update(part)

        for p in procs:
            p.join()

        # Fill any missing entries
        for i in range(len(model_specs)):
            merged.setdefault(i, {name: 0.0 for name in self.fairness_metric_names})

        return merged