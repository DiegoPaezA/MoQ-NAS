""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Refactored Q-NAS algorithm class, with modular replace & crossover.
        Diego Páez Ardila - 2025
"""

import datetime
import os
from pickle import dump, HIGHEST_PROTOCOL

import numpy as np
import time
from collections import defaultdict

from population import QPopulationNetwork, QPopulationParams
from util import (
    delete_old_dirs_v2,
    init_log,
    load_pkl,
    calculate_time,
    backup_cache,
    load_cache,
)


class QNAS(object):
    """ Quantum-Inspired Neural Architecture Search (refactored) """

    def __init__(self, eval_func, experiment_path, objectives, log_file, log_level, data_file):
        """
        Initialize the core QNAS object.

        Args:
            eval_func (callable): Function that evaluates a list of
                (param_dict, net_struct) pairs and returns a NumPy array of raw
                fitness values (shape=(n,)).
            experiment_path (str): Path to directory for logs, caches, etc.
            objectives (list): List of objectives to optimize, e.g. ["accuracy", "latency"].
            log_file (str): Path to file where logs will be written.
            log_level (str): "INFO", "DEBUG", or "NONE" for logging verbosity.
            data_file (str): Path to a .pkl file used to store evolution data
                across generations.
        """
        # --- Basic settings & bookkeeping ---
        self.dtype = np.float64
        self.tolerance = 1e-15

        self.best_so_far = 0.0
        self.best_so_far_id = [0, 0]
        self.current_best_id = [0, 0]
        self.current_gen = 0

        self.data_file = data_file
        self.eval_func = eval_func
        self.experiment_path = experiment_path
        self.objectives = objectives

        self.logger = init_log(log_level, name=__name__, file_path=log_file)

        # These will be set in initialize_qnas(...)
        self.fitnesses = None         # 1D np.array of current-gen fitnesses
        self.raw_fitnesses = None     # 1D np.array of unpenalized fitnesses
        self.generations = None       # max number of generations
        self.update_quantum_gen = None
        self.replace_method = None    # "elitism" or "best"
        self.penalize_number = None
        self.reducing_fns_list = []   # indices of "reducing" layer IDs
        self.penalties = None

        self.random = 0.0             # random seed for this gen
        self.total_eval = 0           # total number of evaluations so far

        self.early_stopping = None
        self.patience = None
        self.early_stopping_counter = 0
        self.last_best_so_far = 0.0
        self.since_last_mutation = 0  # Counter for mutation frequency

        # population crossover settings
        self.en_pop_crossover = None
        self.pop_crossover_rate = None
        self.crossover_frequency = None

        # frequency for saving train loss/acc of best model in CSV
        self.save_data_freq = np.inf

        # Quantum‐population objects (set in initialize_qnas)
        self.qpop_params = None       # type: QPopulationParams
        self.qpop_net = None          # type: QPopulationNetwork

        # Caching of evaluated “genome → fitness” to avoid repeats
        cache_file = os.path.join(self.experiment_path, "cache_backup.pkl")
        self.evaluated = load_cache(cache_file)          # {tuple(genome): float}
        self.eval_history = defaultdict(list)  # {tuple(genome): [raw1, raw2, raw3]}


    def initialize_qnas(self,num_quantum_ind,params_ranges,repetition,max_generations,
                        crossover_rate,update_quantum_gen,replace_method,fn_list,
                        initial_probs,update_quantum_rate,max_num_nodes,reducing_fns_list,
                        patience,early_stopping,save_data_freq=0,penalize_number=0,
                        crossover_frequency=5,en_pop_crossover=False,
                        pop_crossover_rate=0.25,pop_crossover_method="hux",):
        """
        Initialize all QNAS-specific populations and hyperparameters.

        Args:
            num_quantum_ind (int): Number of quantum individuals (Q).
            params_ranges (dict): {param_name: [lower, upper]} ranges for hyperparameters.
            repetition (int): Number of classical individuals per quantum individual.
            max_generations (int): Total number of generations (G).
            crossover_rate (float): Crossover rate for continuous/hyperparameter genes.
            update_quantum_gen (int): Quantum PMFs update frequency (every K generations).
            replace_method (str): "elitism" or "best" selection scheme.
            fn_list (list): List of possible function (layer-ID) names for network chromosome.
            initial_probs (list): Initial probabilities for each function in fn_list.
            update_quantum_rate (float): “Intensity” of quantum update at each step.
            max_num_nodes (int): Maximum number of nodes in the network chromosome.
            reducing_fns_list (list): List of “reducing” function names or indices that incur penalties.
            patience (int): Number of generations without improvement before early stopping.
            early_stopping (bool): Whether to enable early stopping.
            save_data_freq (int, optional): Save best-model stats to CSV every this many generations.
            penalize_number (int, optional): Max allowed “reducing” layers before penalty applies.
            crossover_frequency (int, optional): Apply population-crossover every this many generations.
            en_pop_crossover (bool, optional): Whether to enable population crossover (network side).
            pop_crossover_rate (float, optional): Fraction of new offspring to mix by crossover with parents.
            pop_crossover_method (str, optional): “hux” or “uniform” crossover for network-chromosome.
        """
        # 1) Evolution settings
        self.generations = max_generations
        self.update_quantum_gen = update_quantum_gen
        self.replace_method = replace_method
        self.penalize_number = penalize_number
        self.patience = patience
        self.early_stopping = early_stopping

        # 2) Population crossover settings
        self.en_pop_crossover = en_pop_crossover
        self.pop_crossover_rate = pop_crossover_rate
        self.crossover_frequency = crossover_frequency

        # 3) Reducing‐layer penalty setup
        if reducing_fns_list:
            # We assume fn_list is a list of function names; collect their indices
            self.reducing_fns_list = [
                i for i, name in enumerate(fn_list) if name in reducing_fns_list
            ]
            self.penalties = np.zeros(shape=(num_quantum_ind * repetition,))
        else:
            self.reducing_fns_list = []
            self.penalties = None

        # 4) CSV‐save frequency
        if save_data_freq:
            self.save_data_freq = save_data_freq

        # 5) Build quantum‐population objects
        self.qpop_params = QPopulationParams(
            num_quantum_ind=num_quantum_ind,
            params_ranges=params_ranges,
            repetition=repetition,
            crossover_rate=crossover_rate,
            update_quantum_rate=update_quantum_rate,
        )

        self.qpop_net = QPopulationNetwork(
            num_quantum_ind=num_quantum_ind,
            max_num_nodes=max_num_nodes,
            repetition=repetition,
            update_quantum_rate=update_quantum_rate,
            fn_list=fn_list,
            initial_probs=initial_probs,
            crossover_method=pop_crossover_method,
        )

    def select_population(
        self,
        old_params: np.ndarray, old_nets: np.ndarray, old_pen: np.ndarray,
        old_raw: np.ndarray, new_params: np.ndarray, new_nets: np.ndarray,
        new_pen: np.ndarray, new_raw: np.ndarray,):
        """
        Merge old and new classical populations (using penalized fitness for ordering),
        then keep the top num_classic = (num_quantum_ind × repetition) individuals.
        Returns both penalized and raw fitness for the survivors.

        Args:
            old_params (np.ndarray): (N_old, param_dim) array of old hyperparameter chromosomes.
            old_nets (np.ndarray): (N_old, net_dim) array of old network chromosomes.
            old_pen (np.ndarray): (N_old,) penalized fitness of old individuals.
            old_raw (np.ndarray): (N_old,) raw (unpenalized) fitness of old individuals.
            new_params (np.ndarray): (N_new, param_dim) array of new hyperparameter chromosomes.
            new_nets (np.ndarray): (N_new, net_dim) array of new network chromosomes.
            new_pen (np.ndarray): (N_new,) penalized fitness of new individuals.
            new_raw (np.ndarray): (N_new,) raw (unpenalized) fitness of new individuals.

        Returns:
            sorted_params (np.ndarray): (num_classic, param_dim) hyperparameter chromosomes of survivors.
            sorted_nets (np.ndarray): (num_classic, net_dim) network chromosomes of survivors.
            sorted_pen (np.ndarray): (num_classic,) penalized fitness of survivors.
            sorted_raw (np.ndarray): (num_classic,) raw fitness of survivors.
        """
        # 1) First generation: simply take new population wholesale (both penalized & raw)
        if self.current_gen == 0:
            return new_params, new_nets, new_pen, new_raw

        self.update_best_id(new_pen)

        if self.replace_method == "elitism":
            selected = [0]
        else:
            selected = list(range(len(old_pen)))

        kept_old_pen = old_pen[selected]
        kept_old_raw = old_raw[selected]
        kept_old_params = old_params[selected]
        kept_old_nets   = old_nets[selected]

        all_pen = np.concatenate([kept_old_pen, new_pen])
        all_raw = np.concatenate([kept_old_raw, new_raw])
        all_params = np.concatenate([kept_old_params, new_params])
        all_nets   = np.concatenate([kept_old_nets, new_nets])

        num_classic = self.qpop_params.num_ind * self.qpop_params.repetition
        sorted_pen, sorted_raw, sorted_params, sorted_nets = self.order_pop(
            all_pen, all_raw, all_params, all_nets, selection=range(num_classic)
        )
        
        return sorted_params, sorted_nets, sorted_pen, sorted_raw

    @staticmethod
    def order_pop(fitnesses: np.ndarray,raw_fitnesses: np.ndarray,
                    pop_params: np.ndarray,pop_net: np.ndarray,selection=None,):
        """
        Sort a population by `fitnesses` in descending order, then pick indices in `selection`.

        Args:
            fitnesses (np.ndarray): (N,) array of fitness values used for sorting (penalized).
            raw_fitnesses (np.ndarray): (N,) array of raw (unpenalized) fitness values.
            pop_params (np.ndarray): (N, param_dim) array of hyperparameter chromosomes.
            pop_net (np.ndarray): (N, net_dim) array of network chromosomes.
            selection (iterable, optional): Indices to keep after sorting.

        Returns:
            sorted_fits (np.ndarray): (len(selection),) penalized fitness after sorting & slicing.
            sorted_raw (np.ndarray): (len(selection),) raw fitness after sorting & slicing.
            sorted_params (np.ndarray): (len(selection), param_dim) hyperparameter chromosomes of survivors.
            sorted_nets (np.ndarray): (len(selection), net_dim) network chromosomes of survivors.
        """
        if selection is None:
            selection = range(fitnesses.shape[0])
        idx = np.argsort(fitnesses)[::-1]  # descending
        sorted_params = pop_params[idx][selection]
        sorted_nets = pop_net[idx][selection]
        sorted_fits = fitnesses[idx][selection]
        sorted_raw = raw_fitnesses[idx][selection]
        return sorted_fits, sorted_raw, sorted_params, sorted_nets


    def update_best_id(self, new_fits: np.ndarray):
        """
        Update the IDs of the best individual based on new_fits.

        Args:
            new_fits (np.ndarray): (N,) penalized fitness array for the new population.
        """
        idx_sorted = np.argsort(new_fits)[::-1]
        self.current_best_id = [self.current_gen, int(idx_sorted[0])]
        if new_fits[idx_sorted[0]] > self.best_so_far:
            self.best_so_far_id = self.current_best_id


    def generate_classical(self):
        """
        Sample a batch of classical individuals from the current quantum PMFs.

        Returns:
            new_pop_params (np.ndarray): (num_classic, param_dim) hyperparameter chromosomes.
            new_pop_net (np.ndarray): (num_classic, net_dim) network chromosomes.
        """
        self.random = np.random.rand()
        new_pop_params = self.qpop_params.generate_classical()
        new_pop_net = self.qpop_net.generate_classical()
        self.logger.info("Generated classical networks:\n%s", new_pop_net)
        return new_pop_params, new_pop_net


    def decode_pop(self, pop_params: np.ndarray, pop_net: np.ndarray):
        """
        Decode classical-encoded populations into Python-readable structures.

        Args:
            pop_params (np.ndarray): (N, param_dim) hyperparameter chromosomes.
            pop_net (np.ndarray): (N, net_dim) network chromosomes.

        Returns:
            decoded_params (list of dict): List of length N, each a dict of hyperparameters.
            decoded_nets (list): List of length N, each an actual network structure.
        """
        num_ind = pop_net.shape[0]
        decoded_params = [None] * num_ind
        decoded_nets = [None] * num_ind
        for i in range(num_ind):
            # QPopulationParams.chromosome.decode returns a dict of hyperparams
            decoded_params[i] = self.qpop_params.chromosome.decode(pop_params[i])
            decoded_params[i]["candidate_id"] = i
            # QPopulationNetwork.chromosome.decode returns an actual network structure
            decoded_nets[i] = self.qpop_net.chromosome.decode(pop_net[i, :])
        return decoded_params, decoded_nets


    def eval_pop(self, pop_params: np.ndarray, pop_net: np.ndarray) -> np.ndarray:
        """
        Decode & evaluate each classical individual, with caching & 3-run averaging.
        Return two arrays: penalized_fits and raw_fits (each shape=(N,)).

        Internally:
            1. Decode param-dicts & network-structures.
            2. For each genome-key (tuple of network chromosome):
                - If in self.evaluated (cached), use cached mean fitness.
                - Else, if <3 runs exist in self.eval_history[key], append to eval batch.
                - If ≥3 runs exist, compute mean, cache, and assign to fitness_list.
            3. Batch-evaluate all in eval batch via self.eval_func.
                - Append each raw fitness to self.eval_history[key].
                - Once it reaches 3 runs, compute mean, cache in self.evaluated, assign to fitness_list.
                - If <3 runs, assign raw fitness to fitness_list (to be averaged later).
            4. Build raw_fits array from fitness_list.
            5. Compute penalized_fits by subtracting penalty for “reducing” layers if needed.

        Args:
            pop_params (np.ndarray): (N, param_dim) hyperparameter chromosomes.
            pop_net (np.ndarray): (N, net_dim) network chromosomes.

        Returns:
            penalized_fits (np.ndarray): (N,) penalized fitness values.
            raw_fits (np.ndarray): (N,) unpenalized (raw) fitness values.
        """
        decoded_params, decoded_nets = self.decode_pop(pop_params, pop_net)
        self.logger.info("Evaluating generation %d (size %d)…",
                        self.current_gen, len(decoded_nets))

        num_ind = pop_net.shape[0]
        fitness_list = [None] * num_ind

        # 1) Determine which need evaluation
        to_eval_idx = []
        to_eval_params = []
        to_eval_nets = []
        to_eval_keys = []

        for idx in range(num_ind):
            key = tuple(pop_net[idx].tolist())
            if key in self.evaluated:
                # Already have cached mean
                fitness_list[idx] = self.evaluated[key]
                continue

            hist = self.eval_history[key]
            if len(hist) < 3:
                to_eval_idx.append(idx)
                to_eval_params.append(decoded_params[idx])
                to_eval_nets.append(decoded_nets[idx])
                to_eval_keys.append(key)
            else:
                # We have >=3 runs cached, so compute mean once & cache
                mean_f = sum(hist) / 3.0
                self.evaluated[key] = mean_f
                fitness_list[idx] = mean_f

        # 2) Batch‐evaluate those that still need runs 
        if to_eval_idx:
            results = self.eval_func(to_eval_params, to_eval_nets,
                                    generation=self.current_gen)
            
            metric_key = self.objectives[0] # for now, just use the first objective
            raw_vals = np.array([
                results[idx][metric_key]
                for idx in range(len(to_eval_nets))
            ])

            for i, idx in enumerate(to_eval_idx):
                key = to_eval_keys[i]
                raw_fitness = float(raw_vals[i])
                self.eval_history[key].append(raw_fitness)
                self.total_eval += 1

                if len(self.eval_history[key]) == 3:
                    # On the 3rd run, compute & cache the mean
                    mean_f = sum(self.eval_history[key]) / 3.0
                    self.evaluated[key] = mean_f
                    fitness_list[idx] = mean_f
                    self.logger.debug(
                        "Key %s now has 3 runs; caching mean = %.4f", key, mean_f
                    )
                else:
                    # <3 runs => use raw for selection, but we'll do more runs later
                    fitness_list[idx] = raw_fitness
                    self.logger.debug(
                        "Key %s: run %d/3, raw fitness = %.4f",
                        key,
                        len(self.eval_history[key]),
                        raw_fitness,
                    )
        # 3) Apply penalty if needed (on “reducing layers”)
        raw_fits = np.array(fitness_list, dtype=float)
        penalized_fits = raw_fits.copy()
        if self.penalize_number and self.reducing_fns_list:
            penalties = self.get_penalties(pop_net)
            penalized_fits -= penalties

        return penalized_fits, raw_fits


    def get_penalties(self, pop_net: np.ndarray, penalty_factor: float = 0.01) -> np.ndarray:
        """
        Compute penalty for each network encoding based on “reducing” genes.

        For each network-encoding chromosome (1D int array), count how many genes
        correspond to a “reducing” function (indices in self.reducing_fns_list). If
        the count exceeds self.penalize_number, subtract penalty_factor × (excess count)
        from its raw fitness.

        Args:
            pop_net (np.ndarray): (N, net_dim) array of network chromosomes.
            penalty_factor (float): Factor to multiply the excess count by.

        Returns:
            penalties (np.ndarray): (N,) array where each entry is the penalty to subtract.
        """
        penalties = np.zeros(shape=(pop_net.shape[0],), dtype=float)
        for i, net in enumerate(pop_net):
            unique, counts = np.unique(net, return_counts=True)
            reducing_count = sum(
                counts[j] for j, u in enumerate(unique) if u in self.reducing_fns_list
            )
            if reducing_count > self.penalize_number:
                penalties[i] = (reducing_count - self.penalize_number)
        return penalty_factor * penalties


    def crossover_hyperparams(self, new_pop_params: np.ndarray) -> np.ndarray:
        """
        Apply classical-style crossover on the hyperparameter chromosomes if not in generation 0.

        Uses QPopulationParams.classic_crossover under the hood, passing `self.random`
        as the “distance” parameter. If QPopulationParams lacks a classic_crossover method,
        does nothing.

        Args:
            new_pop_params (np.ndarray): (N, param_dim) new hyperparameter chromosomes.

        Returns:
            new_pop_params (np.ndarray): Possibly-modified chromosomes after crossover.
        """
        if self.current_gen > 0:
            try:
                new_pop_params = self.qpop_params.classic_crossover(
                    new_pop=new_pop_params,
                    distance=self.random,
                )
            except AttributeError:
                # If QPopulationParams has no classic_crossover, skip
                pass
        return new_pop_params


    def crossover_network(self, new_pop_net: np.ndarray) -> np.ndarray:
        """
        Apply population-level crossover on the network chromosomes if enabled.

        Conditions:
            - self.current_gen > 0
            - self.en_pop_crossover is True
            - self.current_gen % self.crossover_frequency == 0

        Takes the top `num_off = int(N * self.pop_crossover_rate)` parents
        (from self.qpop_net.current_pop[:num_off]) and applies QPopulationNetwork.apply_crossover
        to mix them with the first `num_off` offspring in new_pop_net. If QPopulationNetwork
        lacks apply_crossover, does nothing.

        Args:
            new_pop_net (np.ndarray): (N, net_dim) new network chromosomes.

        Returns:
            new_pop_net (np.ndarray): Possibly-modified network chromosomes after crossover.
        """
        if self.current_gen > 0 and getattr(self, "en_pop_crossover", False):
            if self.current_gen % self.crossover_frequency == 0:
                num_off = int(len(new_pop_net) * self.pop_crossover_rate)
                best_current = self.qpop_net.current_pop[:num_off]
                try:
                    new_pop_net[:num_off] = self.qpop_net.apply_crossover(
                        best_current, new_pop_net[:num_off]
                    )
                except AttributeError:
                    # If QPopulationNetwork has no apply_crossover, skip
                    pass
        return new_pop_net


    def log_data(self):
        """
        Log information about the current generation to the logger, including:
            - generation number
            - best-so-far ID and fitness
            - penalized fitnesses array
            - raw fitnesses array
        """
        np.set_printoptions(precision=4)
        self.logger.info(
            "Generation %d complete!\n"
            "- Best so far: %s → %.5f\n"
            "- Penalized fitnesses: %s\n"
            "- Raw fitnesses: %s\n",
            self.current_gen,
            self.best_so_far_id,
            self.best_so_far,
            self.fitnesses,
            self.raw_fitnesses,
        )


    def save_data(self):
        """
        Save all QNAS-related data into self.data_file as a pickle.
        The saved dict (keyed by generation number) contains:
            - time: current timestamp (str)
            - total_eval: total number of architecture evaluations
            - best_so_far: best penalized fitness so far
            - best_so_far_id: [generation, index] of best-so-far
            - fitnesses: 1D array of penalized fitnesses for current generation
            - raw_fitnesses: 1D array of raw fitnesses for current generation
            - lower: QPopulationParams.lower bounds for hyperparameters
            - upper: QPopulationParams.upper bounds for hyperparameters
            - params_pop: current hyperparameter population (NumPy array)
            - net_probs: QPopulationNetwork.probabilities (quantum PMF state for networks)
            - num_net_nodes: QPopulationNetwork.chromosome.num_genes
            - net_pop: current network-chromosome population (NumPy array)
        """
        if self.current_gen == 0:
            data = {}
        else:
            if os.path.exists(self.data_file):
                data = load_pkl(self.data_file)
            else:
                data = {}

        data[self.current_gen] = {
            "time": str(datetime.datetime.now()),
            "total_eval": self.total_eval,
            "best_so_far": self.best_so_far,
            "best_so_far_id": self.best_so_far_id,
            "fitnesses": self.fitnesses,
            "raw_fitnesses": self.raw_fitnesses,
            "lower": self.qpop_params.lower,
            "upper": self.qpop_params.upper,
            "params_pop": self.qpop_params.current_pop,
            "net_probs": self.qpop_net.probabilities,
            "num_net_nodes": self.qpop_net.chromosome.num_genes,
            "net_pop": self.qpop_net.current_pop,
        }

        self.dump_pkl_data(data)


    def dump_pkl_data(self, new_data: dict):
        """
        Overwrite self.data_file with new_data (dictionary) as a pickle.

        Args:
            new_data (dict): Dictionary to write to self.data_file.
        """
        with open(self.data_file, "wb") as f:
            dump(new_data, f, protocol=HIGHEST_PROTOCOL)


    def load_qnas_data(self, file_path: str):
        """
        Load evolution data from an existing pickle (file_path), and assign all
        relevant fields into this QNAS instance so you can resume training.

        Args:
            file_path (str): Path to a pickle file containing a dict keyed by generation.
        """
        log_data = load_pkl(file_path)
        if not os.path.exists(self.data_file):
            self.dump_pkl_data(log_data)

        generation = max(log_data.keys())
        state = log_data[generation]

        self.current_gen = generation
        self.total_eval = state["total_eval"]
        self.best_so_far = state["best_so_far"]
        self.best_so_far_id = state["best_so_far_id"]
        self.qpop_net.chromosome.set_num_genes(state["num_net_nodes"])
        self.fitnesses = state["fitnesses"]
        self.raw_fitnesses = state["raw_fitnesses"]
        self.qpop_params.lower = state["lower"]
        self.qpop_params.upper = state["upper"]
        self.qpop_net.probabilities = state["net_probs"]
        self.qpop_params.current_pop = state["params_pop"]
        self.qpop_net.current_pop = state["net_pop"]

    def check_early_stopping(self) -> bool:
        """
        Compute early-stopping: if `self.best_so_far` fails to improve by at least 0.005
        over `self.patience` consecutive generations, return True to signal stopping.

        Returns:
            bool: True if early stopping criterion is met, False otherwise.
        """
        if self.current_gen > 1:
            if self.last_best_so_far != 0:
                improvement = (self.best_so_far - self.last_best_so_far) / self.last_best_so_far
            else:
                improvement = 0.0

            if improvement > 0.005:
                self.early_stopping_counter = 0
                # self.since_last_mutation = 0
            else:
                self.early_stopping_counter += 1
                # self.since_last_mutation += 1
                # interval = max(1, self.patience // 4)
                # if self.since_last_mutation >= interval:
                #     self.qpop_net.mutate_probabilities(fraction=0.2, intensity=0.1) # 20% pop, 10% noise
                #     self.since_last_mutation = 0
                #     self.logger.info("Mutating quantum population due to stagnation")

            self.logger.info("Early stopping counter: %d", self.early_stopping_counter)
            if self.early_stopping_counter >= self.patience:
                self.logger.info("Early stopping at generation %d!", self.current_gen)
                return True

        self.last_best_so_far = self.best_so_far
        return False


    def update_quantum(self):
        """
        If `self.current_gen > 0` and is a multiple of `self.update_quantum_gen`,
        call `QPopulationParams.update_quantum(...)` and `QPopulationNetwork.update_quantum(...)`
        using `self.random` as the intensity parameter.
        """
        if self.current_gen > 0 and (self.current_gen % self.update_quantum_gen == 0):
            self.qpop_params.update_quantum(intensity=self.random)
            self.qpop_net.update_quantum(intensity=self.random)


    def go_next_gen(self):
        """
        Advance to the next generation:
            1) Perform quantum PMF update
            2) Backup evaluation cache
            3) Save current state to pickle
            4) Log current generation data
            5) Delete model directories for earlier gens, keeping only best ID
            6) Increment `self.current_gen`
        """
        self.update_quantum()
        backup_cache(self.evaluated, file_path=self.experiment_path)
        self.save_data()
        self.log_data()

        best_gen, best_idx = self.best_so_far_id
        best_id = f"{best_gen}_{best_idx}"
        delete_old_dirs_v2(self.experiment_path, self.current_gen, keep_ids=[best_id])
        self.current_gen += 1


    def evolve(self):
        """
        Main evolution loop (modular). Steps per generation:

            (1) If gen == 0:
                    - generate_classical() → (p0, n0)
                    - f0_pen, f0_raw = eval_pop(p0, n0)
                    - assign to self.qpop_* and self.fitnesses/self.raw_fitnesses
            (2) While self.current_gen < max_gen:
                    a) generate_classical() → (new_p, new_n)
                    b) new_p = crossover_hyperparams(new_p)
                    c) new_n = crossover_network(new_n)
                    d) new_f_pen, new_f_raw = eval_pop(new_p, new_n)
                    e) next_p, next_n, next_pen, next_raw = select_population(
                        old_p, old_n, old_pen, old_raw, new_p, new_n, new_f_pen, new_f_raw)
                    f) assign next_p, next_n, next_pen, next_raw to self.qpop_*, self.fitnesses, self.raw_fitnesses
                    g) go_next_gen()  # quantum update + logging + cleanup + increment gen
                    h) if early stopping criterion met, break
            (3) Final logging of total evolution time
        """
        start_time = time.time()
        max_gen = self.generations
        if self.current_gen > 0:
            # Resuming from a prior run: shift max_gen accordingly
            max_gen += (self.current_gen + 1)
            self.current_gen += 1

        # (2) Main loop
        while self.current_gen < max_gen:
            # (2a) Every 5 generations, log ETA
            if self.current_gen > 0 and (self.current_gen % 5 == 0):
                curr_time = time.time()
                h, m, est_h, est_m = calculate_time(
                    start_time, curr_time, self.current_gen, max_gen, end_evol=False
                )
                self.logger.info(
                    "Gen %d: elapsed %dh %dm; ETA %dh %dm",
                    self.current_gen, h, m, est_h, est_m,
                )

            # (2b) Generate classical offspring
            new_p, new_n = self.generate_classical()

            # (2c) Hyperparameter crossover
            new_p = self.crossover_hyperparams(new_p)

            # (2d) Network‐structure crossover
            new_n = self.crossover_network(new_n)

            # (2e) Evaluate new offspring
            new_f_pen, new_f_raw = self.eval_pop(new_p, new_n)

            # (2f) Merge & select next generation
            old_p = self.qpop_params.current_pop
            old_n = self.qpop_net.current_pop
            old_pen = self.fitnesses
            old_raw = self.raw_fitnesses

            next_p, next_n, next_pen, next_raw = self.select_population(
                old_p, old_n, old_pen, old_raw,
                new_p, new_n, new_f_pen, new_f_raw
            )

            # (2g) Assign back
            self.qpop_params.current_pop = next_p
            self.qpop_net.current_pop = next_n
            self.fitnesses               = next_pen
            self.raw_fitnesses           = next_raw
            self.best_so_far             = next_pen[0]

            # (2h) Quantum update + logging + cleanup + increment gen
            self.go_next_gen()

            # (2i) Early stopping?
            if self.early_stopping and self.check_early_stopping():
                break

        total_h, total_m = calculate_time(start_time, time.time())
        self.logger.info("Total evolution time: %d hours and %d minutes", total_h, total_m)