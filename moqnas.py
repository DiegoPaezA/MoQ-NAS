""" 
MoQNAS: Multi-Objective Quantum-Inspired Neural Architecture Search
Based on the refactored QNAS (qnas2.py). 

This implements NSGA-II style Pareto selection over multiple objectives, 
while still using the quantum-population machinery from QNAS to evolve 
hyperparameter distributions. 
"""
import os
import time
import json
import pickle
import numpy as np
from qnas2 import QNAS
from pymoo.indicators.hv import Hypervolume
from util import calculate_time, delete_old_dirs_v2


class MOQNAS(QNAS):
    """Multi-Objective QNAS using NSGA-II style selection.

    This class extends the single-objective QNAS algorithm to handle multiple
    competing objectives. It replaces the simple elitist selection with Pareto
    dominance ranking and crowding distance to maintain a diverse set of
    optimal trade-off solutions (the Pareto front).
    """

    def __init__(self, eval_func, experiment_path, objectives, log_file, log_level, data_file):
        """Initializes the MOQNAS algorithm.

        Args:
            eval_func (callable): A function that evaluates a list of individuals.
                It must return a dictionary mapping each individual's 'candidate_id'
                to its performance metrics dict (e.g., {'accuracy': 0.9, 'latency': 100}).
            experiment_path (str): The root directory for saving all experiment artifacts.
            objectives (list[str]): A list of objective names to optimize.
            log_file (str): The path to the log file.
            log_level (str): Logging verbosity level ("INFO", "DEBUG", or "NONE").
            data_file (str): Path to the .pkl file for saving and resuming evolution.
        """
        super().__init__(eval_func, experiment_path, objectives, log_file, log_level, data_file)
        self.num_objectives = len(objectives)
        self.data_file = data_file
        self.pop_size = None
        self.max_generations = None

    def initialize_moqnas(self,
                        # Core EA Parameters
                        num_quantum_ind, repetition, max_generations, update_quantum_gen,
                        update_quantum_rate,
                        # Hyperparameter Population
                        params_ranges, crossover_rate,
                        # Network Population
                        fn_list, initial_probs, max_num_nodes, reducing_fns_list,
                        # Network Crossover
                        en_pop_crossover=False, pop_crossover_method="hux",
                        pop_crossover_rate=0.25, crossover_frequency=5,
                        # Elite Selection & MOEA/D
                        elite_mode="moead_topk", k_elites=5, pool_factor=2,
                        ema_beta=0.7, rank_weighting=True, ref_dir_method="das-dennis",
                        # Network Architecture Rules
                        terminal_op_name="no_op", pool_op_name="pool", min_active_len=5,
                        truncate_after_noop=True, avoid_consecutive_pool=True,
                        # No-Op Probability Management
                        enforce_noop_in_update=True, noop_max_prob=0.90, noop_ramp_cap=True,
                        # Stopping & Penalties
                        early_stopping=True, patience=10, penalize_number=0,
                        # Misc
                        save_data_freq=1, replace_method="best"):
        """Configures the MOQNAS populations and evolutionary hyperparameters.

        Args:
            num_quantum_ind (int): Number of quantum individuals.
            repetition (int): Number of classical individuals per quantum one.
            max_generations (int): Total number of generations to run.
            update_quantum_gen (int): Frequency (in generations) for quantum updates.
            update_quantum_rate (float): Base learning rate for quantum updates.
            params_ranges (dict): Search space for hyperparameters.
            crossover_rate (float): Crossover rate for hyperparameter chromosomes.
            fn_list (list): List of all possible operation names for network nodes.
            initial_probs (list): Initial probabilities for each operation.
            max_num_nodes (int): The maximum length of a network chromosome.
            reducing_fns_list (list): List of operation names that are penalized.
            en_pop_crossover (bool, optional): If True, enables network crossover.
                Defaults to False.
            pop_crossover_method (str, optional): Crossover type for networks
                ("hux" or "uniform"). Defaults to "hux".
            pop_crossover_rate (float, optional): Fraction of the population to
                be replaced by crossover offspring. Defaults to 0.25.
            crossover_frequency (int, optional): Apply network crossover every N
                generations. Defaults to 5.
            elite_mode (str, optional): Strategy for building quantum update targets.
                Defaults to "moead_topk".
            k_elites (int, optional): Number of elite individuals to use. Defaults to 5.
            pool_factor (int, optional): Multiplier for elite pool size. Defaults to 2.
            ema_beta (float, optional): EMA smoothing factor for global elite
                distributions. Set to 0.0 to disable. Defaults to 0.7.
            rank_weighting (bool, optional): If True, weight elite contributions by
                inverse rank. Defaults to True.
            ref_dir_method (str, optional): Method for generating reference vectors
                for MOEA/D-based updates ('das-dennis' or 'dirichlet'). Defaults to "das-dennis".
            terminal_op_name (str, optional): Name of the terminal operation.
                Defaults to "no_op".
            pool_op_name (str | list, optional): Name or pattern(s) to identify
                pooling layers. Defaults to "pool".
            min_active_len (int, optional): Minimum network length before a
                terminal op is allowed. Defaults to 5.
            truncate_after_noop (bool, optional): If True, forces all subsequent
                nodes to be terminal ops after the first one appears. Defaults to True.
            avoid_consecutive_pool (bool, optional): If True, prevents sampling
                two pooling layers in a row. Defaults to True.
            enforce_noop_in_update (bool, optional): If True, applies architecture
                rules during the quantum update. Defaults to True.
            noop_max_prob (float, optional): The maximum probability for a terminal op.
                Defaults to 0.90.
            noop_ramp_cap (bool, optional): If True, linearly increases the
                terminal op's probability cap over the chromosome length.
                Defaults to True.
            early_stopping (bool, optional): If True, enables early stopping.
                Defaults to True.
            patience (int, optional): Generations to wait for improvement before
                stopping. Defaults to 10.
            penalize_number (int, optional): Max allowed reducing layers before penalty.
                Defaults to 0.
            save_data_freq (int, optional): Save best model stats every N generations.
                Defaults to 1.
            replace_method (str, optional): Inherited from QNAS, not used in MOQNAS.
                Defaults to "best".
        """
        self.pop_size = num_quantum_ind * repetition
        self.max_generations = max_generations
        self.hyperparam_crossover_rate = crossover_rate

        # Initialize the base QNAS class, which sets up the quantum populations
        super().initialize_qnas(
            num_quantum_ind=num_quantum_ind,
            params_ranges=params_ranges,
            repetition=repetition,
            max_generations=max_generations,
            crossover_rate=None,  # Not used in MOQNAS's main loop
            update_quantum_gen=update_quantum_gen,
            replace_method=replace_method,
            fn_list=fn_list,
            initial_probs=initial_probs,
            update_quantum_rate=update_quantum_rate,
            max_num_nodes=max_num_nodes,
            reducing_fns_list=reducing_fns_list,
            patience=patience,
            early_stopping=early_stopping,
            save_data_freq=save_data_freq,
            penalize_number=penalize_number,
            crossover_frequency=crossover_frequency,
            en_pop_crossover=en_pop_crossover,
            pop_crossover_rate=pop_crossover_rate,
            pop_crossover_method=pop_crossover_method,
            elite_mode=elite_mode,
            k_elites=k_elites,
            pool_factor=pool_factor,
            ema_beta=ema_beta,
            rank_weighting=rank_weighting,
            # Pass the new network architecture parameters to the QPopulationNetwork
            terminal_op_name=terminal_op_name,
            pool_op_name=pool_op_name,
            min_active_len=min_active_len,
            truncate_after_noop=truncate_after_noop,
            avoid_consecutive_pool=avoid_consecutive_pool,
            enforce_noop_in_update=enforce_noop_in_update,
            noop_max_prob=noop_max_prob,
            noop_ramp_cap=noop_ramp_cap,
        )

        self.classical_nets = self.qpop_net.generate_classical()
        self.classical_params = self.qpop_params.generate_classical()
        self.fits = None
        self.raw_fits = None
        self.pareto_global_population = None
        self.pareto_global_fitnesses = None
        self.pareto_global_params = None
        self.pareto_global_ids = []
        self.fronts_history = {}

        try:
            with open("config_objectives/cfg_obj.json", "r") as f:
                self.objectives_info = json.load(f)["objectives"]
        except Exception as e:
            self.logger.error(f"Could not load objectives config: {e}")
            self.objectives_info = {}

        objective_names = []
        objective_senses = []
        for active_obj in self.objectives:
            for key, info in self.objectives_info.items():
                if key in active_obj:
                    objective_names.append(key)
                    sense = 'max' if info['goal'] == 'maximize' else 'min'
                    objective_senses.append(sense)
                    break

        # Set up reference directions for MOEA/D-based quantum updates
        self.qpop_net.ref_dir_method = ref_dir_method
        self.qpop_net.set_objective_directions(names=objective_names, sense=objective_senses)
        self.logger.info(f"Set objective directions: {list(zip(objective_names, objective_senses))}")

    def multiobjective_fitness(self) -> np.ndarray:
        """Evaluates the current population across all objectives.

        This method decodes the classical chromosomes, calls the evaluation function,
        and organizes the results into a fitness matrix. It also applies penalties
        to the primary objective if configured.

        Returns:
            np.ndarray: A fitness matrix of shape (pop_size, num_objectives)
                        containing penalized objective values.
        """
        decoded_params, decoded_nets = self.decode_pop(
            self.classical_params, self.classical_nets
        )

        raw_results = self.eval_func(decoded_params, decoded_nets, generation=self.current_gen)
        
        N = len(decoded_nets)
        fits = np.zeros((N, self.num_objectives), dtype=float)
        raw_fits = np.zeros((N, self.num_objectives), dtype=float)

        for i in range(N):
            metrics = raw_results[i]
            for j, obj_name in enumerate(self.objectives[:self.num_objectives]):
                val = metrics.get(obj_name, 0.0)
                raw_fits[i, j] = val
                fits[i, j] = val
        
        if self.penalize_number and self.reducing_fns_list:
            penalties = self.get_penalties(self.classical_nets)
            fits[:, 0] -= penalties

        self.raw_fits = raw_fits
        self.fits = fits
        return fits

    @staticmethod
    def compute_hypervolume_mixed(front_raw: np.ndarray, ref_point=None) -> float:
        """Computes the hypervolume of a Pareto front.

        This method handles mixed optimization goals (max/min) by converting
        all objectives to a minimization problem before calculation. The first
        objective is assumed to be maximization, others minimization.

        Args:
            front_raw (np.ndarray): The raw fitness values of the Pareto front.
            ref_point (np.ndarray, optional): A reference point for the calculation.
                If None, a point is inferred from the front. Defaults to None.

        Returns:
            float: The calculated hypervolume.
        """
        if front_raw is None or len(front_raw) == 0:
            return 0.0

        f = np.array(front_raw, dtype=float, copy=True)
        f[:, 0] = -f[:, 0]  # Flip first objective (e.g., accuracy) to minimization

        if ref_point is None:
            rp = np.max(f, axis=0) + 1e-6
        else:
            rp = np.asarray(ref_point, dtype=float)

        return float(Hypervolume(ref_point=rp).do(f))
    
    @staticmethod
    def dominates(a, b):
        """Checks if solution 'a' Pareto-dominates solution 'b'.

        Assumes the first objective is to be maximized and all others minimized.

        Args:
            a (np.ndarray): Fitness vector of solution 'a'.
            b (np.ndarray): Fitness vector of solution 'b'.

        Returns:
            bool: True if 'a' dominates 'b', False otherwise.
        """
        obj_a = np.array([-a[0]] + list(a[1:]))
        obj_b = np.array([-b[0]] + list(b[1:]))
        return np.all(obj_a <= obj_b) and np.any(obj_a < obj_b)

    def fast_nondominated_sort(self, fitnesses: np.ndarray) -> list[list[int]]:
        """Performs the fast non-dominated sort algorithm from NSGA-II.

        It ranks all individuals in a population into a set of Pareto fronts.

        Args:
            fitnesses (np.ndarray): The fitness matrix of the entire population.

        Returns:
            list[list[int]]: A list of fronts, where each front is a list of
                            individual indices. `fronts[0]` is the best front.
        """
        N = fitnesses.shape[0]
        dominated_sets = [set() for _ in range(N)]
        dom_count = np.zeros(N, dtype=int)
        fronts = [[]]

        for p in range(N):
            for q in range(N):
                if self.dominates(fitnesses[p], fitnesses[q]):
                    dominated_sets[p].add(q)
                elif self.dominates(fitnesses[q], fitnesses[p]):
                    dom_count[p] += 1
            if dom_count[p] == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated_sets[p]:
                    dom_count[q] -= 1
                    if dom_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]

    def crowding_distance(self, fits: np.ndarray, front: list[int]) -> np.ndarray:
        """Calculates the crowding distance for each individual in a front.

        This metric is used as a tie-breaker to promote diversity among solutions
        with the same Pareto rank.

        Args:
            fits (np.ndarray): The fitness matrix of the entire population.
            front (list[int]): A list of indices for individuals in one front.

        Returns:
            np.ndarray: An array of crowding distance values for each individual in the front.
        """
        f = fits[front]
        F, M = f.shape
        dist = np.zeros(F)
        if F <= 2:
            return np.full(F, np.inf)
        
        sorted_idx = np.argsort(f, axis=0)
        dist[sorted_idx[0, :]] = np.inf
        dist[sorted_idx[-1, :]] = np.inf
        
        min_vals = f[sorted_idx[0, :], np.arange(M)]
        max_vals = f[sorted_idx[-1, :], np.arange(M)]
        denom = max_vals - min_vals

        for j in range(M):
            if denom[j] > 1e-9:
                prev = f[sorted_idx[:-2, j], j]
                nxt = f[sorted_idx[2:, j], j]
                dist[sorted_idx[1:-1, j]] += (nxt - prev) / denom[j]
        return dist

    def environmental_selection(self, pop: np.ndarray, fits: np.ndarray) -> tuple:
        """Selects the next generation's population using NSGA-II rules.

        It combines parents and offspring, ranks them using non-dominated sorting,
        and fills the new population front by front. Crowding distance is used
        to select individuals from the last front that can fit.

        Args:
            pop (np.ndarray): The combined population's network chromosomes.
            fits (np.ndarray): The combined population's fitness matrix.

        Returns:
            tuple: A tuple containing the selected population's networks, fitnesses,
                and their original indices from the combined input.
        """
        pop_size = self.pop_size
        fronts = self.fast_nondominated_sort(fits)

        selected_idx = []
        count = 0

        for front in fronts:
            if count >= pop_size:
                break
            
            if count + len(front) <= pop_size:
                selected_idx.extend(front)
                count += len(front)
            else:
                rem = pop_size - count
                if rem > 0:
                    cd = self.crowding_distance(fits, front)
                    top_indices = np.argsort(cd)[-rem:]
                    chosen = [front[i] for i in top_indices]
                    selected_idx.extend(chosen)
                break
        
        selected_idx = np.array(selected_idx, dtype=int)
        return pop[selected_idx], fits[selected_idx], selected_idx

    def random_crossover_hyperparams(self, new_pop: np.ndarray) -> np.ndarray:
        """Applies uniform crossover to hyperparameter chromosomes.

        Each individual in the new population has a chance to be crossed over with
        a randomly selected parent from the previous generation.

        Args:
            new_pop (np.ndarray): The newly generated hyperparameter population.

        Returns:
            np.ndarray: The hyperparameter population after crossover.
        """
        old_pop = self.qpop_params.current_pop
        if old_pop is None or old_pop.shape[0] == 0:
            return new_pop

        mixed = new_pop.copy()
        for i in range(new_pop.shape[0]):
            if np.random.rand() < self.hyperparam_crossover_rate:
                p_idx = np.random.randint(old_pop.shape[0])
                parent = old_pop[p_idx]
                child = new_pop[i]
                mask = np.random.rand(*child.shape) < 0.5
                mixed[i] = np.where(mask, parent, child)
        return mixed

    def record_and_save_history(self):
        """Records the current global Pareto front and its hypervolume, saving to disk."""
        gen_record = {1: []}
        
        for i in range(len(self.pareto_global_ids)):
            individual_data = {
                "id": self.pareto_global_ids[i],
                "accuracy": float(self.pareto_global_fitnesses[i][0]),
                "params": float(self.pareto_global_fitnesses[i][1]),
                "inference_time": float(self.pareto_global_fitnesses[i][2])
            }
            gen_record[1].append(individual_data)

        hv = self.compute_hypervolume_mixed(self.pareto_global_fitnesses)
        gen_record["hypervolume"] = float(hv)
        
        self.fronts_history[self.current_gen] = gen_record
        
        history_path = os.path.join(self.experiment_path, "pareto_history.pkl")
        with open(history_path, "wb") as f:
            pickle.dump(self.fronts_history, f)

    def update_global_pareto_front(self):
        """Updates the global Pareto front archive.

        It combines the current population with the existing archive, removes
        duplicates, and identifies the new non-dominated set.
        """
        curr_ids = self.classical_ids

        if self.pareto_global_population is None:
            all_pop = self.classical_nets.copy()
            all_fits = self.fits.copy()
            all_params = self.classical_params.copy()
            all_ids = curr_ids.copy()
        else:
            all_pop = np.vstack([self.pareto_global_population, self.classical_nets])
            all_fits = np.vstack([self.pareto_global_fitnesses, self.fits])
            all_params = np.vstack([self.pareto_global_params, self.classical_params])
            all_ids = self.pareto_global_ids + curr_ids

        unique_ids, unique_indices = np.unique(all_ids, return_index=True)
        unique_pop = all_pop[unique_indices]
        unique_fits = all_fits[unique_indices]
        unique_params = all_params[unique_indices]
        unique_ids = list(unique_ids)
        
        fronts = self.fast_nondominated_sort(unique_fits)
        idx0 = fronts[0]
        
        self.pareto_global_population = unique_pop[idx0]
        self.pareto_global_fitnesses = unique_fits[idx0]
        self.pareto_global_params = unique_params[idx0]
        self.pareto_global_ids = [unique_ids[i] for i in idx0]
        
        self._last_cd = self.crowding_distance(self.pareto_global_fitnesses,
            list(range(len(self.pareto_global_fitnesses))))
    
    def go_next_gen(self):
        """Orchestrates all end-of-generation tasks.

        This includes updating the global archive, recording history, updating
        the quantum populations based on the archive, logging, and cleanup.
        """
        self.update_global_pareto_front()
        self.record_and_save_history()

        cd = self._last_cd
        sorted_rel = np.argsort(cd)[::-1]
        
        # The best individuals from the global front guide the quantum update
        self.qpop_net.current_pop = self.pareto_global_population[sorted_rel]
        self.qpop_net.current_pop_objs = self.pareto_global_fitnesses[sorted_rel]
        if self.pareto_global_params is not None:
            self.qpop_params.current_pop = self.pareto_global_params[sorted_rel]

        self.update_quantum(self.current_gen)

        hv = self.fronts_history[self.current_gen]['hypervolume']
        self.logger.info(
            "Generation %d: updated global front with %d individuals (HV: %.2f)",
            self.current_gen, len(self.pareto_global_population), hv
        )
        
        is_snapshot = (self.current_gen % 5 == 0) and (self.current_gen > 0)
        delete_old_dirs_v2(self.experiment_path, self.current_gen, 
                        keep_ids=self.pareto_global_ids.copy(), is_snapshot_gen=is_snapshot)
        if self.current_gen == 1:
            delete_old_dirs_v2(self.experiment_path, 0, keep_ids=self.pareto_global_ids.copy())

        self.save_data()
        self.current_gen += 1

    def evolve(self) -> tuple[np.ndarray, np.ndarray]:
        """Runs the main multi-objective evolutionary loop.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the final Pareto
                front's network chromosomes and their corresponding fitness values.
        """
        start_time = time.time()
        
        # --- Generation 0 ---
        p0_params, p0_nets = self.generate_classical()
        self.classical_params = p0_params
        self.classical_nets = p0_nets
        self.qpop_params.current_pop = p0_params
        self.qpop_net.current_pop = p0_nets
        
        f0 = self.multiobjective_fitness()
        self.logger.info("Generation 0: fitnesses:\n%s", f0)
        
        p0_ids = [f"0_{i}" for i in range(len(p0_nets))]
        self.classical_ids = p0_ids
        self.fits = f0
        
        # --- Main Loop (Generations 1 to N) ---
        for gen in range(1, self.max_generations + 1):
            self.current_gen = gen

            children_params, children_nets = self.generate_classical()
            child_ids = [f"{self.current_gen}_{i}" for i in range(len(children_nets))]

            children_params = self.random_crossover_hyperparams(children_params)
            children_nets = self.crossover_network(children_nets)
            
            self.classical_params = children_params
            self.classical_nets = children_nets
            child_fits = self.multiobjective_fitness()
            child_raw = self.raw_fits.copy()

            combined_nets = np.vstack([p0_nets, children_nets])
            combined_fits = np.vstack([f0, child_fits])
            combined_raws = np.vstack([p0_raws, child_raw])
            combined_ids = p0_ids + child_ids
            combined_params = np.vstack([p0_params, children_params])
            
            next_nets, next_fits, survivor_idx = self.environmental_selection(combined_nets, combined_fits)
            
            self.classical_nets = next_nets
            self.fits = next_fits
            self.raw_fits = combined_raws[survivor_idx]
            self.classical_ids = [combined_ids[i] for i in survivor_idx]
            self.classical_params = combined_params[survivor_idx]

            self.go_next_gen()

            if self.early_stopping and self.check_early_stopping():
                break

            p0_params, p0_nets, f0, p0_raws, p0_ids = (
                self.classical_params, self.classical_nets, self.fits, 
                self.raw_fits, self.classical_ids
            )
            
            if self.current_gen > 0 and (self.current_gen % 5 == 0):
                h, m, est_h, est_m = calculate_time(
                    start_time, time.time(), self.current_gen, self.max_generations, end_evol=False)
                self.logger.info("Gen %d: elapsed %dh %dm; ETA %dh %dm", 
                                self.current_gen, h, m, est_h, est_m)

        total_h, total_m = calculate_time(start_time, time.time())
        self.logger.info("Total evolution time: %d hours and %d minutes", total_h, total_m)

        return self.pareto_global_population, self.pareto_global_fitnesses