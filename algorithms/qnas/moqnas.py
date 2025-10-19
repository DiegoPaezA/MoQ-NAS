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
from pymoo.indicators.hv import Hypervolume

from .qnas2 import QNAS
from .qhistory import QHistory
from utils.helpers import calculate_time, delete_old_dirs_v2


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
        self.history = QHistory(self.experiment_path)

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
        
        # load config objective sense
        try:
            with open("configs/cfg_obj.json", "r") as f:
                self.objectives_info = json.load(f)["objectives"]
        except Exception as e:
            raise RuntimeError(f"Failed to load objective config: {e}")

        self.objective_names = []
        self.objective_senses = []
        # Ensure all specified objectives have corresponding info
        # Map the active objectives to their information
        for active_obj in self.objectives:
            match_found = False
            for key, info in self.objectives_info.items():
                if key in active_obj:
                    self.objective_names.append(key)
                    sense = 'max' if info['goal'] == 'maximize' else 'min'
                    self.objective_senses.append(sense)
                    match_found = True
                    break # Move to the next active_obj
            if not match_found:
                print(f"Warning: Could not find a rule for '{active_obj}'")
        
        # moead_topk
        self.qpop_net.ref_dir_method = ref_dir_method
        print(f"Reference direction method set to: {self.qpop_net.ref_dir_method}")
        self.qpop_net.set_objective_directions(names=self.objective_names, sense=self.objective_senses)
        print(f"Set objective directions: {list(zip(self.objective_names, self.objective_senses))}")
        if self.qpop_net._ref_dirs is not None and self.qpop_net._ind_to_dir is not None:
            for i in range(self.qpop_net.num_ind):
                dir_index = self.qpop_net._ind_to_dir[i]
                direction_vector = self.qpop_net._ref_dirs[dir_index]
                direction_str = ", ".join([f"{val:.3f}" for val in direction_vector])
                print(f"Quantum Individual {i}: Direction -> [{direction_str}]")
        else:
            print("Quantum directions have not been assigned yet. Please call 'set_objective_directions'.")
        
    def multiobjective_fitness(self) -> np.ndarray:
        """Compute multi-objective fitness and log a per-individual history record.

        The method:
        1) Decodes the current classical population.
        2) Calls the evaluator to obtain raw metric dictionaries per individual.
        3) Builds `raw_fits` and `fits` (applying penalties to the first objective if configured).
        4) Logs a JSONL record per individual using `_log_eval_history`, including:
            - generation, q_index (from `parent_map`)
            - candidate_id and canonical architecture key
            - primary metric (first objective) and its raw scalar value
            - a metrics payload that contains:
                * the raw evaluator dict
                * `_raw_vector`: full raw objectives
                * `_penalized_vector`: penalized objectives (if any)
                * `_penalty` and `_first_obj_penalized` when penalties apply
        5) Returns the penalized multi-objective `fits`.

        Returns:
            A (N, M) numpy array of penalized objective values for each individual,
            where N is population size and M is `self.num_objectives`.
        """
        # Decode to human-readable form (params + decoded nets).
        decoded_params, decoded_nets = self.decode_pop(self.classical_params, self.classical_nets)

        # Evaluate all individuals on all objectives; returns per-individual dicts.
        raw_results = self.eval_func(decoded_params, decoded_nets, generation=self.current_gen)

        N = len(decoded_nets)
        M = self.num_objectives
        raw_fits = np.zeros((N, M), dtype=float)
        fits     = np.zeros((N, M), dtype=float)

        # Fill raw_fits/fits with evaluator outputs for each objective.
        for i in range(N):
            for j, obj in enumerate(self.objectives[:M]):
                val = raw_results[i][obj]
                raw_fits[i, j] = val
                fits[i, j]     = val

        # Apply optional penalty to the first objective.
        penalty_vec = None
        if self.penalize_number and self.reducing_fns_list:
            penalty_vec = self.get_penalties(self.classical_nets)
            fits[:, 0] -= penalty_vec

        # Expose for downstream consumers / serializers.
        self.raw_fits = raw_fits
        self.fits = fits

        # Build extra payload for logging (vectors + penalty info).
        extra_metrics = []
        for i in range(N):
            em = {
                "_raw_vector": raw_fits[i, :].tolist(),
                "_penalized_vector": fits[i, :].tolist(),
            }
            if penalty_vec is not None:
                em["_penalty"] = float(penalty_vec[i])
                em["_first_obj_penalized"] = float(fits[i, 0])
            extra_metrics.append(em)

        # Log one record per individual (no reuse).
        self._log_eval_history(
            decoded_params,
            decoded_nets,
            results=raw_results,
            raw_fits=raw_fits,                   # 2D; helper will use column 0 as scalar fitness
            pop_net=self.classical_nets,         # encoded rows -> canonical architecture key
            objective_names=self.objectives[:M],
            extra_metrics=extra_metrics,
        )

        return fits

    def compute_hypervolume_mixed(self, front_raw: np.ndarray, ref_point=None) -> float:
        """
        Compute hypervolume for a 3-objective Pareto front where:
            - front_raw[:, 0] = accuracy (to be maximized)
            - front_raw[:, 1] = num_parameters (to be minimized)
            - front_raw[:, 2] = inference_time (to be minimized)

        We first convert everything into minimization form by flipping accuracy → -accuracy,
        then build a reference point slightly above the “worst” in each dimension,
        and finally call pymoo’s Hypervolume on that minimization front.

        Args:
            front_raw (np.ndarray): shape=(N, 3) with columns [acc, params, time].
            ref_point (np.ndarray): shape=(3,) with the reference point for hypervolume calculation.

        Returns:
            float: the hypervolume (in the original mixed‐obj space).
        """
        if front_raw is None or len(front_raw) == 0:
            return 0.0

        f = np.array(front_raw, dtype=float, copy=True)

        # Flip the sign for maximization objectives
        for i, sense in enumerate(self.objective_senses):
            if sense == 'max':
                f[:, i] = -f[:, i]

        # Choose a safe reference point (must be worse than all points for minimization)
        if ref_point is None:
            rp = np.max(f, axis=0) + 1e-6
        else:
            rp = np.asarray(ref_point, dtype=float)
            # Flip the sign for maximization objectives in the reference point as well
            for i, sense in enumerate(self.objective_senses):
                if sense == 'max':
                    rp[i] = -rp[i]

        return float(Hypervolume(ref_point=rp).do(f))
    
    def dominates(self, a, b):
        """
        Determine Pareto domination between two fitness tuples.

        Converts the first objective to minimization by negating it, then checks
        if `a` is no worse in all objectives and strictly better in at least one.

        Args:
            a (tuple or list): Fitness values for candidate a.
            b (tuple or list): Fitness values for candidate b.

        Returns:
            bool: True if a dominates b, False otherwise.
        """
        obj_a = np.array(a, copy=True)
        obj_b = np.array(b, copy=True)

        # Flip the sign for maximization objectives to convert them to minimization
        for i, sense in enumerate(self.objective_senses):
            if sense == 'max':
                obj_a[i] = -obj_a[i]
                obj_b[i] = -obj_b[i]

        return np.all(obj_a <= obj_b) and np.any(obj_a < obj_b)

    def fast_nondominated_sort(self, fitnesses: np.ndarray):
        """
        Perform the “fast non‐dominated sort” of NSGA‐II on the (2*pop_size × n_obj)
        fitness‐matrix. Returns a list of fronts, where each front is a list of indices.

        Args:
            fitnesses (np.ndarray): shape=(N_all, n_obj)

        Returns:
            fronts (list of lists): fronts[0] is list of indices of the first Pareto front,
                                fronts[1] is list of indices of the second front, etc.
        """
        N = fitnesses.shape[0]
        dominated_sets = [set() for _ in range(N)]
        dom_count = np.zeros(N, dtype=int)
        fronts = [[]]

        for p in range(N):
            for q in range(N):
                # if p == q:
                #     continue
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

        # The last appended front will be empty; discard it
        return fronts[:-1]

    def crowding_distance(self, fits, front):
        """
        Compute the crowding distance for individuals in a given front.

        Uses a vectorized approach over all objectives to measure solution density,
        assigning infinite distance to boundary points.

        Args:
            fits (np.ndarray): Fitness array of shape (N, M).
            front (list[int]): Indices of individuals in the front.

        Returns:
            np.ndarray: Crowding distances for each index in `front`.
        """
        f = fits[front]
        F, M = f.shape
        dist = np.zeros(F)
        if F <= 2:
            return np.array([np.inf] * F)
        sorted_idx = np.argsort(f, axis=0)
        dist[sorted_idx[0, :]] = np.inf
        dist[sorted_idx[-1, :]] = np.inf
        min_vals = f[sorted_idx[0, :], np.arange(M)]
        max_vals = f[sorted_idx[-1, :], np.arange(M)]
        denom = max_vals - min_vals
        for j in range(M):
            if denom[j] == 0:
                continue
            prev = f[sorted_idx[:-2, j], j]
            nxt = f[sorted_idx[2:, j], j]
            dist[sorted_idx[1:-1, j]] += (nxt - prev) / denom[j]
        return dist

    def environmental_selection(
        self, pop: np.ndarray, fits: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform NSGA-II selection on a combined population of size (2*pop_size).
        Ranks by Pareto fronts, then uses crowding distance to fill up to pop_size.
        Returns (selected_pop, selected_fits, selected_indices), where
        selected_indices is a 1D int array of length pop_size giving the row-indices
        in `pop` that were chosen.
        """
        pop_size = self.pop_size
        fronts = self.fast_nondominated_sort(fits)

        new_pop = np.zeros((pop_size, pop.shape[1]), dtype=pop.dtype)
        new_fits = np.zeros((pop_size, fits.shape[1]), dtype=float)
        selected_idx = []  # <-- collect indices here
        count = 0

        for front in fronts:
            # Si ya llenamos la población, salimos inmediatamente.
            if count >= pop_size:
                break

            front_size = len(front)
            if count + front_size <= pop_size:
                # Tomamos todo el frente completo
                new_pop[count : count + front_size] = pop[front]
                new_fits[count : count + front_size] = fits[front]
                selected_idx.extend(front)
                count += front_size
            else:
                # Quedan pocos huecos (rem) y el frente es más grande
                rem = pop_size - count
                if rem > 0:
                    cd = self.crowding_distance(fits, front)  # (len(front),)
                    # Seleccionamos los índices con mayor crowding distance
                    top_indices = np.argsort(cd)[-rem:]
                    # Convertimos esos índices relativos en índices absolutos sobre 'pop'
                    chosen = [front[i] for i in top_indices]
                    new_pop[count : count + rem] = pop[chosen]
                    new_fits[count : count + rem] = fits[chosen]
                    selected_idx.extend(chosen)
                    count = pop_size
                break  # Después de llenar con 'rem' individuos, rompemos el bucle
        selected_idx = np.array(selected_idx, dtype=int)
        return new_pop, new_fits, selected_idx

    def random_crossover_hyperparams(self, new_pop: np.ndarray) -> np.ndarray:
        """
        Perform a random‐parent “tournament” crossover on hyperparameters.

        Each child in new_pop has a chance (hyperparam_crossover_rate) to
        be crossed with a random parent from self.qpop_params.current_pop.
        """
        old_pop = self.qpop_params.current_pop
        N_old = 0 if old_pop is None else old_pop.shape[0]
        if N_old == 0:
            return new_pop

        mixed = new_pop.copy()
        for i in range(new_pop.shape[0]):
            if np.random.rand() < self.hyperparam_crossover_rate:
                p_idx = np.random.randint(N_old)
                parent = old_pop[p_idx]
                child = new_pop[i]
                mask = np.random.rand(*child.shape) < 0.5
                mixed[i] = np.where(mask, parent, child)
        return mixed

    def record_and_save_history(self):
            """
            Records the current global Pareto front, calculates its hypervolume,
            and saves the entire history to a pickle file.
            """
            # 1) Build the record for the current generation from the global archive.
            gen_record = {1: []} # Storing the front in a key '1'
            
            for i in range(len(self.pareto_global_ids)):
                individual_data = {"id": self.pareto_global_ids[i]}
                for j, obj_name in enumerate(self.objectives):
                    individual_data[obj_name] = float(self.pareto_global_fitnesses[i][j])
                gen_record[1].append(individual_data)

            # 2) Calculate the hypervolume of the current global front.
            hv = self.compute_hypervolume_mixed(self.pareto_global_fitnesses)
            gen_record["hypervolume"] = float(hv)
            
            # 3) Add the record for this generation to the main history dictionary.
            self.fronts_history[self.current_gen] = gen_record
            
            # 4) Persist the entire history to disk.
            history_path = os.path.join(self.experiment_path, "pareto_history.pkl")
            with open(history_path, "wb") as f:
                pickle.dump(self.fronts_history, f)

    def update_global_pareto_front(self):
        """
        Update the global Pareto archive by merging it with the current population
        and finding the new set of non-dominated solutions.

        This method relies on `self.classical_ids` containing the correct,
        persistent IDs for the individuals in the current population.
        """
        # Use the persistent IDs managed by the `evolve` loop.
        curr_ids = self.classical_ids

        if self.pareto_global_population is None:
            # If the archive is empty, initialize it with the current population.
            all_pop = self.classical_nets.copy()
            all_fits = self.fits.copy()
            all_params = self.classical_params.copy()
            all_ids = curr_ids.copy()
        else:
            # Otherwise, combine the existing archive with the current population.
            all_pop = np.vstack([self.pareto_global_population, self.classical_nets])
            all_fits = np.vstack([self.pareto_global_fitnesses, self.fits])
            all_params = np.vstack([self.pareto_global_params, self.classical_params])
            all_ids = self.pareto_global_ids + curr_ids

        unique_ids, unique_indices = np.unique(all_ids, return_index=True)
        # Filter the combined populations to keep only the first occurrence of each individual.
        unique_pop = all_pop[unique_indices]
        unique_fits = all_fits[unique_indices]
        unique_params = all_params[unique_indices]
        unique_ids = list(unique_ids) # Convert back to a list
        
        # 1) Perform a full non-dominated sort on the combined set.
        fronts = self.fast_nondominated_sort(unique_fits)
        
        # 2) The new global Pareto front consists of all individuals in the first front.
        idx0 = fronts[0]
        
        # 3) Update the global archive class attributes.
        self.pareto_global_population  = unique_pop[idx0]
        self.pareto_global_fitnesses   = unique_fits[idx0]
        self.pareto_global_params      = unique_params[idx0]
        self.pareto_global_ids         = [unique_ids[i] for i in idx0]
        
        # 4) Compute crowding distance on the final, updated global front.
        #    This is stored for the quantum update logic in `go_next_gen`.
        self._last_cd = self.crowding_distance(self.pareto_global_fitnesses,
            list(range(len(self.pareto_global_fitnesses))))
    
    def go_next_gen(self):
        """
        Orchestrates end-of-generation tasks: updating the global archive,
        recording history, updating quantum populations, and cleaning up.
        """
        # 1. Update the global Pareto archive with the latest population's results.
        # This method now updates self.pareto_global_* attributes directly and
        # calculates and stores the crowding distance of the new front in self._last_cd.
        self.update_global_pareto_front()

        # 2. Record the history of the updated global front and save it to disk.
        self.record_and_save_history()
        self.history.flush()

        # 3. Select a diverse subset from the global front to update the quantum populations.
        # We use the crowding distance that was calculated and stored in the previous step.
        cd = self._last_cd
        sorted_rel = np.argsort(cd)[::-1]
        # pick = sorted_rel[:self.qpop_net.num_ind]

        # 4. Set the chosen individuals as the 'parents' for the quantum update.
        self.qpop_net.current_pop = self.pareto_global_population[sorted_rel]
        self.qpop_net.current_pop_objs = self.pareto_global_fitnesses[sorted_rel]
        
        if self.pareto_global_params is not None:
            self.qpop_params.current_pop = self.pareto_global_params[sorted_rel]

        # 5. Trigger the quantum population update (the learning step).
        self.update_quantum(self.current_gen)

        # 6. Log a summary of the generation's results.
        hv = self.fronts_history[self.current_gen]['hypervolume']
        self.logger.info("Generation %d: updated global Pareto front with %d individuals and hypervolume %.2f",
            self.current_gen,
            len(self.pareto_global_population),
            hv,
        )
        display_ids = [str(item) for item in self.pareto_global_ids]
        self.logger.info("Generation %d: current global Pareto IDs:\n%s",
            self.current_gen,
            display_ids,
        )

        fitness_str = np.array2string(
            self.pareto_global_fitnesses,
            separator='  ',
            formatter={'float_kind': lambda x: f"{x:.3f}"}
        )

        self.logger.info(
            "Generation %d global Pareto fitness:\n%s (n=%d)",
            self.current_gen,
            fitness_str,
            len(self.pareto_global_population),
        )
        is_snapshot = (self.current_gen % 5 == 0) and (self.current_gen > 0)
        # 7. Clean up old model directories, keeping only those in the global archive.
        delete_old_dirs_v2(self.experiment_path, self.current_gen, 
                        keep_ids=self.pareto_global_ids.copy(), is_snapshot_gen=is_snapshot)
        if self.current_gen == 1:
            # On the first run, also clean up directories from generation 0.
            delete_old_dirs_v2(self.experiment_path, 0, keep_ids=self.pareto_global_ids.copy())

        # 8. Save other necessary data from the parent class and advance the generation counter.
        self.save_data()
        self.current_gen += 1

    def evolve(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Run MoQNAS for max_generations, maintaining a global Pareto archive.

        Workflow per generation:
        1. Generation 0:
                - (p0_params, p0_nets) = self.generate_classical()
                - f0 = self.multiobjective_fitness()
                - assign fits, raw_fits; record best_so_far
                - DO NOT call go_next_gen for gen=0
        2. For gen in [1..max_generations]:
                a) self.current_gen = gen
                b) children_params, children_nets = self.generate_classical()
                c) child_fits = self.multiobjective_fitness()
                d) Combine parents + children and run NSGA‐II
                e) Assign survivors → self.classical_nets, self.fits, self.raw_fits
                f) Resample hyperparams for next gen: children_params already from generate_classical()
                h) Call self.go_next_gen()    # handles global Pareto archiving, cleanup, logging, ++gen
                i) Early stop if check_early_stopping()
                j) Prepare (p0_params, p0_nets, f0, p0_raws) for next iteration
        3. Return final global Pareto archive
        """
        # 1) Generation 0: sample both hyperparams and nets via generate_classical()
        start_time = time.time()
        p0_params, p0_nets = self.generate_classical()
        self.classical_params = p0_params
        self.classical_nets = p0_nets
        
        self.qpop_params.current_pop = p0_params
        self.qpop_net.current_pop    = p0_nets

        # Evaluate generation 0
        f0 = self.multiobjective_fitness()
        self.logger.info("Generation 0: fitnesses:\n%s", f0)
        
        p0_ids = [f"0_{i}" for i in range(len(p0_nets))]
        self.classical_ids = p0_ids # Initialize a new attribute to hold current IDs
    
        self.fits = f0
        self.raw_fits = self.raw_fits.copy()

        # Record best‐so‐far (by first objective)
        i0 = int(np.nanargmax(f0[:, 0]))
        self.best_so_far = float(f0[i0, 0])
        self.best_so_far_id = [0, i0]

        # Keep copies for parent‐combination in next loop
        p0_raws = self.raw_fits.copy()

        # 2) Main loop: generations 1..max_generations
        for gen in range(1, self.max_generations + 1):
            self.current_gen = gen

            # 2a) Sample children classical population (both params and nets) at once
            children_params, children_nets = self.generate_classical()
            child_ids = [f"{self.current_gen}_{i}" for i in range(len(children_nets))]


            children_params = self.random_crossover_hyperparams(children_params)
            children_nets = self.crossover_network(children_nets)
            
            # 2b) Evaluate children on all objectives
            self.classical_params = children_params
            self.classical_nets = children_nets
            child_fits = self.multiobjective_fitness()
            child_raw = self.raw_fits.copy()

            # 2c) Combine parents + children
            combined_nets = np.vstack([p0_nets, children_nets])   # shape = (2*pop_size, net_dim)
            combined_fits = np.vstack([f0, child_fits])           # shape = (2*pop_size, n_obj)
            combined_raws = np.vstack([p0_raws, child_raw])       # shape = (2*pop_size, n_obj)
            combined_ids = p0_ids + child_ids                    # shape = (2*pop_size,)
            combined_params = np.vstack([p0_params, children_params])
            # 2d) NSGA‐II environmental selection
            next_nets, next_fits, survivor_idx = self.environmental_selection(combined_nets, combined_fits)

            # 2e) Assign survivors
            self.classical_nets = next_nets
            self.fits = next_fits
            self.raw_fits = combined_raws[survivor_idx]
            self.classical_ids = [combined_ids[i] for i in survivor_idx]
            self.classical_params = combined_params[survivor_idx]

            # 2h) Advance generation: update global Pareto, backup, save, log, cleanup, increment gen
            self.go_next_gen()

            # 2i) Early stopping
            if self.early_stopping and self.check_early_stopping():
                break

            # 2j) Prepare for next iteration
            p0_params = self.classical_params
            p0_nets = self.classical_nets
            f0 = self.fits
            p0_raws = self.raw_fits
            p0_ids = self.classical_ids
            if self.current_gen > 0 and (self.current_gen % 5 == 0):
                curr_time = time.time()
                h, m, est_h, est_m = calculate_time(
                    start_time, curr_time, self.current_gen, self.max_generations, end_evol=False
                )
                self.logger.info(
                    "Gen %d: elapsed %dh %dm; ETA %dh %dm",
                    self.current_gen, h, m, est_h, est_m,
                )
        total_h, total_m = calculate_time(start_time, time.time())
        self.logger.info("Total evolution time: %d hours and %d minutes", total_h, total_m)

        # 3) Return final global Pareto archive
        return self.pareto_global_population, self.pareto_global_fitnesses