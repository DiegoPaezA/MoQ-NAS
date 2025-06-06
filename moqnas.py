""" 
MoQNAS: Multi‐Objective Quantum‐Inspired Neural Architecture Search
Based on the refactored QNAS (qnas2.py). 

This implements NSGA-II style Pareto selection over multiple objectives, 
while still using the quantum‐population machinery from QNAS to evolve 
hyperparameter distributions. 
"""
import os
import time
import pickle
import numpy as np
from qnas2 import QNAS
from pymoo.indicators.hv import Hypervolume
from util import calculate_time, delete_old_dirs_v2


class MOQNAS(QNAS):
    """ Multi‐Objective QNAS: extends QNAS to optimize multiple objectives via NSGA‐II. """

    def __init__(
        self,
        eval_func,
        experiment_path,
        objectives,
        log_file,
        log_level,
        data_file,
    ):
        """
        Initialize MoQNAS.

        Args:
            eval_func (callable): 
                Given lists of decoded_params (dict) and decoded_nets (structures), 
                returns a list/array of dicts, each dict mapping metric_name→value.
            experiment_path (str): Path for logs, cache, etc.
            objectives (list of str): Names of metrics to optimize, e.g. ["accuracy","latency"].
            log_file (str): Path to logfile.
            log_level (str): "INFO", "DEBUG", or "NONE".
            data_file (str): Path to pickle file for saving per‐gen data.
        """
        super().__init__(eval_func, experiment_path, objectives, log_file, log_level, data_file)
        self.num_objectives = len(objectives)
        self.data_file = data_file
        # The QNAS base class manages hyperparameter quantum populations (qpop_params)
        # and network quantum populations (qpop_net). We will perform multiobjective selection
        # on the *classical* network‐chromosome population; hyperparameters will continue
        # to be sampled from their quantum PMFs each generation and updated based on
        # the Pareto‐optimal survivors.
        self.pop_size = None          # number of classical nets per generation
        self.max_generations = None   # total generations to run

    def initialize_moqnas(self,num_quantum_ind,params_ranges,repetition,crossover_rate,max_generations,
                        update_quantum_gen,replace_method,fn_list,initial_probs,update_quantum_rate,
                        max_num_nodes,reducing_fns_list,patience,early_stopping,save_data_freq=0,
                        penalize_number=0,crossover_frequency=5,en_pop_crossover=False,
                        pop_crossover_rate=0.25,pop_crossover_method="hux",):
        """
        Initialize MoQNAS (quantum populations + multiobjective settings).

        Args:
            num_quantum_ind (int): Number of quantum individuals.
            params_ranges (dict): Hyperparameter ranges for QPopulationParams.
            repetition (int): # classical per quantum = population size multiplier.
            crossover_rate (float): Probability of applying classical crossover.
            max_generations (int): Total number of generations (G).
            update_quantum_gen (int): Frequency (in generations) to update quantum PMFs.
            replace_method (str): Either "elitism" or "best" (not used in MoQNAS, but inherited).
            fn_list (list): List of possible layer‐IDs (for QPopulationNetwork).
            initial_probs (list): Initial probabilities for each layer‐ID.
            update_quantum_rate (float): Intensity of quantum update each time.
            max_num_nodes (int): Max nodes in each network chromosome.
            reducing_fns_list (list): Which layer‐IDs count as "reducing" for penalty.
            patience (int): Generations to wait without improvement before stopping.
            early_stopping (bool): If True, enable early stopping.
            save_data_freq (int, optional): Gen‐frequency to save best‐model stats.
            penalize_number (int, optional): Max allowed “reducing” layers before penalty.
            crossover_frequency (int, optional): Gen‐interval to apply network crossover.
            en_pop_crossover (bool, optional): Whether to enable network crossover.
            pop_crossover_rate (float, optional): Fraction of offspring to crossover each time.
            pop_crossover_method (str, optional): “hux” or “uniform” method for network crossover.
        """
        # 1) Store multiobjective sizes
        self.pop_size = num_quantum_ind*repetition  # Classical population size
        self.max_generations = max_generations
        self.hyperparam_crossover_rate = crossover_rate  

        # 2) Initialize QNAS (hyperparams & network quantum populations)
        super().initialize_qnas(
            num_quantum_ind=num_quantum_ind,
            params_ranges=params_ranges,
            repetition=repetition,
            max_generations=max_generations,
            crossover_rate=None,            # we will not use hyperparam classical crossover here
            update_quantum_gen=update_quantum_gen,
            replace_method=replace_method,  # not used in MoQNAS; selection is Pareto
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
        )

        # 3) Create the very first classical network population by sampling from quantum
        #    and store it in self.classical_nets
        self.classical_nets = self.qpop_net.generate_classical()  # shape = (pop_size, net_dim)

        # 4) Placeholder for hyperparameter classical sampling (unchanged QNAS logic)
        self.classical_params = self.qpop_params.generate_classical()  # shape = (pop_size, param_dim)

        # 5) Placeholder for current multiobjective fitness arrays
        self.fits = None     # Will be (pop_size × n_obj)
        self.raw_fits = None # Will be (pop_size × n_obj)
        
        # 4) Initialize global Pareto archive empty
        self.pareto_global_population = None
        self.pareto_global_fitnesses = None
        self.pareto_global_ids = []
        self.fronts_history = {}


    def multiobjective_fitness(self) -> np.ndarray:
        """
        Evaluate the current classical population on all objectives.

        Steps:
            1. Decode classical_params and classical_nets via QNAS.decode_pop.
            2. Call self.eval_func(list_of_param_dicts, list_of_net_structures, generation)
                which returns a list (length=pop_size) of dicts, each mapping metric_name→value.
            3. Build a (pop_size × n_obj) array, selecting metrics in the order of self.objectives.
            4. Apply the same “penalty” from QNAS to the first objective column (if penalize_number>0).
        
        Returns:
            fits (np.ndarray): shape=(pop_size, n_obj) of penalized objective values.
        """
        # 1) Decode current classical to human‐readable
        decoded_params, decoded_nets = self.decode_pop(
            self.classical_params, self.classical_nets
        )

        # 2) Evaluate all individuals: eval_func returns list/array of dicts
        raw_results = self.eval_func(decoded_params, decoded_nets, generation=self.current_gen)
        # raw_results[i] is a dict of metric_name→value for individual i

        N = len(decoded_nets)
        fits = np.zeros((N, self.num_objectives), dtype=float)
        raw_fits = np.zeros((N, self.num_objectives), dtype=float)

        # 3) Fill each column j with metric self.objectives[j]
        for i in range(N):
            metrics = raw_results[i]
            for j, obj_name in enumerate(self.objectives[: self.num_objectives]):
                val = metrics[obj_name]
                raw_fits[i, j] = val
                fits[i, j] = val

        # 4) Apply penalty to first objective if needed (QNAS.get_penalties uses network topology)
        if self.penalize_number and self.reducing_fns_list:
            penalties = self.get_penalties(self.classical_nets)
            fits[:, 0] -= penalties

        # 5) Store raw_fits for logging / saving
        self.raw_fits = raw_fits
        self.fits = fits

        return fits
    @staticmethod
    def compute_hypervolume_mixed(front_raw: np.ndarray, ε: float = 1e-6) -> float:
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
            ε (float): tiny margin to add to the reference point.

        Returns:
            float: the hypervolume (in the original mixed‐obj space).
        """
        if front_raw.size == 0:
            return 0.0

        # 1) Build minimization front: acc → -acc, params & time unchanged
        front_min = front_raw.copy()
        front_min[:, 0] *= -1.0   # flip accuracy

        # 2) Construct the reference point (for each column, take max(front_min[:,j]) + ε)
        ref = np.max(front_min, axis=0) + ε

        # 3) Call pymoo’s Hypervolume (which expects a minimization front and ref_point)
        hv_indicator = Hypervolume(ref_point=ref)
        hv_value     = hv_indicator(front_min)

        return float(hv_value)

    @staticmethod
    def dominates(a: np.ndarray, b: np.ndarray) -> bool:
        """
        Return True if vector a Pareto‐dominates vector b
        under the convention that *larger* is better for all objectives
        except we penalize the first objective in multiobjective_fitness,
        so here we assume a and b are all to‐be maximized.

        For strict Pareto dominance: a_i >= b_i for all i, and a_i > b_i for some i.

        Args:
            a (np.ndarray): shape=(n_obj,)
            b (np.ndarray): shape=(n_obj,)

        Returns:
            bool: True if a dominates b, False otherwise.
        """
        return np.all(a >= b) and np.any(a > b)

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
                if p == q:
                    continue
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

    # def crowding_distance(self, fitnesses: np.ndarray, front: list) -> np.ndarray:
    #     """
    #     Compute the crowding distance for a given Pareto front.

    #     Args:
    #         fitnesses (np.ndarray): shape=(N_all, M) array of fitness values.
    #         front (list of int): indices of individuals in the same Pareto front.

    #     Returns:
    #         distances (np.ndarray): shape=(len(front),) crowding distances (boundary points = +inf).
    #     """
    #     f = fitnesses[front]
    #     F, M = f.shape
    #     if F <= 2:
    #         return np.full(F, np.inf, dtype=float)

    #     sorted_idx = np.argsort(f, axis=0)   # (F, M)
    #     distances = np.zeros(F, dtype=float)
    #     distances[sorted_idx[0,  :]] = np.inf
    #     distances[sorted_idx[-1, :]] = np.inf

    #     f_min = f[sorted_idx[0,  :], np.arange(M)]
    #     f_max = f[sorted_idx[-1, :], np.arange(M)]
    #     denom = f_max - f_min

    #     f_sorted = np.take_along_axis(f, sorted_idx, axis=0)  # (F, M)
    #     diff_all  = f_sorted[2:, :] - f_sorted[:-2, :]        # (F-2, M)

    #     zero_denom = (denom == 0)
    #     denom_safe = denom.copy()
    #     denom_safe[zero_denom] = 1.0
    #     normalized_diff = diff_all / denom_safe[np.newaxis, :]
    #     normalized_diff[:, zero_denom] = 0.0

    #     interior_idx = sorted_idx[1:-1, :]     # (F-2, M)
    #     flat_positions = interior_idx.reshape(-1)      # (F-2)*M
    #     flat_values    = normalized_diff.reshape(-1)   # (F-2)*M

    #     np.add.at(distances, flat_positions, flat_values)

    #     return distances
    
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

    def update_global_pareto_front(self):
        """
        Update the global Pareto buffer by merging the existing archive with the
        current generation’s population, performing a full non‐dominated sort,
        and then filtering front 0 by crowding distance.

        Returns:
            tuple:
                all_pop (ndarray): stacked [old_archive; current_pop]
                all_fits (ndarray): stacked [old_archive_fits; current_fits]
                all_ids (list):    concatenated [old_archive_ids + current_ids]
                fronts (list of lists): full list of Pareto fronts on all_fits
        """
        # Build IDs for each individual in the current classical population
        curr_ids = [f"{self.current_gen}_{i}" for i in range(len(self.classical_nets))]

        if self.pareto_global_population is None:
            # If archive is empty, start with current population
            all_pop = self.classical_nets.copy()
            all_fits = self.fits.copy()
            all_ids = curr_ids.copy()
        else:
            # Otherwise, stack old archive + current pop
            all_pop = np.vstack([self.pareto_global_population, self.classical_nets])
            all_fits = np.vstack([self.pareto_global_fitnesses, self.fits])
            all_ids = self.pareto_global_ids + curr_ids

        # 1) Full Pareto sort on combined fitnesses
        fronts = self.fast_nondominated_sort(all_fits)
        idx0 = fronts[0]

        # 2) Take only front 0 entries
        pop0 = all_pop[idx0]
        fit0 = all_fits[idx0]
        ids0 = [all_ids[i] for i in idx0]

        # 3) Compute crowding distances on front 0
        cd = self.crowding_distance(fit0, list(range(len(pop0))))
        # Keep boundary (inf) or any with cd > 0
        mask = np.isinf(cd) | (cd > 0)

        # 4) Update the archive to only those survivors
        self.pareto_global_population = pop0[mask]
        self.pareto_global_fitnesses = fit0[mask]
        self.pareto_global_ids = [ids0[i] for i, keep in enumerate(mask) if keep]

        return all_pop, all_fits, all_ids, fronts


    def record_global_fronts_history(self, fronts_info=None, hv=None):
        """
        Record Pareto fronts into self.fronts_history (including hypervolume) and persist to disk.

        If fronts_info is provided, it should be a tuple
        (all_pop, all_fits, all_ids, fronts) representing the combined
        archive + current population and the full Pareto fronts on them.
        Otherwise, recompute fronts from self.pareto_global_*.

        Args:
            fronts_info (tuple, optional):
                all_pop (np.ndarray): stacked [old_archive; current_pop]
                all_fits (np.ndarray): stacked [old_archive_fits; current_fits]
                all_ids (list):     [old_archive_ids + current_ids]
                fronts (list of lists): Pareto fronts on all_fits
            hv (float, optional):
                The hypervolume value for this generation. If provided, it will
                be saved alongside the fronts.
        """
        all_pop, all_fit, all_ids, fronts = fronts_info

        # Build a dict of fronts (level → list of individuals’ data)
        gen_global = {}
        for level, front in enumerate(fronts, start=1):
            gen_global[level] = [
                {
                    "id": all_ids[i],
                    "accuracy": float(all_fit[i][0]),
                    "params": float(all_fit[i][1]),
                    "inference_time": float(all_fit[i][2])
                }
                for i in front
            ]

        if hv is not None:
            gen_global["hypervolume"] = float(hv)

        # Save into fronts_history, keyed by this generation number
        self.fronts_history[self.current_gen] = gen_global

        # Dump to disk
        with open(os.path.join(self.experiment_path, "pareto_history.pkl"), "wb") as f:
            pickle.dump(self.fronts_history, f)

    def go_next_gen(self):
        """
        Archive and record the current global Pareto front, then advance to the next generation.

        Overrides QNAS.go_next_gen. Steps:
            1. Build combined_pop, combined_fits, combined_ids from current & previous populations.
            2. Update global Pareto front via update_global_pareto_front().
            3. Record full Pareto fronts history via record_global_fronts_history().
            4. Delete old model directories, keeping only current global Pareto IDs.
            5. Log a summary line, then increment self.current_gen.
        """
        all_pop, all_fits, all_ids, fronts = self.update_global_pareto_front()
        
        hv = self.compute_hypervolume_mixed(self.pareto_global_fitnesses)
        self.logger.info(f"Gen {self.current_gen} → Hypervolume = {hv:.3f}")
        
        self.record_global_fronts_history((all_pop, all_fits, all_ids, fronts), hv=hv)

        # Delete old model directories, keep only global Pareto IDs
        delete_old_dirs_v2(self.experiment_path,self.current_gen,keep_ids=self.pareto_global_ids.copy())
        
        if self.current_gen == 1:
            delete_old_dirs_v2(self.experiment_path, 0, keep_ids=self.pareto_global_ids.copy())
            
        # Log global front size
        self.logger.info(
            "Generation %d global Pareto front size: %d",
            self.current_gen,
            len(self.pareto_global_population),
        )
        self.save_data()
        # Finally, increment generation counter
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
                g) Identify Pareto front in combined, assign to qpop_*.current_pop
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
        self.fits = f0
        self.raw_fits = self.raw_fits.copy()

        # Record best‐so‐far (by first objective)
        self.best_so_far = self.fits[0, 0]
        self.best_so_far_id = [0, 0]

        # Keep copies for parent‐combination in next loop
        p0_raws = self.raw_fits.copy()

        # 2) Main loop: generations 1..max_generations
        for gen in range(1, self.max_generations + 1):
            self.current_gen = gen

            # 2a) Sample children classical population (both params and nets) at once
            children_params, children_nets = self.generate_classical()

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

            # 2d) NSGA‐II environmental selection
            next_nets, next_fits, survivor_idx = self.environmental_selection(combined_nets, combined_fits)

            # 2e) Assign survivors
            self.classical_nets = next_nets
            self.fits = next_fits
            self.raw_fits = combined_raws[survivor_idx]

            # 2f) Resample hyperparams for next generation via generate_classical()
            #     (we will overwrite these again at the top of the next loop iteration anyway)
            self.classical_params = children_params

            # 2g) Identify Pareto front in combined_fits and assign to quantum populations
            fronts = self.fast_nondominated_sort(combined_fits)
            pareto_front = fronts[0]
            pareto_nets = combined_nets[pareto_front]
            pareto_params = np.vstack([p0_params, children_params])[pareto_front]

            # Assign Pareto survivors into quantum current_pop
            self.qpop_net.current_pop = pareto_nets
            self.qpop_params.current_pop = pareto_params

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
