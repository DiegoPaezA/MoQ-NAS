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
from util import backup_cache, delete_old_dirs_v2


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
        # The QNAS base class manages hyperparameter quantum populations (qpop_params)
        # and network quantum populations (qpop_net). We will perform multiobjective selection
        # on the *classical* network‐chromosome population; hyperparameters will continue
        # to be sampled from their quantum PMFs each generation and updated based on
        # the Pareto‐optimal survivors.
        self.pop_size = None          # number of classical nets per generation
        self.max_generations = None   # total generations to run

    def initialize_moqnas(self,num_quantum_ind,params_ranges,repetition,pop_size,max_generations,
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
            pop_size (int): Number of classical networks to keep each generation.
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
        self.pop_size = pop_size
        self.max_generations = max_generations

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
        self.raw_fits = raw_fits.copy()
        self.fits = fits.copy()

        return fits

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

    def crowding_distance(self, fitnesses: np.ndarray, front: list) -> np.ndarray:
        """
        Compute the crowding distance for each individual in a given Pareto front.

        Args:
            fitnesses (np.ndarray): shape=(N_all, n_obj)
            front (list): list of indices into fitnesses for the current front.

        Returns:
            distances (np.ndarray): shape=(len(front),) of crowding distances.
                                    Boundary points get +inf.
        """
        f = fitnesses[front]
        F, M = f.shape
        distances = np.zeros(F, dtype=float)
        if F <= 2:
            return np.array([np.inf] * F)

        # For each objective dimension j
        sorted_idx = np.argsort(f, axis=0)
        distances[sorted_idx[0, :]] = np.inf
        distances[sorted_idx[-1, :]] = np.inf
        f_min = f[sorted_idx[0, :], np.arange(M)]
        f_max = f[sorted_idx[-1, :], np.arange(M)]
        denom = f_max - f_min

        for j in range(M):
            if denom[j] == 0:
                continue
            prev_vals = f[sorted_idx[:-2, j], j]
            next_vals = f[sorted_idx[2:, j], j]
            # accumulate distance; note sorted_idx[1:-1, j] are the interior points
            distances[sorted_idx[1:-1, j]] += (next_vals - prev_vals) / denom[j]

        return distances

    def environmental_selection(
        self, pop: np.ndarray, fits: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform NSGA-II selection on a combined population of size (2*pop_size).
        Ranks by Pareto fronts, then uses crowding distance to fill up to pop_size.
        """
        pop_size = self.pop_size
        fronts = self.fast_nondominated_sort(fits)

        new_pop = np.zeros((pop_size, pop.shape[1]), dtype=pop.dtype)
        new_fits = np.zeros((pop_size, fits.shape[1]), dtype=float)
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
                    count = pop_size
                break  # Después de llenar con 'rem' individuos, rompemos el bucle

        return new_pop, new_fits

    def reproduce(self) -> np.ndarray:
        """
        Generate a new classical offspring population (size = pop_size) from the
        current classical network population (self.classical_nets) using:
            - Binary tournament selection (based on objective 0, the primary objective).
            - Uniform crossover (mask = 0.5).
            - QNAS’s mutate() (which randomly applies one of swap, block, neighbor, or gene mutation).

        Note: We ignore hyperparameter reproduction here; instead, hyperparams are always
        re‐sampled from quantum (self.qpop_params) each generation in evolve().

        Returns:
            new_offspring (np.ndarray): shape=(pop_size, net_dim) of children network chromosomes.
        """
        parents = self.classical_nets.copy()
        N = parents.shape[0]
        new_pop = []

        # (Optional) Elitism: carry forward the single best network by objective 0
        if getattr(self, "elitism", False):
            best_idx = np.argmax(self.fits[:, 0])
            new_pop.append(parents[best_idx].copy())

        while len(new_pop) < self.pop_size:
            # Tournament for parent1
            i1, i2 = np.random.choice(N, 2, replace=False)
            if self.fits[i1, 0] > self.fits[i2, 0]:
                p1 = parents[i1]
            else:
                p1 = parents[i2]
            # Tournament for parent2
            j1, j2 = np.random.choice(N, 2, replace=False)
            if self.fits[j1, 0] > self.fits[j2, 0]:
                p2 = parents[j1]
            else:
                p2 = parents[j2]

            # Uniform crossover
            if np.random.rand() < self.crossover_rate:
                mask = np.random.rand(*p1.shape) < 0.5
                c1 = np.where(mask, p1, p2)
                c2 = np.where(mask, p2, p1)
            else:
                c1 = p1.copy()
                c2 = p2.copy()

            # Mutation
            c1 = self.mutate(c1)
            c2 = self.mutate(c2)

            new_pop.append(c1)
            if len(new_pop) < self.pop_size:
                new_pop.append(c2)

        return np.array(new_pop, dtype=parents.dtype)
    
    def update_global_pareto_front(self, combined_pop: np.ndarray, combined_fits: np.ndarray, combined_ids: list):
        """
        Update the global Pareto buffer by merging existing archive with current combined population.

        Steps:
            1. If archive is empty, set archive = combined_pop, combined_fits, combined_ids.
            2. Else, vertically stack archive + combined.
            3. Perform fast_nondominated_sort on merged fitnesses.
            4. Extract front0 indices; apply crowding to that front to filter extremes.
            5. Store the filtered front0 as new global archive, update pareto_global_ids.
        
        Args:
            combined_pop (np.ndarray): shape=(N_all, net_dim) merged parent+offspring networks.
            combined_fits (np.ndarray): shape=(N_all, n_obj) merged fitness array.
            combined_ids (list of str): length N_all, IDs like "gen_candidateidx".
        
        Returns:
            None (updates self.pareto_global_* attributes in place).
        """
        # 1) Merge with existing archive if any
        if self.pareto_global_population is None:
            all_pop = combined_pop.copy()
            all_fits = combined_fits.copy()
            all_ids = combined_ids.copy()
        else:
            all_pop = np.vstack([self.pareto_global_population, combined_pop])
            all_fits = np.vstack([self.pareto_global_fitnesses, combined_fits])
            all_ids = self.pareto_global_ids + combined_ids

        # 2) Fast non‐dominated sort on all_fits
        fronts = self.fast_nondominated_sort(all_fits)
        front0 = fronts[0]
        pop0 = all_pop[front0]
        fit0 = all_fits[front0]
        ids0 = [all_ids[i] for i in front0]

        # 3) Compute crowding distance on front0, keep boundary + high‐distance solutions
        cd = self.crowding_distance(fit0, list(range(len(pop0))))
        keep_mask = np.isinf(cd) | (cd > 0)
        self.pareto_global_population = pop0[keep_mask]
        self.pareto_global_fitnesses = fit0[keep_mask]
        self.pareto_global_ids = [ids0[i] for i, k in enumerate(keep_mask) if k]

    def record_global_fronts_history(self, fronts_info):
        """
        Record all global Pareto fronts into a nested dictionary and persist to disk.

        Args:
            fronts_info (tuple): (all_pop, all_fits, all_ids, fronts) from update_global_pareto_front
        """
        all_pop, all_fits, all_ids, fronts = fronts_info
        gen_global = {}
        for level, front in enumerate(fronts, start=1):
            gen_global[level] = [
                {
                    "id": all_ids[i],
                    **{
                        self.objectives[j]: float(all_fits[i][j])
                        for j in range(self.num_objectives)
                    }
                }
                for i in front
            ]
        self.fronts_history[self.current_gen] = gen_global

        # Persist to disk
        hist_file = os.path.join(self.experiment_path, "pareto_history.pkl")
        with open(hist_file, "wb") as f:
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
        # Build combined_pop, combined_fits, combined_ids:
        # We need to combine existing archive with the current generation’s population.
        # However, QNAS’s go_next_gen is called only after we assign Pareto survivors into
        # self.qpop_* .current_pop. Therefore “current population” here refers to self.classical_nets
        # and self.fits (the first front of the last environmental_selection).
        pop_curr = self.classical_nets
        fits_curr = self.fits
        ids_curr = [f"{self.current_gen}_{i}" for i in range(len(pop_curr))]

        # If no existing archive, combined = current; else merge
        if self.pareto_global_population is None:
            all_pop = pop_curr.copy()
            all_fits = fits_curr.copy()
            all_ids = ids_curr.copy()
        else:
            all_pop = np.vstack([self.pareto_global_population, pop_curr])
            all_fits = np.vstack([self.pareto_global_fitnesses, fits_curr])
            all_ids = self.pareto_global_ids + ids_curr

        # Perform fast non‐dominated sort on all_fits
        fronts = self.fast_nondominated_sort(all_fits)

        # Before filtering via crowding, record entire fronts history
        self.record_global_fronts_history((all_pop, all_fits, all_ids, fronts))

        # Filter front0 by crowding distance
        front0 = fronts[0]
        pop0 = all_pop[front0]
        fit0 = all_fits[front0]
        ids0 = [all_ids[i] for i in front0]
        cd = self.crowding_distance(fit0, list(range(len(pop0))))
        keep_mask = np.isinf(cd) | (cd > 0)
        self.pareto_global_population = pop0[keep_mask]
        self.pareto_global_fitnesses = fit0[keep_mask]
        self.pareto_global_ids = [ids0[i] for i, k in enumerate(keep_mask) if k]

        # Delete old model directories, keep only global Pareto IDs
        delete_old_dirs_v2(
            self.experiment_path,
            self.current_gen,
            keep_ids=self.pareto_global_ids.copy(),
        )
        if self.current_gen == 1:
            delete_old_dirs_v2(self.experiment_path, 0, keep_ids=self.pareto_global_ids.copy())

        # Log global front size
        self.logger.info(
            "Generation %d global Pareto front size: %d",
            self.current_gen,
            len(self.pareto_global_population),
        )

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
        p0_params, p0_nets = self.generate_classical()
        self.classical_params = p0_params
        self.classical_nets = p0_nets

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
            next_nets, next_fits = self.environmental_selection(combined_nets, combined_fits)

            # 2e) Assign survivors
            self.classical_nets = next_nets
            self.fits = next_fits

            # Extract corresponding raw_fits for survivors
            next_raw = []
            for net in next_nets:
                idx = np.where((combined_nets == net).all(axis=1))[0][0]
                next_raw.append(combined_raws[idx])
            self.raw_fits = np.array(next_raw)

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
            p0_params = self.classical_params.copy()
            p0_nets = self.classical_nets.copy()
            f0 = self.fits.copy()
            p0_raws = self.raw_fits.copy()

        # 3) Return final global Pareto archive
        return self.pareto_global_population, self.pareto_global_fitnesses
