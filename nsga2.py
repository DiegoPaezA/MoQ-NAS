import os
import time
import pickle
import numpy as np
from ga import GA
from util import backup_cache, delete_old_dirs_v2

class NSGA2(GA):
    """
    NSGA-II multi-objective genetic algorithm implementation.

    Extends a base GA to handle an arbitrary number of objectives (>=2). Implements:
      - Evaluation caching to avoid redundant fitness computations.
      - Fast non-dominated sorting with dynamic number of objectives.
      - Vectorized crowding distance calculation for diversity preservation.
      - Preallocated arrays for offspring generation and environmental selection.
      - Global Pareto front archive with crowding-based filtering.
      - History recording of all Pareto fronts per generation.
    """
    def __init__(self, eval_func, experiment_path, log_file, log_level, data_file):
        """
        Initialize NSGA2 instance.

        Args:
            eval_func (callable): Function to evaluate individuals. Should accept
                (decoded_params_list, decoded_nets_list, generation) and return
                a list of fitness tuples.
            experiment_path (str): Base directory for logging and archives.
            log_file (str): File path for logging.
            log_level (int): Logging level.
            data_file (str): Path to initial data or checkpoint.
        """
        super().__init__(eval_func, experiment_path, log_file, log_level, data_file)
        self.eval_cache = {}
        self.pareto_global_population = None
        self.pareto_global_fitnesses = None
        self.pareto_global_ids = []
        self.fronts_history = {}
        self.num_objectives = None

    def evaluate_population(self):
        """
        Evaluate the current population, using a cache to avoid re-evaluating.

        Decodes the population, hashes individuals to detect repeats, and updates
        fitnesses. Sets self.fitnesses and num_objectives on first run.

        Returns:
            np.ndarray: Array of shape (population_size, num_objectives) with fitnesses.
        """
        decoded_nets, decoded_params = self.decode_pop()
        pop = self.population
        N = len(pop)
        fits = [None] * N
        keys = [pop[i].tobytes() for i in range(N)]
        to_eval = []
        for i, k in enumerate(keys):
            if k in self.eval_cache:
                fits[i] = self.eval_cache[k]
            else:
                to_eval.append(i)
        if to_eval:
            sub_nets = [decoded_nets[i] for i in to_eval]
            sub_params = [decoded_params[i] for i in to_eval]
            sub_fits = self.eval_func(sub_params, sub_nets, generation=self.current_gen)
            for idx, fit in zip(to_eval, sub_fits):
                fits[idx] = fit
                self.eval_cache[keys[idx]] = fit
        fit_array = np.array(fits, dtype=float)
        self.fitnesses = fit_array
        if self.num_objectives is None:
            self.num_objectives = fit_array.shape[1]
        return self.fitnesses

    @staticmethod
    def dominates(a, b):
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
        obj_a = np.array([-a[0]] + list(a[1:]))
        obj_b = np.array([-b[0]] + list(b[1:]))
        return np.all(obj_a <= obj_b) and np.any(obj_a < obj_b)

    def fast_nondominated_sort(self, fits):
        """
        Perform non-dominated sorting on a set of fitness vectors.

        Args:
            fits (np.ndarray): Array of shape (N, M) for N individuals and M objectives.

        Returns:
            List[List[int]]: A list of fronts, each a list of individual indices.
                Fronts[0] is the first Pareto front, etc.
        """
        N = len(fits)
        dominated = [set() for _ in range(N)]
        dom_count = np.zeros(N, dtype=int)
        fronts = [[]]
        for p in range(N):
            for q in range(N):
                if self.dominates(fits[p], fits[q]):
                    dominated[p].add(q)
                elif self.dominates(fits[q], fits[p]):
                    dom_count[p] += 1
            if dom_count[p] == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated[p]:
                    dom_count[q] -= 1
                    if dom_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
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

    def environmental_selection(self, pop, fits):
        """
        Select the next generation population by Pareto rank and crowding.

        Fills new population front by front until population_size is reached,
        using crowding distance to select within a partial front.

        Args:
            pop (np.ndarray): Combined parent+offspring population.
            fits (np.ndarray): Corresponding fitness array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Selected population and fitness arrays.
        """
        pop_size = self.population_size
        M = fits.shape[1]
        fronts = self.fast_nondominated_sort(fits)
        # Preallocate result arrays
        new_pop = np.empty((pop_size, pop.shape[1]), dtype=pop.dtype)
        new_fits = np.empty((pop_size, M), dtype=float)
        count = 0
        for front in fronts:
            front_size = len(front)
            # If entire front fits
            if count + front_size <= pop_size:
                new_pop[count:count+front_size] = pop[front]
                new_fits[count:count+front_size] = fits[front]
                count += front_size
            else:
                rem = pop_size - count
                if rem > 0:
                    # Compute crowding distance and select top rem individuals
                    cd = self.crowding_distance(fits, front)
                    sel_indices = np.argsort(cd)[-rem:]
                    chosen = [front[i] for i in sel_indices]
                    new_pop[count:count+rem] = pop[chosen]
                    new_fits[count:count+rem] = fits[chosen]
                # Either way, stop filling
                break
        return new_pop, new_fits
    
    def tournament_select(self, pop, fits, rank, crowd):
        """
        Select one individual via binary tournament on rank and crowding distance.

        Args:
            pop (np.ndarray): population array.
            fits (np.ndarray): fitness array (not used here).
            rank (np.ndarray): Pareto rank per individual.
            crowd (np.ndarray): crowding distance per individual.

        Returns:
            np.ndarray: selected individual.
        """
        i, j = np.random.choice(len(pop), 2, replace=False)
        if rank[i] < rank[j]:
            return pop[i]
        if rank[j] < rank[i]:
            return pop[j]
        return pop[i] if crowd[i] > crowd[j] else pop[j]
        i, j = np.random.choice(len(pop), 2, replace=False)
        if rank[i] < rank[j]: return pop[i]
        if rank[j] < rank[i]: return pop[j]
        return pop[i] if crowd[i] > crowd[j] else pop[j]
    
    def generate_offspring(self):
        """
        Create offspring population using tournament selection, crossover, and mutation.

        Respects Pareto rank and crowding distance in tournament selection.
        Preallocates offspring array for performance.
        """
        pop_size = self.population_size
        gene_len = self.population.shape[1]
        new_pop = np.empty((pop_size, gene_len), dtype=self.population.dtype)
        fronts = self.fast_nondominated_sort(self.fitnesses)
        rank = np.empty(len(self.population), dtype=int)
        crowd = np.zeros(len(self.population))
        for r, fr in enumerate(fronts):
            cd = self.crowding_distance(self.fitnesses, fr)
            for i, idx in enumerate(fr):
                rank[idx] = r
                crowd[idx] = cd[i]
        i = 0
        while i < pop_size:
            p1 = self.tournament_select(self.population, self.fitnesses, rank, crowd)
            p2 = self.tournament_select(self.population, self.fitnesses, rank, crowd)
            if np.random.rand() < self.crossover_rate:
                c1, c2 = self.crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            new_pop[i] = self.mutate(c1)
            if i + 1 < pop_size:
                new_pop[i+1] = self.mutate(c2)
            i += 2
        self.population = new_pop

    def update_global_pareto_front(self):
        """
        Update the global Pareto buffer by combining history with current pop,
        computing non-dominated fronts, and filtering front 0 by crowding.

        Returns:
            tuple: (all_pop, all_fit, all_ids, fronts) for history recording.
        """
        curr_ids = [f"{self.current_gen}_{i}" for i in range(len(self.population))]
        if self.pareto_global_population is None:
            all_pop, all_fit, all_ids = self.population.copy(), self.fitnesses.copy(), curr_ids
        else:
            all_pop = np.vstack([self.pareto_global_population, self.population])
            all_fit = np.vstack([self.pareto_global_fitnesses, self.fitnesses])
            all_ids = self.pareto_global_ids + curr_ids
        fronts = self.fast_nondominated_sort(all_fit)
        idx0 = fronts[0]
        pop0, fit0 = all_pop[idx0], all_fit[idx0]
        ids0 = [all_ids[i] for i in idx0]
        cd = self.crowding_distance(fit0, list(range(len(pop0))))
        mask = np.isinf(cd) | (cd > 0)
        self.pareto_global_population = pop0[mask]
        self.pareto_global_fitnesses = fit0[mask]
        self.pareto_global_ids = [ids0[i] for i, keep in enumerate(mask) if keep]
        return all_pop, all_fit, all_ids, fronts

    def record_global_fronts_history(self, fronts_info):
        """
        Persist all global Pareto fronts into the history dictionary and disk.

        Args:
            fronts_info (tuple): Output of update_global_pareto_front.
        """
        all_pop, all_fit, all_ids, fronts = fronts_info
        gen_global = {}
        for level, front in enumerate(fronts, start=1):
            gen_global[level] = [
                {"id": all_ids[i],
                    "accuracy": float(all_fit[i][0]),
                    "params": float(all_fit[i][1]),
                    "inference_time": float(all_fit[i][2])}
                for i in front
            ]
        self.fronts_history[self.current_gen] = gen_global
        with open(os.path.join(self.experiment_path, "pareto_history.pkl"), "wb") as f:
            pickle.dump(self.fronts_history, f)

    def go_next_gen(self):
        """
        Archive and record the current global Pareto front, then increment generation.

        - Updates global buffer via update_global_pareto_front.
        - Records full Pareto fronts history.
        - Calls delete_old_dirs_v2 to archive front 0 only.
        - Preserves gen0 archive on first call.
        """
        fronts_info = self.update_global_pareto_front()
        self.record_global_fronts_history(fronts_info)
        delete_old_dirs_v2(
            self.experiment_path,
            self.current_gen,
            keep_ids=self.pareto_global_ids.copy()
        )
        if self.current_gen == 1:
            delete_old_dirs_v2(self.experiment_path, 0, keep_ids=self.pareto_global_ids.copy())
        self.current_gen += 1

    def evolve(self):
        """
        Run the evolutionary process for num_generations.

        Workflow per generation:
            1. Generate and evaluate offspring.
            2. Combine parents and offspring for environmental selection.
            3. Archive and record Pareto fronts via go_next_gen.
            4. Optionally terminate via early stopping.

        Returns:
            tuple: (pareto_global_population, pareto_global_fitnesses)
        """
        start = time.time()
        fits_old = self.evaluate_population()
        pop_old = self.population.copy()
        self.best_so_far = np.max(fits_old[:, 0])
        self.last_best_so_far = self.best_so_far
        self.current_gen = 1
        while self.current_gen < self.num_generations:
            self.generate_offspring()
            fits_new = self.evaluate_population()
            combined_pop = np.vstack([pop_old, self.population])
            combined_fits = np.vstack([fits_old, fits_new])
            self.population, self.fitnesses = self.environmental_selection(
                combined_pop, combined_fits)
            self.go_next_gen()
            pop_old, fits_old = self.population.copy(), self.fitnesses.copy()
            if self.check_early_stopping():
                break
        elapsed = time.time() - start
        self.logger.info("Total evolution time: %.2f seconds", elapsed)
        return self.pareto_global_population, self.pareto_global_fitnesses
