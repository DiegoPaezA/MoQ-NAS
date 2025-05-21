import os
import time
import pickle
import numpy as np
from ga import GA
from util import backup_cache, delete_old_dirs_v2

class NSGA2(GA):
    """
    Multi-objective extension of GA using NSGA-II.
    Tracks global Pareto fronts and history per generation.
    Archives only solutions in the global Pareto front.
    """
    def __init__(self, eval_func, experiment_path, log_file, log_level, data_file):
        super().__init__(eval_func, experiment_path, log_file, log_level, data_file)
        # Buffers for global Pareto front
        self.pareto_global_population = None
        self.pareto_global_fitnesses = None
        self.pareto_global_ids = []
        # History: generation -> {level: [records]}
        self.fronts_history = {}

    def evaluate_population(self):
        decoded_nets, decoded_params = self.decode_pop()
        raw_fits = self.eval_func(decoded_params, decoded_nets, generation=self.current_gen)
        self.fitnesses = np.array(raw_fits, dtype=float)
        return self.fitnesses

    @staticmethod
    def dominates(a, b):
        # maximize accuracy => minimize -accuracy
        obj_a = np.array([-a[0], a[1], a[2]])
        obj_b = np.array([-b[0], b[1], b[2]])
        return np.all(obj_a <= obj_b) and np.any(obj_a < obj_b)

    def fast_nondominated_sort(self, fits):
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
        size = len(front)
        dist = np.zeros(size)
        if size <= 2:
            return np.array([np.inf]*size)
        for m in range(fits.shape[1]):
            vals = fits[front, m]
            sorted_idx = np.argsort(vals)
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            fmin, fmax = vals[sorted_idx[0]], vals[sorted_idx[-1]]
            if fmax == fmin:
                continue
            for k in range(1, size-1):
                prev_i = sorted_idx[k-1]
                next_i = sorted_idx[k+1]
                dist[sorted_idx[k]] += (vals[next_i] - vals[prev_i])/(fmax - fmin)
        return dist

    def environmental_selection(self, pop, fits):
        fronts = self.fast_nondominated_sort(fits)
        new_pop, new_fits = [], []
        for front in fronts:
            if len(new_pop) + len(front) <= self.population_size:
                new_pop.extend(pop[front])
                new_fits.extend(fits[front])
            else:
                dist = self.crowding_distance(fits, front)
                order = np.argsort(dist)[::-1]
                slots = self.population_size - len(new_pop)
                chosen = [front[i] for i in order[:slots]]
                new_pop.extend(pop[chosen])
                new_fits.extend(fits[chosen])
                break
        return np.array(new_pop), np.array(new_fits)

    def tournament_select(self, pop, fits, rank, crowd):
        i, j = np.random.choice(len(pop), 2, replace=False)
        if rank[i] < rank[j]: return pop[i]
        if rank[j] < rank[i]: return pop[j]
        return pop[i] if crowd[i] > crowd[j] else pop[j]

    def generate_offspring(self):
        fronts = self.fast_nondominated_sort(self.fitnesses)
        rank = np.zeros(len(self.population), dtype=int)
        for r, front in enumerate(fronts):
            for idx in front:
                rank[idx] = r
        crowd = np.zeros(len(self.population))
        for front in fronts:
            cd = self.crowding_distance(self.fitnesses, front)
            for i, idx in enumerate(front): crowd[idx] = cd[i]
        new_pop = []
        while len(new_pop) < self.population_size:
            p1 = self.tournament_select(self.population, self.fitnesses, rank, crowd)
            p2 = self.tournament_select(self.population, self.fitnesses, rank, crowd)
            if np.random.rand() < self.crossover_rate:
                c1, c2 = self.crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            new_pop.extend([self.mutate(c1), self.mutate(c2)])
        self.population = np.array(new_pop[:self.population_size])

    def update_global_pareto_front(self):
        """
        Combines historical global Pareto buffer with current population,
        computes global fronts, applies crowding-distance filtering on front 0
        to remove overly crowded or duplicate solutions, and updates the
        buffer of global Pareto solutions.

        Returns:
            all_pop (np.ndarray): combined individuals
            all_fit (np.ndarray): combined fitnesses
            all_ids (List[str]): combined IDs
            fronts (List[List[int]]): all Pareto fronts
        """
        # 1) Build IDs for current population
        curr_ids = [f"{self.current_gen}_{i}" for i in range(len(self.population))]

        # 2) Combine with historical buffer
        if self.pareto_global_population is None:
            all_pop = self.population.copy()
            all_fit = self.fitnesses.copy()
            all_ids = curr_ids
        else:
            all_pop = np.vstack([self.pareto_global_population, self.population])
            all_fit = np.vstack([self.pareto_global_fitnesses, self.fitnesses])
            all_ids = self.pareto_global_ids + curr_ids

        # 3) Fast non-dominated sort on combined set
        fronts = self.fast_nondominated_sort(all_fit)

        # 4) Extract first front (non-dominated)
        idx0 = list(fronts[0])
        pop0 = all_pop[idx0]
        fit0 = all_fit[idx0]
        ids0 = [all_ids[i] for i in idx0]

        # 5) Compute crowding distance on front 0 to ensure diversity
        #    and remove overcrowded or duplicate points
        distances = self.crowding_distance(fit0, list(range(len(fit0))))
        # Keep extremes (infinite distance) or any point with positive crowding
        mask = np.isinf(distances) | (distances > 0)
        pop_filtered = pop0[mask]
        fit_filtered = fit0[mask]
        ids_filtered = [ids0[i] for i, keep in enumerate(mask) if keep]

        # 6) Update global Pareto buffer
        self.pareto_global_population = pop_filtered
        self.pareto_global_fitnesses = fit_filtered
        self.pareto_global_ids = ids_filtered

        return all_pop, all_fit, all_ids, fronts

    def record_global_fronts_history(self):
        # build combined pool and obtain all fronts
        all_pop, all_fit, all_ids, fronts = self.update_global_pareto_front()
        gen_global = {}
        for level, front in enumerate(fronts, start=1):
            gen_global[level] = [
                {"id": all_ids[idx],
                    "accuracy": float(all_fit[idx][0]),
                    "params": float(all_fit[idx][1]),
                    "inference_time": float(all_fit[idx][2]),
                    "individual": all_pop[idx].tolist()
                } for idx in front
            ]
        # record and persist
        self.fronts_history[self.current_gen] = gen_global
        self.save_history_pickle()

    def save_history_pickle(self, filename=None):
        if filename is None:
            filename = os.path.join(self.experiment_path, "pareto_history.pkl")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self.fronts_history, f)

    def go_next_gen(self):
        self.save_data()
        backup_cache(self.evaluated, file_path=self.experiment_path)
        # record all global fronts this generation
        self.record_global_fronts_history()
        # archive only front0 individuals
        keep_ids = self.pareto_global_ids.copy()
        delete_old_dirs_v2(self.experiment_path, self.current_gen, keep_ids=keep_ids)
        if self.current_gen == 1:
                delete_old_dirs_v2(self.experiment_path,0,keep_ids=keep_ids)
        self.current_gen += 1

    def evolve(self):
        start_time = time.time()
        self.logger.info("=== NSGA2 Evolution ===")
        # 0) Evaluar generación 0
        fits_old = self.evaluate_population()
        pop_old  = self.population.copy()
        
        self.best_so_far = np.max(fits_old[:, 0])
        self.last_best_so_far = self.best_so_far
        
        self.current_gen += 1
        while self.current_gen < self.num_generations:
            # 1) Generar y evaluar descendencia
            self.generate_offspring()            # produce self.population (hijos)
            fits_new = self.evaluate_population()

            # 2) Combinar padres e hijos y seleccionar
            combined_pop  = np.vstack([pop_old, self.population])
            combined_fits = np.vstack([fits_old, fits_new])
            self.population, self.fitnesses = self.environmental_selection(
                combined_pop, combined_fits
            )

            # 3) Avanzar flujo normal de NSGA2
            self.go_next_gen()

            # 4) Preparar para la siguiente iteración
            pop_old  = self.population.copy()
            fits_old = self.fitnesses.copy()

            if self.check_early_stopping(): break
            
        total_time = time.time() - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, _ = divmod(rem, 60)
        self.logger.info(f"Total evolution time: {hours} hours and {minutes} minutes")
        return self.pareto_global_population, self.pareto_global_fitnesses
