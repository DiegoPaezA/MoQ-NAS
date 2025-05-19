import os
import time
import json
import numpy as np
from ga import GA
from util import backup_cache, delete_old_dirs_v2

class NSGA2(GA):
    """
    Multi-objective extension of the GA class using the NSGA-II algorithm.
    Assumes eval_func returns a tuple (accuracy, num_params, inference_time) per individual.
    Maintains a global Pareto front across all generations and archives only those.
    """
    def __init__(self, eval_func, experiment_path, log_file, log_level, data_file):
        super().__init__(eval_func, experiment_path, log_file, log_level, data_file)
        # Initialize global Pareto front buffers lazily
        self.pareto_global_population = None       # Will be set after first population evaluation
        self.pareto_global_fitnesses = None        # Will be set accordingly
        self.pareto_global_ids = []                # List of "gen_idx" strings

    def evaluate_population(self):
        decoded_nets, decoded_params = self.decode_pop()
        raw_fits = self.eval_func(decoded_params, decoded_nets, generation=self.current_gen)
        self.fitnesses = np.array(raw_fits, dtype=float)
        return self.fitnesses

    @staticmethod
    def dominates(a, b):
        # Convert maximize accuracy to minimization of -accuracy
        obj_a = np.array([-a[0], a[1], a[2]])
        obj_b = np.array([-b[0], b[1], b[2]])
        return np.all(obj_a <= obj_b) and np.any(obj_a < obj_b)

    def fast_nondominated_sort(self, fits):
        N = len(fits)
        dominated_sets = [set() for _ in range(N)]
        domination_counts = np.zeros(N, dtype=int)
        fronts = [[]]
        for p in range(N):
            for q in range(N):
                if self.dominates(fits[p], fits[q]):
                    dominated_sets[p].add(q)
                elif self.dominates(fits[q], fits[p]):
                    domination_counts[p] += 1
            if domination_counts[p] == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated_sets[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        return fronts[:-1]

    def crowding_distance(self, fits, front):
        size = len(front)
        dist = np.zeros(size)
        if size <= 2:
            return np.array([np.inf] * size)
        for obj in range(fits.shape[1]):
            vals = fits[front, obj]
            sorted_idx = np.argsort(vals)
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            fmin, fmax = vals[sorted_idx[0]], vals[sorted_idx[-1]]
            if fmax == fmin:
                continue
            for k in range(1, size - 1):
                dist[sorted_idx[k]] += (vals[sorted_idx[k+1]] - vals[sorted_idx[k-1]]) / (fmax - fmin)
        return dist

    def environmental_selection(self, pop, fits):
        fronts = self.fast_nondominated_sort(fits)
        new_pop, new_fits = [], []
        for front in fronts:
            if len(new_pop) + len(front) <= self.population_size:
                new_pop.extend(pop[front])
                new_fits.extend(fits[front])
            else:
                distances = self.crowding_distance(fits, front)
                idx_sorted = np.argsort(distances)[::-1]
                slots = self.population_size - len(new_pop)
                chosen = [front[i] for i in idx_sorted[:slots]]
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
        new_pop = []
        fronts = self.fast_nondominated_sort(self.fitnesses)
        rank = np.empty(len(self.population), dtype=int)
        for r, front in enumerate(fronts):
            for idx in front:
                rank[idx] = r
        crowd = np.zeros(len(self.population))
        for front in fronts:
            cd = self.crowding_distance(self.fitnesses, front)
            for i, idx in enumerate(front): crowd[idx] = cd[i]
        while len(new_pop) < self.population_size:
            p1 = self.tournament_select(self.population, self.fitnesses, rank, crowd)
            p2 = self.tournament_select(self.population, self.fitnesses, rank, crowd)
            if np.random.rand() < self.crossover_rate:
                c1, c2 = self.crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            new_pop.append(self.mutate(c1))
            new_pop.append(self.mutate(c2))
        self.population = np.array(new_pop[:self.population_size])

    def update_global_pareto_front(self):
        # IDs for current population
        curr_ids = [f"{self.current_gen}_{i}" for i in range(len(self.population))]
        # First update: no prior global front
        if self.pareto_global_population is None:
            self.pareto_global_population = self.population.copy()
            self.pareto_global_fitnesses = self.fitnesses.copy()
            self.pareto_global_ids = curr_ids
        else:
            # Combine historical and current
            all_pop = np.vstack([self.pareto_global_population, self.population])
            all_fit = np.vstack([self.pareto_global_fitnesses, self.fitnesses])
            all_ids = self.pareto_global_ids + curr_ids
            # Compute new global front
            fronts = self.fast_nondominated_sort(all_fit)
            idx_front = fronts[0]
            self.pareto_global_population = all_pop[idx_front]
            self.pareto_global_fitnesses = all_fit[idx_front]
            self.pareto_global_ids = [all_ids[i] for i in idx_front]

    def save_global_pareto_front(self, filename=None):
        if filename is None:
            filename = os.path.join(self.experiment_path, "pareto_global_front.json")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        results = []
        for id_str, fit, indiv in zip(self.pareto_global_ids, self.pareto_global_fitnesses, self.pareto_global_population):
            results.append({
                "id": id_str,
                "accuracy": float(fit[0]),
                "params": float(fit[1]),
                "inference_time": float(fit[2]),
                "individual": indiv.tolist()
            })
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Global Pareto front guardado en {filename}")

    def go_next_gen(self):
        # 1) save metrics & cache
        self.save_data()
        backup_cache(self.evaluated, file_path=self.experiment_path)
        # 2) update global Pareto
        self.update_global_pareto_front()
        self.save_global_pareto_front()
        # 3) archive only global Pareto front
        keep_ids = self.pareto_global_ids.copy()
        delete_old_dirs_v2(self.experiment_path, self.current_gen, keep_ids=keep_ids)
        # 4) advance generation counter
        self.current_gen += 1

    def evolve(self):
        start_time = time.time()
        while self.current_gen < self.num_generations:
            fits_old = self.evaluate_population()
            pop_old = self.population.copy()
            self.generate_offspring()
            fits_new = self.evaluate_population()
            combined_pop = np.vstack([pop_old, self.population])
            combined_fits = np.vstack([fits_old, fits_new])
            self.population, self.fitnesses = self.environmental_selection(combined_pop, combined_fits)
            self.go_next_gen()
            if self.check_early_stopping(): break
        elapsed = time.time() - start_time
        self.logger.info("Total evolution time: %.2f seconds", elapsed)
        return self.population, self.fitnesses
