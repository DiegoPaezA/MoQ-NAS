import os
import time
import datetime
import numpy as np
from ga import GA

class NSGA2(GA):
    """
    Multi-objective extension of the GA class using the NSGA-II algorithm.
    Assumes eval_func returns a tuple (accuracy, num_params, inference_time) per individual.
    """
    def __init__(self, eval_func, experiment_path, log_file, log_level, data_file):
        super().__init__(eval_func, experiment_path, log_file, log_level, data_file)

    def evaluate_population(self):
        """
        Evaluate all individuals and store a (pop_size x 3) fitness array:
        [accuracy, num_params, inference_time]
        """
        decoded_nets, decoded_params = self.decode_pop()
        raw_fits = self.eval_func(decoded_params, decoded_nets, generation=self.current_gen)
        self.fitnesses = np.array(raw_fits, dtype=float)
        return self.fitnesses

    @staticmethod
    def dominates(a, b):
        """Return True if a dominates b (maximize accuracy, minimize params/time)."""
        # Convert to minimization objectives: (-accuracy, params, time)
        obj_a = np.array([-a[0], a[1], a[2]])
        obj_b = np.array([-b[0], b[1], b[2]])
        return np.all(obj_a <= obj_b) and np.any(obj_a < obj_b)

    def fast_nondominated_sort(self, fits):
        N = len(fits)
        S = [set() for _ in range(N)]
        n = np.zeros(N, dtype=int)
        fronts = [[]]
        for p in range(N):
            for q in range(N):
                if self.dominates(fits[p], fits[q]):
                    S[p].add(q)
                elif self.dominates(fits[q], fits[p]):
                    n[p] += 1
            if n[p] == 0:
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        return fronts[:-1]

    def crowding_distance(self, fits, front):
        """
        Compute crowding distance for a given Pareto front.
        """
        size = len(front)
        dist = np.zeros(size)
        if size <= 2:
            # Both boundary points get infinite distance
            return np.array([np.inf]*size)
        # For each objective
        for obj in range(fits.shape[1]):
            vals = fits[front, obj]
            sorted_idx = np.argsort(vals)
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            fmin, fmax = vals[sorted_idx[0]], vals[sorted_idx[-1]]
            if fmax == fmin:
                continue
            for k in range(1, size-1):
                i = sorted_idx[k]
                prev_i = sorted_idx[k-1]
                next_i = sorted_idx[k+1]
                dist[i] += (vals[next_i] - vals[prev_i]) / (fmax - fmin)
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
        # Compare Pareto rank
        if rank[i] < rank[j]:
            return pop[i]
        if rank[j] < rank[i]:
            return pop[j]
        # Same rank -> crowding distance
        return pop[i] if crowd[i] > crowd[j] else pop[j]

    def generate_offspring(self):
        new_pop = []
        # Compute rank and crowding for current population
        fronts = self.fast_nondominated_sort(self.fitnesses)
        rank = np.empty(len(self.population), dtype=int)
        for r, front in enumerate(fronts):
            for idx in front:
                rank[idx] = r
        crowd = np.zeros(len(self.population))
        for front in fronts:
            cd = self.crowding_distance(self.fitnesses, front)
            for i, idx in enumerate(front):
                crowd[idx] = cd[i]
        # Generate offspring
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
            if self.check_early_stopping():
                break
        return self.population, self.fitnesses
