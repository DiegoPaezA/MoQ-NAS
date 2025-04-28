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
        # Decode networks
        decoded_nets = self.decode_pop(None, self.population)
        # Evaluate them
        raw_fits = self.eval_func(None, decoded_nets, generation=self.current_gen)
        # Convert list of tuples to numpy array
        self.fitnesses = np.array(raw_fits, dtype=float)
        return self.fitnesses

    @staticmethod
    def dominates(a, b):
        """Return True if a dominates b (minimize all objectives except accuracy)."""
        # We want to maximize accuracy but minimize params and time.
        # Transform accuracy into a minimizable objective: obj0 = -accuracy
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
        m = fits.shape[1]
        dist = np.zeros(len(front))
        for obj in range(m):
            vals = fits[front, obj]
            idx = np.argsort(vals)
            dist[idx[0]] = dist[idx[-1]] = np.inf
            fmin, fmax = vals[idx[0]], vals[idx[-1]]
            if fmax == fmin:
                continue
            for i in idx[1:-1]:
                dist[i] += (vals[i+1] - vals[i-1]) / (fmax - fmin)
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
                idx = np.argsort(dist)[::-1]
                slots = self.population_size - len(new_pop)
                chosen = [front[i] for i in idx[:slots]]
                new_pop.extend(pop[chosen])
                new_fits.extend(fits[chosen])
                break
        return np.array(new_pop), np.array(new_fits)

    def tournament_select(self, pop, fits, rank, crowd):
        i, j = np.random.choice(len(pop), 2, replace=False)
        # compare rank first
        if rank[i] < rank[j]: return pop[i]
        if rank[j] < rank[i]: return pop[j]
        # same rank -> crowding distance
        if crowd[i] > crowd[j]: return pop[i]
        return pop[j]

    def generate_offspring(self):
        new_pop = []
        # First, calculate rank & crowd for current pop
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
            # Evaluate current pop -> self.fitnesses (pop_size x 3)
            fits_old = self.evaluate_population()
            pop_old = self.population.copy()

            # Create offspring
            self.generate_offspring()
            fits_new = self.evaluate_population()

            # Combine parents + offspring
            combined_pop = np.vstack([pop_old, self.population])
            combined_fits = np.vstack([fits_old, fits_new])
            # Environmental selection
            self.population, self.fitnesses = self.environmental_selection(
                combined_pop, combined_fits
            )

            # Usual logging/backups
            self.go_next_gen()

            if self.check_early_stopping():
                break

        return self.population, self.fitnesses
