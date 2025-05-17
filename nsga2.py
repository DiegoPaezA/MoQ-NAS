import time
import numpy as np
from ga import GA
from util import backup_cache, delete_old_dirs_v2

class NSGA2(GA):
    """
    Multi-objective extension of the GA class using the NSGA-II algorithm.
    Assumes eval_func returns a tuple (accuracy, num_params, inference_time) per individual.
    """
    def __init__(self, eval_func, experiment_path, log_file, log_level, data_file):
        super().__init__(eval_func, experiment_path, log_file, log_level, data_file)

    def evaluate_population(self):
        decoded_nets, decoded_params = self.decode_pop()
        raw_fits = self.eval_func(decoded_params, decoded_nets, generation=self.current_gen)
        self.fitnesses = np.array(raw_fits, dtype=float)
        return self.fitnesses

    @staticmethod
    def dominates(a, b):
        """
        Determines whether solution 'a' dominates solution 'b' in a multi-objective 
        optimization context.

        In this implementation, the objectives are:
            - Maximizing accuracy (converted to minimization by negating accuracy)
            - Minimizing number of parameters
            - Minimizing inference time

        A solution 'a' is said to dominate solution 'b' if 'a' is no worse than 'b' 
        in all objectives, and strictly better in at least one objective.

        Args:
            a (tuple or list): Objective values for solution 'a' 
            in the form (accuracy, params, time).
            b (tuple or list): Objective values for solution 'b' 
            in the form (accuracy, params, time).

        Returns:
            bool: True if 'a' dominates 'b', False otherwise.
        """
        # Convert to minimization objectives: (-accuracy, params, time)
        obj_a = np.array([-a[0], a[1], a[2]])
        obj_b = np.array([-b[0], b[1], b[2]])
        return np.all(obj_a <= obj_b) and np.any(obj_a < obj_b)

    def fast_nondominated_sort(self, fits):
        """
        Perform the fast non-dominated sorting algorithm for multi-objective optimization.

        Args:
            fits (np.ndarray): A (N x M) array of fitness values, where N is 
            the number of individuals and M is the number of objectives.

        Returns:
            List[List[int]]: A list of fronts, where each front is a list of indices 
            corresponding to individuals in that front. The first front contains the 
            indices of non-dominated individuals, the second front contains the indices 
            of individuals dominated only by those in the first front, and so on.

        Notes:
            - This method assumes the existence of a `dominates` method that determines if 
            one individual dominates another.
            - The sorting is based on Pareto dominance: an individual p dominates q if p 
            is no worse than q in all objectives and better in at least one objective.
        """
        # fits: (num_individuals x 3) array of fitness values
        num_individuals = len(fits)
        dominated_sets = [set() for _ in range(num_individuals)]
        domination_counts = np.zeros(num_individuals, dtype=int)
        fronts = [[]]
        for p in range(num_individuals):
            for q in range(num_individuals):
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
        """
        Calculates the crowding distance for individuals in a given front 
        for multi-objective optimization.

        The crowding distance is a measure used in evolutionary algorithms (such as NSGA-II) 
        to estimate the density of solutions surrounding a particular solution in the objective 
        space. It helps to maintain diversity in the population by favoring individuals that are 
        in less crowded regions.

        Args:
            fits (np.ndarray): A 2D array of shape (population_size, num_objectives) containing 
            the fitness values of all individuals. front (list or np.ndarray): Indices of 
            individuals that belong to the current front.

        Returns:
            np.ndarray: An array of crowding distances for each individual in the front. Boundary 
            individuals are assigned an infinite distance.
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
        """
        Performs the environmental selection step in the NSGA-II algorithm.

        This method selects the next generation population from the current population
        and their fitness values using non-dominated sorting and crowding distance.
        It first fills the new population with entire non-dominated fronts until the
        population size limit is reached. If the last front cannot be fully accommodated,
        individuals from this front are selected based on their crowding distance.

        Args:
            pop (np.ndarray): The current population, typically an array of individuals.
            fits (np.ndarray): The fitness values corresponding to the population.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The selected new population and their fitness values,
            both as numpy arrays, with size equal to the specified population size.
        """
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
        """
        Selects an individual from the population using binary tournament selection 
        based on Pareto rank and crowding distance.

        Args:
            pop (list): The population of individuals to select from.
            fits (list or np.ndarray): Fitness values of the individuals 
            (not used in selection logic here).
            rank (list or np.ndarray): Pareto rank of each individual (lower is better).
            crowd (list or np.ndarray): Crowding distance of each individual (higher is better).

        Returns:
            object: The selected individual from the population.
        """
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

    def go_next_gen(self):
        """
        Advances the evolutionary algorithm to the next generation.

        This method performs the following steps:
            1. Saves current metrics and evaluation cache.
            2. Computes the current Pareto front using non-dominated sorting.
            3. Archives/prunes individuals, keeping only those on the current Pareto front.
            4. Increments the generation counter.

        Side Effects:
            - Writes data to disk for metrics and cache.
            - Deletes directories of non-Pareto individuals from the experiment path.
            - Updates the current generation counter.
        """
        # 1) save metrics & cache
        self.save_data()
        backup_cache(self.evaluated, file_path=self.experiment_path)

        # 2) compute current Pareto front
        fronts = self.fast_nondominated_sort(self.fitnesses)
        pareto = fronts[0]
        # build the keep_ids in the same "gen_idx" format
        keep_ids = [f"{self.current_gen}_{idx}" for idx in pareto]

        # 3) prune/archive exactly those front members
        delete_old_dirs_v2(self.experiment_path, self.current_gen, keep_ids=keep_ids)

        # 4) advance generation counter
        self.current_gen += 1

    def evolve(self):
        """
        Runs the evolutionary process for a specified number of generations or until 
        early stopping is triggered.

        The method performs the following steps in each generation:
            1. Evaluates the current population (parents).
            2. Generates offspring and evaluates them.
            3. Combines parents and offspring populations and their fitnesses.
            4. Selects the next generation population using environmental selection.
            5. Archives the current Pareto front and advances to the next generation.
            6. Checks for early stopping criteria.

        Tracks and logs the total evolution time upon completion.

        Returns:
            tuple: The final population and their corresponding fitnesses.
        """
        start_time = time.time()
        while self.current_gen < self.num_generations:
            # Evaluate parents
            fits_old = self.evaluate_population()
            pop_old  = self.population.copy()
            # Create offspring & evaluate them
            self.generate_offspring()
            fits_new = self.evaluate_population()
            # Combine & select
            combined_pop  = np.vstack([pop_old, self.population])
            combined_fits = np.vstack([fits_old, fits_new])
            self.population, self.fitnesses = self.environmental_selection(
                combined_pop, combined_fits
            )
            # Archive Pareto front, advance
            self.go_next_gen()

            if self.check_early_stopping():
                break
        total_time = time.time() - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, _ = divmod(rem, 60)
        self.logger.info("Total evolution time: %s hours and %s minutes", hours, minutes)

        return self.population, self.fitnesses
