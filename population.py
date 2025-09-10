""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Quantum population classes.

    Refactored QPopulationNetwork with new Elite Method for the Update of the Quantum Population
    - Added new crossover methods: HUX and Uniform.
    - Improved mutation strategies for better exploration of the search space.
    - Added Metrics to track the probabilities evolution. 
    Diego Páez Ardila - 2025
"""
import os
import csv
import numpy as np

from chromosome import QChromosomeParams, QChromosomeNetwork


class QPopulation(object):
    """ QNAS Population to be evolved. """

    def __init__(self, num_quantum_ind, repetition, update_quantum_rate):
        """ Initialize QPopulation.

        Args:
            num_quantum_ind: (int) number of quantum individuals.
            repetition: (int) ratio between the number of classic individuals in the classic
                population and the quantum individuals in the quantum population.
            update_quantum_rate: (float) probability that a quantum gene will be updated.
        """

        self.dtype = np.float64  # Type of quantum population arrays.

        self.chromosome = None
        self.current_pop = None
        self.current_pop_objs = None
        self.num_ind = num_quantum_ind

        self.repetition = repetition
        self.update_quantum_rate = update_quantum_rate

    def initialize_qpop(self):
        raise NotImplementedError('initialize_qpop() must be implemented in sub classes')

    def generate_classical(self):
        raise NotImplementedError('generate_classical() must be implemented in sub classes')

    def update_quantum(self, intensity):
        raise NotImplementedError('update_quantum() must be implemented in sub classes')


class QPopulationParams(QPopulation):
    """ QNAS Chromosomes for the hyperparameters to be evolved. """

    def __init__(self, num_quantum_ind, params_ranges, repetition, crossover_rate,
                update_quantum_rate):
        """ Initialize QPopulationParams.

        Args:
            num_quantum_ind: (int) number of quantum individuals.
            params_ranges: {'parameter_name': [parameter_lower_limit, parameter_upper_limit]}.
            repetition: (int) ratio between the number of classic individuals in the classic
                population and the quantum individuals in the quantum population.
            crossover_rate: (float) crossover rate.
            update_quantum_rate: (float) probability that a quantum gene will be updated.
        """

        super(QPopulationParams, self).__init__(num_quantum_ind, repetition,
                                                update_quantum_rate)

        self.tolerance = 1.e-15  # Tolerance to compare floating point

        self.lower = None
        self.upper = None
        self.crossover = crossover_rate

        self.chromosome = QChromosomeParams(params_ranges, self.dtype)

        self.initial_lower, self.initial_upper = self.chromosome.initialize_qgenes()

        self.initialize_qpop()

    def initialize_qpop(self):
        """ Initialize quantum population with *self.num_ind* individuals. """

        self.lower = np.tile(self.initial_lower, (self.num_ind, 1))
        self.upper = np.tile(self.initial_upper, (self.num_ind, 1))

    def classic_crossover(self, new_pop, distance):
        """ Perform arithmetic crossover of the old classic population with the new one.

        Args:
            new_pop: float numpy array representing the new classical population.
            distance: (float) random distance for arithmetic crossover (range = [0, 1]).
        """

        mask = np.random.rand(self.num_ind * self.repetition, self.chromosome.num_genes)
        idx = np.where(mask <= self.crossover)
        new_pop[idx] = new_pop[idx] + (self.current_pop[idx] - new_pop[idx]) * distance

        return new_pop

    def generate_classical(self):
        """ Generate a specific number of classical individuals from the observation of quantum
            individuals. This number is equal to (*num_ind* x *repetition*).
        """

        random_numbers = np.random.rand(self.num_ind * self.repetition,
                                        self.chromosome.num_genes).astype(self.dtype)

        new_pop = random_numbers * np.tile(self.upper - self.lower, (self.repetition, 1)) \
            + np.tile(self.lower, (self.repetition, 1))

        return new_pop

    def update_quantum(self, intensity):
        """ Update self.lower and self.upper.

        Args:
            intensity: (float) value defining the maximum intensity of the update.
        """

        random = np.random.rand(self.num_ind, self.chromosome.num_genes)
        mask = np.where(random <= self.update_quantum_rate)

        max_genes = np.max(self.current_pop, axis=0)
        min_genes = np.min(self.current_pop, axis=0)
        diff = np.tile(max_genes - min_genes, (self.num_ind, 1))

        update = self.current_pop[mask] - self.lower[mask] - (diff[mask] / 2)
        self.lower[mask] += intensity * update

        update = self.current_pop[mask] - self.upper[mask] + (diff[mask] / 2)
        self.upper[mask] += intensity * update
        # Correct limits (truncate) if they get out of the initial boundaries
        for i in range(self.num_ind):
            idx = np.where(self.lower[i] - self.initial_lower < -self.tolerance)
            self.lower[i][idx] = self.initial_lower[idx]
            idx = np.where(self.upper[i] - self.initial_upper > self.tolerance)
            self.upper[i][idx] = self.initial_upper[idx]


class QPopulationNetwork(QPopulation):
    """QNAS Chromosomes for the networks to be evolved."""

    def __init__(self, num_quantum_ind: int, max_num_nodes: int, repetition: int, 
                update_quantum_rate: float, fn_list: list, initial_probs: list, 
                crossover_method: str = 'hux', elite_mode: str = "global_k", 
                k_elites: int = 5, pool_factor: int = 2, ema_beta: float = 0.7, 
                rank_weighting: bool = True, terminal_op_name: str = "no_op", 
                pool_op_name: str | list[str] = "pool", min_active_len: int = 5,
                truncate_after_noop: bool = True, avoid_consecutive_pool: bool = True,
                enforce_noop_in_update: bool = True, noop_max_prob: float = 0.90,
                noop_ramp_cap: bool = True, experiment_path: str = ""):
        """Initializes the QPopulationNetwork.

        Args:
            num_quantum_ind (int): Number of quantum individuals.
            max_num_nodes (int): Maximum number of nodes (genes) in a network.
            repetition (int): Ratio of classical to quantum individuals.
            update_quantum_rate (float): Base probability for updating a quantum gene.
            fn_list (list): List of possible functions (operations) for each node.
            initial_probs (list): Defines initial probabilities for each function. If
                empty, a uniform distribution is used.
            crossover_method (str, optional): Crossover method to use.
                Options are "hux" or "uniform". Defaults to 'hux'.
            elite_mode (str, optional): Strategy to build target distributions from elites.
                Options: "single", "global_k", "bootstrap_k", "moead_topk".
                Defaults to "global_k".
            k_elites (int, optional): Number of elites for building target distributions.
                Defaults to 5.
            pool_factor (int, optional): Multiplier for elite pool size in "bootstrap_k".
                Defaults to 2.
            ema_beta (float, optional): EMA factor for global elite distributions.
                Set to 0.0 to disable. Defaults to 0.7.
            rank_weighting (bool, optional): If True, weights elite contributions by
                inverse rank. Defaults to True.
            terminal_op_name (str, optional): The exact name of the terminal operation
                (e.g., "no_op"). Defaults to "no_op".
            pool_op_name (str | list[str], optional): Pattern(s) to identify pooling
                operations. Defaults to "pool".
            min_active_len (int, optional): Minimum number of active nodes before a
                terminal operation is allowed. Defaults to 5.
            truncate_after_noop (bool, optional): If True, all subsequent nodes after the
                first terminal op are set to be terminal ops. Defaults to True.
            avoid_consecutive_pool (bool, optional): If True, prevents two pooling
                operations from being sampled in a row. Defaults to True.
            enforce_noop_in_update (bool, optional): If True, applies structural rules
                for the terminal op during quantum updates. Defaults to True.
            noop_max_prob (float, optional): Maximum probability allowed for the
                terminal operation after `min_active_len`. Defaults to 0.90.
            noop_ramp_cap (bool, optional): If True, linearly increases the `noop_max_prob`
                from 0 to its max value between `min_active_len` and the last node.
                Defaults to True.
            experiment_path (str, optional): Path to the experiment directory for
                saving metrics. Defaults to "".
        """
        super(QPopulationNetwork, self).__init__(num_quantum_ind, repetition,
                                                update_quantum_rate)
        self.probabilities = None

        self.chromosome = QChromosomeNetwork(max_num_nodes, fn_list, self.dtype)

        self.max_update = 0.05
        self.max_prob = 0.90
        self.min_prob = max(1e-8, 0.01 / self.chromosome.num_functions)

        self.elite_mode = elite_mode
        self.k_elites = k_elites
        self.pool_factor = pool_factor
        self.ema_beta = ema_beta
        self.rank_weighting = rank_weighting

        self._U_total = None

        # Update-rate schedule (probability a (ind,gene) gets updated)
        self.rate_start = max(0.2, self.update_quantum_rate)
        self.rate_boost = 0.40
        self.rate_end = self.update_quantum_rate

        self.metrics_output = os.path.join(experiment_path, "qpop_update", "metrics_output.csv")
        os.makedirs(os.path.dirname(self.metrics_output), exist_ok=True)
        self._last_dump_idx = 0

        self.initial_probs = self.chromosome.initialize_qgenes(initial_probs=initial_probs)
        self.crossover_method = crossover_method

        self.objective_names = None
        self.objective_sense = None
        self.num_objectives = None

        self.moead_q_low = 0.30
        self.moead_q_high = 0.90
        self.topP_mult = 5

        self._ref_dirs = None
        self._ind_to_dir = None
        self.ref_dir_method = "das-dennis"

        self.fn_list = list(fn_list)
        self.terminal_op_name = terminal_op_name
        self.min_active_len = int(min_active_len)
        self.truncate_after_noop = bool(truncate_after_noop)
        self.avoid_consecutive_pool = bool(avoid_consecutive_pool)

        try:
            self.no_op_id = int(self.fn_list.index(self.terminal_op_name))
        except ValueError:
            self.no_op_id = None

        patterns = pool_op_name if isinstance(pool_op_name, (list, tuple, set)) else [pool_op_name]
        patterns = [str(p) for p in patterns]
        self.pool_ids = [i for i, name in enumerate(self.fn_list)
                         if any(pat in str(name) for pat in patterns)]

        self.enforce_noop_in_update = bool(enforce_noop_in_update)
        self.noop_max_prob = float(noop_max_prob)
        self.noop_ramp_cap = bool(noop_ramp_cap)

        self.initialize_qpop()
        self._init_metrics_min()

    def set_schedule_total_updates(self, total_updates: int):
        """Sets the total number of updates for scheduling learning rates.

        Args:
            total_updates (int): The total expected number of calls to `update_quantum`.
        """
        self._U_total = max(1, int(total_updates))

    def initialize_qpop(self):
        """Initializes the quantum population probabilities.

        Creates the `self.probabilities` array with shape
        (num_ind, num_genes, num_functions) and applies the initial static
        mask for the terminal operation if enabled.
        """
        self.probabilities = np.tile(self.initial_probs, (self.num_ind, self.chromosome.num_genes, 1))
        if getattr(self, "enforce_noop_in_update", True):
            self._enforce_noop_static_mask_global()

    def generate_classical(self) -> np.ndarray:
        """Generates a classical population from quantum probabilities.

        This method samples from the quantum probability distributions to create
        a set of classical individuals (architectures). It enforces structural
        rules during generation, such as:
        - Truncating architectures after a terminal operation.
        - Preventing consecutive pooling operations.

        Returns:
            np.ndarray: A 2D array of shape (N, L) representing the classical
                        population, where N is the number of individuals and L
                        is the number of genes.
        """
        F = self.chromosome.num_functions
        L = self.chromosome.num_genes
        N = self.num_ind * self.repetition

        new_pop = np.zeros((N, L), dtype=np.int32)
        base_prob = np.tile(self.probabilities, (self.repetition, 1, 1))

        def _renorm_or_fallback(p, base=None):
            """Safely renormalizes a probability vector, with fallbacks."""
            s = p.sum()
            if s > 1e-8:
                return p / s
            if base is not None:
                sb = base.sum()
                if sb > 1e-8:
                    return base / sb
            return np.full_like(p, 1.0 / len(p), dtype=float)

        for ind in range(N):
            prev_was_pool = False
            truncated = False
            for node in range(L):
                if truncated:
                    new_pop[ind, node] = self.no_op_id if self.no_op_id is not None else 0
                    continue

                p = base_prob[ind, node, :].astype(float, copy=True)

                if (self.no_op_id is not None) and (node < self.min_active_len):
                    p[self.no_op_id] = 0.0

                if self.avoid_consecutive_pool and prev_was_pool and self.pool_ids:
                    p[self.pool_ids] = 0.0

                p = _renorm_or_fallback(p, base=p)
                choice = np.random.choice(F, p=p)
                new_pop[ind, node] = choice
                prev_was_pool = (choice in self.pool_ids)

                if (self.truncate_after_noop and
                        (self.no_op_id is not None) and
                        (choice == self.no_op_id) and
                        (node >= self.min_active_len - 1)):
                    if node + 1 < L:
                        new_pop[ind, node + 1: L] = self.no_op_id
                    truncated = True
        return new_pop

    def _noop_cap_for_gene(self, j: int) -> float:
        """Calculates the maximum allowed probability for the terminal op at a given gene.

        Args:
            j (int): The index of the gene (node).

        Returns:
            float: The probability cap, ranging from 0.0 to `self.noop_max_prob`.
                    Returns 0.0 for genes before `min_active_len`.
        """
        if getattr(self, "no_op_id", None) is None:
            return 1.0
        if j < int(self.min_active_len):
            return 0.0
        if not getattr(self, "noop_ramp_cap", True):
            return float(getattr(self, "noop_max_prob", 0.90))
        L = self.chromosome.num_genes
        maxcap = float(getattr(self, "noop_max_prob", 0.90))
        if L <= self.min_active_len:
            return maxcap
        alpha = (j - self.min_active_len + 1) / float(L - self.min_active_len + 1)
        return float(min(maxcap, max(0.0, alpha * maxcap)))

    def _apply_noop_caps_rows(self, P_rows: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Applies terminal op caps to selected rows of a probability matrix.

        This is used before the quantum update to enforce architectural constraints
        on the current probabilities.

        Args:
            P_rows (np.ndarray): Probability vectors to be modified.
            rows (np.ndarray): Indices of the quantum individuals.
            cols (np.ndarray): Indices of the genes corresponding to the rows.

        Returns:
            np.ndarray: The modified and renormalized probability vectors.
        """
        if (getattr(self, "no_op_id", None) is None) or (not getattr(self, "enforce_noop_in_update", True)):
            return P_rows
        P = P_rows.copy()
        F = P.shape[1]
        eps = 1e-12
        for i in range(P.shape[0]):
            j = int(cols[i])
            cap = self._noop_cap_for_gene(j)
            if cap <= 0.0:
                P[i, self.no_op_id] = 0.0
            else:
                P[i, self.no_op_id] = min(P[i, self.no_op_id], cap)
            s = P[i].sum()
            if s <= eps:
                P[i].fill(1.0 / F)
            else:
                P[i] /= s
        return P

    def _mask_noop_in_qrows(self, q_rows: np.ndarray, rows: np.ndarray, cols: np.ndarray, prior_P: np.ndarray) -> np.ndarray:
        """Masks the terminal op in target distribution rows (q_rows).

        Ensures that the "winner" operation in the target distribution `q` cannot
        be the terminal op if it is forbidden at that position.

        Args:
            q_rows (np.ndarray): Target probability distributions.
            rows (np.ndarray): Indices of the quantum individuals.
            cols (np.ndarray): Indices of the genes.
            prior_P (np.ndarray): The original probability distributions, used as a fallback.

        Returns:
            np.ndarray: The modified and renormalized target distributions.
        """
        if (getattr(self, "no_op_id", None) is None) or (not getattr(self, "enforce_noop_in_update", True)):
            return q_rows
        Q = q_rows.copy()
        eps = 1e-12
        for i in range(Q.shape[0]):
            j = int(cols[i])
            cap = self._noop_cap_for_gene(j)
            if cap <= 0.0:
                Q[i, self.no_op_id] = 0.0
            s = Q[i].sum()
            if s <= eps:
                row = prior_P[i]
                row = row / max(row.sum(), eps)
                Q[i] = row
            else:
                Q[i] /= s
        return Q

    def _enforce_noop_static_mask_global(self):
        """Globally enforces that the terminal op probability is zero before `min_active_len`.

        This method is called after initialization and updates to ensure the entire
        quantum population respects this hard constraint.
        """
        if getattr(self, "no_op_id", None) is None or (self.min_active_len <= 0):
            return
        P = self.probabilities
        P[:, :self.min_active_len, self.no_op_id] = 0.0
        sums = P[:, :self.min_active_len, :].sum(axis=-1, keepdims=True)
        sums = np.where(sums <= 1e-12, 1e-12, sums)
        P[:, :self.min_active_len, :] = P[:, :self.min_active_len, :] / sums

    def hux_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Performs Half-Uniform Crossover (HUX) on two parents.

        HUX identifies differing genes and swaps exactly half of them.

        Args:
            parent1 (np.ndarray): The first parent chromosome.
            parent2 (np.ndarray): The second parent chromosome.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the two offspring.
        """
        differing_indices = np.where(parent1 != parent2)[0]
        num_swaps = len(differing_indices) // 2
        swap_indices = np.random.choice(differing_indices, num_swaps, replace=False)
        offspring1, offspring2 = parent1.copy(), parent2.copy()
        offspring1[swap_indices], offspring2[swap_indices] = parent2[swap_indices], parent1[swap_indices]
        return offspring1, offspring2

    def uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Performs Uniform Crossover on two parents.

        A binary mask is created, and genes are swapped between parents
        where the mask is True.

        Args:
            parent1 (np.ndarray): The first parent chromosome.
            parent2 (np.ndarray): The second parent chromosome.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the two offspring.
        """
        chromosome_length = len(parent1)
        crossover_mask = np.random.randint(0, 2, size=chromosome_length).astype(bool)
        offspring1, offspring2 = parent1.copy(), parent2.copy()
        offspring1[crossover_mask], offspring2[crossover_mask] = parent2[crossover_mask], parent1[crossover_mask]
        return offspring1, offspring2

    def apply_crossover(self, best_current_pop: np.ndarray, new_pop: np.ndarray) -> np.ndarray:
        """Applies the selected crossover method to generate offspring.

        It pairs individuals from `best_current_pop` and `new_pop` as parents.

        Args:
            best_current_pop (np.ndarray): Best individuals from the current population.
            new_pop (np.ndarray): Individuals from the newly generated population.

        Returns:
            np.ndarray: A population of offspring.
        """
        offspring = []
        for parent1, parent2 in zip(best_current_pop, new_pop):
            if self.crossover_method == 'hux':
                child1, child2 = self.hux_crossover(parent1, parent2)
            elif self.crossover_method == 'uniform':
                child1, child2 = self.uniform_crossover(parent1, parent2)
            else:
                raise ValueError(f"Unknown crossover method: {self.crossover_method}")
            offspring.extend([child1, child2])
        return np.array(offspring[:len(new_pop)])

    def set_crossover_method(self, method: str):
        """Sets the crossover method.

        Args:
            method (str): The name of the method ('hux' or 'uniform').

        Raises:
            ValueError: If the method is not supported.
        """
        if method in ['hux', 'uniform']:
            self.crossover_method = method
        else:
            raise ValueError(f"Unknown crossover method: {method}")

    def _init_metrics_min(self):
        """Initializes the metrics dictionary and tracking variables."""
        self.metrics = {
            "epoch": [], "update_idx": [], "quantum_update_rate": [],
            "max_update": [], "entropy_mean": [], "kl_mean": [],
            "frac_onehot_0p9": [],
        }
        self._last_P = None
        self._update_counter = 0

    def _log_update_metrics_min(self, epoch_idx: int | None = None):
        """Logs key metrics about the quantum population's state after an update.

        Args:
            epoch_idx (int | None, optional): The current epoch index. Defaults to None.
        """
        P = self.probabilities
        H_mean = float(self._entropy_rows(P).mean())
        if self._last_P is not None:
            KL_mean = float(self._kl_rows(
                P.reshape(-1, P.shape[-1]),
                self._last_P.reshape(-1, self._last_P.shape[-1])
            ).mean())
        else:
            KL_mean = float("nan")
        frac_oh = self._frac_onehot(P, thr=0.9)

        self.metrics["epoch"].append(int(epoch_idx) if epoch_idx is not None else len(self.metrics["epoch"]))
        self.metrics["update_idx"].append(self._update_counter)
        self.metrics["quantum_update_rate"].append(self.update_quantum_rate)
        self.metrics["max_update"].append(self.max_update)
        self.metrics["entropy_mean"].append(H_mean)
        self.metrics["kl_mean"].append(KL_mean)
        self.metrics["frac_onehot_0p9"].append(frac_oh)

        self._last_P = P.copy()
        self._update_counter += 1

    def save_metrics_csv(self, path_csv: str, overwrite: bool = False):
        """Saves the logged metrics to a CSV file.

        Args:
            path_csv (str): The path to the output CSV file.
            overwrite (bool, optional): If True, overwrites the file. Otherwise, appends.
                                        Defaults to False.
        """
        cols = ["epoch", "update_idx", "quantum_update_rate", "max_update",
                "entropy_mean", "kl_mean", "frac_onehot_0p9"]
        total = len(self.metrics["update_idx"])
        start = 0 if overwrite or not os.path.exists(path_csv) else getattr(self, "_last_dump_idx", 0)
        if start >= total:
            return

        mode = "w" if overwrite or not os.path.exists(path_csv) else "a"
        write_header = (mode == "w")
        with open(path_csv, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if write_header:
                w.writeheader()
            for i in range(start, total):
                row = {k: self.metrics[k][i] for k in cols}
                w.writerow(row)
        self._last_dump_idx = total

    def _das_dennis(self, M: int, H: int) -> np.ndarray:
        """Generates uniformly spaced reference directions on a simplex. (Das-Dennis method)

        Args:
            M (int): Number of objectives.
            H (int): Number of divisions along each objective axis.

        Returns:
            np.ndarray: An array of reference direction vectors.
        """
        if H == 0:
            return np.array([[1.0]]) if M == 1 else np.empty((0, M), dtype=float)

        def gen_partitions(n_rem, k_rem):
            if k_rem == 1:
                yield (n_rem,)
                return
            for i in range(n_rem + 1):
                for p in gen_partitions(n_rem - i, k_rem - 1):
                    yield (i,) + p

        partitions = list(gen_partitions(H, M))
        return np.array(partitions, dtype=float) / float(H)

    def _make_ref_directions(self, M: int, D: int) -> np.ndarray:
        """Creates reference direction vectors for MOEA/D.

        Tries to use the systematic Das-Dennis method first, falling back to a
        Dirichlet distribution if a suitable number of points cannot be generated.

        Args:
            M (int): Number of objectives.
            D (int): Number of directions to generate.

        Returns:
            np.ndarray: Array of shape (D, M) containing the direction vectors.
        """
        rng = np.random.default_rng(12345)
        if self.ref_dir_method == 'dirichlet':
            if M > D:
                return rng.dirichlet(alpha=np.ones(M), size=D)
            extreme_dirs = np.eye(M, dtype=float)
            num_random_dirs = D - M
            if num_random_dirs > 0:
                random_dirs = rng.dirichlet(alpha=np.ones(M), size=num_random_dirs)
                all_dirs = np.vstack((extreme_dirs, random_dirs))
            else:
                all_dirs = extreme_dirs[:D]
            rng.shuffle(all_dirs)
            return all_dirs

        for H in range(1, 10):
            dirs = self._das_dennis(M, H)
            if dirs.shape[0] >= D:
                return dirs[:D, :]
        return rng.dirichlet(alpha=np.ones(M), size=D)

    def set_objective_directions(self, names: list, sense: list | None = None, D: int | None = None):
        """Defines objectives and assigns a fixed reference direction to each individual.

        Args:
            names (list): A list of objective names.
            sense (list | None, optional): A list of "min" or "max" for each objective.
                                           Defaults to all "min".
            D (int | None, optional): The number of reference directions. Defaults to
                                      the number of quantum individuals.
        """
        self.objective_names = list(names)
        self.num_objectives = len(self.objective_names)
        self.objective_sense = [("min" if s is None else str(s).lower())
                                for s in (sense or ["min"] * self.num_objectives)]
        if D is None:
            D = self.num_ind
        self._ref_dirs = self._make_ref_directions(self.num_objectives, D)
        self._ind_to_dir = np.array([i % D for i in range(self.num_ind)], dtype=int)

    def _normalize_objectives_01(self, objs: np.ndarray) -> np.ndarray:
        """Normalizes objective values to a [0, 1] range, oriented for maximization.

        Minimization objectives are inverted (1 - normalized_value).

        Args:
            objs (np.ndarray): Raw objective values of shape (E, M).

        Returns:
            np.ndarray: Normalized values of shape (E, M), where higher is better.
        """
        E, M = objs.shape
        g = np.empty_like(objs, dtype=float)
        for m in range(M):
            col = objs[:, m].astype(float)
            lo, hi = np.min(col), np.max(col)
            if hi - lo < 1e-12:
                norm = np.zeros_like(col)
            else:
                norm = (col - lo) / (hi - lo)
            g[:, m] = (1.0 - norm) if (self.objective_sense[m] == "min") else norm
        return g

    def _score_weighted_sum(self, g: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """Calculates a weighted sum score for each individual.

        Args:
            g (np.ndarray): Maximization-oriented normalized objectives (E, M).
            lam (np.ndarray): A weight vector (M,).

        Returns:
            np.ndarray: A score for each individual (E,).
        """
        return g.dot(lam)

    def _quantile_thresholds(self, g: np.ndarray, q_low: float = 0.3, q_high: float = 0.90) -> dict:
        """Calculates quantile-based thresholds for each objective.

        Args:
            g (np.ndarray): Maximization-oriented normalized objectives.
            q_low (float, optional): The lower quantile. Defaults to 0.3.
            q_high (float, optional): The upper quantile. Defaults to 0.90.

        Returns:
            dict: A dictionary with "min_ok" and "max_ok" threshold vectors.
        """
        min_ok = np.quantile(g, q_low, axis=0)
        max_ok = np.quantile(g, q_high, axis=0)
        return {"min_ok": min_ok, "max_ok": max_ok}

    def _mask_constraints(self, g: np.ndarray, cons: dict, hard: dict | None = None) -> np.ndarray:
        """Creates a boolean mask to filter individuals based on constraints.

        Args:
            g (np.ndarray): Maximization-oriented normalized objectives.
            cons (dict): Dictionary of constraints, e.g., from `_quantile_thresholds`.
            hard (dict | None, optional): Additional hard constraints. Defaults to None.

        Returns:
            np.ndarray: A boolean array indicating which individuals pass the filter.
        """
        E, M = g.shape
        mask = np.ones(E, dtype=bool)
        if cons and ("min_ok" in cons):
            min_ok = cons["min_ok"]
            if hard and ("min_idx" in hard):
                for m in hard["min_idx"]:
                    mask &= (g[:, m] >= float(min_ok[m]))
            else:
                for m in range(M):
                    mask &= (g[:, m] >= float(min_ok[m]))
        return mask

    def _select_topk_stratified(self, scores: np.ndarray, K: int, P_mult: int = 5) -> np.ndarray:
        """Selects K individuals using stratified sampling.

        It first selects the top P individuals (P > K), splits them into K
        groups, and picks the best from each group.

        Args:
            scores (np.ndarray): The scores for each individual.
            K (int): The final number of individuals to select.
            P_mult (int, optional): The multiplier to determine the initial pool size P.

        Returns:
            np.ndarray: The indices of the K selected individuals.
        """
        E = scores.shape[0]
        K = int(min(max(1, K), E))
        P = int(min(P_mult * K, E))
        order = np.argsort(-scores)
        topP = order[:P]
        splits = np.array_split(topP, K)
        picks = [seg[0] for seg in splits if seg.size > 0]
        return np.array(picks[:K], dtype=int)

    def _entropy_rows(self, P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Calculates the normalized entropy for each probability vector in P.

        Args:
            P (np.ndarray): An array of probability distributions.
            eps (float, optional): A small value for numerical stability.

        Returns:
            np.ndarray: The calculated entropy for each row.
        """
        F = P.shape[-1]
        P = np.clip(P, eps, 1.0)
        H = -(P * np.log(P)).sum(axis=-1) / np.log(F)
        return H

    def _kl_rows(self, P_new: np.ndarray, P_old: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Calculates the KL divergence between two sets of probability vectors.

        Args:
            P_new (np.ndarray): The new probability distributions.
            P_old (np.ndarray): The old probability distributions.
            eps (float, optional): A small value for numerical stability.

        Returns:
            np.ndarray: The KL divergence for each pair of rows.
        """
        P_new = np.clip(P_new, eps, 1.0)
        P_old = np.clip(P_old, eps, 1.0)
        return (P_new * (np.log(P_new) - np.log(P_old))).sum(axis=-1)

    def _frac_onehot(self, P: np.ndarray, thr: float = 0.9) -> float:
        """Calculates the fraction of probability vectors that are nearly one-hot.

        Args:
            P (np.ndarray): An array of probability distributions.
            thr (float, optional): The threshold to consider a vector "one-hot".

        Returns:
            float: The fraction of vectors where the max probability exceeds the threshold.
        """
        return float(np.mean(P.max(axis=-1) > thr))

    def _elite_weights(self, E: int) -> np.ndarray:
        """Generates weights for elites, optionally based on rank.

        Args:
            E (int): The number of elites.

        Returns:
            np.ndarray: A normalized weight vector of size E.
        """
        if not self.rank_weighting:
            return np.ones(E, dtype=float) / max(E, 1)
        ranks = np.arange(E, dtype=float) + 1.0
        w = 1.0 / ranks
        return w / w.sum()

    def _build_q_global(self, elites_choices: np.ndarray, F: int) -> np.ndarray:
        """Builds a global target distribution `q` from elite individuals.

        A weighted histogram is created for each gene across all elites.

        Args:
            elites_choices (np.ndarray): The chromosomes of elite individuals (E, L).
            F (int): The number of possible functions (alleles).

        Returns:
            np.ndarray: The global target distribution of shape (L, F).
        """
        E, L = elites_choices.shape
        w = self._elite_weights(E)
        counts = np.zeros((L, F), dtype=float)
        for e in range(E):
            counts[np.arange(L), elites_choices[e]] += w[e]
        q = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)

        if self.ema_beta and self.ema_beta > 0.0:
            if not hasattr(self, "_q_ema") or self._q_ema.shape != q.shape:
                self._q_ema = q.copy()
            self._q_ema = self.ema_beta * self._q_ema + (1 - self.ema_beta) * q
            q = self._q_ema
        return q

    def _build_q_bootstrap_rows(self, pool_choices: np.ndarray, rows: np.ndarray,
                                cols: np.ndarray, F: int, k: int,
                                weights: np.ndarray = None,
                                rng: np.random.Generator = None) -> np.ndarray:
        """Builds target distributions by bootstrapping from an elite pool.

        For each specified (individual, gene), it samples `k` elites from the
        pool and creates a histogram to form the target distribution.

        Args:
            pool_choices (np.ndarray): Chromosomes of the elite pool (E_pool, L).
            rows (np.ndarray): Indices of quantum individuals to update.
            cols (np.ndarray): Indices of genes to update.
            F (int): Number of functions.
            k (int): Number of elites to sample for each update.
            weights (np.ndarray, optional): Weights for sampling from the pool.
            rng (np.random.Generator, optional): A random number generator.

        Returns:
            np.ndarray: Target distributions `q_rows` of shape (K_sel, F).
        """
        if rng is None:
            rng = np.random.default_rng()

        E_pool, L = pool_choices.shape
        K_sel = rows.size
        q_rows = np.zeros((K_sel, F), dtype=float)
        if weights is None:
            weights = np.ones(E_pool, dtype=float) / max(E_pool, 1)

        for i in range(K_sel):
            c = cols[i]
            sel = rng.choice(E_pool, size=k, replace=True, p=weights)
            ops = pool_choices[sel, c]
            for op in ops:
                q_rows[i, op] += 1.0
            s = q_rows[i].sum()
            q_rows[i] = (q_rows[i] / s) if s > 0 else (1.0 / F)
        return q_rows

    def _build_q_moead_topk_rows(self, pool_choices: np.ndarray, pool_objs: np.ndarray,
                                  rows: np.ndarray, cols: np.ndarray, F: int, K: int) -> np.ndarray:
        """Builds target distributions using the MOEA/D-TopK strategy.

        For each quantum individual to be updated, it uses its assigned reference
        direction to select the best `K` individuals from the elite pool and
        builds a target distribution from them.

        Args:
            pool_choices (np.ndarray): Chromosomes of the elite pool (E_pool, L).
            pool_objs (np.ndarray): Objective values of the elite pool (E_pool, M).
            rows (np.ndarray): Indices of quantum individuals to update.
            cols (np.ndarray): Indices of genes to update.
            F (int): Number of functions.
            K (int): Number of elites to select.

        Returns:
            np.ndarray: Target distributions `q_rows` of shape (K_sel, F).
        """
        assert self._ref_dirs is not None and self._ind_to_dir is not None
        E_pool, L = pool_choices.shape
        K_sel = rows.size
        q_rows = np.zeros((K_sel, F), dtype=float)

        g = self._normalize_objectives_01(pool_objs)
        cons = self._quantile_thresholds(g, q_low=self.moead_q_low, q_high=self.moead_q_high)

        for r in range(K_sel):
            i = int(rows[r]); j = int(cols[r])
            lam = self._ref_dirs[self._ind_to_dir[i]]

            mask = self._mask_constraints(g, cons, hard=None)
            idx = np.where(mask)[0]
            if idx.size < K:
                idx = np.arange(E_pool)

            s = self._score_weighted_sum(g[idx, :], lam)
            pick_rel = self._select_topk_stratified(s, K=K, P_mult=self.topP_mult)
            pick = idx[pick_rel]

            ops = pool_choices[pick, j].astype(int)
            counts = np.bincount(ops, minlength=F).astype(float)
            ssum = counts.sum()
            q_rows[r, :] = counts / ssum if ssum > 0 else (1.0 / F)
        return q_rows

    def _sample_intensity(self, lo: float = 0.5, hi: float = 1.0) -> float:
        """Samples an update intensity from a Beta distribution.

        Args:
            lo (float, optional): The minimum intensity. Defaults to 0.5.
            hi (float, optional): The maximum intensity. Defaults to 1.0.

        Returns:
            float: A sampled intensity value.
        """
        u = np.random.beta(2.0, 5.0)
        return lo + (hi - lo) * u

    def _suggest_max_update(self) -> float:
        """Suggests a base `max_update` value based on the number of functions.

        Returns:
            float: The suggested `max_update` value.
        """
        F = self.chromosome.num_functions
        if F <= 12: return 0.07
        if F <= 32: return 0.05
        return 0.04

    def _lin_schedule(self, t: float, T: float, start: float, end: float) -> float:
        """Calculates a value based on a linear schedule.

        Args:
            t (float): Current step.
            T (float): Total steps.
            start (float): Start value.
            end (float): End value.

        Returns:
            float: The scheduled value.
        """
        t = max(0, min(int(t), int(T)))
        return start + (end - start) * (t / float(T + 1e-12))

    def _cosine_schedule(self, t: float, T: float, start: float, end: float) -> float:
        """Calculates a value based on a cosine annealing schedule.

        Args:
            t (float): Current step.
            T (float): Total steps.
            start (float): Start value.
            end (float): End value.

        Returns:
            float: The scheduled value.
        """
        t = max(0, min(int(t), int(T)))
        w = 0.5 * (1.0 + np.cos(np.pi * t / float(T)))
        return end + (start - end) * w

    def _update(self, chromosomes: np.ndarray, idx: np.ndarray, update_value: np.ndarray) -> np.ndarray:
        """Applies the quantum update rule to a batch of probability vectors.

        It increases the probability of the 'winner' gene (at `idx`) and
        proportionally decreases the probabilities of the others. It ensures
        probabilities stay within the [min_prob, max_prob] bounds and renormalizes.

        Args:
            chromosomes (np.ndarray): A batch of probability vectors to update.
            idx (np.ndarray): The indices of the winner gene for each vector.
            update_value (np.ndarray): The amount to add to the winner's probability.

        Returns:
            np.ndarray: The updated and normalized probability vectors.
        """
        idx0 = np.arange(chromosomes.shape[0])
        current = chromosomes[idx0, idx]
        headroom = np.maximum(self.max_prob - current, 0.0)
        update_array = np.minimum(update_value, headroom)
        sum_values = current + update_array

        chromosomes[idx0, idx] = 0.0
        totals = np.sum(chromosomes, axis=1)
        totals = np.where(totals == 0, 1e-8, totals)
        decrease = (update_array / totals).reshape(-1, 1) * chromosomes
        chromosomes -= decrease
        chromosomes[idx0, idx] = sum_values

        chromosomes = np.maximum(chromosomes, self.min_prob)
        chromosomes /= np.sum(chromosomes, axis=1, keepdims=True)
        return chromosomes

    def update_quantum(self, intensity: float | None = None, current_gen: int | None = None):
        """Performs the main quantum population update.

        This method orchestrates the entire update process:
        1. Schedules the learning rate (`update_quantum_rate`) and max update value.
        2. Selects a random subset of (individual, gene) pairs to update.
        3. Based on `self.elite_mode`, constructs a target probability
            distribution `q_rows` for each selected gene.
        4. Calculates the update step size based on the target distribution's confidence.
        5. Calls the `_update` method to apply the changes.
        6. Enforces global constraints and logs metrics.

        Args:
            intensity (float | None, optional): A factor to scale the update step size.
                                                If None, it's sampled randomly. Defaults to None.
            current_gen (int | None, optional): The current generation/epoch index, used
                                                for scheduling and logging. Defaults to None.
        """
        u = getattr(self, "_update_counter", 0)
        U_total = getattr(self, "_U_total", None)
        if U_total is None:
            U_total = max(1, (current_gen or 0) // 1)
        legacy = (self.elite_mode == "old")

        if legacy:
            self.update_quantum_rate = float(self.update_quantum_rate)
        else:
            self.update_quantum_rate = self._cosine_schedule(u, U_total, self.rate_boost, self.rate_end)

        if intensity is None:
            intensity = self._sample_intensity(lo=0.5, hi=1.0)

        if legacy:
            self.max_update = float(self.max_update)
        else:
            base = self._suggest_max_update()
            mult = self._cosine_schedule(u, U_total, start=1.5, end=0.8)
            self.max_update = base * mult
        eta_base = float(intensity) * float(self.max_update)

        F = int(self.chromosome.num_functions)

        rand = np.random.rand(self.num_ind, self.chromosome.num_genes)
        rows, cols = np.where(rand <= self.update_quantum_rate)
        if rows.size == 0:
            return

        if legacy:
            E = min(self.num_ind, self.current_pop.shape[0])
            best_classic = self.current_pop[:E]
            winners = best_classic[rows, cols]
            self.probabilities[rows, cols, :] = self._update(
                self.probabilities[rows, cols, :], winners, eta_base)
            self._log_update_metrics_min(epoch_idx=current_gen)
            self.save_metrics_csv(self.metrics_output)
            return

        P_sel = self.probabilities[rows, cols, :].astype(float, copy=True)
        P_sel = self._apply_noop_caps_rows(P_sel, rows, cols)

        if self.elite_mode == "single":
            E = min(self.num_ind, self.current_pop.shape[0])
            best_classic = self.current_pop[:E]
            winners_single = best_classic[rows, cols]
            q_rows = np.zeros_like(P_sel)
            q_rows[np.arange(rows.size), winners_single] = 1.0
        elif self.elite_mode == "global_k":
            E = min(self.k_elites, self.current_pop.shape[0])
            topk = self.current_pop[:E]
            q_full = self._build_q_global(topk, F)
            q_rows = q_full[cols, :]
        elif self.elite_mode == "bootstrap_k":
            E_pool = min(self.current_pop.shape[0], max(self.k_elites, self.pool_factor * self.k_elites))
            pool = self.current_pop[:E_pool]
            weights = self._elite_weights(E_pool)
            q_rows = self._build_q_bootstrap_rows(pool, rows, cols, F, k=self.k_elites, weights=weights)
        elif self.elite_mode == "moead_topk":
            E_pool = self.current_pop.shape[0]
            pool_choices = self.current_pop[:E_pool]
            pool_objs = self.current_pop_objs[:E_pool, :]
            q_rows = self._build_q_moead_topk_rows(
                pool_choices=pool_choices, pool_objs=pool_objs,
                rows=rows, cols=cols, F=F, K=self.k_elites)
        else:
            raise ValueError(f"Unknown elite_mode: {self.elite_mode}")

        q_rows = self._mask_noop_in_qrows(q_rows, rows, cols, prior_P=P_sel)
        winners = np.argmax(q_rows, axis=1)
        consensus = q_rows[np.arange(q_rows.shape[0]), winners]
        bump = eta_base * np.maximum(consensus, 1e-8)

        updated = self._update(P_sel, winners, bump)
        self.probabilities[rows, cols, :] = updated

        if getattr(self, "enforce_noop_in_update", True):
            self._enforce_noop_static_mask_global()

        self._log_update_metrics_min(epoch_idx=current_gen)
        self.save_metrics_csv(self.metrics_output)