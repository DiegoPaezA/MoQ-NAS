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
    """ QNAS Chromosomes for the networks to be evolved. """

    def __init__(self, num_quantum_ind, max_num_nodes, repetition, update_quantum_rate,
                fn_list, initial_probs, crossover_method='hux', elite_mode="global_k",
                k_elites=5, pool_factor=2, ema_beta=0.7, rank_weighting=True, experiment_path:str=""):
        """ Initialize QPopulationNetwork.

        Args:
            num_quantum_ind: (int) number of quantum individuals.
            max_num_nodes: (int) maximum number of nodes of the network, which will be the
                number of genes in a individual.
            repetition: (int) ratio between the number of classic individuals in the classic
                population and the quantum individuals in the quantum population.
            update_quantum_rate: (float) probability that a quantum gene will be updated.
            fn_list: list of possible functions.
            initial_probs: list defining the initial probabilities for each function; if empty,
                the algorithm will give the same probability for each function.
            elite_mode (str, optional): Strategy to build target distributions from elites.
                Options are "single", "global_k" or "bootstrap_k". Defaults to "global_k".
            k_elites (int, optional): Number of elites used when elite_mode requires it.
                Defaults to 5.
            pool_factor (int, optional): Multiplier to determine the elite pool size when
                using "bootstrap_k". Defaults to 2.
            ema_beta (float, optional): Exponential moving average factor applied when
                computing global elite distributions. Set to 0.0 to disable. Defaults to 0.7.
            rank_weighting (bool, optional): If True, weight elite contributions by
                inverse rank. Defaults to True.
            experiment_path (str): Path to the experiment directory.
        """

        super(QPopulationNetwork, self).__init__(num_quantum_ind, repetition,
                                                update_quantum_rate)
        self.probabilities = None

        self.chromosome = QChromosomeNetwork(max_num_nodes, fn_list, self.dtype)

        self.max_update = 0.05
        self.max_prob = 0.90
        self.min_prob = max(1e-8, 0.01 / self.chromosome.num_functions)

        self.elite_mode = elite_mode  # "single" | "global_k" | "bootstrap_k"
        self.k_elites = k_elites
        self.pool_factor = pool_factor
        self.ema_beta = ema_beta
        self.rank_weighting = rank_weighting
        
        self._U_total = None
        
        # Update-rate schedule (probability a (ind,gene) gets updated)
        self.rate_start = max(0.2, self.update_quantum_rate)  # don’t go below current
        self.rate_boost = 0.40                                # early exploration target
        self.rate_end   = self.update_quantum_rate            # cool to your baseline

        self.metrics_output = os.path.join(experiment_path, "qpop_update", "metrics_output.csv")
        os.makedirs(os.path.dirname(self.metrics_output), exist_ok=True)
        self._last_dump_idx = 0

        self.initial_probs = self.chromosome.initialize_qgenes(initial_probs=initial_probs)
        self.crossover_method = crossover_method  # Crossover method selection
        self.initialize_qpop()
        self._init_metrics_min()

    def set_schedule_total_updates(self, total_updates: int):
        self._U_total = max(1, int(total_updates))

    def initialize_qpop(self):
        """ Initialize quantum population with *self.num_ind* individuals. """

        # Shape = (num_ind, num_nodes, num_functions)
        self.probabilities = np.tile(self.initial_probs, (self.num_ind,
                                                        self.chromosome.num_genes, 1))

    def generate_classical(self):
        """ Generate a specific number of classical individuals from the observation of quantum
            individuals. This number is equal to (*num_ind* x *repetition*).
        """

        def sample(idx0, idx1):
            return np.random.choice(size, p=temp_prob[idx0, idx1, :])

        size = self.chromosome.num_functions
        new_pop = np.zeros(shape=(self.num_ind * self.repetition, self.chromosome.num_genes),
                            dtype=np.int32)

        temp_prob = np.tile(self.probabilities, (self.repetition, 1, 1))
        
        for ind in range(self.num_ind * self.repetition):
            for node in range(self.chromosome.num_genes):
                new_pop[ind, node] = sample(ind, node)

        return new_pop
    
    def hux_crossover(self, parent1, parent2):
        """ Perform Half Uniform Crossover (HUX) between two parent chromosomes. """
        differing_indices = np.where(parent1 != parent2)[0]
        num_swaps = len(differing_indices) // 2
        swap_indices = np.random.choice(differing_indices, num_swaps, replace=False)
        offspring1, offspring2 = parent1.copy(), parent2.copy()
        offspring1[swap_indices], offspring2[swap_indices] = parent2[swap_indices], parent1[swap_indices]
        return offspring1, offspring2

    def uniform_crossover(self, parent1, parent2):
        """ Perform Uniform Crossover with a crossover mask between two parent chromosomes. """
        chromosome_length = len(parent1)
        crossover_mask = np.random.randint(0, 2, size=chromosome_length).astype(bool)
        offspring1, offspring2 = parent1.copy(), parent2.copy()
        offspring1[crossover_mask], offspring2[crossover_mask] = parent2[crossover_mask], parent1[crossover_mask]
        return offspring1, offspring2

    def apply_crossover(self, best_current_pop, new_pop):
        """ Apply the selected crossover method between best individuals of the current and new populations. 
        
        Args:
            best_current_pop: numpy array representing the best individuals from the current population.
            new_pop: numpy array representing the new population.
        
        Returns:
            A population of offspring resulting from the selected crossover method.
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

        return np.array(offspring[:len(new_pop)])  # Ensure offspring size matches new_pop size

    def set_crossover_method(self, method):
        """ Set the crossover method for this population. """
        if method in ['hux', 'uniform']:
            self.crossover_method = method
        else:
            raise ValueError(f"Unknown crossover method: {method}")
        
    def mutate_probabilities(self, fraction: float = 0.2, intensity: float = 0.1):
        """
        Perform exploratory mutation on a subset of quantum individuals’ probability distributions.

        This method selects a fraction of the population at random and blends each selected
        individual’s probability tensor with uniform noise. After mixing, values are clipped
        to enforce a minimum probability floor and then renormalized so that each distribution
        still sums to 1.

        Args:
            fraction (float, optional): Proportion of individuals to mutate, in the range [0.0, 1.0].
                Defaults to 0.2 (i.e., 20% of the population).
            intensity (float, optional): Mixing weight for the noise, in the range [0.0, 1.0].
                A value of 0.0 leaves the original probabilities unchanged; 1.0 replaces them
                entirely with noise. Defaults to 0.1 (i.e., 10% noise).
        """
        if not (0.0 <= fraction <= 1.0):
            raise ValueError(f"fraction must be between 0.0 and 1.0, got {fraction}")
        if not (0.0 <= intensity <= 1.0):
            raise ValueError(f"intensity must be between 0.0 and 1.0, got {intensity}")

        # Determine how many individuals to mutate (at least one)
        num_mut = max(1, int(self.num_ind * fraction))
        idx = np.random.choice(self.num_ind, num_mut, replace=False)

        noise = np.random.rand(num_mut,self.chromosome.num_genes,self.chromosome.num_functions)
        mutated = (1 - intensity) * self.probabilities[idx] + intensity * noise
        mutated = np.maximum(mutated, self.min_prob)
        mutated /= mutated.sum(axis=2, keepdims=True)
        self.probabilities[idx] = mutated

    def _init_metrics_min(self):
        self.metrics = {
            "epoch": [],
            "update_idx": [],
            "quantum_update_rate": [],
            "max_update": [],
            "entropy_mean": [],
            "kl_mean": [],            # puedes no llenar si no quieres KL
            "frac_onehot_0p9": [],
            # "noop_mass_mean": []
        }
        self._last_P = None          # para KL entre updates
        self._update_counter = 0     # contador de actualizaciones
        
    def _log_update_metrics_min(self, epoch_idx=None):
        P = self.probabilities  # (Nq, L, F)

        # Entropía media (promediando Nq y L)
        H_mean = float(self._entropy_rows(P).mean())

        # KL medio vs. último P (opcional)
        if self._last_P is not None:
            KL_mean = float(self._kl_rows(
                P.reshape(-1, P.shape[-1]),
                self._last_P.reshape(-1, self._last_P.shape[-1])
            ).mean())
        else:
            KL_mean = float("nan")  # o 0.0 si prefieres

        # Colapso (fracción de filas casi one-hot)
        frac_oh = self._frac_onehot(P, thr=0.9)

        # # Masa en no_op (si tienes noop_id)
        # if hasattr(self, "noop_id"):
        #     noop_mass = float(P[..., self.noop_id].mean())
        # else:
        #     noop_mass = float("nan")

        # Guardar en buffers
        self.metrics["epoch"].append(int(epoch_idx) if epoch_idx is not None else len(self.metrics["epoch"]))
        self.metrics["update_idx"].append(self._update_counter)
        self.metrics["quantum_update_rate"].append(self.update_quantum_rate)
        self.metrics["max_update"].append(self.max_update)
        self.metrics["entropy_mean"].append(H_mean)
        self.metrics["kl_mean"].append(KL_mean)
        self.metrics["frac_onehot_0p9"].append(frac_oh)
        # self.metrics["noop_mass_mean"].append(noop_mass)

        # actualizar referencia para próximo KL y contador
        self._last_P = P.copy()
        self._update_counter += 1

    def save_metrics_csv(self, path_csv: str, overwrite: bool = False):
        cols = ["epoch","update_idx","quantum_update_rate","max_update",
        "entropy_mean","kl_mean","frac_onehot_0p9"]

        total = len(self.metrics["update_idx"])
        start = 0 if overwrite or (not os.path.exists(path_csv)) else getattr(self, "_last_dump_idx", 0)
        if start >= total:
            return  # nada nuevo que escribir

        mode = "w" if overwrite or (not os.path.exists(path_csv)) else "a"
        write_header = (mode == "w")

        with open(path_csv, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if write_header:
                w.writeheader()
            for i in range(start, total):
                row = {k: self.metrics[k][i] for k in cols}
                w.writerow(row)

        self._last_dump_idx = total

    def _entropy_rows(self, P, eps=1e-12):
        F = P.shape[-1]
        P = np.clip(P, eps, 1.0)
        H = -(P * np.log(P)).sum(axis=-1) / np.log(F)
        return H  # misma shape que P[...,:]

    def _kl_rows(self, P_new, P_old, eps=1e-12):
        P_new = np.clip(P_new, eps, 1.0)
        P_old = np.clip(P_old, eps, 1.0)
        return (P_new * (np.log(P_new) - np.log(P_old))).sum(axis=-1)

    def _frac_onehot(self, P, thr=0.9):
        return float(np.mean(P.max(axis=-1) > thr))
    
    def _elite_weights(self, E: int) -> np.ndarray:
        """w_e ∝ 1/rank; e=0 es el mejor."""
        if not self.rank_weighting:
            return np.ones(E, dtype=float) / max(E, 1)
        ranks = np.arange(E, dtype=float) + 1.0  # 1..E
        w = 1.0 / ranks
        return w / w.sum()

    def _build_q_global(self, elites_choices: np.ndarray, F: int) -> np.ndarray:
        """
        Histograma global por gen: q (L, F).
        elites_choices: (E, L) índices de función [0..F)
        """
        E, L = elites_choices.shape
        w = self._elite_weights(E)                    # (E,)
        counts = np.zeros((L, F), dtype=float)
        for e in range(E):
            counts[np.arange(L), elites_choices[e]] += w[e]
        q = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)

        # EMA opcional
        if self.ema_beta and self.ema_beta > 0.0:
            if not hasattr(self, "_q_ema") or self._q_ema.shape != q.shape:
                self._q_ema = q.copy()
            self._q_ema = self.ema_beta * self._q_ema + (1 - self.ema_beta) * q
            q = self._q_ema
        return q  # (L, F)

    def _build_q_bootstrap_rows(self,
        pool_choices: np.ndarray,  # (E_pool, L)
        rows: np.ndarray,          # idxs de individuos cuánticos seleccionados
        cols: np.ndarray,          # idxs de genes seleccionados (alineado con rows)
        F: int,
        k: int,
        weights: np.ndarray = None,
        rng: np.random.Generator = None,) -> np.ndarray:
        """
        Devuelve q_rows (K_sel, F): para cada fila seleccionada (ind, gen=cols[i]),
        muestrea k élites del pool (con reemplazo y ponderación) y arma el histograma.
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
            sel = rng.choice(E_pool, size=k, replace=True, p=weights)  # (k,)
            ops = pool_choices[sel, c]                                 # (k,)
            # histograma uniforme
            for op in ops:
                q_rows[i, op] += 1.0
            s = q_rows[i].sum()
            q_rows[i] = (q_rows[i] / s) if s > 0 else (1.0 / F)
        return q_rows  # (K_sel, F)

    def _sample_intensity(self, lo=0.5, hi=1.0):
        # Beta(2,5): más masa hacia valores pequeños; reescala a [lo, hi]
        u = np.random.beta(2.0, 5.0)
        return lo + (hi - lo) * u

    def _suggest_max_update(self):
        # k=5 (actualización cada 5 épocas) y F variable:
        # más F -> paso base más pequeño (suave)
        F = self.chromosome.num_functions
        base = 0.07 if F <= 12 else 0.05 if F <= 32 else 0.04
        return base

    def _lin_schedule(self, t, T, start, end):
        t = max(0, min(int(t), int(T)))
        return start + (end - start) * (t / float(T + 1e-12))

    def _cosine_schedule(self, t, T, start, end):
        t = max(0, min(int(t), int(T)))
        w = 0.5 * (1.0 + np.cos(np.pi * t / float(T)))  # 1→0
        return end + (start - end) * w

    def _update_tilt_with_q(self, P_rows: np.ndarray, q_rows: np.ndarray, eta_base: float) -> np.ndarray:
        """
        P_rows: (K_sel, F) PMFs a actualizar
        q_rows: (K_sel, F) objetivos por fila (global_k o bootstrap_k)
        eta_base: paso base (intensity * max_update)
        """
        eps = 1e-12

        # paso adaptativo por fila (consenso^2)
        consensus = q_rows.max(axis=1, keepdims=True)  # (K_sel,1)
        eta = eta_base * consensus

        tilt = np.exp(eta * q_rows)                    # (K_sel,F)
        P_new = P_rows * tilt
        P_new /= np.maximum(P_new.sum(axis=1, keepdims=True), eps)

        # piso/techo + renorm
        P_new = np.clip(P_new, self.min_prob, self.max_prob)
        P_new /= np.maximum(P_new.sum(axis=1, keepdims=True), eps)
        return P_new

    def _update_tilt(self, P_rows, winners, eta):
        eps = 1e-12
        K, F = P_rows.shape
        q = np.zeros_like(P_rows)
        q[np.arange(K), winners] = 1.0      # objetivo mínimo viable: one-hot

        tilt = np.exp(eta * q)              # solo el ganador sube un poco
        P_new = P_rows * tilt
        P_new /= np.maximum(P_new.sum(axis=1, keepdims=True), eps)

        P_new = np.clip(P_new, self.min_prob, self.max_prob)
        P_new /= np.maximum(P_new.sum(axis=1, keepdims=True), eps)
        return P_new

    def _update(self, chromosomes, idx, update_value):
        """
        Modify *chromosomes* by adding *update_value* to the genes indicated by *idx* and
        subtracting *update_value* from the other genes proportional to the size of each
        probability. Refuerza la robustez numérica asegurando límites mínimo y renormalización.

        Args:
            chromosomes (np.ndarray): shape (N, M)
            idx (int): índice del gen a incrementar
            update_value (float): valor a añadir (antes limitado por max_prob)

        Returns:
            np.ndarray: matriz de probabilidades actualizada y normalizada
        """
        idx0 = np.arange(chromosomes.shape[0])

        current = chromosomes[idx0, idx]
        headroom = np.maximum(self.max_prob - current, 0.0)
        update_array = np.minimum(update_value, headroom)   # <-- antes era todo o nada
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

    def update_quantum(self, intensity=None, current_gen=None):
        """
        Actualiza self.probabilities según self.elite_mode:
        - "old": baseline aditivo (tu método previo) emparejado 1–a–1 con los top N_q clásicos.
        - "single": construye q one-hot por fila a partir de los top N_q, pero aplica *actualización aditiva* (Option A).
        - "global_k": un q por gen a partir del top-K global (mismo para todas las filas), luego *aditiva*.
        - "bootstrap_k": un q por fila, muestreando K élites de un pool mayor, luego *aditiva*.
        """
        # --- progreso en espacio de "updates" (no épocas)
        u = getattr(self, "_update_counter", 0)
        U_total = getattr(self, "_U_total", None)
        if U_total is None:
            # fallback si QNAS no llamó set_schedule_total_updates
            U_total = max(1, (current_gen or 0) // 1)

        # --- Scheduler: prob. de actualizar (exploración alta → baseline)
        self.update_quantum_rate = self._cosine_schedule(u, U_total, self.rate_boost, self.rate_end)

        # --- Intensidad estocástica (puedes programarla si quieres)
        if intensity is None:
            intensity = self._sample_intensity(lo=0.5, hi=1.0)  # Beta(2,5) reescalada

        # --- Scheduler: tamaño de paso (fuerte al inicio → suave al final)
        base = self._suggest_max_update()  # 0.07 (F<=12), 0.05 (<=32), else 0.04

        # cosine multiplier: stronger early, cooler later
        mult = self._cosine_schedule(u, U_total, start=1.5, end=0.8)
        # e.g., if base=0.05 → starts at 0.08, ends at 0.04
        # if base=0.07 → starts at ~0.112, ends at ~0.056

        self.max_update = base * mult
        eta_base = float(intensity) * float(self.max_update)

        F = int(self.chromosome.num_functions)

        # --- Seleccionar (ind, gen) a actualizar
        rand = np.random.rand(self.num_ind, self.chromosome.num_genes)
        rows, cols = np.where(rand <= self.update_quantum_rate)
        if rows.size == 0:
            return  # nada que hacer

        # --- Rama "old": tu baseline aditivo emparejado 1–a–1 (sin q_rows)
        if self.elite_mode == "old":
            E = min(self.num_ind, self.current_pop.shape[0])
            best_classic = self.current_pop[:E]          # (E, L)
            winners = best_classic[rows, cols]           # (K_sel,)
            # paso constante = eta_base (tu updater maneja headroom + renorm)
            self.probabilities[rows, cols, :] = self._update(
                self.probabilities[rows, cols, :],
                winners,
                eta_base
            )
            self._log_update_metrics_min(epoch_idx=current_gen)
            self.save_metrics_csv(self.metrics_output)
            return

        if self.elite_mode == "single":
            # one-hot por fila a partir de los top N_q clásicos
            E = min(self.num_ind, self.current_pop.shape[0])
            best_classic = self.current_pop[:E]                 # (E, L)
            winners_single = best_classic[rows, cols]           # (K_sel,)
            q_rows[np.arange(rows.size), winners_single] = 1.0  # one-hot por fila

        elif self.elite_mode == "global_k":
            # mismo top-K para todos, q por gen
            E = min(self.k_elites, self.current_pop.shape[0])
            topk   = self.current_pop[:E]                 # (E, L)
            q_full = self._build_q_global(topk, F)        # (L, F)
            q_rows = q_full[cols, :]                      # (K_sel, F)

        elif self.elite_mode == "bootstrap_k":
            # por fila: muestrear K élites de un pool grande
            E_pool = min(self.current_pop.shape[0], max(self.k_elites, self.pool_factor * self.k_elites))
            pool    = self.current_pop[:E_pool]           # (E_pool, L)
            weights = self._elite_weights(E_pool)         # (E_pool,)
            q_rows  = self._build_q_bootstrap_rows(pool, rows, cols, F, k=self.k_elites, weights=weights)

        else:
            raise ValueError(f"elite_mode desconocido: {self.elite_mode}")

        # --- OPTION A: actualización *aditiva* guiada por q_rows ---
        # Ganadores por fila (argmax del target)
        winners = np.argmax(q_rows, axis=1)  # (K_sel,)
        # "Consenso" = confianza del ganador en q_rows ∈ [0,1]
        consensus = q_rows[np.arange(q_rows.shape[0]), winners]  # (K_sel,)
        # Paso por fila: escala eta por consenso (con un piso numérico pequeño)
        bump = eta_base * np.maximum(consensus, 1e-8)            # (K_sel,)

        # Aplicar updater aditivo (maneja headroom, piso, techo, renorm)
        self.probabilities[rows, cols, :] = self._update(
            self.probabilities[rows, cols, :],  # (K_sel, F)
            winners,                             # (K_sel,) índices
            bump                                 # (K_sel,) pasos por fila
        )

        # --- Métricas + volcado CSV
        self._log_update_metrics_min(epoch_idx=current_gen)
        self.save_metrics_csv(self.metrics_output)
