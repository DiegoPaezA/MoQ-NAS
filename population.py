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
        
        self.objective_names = None          # p.ej. ["acc","params","lat"]
        self.objective_sense = None          # p.ej. ["max","min","min"]
        self.num_objectives  = None

        self.moead_q_low  = 0.30             # cuantiles para filtros (min-ok)
        self.moead_q_high = 0.90             # opcional (no usado abajo, pero disponible)
        self.topP_mult    = 5                # Top-P = topP_mult * K para estratificar

        # Direcciones de referencia (simplex) y asignación por individuo
        self._ref_dirs  = None               # (D, M)
        self._ind_to_dir = None              # (num_ind,)
        self.ref_dir_method = "das-dennis"  # 'das-dennis' | 'dirichlet'
        
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
    # -------------------------------------------------------------------
    # 3) Direcciones de referencia (MOEA/D) y asignación persistente
    # -------------------------------------------------------------------
    def _das_dennis(self, M: int, H: int):
            """
            Genera direcciones de referencia uniformemente espaciadas en un simplex.
            Versión corregida y robusta.
            """
            if H == 0:
                if M == 1:
                    return np.array([[1.0]], dtype=float)
                else:
                    return np.empty((0, M), dtype=float)

            def gen_partitions(n_remaining, k_remaining):
                """Generador recursivo para particiones de enteros."""
                if k_remaining == 1:
                    yield (n_remaining,)
                    return
                for i in range(n_remaining + 1):
                    for p in gen_partitions(n_remaining - i, k_remaining - 1):
                        yield (i,) + p

            partitions = list(gen_partitions(H, M))
            dirs = np.array(partitions, dtype=float) / float(H)
            return dirs

    def _make_ref_directions(self, M: int, D: int) -> np.ndarray:
            """
            Genera direcciones de referencia. El método 'dirichlet' ahora está "anclado"
            con vectores extremos para cada objetivo.
            """
            import numpy as np
            if D <= 0: D = self.num_ind
            rng = np.random.default_rng(12345)

            # --- Opción 1: Dirichlet Anclado (Híbrido) ---
            if self.ref_dir_method == 'dirichlet':
                if M > D:
                    # Caso extremo: no hay suficientes individuos para anclar cada objetivo.
                    # Se recurre a Dirichlet estándar.
                    return rng.dirichlet(alpha=np.ones(M), size=D)

                # 1. Crear los M focos extremos (uno por cada objetivo)
                extreme_dirs = np.eye(M, dtype=float)

                # 2. Calcular cuántas direcciones aleatorias faltan
                num_random_dirs = D - M

                # 3. Generar las direcciones aleatorias restantes (si es necesario)
                if num_random_dirs > 0:
                    random_dirs = rng.dirichlet(alpha=np.ones(M), size=num_random_dirs)
                    # 4. Combinar los vectores extremos y los aleatorios
                    all_dirs = np.vstack((extreme_dirs, random_dirs))
                else:
                    # Si D <= M, solo usamos los vectores extremos necesarios
                    all_dirs = extreme_dirs[:D]

                # 5. Barajar el conjunto final para una asignación imparcial de roles
                rng.shuffle(all_dirs)
                return all_dirs

            # --- Opción 2 (default): Intentar Das-Dennis (sistemático) con fallback a Dirichlet ---
            for H in range(1, 10):
                dirs = self._das_dennis(M, H)
                if dirs.shape[0] >= D:
                    return dirs[:D, :]

            # Fallback a Dirichlet si Das-Dennis falla
            return rng.dirichlet(alpha=np.ones(M), size=D)

    def set_objective_directions(self, names, sense=None, D=None):
        """Define objetivos y direcciones; asigna a cada individuo una dirección fija."""
        self.objective_names = list(names)
        self.num_objectives = len(self.objective_names)
        self.objective_sense = [("min" if s is None else str(s).lower())
                                for s in (sense or ["min"] * self.num_objectives)]
        if D is None:
            D = self.num_ind
        self._ref_dirs = self._make_ref_directions(self.num_objectives, D)  # (D,M)
        self._ind_to_dir = np.array([i % D for i in range(self.num_ind)], dtype=int)

    # ---------- Normalización + puntuación + filtros ----------

    def _normalize_objectives_01(self, objs: np.ndarray) -> np.ndarray:
        """
        objs: (E, M) crudos. Devuelve g en [0,1] orientado a MAX (min -> 1 - norm).
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
        """g: (E, M) MAX-oriented; lam: (M,). Retorna (E,), mayor=mejor."""
        return g.dot(lam)

    def _quantile_thresholds(self, g: np.ndarray, q_low=0.3, q_high=0.90):
        """Umbrales por cuantiles en g (MAX-oriented)."""
        min_ok = np.quantile(g, q_low, axis=0)
        max_ok = np.quantile(g, q_high, axis=0)
        return {"min_ok": min_ok, "max_ok": max_ok}

    def _mask_constraints(self, g: np.ndarray, cons: dict, hard: dict | None = None) -> np.ndarray:
        """
        Exige g[:,m] >= min_ok[m] a todos (o a subset en hard['min_idx']) para filtrar basura.
        """
        E, M = g.shape
        mask = np.ones(E, dtype=bool)
        if cons is not None and ("min_ok" in cons):
            min_ok = cons["min_ok"]
            if hard and ("min_idx" in hard):
                for m in hard["min_idx"]:
                    mask &= (g[:, m] >= float(min_ok[m]))
            else:
                for m in range(M):
                    mask &= (g[:, m] >= float(min_ok[m]))
        return mask

    def _select_topk_stratified(self, scores: np.ndarray, K: int, P_mult: int = 5) -> np.ndarray:
        """
        Top-K estratificado: toma Top-P (=P_mult*K), divide en K segmentos y elige 1 de cada.
        """
        E = scores.shape[0]
        K = int(min(max(1, K), E))
        P = int(min(P_mult * K, E))
        order = np.argsort(-scores)     # desc
        topP = order[:P]
        splits = np.array_split(topP, K)
        picks = [seg[0] for seg in splits if seg.size > 0]
        return np.array(picks[:K], dtype=int)

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
    
    def _build_q_moead_topk_rows(self,
                                pool_choices: np.ndarray,   # (E_pool, L)
                                pool_objs: np.ndarray,      # (E_pool, M)
                                rows: np.ndarray, cols: np.ndarray,
                                F: int, K: int) -> np.ndarray:
        """
        Para cada fila (ind i, gen j): normaliza objs->g (MAX), toma lam_i, filtra por cuantiles,
        puntúa con WS, Top-K estratificado, histograma->q_rows.
        """
        assert self._ref_dirs is not None and self._ind_to_dir is not None
        E_pool, L = pool_choices.shape
        K_sel = rows.size
        q_rows = np.zeros((K_sel, F), dtype=float)

        # 1) normaliza objetivos e impone constraints
        g = self._normalize_objectives_01(pool_objs)  # (E_pool, M)
        cons = self._quantile_thresholds(g, q_low=self.moead_q_low, q_high=self.moead_q_high)

        for r in range(K_sel):
            i = int(rows[r]); j = int(cols[r])
            lam = self._ref_dirs[self._ind_to_dir[i]]  # (M,)

            mask = self._mask_constraints(g, cons, hard=None)
            idx = np.where(mask)[0]
            if idx.size < K:
                idx = np.arange(E_pool)

            s = self._score_weighted_sum(g[idx, :], lam)  # (E_mask,)
            pick_rel = self._select_topk_stratified(s, K=K, P_mult=self.topP_mult)
            pick = idx[pick_rel]

            ops = pool_choices[pick, j].astype(int)
            counts = np.bincount(ops, minlength=F).astype(float)
            ssum = counts.sum()
            q_rows[r, :] = counts / ssum if ssum > 0 else (1.0 / F)
        return q_rows

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
            
        legacy = (self.elite_mode == "old")

        # --- Tasa de actualización efectiva (si legacy, NO usar scheduler)
        if legacy:
            self.update_quantum_rate = float(self.update_quantum_rate)  # tal cual estaba
        else:
            # --- Scheduler: prob. de actualizar (exploración alta → baseline)
            self.update_quantum_rate = self._cosine_schedule(u, U_total, self.rate_boost, self.rate_end)

        # --- Intensidad estocástica (puedes programarla si quieres)
        if intensity is None:
            intensity = self._sample_intensity(lo=0.5, hi=1.0)  # Beta(2,5) reescalada

        if legacy:
            self.max_update = float(self.max_update)   # respeta el valor fijo que ya tengas
        else:
            base = self._suggest_max_update()         # 0.07/0.05/0.04 según F
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
        if legacy:
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
        
        # --- Construir q_rows según elite_mode (todas terminan con q_rows: (K_sel, F))
        P_sel = self.probabilities[rows, cols, :].astype(float, copy=True)  # (K_sel, F)
        
        if self.elite_mode == "single":
            # one-hot por fila a partir de los top N_q clásicos
            E = min(self.num_ind, self.current_pop.shape[0])
            best_classic = self.current_pop[:E]                 # (E, L)
            winners_single = best_classic[rows, cols]           # (K_sel,)
            q_rows = np.zeros_like(P_sel)
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
            
        elif self.elite_mode == "moead_topk":
            # Pool para MOEA/D: usa top E_pool de la población actual
            E_pool = self.current_pop.shape[0] # min(self.current_pop.shape[0], max(self.k_elites, self.pool_factor * self.k_elites))
            pool_choices = self.current_pop[:E_pool]         # (E_pool, L)
            pool_objs    = self.current_pop_objs[:E_pool, :] # (E_pool, M)  <-- asegúrate de setearla cada gen
            
            # Construye q_rows con moead_topk
            q_rows = self._build_q_moead_topk_rows(
                pool_choices=pool_choices,
                pool_objs=pool_objs,
                rows=rows, cols=cols,
                F=F, K=self.k_elites
            )

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
