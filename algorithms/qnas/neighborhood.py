from collections import deque
from dataclasses import dataclass
import numpy as np
import hashlib

from dataclasses import dataclass
import numpy as np
import hashlib
from typing import List, Tuple

@dataclass(frozen=True)
class NeighborhoodEntry:
    arch: np.ndarray      # (G,) integer genotype
    objs: np.ndarray      # (M,) objective vector (M=1 for single-objective)
    gen:  int
    q_idx: int
    sig:  int             # hash signature for deduplication


def _geno_sig(arch: np.ndarray) -> int:
    """Stable 64-bit signature for an integer genotype."""
    a = arch.astype(np.int32, copy=False)
    return int(hashlib.blake2b(a.tobytes(), digest_size=8).hexdigest(), 16)

class NeighborhoodArchive:
    """
    Rolling elite history for a single neighborhood (one q_ind).

    This archive ingests whole evaluated batches and maintains a top-K-by-score
    history OVER TIME (best-so-far), with optional deduplication by genotype
    signature. Unlike deque(maxlen=K), we never evict before we decide which
    entries to keep; pruning happens explicitly after ingest so historical
    champions are preserved as long as they remain within the top-K.

    Workflow per generation:
        1) update_from_batch(archs, objs, gen, q_idx)
            - Upsert all candidates from the batch.
            - If a genotype reappears, keep the record with the better score.
        2) Prune to the top K entries by score (descending).

    Methods:
        - update_from_batch: ingest whole batch, upsert, prune to top-K.
        - topk(k): return current top-k (architectures and scores).
        - champion(): return the single best entry (or (None, None) if empty).
    """

    def __init__(self, k_hist: int = 10, dedup: bool = True):
        self._k: int = int(k_hist)
        self._dedup: bool = bool(dedup)
        self._items: List[NeighborhoodEntry] = []  # maintained sorted (desc) AFTER prune

    def __len__(self) -> int:
        return len(self._items)

    def _best_index_by_sig(self) -> dict[int, int]:
        """Build a mapping: genotype signature -> index in self._items."""
        return {e.sig: i for i, e in enumerate(self._items)}

    def update_from_batch(
        self,
        archs: np.ndarray,
        objs: np.ndarray,
        gen: int,
        q_idx: int,
    ) -> None:
        """
        Upsert the entire batch, then prune to top-K by score (descending).
        For single-objective, objs[:, 0] is used as the score to maximize.

        Args:
            archs: (N, G) integer genotypes for this neighborhood in this batch.
            objs:  (N,) or (N, M) objective values (maximize). For M=1, use objs[:,0].
            gen:   Current generation index (for record-keeping).
            q_idx: Neighborhood index (for record-keeping).
        """
        if archs is None or archs.size == 0:
            return
        if archs.ndim == 1:
            archs = archs[None, :]
        if objs.ndim == 1:
            objs = objs[:, None]

        # Build index for dedup upsert (best per signature)
        sig2pos = self._best_index_by_sig() if self._dedup else {}

        # 1) Upsert every candidate from the batch
        for a, o in zip(archs, objs):
            sig = _geno_sig(a)
            if self._dedup and sig in sig2pos:
                pos = sig2pos[sig]
                # Keep better score (single-objective: o[0])
                if float(o[0]) > float(self._items[pos].objs[0]):
                    self._items[pos] = NeighborhoodEntry(
                        arch=a.copy(),
                        objs=o.astype(float, copy=True),
                        gen=int(gen),
                        q_idx=int(q_idx),
                        sig=sig
                    )
                # else: existing entry is better; keep it
            else:
                self._items.append(
                    NeighborhoodEntry(
                        arch=a.copy(),
                        objs=o.astype(float, copy=True),
                        gen=int(gen),
                        q_idx=int(q_idx),
                        sig=sig
                    )
                )
                if self._dedup:
                    sig2pos[sig] = len(self._items) - 1

        # 2) Prune to top-K by score (descending), preserving best historical entries
        if self._items:
            self._items.sort(key=lambda e: float(e.objs[0]), reverse=True)
            if len(self._items) > self._k:
                self._items = self._items[: self._k]

    def topk(self, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the current top-k architectures and scores.

        Args:
            k: Number of entries to return (<= archive size).

        Returns:
            (A, O):
                A: (<=k, G) integer genotypes (architectures)
                O: (<=k, M) objective vectors (M=1 for single-objective)
        """
        if not self._items:
            return (np.empty((0, 0), dtype=int),
                    np.empty((0, 0), dtype=float))
        # _items is already kept sorted (desc) after pruning
        pick = self._items[: min(k, len(self._items))]
        A = np.vstack([e.arch for e in pick])
        O = np.vstack([e.objs for e in pick])
        return A, O

    def champion(self) -> Tuple[np.ndarray | None, np.ndarray | None]:
        """
        Return the single best entry as (A:(1,G), O:(1,M)), or (None, None) if empty.
        """
        A, O = self.topk(1)
        if A.size == 0:
            return None, None
        return A, O


class NeighborhoodRegistry:
    """Holds a NeighborhoodArchive per q_ind and provides bank-level ops."""
    def __init__(self, num_neighborhoods: int, k_hist: int = 10):
        self.num_neighborhoods = int(num_neighborhoods)
        self.neighborhoods = [NeighborhoodArchive(k_hist=k_hist)
                            for _ in range(self.num_neighborhoods)]

    # ---- Ingest: full batch routed by parent_map ----
    def add_full_batch_by_parent(
        self,
        parent_map: np.ndarray,   # shape (Nq*R,)
        archs: np.ndarray,        # shape (Nq*R, G)
        objs: np.ndarray,         # shape (Nq*R,) or (Nq*R, M)
        gen: int,
    ) -> None:
        """Ingest the entire evaluated batch. Each neighborhood archive upserts
        all its items (keep best per genotype) and prunes to its top-k_hist."""
        if objs.ndim == 1:
            objs = objs[:, None]
        for q in range(self.num_neighborhoods):
            idx = np.where(parent_map == q)[0]
            if idx.size == 0:
                continue
            self.neighborhoods[q].update_from_batch(
                archs=archs[idx], objs=objs[idx], gen=gen, q_idx=q
            )

    # ---- Reads: champions / top-k ----
    def champions(self):
        """Return one champion per neighborhood stacked as (Nq,G) and (Nq,M).
        If any neighborhood has an empty archive, returns (None, None)."""
        A_list, O_list = [], []
        for nb in self.neighborhoods:
            A, O = nb.champion()
            if A is None:
                return None, None
            A_list.append(A[0]); O_list.append(O[0])
        return np.vstack(A_list), np.vstack(O_list)

    def champions_or_best_of_batch(
        self,
        parent_map: np.ndarray,
        archs: np.ndarray,
        scores: np.ndarray,  # single-objective: (Nq*R,) or (Nq*R,1)
    ):
        """One champion per neighborhood; fallback to best-of-batch if archive empty."""
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores.ravel()

        champs_A, champs_O = [], []
        for q in range(self.num_neighborhoods):
            # Try archive champion first
            A, O = self.neighborhoods[q].champion()
            if A is not None:
                champs_A.append(A[0]); champs_O.append(O[0])
                continue

            # Fallback: best-of-batch for this neighborhood
            idx = np.where(parent_map == q)[0]
            if idx.size == 0:
                # Soft fallback: zeros row (should be rare)
                champs_A.append(np.zeros((archs.shape[1],), dtype=archs.dtype))
                champs_O.append(np.array([0.0], dtype=float))
            else:
                j = idx[int(np.argmax(scores[idx]))]
                champs_A.append(archs[j])
                champs_O.append(np.array([float(scores[j])], dtype=float))

        return np.vstack(champs_A), np.vstack(champs_O)

    def per_neighborhood_topk(self, k_each: int = 2):
        """Balanced pool: concat top-k from each neighborhood archive.
        Returns (A:(<=Nq*k_each,G), O:(<=Nq*k_each,M)) or (None,None) if all empty."""
        A_list, O_list = [], []
        for nb in self.neighborhoods:
            A, O = nb.topk(k_each)
            if A.size:
                A_list.append(A); O_list.append(O)
        if not A_list:
            return None, None
        return np.vstack(A_list), np.vstack(O_list)