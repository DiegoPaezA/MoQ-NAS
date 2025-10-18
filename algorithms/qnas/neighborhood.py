from collections import deque
from dataclasses import dataclass
import numpy as np
import hashlib

@dataclass(frozen=True)
class NeighborhoodEntry:
    arch: np.ndarray      # (G,) int genotype
    objs: np.ndarray      # (M,) objective vector (M=1 for single obj)
    gen:  int
    q_idx: int
    sig:  int             # hash for dedup

def _geno_sig(arch: np.ndarray) -> int:
    a = arch.astype(np.int32, copy=False)
    return int(hashlib.blake2b(a.tobytes(), digest_size=8).hexdigest(), 16)

class NeighborhoodArchive:
    """Rolling elite history for one neighborhood (one q_ind).
    Ingest whole batches, keep best score per genotype, prune to top-k_hist."""
    def __init__(self, k_hist: int = 10, dedup: bool = True):
        self._dq = deque(maxlen=k_hist)
        self._dedup = dedup

    def __len__(self) -> int:
        return len(self._dq)

    def _best_index_by_sig(self) -> dict[int, int]:
        """Map: genotype signature -> index in deque."""
        return {e.sig: i for i, e in enumerate(self._dq)}

    def update_from_batch(self, archs: np.ndarray, objs: np.ndarray, gen: int, q_idx: int) -> None:
        """Upsert the entire batch, then prune to top-k_hist by score (single-objective: objs[:,0])."""
        if archs is None or archs.size == 0:
            return
        if archs.ndim == 1: archs = archs[None, :]
        if objs.ndim == 1:  objs  = objs[None, :]

        sig2pos = self._best_index_by_sig() if self._dedup else {}

        # Upsert: keep best score seen for each genotype
        for a, o in zip(archs, objs):
            sig = _geno_sig(a)
            if self._dedup and sig in sig2pos:
                pos = sig2pos[sig]
                if float(o[0]) > float(self._dq[pos].objs[0]):  # single-objective
                    self._dq[pos] = NeighborhoodEntry(a.copy(), o.astype(float, copy=True), int(gen), int(q_idx), sig)
                continue
            self._dq.append(NeighborhoodEntry(a.copy(), o.astype(float, copy=True), int(gen), int(q_idx), sig))
            if self._dedup:
                sig2pos[sig] = len(self._dq) - 1

        # Prune to top-k_hist by score (descending)
        if len(self._dq) > self._dq.maxlen:
            arr = list(self._dq)
            arr.sort(key=lambda e: float(e.objs[0]), reverse=True)  # single-objective
            self._dq.clear()
            self._dq.extend(arr[: self._dq.maxlen])

    def topk(self, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Return top-k (A:(<=k,G), O:(<=k,M)) by current scores."""
        if not self._dq:
            return np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=float)
        arr = sorted(self._dq, key=lambda e: float(e.objs[0]), reverse=True)
        pick = arr[: min(k, len(arr))]
        A = np.vstack([e.arch for e in pick])
        O = np.vstack([e.objs for e in pick])
        return A, O

    def champion(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return the single best entry (A:(1,G), O:(1,M)) or (None,None) if empty."""
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