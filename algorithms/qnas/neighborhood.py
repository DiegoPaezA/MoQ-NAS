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
    # stable 64-bit signature for integer genotype
    a = arch.astype(np.int32, copy=False)
    return int(hashlib.blake2b(a.tobytes(), digest_size=8).hexdigest(), 16)

class NeighborhoodArchive:
    """Rolling elite buffer for one neighborhood (one q_ind)."""
    def __init__(self, k_hist: int = 10, dedup: bool = True):
        self._dq = deque(maxlen=k_hist)
        self._dedup = dedup

    def add_many(self, archs: np.ndarray, objs: np.ndarray, gen: int, q_idx: int):
        if archs is None or archs.size == 0:
            return
        if archs.ndim == 1: archs = archs[None, :]
        if objs.ndim == 1:  objs  = objs[None, :]
        seen = {e.sig for e in self._dq} if self._dedup else set()
        for a, o in zip(archs, objs):
            sig = _geno_sig(a)
            if sig in seen:
                continue
            self._dq.append(NeighborhoodEntry(
                arch=a.copy(), objs=o.astype(float, copy=True),
                gen=int(gen), q_idx=int(q_idx), sig=sig
            ))
            seen.add(sig)

    def topk(self, k: int = 1, maximize: bool = True):
        if not self._dq:
            return np.empty((0,0), dtype=int), np.empty((0,0), dtype=float)
        arr = list(self._dq)
        arr.sort(key=lambda e: float(e.objs[0]), reverse=maximize)  # single obj by default
        pick = arr[:min(k, len(arr))]
        A = np.vstack([e.arch for e in pick])
        O = np.vstack([e.objs for e in pick])
        return A, O

    def champion(self):
        A, O = self.topk(1)
        if A.size == 0: return None, None
        return A, O

class NeighborhoodRegistry:
    """Holds a NeighborhoodArchive per q_ind and provides bank-level ops."""
    def __init__(self, num_neighborhoods: int, k_hist: int = 10):
        self.neighborhoods = [NeighborhoodArchive(k_hist=k_hist) for _ in range(num_neighborhoods)]
        self.num_neighborhoods = num_neighborhoods

    def add_batch_by_parent(self, parent_map: np.ndarray, archs: np.ndarray, objs: np.ndarray, gen: int):
        # parent_map: (Nq*R,), archs:(Nq*R,G), objs:(Nq*R,M)
        for q in range(self.num_neighborhoods):
            idx = np.where(parent_map == q)[0]
            if idx.size:
                self.neighborhoods[q].add_many(archs[idx], objs[idx], gen=gen, q_idx=q)

    def champions(self):
        """Stack one champion per neighborhood -> A:(Nq,G), O:(Nq,M)."""
        A_list, O_list = []
        A_list, O_list = [], []
        for nb in self.neighborhoods:
            A, O = nb.champion()
            if A is None: return None, None
            A_list.append(A[0]); O_list.append(O[0])
        return np.vstack(A_list), np.vstack(O_list)

    def champions_or_best_of_batch(self, parent_map: np.ndarray, archs: np.ndarray, scores: np.ndarray):
        """
        Preferred: returns one champion per neighborhood.
        If a neighborhood archive is empty (or later ends up empty due to filtering),
        picks the best-of-batch for that neighborhood using `scores`.
        """
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores.ravel()

        Nq = self.num_neighborhoods
        champs_A, champs_O = [], []
        for q in range(Nq):
            # Try archive first
            A, O = self.neighborhoods[q].champion()
            if A is not None:
                champs_A.append(A[0]); champs_O.append(O[0])
                continue

            # Fallback: best-of-batch for this neighborhood
            idx = np.where(parent_map == q)[0]
            if idx.size == 0:
                # still empty: create a zero row or raise; here we soft-fail with zeros
                champs_A.append(np.zeros((archs.shape[1],), dtype=archs.dtype))
                champs_O.append(np.array([0.0], dtype=float))
            else:
                j = idx[int(np.argmax(scores[idx]))]
                champs_A.append(archs[j])
                champs_O.append(np.array([float(scores[j])], dtype=float))

        return np.vstack(champs_A), np.vstack(champs_O)

    def per_neighborhood_topk(self, k_each: int = 2):
        """Balanced pool: concat top-k from each neighborhood."""
        A_list, O_list = [], []
        for nb in self.neighborhoods:
            A, O = nb.topk(k_each)
            if A.size:
                A_list.append(A); O_list.append(O)
        if not A_list:
            return None, None
        return np.vstack(A_list), np.vstack(O_list)
