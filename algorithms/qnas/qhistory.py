from __future__ import annotations
import os, json, datetime, pickle
from typing import Dict, Tuple, Any, Optional

class QHistory:
    """Lightweight evaluation history.

    Persists each evaluation as an append-only JSONL record and maintains a
    small in-memory index with basic statistics (mean fitness, runs, last_gen).
    This module does **not** perform any reuse/lookup; it only logs.

    The history is useful for auditability, experiment analytics, and
    post-hoc diagnostics (e.g., distribution of fitness per gen or per quantum
    individual).
    """

    def __init__(self, base_dir: str, log_name: str = "arch_history.jsonl"):
        """Initialize the history writer and try to load the index.

        Args:
            base_dir: Base experiment directory where the `history/` folder
                will be created if it doesn't exist.
            log_name: Filename for the JSONL log (append-only).
        """
        self.dir = os.path.join(base_dir, "history")
        os.makedirs(self.dir, exist_ok=True)
        self.log_path = os.path.join(self.dir, log_name)
        self.idx_path = os.path.join(self.dir, "arch_index.pkl")
        # Index is only for quick stats; it is not used for cache/lookup.
        self.index: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load the stats index from disk, if present.

        Notes:
            This method is best-effort: if anything fails, it simply starts
            with an empty index.
        """
        try:
            if os.path.exists(self.idx_path):
                with open(self.idx_path, "rb") as f:
                    self.index = pickle.load(f) or {}
        except Exception:
            self.index = {}

    def flush(self) -> None:
        """Persist the in-memory stats index to disk.

        Notes:
            Called occasionally (e.g., every N writes) and at the end of
            a generation to avoid losing stats on crashes.
        """
        try:
            with open(self.idx_path, "wb") as f:
                pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            # Non-fatal: best-effort persistence.
            pass

    @staticmethod
    def key_from_row(row_like) -> Tuple[int, ...]:
        """Canonicalize an architecture row into a hashable key (tuple of ints).

        Accepts numpy arrays, lists, or any iterable of numbers and returns a
        tuple[int, ...] suitable as a dictionary key.

        Args:
            row_like: Array-like encoding of an architecture (e.g., encoded
                chromosome row or already-decoded but stable mapping).

        Returns:
            Tuple of ints representing the architecture key.
        """
        try:
            import numpy as np
            return tuple(int(x) for x in np.asarray(row_like).astype(int).ravel().tolist())
        except Exception:
            return tuple(int(x) for x in row_like)

    def log(
        self,
        *,
        gen: int,
        q_index: int,
        candidate_id: int,
        arch_key: Tuple[int, ...],
        primary_metric: str,
        fitness: float,
        metrics: Optional[dict] = None,
    ) -> None:
        """Append one evaluation record (JSONL) and update basic stats.

        Args:
            gen: Current generation number at the time of evaluation.
            q_index: Quantum individual id (from the parent map). Use -1 if unknown.
            candidate_id: Stable per-individual identifier if available; otherwise
                the position/index in the evaluated batch.
            arch_key: Canonical architecture key for exact identity of the design.
            primary_metric: Name of the primary objective (e.g., "accuracy").
            fitness: Scalar fitness value for the primary metric (raw value).
            metrics: Optional payload of additional metrics (dict). Can include
                full metric dict returned by the evaluator, including penalized
                versions and vectors for multi-objective setups.

        Side effects:
            - Appends one JSON object to `arch_history.jsonl`
            - Updates the in-memory index (mean, runs, last_gen), occasionally
              flushing it to disk to reduce I/O.
        """
        rec = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "gen": int(gen),
            "q_index": int(q_index),
            "candidate_id": int(candidate_id),
            "primary_metric": str(primary_metric),
            "fitness": float(fitness),
            "arch": list(arch_key),
            "metrics": metrics or {}
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Update basic stats (no lookup/reuse).
        H = self.index.get(arch_key)
        if H is None:
            H = {"mean": 0.0, "runs": 0, "last_gen": -1}
            self.index[arch_key] = H
        H["mean"] = (H["mean"] * H["runs"] + float(fitness)) / (H["runs"] + 1)
        H["runs"] += 1
        H["last_gen"] = int(gen)

        # Flush occasionally to avoid excessive I/O.
        if (H["runs"] % 50) == 0:
            self.flush()
