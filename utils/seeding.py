"""Global RNG seeding utilities for reproducible evolutionary runs.

Centralizes the seeding of every random number generator used by the
search pipeline (``random``, ``numpy``, ``torch`` CPU/CUDA) so that
experiments launched with the same ``--seed`` are comparable run-to-run.
"""
import os
import random

import numpy as np
import torch


def set_global_seeds(seed: int = 42, deterministic: bool = True) -> None:
    """Seed all RNGs used by the evolutionary search for reproducible runs.

    Parameters
    ----------
    seed : int, default 42
        Seed applied to ``random``, ``numpy`` and ``torch`` (CPU and CUDA).
    deterministic : bool, default True
        If True, force cuDNN into deterministic mode and disable benchmark
        autotuning. Trades some GPU throughput for run-to-run stability.

    Notes
    -----
    Architecture sampling is fully deterministic under this seeding:
    repeated runs produce bit-exact chromosomes/parameter counts.
    Trained accuracy is NOT reproducible run-to-run, because candidates
    are trained in parallel worker threads that draw weight
    initialization from the shared global RNG, so the interleaving is
    scheduler-dependent (observed spread ~1-3 pp on 1-epoch smoke runs).
    Operator correctness in Block D is therefore verified with synthetic
    parity scripts and architecture-level diffs, not bit-exact
    end-to-end accuracy/hypervolume comparisons.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
