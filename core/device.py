"""Device selection shared by the evolution, fairness and retrain workers.

Supported backends: CUDA (Linux servers), MPS (Apple Silicon GPU) and CPU.

Importing this module queries no backend; every check happens inside the
functions. ``pick_device`` and ``device_slots`` must only be called from a
worker process: ``torch.cuda.is_available()``/``device_count()`` in the parent
of the evolution register a ``pthread_atfork`` handler in the NVIDIA driver and
break every worker forked afterwards.
"""
import sys
import warnings

import torch


def mps_available() -> bool:
    """True when the Apple GPU backend can be used."""
    mps = getattr(torch.backends, 'mps', None)
    return mps is not None and mps.is_available()


def pick_device(index: int = 0) -> str:
    """Device for ONE worker: ``'cuda:<index % n>'`` > ``'mps'`` > ``'cpu'``.

    On CUDA it also makes that GPU the current one, as the workers did before
    this module existed. On macOS a fall back to CPU is announced with a
    warning, because training there without the GPU is rarely intended.
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            gpu_idx = index % num_gpus
            torch.cuda.set_device(gpu_idx)
            return f'cuda:{gpu_idx}'
    if mps_available():
        return 'mps'
    if sys.platform == 'darwin':
        warnings.warn("The Apple GPU (MPS) is not available; this worker will train on the CPU.")
    return 'cpu'


def device_slots() -> int:
    """Number of GPUs a worker can see: the CUDA count, 1 with MPS, 0 on CPU."""
    try:
        num_gpus = torch.cuda.device_count()
    except Exception:
        num_gpus = 0
    if num_gpus > 0:
        return num_gpus
    return 1 if mps_available() else 0


def _kind(device) -> str:
    return str(device).split(':')[0]


def synchronize(device) -> None:
    """Wait for the pending GPU work of ``device`` (no-op on CPU)."""
    kind = _kind(device)
    if kind == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif kind == 'mps':
        torch.mps.synchronize()


def empty_cache(device) -> None:
    """Release the cached GPU memory of ``device`` (no-op on CPU)."""
    kind = _kind(device)
    if kind == 'cuda':
        torch.cuda.empty_cache()
    elif kind == 'mps':
        torch.mps.empty_cache()


def check_precision(device, precision: str) -> None:
    """Fail early when ``precision`` cannot run on ``device``.

    Mixed precision on MPS needs PyTorch >= 2.8: older versions silently
    disable bf16 autocast and their GradScaler fails on the first fp16 step.
    """
    if _kind(device) != 'mps' or precision not in ('fp16', 'bf16'):
        return
    version = tuple(int(v) for v in torch.__version__.split('+')[0].split('.')[:2])
    if version < (2, 8):
        raise RuntimeError(
            f"precision='{precision}' on the Apple GPU (MPS) needs PyTorch >= 2.8, "
            f"found {torch.__version__}. Use precision='fp32' or upgrade PyTorch.")
