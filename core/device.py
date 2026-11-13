"""Device selection shared by the evolution, fairness and retrain workers.

Supported backends: CUDA (Linux servers), MPS (Apple Silicon GPU) and CPU.

Importing this module queries no backend; every check happens inside the
functions. ``pick_device`` and ``device_slots`` must only be called from a
worker process: ``torch.cuda.is_available()``/``device_count()`` in the parent
of the evolution register a ``pthread_atfork`` handler in the NVIDIA driver and
break every worker forked afterwards.
"""
import os
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
    this module existed. On macOS a fall back to CPU is an error unless
    ``MOQNAS_ALLOW_CPU=1`` (see ``_cpu_fallback``).
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            gpu_idx = index % num_gpus
            torch.cuda.set_device(gpu_idx)
            return f'cuda:{gpu_idx}'
    if mps_available():
        return 'mps'
    return _cpu_fallback()


def _cpu_fallback() -> str:
    """``'cpu'``; on macOS only when ``MOQNAS_ALLOW_CPU=1``, otherwise an error.

    Training on a Mac without its GPU is never intended, and a silent fall back
    would only show up as a run that is many times slower. Linux keeps the
    historical behaviour (CPU without complaint).
    """
    if sys.platform == 'darwin':
        if os.environ.get('MOQNAS_ALLOW_CPU') != '1':
            raise RuntimeError(
                "The Apple GPU (MPS) is not available, so training would run on the CPU. "
                "Check the PyTorch build (python -m core.device) or set MOQNAS_ALLOW_CPU=1 "
                "to run on the CPU on purpose.")
        warnings.warn("The Apple GPU (MPS) is not available; training on the CPU (MOQNAS_ALLOW_CPU=1).")
    return 'cpu'


def require_gpu_on_macos() -> None:
    """Early check for launchers: fail before any work if a Mac has no MPS.

    A no-op on every other platform and it never queries CUDA, so it is safe in
    the parent process of the evolution.
    """
    if sys.platform == 'darwin' and not mps_available():
        _cpu_fallback()


def resolve_device(name=None) -> str:
    """Device for a single-process script from its ``--device`` argument.

    ``None``/``'auto'`` -> ``'cuda'`` > ``'mps'`` > ``'cpu'``; any other value is
    returned unchanged (``'cuda'`` stays ``'cuda'``), after checking that a
    requested ``'mps'`` exists. Not for the parent of the evolution: the
    automatic choice queries CUDA.
    """
    if name is None or str(name).lower() == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if mps_available():
            return 'mps'
        return _cpu_fallback()
    if _kind(name) == 'mps' and not mps_available():
        raise RuntimeError("device 'mps' was requested but the Apple GPU (MPS) is not available.")
    return str(name)


def autocast_device_type(device) -> str:
    """``device_type`` for ``torch.amp.autocast``/``GradScaler``: cuda, mps or cpu."""
    kind = _kind(device)
    return kind if kind in ('cuda', 'mps') else 'cpu'


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


def _preflight() -> int:
    """``python -m core.device``: report the environment and run one bf16 and
    one fp16 training step on the GPU. Exit code 0 only if a GPU trained."""
    import platform
    print(f"python   : {platform.python_version()} ({sys.platform}, {platform.machine()})")
    print(f"torch    : {torch.__version__}")
    try:
        device = resolve_device('auto')
    except RuntimeError as exc:
        print(f"device   : NONE - {exc}")
        return 1
    print(f"device   : {device}")
    if device == 'cpu':
        print("result   : FAIL - no GPU; training would run on the CPU")
        return 1
    if _kind(device) == 'cuda':
        print(f"gpus     : {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    ok = True
    kind = autocast_device_type(device)
    for precision, dtype in (('bf16', torch.bfloat16), ('fp16', torch.float16)):
        try:
            check_precision(device, precision)
            net = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3), torch.nn.BatchNorm2d(8), torch.nn.ReLU(),
                                      torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                      torch.nn.Linear(8, 2)).to(device)
            opt = torch.optim.AdamW(net.parameters())
            scaler = torch.amp.GradScaler(kind, enabled=(precision == 'fp16'))
            x = torch.randn(8, 3, 32, 32, device=device)
            y = torch.randint(0, 2, (8,), device=device)
            with torch.amp.autocast(kind, dtype=dtype):
                out = net(x)
                loss = torch.nn.functional.cross_entropy(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            synchronize(device)
            if out.dtype != dtype or not torch.isfinite(loss).item():
                raise RuntimeError(f"autocast produced {out.dtype}, loss {loss.item()}")
            print(f"{precision} step: ok (output {out.dtype}, loss {loss.item():.4f})")
        except Exception as exc:
            ok = False
            print(f"{precision} step: FAIL - {exc}")
    print("result   : " + ("OK - the GPU trains in mixed precision" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(_preflight())
