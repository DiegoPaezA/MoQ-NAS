"""Health check of finished (or running) experiments.

A trainer exception does not stop a search: the candidate is scored 0.0 and the
run goes on. This script makes those silent failures visible.

    python scripts/health_check.py <experiment_root> [--expect_device mps]

For every experiment found under the root (a folder with ``launcher.log`` or
``log_QNAS.txt``) it reports the devices the workers used, the candidates
recorded in ``archive/`` and how many of them have ``best_accuracy`` 0.0, and
the failure markers found in the logs. Exit code 1 if anything is wrong.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

LOG_NAMES = ('launcher.log', 'log_QNAS.txt')
MARKERS = {
    'scoring 0.0': 'candidates scored 0.0 after a training error',
    'Cannot re-initialize': 'CUDA initialised in a forked worker',
    'Traceback (most recent call last)': 'Python tracebacks',
    'All evaluation workers exited': 'every evaluation worker died',
    'out of memory': 'out-of-memory errors',
}
DEVICE_LINE = re.compile(r'Worker \d+ using device (\S+)')
PARAM = re.compile(r'^(device|best_accuracy):\s*(\S+)\s*$', re.M)


def check(exp_dir: Path, expect_device):
    problems = []
    markers, worker_devices = Counter(), Counter()
    for name in LOG_NAMES:
        log = exp_dir / name
        if not log.exists():
            continue
        text = log.read_text(errors='replace')
        for marker in MARKERS:
            markers[marker] += text.count(marker)
        worker_devices.update(DEVICE_LINE.findall(text))

    candidate_devices, zeros, total = Counter(), [], 0
    for params in sorted((exp_dir / 'archive').glob('*/training_params.txt')):
        fields = dict(PARAM.findall(params.read_text(errors='replace')))
        total += 1
        candidate_devices[fields.get('device', '?')] += 1
        try:
            if float(fields.get('best_accuracy', 'nan')) == 0.0:
                zeros.append(params.parent.name)
        except ValueError:
            pass

    for marker, count in markers.items():
        if count:
            problems.append(f"{count} x '{marker}' ({MARKERS[marker]})")
    if zeros:
        problems.append(f"{len(zeros)} archived candidates with best_accuracy 0.0: {zeros[:10]}")
    used = {d.split(':')[0] for d in list(worker_devices) + list(candidate_devices)} - {'?'}
    if 'cpu' in used:
        problems.append("training ran on the CPU")
    if expect_device and used and used != {expect_device}:
        problems.append(f"expected device '{expect_device}', found {sorted(used)}")

    print(f"{exp_dir}")
    print(f"  workers   : {dict(worker_devices) or 'no device lines in the logs'}")
    print(f"  candidates: {total} archived, devices {dict(candidate_devices)}, {len(zeros)} with accuracy 0.0")
    print("  status    : " + ("OK" if not problems else "PROBLEMS"))
    for problem in problems:
        print(f"    - {problem}")
    return not problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', type=Path)
    parser.add_argument('--expect_device', choices=['cuda', 'mps', 'cpu'], default=None,
                        help="Fail if the workers used any other device.")
    args = parser.parse_args()

    exp_dirs = sorted({log.parent for name in LOG_NAMES for log in args.root.rglob(name)})
    if not exp_dirs:
        print(f"No experiments (launcher.log / log_QNAS.txt) under {args.root}")
        return 1
    results = [check(d, args.expect_device) for d in exp_dirs]
    print(f"\n{sum(results)}/{len(results)} experiments healthy")
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
