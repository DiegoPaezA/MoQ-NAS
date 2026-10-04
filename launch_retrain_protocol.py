#!/usr/bin/env python3
"""Protocol-driven retrain launcher (retrain_2026 stage).

Builds one retrain_parallel.py job per search run listed in the candidates CSV,
with EVERY hyperparameter taken from a single protocol YAML and passed
explicitly on the command line. Nothing is inherited from the search
log_params_evolution.txt (search logs differ in precision, accuracy
aggregation and dataset-config paths), so all algorithms of a case are
retrained under exactly the same protocol.

Resolution order for each job (later wins):
    protocol.common -> protocol.datasets[<dataset>] -> protocol.cases[<case>]
    -> protocol.profiles[<profile>] (if --profile) -> CLI overrides

Usage (from the MoQ-NAS repo root on the cluster):
    python launch_retrain_protocol.py --cases C1_triobj --roles best_acc knee compact strat10 front \\
        --seeds 1 --tag F13v1 --gpus 0 1 --dry-run
    python launch_retrain_protocol.py --cases C3_fairness_two C3_fairness_three --roles all \\
        --profile fairness_R1 --seeds 1 2 3 --tag fairR1 --gpus 0 1

Outputs, inside each run folder: archive/<id>/retrain_<tag>_s<seed>/,
retrain_results_<tag>.txt and retrain_failures_<tag>.csv. One log per job and a
launch manifest go to --logs-dir.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import OrderedDict

import yaml

RETRAIN_SCRIPT = 'retrain_parallel.py'

# protocol key -> retrain_parallel.py option (value options)
VALUE_KEYS = {
    'dataset': 'dataset', 'data_path': 'data_path', 'config_path_dataset': 'config_path_dataset',
    'max_epochs': 'max_epochs', 'epochs_to_eval': 'epochs_to_eval',
    'patience_retrain': 'patience_retrain', 'delta_fraction': 'delta_fraction',
    'lr_scheduler': 'lr_scheduler', 'optimizer': 'optimizer',
    'learning_rate': 'learning_rate', 'weight_decay': 'weight_decay',
    'precision': 'precision', 'grad_clip_norm': 'grad_clip_norm',
    'eval_window_agg': 'eval_window_agg', 'train_split': 'train_split', 'split_seed': 'split_seed',
    'limit_data_value': 'limit_data_value', 'num_workers': 'num_workers',
    'batch_size': 'batch_size', 'eval_batch_size': 'eval_batch_size',
    'augmentation_policy': 'augmentation_policy',
}
FLAG_KEYS = ('data_augmentation', 'limit_data', 'keep_metrics')
# Keys every job must define after resolution (fail fast instead of falling back to CLI defaults).
REQUIRED = ('dataset', 'data_path', 'config_path_dataset', 'max_epochs', 'epochs_to_eval',
            'patience_retrain', 'lr_scheduler', 'optimizer', 'learning_rate', 'weight_decay',
            'precision', 'eval_window_agg', 'batch_size', 'eval_batch_size', 'train_split', 'split_seed')


def case_key(case: str) -> str:
    """CSV case -> protocol `cases` key (both fairness formulations share one entry)."""
    return 'C3_fairness' if case.startswith('C3_fairness') else case


def resolve(protocol: dict, dataset: str, case: str, profile: str | None, cli: dict) -> dict:
    spec = dict(protocol.get('common', {}))
    if dataset not in protocol.get('datasets', {}):
        raise KeyError(f"Dataset '{dataset}' not defined in protocol 'datasets'.")
    spec.update(protocol['datasets'][dataset])
    spec.update(protocol.get('cases', {}).get(case_key(case), {}))
    if profile:
        spec.update(protocol['profiles'][profile])
    spec.update({k: v for k, v in cli.items() if v is not None})
    if spec.get('epochs_to_eval') is None and spec.get('max_epochs') is not None:
        spec['epochs_to_eval'] = spec['max_epochs']
    if spec.get('patience_retrain') is None and spec.get('max_epochs') is not None:
        spec['patience_retrain'] = spec['max_epochs']
    missing = [k for k in REQUIRED if spec.get(k) is None]
    if missing:
        raise ValueError(f"Protocol leaves {missing} undefined for case={case}, dataset={dataset}, "
                         f"profile={profile}. Pass them explicitly (e.g. --max-epochs).")
    return spec


def build_argv(run_path: str, ids: list[str], spec: dict, seeds: list[int], tag: str,
               workers: int, python: str) -> list[str]:
    argv = [python, RETRAIN_SCRIPT, '--experiment_path', run_path, '--ids', *ids]
    for key, opt in VALUE_KEYS.items():
        if key not in spec:
            continue
        value = spec[key]
        if key == 'grad_clip_norm' and value is None:
            value = 'none'
        if key == 'lr_scheduler' and value in (None, 'None', 'none'):
            value = 'None'
        if value is None:
            continue
        argv += [f'--{opt}', str(value)]
    for flag in FLAG_KEYS:
        if spec.get(flag):
            argv.append(f'--{flag}')
    argv += ['--seeds', *[str(s) for s in seeds], '--tag', tag,
             '--max_parallel_workers', str(workers), '--log_level', 'INFO']
    return argv


def load_jobs(args, protocol):
    with open(args.candidates) as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r['case'] in args.cases]
    if args.algos:
        rows = [r for r in rows if r['algo'] in args.algos]
    if args.datasets:
        rows = [r for r in rows if r['dataset'] in args.datasets]
    if args.runs:
        rows = [r for r in rows if r['local_dir'] in args.runs]
    if 'all' not in args.roles:
        rows = [r for r in rows if any(role in r['role'].split('+') for role in args.roles)]
    if args.ids:
        rows = [r for r in rows if r['id'] in args.ids]

    cli = {'max_epochs': args.max_epochs, 'epochs_to_eval': args.max_epochs,
           'patience_retrain': args.max_epochs} if args.max_epochs else {}
    if args.smoke:
        # Code-path check only: 2 epochs on a 2k-image subset, one candidate per case.
        cli.update(max_epochs=2, epochs_to_eval=2, patience_retrain=2,
                   limit_data=True, limit_data_value=2000)
    if args.smoke:
        # Cheapest candidate first, so the smoke test checks the code path quickly.
        def cost(r):
            return (float(r.get('flops') or 0) or float(r.get('params') or 0))
        rows = sorted(rows, key=cost)
    jobs = OrderedDict()
    for r in rows:
        job = jobs.setdefault(r['local_dir'], {'meta': r, 'ids': []})
        job['ids'].append(r['id'])
    if args.smoke:
        first_per_case = OrderedDict()
        for local_dir, job in jobs.items():
            first_per_case.setdefault(job['meta']['case'], (local_dir, job))
        jobs = OrderedDict((ld, dict(job, ids=job['ids'][:1])) for ld, job in first_per_case.values())
    out = []
    for local_dir, job in jobs.items():
        m = job['meta']
        spec = resolve(protocol, m['dataset'], m['case'], args.profile, cli)
        run_path = os.path.join(args.runs_root, local_dir)
        if not os.path.isfile(os.path.join(run_path, 'log_params_evolution.txt')):
            raise FileNotFoundError(f"Run not found: {run_path} (check --runs-root)")
        out.append(dict(local_dir=local_dir, case=m['case'], dataset=m['dataset'], algo=m['algo'],
                        run=m['run'], ids=job['ids'], spec=spec,
                        argv=build_argv(run_path, job['ids'], spec, args.seeds, args.tag,
                                        args.workers_per_job, args.python)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--protocol', default='retrain_matrices/protocol_F13v1.yaml')
    ap.add_argument('--candidates', default='retrain_matrices/literature_retrain_candidates.csv')
    ap.add_argument('--runs-root', default='retrain_2026/runs')
    ap.add_argument('--logs-dir', default='retrain_2026/logs')
    ap.add_argument('--cases', nargs='+', required=True,
                    help='C1_triobj AF_std_biobj C2_medmnist C3_fairness_two C3_fairness_three')
    ap.add_argument('--roles', nargs='+', default=['best_acc', 'knee', 'compact', 'strat10'],
                    help="Candidate roles from the CSV, or 'all'.")
    ap.add_argument('--algos', nargs='*', default=None)
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--runs', nargs='*', default=None, help='Restrict to these local_dir values.')
    ap.add_argument('--ids', nargs='*', default=None, help='Restrict to these candidate IDs.')
    ap.add_argument('--profile', default=None, help='Protocol profile (fairness_R1, fairness_R2).')
    ap.add_argument('--max-epochs', type=int, default=None,
                    help='Override max_epochs (also epochs_to_eval and patience). Required for fairness_R2.')
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--tag', required=True, help='Protocol tag, e.g. F13v1, fairR1, fairR2.')
    ap.add_argument('--gpus', type=int, nargs='+', default=[0])
    ap.add_argument('--jobs-per-gpu', type=int, default=1, help='Concurrent run-jobs per GPU.')
    ap.add_argument('--workers-per-job', type=int, default=2,
                    help='Candidates trained concurrently inside one job (retrain_parallel workers).')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--smoke', action='store_true',
                    help="Smoke test: 2 epochs, 2000 images, first run and first candidate of each case; "
                         "tag becomes <tag>_smoke so real results are never touched.")
    args = ap.parse_args()
    if args.smoke:
        args.tag = f"{args.tag}_smoke"

    with open(args.protocol) as f:
        protocol = yaml.safe_load(f)
    jobs = load_jobs(args, protocol)
    if not jobs:
        sys.exit('No jobs after filtering.')

    n_trainings = sum(len(j['ids']) for j in jobs) * len(args.seeds)
    print(f"# protocol={protocol.get('protocol')} profile={args.profile} tag={args.tag} "
          f"jobs={len(jobs)} candidates={sum(len(j['ids']) for j in jobs)} seeds={args.seeds} "
          f"trainings={n_trainings}", file=sys.stderr)

    if args.dry_run:
        for j in jobs:
            print(f"# {j['case']} {j['algo']} {j['run']} ids={len(j['ids'])} precision={j['spec']['precision']}")
            print(' '.join(j['argv']))
        return 0

    os.makedirs(args.logs_dir, exist_ok=True)
    manifest_path = os.path.join(args.logs_dir, f'launch_manifest_{args.tag}.csv')
    new_manifest = not os.path.isfile(manifest_path)
    manifest = open(manifest_path, 'a', newline='')
    mw = csv.writer(manifest)
    if new_manifest:
        mw.writerow(['tag', 'local_dir', 'n_ids', 'seeds', 'gpu', 'start', 'end', 'returncode', 'log', 'cmd'])

    slots = [g for g in args.gpus for _ in range(max(1, args.jobs_per_gpu))]
    pending, running, failed = list(jobs), {}, []
    while pending or running:
        free = [i for i in range(len(slots)) if i not in running]
        while free and pending:
            si, job = free.pop(0), pending.pop(0)
            safe = job['local_dir'].replace('/', '__')
            log_path = os.path.join(args.logs_dir, f"{args.tag}__{safe}.log")
            log = open(log_path, 'a')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slots[si]))
            proc = subprocess.Popen(job['argv'], env=env, stdout=log, stderr=subprocess.STDOUT)
            running[si] = (job, proc, log, time.strftime('%Y-%m-%d %H:%M:%S'), log_path)
            print(f"[launch] GPU{slots[si]} {job['case']} {job['algo']} {job['run']} ids={len(job['ids'])}")
        time.sleep(5)
        for si, (job, proc, log, start, log_path) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            mw.writerow([args.tag, job['local_dir'], len(job['ids']), ' '.join(map(str, args.seeds)),
                         slots[si], start, time.strftime('%Y-%m-%d %H:%M:%S'), rc, log_path,
                         ' '.join(job['argv'])])
            manifest.flush()
            status = 'OK' if rc == 0 else f'FAIL(rc={rc})'
            if rc != 0:
                failed.append(job['local_dir'])
            print(f"[done]   GPU{slots[si]} {status}: {job['case']} {job['algo']} {job['run']}")
            del running[si]
    manifest.close()
    print(f"\n=== {len(jobs)} job(s); {len(failed)} with failures (see retrain_failures_{args.tag}.csv in each run) ===")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
