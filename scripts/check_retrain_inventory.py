#!/usr/bin/env python3
"""Verify that the runs and candidates listed for retraining exist on this machine.

Reads retrain_matrices/literature_retrain_candidates.csv and, for every run
directory and candidate ID, checks the files that retrain_parallel.py needs:

  <run>/log_params_evolution.txt          (train / QNAS / fn_dict sections)
  <run>/archive/<id>/training_params.txt  (non-empty, with net_list)

It also reports the dataset config referenced by each run log
(config_path_dataset), because older logs point to configs/<name>.yaml, which
no longer exists (now dataset_configs/), and some have no such key at all.

Run directories are resolved under one or more --root folders, trying the CSV
`local_dir` as is, without its first component, and finally by searching for
a directory whose path ends in <experiment_*>/.../<run>.

Usage (from the MoQ-NAS repo root on the cluster):
    python scripts/check_retrain_inventory.py --root . --root /path/to/old/results
    python scripts/check_retrain_inventory.py --root . --tiers A B --roles best_acc knee compact

Outputs (in --out-dir, default retrain_inventory/):
    runs_report.csv     one row per run: resolved path, status, missing counts
    missing_ids.csv     candidates whose training_params.txt is missing/empty
    upload_files.txt    files to copy from the Mac's data/ folder for missing
                        runs/IDs (paths relative to data/, for rsync --files-from)
"""

import argparse
import csv
import os
import sys
from collections import OrderedDict

import yaml

DEFAULT_CSV = 'retrain_matrices/literature_retrain_candidates.csv'
REQUIRED_SECTIONS = ('train', 'QNAS', 'fn_dict')


def index_run_dirs(roots, max_depth=7):
    """Map run basename (e.g. exp22_repeat_1) -> list of absolute paths."""
    index = {}
    for root in roots:
        root = os.path.abspath(root)
        base_depth = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, _ in os.walk(root):
            depth = dirpath.count(os.sep) - base_depth
            # Never descend into per-candidate folders; they are huge and irrelevant.
            dirnames[:] = [d for d in dirnames
                           if d not in ('archive', 'results', 'snapshots', 'qpop_update', 'metrics')]
            if depth >= max_depth:
                dirnames[:] = []
            name = os.path.basename(dirpath)
            if '_repeat_' in name:
                index.setdefault(name, []).append(dirpath)
    return index


def resolve_run(local_dir, roots, index):
    parts = local_dir.split('/')
    for root in roots:
        for rel in (local_dir, '/'.join(parts[1:])):
            cand = os.path.join(root, rel)
            if rel and os.path.isfile(os.path.join(cand, 'log_params_evolution.txt')):
                return os.path.abspath(cand), 'direct'
    exp_dir = next((p for p in parts if p.startswith('experiment_')), None)
    run = parts[-1]
    tail = [p for p in parts[parts.index(exp_dir):]] if exp_dir else [run]
    matches = [p for p in index.get(run, []) if p.replace(os.sep, '/').endswith('/'.join(tail))]
    if not matches and exp_dir:
        matches = [p for p in index.get(run, []) if f'/{exp_dir}/' in p.replace(os.sep, '/') + '/']
    if len(matches) == 1:
        return matches[0], 'search'
    if len(matches) > 1:
        return ';'.join(matches), 'ambiguous'
    return '', 'not_found'


def check_log(run_path):
    log = os.path.join(run_path, 'log_params_evolution.txt')
    if not os.path.isfile(log):
        return 'missing', '', ''
    try:
        with open(log) as f:
            data = yaml.safe_load(f)
    except Exception as exc:  # noqa: BLE001 - report any parse failure
        return f'unparseable: {exc.__class__.__name__}', '', ''
    missing = [s for s in REQUIRED_SECTIONS if s not in (data or {})]
    if missing:
        return f'missing sections {missing}', '', ''
    train = data['train'] or {}
    return 'ok', str(train.get('config_path_dataset', '')), str(train.get('dataset', ''))


def check_candidate(run_path, cid):
    tp = os.path.join(run_path, 'archive', cid, 'training_params.txt')
    if not os.path.isfile(tp) or os.path.getsize(tp) == 0:
        return 'missing'
    try:
        with open(tp) as f:
            data = yaml.safe_load(f)
    except Exception:  # noqa: BLE001
        return 'unparseable'
    if not isinstance(data, dict) or not data.get('net_list'):
        return 'no_net_list'
    return 'ok'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', default=DEFAULT_CSV)
    ap.add_argument('--root', action='append', default=None,
                    help='Folder(s) where experiment results live. Repeatable. Default: current dir.')
    ap.add_argument('--tiers', nargs='*', default=None, help='Filter tiers (A B C D).')
    ap.add_argument('--roles', nargs='*', default=None,
                    help='Filter roles (e.g. best_acc knee compact strat10 front).')
    ap.add_argument('--out-dir', default='retrain_inventory')
    args = ap.parse_args()
    roots = args.root or ['.']

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    if args.tiers:
        rows = [r for r in rows if r['tier'] in args.tiers]
    if args.roles:
        rows = [r for r in rows if any(role in r['role'].split('+') for role in args.roles)]
    if not rows:
        sys.exit('No candidates after filtering.')

    runs = OrderedDict()
    for r in rows:
        runs.setdefault(r['local_dir'], {'meta': r, 'ids': []})['ids'].append(r['id'])

    print(f'Indexing run folders under: {roots} ...', flush=True)
    index = index_run_dirs(roots)

    os.makedirs(args.out_dir, exist_ok=True)
    report, missing_ids, upload = [], [], []
    for local_dir, info in runs.items():
        m = info['meta']
        path, how = resolve_run(local_dir, roots, index)
        row = dict(tier=m['tier'], case=m['case'], dataset=m['dataset'], algo=m['algo'], run=m['run'],
                   local_dir=local_dir, cluster_path=path, resolved_by=how, n_ids=len(info['ids']),
                   log='', config_path_dataset='', config_exists='', dataset_in_log='',
                   ids_ok=0, ids_missing=0, status='')
        if how in ('not_found', 'ambiguous'):
            row['status'] = 'RUN_NOT_FOUND' if how == 'not_found' else 'AMBIGUOUS'
            if how == 'not_found':
                upload += [f'{local_dir}/log_params_evolution.txt', f'{local_dir}/pareto_history.pkl']
                upload += [f'{local_dir}/archive/{cid}/training_params.txt' for cid in info['ids']]
            report.append(row)
            continue
        log_status, cfg, ds = check_log(path)
        row.update(log=log_status, config_path_dataset=cfg, dataset_in_log=ds)
        row['config_exists'] = 'yes' if cfg and os.path.isfile(cfg) else ('MISSING_KEY' if not cfg else 'no')
        if log_status != 'ok':
            upload.append(f'{local_dir}/log_params_evolution.txt')
        for cid in info['ids']:
            st = check_candidate(path, cid)
            if st == 'ok':
                row['ids_ok'] += 1
            else:
                row['ids_missing'] += 1
                missing_ids.append(dict(local_dir=local_dir, cluster_path=path, id=cid, problem=st))
                upload.append(f'{local_dir}/archive/{cid}/training_params.txt')
        problems = []
        if log_status != 'ok':
            problems.append('LOG')
        if row['ids_missing']:
            problems.append('IDS')
        if row['config_exists'] != 'yes':
            problems.append('DATASET_CONFIG')
        row['status'] = 'OK' if not problems else '+'.join(problems)
        report.append(row)

    with open(os.path.join(args.out_dir, 'runs_report.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)
    with open(os.path.join(args.out_dir, 'missing_ids.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['local_dir', 'cluster_path', 'id', 'problem'])
        w.writeheader()
        w.writerows(missing_ids)
    with open(os.path.join(args.out_dir, 'upload_files.txt'), 'w') as f:
        f.write('\n'.join(dict.fromkeys(upload)) + ('\n' if upload else ''))

    # Console summary
    print(f'\n{"tier":4} {"case":22} {"dataset":13} {"algo":6} {"run":16} {"status":22} ids_ok/n  path')
    for r in report:
        print(f'{r["tier"]:4} {r["case"]:22} {r["dataset"]:13} {r["algo"]:6} {r["run"]:16} '
              f'{r["status"]:22} {r["ids_ok"]:>3}/{r["n_ids"]:<4} {r["cluster_path"] or "-"}')
    n_ok = sum(r['status'] == 'OK' for r in report)
    print(f'\nRuns: {len(report)} | OK: {n_ok} | with problems: {len(report) - n_ok}')
    print(f'Candidates checked: {len(rows)} | missing/invalid: {len(missing_ids)}')
    cfgs = sorted({(r["config_path_dataset"] or "<none>", r["config_exists"]) for r in report if r["log"] == "ok"})
    print('Dataset configs referenced by the logs (path, exists here):')
    for c, e in cfgs:
        print(f'  {c:45} {e}')
    print(f'\nReports written to {args.out_dir}/ (runs_report.csv, missing_ids.csv, upload_files.txt)')
    if upload:
        print('To upload what is missing, run on the Mac from the data/ folder, e.g.:\n'
              f'  rsync -av --files-from={args.out_dir}/upload_files.txt ./ <user>@<cluster>:<results_root>/')


if __name__ == '__main__':
    main()
