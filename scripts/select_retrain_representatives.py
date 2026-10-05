#!/usr/bin/env python3
"""Stage 2-3 of the two-stage retrain selection (doc §2, "Selección en dos etapas").

After the 1-seed screening, per search run:
  2. re-Pareto the screened networks with the retrain VALIDATION accuracy (never test);
  3. pick representatives with rules fixed before looking at test:
       A  lowest validation error
       P  fewest parameters            F  fewest MACs
       K  knee: min Euclidean distance to the ideal after min-max normalisation within the front
       budgets: lowest validation error with params <= {0.25, 0.5, 1.0, 1.5} M (C1_triobj)
                or MACs <= {50, 100, 250, 500} M (AF_std_biobj)
     All rules are applied to the re-Pareto front of the case objectives:
       C1_triobj:    (val error, params)                 AF_std_biobj: (val error, MACs)
     Case 1 leaves out the search Time_CUDA, which is not reproducible (doc §8.1; decision 2026-10-05).

The output is a candidates CSV (same columns as literature_retrain_candidates.csv plus the rules and the
validation accuracy) that launch_retrain_protocol.py takes with --candidates, for the confirmation with
seeds 11 12 13. Test accuracy is never read. Only the standard library is used, so it runs on the Mac
against the per-host mirrors of scripts/sync_retrain_results.sh.

Usage:
  python scripts/select_retrain_representatives.py --case C1_triobj \\
      --runs-root ../retrain_2026/cluster/dualgpu1/retrain_2026/runs \\
      --out retrain_matrices/confirm_C1_triobj_F13v1c.csv
  (--partial selects only in the runs whose screening is complete, for checks before the end;
   --objectives overrides the front objectives, e.g. 'val_err params cuda_time' for the search objectives.)
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import OrderedDict

CASES = {
    # objectives of the re-Pareto front (all minimised), screened roles, budget axis and budgets
    'C1_triobj': dict(objectives=('val_err', 'params'),  # no Time_CUDA: not reproducible (doc §8.1)
                      roles=('best_acc', 'knee', 'compact', 'strat10'),
                      budget_key='params', budgets=(0.25e6, 0.5e6, 1.0e6, 1.5e6), budget_unit='M params'),
    'AF_std_biobj': dict(objectives=('val_err', 'macs'),
                         roles=None,  # the whole filtered front was screened (--roles all)
                         budget_key='macs', budgets=(50e6, 100e6, 250e6, 500e6), budget_unit='M MACs'),
}


def dominates(a, b, keys):
    return all(a[k] <= b[k] for k in keys) and any(a[k] < b[k] for k in keys)


def pareto(nets, keys):
    return [n for n in nets if not any(dominates(m, n, keys) for m in nets if m is not n)]


def tiebreak(n):
    return (n['val_err'], n['params'], n['macs'], n['id'])


def knee(front, keys):
    lo = {k: min(n[k] for n in front) for k in keys}
    hi = {k: max(n[k] for n in front) for k in keys}

    def dist(n):
        return math.sqrt(sum(((n[k] - lo[k]) / (hi[k] - lo[k]) if hi[k] > lo[k] else 0.0) ** 2 for k in keys))
    return min(front, key=lambda n: (dist(n), tiebreak(n)))


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(x) < 3:
        return float('nan')
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float('nan')


def load_screened(rows, runs_root, tag, seed):
    """Per run: screened rows joined with the retrain validation accuracy of `seed`; missing ids listed."""
    runs = OrderedDict()
    for r in rows:
        runs.setdefault(r['local_dir'], []).append(r)
    out = OrderedDict()
    for local_dir, rs in runs.items():
        path = os.path.join(runs_root, local_dir, f'retrain_results_{tag}.txt')
        results = json.load(open(path)) if os.path.isfile(path) else {}
        nets, missing = [], []
        for r in rs:
            rep = results.get(r['id'], {}).get(f'seed_{seed}', {})
            if rep.get('status') != 'OK' or rep.get('best_accuracy') is None:
                missing.append(r['id'])
                continue
            nets.append(dict(row=r, id=r['id'], val_acc=float(rep['best_accuracy']),
                             val_err=100.0 - float(rep['best_accuracy']), proxy_acc=float(r['proxy_acc']),
                             params=float(r['params']), macs=float(r['flops']) / 2.0,  # total_flops = 2 x MACs
                             cuda_time=float(r['cuda_time'] or 'nan')))
        out[local_dir] = dict(meta=rs[0], nets=nets, missing=missing)
    return out


def select_run(nets, spec):
    keys = spec['objectives']
    front = sorted(pareto(nets, keys), key=tiebreak)
    rules = OrderedDict()
    rules['A'] = min(front, key=tiebreak)
    rules['P'] = min(front, key=lambda n: (n['params'],) + tiebreak(n))
    rules['F'] = min(front, key=lambda n: (n['macs'],) + tiebreak(n))
    rules['K'] = knee(front, keys)
    for b in spec['budgets']:
        within = [n for n in front if n[spec['budget_key']] <= b]
        label = f"B{b / 1e6:g}"
        rules[label] = min(within, key=tiebreak) if within else None
    picked = OrderedDict()
    for label, n in rules.items():
        if n is not None:
            picked.setdefault(n['id'], (n, []))[1].append(label)
    return front, rules, picked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--case', required=True, choices=sorted(CASES))
    ap.add_argument('--runs-root', required=True, help='Folder that holds <local_dir>/retrain_results_<tag>.txt')
    ap.add_argument('--candidates', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                                         'retrain_matrices', 'literature_retrain_candidates.csv'))
    ap.add_argument('--tag', default='F13v1c')
    ap.add_argument('--seed', type=int, default=1, help='Screening seed (its validation drives the selection).')
    ap.add_argument('--out', required=True, help='Confirmation candidates CSV (a .md report is written next to it).')
    ap.add_argument('--partial', action='store_true', help='Select only in the runs whose screening is complete.')
    ap.add_argument('--objectives', nargs='+', choices=['val_err', 'params', 'macs', 'cuda_time'], default=None,
                    help="Override the front objectives (e.g. 'val_err params cuda_time' to include the search "
                         "Time_CUDA in the Case 1 front). Default: the case objectives.")
    args = ap.parse_args()
    spec = dict(CASES[args.case])
    if args.objectives:
        if 'val_err' not in args.objectives:
            ap.error("--objectives must include val_err")
        spec['objectives'] = tuple(args.objectives)

    with open(args.candidates) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = [r for r in reader if r['case'] == args.case]
    if spec['roles']:
        rows = [r for r in rows if any(role in r['role'].split('+') for role in spec['roles'])]
    runs = load_screened(rows, args.runs_root, args.tag, args.seed)

    incomplete = {ld: r['missing'] for ld, r in runs.items() if r['missing']}
    if incomplete and not args.partial:
        for ld, miss in incomplete.items():
            print(f"incomplete: {ld}: {len(miss)} of {len(runs[ld]['missing']) + len(runs[ld]['nets'])} "
                  f"screened nets without an OK seed_{args.seed}", file=sys.stderr)
        sys.exit('Screening not finished; nothing written (use --partial for a check on the complete runs).')

    out_rows, report = [], [f"# Representantes {args.case} (tag {args.tag}, validación de la semilla {args.seed})", "",
                            f"Frente: {', '.join(spec['objectives'])}. Presupuestos: "
                            f"{', '.join(f'{b / 1e6:g}' for b in spec['budgets'])} {spec['budget_unit']}.", ""]
    report.append("| Corrida | Algoritmo | Cribadas | En el frente (val) | Dominadas tras el retrain | "
                  "Spearman proxy–val | Representantes |")
    report.append("|---|---|---|---|---|---|---|")
    detail = []
    for ld, run in runs.items():
        if ld in incomplete:
            report.append(f"| {run['meta']['run']} | {run['meta']['algo']} | {len(run['nets'])}"
                          f"/{len(run['nets']) + len(run['missing'])} | — | — | — | incompleta |")
            continue
        nets = run['nets']
        front, rules, picked = select_run(nets, spec)
        rho = spearman([n['proxy_acc'] for n in nets], [n['val_acc'] for n in nets])
        report.append(f"| {run['meta']['run']} | {run['meta']['algo']} | {len(nets)} | {len(front)} | "
                      f"{len(nets) - len(front)} | {rho:.2f} | {len(picked)} |")
        detail += ["", f"## {ld}", "", "| Regla | id | val acc | params | MACs (M) |", "|---|---|---|---|---|"]
        for label, n in rules.items():
            if n is None:
                detail.append(f"| {label} | (ninguna dentro del presupuesto) | | | |")
            else:
                detail.append(f"| {label} | {n['id']} | {n['val_acc']:.2f} | {n['params'] / 1e6:.3f} M | "
                              f"{n['macs'] / 1e6:.1f} |")
        for cid, (n, labels) in picked.items():
            row = dict(n['row'])
            row['role'] = '+'.join(labels)
            row['val_acc_screen'] = f"{n['val_acc']:.2f}"
            out_rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(fields) + ['val_acc_screen'])
        w.writeheader()
        w.writerows(out_rows)
    report_path = os.path.splitext(args.out)[0] + '_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report + detail) + '\n')
    n_runs = len(runs) - len(incomplete)
    print(f"{args.case}: {len(out_rows)} representatives from {n_runs} complete run(s) -> {args.out} "
          f"(report: {report_path}); confirmation trainings with 3 seeds: {3 * len(out_rows)}")
    if incomplete:
        print(f"skipped {len(incomplete)} incomplete run(s) (--partial)")


if __name__ == '__main__':
    main()
