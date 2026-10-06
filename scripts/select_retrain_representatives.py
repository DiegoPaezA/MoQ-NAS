#!/usr/bin/env python3
"""Stage 2-3 of the two-stage retrain selection (doc §2, "Selección en dos etapas").

After the 1-seed screening, per search run:
  2. re-Pareto the screened networks with the retrain VALIDATION accuracy (never test);
  3. pick representatives with rules fixed before looking at test:
       A  lowest validation error
       K  knee: min Euclidean distance to the ideal after min-max normalisation within the front
       C  compact: least complexity among the networks within 5 pp of the validation accuracy of A
       budgets (C1_triobj, AF_std_biobj): lowest validation error with params <= {0.25, 0.5, 1.0, 1.5} M
                or MACs <= {50, 100, 250, 500} M
     The extreme "fewest parameters / MACs" points are not representatives: they are near-trivial networks
     (70-77% validation accuracy on CIFAR-10) and the reported networks must be useful and competitive
     (decision 2026-10-05).
     Unstable choices: if a rule keeps its network in fewer than 70% of 500 noise repetitions (validation accuracy
     + N(0, 0.36 pp)), the network that the same rule picks most often among the others is also confirmed (label
     '<rule>~'); both are reported and the test set is never used to choose between them (decision 2026-10-05).
     All rules are applied to the re-Pareto front of the case objectives:
       C1_triobj, C2_medmnist: (val error, params)       AF_std_biobj: (val error, MACs)
     The search Time_CUDA is left out everywhere: it is not reproducible on a shared GPU (doc §8.1).

The output is a candidates CSV (same columns as literature_retrain_candidates.csv plus the rules and the
validation accuracy) that launch_retrain_protocol.py takes with --candidates, for the confirmation with
seeds 11 12 13. Test accuracy is never read. Only the standard library is used, so it runs on the Mac
against the per-host mirrors of scripts/sync_retrain_results.sh.

Usage:
  python scripts/select_retrain_representatives.py --case C1_triobj \\
      --runs-root ../retrain_2026/cluster/dualgpu1/retrain_2026/runs \\
      --out retrain_matrices/confirm_C1_triobj_F13v1c.csv
  python scripts/select_retrain_representatives.py --case C2_medmnist --tag PMedW \\
      --runs-root <mirror>/retrain_2026/runs --out retrain_matrices/confirm_C2_medmnist_PMedW.csv
  (--partial selects only in the runs whose screening is complete, for checks before the end;
   --objectives overrides the front objectives, e.g. 'val_err params cuda_time' for the search objectives.)
"""

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import OrderedDict

COMPACT_MARGIN = 5.0  # pp of validation accuracy below rule A allowed for rule C
STABILITY_THRESHOLD = 0.70  # below this, the rule's most frequent alternative under noise is also confirmed

CASES = {
    # objectives of the re-Pareto front (all minimised), screened set, complexity axis, budgets
    'C1_triobj': dict(objectives=('val_err', 'params'),  # no Time_CUDA: not reproducible (doc §8.1)
                      roles=('best_acc', 'knee', 'compact', 'strat10'), strat5=False, complexity='params',
                      search_has_time=True,
                      budgets=(0.25e6, 0.5e6, 1.0e6, 1.5e6), budget_unit='M params'),
    'AF_std_biobj': dict(objectives=('val_err', 'macs'),
                         roles=None, strat5=False,  # the whole filtered front was screened (--roles all)
                         complexity='macs', budgets=(50e6, 100e6, 250e6, 500e6), budget_unit='M MACs'),
    'C2_medmnist': dict(objectives=('val_err', 'params'),  # no Time_CUDA (doc §8.1)
                        roles=None, strat5=True, search_has_time=True,  # boolean strat5 column: 5 per run
                        complexity='params', budgets=(), budget_unit='M params'),
}


def dominates(a, b, keys):
    return all(a[k] <= b[k] for k in keys) and any(a[k] < b[k] for k in keys)


def pareto(nets, keys):
    return [n for n in nets if not any(dominates(m, n, keys) for m in nets if m is not n)]


def tiebreak(n):
    return (n['val_err'], n['params'], n['macs'], n['id'])


def knee_distances(front, keys, log_keys=()):
    """Distance of each front member to the ideal after min-max scaling (log10 first for log_keys)."""
    val = {k: (lambda n, k=k: math.log10(n[k])) if k in log_keys else (lambda n, k=k: n[k]) for k in keys}
    lo = {k: min(val[k](n) for n in front) for k in keys}
    hi = {k: max(val[k](n) for n in front) for k in keys}
    return {n['id']: math.sqrt(sum(((val[k](n) - lo[k]) / (hi[k] - lo[k]) if hi[k] > lo[k] else 0.0) ** 2
                                   for k in keys)) for n in front}


def knee(front, keys, log_keys=()):
    d = knee_distances(front, keys, log_keys)
    return min(front, key=lambda n: (d[n['id']], tiebreak(n)))


def kendall_tau_b(x, y):
    n = len(x)
    if n < 3:
        return float('nan')
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = x[i] - x[j], y[i] - y[j]
            if a == 0 and b == 0:
                continue
            if a == 0:
                tx += 1
            elif b == 0:
                ty += 1
            elif (a > 0) == (b > 0):
                conc += 1
            else:
                disc += 1
    den = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / den if den else float('nan')


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


def select_run(nets, spec, margin=COMPACT_MARGIN, log_knee=False, objectives=None):
    keys = tuple(objectives or spec['objectives'])
    front = sorted(pareto(nets, keys), key=tiebreak)
    cx = spec['complexity']
    rules = OrderedDict()
    rules['A'] = min(front, key=tiebreak)
    rules['K'] = knee(front, keys, log_keys=[k for k in keys if k != 'val_err'] if log_knee else ())
    near = [n for n in front if n['val_acc'] >= rules['A']['val_acc'] - margin]
    rules['C'] = min(near, key=lambda n: (n[cx],) + tiebreak(n))
    for b in spec['budgets']:
        within = [n for n in front if n[cx] <= b]
        label = f"B{b / 1e6:g}"
        rules[label] = min(within, key=tiebreak) if within else None
    picked = OrderedDict()
    for label, n in rules.items():
        if n is not None:
            picked.setdefault(n['id'], (n, []))[1].append(label)
    return front, rules, picked


def analyse_run(nets, spec, front, rules, noise):
    """Validation-only evidence that supports the selection of one run (doc: selection analysis)."""
    keys = spec['objectives']
    cx = spec['complexity']
    out = {}
    # proxy -> validation
    out['spearman'] = spearman([n['proxy_acc'] for n in nets], [n['val_acc'] for n in nets])
    out['kendall'] = kendall_tau_b([n['proxy_acc'] for n in nets], [n['val_acc'] for n in nets])
    out['n_screened'], out['front_size'] = len(nets), len(front)
    out['n_dominated'] = len(nets) - len(front)
    ids = {n['id'] for n in front}

    def shifted(n, d):
        return dict(n, val_err=n['val_err'] - d)
    # dominated networks that would enter the front with +noise accuracy, and members that would leave with -noise
    out['borderline_out'] = sum(1 for n in nets if n['id'] not in ids
                                and not any(dominates(m, shifted(n, noise), keys) for m in nets if m is not n))
    out['borderline_in'] = sum(1 for n in front
                               if any(dominates(m, shifted(n, -noise), keys) for m in nets if m is not n))
    proxy_best = max(nets, key=lambda n: (n['proxy_acc'], -n['params']))
    out['proxy_best_is_A'] = proxy_best['id'] == rules['A']['id']
    proxy_reps = {n['id'] for n in nets if {'best_acc', 'knee', 'compact'} & set(n['row']['role'].split('+'))}
    out['proxy_reps'] = len(proxy_reps)
    out['proxy_reps_kept'] = len(proxy_reps & {n['id'] for n in rules.values() if n is not None})
    # stability of each pick: runner-up under the same rule and its margin
    per_rule = OrderedDict()
    order = sorted(front, key=tiebreak)
    a_acc = rules['A']['val_acc']
    for label, n in rules.items():
        if n is None:
            continue
        if label == 'A':
            alt = order[1] if len(order) > 1 else None
            gap = n['val_acc'] - alt['val_acc'] if alt else float('nan')
            within = gap < noise if alt else False
            note = f"runner-up {alt['id']} at {gap:.2f} pp" if alt else 'single-member front'
        elif label == 'C':
            slack = n['val_acc'] - (a_acc - COMPACT_MARGIN)
            near = sorted([m for m in front if m['val_acc'] >= a_acc - COMPACT_MARGIN], key=lambda m: m[cx])
            alt = near[1] if len(near) > 1 else None
            gap = slack
            within = slack < noise
            note = f"{slack:.2f} pp above the A-{COMPACT_MARGIN:g} threshold" + (f"; next: {alt['id']}" if alt else '')
        elif label == 'K':
            d = knee_distances(front, keys)
            ranked = sorted(front, key=lambda m: (d[m['id']], tiebreak(m)))
            alt = ranked[1] if len(ranked) > 1 else None
            gap = (d[alt['id']] - d[n['id']]) if alt else float('nan')
            within = False
            note = f"distance {d[n['id']]:.3f}; runner-up {alt['id']} at {d[alt['id']]:.3f}" if alt else ''
        else:  # budget
            b = float(label[1:]) * 1e6
            within_b = sorted([m for m in front if m[cx] <= b], key=tiebreak)
            alt = within_b[1] if len(within_b) > 1 else None
            gap = n['val_acc'] - alt['val_acc'] if alt else float('nan')
            within = gap < noise if alt else False
            note = f"runner-up {alt['id']} at {gap:.2f} pp" if alt else 'only network within the budget'
        per_rule[label] = dict(id=n['id'], val_acc=n['val_acc'], complexity=n[cx], runner_up=alt['id'] if alt else '',
                               gap=gap, within_noise=within, note=note)
    # sensitivity to the fixed parameters of the rules
    variants = OrderedDict([('C 3 pp', dict(margin=3.0)), ('C 7 pp', dict(margin=7.0)), ('K log', dict(log_knee=True))])
    if not math.isnan(nets[0]['cuda_time']) and 'cuda_time' not in keys and spec.get('search_has_time'):
        variants['+Time_CUDA'] = dict(objectives=tuple(keys) + ('cuda_time',))
    sens = OrderedDict()
    for name, kw in variants.items():
        f2, r2, _ = select_run(nets, spec, **kw)
        changed = [lab for lab in rules if rules[lab] is not None and (r2.get(lab) is None or r2[lab]['id'] != rules[lab]['id'])]
        jac = len(ids & {m['id'] for m in f2}) / len(ids | {m['id'] for m in f2})
        sens[name] = dict(changed=changed, front_jaccard=jac,
                          picks={lab: (r2[lab]['id'] if r2.get(lab) else '-') for lab in rules})
    # stability under seed noise: perturb every validation accuracy with N(0, noise) and redo the selection
    rng = random.Random(20261005)
    same = {lab: 0 for lab in per_rule}
    alts = {lab: {} for lab in per_rule}
    in_front = {n['id']: 0 for n in nets}
    draws = 500
    for _ in range(draws):
        noisy = []
        for n in nets:
            e = rng.gauss(0.0, noise)
            noisy.append(dict(n, val_acc=n['val_acc'] + e, val_err=n['val_err'] - e))
        f3, r3, _ = select_run(noisy, spec)
        for m in f3:
            in_front[m['id']] += 1
        for lab in per_rule:
            if r3.get(lab) is not None and r3[lab]['id'] == per_rule[lab]['id']:
                same[lab] += 1
            elif r3.get(lab) is not None:
                alts[lab][r3[lab]['id']] = alts[lab].get(r3[lab]['id'], 0) + 1
    for lab in per_rule:
        per_rule[lab]['stability'] = same[lab] / draws
        best_alt = max(alts[lab].items(), key=lambda kv: (kv[1], kv[0]), default=(None, 0))
        per_rule[lab]['alt_id'], per_rule[lab]['alt_freq'] = best_alt[0], best_alt[1] / draws
    out['front_stability'] = {i: c / draws for i, c in in_front.items()}
    out['per_rule'], out['sensitivity'] = per_rule, sens
    return out


def write_analysis(path_base, case, spec, analysed, noise):
    """Per-run and per-rule CSVs plus a markdown section that justifies the selection."""
    run_rows, rule_rows, md = [], [], []
    for ld, meta, a in analysed:
        run_rows.append(dict(case=case, dataset=meta['dataset'], algo=meta['algo'], run=meta['run'],
                             n_screened=a['n_screened'], front_size=a['front_size'], n_dominated=a['n_dominated'],
                             borderline_out=a['borderline_out'], borderline_in=a['borderline_in'],
                             spearman=f"{a['spearman']:.3f}", kendall=f"{a['kendall']:.3f}",
                             proxy_best_is_A=a['proxy_best_is_A'], proxy_reps=a['proxy_reps'],
                             proxy_reps_kept=a['proxy_reps_kept'],
                             **{f"sens_{k.replace(' ', '_').replace('+', 'plus_')}": '+'.join(v['changed']) or 'none'
                                for k, v in a['sensitivity'].items()}))
        for lab, r in a['per_rule'].items():
            rule_rows.append(dict(case=case, dataset=meta['dataset'], algo=meta['algo'], run=meta['run'], rule=lab,
                                  id=r['id'], val_acc=f"{r['val_acc']:.2f}", complexity=f"{r['complexity']:.0f}",
                                  runner_up=r['runner_up'], gap=f"{r['gap']:.3f}", within_noise=r['within_noise'],
                                  stability=f"{r['stability']:.3f}", noise_alt=r['alt_id'] or '',
                                  noise_alt_freq=f"{r['alt_freq']:.3f}",
                                  alt_confirmed=r['stability'] < STABILITY_THRESHOLD and bool(r['alt_id']),
                                  **{f"alt_{k.replace(' ', '_').replace('+', 'plus_')}": v['picks'][lab]
                                     for k, v in a['sensitivity'].items()}))
    for name, rows in (('runs', run_rows), ('rules', rule_rows)):
        if rows:
            with open(f"{path_base}_analysis_{name}.csv", 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    md += ["", "# Análisis de la selección (solo validación)", "",
           f"Ruido de referencia entre semillas: {noise:g} pp (sd máxima entre semillas de una misma red en los "
           "reentrenamientos previos). \"Frontera\": redes dominadas que entrarían al frente con +ruido "
           "(fuera) o miembros que saldrían con −ruido (dentro).", "",
           "## Proxy → validación por algoritmo", "",
           "| Algoritmo | Corridas | ρ Spearman (media) | τ Kendall (media) | Dominadas / cribadas | Frontera fuera / dentro | "
           "Mejor proxy = A | Representantes proxy conservados |", "|---|---|---|---|---|---|---|---|"]
    by_algo = OrderedDict()
    for ld, meta, a in analysed:
        by_algo.setdefault(meta['algo'], []).append(a)
    for algo, lst in by_algo.items():
        m = lambda key: sum(x[key] for x in lst) / len(lst)
        md.append(f"| {algo} | {len(lst)} | {m('spearman'):.2f} | {m('kendall'):.2f} | "
                  f"{sum(x['n_dominated'] for x in lst)} / {sum(x['n_screened'] for x in lst)} | "
                  f"{sum(x['borderline_out'] for x in lst)} / {sum(x['borderline_in'] for x in lst)} | "
                  f"{sum(x['proxy_best_is_A'] for x in lst)} / {len(lst)} | "
                  f"{sum(x['proxy_reps_kept'] for x in lst)} / {sum(x['proxy_reps'] for x in lst)} |")
    md += ["", "## Estabilidad de cada elección", "",
           "Estabilidad: fracción de 500 simulaciones con ruido N(0, ruido) en la accuracy de validación de todas las "
           "redes en las que la regla elige la misma red. Margen: pp de accuracy sobre la siguiente (A, presupuestos), pp "
           "sobre el umbral A−5 (C) o diferencia de distancia normalizada al ideal (K).", "",
           f"Si la estabilidad es < {100 * STABILITY_THRESHOLD:.0f}%, también se confirma la alternativa más frecuente de la "
           "misma regla en las simulaciones (etiqueta `<regla>~`).", "",
           "| Corrida | Regla | id | val acc | Margen | Estabilidad | Alternativa más frecuente | ¿Se confirma también? | Detalle |",
           "|---|---|---|---|---|---|---|---|---|"]
    for ld, meta, a in analysed:
        for lab, r in a['per_rule'].items():
            extra = r['stability'] < STABILITY_THRESHOLD and r['alt_id']
            alt = f"{r['alt_id']} ({100 * r['alt_freq']:.0f}%)" if r['alt_id'] else '—'
            md.append(f"| {meta['algo']} {meta['run']} | {lab} | {r['id']} | {r['val_acc']:.2f} | "
                      f"{r['gap']:.3f} | {100 * r['stability']:.0f}% | {alt} | "
                      f"{'sí' if extra else 'no'} | {r['note']} |")
    names = list(analysed[0][2]['sensitivity']) if analysed else []
    md += ["", "## Sensibilidad a los parámetros fijados", "",
           "Red que elige cada regla afectada con cada variante (— = ninguna cambia) y Jaccard del frente frente al de "
           "referencia.", "",
           "| Corrida | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for ld, meta, a in analysed:
        md.append(f"| {meta['algo']} {meta['run']} | " + " | ".join(
            (', '.join(f"{lab}: {a['per_rule'][lab]['id']}→{v['picks'][lab]}" for lab in v['changed']) or '—')
            + f" (J={v['front_jaccard']:.2f})" for v in a['sensitivity'].values()) + " |")
    tot = len(analysed)
    stab = [r['stability'] for _, _, a in analysed for r in a['per_rule'].values()]
    by_rule = OrderedDict()
    for _, _, a in analysed:
        for lab, r in a['per_rule'].items():
            by_rule.setdefault(lab, []).append(r['stability'])
    md += ["", "Estabilidad media por regla: " + ", ".join(
        f"{lab} {100 * sum(v) / len(v):.0f}%" for lab, v in by_rule.items()) + "."]
    md += ["", "Resumen: " + "; ".join(
        f"{n}: cambia alguna regla en {sum(1 for _, _, a in analysed if a['sensitivity'][n]['changed'])}/{tot} corridas"
        for n in names) + f"; elecciones dentro del ruido: "
           f"{sum(r['within_noise'] for _, _, a in analysed for r in a['per_rule'].values())}/"
           f"{sum(len(a['per_rule']) for _, _, a in analysed)}."]
    return md


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
    ap.add_argument('--noise-pp', type=float, default=0.36,
                    help='Seed-to-seed noise of the validation accuracy used to flag fragile choices (pp).')
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
    if spec['strat5']:
        rows = [r for r in rows if str(r.get('strat5', '')).lower() == 'true']
    runs = load_screened(rows, args.runs_root, args.tag, args.seed)

    incomplete = {ld: r['missing'] for ld, r in runs.items() if r['missing']}
    if incomplete and not args.partial:
        for ld, miss in incomplete.items():
            print(f"incomplete: {ld}: {len(miss)} of {len(runs[ld]['missing']) + len(runs[ld]['nets'])} "
                  f"screened nets without an OK seed_{args.seed}", file=sys.stderr)
        sys.exit('Screening not finished; nothing written (use --partial for a check on the complete runs).')

    out_rows, report = [], [f"# Representantes {args.case} (tag {args.tag}, validación de la semilla {args.seed})", "",
                            f"Frente: {', '.join(spec['objectives'])}. Compacto (C): menor complejidad a "
                            f"<= {COMPACT_MARGIN:g} pp de A. Presupuestos: "
                            f"{', '.join(f'{b / 1e6:g}' for b in spec['budgets']) or 'ninguno'} "
                            f"{spec['budget_unit'] if spec['budgets'] else ''}.", ""]
    report.append("| Corrida | Algoritmo | Cribadas | En el frente (val) | Dominadas tras el retrain | "
                  "Spearman proxy–val | Representantes |")
    report.append("|---|---|---|---|---|---|---|")
    detail, analysed = [], []
    for ld, run in runs.items():
        if ld in incomplete:
            report.append(f"| {run['meta']['run']} | {run['meta']['algo']} | {len(run['nets'])}"
                          f"/{len(run['nets']) + len(run['missing'])} | — | — | — | incompleta |")
            continue
        nets = run['nets']
        front, rules, picked = select_run(nets, spec)
        analysed.append((ld, run['meta'], analyse_run(nets, spec, front, rules, args.noise_pp)))
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
        a_run = analysed[-1][2]
        by_id = {m['id']: m for m in nets}
        for lab, r in a_run['per_rule'].items():
            if r['stability'] < STABILITY_THRESHOLD and r['alt_id']:
                picked.setdefault(r['alt_id'], (by_id[r['alt_id']], []))[1].append(f"{lab}~")
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
    analysis = write_analysis(os.path.splitext(args.out)[0], args.case, spec, analysed, args.noise_pp)
    with open(report_path, 'w') as f:
        f.write('\n'.join(report + detail + analysis) + '\n')
    n_runs = len(runs) - len(incomplete)
    print(f"{args.case}: {len(out_rows)} representatives from {n_runs} complete run(s) -> {args.out} "
          f"(report: {report_path}); confirmation trainings with 3 seeds: {3 * len(out_rows)}")
    if incomplete:
        print(f"skipped {len(incomplete)} incomplete run(s) (--partial)")


if __name__ == '__main__':
    main()
