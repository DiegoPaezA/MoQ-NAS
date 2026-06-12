# Running Experiments with the Launcher — A Practical Guide

This guide explains how to run MoQ-NAS experiments with the matrix launcher
(`launch.py`): how to write a matrix file, how the config files fit together,
how to manage GPUs (including using several GPUs for a single experiment), and
how to resume an interrupted batch so you can simply relaunch and have the
system pick up where it left off.

If you only need to launch **one** run, you can call `run_all_evolution.py`
directly (see [§7](#7-single-run-without-the-launcher)). For anything with
repeats, several algorithms, or several GPUs, use the launcher.

---

## 1. The two layers of configuration

There are two separate kinds of YAML files; do not confuse them.

| Layer | Directory | Defines |
|---|---|---|
| **Experiment config** | `experiment_configs/` | The search space (`function_dict`), algorithm hyperparameters, training settings (`max_epochs`, `optimizer`, `precision`, `eval_window_agg`), the **objectives**, and the metrics. One config = one experiment recipe. |
| **Dataset metadata** | `dataset_configs/` | Per-dataset metadata (`cifar10.yaml`, …) and the objective senses (`cfg_obj.json`, which says whether each objective is maximized or minimized). |
| **Experiment matrix** | `experiment_matrices/` | A *batch* description for the launcher: which configs/algorithms to run, how many repeats, which GPUs, and whether to resume. It points at the experiment configs above. |

A matrix never duplicates an experiment config — it **references** one per cell.

### Key experiment-config fields you will touch most

```yaml
train:
  max_epochs: 50
  epochs_to_eval: 5
  eval_window_agg: max          # max | mean | last — how best_accuracy is
                                # aggregated over the last epochs_to_eval epochs
  optimizer: AdamW
  precision: fp16               # fp32 | fp16 | bf16 (bf16 needs Ampere/Ada)
  multi_objective: true
  objectives: ['best_accuracy', 'total_flops']
  metrics:
    - name: Accuracy
    - name: HardwareMetrics     # produces total_params, total_flops, cuda_inference_time
    - name: ScalarizedFitness
```

- **`objectives`** are validated at startup against `dataset_configs/cfg_obj.json`
  and the configured `metrics`. A typo or an unproducible objective aborts the run
  with a clear message.
- Using deterministic objectives (`total_flops`, `total_params`) instead of the
  measured `cuda_inference_time` makes a whole run **bit-reproducible** — useful
  for comparisons and for testing resume.
- **`eval_window_agg`** only changes how the scalar proxy accuracy is computed; the
  saved model (`best_model.pth`) is always the best-val-accuracy epoch.

---

## 2. Anatomy of a matrix file

```yaml
exp_root: experiment_cifar10_acc_flops   # root output directory
gpus: [0]                # GPU pool
gpus_per_run: 1          # GPUs assigned to each run (see §4)
repeats: 3               # how many times to repeat each experiment
seed_base: 42            # repeat i (1-based) uses seed = seed_base + i
resume: false            # or pass --resume on the command line (see §5)

defaults:                # arguments shared by every run -> --key value / --key (bool true)
  data_path: datasets/cifar10_data
  dataset: cifar10
  config_path_dataset: dataset_configs/cifar10.yaml
  log_level: INFO

experiments:
  - algo: moqnas
    config: experiment_configs/cifar_mo/config0_3_acc_flops.yaml
    name: exp10
    overrides: {optimizer: AdamW}        # per-cell args, override defaults
    flags: [--multi_objective]           # literal flags appended verbatim
```

How values become CLI arguments (`_arg_tokens`):

- `key: value` → `--key value`
- `key: true` → `--key` (a store-true flag); `key: false` → omitted
- `key: [a, b]` → `--key a b`
- `flags: [--x, --y]` → appended literally (use this for store-true flags like
  `--multi_objective` or for the inverted `--no-...` rule flags)

Each cell expands to one `run_all_evolution.py` call; its output directory is
`<exp_root>/<algo>/<name>_repeat_<i>`. The launcher writes the exact command
(with `CUDA_VISIBLE_DEVICES`) into `launch_command.txt` in that directory.

**Always preview first** — `--dry-run` prints every expanded command and runs
nothing:

```bash
python launch.py experiment_matrices/your_matrix.yaml --dry-run
```

> Note: `population_size` and `num_generations` apply to the GA family
> (`ga`, `nsga2`, `nsga3`, `moead`). MO-QNAS reads its population from the
> config; `--num_generations` only overrides its `max_generations`.

---

## 3. Example: an accuracy + FLOPs experiment

This is the matrix shipped as `experiment_matrices/acc_flops.yaml`: the four
multi-objective algorithms on the `(best_accuracy, total_flops)` objective set,
3 repeats each, on one GPU.

```yaml
exp_root: experiment_cifar10_acc_flops
gpus: [0]
repeats: 3
seed_base: 42
defaults:
  data_path: datasets/cifar10_data
  dataset: cifar10
  config_path_dataset: dataset_configs/cifar10.yaml
  log_level: INFO
  multi_objective: true       # keep the config's multi-objective setting
  population_size: 20         # nsga2/nsga3/moead (ignored by moqnas)
  num_generations: 150
experiments:
  - {algo: moqnas, config: experiment_configs/cifar_mo/config0_3_acc_flops.yaml, name: moqnas}
  - {algo: nsga2,  config: experiment_configs/cifar_mo/config0_3_acc_flops.yaml, name: nsga2}
  - {algo: nsga3,  config: experiment_configs/cifar_mo/config0_3_acc_flops.yaml, name: nsga3,
     overrides: {ref_divisions: 12}}
  - {algo: moead,  config: experiment_configs/cifar_mo/config0_3_acc_flops.yaml, name: moead,
     overrides: {ref_divisions: 12, moead_T: 20, moead_scalar: tchebycheff, moead_pneighbor: 0.9}}
```

The objective set comes from the config
(`config0_3_acc_flops.yaml` declares `objectives: [best_accuracy, total_flops]`).
To run a different objective set, point the cells at a different config — e.g.
`config0_3_flops.yaml` for `[best_accuracy, total_params, total_flops]`, or write
your own config with the `objectives` and `metrics` you need.

> `multi_objective: true` is required in `defaults` for multi-objective runs: the
> CLI flag defaults to false and would otherwise override the config's value.

Run it:

```bash
python launch.py experiment_matrices/acc_flops.yaml --dry-run   # preview
python launch.py experiment_matrices/acc_flops.yaml             # run
```

---

## 4. Managing GPUs

The launcher groups the `gpus` pool into **slots** of `gpus_per_run` GPUs each;
one run occupies one slot at a time, and as many runs execute concurrently as
there are slots.

### Run many experiments in parallel (one GPU each)

```yaml
gpus: [0, 1, 2, 3]
gpus_per_run: 1     # (default) -> 4 slots -> up to 4 runs at once, one GPU each
```

This is the throughput-oriented setup: with 12 cells (4 algorithms × 3 repeats)
and 4 slots, four runs proceed at a time and the rest queue automatically. A
single failed run never stops the others; a summary is printed at the end.

### Use several GPUs for one experiment

Set `gpus_per_run` to the number of GPUs each run should use. The pool is split
into slots of that size:

```yaml
gpus: [0, 1]
gpus_per_run: 2     # -> 1 slot of 2 GPUs; each run sees CUDA_VISIBLE_DEVICES=0,1
```

Within that run, candidate trainings are distributed across both GPUs by the
work-stealing scheduler (a single GPU is never oversubscribed by two runs). With
`gpus: [0,1,2,3]` and `gpus_per_run: 2` you get two slots (`[0,1]` and `[2,3]`):
two runs at a time, each on two GPUs.

> When is multi-GPU-per-run worth it? Only when one run actually saturates a
> single GPU (large population, full epochs/models). For many small candidates,
> running more experiments in parallel (`gpus_per_run: 1`) uses the hardware
> better, because the per-run setup cost dominates over GPU compute.

---

## 5. Resuming an interrupted batch

Every run — MO-QNAS and the whole GA family — writes a `checkpoint.pkl` at each
generation boundary. If a batch is interrupted (power loss, preemption, a manual
kill), **relaunch the same matrix with `--resume`**:

```bash
python launch.py experiment_matrices/acc_flops.yaml --resume
```

What happens:

1. The launcher prints a **pre-flight status** for every cell — its checkpoint's
   completed generation, or "no checkpoint (fresh)":

   ```
   [resume] checkpoint status per cell:
              gen 137  experiment_cifar10_acc_flops/moqnas/moqnas_repeat_1
   no checkpoint (fresh)  experiment_cifar10_acc_flops/nsga2/nsga2_repeat_1
   ...
   ```

2. Each cell is relaunched with `--resume`:
   - A **finished** run loads its final checkpoint and exits almost immediately
     (its generation loop has nothing left to do) — effectively skipped.
   - An **interrupted** run continues from its last saved generation, bit-identically
     to an uninterrupted run.
   - A **fresh** cell (no checkpoint) starts from generation 0.

So you do not track which experiments completed by hand: rerun the same launcher
command with `--resume` and the batch converges to completion. You can rerun it
as many times as needed.

You can also make resume the matrix default with `resume: true` at the top level;
the `--resume` flag overrides it. Without either, an existing checkpoint is
**ignored** and the run restarts from generation 0 — the safe default for a fresh
launch (so you never silently continue an old run by accident).

### Configuration guard

Resume refuses to continue if the run configuration differs from the checkpoint
(different objectives, population size, number of generations, precision, seed,
etc.) and aborts naming the differing field. Relaunch with the **same matrix** to
avoid this — that is exactly why reusing the matrix file is the intended workflow.

---

## 6. What each experiment directory contains

After (or during) a run, `<exp_root>/<algo>/<name>_repeat_<i>/` holds:

| File | Meaning |
|---|---|
| `launch_command.txt` | The exact command + `CUDA_VISIBLE_DEVICES` used (reproducibility). |
| `launcher.log` | The run's stdout/stderr (per-candidate metrics, generation summaries). |
| `checkpoint.pkl` | The resumable search state at the last generation boundary. |
| `pareto_history.pkl` | Per-generation Pareto front + hypervolume (multi-objective runs). |
| `eval_cache.pkl` | The evaluation cache (only when `--use_cache`/`use_cache: true`). |
| `log_params_evolution.txt` | The fully resolved parameters of the run. |

---

## 7. Single run without the launcher

For a one-off run, call the entry point directly. The launcher is just a wrapper
around this:

```bash
python run_all_evolution.py --algo nsga2 \
    --config_file experiment_configs/cifar_mo/config0_3_acc_flops.yaml \
    --experiment_path experiment_cifar10_acc_flops/nsga2/nsga2_repeat_1 \
    --data_path datasets/cifar10_data --dataset cifar10 \
    --config_path_dataset dataset_configs/cifar10.yaml \
    --multi_objective --population_size 20 --num_generations 150 \
    --seed 42 --log_level INFO

# Two GPUs for this single run (candidates balanced across both):
CUDA_VISIBLE_DEVICES=0,1 python run_all_evolution.py --algo nsga2 ... 

# Resume this run after an interruption:
python run_all_evolution.py --algo nsga2 ... --resume
```

`--use_cache` enables the evaluation cache (reuses metrics of architectures already
seen). The cache key includes the objectives, precision and `eval_window_agg`, so a
cached `max`/`fp16` result is never reused for a `mean`/`bf16` run.

---

## 8. Quick reference

```bash
# Preview the expanded commands
python launch.py experiment_matrices/M.yaml --dry-run

# Run a batch
python launch.py experiment_matrices/M.yaml

# Resume an interrupted batch (rerun as needed; finished cells are skipped)
python launch.py experiment_matrices/M.yaml --resume
```

| Matrix key | Purpose |
|---|---|
| `exp_root` | Root output directory. |
| `gpus` | GPU pool, e.g. `[0, 1]`. |
| `gpus_per_run` | GPUs per run (default 1). Pool is split into slots of this size. |
| `repeats` / `seed_base` | Repeats per experiment; repeat *i* uses seed `seed_base + i`. |
| `resume` | `true` to resume by default (overridden by `--resume`). |
| `defaults` | Args shared by every cell. |
| `experiments[].overrides` | Per-cell args (override `defaults`). |
| `experiments[].flags` | Literal flags appended verbatim (e.g. `--multi_objective`). |
