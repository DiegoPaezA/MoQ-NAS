import argparse
import copy
import os
import pickle
import yaml
import multiprocessing as mp
import torch
import traceback
from concurrent.futures import ProcessPoolExecutor

from core.cnn import input, master
from utils.helpers import load_log_params_evolution, init_log, save_results_file


def parse_pareto_ids(exp_path: str, top_n: int | None = None, sort_by: str | None = None):
    """
    Load candidate IDs from the final Pareto front that exist on disk.
    Reads IDs from pareto_history.pkl, then filters them to ensure a
    corresponding directory exists in the 'archive' folder.
    """
    history_file = os.path.join(exp_path, "pareto_history.pkl")
    archive_dir = os.path.join(exp_path, "archive")
    ids = []

    try:
        if os.path.isfile(history_file):
            with open(history_file, "rb") as f:
                history = pickle.load(f)

            if history:
                last_gen = max(history.keys())
                front = history[last_gen].get(1, [])

                if sort_by:
                    is_reversed = sort_by not in ['params', 'inference_time']
                    front = sorted(front, key=lambda x: x.get(sort_by, 0), reverse=is_reversed)

                if os.path.isdir(archive_dir):
                    existing_dirs = set(os.listdir(archive_dir))
                else:
                    existing_dirs = set()

                all_front_ids = [rec.get("id") for rec in front if rec.get("id")]
                ids = [cid for cid in all_front_ids if cid in existing_dirs]

    except (pickle.UnpicklingError, EOFError) as e:
        print(f"Warning: Could not load {history_file}. It may be corrupted. Error: {e}")

    if not ids and os.path.isdir(archive_dir):
        print("Warning: No valid IDs found in pareto_history.pkl. "
                "Falling back to alphabetical list of models in archive directory.")
        ids = sorted(d for d in os.listdir(archive_dir)
                    if os.path.isdir(os.path.join(archive_dir, d)))

    if top_n:
        ids = ids[:top_n]

    return ids


def load_candidate_params(archive_dir: str, cid: str, logger):
    """
    Safely loads network parameters, handling all potential errors.
    Returns (net_list, backbone_name, backbone_percentage, evolution_metrics).
    evolution_metrics contains objective values measured during evolution
    (e.g. total_flops, best_accuracy, fairness_spd) so they can be attached
    to the retrain results without recomputing them.
    """
    # Objective and hardware metrics measured during evolution that are worth
    # keeping in the retrain summary. Config params (batch_size, seed, etc.) excluded.
    _METRIC_KEYS = {
        'best_accuracy', 'total_flops', 'total_params', 'cuda_inference_time',
        'model_memory_usage', 'training_time', 'best_validation_loss',
        'fairness_spd', 'fairness_mean_tpr', 'fairness_score',
    }
    _SKIP = {'net_list', 'fn_dict', 'backbone_name', 'backbone_percentage',
             'generation', 'individual'}
    params_file = os.path.join(archive_dir, cid, "training_params.txt")
    try:
        with open(params_file, "r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            logger.error(f"Content of {params_file} is not a valid dictionary for candidate {cid}. Skipping.")
            return None, None, None, {}
        net_list = data.get("net_list", [])
        backbone = data.get("backbone_name")
        backbone_pct = data.get("backbone_percentage", 0.0)
        evolution_metrics = {k: v for k, v in data.items()
                             if k in _METRIC_KEYS and v is not None}
        return net_list, backbone, backbone_pct, evolution_metrics
    except Exception as e:
        logger.error(f"Failed to load or parse {params_file} for candidate {cid}. Error: {e}. Skipping.")
        return None, None, None, {}


def _parse_clip(value):
    """CLI value for --grad_clip_norm: 'none'/'null'/'0' disables clipping, else a float."""
    if value is None:
        return None
    if str(value).strip().lower() in ('none', 'null', '0', '0.0', 'off'):
        return 'disabled'
    return float(value)


def _set_global_seed(seed: int):
    """Seed Python, NumPy and Torch (CPU + CUDA) for one retrain repetition."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _environment_info():
    """Software/hardware context saved with every retrain result."""
    import platform
    import subprocess
    info = {
        'hostname': platform.node(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
    }
    try:
        repo = os.path.dirname(os.path.abspath(__file__))
        info['git_commit'] = subprocess.check_output(
            ['git', '-C', repo, 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:
        info['git_commit'] = 'unknown'
    return info


def _classify_result(res):
    """Map a master.retrain() return value to an explicit status string."""
    import math
    if res is None:
        return 'FAILED_OOM'  # master.retrain returns None only on CUDA out-of-memory
    if not isinstance(res, dict):
        return 'FAILED_UNKNOWN'
    test_loss = res.get('test_loss')
    train_losses = res.get('training_losses') or []
    has_nan = any(isinstance(v, float) and math.isnan(v) for v in train_losses[-5:])
    if test_loss is not None and isinstance(test_loss, float) and math.isnan(test_loss):
        has_nan = True
    if has_nan:
        return 'FAILED_NAN'
    if res.get('test_accuracy') is None:
        return 'FAILED_NO_TEST'
    return 'OK'


def _apply_protocol_overrides(params, args):
    """Explicit protocol values (never inherited from the search log when given)."""
    override_keys = [
        "max_epochs", "epochs_to_eval", "batch_size", "eval_batch_size",
        "limit_data", "lr_scheduler", "optimizer", "data_augmentation",
        "num_workers", "save_checkpoints_epochs", "patience_retrain", "delta_fraction",
        "config_path_dataset", "precision", "learning_rate", "weight_decay",
        "train_split", "split_seed", "eval_window_agg", "limit_data_value",
    ]
    for k in override_keys:
        v = getattr(args, k, None)
        if v is not None:
            params[k] = v
    if args.precision is not None:
        # An explicit precision wins over the legacy boolean in resolve_precision;
        # drop the boolean anyway so saved params are unambiguous.
        params.pop('mixed_precision', None)
    clip = _parse_clip(args.grad_clip_norm)
    if clip == 'disabled':
        params['grad_clip_norm'] = None
    elif clip is not None:
        params['grad_clip_norm'] = clip


def _rep_dir_name(tag, seed, rep):
    if seed is not None:
        return f"retrain_{tag}_s{seed}"
    return f"retrain_parallel_{rep + 1}" if tag == 'parallel' else f"retrain_{tag}_{rep + 1}"


def _rep_key(seed, rep):
    return f"seed_{seed}" if seed is not None else f"retrain_{rep + 1}"


def worker(task_args):
    """
    Retrain one candidate for every requested seed/repetition and return
    (cid, results, failures). Failures are explicit statuses, never silent.
    """
    cid, base_spec, fn_dict, args, device, log_file = task_args
    logger = init_log(args.log_level, name=f"worker-{cid}", file_path=log_file)

    try:
        archive_dir = os.path.join(args.experiment_path, "archive")
        net_list, backbone, backbone_pct, evolution_metrics = load_candidate_params(archive_dir, cid, logger)

        if net_list is None:
            return cid, {"error": "Failed to load parameters"}, [
                dict(id=cid, seed='', status='FAILED_LOAD', message='training_params.txt missing or invalid')]

        params = copy.deepcopy(base_spec)
        params.update({
            "data_path": args.data_path,
            "dataset": args.dataset,
            "device": device,
            "phase": "retrain",
        })
        if backbone: params["backbone_name"] = backbone
        if backbone_pct: params["backbone_percentage"] = backbone_pct

        _apply_protocol_overrides(params, args)

        if args.network_config:
            params['network_config'] = args.network_config
        if not args.keep_metrics:
            params['metrics'] = [{'name': 'Accuracy'}]
            params.pop('artifacts', None)

        logger.info(f"Creating DataLoader for {cid} on {device}")
        loader = input.GenericDataLoader(params=params)

        seeds = list(args.seeds) if args.seeds else [None] * args.num_repetitions
        env = _environment_info()
        results, failures = {}, []
        for rep, seed in enumerate(seeds):
            if seed is not None:
                _set_global_seed(seed)
                params['loader_seed'] = seed  # batch order; split_seed stays fixed
            # Fresh loaders per repetition so each seed starts from its own RNG state.
            train_loader, val_loader = loader.get_loader(pin_memory_device=device)
            test_loader = loader.get_loader(for_train=False, pin_memory_device=device)
            if rep == 0:
                logger.info(f"Train loader: {len(train_loader.dataset)} samples, "
                            f"Validation loader: {len(val_loader.dataset)} samples, "
                            f"Test loader: {len(test_loader.dataset)} samples")

            params["experiment_path"] = os.path.join(archive_dir, cid, _rep_dir_name(args.tag, seed, rep))
            logger.info(f"Starting retraining for {cid} seed={seed} rep={rep + 1} on {device}")
            try:
                res = master.retrain(params=params, fn_dict=fn_dict, net_list=net_list,
                                     train_loader=train_loader, val_loader=val_loader,
                                     test_loader=test_loader)
                status = _classify_result(res)
                message = ''
            except Exception as exc:  # keep going with the remaining seeds
                res, status, message = None, 'FAILED_EXCEPTION', f"{exc.__class__.__name__}: {exc}"
                logger.error(f"{cid} seed={seed}: {traceback.format_exc()}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            entry = res if isinstance(res, dict) else {}
            entry.update({'status': status, 'seed': seed, 'protocol_tag': args.tag,
                          'evolution_metrics': evolution_metrics, 'env': env})
            if message:
                entry['error'] = message
            results[_rep_key(seed, rep)] = entry
            if status != 'OK':
                failures.append(dict(id=cid, seed=seed if seed is not None else '', status=status, message=message))

        return cid, results, failures

    except Exception:
        error_message = f"Worker {cid} crashed unexpectedly.\n{traceback.format_exc()}"
        return cid, {"error": error_message}, [
            dict(id=cid, seed='', status='FAILED_WORKER', message=error_message.splitlines()[-1])]


def _merge_results(path, new_results):
    """Merge per candidate and per seed, so later seeds never overwrite earlier ones."""
    import json
    merged = {}
    if os.path.isfile(path):
        try:
            with open(path) as f:
                merged = json.load(f)
        except Exception:
            backup = path + '.corrupt'
            os.replace(path, backup)
            merged = {}
    for cid, reps in new_results.items():
        merged.setdefault(cid, {}).update(reps)
    return merged


def _append_failures(path, rows):
    import csv
    if not rows:
        return
    write_header = not os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'seed', 'status', 'message'])
        if write_header:
            w.writeheader()
        w.writerows(rows)


def main(arguments):
    log_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, 'retrain_parallel.log')

    logger = init_log("INFO", name=__name__, file_path=log_file)

    try:
        config = load_log_params_evolution(arguments.experiment_path)
        train_spec = config['train_spec']
        fn_dict = config['fn_dict']

        candidate_ids = arguments.ids or parse_pareto_ids(
            arguments.experiment_path, arguments.top_n, arguments.sort_by
        )
        logger.info(f"Found {len(candidate_ids)} valid candidates to retrain: {candidate_ids}")
        if not candidate_ids:
            logger.error("No candidate IDs found to retrain.")
            return 1

        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if not devices: devices = ["cpu"]

        if arguments.max_parallel_workers:
            num_workers = min(arguments.max_parallel_workers, len(candidate_ids))
            logger.info(f"Using {num_workers} parallel workers as specified by --max_parallel_workers.")
        else:
            num_workers = len(devices)

        tasks = []
        for i, cid in enumerate(candidate_ids):
            device = devices[i % len(devices)]
            tasks.append((cid, train_spec, fn_dict, arguments, device, log_file))

        final_results, all_failures = {}, []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_results = executor.map(worker, tasks)
            logger.info(f"Starting retraining for {len(tasks)} models on {len(devices)} devices...")
            for cid, res, failures in future_results:
                all_failures.extend(failures)
                if isinstance(res, dict) and "error" in res and len(res) == 1:
                    logger.error(f"--- Worker Error for Candidate {cid} ---\n{res['error']}\n"
                                 f"-------------------------------------------")
                    continue
                final_results[cid] = res
                bad = [f"{k}:{v.get('status')}" for k, v in res.items() if v.get('status') != 'OK']
                if bad:
                    logger.error(f"Candidate {cid} finished with failures: {bad}")
                else:
                    logger.info(f"Successfully finished retraining for candidate {cid}.")

        results_name = ('retrain_results_parallel.txt' if arguments.tag == 'parallel'
                        else f'retrain_results_{arguments.tag}.txt')
        results_path = os.path.join(arguments.experiment_path, results_name)
        final_results = _merge_results(results_path, final_results)
        save_results_file(arguments.experiment_path, final_results, file_name=results_name)
        _append_failures(os.path.join(arguments.experiment_path, f'retrain_failures_{arguments.tag}.csv'),
                         all_failures)
        if all_failures:
            logger.error(f"{len(all_failures)} failed repetition(s); see retrain_failures_{arguments.tag}.csv")
            return 2
        return 0

    except Exception as e:
        logger.critical(f"A critical error occurred in the main process: {e}", exc_info=True)
        return 1
    finally:
        logger.info("Retraining script finished.")


if __name__ == '__main__':
    # It's important to set the start method for CUDA + multiprocessing
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--ids', nargs='+', default=None)
    parser.add_argument('--sort_by', type=str, default=None, choices=['accuracy', 'params', 'inference_time'])
    parser.add_argument('--max_parallel_workers', type=int, default=None,
                        help='Manually set the number of parallel workers. '
                            'Defaults to the number of available GPUs.')
    parser.add_argument('--top_n', type=int, default=None)
    parser.add_argument('--log_level', choices=['NONE', 'INFO', 'DEBUG'], default='INFO')
    parser.add_argument('--max_epochs', type=int, default=25)
    parser.add_argument('--epochs_to_eval', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=1000)
    parser.add_argument('--limit_data', action='store_true')
    parser.add_argument('--limit_data_value', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_repetitions', type=int, default=1,
                        help='Unseeded repetitions (legacy). Ignored when --seeds is given.')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='One seeded repetition per value (weight init + batch order). '
                             'The train/val split stays fixed by split_seed.')
    parser.add_argument('--tag', type=str, default='parallel',
                        help="Protocol tag: names the results file (retrain_results_<tag>.txt), "
                             "the failures file and the per-seed folders (retrain_<tag>_s<seed>).")
    parser.add_argument('--lr_scheduler', type=str, default="multistep")
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--precision', type=str, default=None, choices=['fp32', 'fp16', 'bf16'],
                        help='Explicit precision; overrides precision/mixed_precision from the search log.')
    parser.add_argument('--grad_clip_norm', type=str, default=None,
                        help="Gradient clipping max norm, or 'none' to disable. Default: trainer default (1.0).")
    parser.add_argument('--config_path_dataset', type=str, default=None,
                        help='Dataset YAML (e.g. dataset_configs/cifar10.yaml); older logs point to configs/.')
    parser.add_argument('--train_split', type=float, default=None)
    parser.add_argument('--split_seed', type=int, default=None)
    parser.add_argument('--eval_window_agg', type=str, default='max', choices=['max', 'mean', 'last'],
                        help='Aggregation of the reported validation accuracy over the evaluated epochs '
                             '(retrain evaluates every epoch). Default: max (best epoch).')
    parser.add_argument('--data_augmentation', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_checkpoints_epochs', type=int, default=5)
    parser.add_argument('--patience_retrain', type=int, default=25)
    parser.add_argument('--delta_fraction', type=float, default=0.005)
    parser.add_argument('--network_config', type=str, default=None,
                        choices=['default', 'dense', 'backbone'],
                        help='Override network_config from the evolution config.')
    parser.add_argument('--keep_metrics', action='store_true',
                        help='Keep the full metric suite from the evolution config '
                             '(e.g. HardwareMetrics, FairnessMetric). '
                             'Default: use Accuracy only during retrain.')

    arguments = parser.parse_args()
    raise SystemExit(main(arguments))
