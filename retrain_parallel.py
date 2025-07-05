import argparse
import os
import pickle
import yaml
import multiprocessing as mp
import torch

from cnn import input, master
from util import load_log_params_evolution, init_log, save_results_file


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

                # 1. Get a set of all directories that actually exist in the archive.
                if os.path.isdir(archive_dir):
                    existing_dirs = set(os.listdir(archive_dir))
                else:
                    existing_dirs = set()

                # 2. Load all IDs from the history file's front.
                all_front_ids = [rec.get("id") for rec in front if rec.get("id")]
                
                # 3. Keep only the IDs that have an existing directory.
                ids = [cid for cid in all_front_ids if cid in existing_dirs]

    except (pickle.UnpicklingError, EOFError) as e:
        print(f"Warning: Could not load {history_file}. It may be corrupted. Error: {e}")

    # Fallback logic remains the same
    if not ids and os.path.isdir(archive_dir):
        print("Warning: No valid IDs found in pareto_history.pkl. "
            "Falling back to alphabetical list of models in archive directory.")
        ids = sorted(d for d in os.listdir(archive_dir)
                    if os.path.isdir(os.path.join(archive_dir, d)))

    if top_n:
        ids = ids[:top_n]

    return ids


def load_candidate_params(archive_dir: str, cid: str):
    params_file = os.path.join(archive_dir, cid, "training_params.txt")
    with open(params_file, "r") as f:
        data = yaml.safe_load(f)
    net_list = data.get("net_list", [])
    backbone = data.get("backbone_name")
    backbone_pct = data.get("backbone_percentage", 0.0)
    return net_list, backbone, backbone_pct


def worker(cid: str, base_spec: dict, fn_dict: dict, args, device: str, queue: mp.Queue):
    logger = init_log(args.log_level, name=f"worker-{cid}")
    archive_dir = os.path.join(args.experiment_path, "archive")
    net_list, backbone, backbone_pct = load_candidate_params(archive_dir, cid)

    params = dict(base_spec)
    params.update({
        "data_path": args.data_path,
        "dataset": args.dataset,
        "device": device,
    })
    if backbone:
        params["backbone_name"] = backbone
    if backbone_pct:
        params["backbone_percentage"] = backbone_pct

    override_keys = [
        "max_epochs", "epochs_to_eval", "batch_size", "eval_batch_size",
        "limit_data", "lr_scheduler", "optimizer", "data_augmentation",
        "num_workers", "save_checkpoints_epochs", "patience_retrain",
        "delta_fraction",
    ]
    for k in override_keys:
        v = getattr(args, k, None)
        if v is not None:
            params[k] = v

    results = {}
    for rep in range(args.num_repetitions):
        params["experiment_path"] = os.path.join(
            archive_dir, cid, f"retrain_parallel_{rep+1}")
        logger.info(f"Retraining {cid} repetition {rep+1} on {device}")
        loader = input.GenericDataLoader(params=params)
        train_loader, val_loader = loader.get_loader(pin_memory_device=device)
        test_loader = loader.get_loader(for_train=False, pin_memory_device=device)
        res = master.retrain(params=params, fn_dict=fn_dict, net_list=net_list,
                            train_loader=train_loader, val_loader=val_loader,
                            test_loader=test_loader)
        results[f"retrain_{rep+1}"] = res
    queue.put((cid, results))


def main(arguments):
    logger = init_log(arguments.log_level, name=__name__)
    config = load_log_params_evolution(arguments.experiment_path)
    train_spec = config['train_spec']
    fn_dict = config['fn_dict']

    if arguments.ids:
        candidate_ids = arguments.ids
    else:
        candidate_ids = parse_pareto_ids(
            arguments.experiment_path,
            arguments.top_n,
            arguments.sort_by  # Pass the new argument here
        )
    logger.info(f"Candidates to retrain: {candidate_ids}")
    if not candidate_ids:
        logger.error("No candidate IDs found to retrain.")
        return

    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if not devices:
        devices = ["cpu"]
    queue = mp.Queue()
    results = {}

    mp.set_start_method('spawn', force=True)

    for chunk_start in range(0, len(candidate_ids), len(devices)):
        procs = []
        chunk = candidate_ids[chunk_start:chunk_start + len(devices)]
        for idx, cid in enumerate(chunk):
            dev = devices[idx % len(devices)]
            p = mp.Process(target=worker,
                            args=(cid, train_spec, fn_dict, arguments, dev, queue))
            procs.append(p)
            p.start()
        for _ in procs:
            cid, res = queue.get()
            results[cid] = res
        for p in procs:
            p.join()

    save_results_file(arguments.experiment_path, results,
                        file_name='retrain_results_parallel.txt')
    logger.info("Retraining finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Path to the evolution experiment.')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to input data.')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'pathmnist', 'octmnist',
                                'tissuemnist', 'organamnist', 'organcmnist',
                                'atleta_axial', 'atleta_coronal'])
    parser.add_argument('--ids', nargs='+', default=None,
                        help='Specific candidate IDs to retrain.')
    parser.add_argument('--sort_by', type=str, default=None,
                        choices=['accuracy', 'params', 'inference_time'],
                        help='Metric to sort the Pareto front by before retraining.')
    parser.add_argument('--top_n', type=int, default=None,
                        help='Number of top models from final Pareto front.')
    parser.add_argument('--log_level', choices=['NONE', 'INFO', 'DEBUG'],
                        default='NONE')
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--epochs_to_eval', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=1000)
    parser.add_argument('--limit_data', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_repetitions', type=int, default=1)
    parser.add_argument('--lr_scheduler', type=str, default="multistep",
                        choices=['cosine', 'reduce_on_plateau', 'exponential',
                                'multistep', 'None'])
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['RMSProp', 'Adam', 'AdamW', 'SGD'])
    parser.add_argument('--data_augmentation', action='store_true')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_checkpoints_epochs', type=int, default=5)
    parser.add_argument('--patience_retrain', type=int, default=25)
    parser.add_argument('--delta_fraction', type=float, default=0.005)
    arguments = parser.parse_args()
    main(arguments)