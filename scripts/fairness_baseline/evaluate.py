# moq-nas/scripts/fairness_baseline/evaluate.py
import sys
import argparse
import torch
from pathlib import Path
import json
from types import SimpleNamespace # Use a standard library instead of yacs

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from core.cnn.metrics.fairness import FairnessMetric
from core.fairness.models import make_baseline_model

def print_detailed_results(arch, data):
    """
    Prints a formatted summary of model performance.
    This function is robust and can handle two types of dictionary structures:
    1. A nested structure with 'soft_results' and/or 'hard_results'.
    2. A single, flat structure with 'per_group_tpr' and 'metrics' at the top level.
    """
    print(f"\n{'='*15} Results for [{arch}] {'='*15}")

    # --- Helper function to print a specific set of results (no changes needed here) ---
    def display_metrics(result_type, results):
        print(f"\n--- {result_type} Results ---")
        if 'per_group_tpr' in results:
            for group, tpr in results['per_group_tpr'].items():
                print(f"  - Group: {group:<20} | TPR: {tpr:.4f}")
        
        if 'metrics' in results:
            metrics = results['metrics']
            print(f"\n  Summary Metrics:")
            min_tpr = metrics.get('min_group_tpr', 'N/A')
            max_gap = metrics.get('max_min_gap', 'N/A')
            spd_sum = metrics.get('spd_sum', 'N/A')
            fairness = metrics.get('fairness_score', 'N/A')
            
            print(f"  - Min Group TPR : {min_tpr:.4f}" if isinstance(min_tpr, float) else f"  - Min Group TPR : {min_tpr}")
            print(f"  - Max-Min Gap   : {max_gap:.4f}" if isinstance(max_gap, float) else f"  - Max-Min Gap   : {max_gap}")
            print(f"  - SPD Sum       : {spd_sum:.4f}" if isinstance(spd_sum, float) else f"  - SPD Sum       : {spd_sum}")
            print(f"  - Fairness Score: {fairness:.4f}" if isinstance(fairness, float) else f"  - Fairness Score: {fairness}")

    # --- Main logic to determine the dictionary structure ---

    # Case 1: Check for the nested structure ('soft'/'hard' results)
    if 'soft_results' in data or 'hard_results' in data:
        if 'soft_results' in data:
            display_metrics("Soft", data['soft_results'])
        if 'hard_results' in data:
            display_metrics("Hard", data['hard_results'])
    
    # Case 2: Check for the flat structure ('per_group_tpr' at the top level)
    elif 'per_group_tpr' in data:
        display_metrics("Overall", data) # The data itself is the results dictionary
    
    # Fallback if no recognizable structure is found
    else:
        print("  No recognizable result structure found.")
    
    print(f"{'='*50}\n")


def evaluate_one_model(arch: str, ckpt_path: str, device: torch.device, args: argparse.Namespace):
    """
    Evaluates a single model checkpoint using the centralized FairnessMetric class.
    """
    print(f"\n--- Evaluating [{arch}] on [{args.dataset_name}] ---")
    print(f"Loading checkpoint: {ckpt_path}")

    # --- Step 1: Build the Model and Load its Weights ---
    model = make_baseline_model(arch, num_classes=2)
    # Using weights_only=True is recommended for safety
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    # --- Step 2: Create a Self-Contained Config Object ---
    cfg = SimpleNamespace()
    cfg.FAIRNESS = SimpleNamespace()
    cfg.FAIRNESS.EVAL_DATASET = args.dataset_name
    cfg.FAIRNESS.EVAL_DATASET_PATH = args.csv_path
    cfg.FAIRNESS.BETA = args.beta
    cfg.FAIRNESS.OBJECTIVE = "fairness_score" # This is the expected value
    cfg.FAIRNESS.CACHE_DIR = args.cache_dir

    cfg.TRAIN = SimpleNamespace(BATCH_SIZE=args.batch_size)
    cfg.SYSTEM = SimpleNamespace(NUM_WORKERS=args.num_workers)
    cfg.CNN = SimpleNamespace(INPUT_SIZE=args.img_size)
    
    # --- Step 3: Instantiate and Run the Fairness Metric ---
    fairness_metric = FairnessMetric(cfg=cfg)
    results = fairness_metric.compute(model)
    
    # --- Step 4: Print the Key Results ---
    print(f"\n--- Results for [{arch}] on [{args.dataset_name}] ---")
    print_detailed_results(arch, results)
        
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline models for fairness using the core FairnessMetric.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # ... (the rest of the main function is unchanged) ...
    parser.add_argument('--ckpt_dir', type=str, required=True, help="Directory containing model checkpoints.")
    parser.add_argument('--dataset_name', type=str, required=True, choices=['fairface', 'facet'],
                        help="Name of the evaluation dataset.")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to the evaluation CSV file.")
    parser.add_argument('--filter', type=str, default=None,
                        help="Optional: Only evaluate checkpoints whose filenames contain this string (e.g., 'personbin').")
    parser.add_argument('--beta', type=float, default=0.2, help="Beta value for the fairness score calculation.")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None, help="e.g., 'cuda:0'. Default picks best available.")
    parser.add_argument('--cache_dir', type=str, default=None,
                        help="Optional directory to cache cropped images for FACET to speed up re-runs.")

    args = parser.parse_args()


    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_dir = Path(args.ckpt_dir)
    all_results = {}

    for ckpt_path in sorted(ckpt_dir.glob("*.pt")):
        if args.filter and args.filter not in ckpt_path.name:
            continue

        try:
            arch = ckpt_path.stem.split('_data_')[1]
            results = evaluate_one_model(arch, str(ckpt_path), device, args)
            all_results[str(ckpt_path.name)] = results
        except Exception as e:
            print(f"\n[ERROR] Could not evaluate {ckpt_path.name}. Reason: {e}\n")

    output_filename = f"fairness_results_{args.dataset_name}"
    if args.filter:
        output_filename += f"_{args.filter}"
    output_path = ckpt_dir / f"{output_filename}.json"
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n✅ Saved all results to {output_path}")

if __name__ == "__main__":
    main()