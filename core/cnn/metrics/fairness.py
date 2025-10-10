# moq-nas/core/cnn/metrics/fairness.py
import torch
import numpy as np
from typing import Dict
from collections import defaultdict

from .base import BaseMetric
# We import the dataloader factory from our new fairness module
from core.fairness.data import create_eval_loader

class FairnessMetric(BaseMetric):
    """
    Computes fairness metrics by performing a separate evaluation on a specified
    fairness benchmark dataset (e.g., FACET or FairFace).

    This metric is self-contained and is triggered during the `compute` step
    of the NAS evaluation phase. It is designed to be driven by the `FAIRNESS`
    section in the .yaml configuration file. It returns a single score for
    optimization but logs a detailed dictionary with all results for later analysis.
    """
    # The 'name' must match the objective name in the .yaml config
    # for the MetricFactory to recognize it.
    name = "FairnessMetric" 

    def __init__(self, **kwargs):
        """
        Initializes the FairnessMetric with individual parameters.
        
        Args:
            model (torch.nn.Module): The model instance to evaluate.
            device (str): The device for computation ('cuda' or 'cpu').
            eval_dataset_name (str): The name of the fairness dataset (e.g., 'facet').
            eval_dataset_path (str): The path to the fairness dataset's CSV file.
            optimization_objective (str): The specific metric to use for optimization (e.g., 'spd_sum').
            beta (float): The beta parameter for the fairness score calculation.
            cache_dir (str, optional): Directory to cache preprocessed images.
            batch_size (int, optional): The batch size for the evaluation loader.
            positive_class_idx (int, optional): The index of the positive class in model outputs.
            eval_skintone_method (str, optional): 'soft' or 'hard' method for FACET evaluation.
        """
        super().__init__()
        
        # --- Save the "blueprint" for cloning ---
        self._init_args = kwargs
        # ------------------------------------
        
        # Extract required parameters from the stored blueprint
        self.model = self._init_args.get('model')
        self.device = self._init_args.get('device')
        self.eval_dataset_name = self._init_args.get('eval_dataset_name')
        self.eval_dataset_path = self._init_args.get('eval_dataset_path')
        self.optimization_objective = self._init_args.get('optimization_objective', '').lower()
        self.beta = self._init_args.get('beta')
        self.cache_dir = self._init_args.get('cache_dir')
        self.batch_size = self._init_args.get('batch_size', 256)
        self.positive_class_idx = self._init_args.get('positive_class_idx', 1)
        self.eval_skintone_method = self._init_args.get('eval_skintone_method', 'soft').lower()
        
        # A check to ensure critical parameters were passed during initialization
        if not all([self.model, self.device, self.eval_dataset_name, self.beta]):
            raise ValueError(f"FairnessMetric is missing required arguments. Provided: {list(self._init_args.keys())}")

        self._results = {}

    def reset(self):
        """Resets cached results."""
        self._results = {}

    def update(self, outputs: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor = None):
        """This metric performs its own evaluation in compute(), so this is not used."""
        pass

    def compute(self, epoch_results=None) -> Dict[str, float]:
        """
        Runs the full fairness evaluation and returns a dictionary of metrics.
        The main optimization objective is one of the keys in this dictionary.
        """
        if self._results:
            return self._results

        # Create the evaluation dataloader using our factory
        dataloader = create_eval_loader(
            dataset_name=self.eval_dataset_name,
            csv_path=self.eval_dataset_path,
            batch_size=self.batch_size,
            cache_dir=self.cache_dir
        )
        
        if self.eval_dataset_name.lower() == 'facet':
            # Calculate both soft and hard metrics
            if self.eval_skintone_method == 'soft':
                tpr_ = self._compute_tpr_per_skintone_soft(dataloader)
            elif self.eval_skintone_method == 'hard':
                tpr_ = self._compute_tpr_per_skintone_hard(dataloader)
            else:
                raise ValueError(f"Unknown eval_skintone_method: {self.eval_skintone_method}. Choose 'soft' or 'hard'.")


            summary_metrics = self._compute_summary_metrics(tpr_)
            self._results["per_group_tpr"] = tpr_
            self._results["metrics"] = summary_metrics
            self._results["fairness_score"] = summary_metrics.get(self.optimization_objective, 0.0)

        elif self.eval_dataset_name.lower() == 'fairface':
            per_group_tpr = self._compute_tpr_per_group(dataloader)
            summary_metrics = self._compute_summary_metrics(per_group_tpr)
            self._results["per_group_tpr"] = per_group_tpr
            self._results["metrics"] = summary_metrics
            self._results["fairness_score"] = summary_metrics.get(self.optimization_objective, 0.0)

        return self._results

    def _compute_tpr_per_skintone_hard(self, loader) -> Dict[str, float]:
            """Computes TPR for FACET by deriving 'hard' labels via argmax."""
            group_correct = defaultdict(int)
            group_total = defaultdict(int)

            with torch.no_grad():
                for inputs, soft_labels in loader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    preds = outputs.argmax(dim=1).cpu()

                    # Derive hard labels from soft probabilities
                    hard_labels = soft_labels.argmax(dim=1)

                    for i in range(len(preds)):
                        # Skin tones are 1-10, so add 1 to the index
                        group_idx = hard_labels[i].item() + 1
                        group_total[group_idx] += 1
                        if preds[i] == self.positive_class_idx:
                            group_correct[group_idx] += 1
            
            per_tone_tpr = {
                str(tone): float(group_correct[tone] / group_total[tone]) if group_total[tone] > 0 else 0.0
                for tone in sorted(group_total.keys())
            }
            return per_tone_tpr

    def _compute_tpr_per_skintone_soft(self, loader) -> Dict[str, float]:
            """Computes TPR for FACET using 'soft' probability-weighted labels."""
            denominator = defaultdict(float)
            numerator = defaultdict(float)

            with torch.no_grad():
                for inputs, soft_labels in loader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    preds = outputs.argmax(dim=1).cpu()

                    for i in range(len(preds)):
                        pred = preds[i]
                        for tone_idx in range(soft_labels.shape[1]):
                            prob = soft_labels[i, tone_idx].item()
                            if prob > 0:
                                denominator[tone_idx + 1] += prob
                                if pred == self.positive_class_idx:
                                    numerator[tone_idx + 1] += prob
            
            per_tone_tpr = {
                str(tone): float(numerator[tone] / denominator[tone]) if denominator[tone] > 0 else 0.0
                for tone in sorted(denominator.keys())
            }
            return per_tone_tpr

    def _compute_tpr_per_group(self, loader) -> Dict[str, float]:
        """Computes TPR for each demographic group in FairFace."""
        group_tpr = defaultdict(float)
        group_counts = defaultdict(int)
        label_map = {v: k for k, v in loader.dataset.race_to_idx.items()}

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds = outputs.argmax(dim=1).cpu()
                
                for i in range(len(labels)):
                    label_idx = labels[i].item()
                    group_name = label_map[label_idx]
                    if preds[i] == self.positive_class_idx:
                        group_tpr[group_name] += 1
                    group_counts[group_name] += 1
            
        final_tpr = {}
        for group_name, total in group_counts.items():
            if total > 0:
                final_tpr[group_name] = float(group_tpr[group_name] / total)
        
        return dict(sorted(final_tpr.items()))
        
    def _compute_summary_metrics(self, per_group_tpr: Dict[str, float]) -> Dict[str, float]:
        """Calculates spd_sum, gap, and the final fairness_score from TPRs."""
        if not per_group_tpr:
            return {"min_group_tpr": 0.0, "max_min_gap": 0.0, "spd_sum": 0.0, "fairness_score": 0.0}

        tprs = np.array(list(per_group_tpr.values()))
        min_tpr = np.min(tprs)
        max_tpr = np.max(tprs)
        
        spd_sum = np.sum(tprs - min_tpr)
        fairness_score = max(0.0, (self.beta - spd_sum) / self.beta)
        max_min_gap = max_tpr - min_tpr

        return {
            "min_group_tpr": float(min_tpr),
            "max_min_gap": float(max_min_gap),
            "spd_sum": float(spd_sum),
            "fairness_raw": float(fairness_score),
        }