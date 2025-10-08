# moq-nas/core/cnn/metrics/fairness.py

from typing import Dict
from unittest import loader
import numpy as np
import torch
from tqdm import tqdm
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
    name = "fairness_score" 

    def __init__(self, cfg, **kwargs):
        """
        Initializes the FairnessMetric based on the global configuration.
        
        Args:
            cfg: The global configuration object (yacs).
        """
        super().__init__()
        # Read parameters from the FAIRNESS section of the .yaml file
        self.eval_dataset_name = cfg.FAIRNESS.EVAL_DATASET
        self.eval_dataset_path = cfg.FAIRNESS.EVAL_DATASET_PATH
        self.beta = cfg.FAIRNESS.BETA
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.cache_dir = getattr(cfg.FAIRNESS, 'CACHE_DIR', None)
        # The primary objective this metric will return for optimization
        self.optimization_objective = cfg.FAIRNESS.OBJECTIVE.lower() # e.g., 'spd_sum' or 'fairness_score'
        self._results = {}

    def reset(self):
        """Resets cached results before a new evaluation."""
        self._results = {}

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """This metric performs its own evaluation, so the standard update step is not used."""
        pass

    def compute(self, model: torch.nn.Module, **kwargs) -> float:
        """
        Runs the full fairness evaluation and returns the main objective's value.

        Results are cached to avoid re-computation.
        
        Args:
            model (torch.nn.Module): The trained model to be evaluated.
        
        Returns:
            float: The value of the fairness metric to be optimized (e.g., spd_sum).
                For spd_sum, a lower value is better.
        """
        if self._results:
            return self._results['metrics'][self.optimization_objective]

        device = next(model.parameters()).device
        
        # 1. Automatically create the correct evaluation dataloader using our factory
        dataloader = create_eval_loader(
            dataset_name=self.eval_dataset_name,
            csv_path=self.eval_dataset_path,
            batch_size=self.batch_size,
            cache_dir=self.cache_dir
        )
        
        if self.eval_dataset_name.lower() == 'facet':
            # Calculate both soft and hard metrics
            soft_tpr = self._compute_tpr_per_skintone_soft(model, dataloader, device)
            hard_tpr = self._compute_tpr_per_skintone_hard(model, dataloader, device)
            
            soft_metrics = self._compute_summary_metrics(soft_tpr)
            hard_metrics = self._compute_summary_metrics(hard_tpr)
            results = {
                "soft_results": {"per_group_tpr": soft_tpr, "metrics": soft_metrics},
                "hard_results": {"per_group_tpr": hard_tpr, "metrics": hard_metrics},
                "fairness_core_soft": soft_metrics.get(self.optimization_objective, 0.0),
                "fairness_core_hard": hard_metrics.get(self.optimization_objective, 0.0)
            }

        elif self.eval_dataset_name.lower() == 'fairface':
            per_group_tpr = self._compute_tpr_per_group(model, dataloader, device)
            summary_metrics = self._compute_summary_metrics(per_group_tpr)
            results = {
                "per_group_tpr": per_group_tpr,
                "metrics": summary_metrics,
                "fairness_core": summary_metrics.get(self.optimization_objective, 0.0)
            }

        return results


    def _compute_tpr_per_group(self, model, loader, device) -> Dict[str, float]:
        """Computes True Positive Rate (TPR) for each demographic group in FairFace."""
        group_tpr = defaultdict(float)
        group_counts = defaultdict(int)
        label_map = {v: k for k, v in loader.dataset.race_to_idx.items()}

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc=f"Evaluating [{self.eval_dataset_name}]"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                
                for i in range(len(labels)):
                    label_idx = labels[i].item()
                    group_name = label_map[label_idx]
                    if preds[i] == 1:
                        group_tpr[group_name] += 1
                    group_counts[group_name] += 1
            
        final_tpr = {}
        for group_name, total in group_counts.items():
            if total > 0:
                final_tpr[group_name] = float(group_tpr[group_name] / total)
        
        return dict(sorted(final_tpr.items()))

    def _compute_tpr_per_skintone_hard(self, model, loader, device) -> Dict[str, float]:
            """Computes TPR for FACET using 'hard' labels (argmax of probabilities)."""
            group_correct = defaultdict(int)
            group_total = defaultdict(int)

            with torch.no_grad():
                for inputs, soft_labels in tqdm(loader, desc=f"Evaluating [FACET - Hard]"):
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    preds = outputs.argmax(dim=1).cpu()
                    
                    # Convert soft labels to hard labels
                    hard_labels = soft_labels.argmax(dim=1)

                    for i in range(len(preds)):
                        # Skin tones are 1-10, so we add 1 to the index
                        group_idx = hard_labels[i].item() + 1
                        group_total[group_idx] += 1
                        if preds[i] == 1: # Correctly predicted "face"
                            group_correct[group_idx] += 1
            
            per_tone_tpr = {
                str(tone): float(group_correct[tone] / group_total[tone]) if group_total[tone] > 0 else 0.0
                for tone in sorted(group_total.keys())
            }
            return per_tone_tpr

    def _compute_tpr_per_skintone_soft(self, model, loader, device) -> Dict[str, float]:
        """Computes TPR for FACET using 'soft' probability-weighted labels."""
        denominator = defaultdict(float)
        numerator = defaultdict(float)

        with torch.no_grad():
            for inputs, soft_labels in tqdm(loader, desc=f"Evaluating [FACET - Soft]"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1).cpu()

                for i in range(len(preds)):
                    pred = preds[i]
                    for tone_idx in range(soft_labels.shape[1]):
                        prob = soft_labels[i, tone_idx].item()
                        if prob > 0:
                            denominator[tone_idx + 1] += prob
                            if pred == 1:
                                numerator[tone_idx + 1] += prob
        
        per_tone_tpr = {
            str(tone): float(numerator[tone] / denominator[tone]) if denominator[tone] > 0 else 0.0
            for tone in sorted(denominator.keys())
        }
        return per_tone_tpr

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
            "fairness_score": float(fairness_score),
        }