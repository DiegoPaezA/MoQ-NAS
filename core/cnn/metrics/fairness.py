# moq-nas/core/cnn/metrics/fairness.py

from typing import Dict
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
        super().__init__(cfg, **kwargs)
        # Read parameters from the FAIRNESS section of the .yaml file
        self.eval_dataset_name = cfg.FAIRNESS.EVAL_DATASET
        self.eval_dataset_path = cfg.FAIRNESS.EVAL_DATASET_PATH
        self.beta = cfg.FAIRNESS.BETA
        self.batch_size = cfg.TRAIN.BATCH_SIZE
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
            batch_size=self.batch_size
        )
        
        # 2. Run the appropriate evaluation loop based on the dataset name
        if self.eval_dataset_name.lower() == 'facet':
            per_group_tpr = self._evaluate_on_facet(model, dataloader, device)
        # Add FairFace evaluation logic here if needed in the future
        elif self.eval_dataset_name.lower() == 'fairface':
            per_group_tpr = self._evaluate_on_fairface(model, dataloader, device)
        else:
            raise NotImplementedError(f"Fairness evaluation for '{self.eval_dataset_name}' is not implemented.")
            
        # 3. Calculate summary metrics from the per-group results
        metrics = self._compute_summary_metrics(per_group_tpr)

        self._results = {
            "per_group_tpr": per_group_tpr,
            "metrics": metrics
        }
        
        # Log the full dictionary of results for detailed analysis
        self.log(self._results)
        
        # Return the single objective value for the NAS to optimize.
        # Note: NSGA-II minimizes objectives by default. `spd_sum` is ideal
        # because a lower value means a fairer model.
        return self._results['metrics'][self.optimization_objective]

    def _evaluate_on_facet(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, float]:
        """Runs the soft-label evaluation loop for the FACET dataset."""
        model.eval()
        numerator = defaultdict(float)
        denominator = defaultdict(float)

        with torch.no_grad():
            for images, soft_labels in tqdm(dataloader, desc=f"Evaluating Fairness on {self.eval_dataset_name}"):
                images = images.to(device, non_blocking=True)
                soft_labels = soft_labels.cpu().numpy()
                
                logits = model(images)
                preds = logits.argmax(dim=1).cpu().numpy()

                for i in range(images.size(0)):
                    pred = preds[i] # 0 or 1
                    # Iterate through the 10 skin tones
                    for tone_idx in range(soft_labels.shape[1]):
                        prob = soft_labels[i, tone_idx]
                        if prob > 0:
                            denominator[tone_idx + 1] += prob
                            if pred == 1: # Positive prediction
                                numerator[tone_idx + 1] += prob
        
        per_tone_tpr = {
            str(tone): numerator[tone] / denominator[tone] if denominator[tone] > 0 else 0.0
            for tone in sorted(denominator.keys())
        }
        return per_tone_tpr

    def _compute_summary_metrics(self, per_group_tpr: Dict[str, float]) -> Dict[str, float]:
        """Calculates spd_sum, gap, and the final fairness_score from TPRs."""
        if not per_group_tpr:
            return {"min_group_tpr": 0.0, "max_min_gap": 0.0, "spd_sum": 0.0, "fairness_score": 0.0}

        tprs = np.array(list(per_group_tpr.values()))
        min_tpr = np.min(tprs)
        
        # Sum of Pairwise Differences (SPD) from the worst-performing group
        spd_sum = np.sum(tprs - min_tpr)
        
        # fairness_score (higher is better)
        fairness_score = max(0.0, (self.beta - spd_sum) / self.beta)
        
        return {
            "min_group_tpr": float(min_tpr),
            "max_min_gap": float(np.max(tprs) - min_tpr),
            "spd_sum": float(spd_sum),
            "fairness_score": float(fairness_score),
        }