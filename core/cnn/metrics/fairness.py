import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.amp import autocast
from typing import Dict, List, Optional, Tuple
from .base import BaseMetric

class FairFaceMetrics(BaseMetric):
    """
    Computes fairness metrics by evaluating the model on the FairFace dataset.

    This metric calculates the True Positive Rate (TPR) for different racial
    groups and then computes a final fairness score based on the gap between
    the highest and lowest TPR. It performs its own evaluation loop on the
    FairFace dataset, triggered during the `compute` step.
    """
    name = "fairface_metrics"

    def __init__(self, model: torch.nn.Module, fairface_csv: str, transforms,
                 device: str, batch_size: int = 128, beta: float = 0.2):
        """
        Initializes the FairFace metric evaluator.

        Args:
            model (torch.nn.Module): The model to be evaluated.
            fairface_csv (str): Path to the FairFace dataset CSV file.
            transforms: The torchvision transforms to apply to the images.
            device (str): The device ('cuda' or 'cpu') for evaluation.
            batch_size (int): The batch size for the internal evaluation loop.
            beta (float): A tolerance parameter for the final fairness score.
        """
        self.model = model
        self.fairface_df = pd.read_csv(fairface_csv)
        self.transforms = transforms
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.beta = beta
        self._results = {}

    def reset(self):
        """Resets the cached results for the epoch."""
        self._results = {}

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """This metric is computed once, so the standard update step is not used."""
        pass

    def compute(self) -> dict:
        """
        Runs the full evaluation on the FairFace dataset and computes fairness.
        The results are cached to avoid re-computation within the same epoch.
        """
        if self._results:
            return self._results

        # 1. Evaluate on FairFace to get per-group TPRs
        per_group_tpr, _, _ = self._evaluate_on_fairface()

        # 2. Calculate final fairness score from the TPRs
        vals = list(per_group_tpr.values())
        tpr_min, tpr_max = (min(vals), max(vals)) if vals else (0.0, 0.0)
        
        # Sum of Per-group-Differences from the minimum TPR
        spd_sum = sum(v - tpr_min for v in vals)
        
        # The fairness score is scaled by the beta tolerance
        fairness_score = max(0.0, (self.beta - spd_sum) / self.beta)

        self._results = {
            "fairface_per_group_tpr": per_group_tpr,
            "fairface_min_tpr": float(tpr_min),
            "fairface_tpr_gap": float(tpr_max - tpr_min),
            "fairface_spd_sum": float(spd_sum),
            "fairface_fairness_score": float(fairness_score)
        }
        return self._results

    def _evaluate_on_fairface(self) -> Tuple[Dict[str, float], Dict[str, int], int]:
        """Internal method to run inference on the FairFace dataset."""
        self.model.eval()
        groups = sorted(self.fairface_df["race"].unique().tolist())
        correct = {g: 0 for g in groups}
        total = {g: 0 for g in groups}
        amp_device = "cuda" if self.device.type == "cuda" else "cpu"
        
        xs, gs = [], []
        with torch.no_grad():
            for _, row in self.fairface_df.iterrows():
                try:
                    img = Image.open(row["image_path"]).convert("RGB")
                except Exception:
                    continue
                xs.append(self.transforms(img))
                gs.append(row["race"])
                
                if len(xs) >= self.batch_size:
                    self._process_batch(xs, gs, correct, total, amp_device)
                    xs, gs = [], []

            if xs: # Process the last batch
                self._process_batch(xs, gs, correct, total, amp_device)

        per_group_tpr = {g: (correct[g] / total[g] if total[g] > 0 else 0.0) for g in groups}
        n_total = int(sum(total.values()))
        return per_group_tpr, total, n_total

    def _process_batch(self, xs, gs, correct, total, amp_device):
        """Helper to process a single batch of images."""
        x = torch.stack(xs).to(self.device)
        with autocast(device_type=amp_device, enabled=(amp_device == "cuda")):
            logits = self.model(x)
        preds = logits.argmax(1).cpu().tolist()
        for p, g in zip(preds, gs):
            # Ground truth is "face" (class 1) for all FairFace rows
            correct[g] += int(p == 1)
            total[g] += 1

class FacetMetrics(BaseMetric):
    """
    Computes fairness metrics based on skin tone groups from the Facet dataset.

    This metric accumulates model predictions and soft probabilities over an
    epoch. The `compute` step then calculates detailed fairness statistics
    using both 'hard' (argmax) and 'soft' (weighted average) grouping methods,
    preserving the full nested dictionary output.
    """
    name = "facet_metrics"

    def __init__(self, mode: str = "both", beta: float = 0.2):
        self.mode = mode
        self.beta = beta
        self.all_soft_probs = []
        self.all_labels = []

    def reset(self):
        self.all_soft_probs = []
        self.all_labels = []

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        # 'labels' are the 0/1 predictions, 'outputs' are the 10-class skin tone logits.
        self.all_labels.append(labels.cpu())
        self.all_soft_probs.append(outputs.softmax(dim=-1).cpu())

    def compute(self) -> dict:
        """
        Computes the final fairness metrics, returning the full detailed dictionary.
        """
        if not self.all_labels:
            return {"hard": {}, "soft": {}}

        preds = torch.cat(self.all_labels).numpy().astype(int)
        soft_probs = torch.cat(self.all_soft_probs).numpy()
        
        # --- Direct call to your original logic, now as a private method ---
        return self._fairness_from_preds(preds, soft_probs)

    def _fairness_from_preds(self, preds: np.ndarray, P: np.ndarray) -> Dict:
        """
        Encapsulates the original `fairness_from_preds` logic.
        """
        tone_ix = np.arange(P.shape[1])
        out = {}

        # HARD: argmax tone
        if self.mode in ("hard", "both"):
            hard = P.argmax(axis=1)
            per = {}
            counts = {}
            for g in tone_ix:
                mask = (hard == g)
                n = int(mask.sum())
                counts[str(g + 1)] = n
                per[str(g + 1)] = float(preds[mask].mean()) if n > 0 else 0.0
            
            vals = list(per.values())
            acc_min, acc_max = (min(vals), max(vals)) if vals else (0.0, 0.0)
            spd_sum = sum(v - acc_min for v in vals)
            fair = max(0.0, (self.beta - spd_sum) / self.beta)
            
            out["hard"] = {
                "per_tone_tpr": per,
                "counts": counts,
                "overall_mean_tpr": float(sum(vals) / max(1, len(vals))),
                "metrics": {
                    "min_group_tpr": float(acc_min),
                    "max_min_gap": float(acc_max - acc_min),
                    "spd_sum": float(spd_sum),
                    "fairness": float(fair)
                }
            }

        # SOFT: expected TPR by tone
        if self.mode in ("soft", "both"):
            per = {}
            denom = {}
            for g in tone_ix:
                pg = P[:, g]
                denom_g = float(pg.sum())
                nume_g = float((preds * pg).sum())
                denom[str(g + 1)] = denom_g
                per[str(g + 1)] = (nume_g / denom_g) if denom_g > 0 else 0.0
            
            vals = list(per.values())
            acc_min, acc_max = (min(vals), max(vals)) if vals else (0.0, 0.0)
            spd_sum = sum(v - acc_min for v in vals)
            fair = max(0.0, (self.beta - spd_sum) / self.beta)

            out["soft"] = {
                "per_tone_tpr": per,
                "denom": denom,
                "overall_mean_tpr": float(sum(vals) / max(1, len(vals))),
                "metrics": {
                    "min_group_tpr": float(acc_min),
                    "max_min_gap": float(acc_max - acc_min),
                    "spd_sum": float(spd_sum),
                    "fairness": float(fair)
                }
            }
            
        return out