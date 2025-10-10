# (markdown) 
# Patch: VRAM-safe, GPU-agnostic fairness evaluation (works on L40S, 3090, etc.)

# moq-nas/core/cnn/metrics/fairness.py
import torch
import numpy as np
from typing import Dict
from collections import defaultdict
from torch.amp import autocast

from .base import BaseMetric
from core.fairness.data import create_eval_loader

class FairnessMetric(BaseMetric):
    name = "FairnessMetric"

    @property
    def is_post_processing(self) -> bool:
        return True

    def __init__(self, **kwargs):
        super().__init__()
        self._init_args = kwargs

        self.model = self._init_args.get('model')
        device_str = self._init_args.get('device')
        self.device = torch.device(device_str) if device_str else torch.device('cpu')
        self.eval_dataset_name = self._init_args.get('eval_dataset_name')
        self.eval_dataset_path = self._init_args.get('eval_dataset_path')
        self.optimization_objective = self._init_args.get('optimization_objective', '').lower()
        self.beta = self._init_args.get('beta')
        self.cache_dir = self._init_args.get('cache_dir')
        self.batch_size = self._init_args.get('batch_size_fairness', 64)
        self.microbatch = int(self._init_args.get('batch_size_fairness_micro', 2))  # NEW: hard VRAM cap
        self.positive_class_idx = self._init_args.get('positive_class_idx', 1)
        self.eval_skintone_method = self._init_args.get('eval_skintone_method', 'soft').lower()
        self.phase = self._init_args.get('phase', 'evolution').lower()
        self.empty_cache_every = int(self._init_args.get('empty_cache_every', 8))  # NEW: periodic cache clear

        if not all([self.model, self.device, self.eval_dataset_name, self.beta]):
            raise ValueError(f"FairnessMetric is missing required arguments. Provided: {list(self._init_args.keys())}")

        # (Optional) helps on Ampere+ including 3090/L40S
        try:
            torch.set_float32_matmul_precision('medium')
        except Exception:
            pass

        self._results = {}

    # ---------- NEW: helpers for portable, safe inference ----------

    def _autocast_kwargs(self):
        """
        Prefer bf16 when supported (A100/L40S), otherwise fp16 (e.g., 3090).
        Works across most CUDA GPUs. Disabled on CPU.
        """
        if self.device.type == 'cuda':
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            return dict(device_type='cuda', dtype=(torch.bfloat16 if use_bf16 else torch.float16), enabled=True)
        return dict(device_type='cpu', enabled=False)

    @torch.no_grad()
    def _forward_microbatched(self, inputs):
        """
        Run forward in microbatches to bound peak VRAM.
        Returns logits on CPU to free GPU memory ASAP.
        """
        outs = []
        # Ensure inputs on device
        inputs = inputs.to(self.device, non_blocking=True)
        # Split current batch into tiny microbatches
        for mb in inputs.split(self.microbatch, dim=0):
            with autocast(**self._autocast_kwargs()):
                logits_mb = self.model(mb)
            outs.append(logits_mb.detach().cpu())  # move off GPU immediately
            del mb, logits_mb
        logits = torch.cat(outs, dim=0)
        del outs, inputs
        return logits

    # ---------------------------------------------------------------

    def reset(self):
        self._results = {}

    def update(self, outputs: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor = None):
        pass

    def compute(self, epoch_results=None) -> Dict[str, float]:
        if self._results:
            return self._results

        dataloader = create_eval_loader(
            dataset_name=self.eval_dataset_name,
            csv_path=self.eval_dataset_path,
            batch_size=self.batch_size,   # outer batch (throughput) — microbatched inside forward
            cache_dir=self.cache_dir,
            phase=self.phase
        )

        if self.eval_dataset_name.lower() == 'facet':
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
        del dataloader
        return self._results

    def _compute_tpr_per_skintone_hard(self, loader) -> Dict[str, float]:
        group_correct = defaultdict(int)
        group_total = defaultdict(int)

        step = 0
        with torch.inference_mode():
            for inputs, soft_labels in loader:
                logits = self._forward_microbatched(inputs)  # logits on CPU
                preds = logits.argmax(dim=1)

                hard_labels = soft_labels.argmax(dim=1)  # on CPU already

                for i in range(len(preds)):
                    group_idx = int(hard_labels[i].item()) + 1  # 1..10
                    group_total[group_idx] += 1
                    if int(preds[i].item()) == self.positive_class_idx:
                        group_correct[group_idx] += 1

                step += 1
                if self.device.type == 'cuda' and step % self.empty_cache_every == 0:
                    torch.cuda.empty_cache()

        per_tone_tpr = {
            str(tone): float(group_correct[tone] / group_total[tone]) if group_total[tone] > 0 else 0.0
            for tone in sorted(group_total.keys())
        }
        return per_tone_tpr

    def _compute_tpr_per_skintone_soft(self, loader) -> Dict[str, float]:
        denominator = defaultdict(float)
        numerator = defaultdict(float)

        step = 0
        with torch.inference_mode():
            for inputs, soft_labels in loader:
                logits = self._forward_microbatched(inputs)  # CPU
                preds = logits.argmax(dim=1)

                # soft_labels is CPU; iterate without moving to GPU
                B, T = soft_labels.shape
                for i in range(B):
                    pred_i = int(preds[i].item())
                    row = soft_labels[i]
                    # accumulate weighted contributions
                    for tone_idx in range(T):
                        prob = float(row[tone_idx].item())
                        if prob > 0.0:
                            key = tone_idx + 1
                            denominator[key] += prob
                            if pred_i == self.positive_class_idx:
                                numerator[key] += prob

                step += 1
                if self.device.type == 'cuda' and step % self.empty_cache_every == 0:
                    torch.cuda.empty_cache()

        per_tone_tpr = {
            str(tone): float(numerator[tone] / denominator[tone]) if denominator[tone] > 0 else 0.0
            for tone in sorted(denominator.keys())
        }
        return per_tone_tpr

    def _compute_tpr_per_group(self, loader) -> Dict[str, float]:
        group_tpr = defaultdict(float)
        group_counts = defaultdict(int)
        label_map = {v: k for k, v in loader.dataset.race_to_idx.items()}

        step = 0
        with torch.inference_mode():
            for inputs, labels in loader:
                logits = self._forward_microbatched(inputs)  # CPU
                preds = logits.argmax(dim=1)

                for i in range(len(labels)):
                    label_idx = int(labels[i].item())
                    group_name = label_map[label_idx]
                    if int(preds[i].item()) == self.positive_class_idx:
                        group_tpr[group_name] += 1.0
                    group_counts[group_name] += 1

                step += 1
                if self.device.type == 'cuda' and step % self.empty_cache_every == 0:
                    torch.cuda.empty_cache()

        final_tpr = {}
        for group_name, total in group_counts.items():
            if total > 0:
                final_tpr[group_name] = float(group_tpr[group_name] / total)
        return dict(sorted(final_tpr.items()))

    def _compute_summary_metrics(self, per_group_tpr: Dict[str, float]) -> Dict[str, float]:
        if not per_group_tpr:
            return {"min_group_tpr": 0.0, "max_min_gap": 0.0, "spd_sum": 0.0, "fairness_raw": 0.0}

        tprs = np.array(list(per_group_tpr.values()), dtype=np.float32)
        min_tpr = float(np.min(tprs))
        max_tpr = float(np.max(tprs))
        spd_sum = float(np.sum(tprs - min_tpr))
        fairness_score = max(0.0, (self.beta - spd_sum) / self.beta)
        max_min_gap = max_tpr - min_tpr

        return {
            "min_group_tpr": min_tpr,
            "max_min_gap": max_min_gap,
            "spd_sum": spd_sum,
            "fairness_raw": fairness_score,
        }
