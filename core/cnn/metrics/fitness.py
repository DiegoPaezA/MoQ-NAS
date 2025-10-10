import torch
from typing import Dict
from .base import BaseMetric

class ScalarizedFitness(BaseMetric):
    """
    Computes a single, scalarized fitness value from multiple metric results.

    This metric is a post-processor that takes the final results from other
    metrics (like accuracy, loss, and hardware stats) and combines them into a
    weighted fitness score, ideal for single-objective NAS algorithms.
    It does not accumulate data per batch; its logic is executed once at the
    end of an evaluation epoch.
    """
    name = "scalarized_fitness"

    def __init__(self, metric_type: str, max_params: float, max_inference_time: float, **kwargs):
        """
        Initializes the scalarized fitness calculator.

        Args:
            metric_type (str): The primary metric to use ('accuracy' or 'loss').
            max_params (float): The threshold for the maximum number of parameters.
            max_inference_time (float): The threshold for the maximum inference time.
        """
        super().__init__(metric_type=metric_type, max_params=max_params, max_inference_time=max_inference_time, **kwargs)
        self.metric_type = metric_type
        self.T_p = max_params
        self.T_t = max_inference_time

    def reset(self):
        """This metric is stateless across batches, so reset does nothing."""
        pass

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """This metric is stateless across batches, so update does nothing."""
        pass

    def compute(self, epoch_results: Dict) -> Dict:
        """
        Calculates the fitness value using results from other metrics.

        Args:
            epoch_results (Dict): A dictionary containing the computed values
                                from all other metrics for the current epoch.

        Returns:
            Dict: A dictionary containing the final 'scalar_multi_objective' score.
        """
        # 1. Extract required values from the results dictionary
        if self.metric_type == 'accuracy':
            metric_value = epoch_results.get('accuracy', 0.0)
        elif self.metric_type == 'loss':
            metric_value = epoch_results.get('loss', float('inf'))
        else:
            raise ValueError(f"Invalid metric_type: {self.metric_type}")

        params = epoch_results.get('total_params', 0)
        inference_time = epoch_results.get('cuda_inference_time', 0)
        
        # 2. Encapsulate the original mofitness logic
        return {
            "scalar_multi_objective": self._mofitness(
                metric_value, params, inference_time
            )
        }

    def _mofitness(self, metric_value, params, inference_time) -> float:
        """
        Encapsulates the original weighted fitness function logic.
        """
        # Handle the primary metric
        if self.metric_type == 'accuracy':
            primary_fitness = metric_value / 100.0 if metric_value > 1 else metric_value
        else: # 'loss'
            primary_fitness = 1 / (1 + metric_value)

        # Assign weights based on thresholds
        w_p = -0.01 if params <= self.T_p else -1
        w_t = -0.01 if inference_time <= self.T_t else -1

        params_ratio = params / self.T_p if self.T_p > 0 else float('inf')
        inference_time_ratio = inference_time / self.T_t if self.T_t > 0 else float('inf')

        params_factor = params_ratio ** w_p if params_ratio > 0 else 0
        inference_time_factor = inference_time_ratio ** w_t if inference_time_ratio > 0 else 0
        
        fitness_value = primary_fitness * params_factor * inference_time_factor
        return fitness_value * 100.0
    
class ValidationLossFitness(BaseMetric):
    """
    Computes a fitness value directly from the validation loss.

    This metric transforms the validation loss into a fitness score,
    where a lower loss results in a higher fitness value.
    The result is scaled to a range of 0-100.
    """
    name = "fitness_val_loss"

    def __init__(self, **kwargs):
        """Initializes the validation loss fitness calculator."""
        super().__init__(**kwargs)

    def reset(self):
        """This metric is stateless across batches, so reset does nothing."""
        pass

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """This metric is stateless across batches, so update does nothing."""
        pass

    def compute(self, epoch_results: Dict) -> Dict:
        """
        Calculates the fitness value from the validation loss.

        Args:
            epoch_results (Dict): A dictionary containing the computed values
                                from all other metrics for the current epoch.

        Returns:
            Dict: A dictionary containing the 'fitness_val_loss' score.
        """
        validation_loss = epoch_results.get('loss', float('inf'))

        if validation_loss == float('inf'):
            fitness_value = 0.0
        else:
            # Transform loss to a fitness value (lower loss = higher fitness)
            fitness_value = (1 / (1 + validation_loss)) * 100.0

        return {
            self.name: fitness_value
        }