import torch
from .base import BaseMetric
from .base_hardware import ModelMetrics 

class HardwareMetrics(BaseMetric):
    """
    Computes hardware and model complexity metrics once per evaluation.
    This includes inference time, parameter count, FLOPs, and memory usage.
    """
    name = "hardware_metrics"

    def __init__(self, model, device, input_shape):
        """
        Initializes the hardware metric calculator.
        Args:
            model (torch.nn.Module): The model to be evaluated.
            device (str): The device ('cuda' or 'cpu') to run on.
            input_shape (tuple): The shape of a single input sample.
        """
        self._init_args = locals()
        del self._init_args['self']
        self.model_metrics = ModelMetrics(model, device)
        self.input_shape = input_shape
        self._results = {}

    def reset(self):
        """Resets the cached results."""
        self._results = {}

    def update(self, outputs: torch.Tensor, labels: torch.Tensor):
        """This metric is computed once, so the update step does nothing."""
        pass

    def compute(self) -> dict:
        """Computes all hardware metrics if not already cached for this epoch."""
        if self._results:
            return self._results
            
        dummy_batch = torch.randn((10, *self.input_shape))
        
        self._results = {
            "cuda_inference_time": self.model_metrics.measure_inference_time(dummy_batch),
            "total_params": self.model_metrics.measure_parameters(),
            "total_flops": self.model_metrics.measure_flops(self.input_shape),
            "model_memory_usage": self.model_metrics.measure_memory(self.input_shape) / (1024**2)
        }
        return self._results