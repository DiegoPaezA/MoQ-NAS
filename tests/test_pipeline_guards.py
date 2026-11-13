"""Unit tests for the guards around device handling outside core/device.py.

Run from the repository root:  python -m unittest tests.test_pipeline_guards -v
"""
import multiprocessing as mp
import unittest
from unittest import mock

import torch

from core.cnn.metrics.fairness import FairnessMetric
from core.evaluation import EvalPopulation


def _metric(device, precision):
    return FairnessMetric(model=torch.nn.Identity(), device=device, eval_dataset_name='facet',
                          beta=1.0, precision=precision)


class FairnessAutocast(unittest.TestCase):
    def test_cpu_is_always_disabled(self):
        for precision in ('fp32', 'fp16', 'bf16'):
            self.assertEqual(_metric('cpu', precision)._autocast_kwargs(),
                             dict(device_type='cpu', enabled=False))

    def test_mps(self):
        self.assertEqual(_metric('mps', 'fp32')._autocast_kwargs(), dict(device_type='mps', enabled=False))
        with mock.patch.object(torch, '__version__', '2.14.1'):
            self.assertEqual(_metric('mps', 'bf16')._autocast_kwargs(),
                             dict(device_type='mps', dtype=torch.bfloat16, enabled=True))
            self.assertEqual(_metric('mps', 'fp16')._autocast_kwargs(),
                             dict(device_type='mps', dtype=torch.float16, enabled=True))
        with mock.patch.object(torch, '__version__', '2.5.1'):
            with self.assertRaises(RuntimeError):
                _metric('mps', 'bf16')._autocast_kwargs()

    def test_cuda(self):
        self.assertEqual(_metric('cuda:0', 'fp32')._autocast_kwargs(), dict(device_type='cuda', enabled=False))
        with mock.patch.object(torch.cuda, 'is_bf16_supported', return_value=True):
            self.assertEqual(_metric('cuda:0', 'bf16')._autocast_kwargs(),
                             dict(device_type='cuda', dtype=torch.bfloat16, enabled=True))
        self.assertEqual(_metric('cuda:0', 'fp16')._autocast_kwargs(),
                         dict(device_type='cuda', dtype=torch.float16, enabled=True))


class _Dead:
    exitcode = -9

    @staticmethod
    def is_alive():
        return False


class NextResult(unittest.TestCase):
    def test_returns_a_queued_result(self):
        q = mp.get_context('spawn').Queue()
        q.put((3, {'best_accuracy': 1.0}, None))
        self.assertEqual(EvalPopulation._next_result(q, [_Dead()])[0], 3)

    def test_raises_when_every_worker_died(self):
        q = mp.get_context('spawn').Queue()
        import queue as queue_mod
        with mock.patch.object(q, 'get', side_effect=queue_mod.Empty):
            with self.assertRaisesRegex(RuntimeError, 'All evaluation workers exited'):
                EvalPopulation._next_result(q, [_Dead(), _Dead()])


if __name__ == '__main__':
    unittest.main()
