"""Unit tests for core/device.py. Backends are simulated; no GPU is needed.

Run from the repository root:  python -m unittest tests.test_device -v
"""
import sys
import unittest
import warnings
from unittest import mock

import torch

from core import device as dev


def _cuda(available, count=0):
    return [mock.patch.object(torch.cuda, 'is_available', return_value=available),
            mock.patch.object(torch.cuda, 'device_count', return_value=count),
            mock.patch.object(torch.cuda, 'set_device')]


class _Case(unittest.TestCase):
    def use(self, patches):
        mocks = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)
        return mocks


class PickDevice(_Case):
    def test_cuda_round_robin_and_sets_current_gpu(self):
        *_, set_device = self.use(_cuda(True, 2))
        self.assertEqual(dev.pick_device(0), 'cuda:0')
        self.assertEqual(dev.pick_device(3), 'cuda:1')
        set_device.assert_called_with(1)

    def test_cuda_wins_over_mps(self):
        self.use(_cuda(True, 1) + [mock.patch.object(dev, 'mps_available', return_value=True)])
        self.assertEqual(dev.pick_device(5), 'cuda:0')

    def test_mps_when_no_cuda(self):
        self.use(_cuda(False) + [mock.patch.object(dev, 'mps_available', return_value=True)])
        self.assertEqual(dev.pick_device(4), 'mps')

    def test_cpu_fallback_is_an_error_on_macos_only(self):
        self.use(_cuda(False) + [mock.patch.object(dev, 'mps_available', return_value=False)])
        with mock.patch.object(sys, 'platform', 'darwin'), mock.patch.dict('os.environ', clear=False) as env:
            env.pop('MOQNAS_ALLOW_CPU', None)
            with self.assertRaises(RuntimeError):
                dev.pick_device()
            with self.assertRaises(RuntimeError):
                dev.resolve_device('auto')
            with self.assertRaises(RuntimeError):
                dev.require_gpu_on_macos()
            env['MOQNAS_ALLOW_CPU'] = '1'
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                self.assertEqual(dev.pick_device(), 'cpu')
                self.assertEqual(len(w), 1)
        with mock.patch.object(sys, 'platform', 'linux'), warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(dev.pick_device(), 'cpu')
            self.assertEqual(dev.resolve_device(None), 'cpu')
            dev.require_gpu_on_macos()
            self.assertEqual(len(w), 0)


class ResolveDevice(_Case):
    def test_auto_prefers_cuda_then_mps(self):
        self.use(_cuda(True, 2) + [mock.patch.object(dev, 'mps_available', return_value=True)])
        self.assertEqual(dev.resolve_device('auto'), 'cuda')
        self.assertEqual(dev.resolve_device(None), 'cuda')

    def test_auto_mps(self):
        self.use(_cuda(False) + [mock.patch.object(dev, 'mps_available', return_value=True)])
        self.assertEqual(dev.resolve_device('auto'), 'mps')

    def test_explicit_device_is_kept(self):
        self.use(_cuda(False) + [mock.patch.object(dev, 'mps_available', return_value=False)])
        self.assertEqual(dev.resolve_device('cuda'), 'cuda')
        self.assertEqual(dev.resolve_device('cuda:1'), 'cuda:1')
        self.assertEqual(dev.resolve_device('cpu'), 'cpu')
        with self.assertRaises(RuntimeError):
            dev.resolve_device('mps')

    def test_autocast_device_type(self):
        self.assertEqual(dev.autocast_device_type('cuda:1'), 'cuda')
        self.assertEqual(dev.autocast_device_type(torch.device('mps')), 'mps')
        self.assertEqual(dev.autocast_device_type('cpu'), 'cpu')

    def test_require_gpu_on_macos_never_queries_cuda(self):
        with mock.patch.object(torch.cuda, 'is_available', side_effect=AssertionError('cuda queried')), \
                mock.patch.object(torch.cuda, 'device_count', side_effect=AssertionError('cuda queried')), \
                mock.patch.object(dev, 'mps_available', return_value=True):
            for platform in ('darwin', 'linux'):
                with mock.patch.object(sys, 'platform', platform):
                    dev.require_gpu_on_macos()


class DeviceSlots(_Case):
    def test_counts(self):
        self.use(_cuda(True, 2) + [mock.patch.object(dev, 'mps_available', return_value=True)])
        self.assertEqual(dev.device_slots(), 2)

    def test_mps_is_one_slot_and_cpu_is_zero(self):
        self.use(_cuda(False, 0))
        with mock.patch.object(dev, 'mps_available', return_value=True):
            self.assertEqual(dev.device_slots(), 1)
        with mock.patch.object(dev, 'mps_available', return_value=False):
            self.assertEqual(dev.device_slots(), 0)


class Helpers(_Case):
    def test_synchronize_and_empty_cache_dispatch(self):
        with mock.patch.object(torch.cuda, 'is_available', return_value=True), \
             mock.patch.object(torch.cuda, 'synchronize') as cs, \
             mock.patch.object(torch.cuda, 'empty_cache') as ce, \
             mock.patch.object(torch.mps, 'synchronize') as ms, \
             mock.patch.object(torch.mps, 'empty_cache') as me:
            dev.synchronize('cuda:1'); dev.empty_cache(torch.device('cuda:0'))
            self.assertEqual((cs.call_count, ce.call_count, ms.call_count, me.call_count), (1, 1, 0, 0))
            dev.synchronize('mps'); dev.empty_cache(torch.device('mps'))
            self.assertEqual((ms.call_count, me.call_count), (1, 1))
            dev.synchronize('cpu'); dev.empty_cache('cpu')
            self.assertEqual((cs.call_count, ce.call_count, ms.call_count, me.call_count), (1, 1, 1, 1))

    def test_check_precision(self):
        with mock.patch.object(torch, '__version__', '2.5.1'):
            dev.check_precision('mps', 'fp32')
            dev.check_precision('cuda:0', 'bf16')
            for precision in ('fp16', 'bf16'):
                with self.assertRaises(RuntimeError):
                    dev.check_precision(torch.device('mps'), precision)
        with mock.patch.object(torch, '__version__', '2.14.1+cu126'):
            dev.check_precision('mps', 'bf16')
            dev.check_precision('mps', 'fp16')


class ParentPurity(unittest.TestCase):
    def test_importing_modules_never_queries_cuda(self):
        """The evolution parent must not call torch.cuda.is_available()/device_count()."""
        import subprocess
        code = (
            "import torch\n"
            "def boom(*a, **k): raise SystemExit('torch.cuda queried in the parent')\n"
            "torch.cuda.is_available = boom; torch.cuda.device_count = boom\n"
            "import core.device, core.evaluation\n"
            "from core.evaluation import EvalPopulation\n"
            "e = EvalPopulation.__new__(EvalPopulation)\n"
            "e.train_params = {'workers_per_gpu': 2}\n"
            "assert e._num_workers(8) >= 1\n"
            "print('pure')\n")
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertIn('pure', out.stdout, out.stderr[-400:])


if __name__ == '__main__':
    unittest.main()
