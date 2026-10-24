"""Tests for the sampler, Huber, IDF, dropout, and guide-weight review fixes.

These assert the closed forms. A loop that only records a call at t=0 would
still pass on the old DDPM sampler, because that call already happened and
its output was discarded.
"""

import math
import unittest

import numpy as np
import torch
import yaml

from diff_models import ResidualBlock
from main_model import CSDI_Forecasting
from tests.synthetic_factory import xattn_config


class _StepRecorder(torch.nn.Module):
    def __init__(self, value=4.0):
        super().__init__()
        self.steps = []
        self.value = value

    def forward(self, diff_input, side_info, diffusion_step, *args, **kwargs):
        self.steps.append(int(diffusion_step.reshape(-1)[0].item()))
        return torch.full_like(diff_input[:, 0], self.value)


def _sampler_model(**diffusion_overrides):
    num_steps = int(diffusion_overrides.get("num_steps", 4))
    config = xattn_config(decomp=False, lookback=8, pred=4, num_steps=num_steps, layers=1, channels=32)
    config["model"]["pattern_residual_diffusion"] = False
    config["model"]["text_injection"] = "off"
    config["diffusion"].update(diffusion_overrides)
    return CSDI_Forecasting(config, torch.device("cpu"), 2, [8, 4])


def _run_impute(model):
    observed = torch.arange(16, dtype=torch.float32).view(1, 1, 16).repeat(1, 2, 1)
    observed = observed + torch.tensor([0.0, 3.0]).view(1, 2, 1)
    cond_mask = torch.zeros(1, 2, 16)
    cond_mask[:, :, :8] = 1.0
    samples = model.impute(observed, cond_mask, side_info=None, n_samples=1)
    return observed, cond_mask, samples


def _invert_revin(observed, cond_mask, samples):
    cond_count = torch.sum(cond_mask, dim=2, keepdim=True)
    safe_count = cond_count.clamp(min=1.0)
    means = torch.sum(observed * cond_mask, dim=2, keepdim=True) / safe_count
    var_denom = (cond_count - 1.0).clamp(min=1.0)
    stdev = torch.sqrt(
        torch.sum((observed - means) ** 2 * cond_mask, dim=2, keepdim=True) / var_denom + 1e-5
    )
    return (samples[:, 0] - means) / stdev


class SamplerIndexTests(unittest.TestCase):
    def test_ddim_denoiser_receives_schedule_timesteps(self):
        model = _sampler_model(
            num_steps=20,
            ddim=True,
            sample_steps=4,
            sample_method="linear",
            sampler_fix=True,
        )
        time_steps, _ = model._build_ddim_schedule()
        recorder = _StepRecorder()
        model.diffmodel = recorder
        _run_impute(model)
        expected = [int(time_steps[t]) for t in range(model.sample_steps - 1, -1, -1)]
        self.assertEqual(recorder.steps, expected)
        self.assertEqual(expected, [16, 11, 6, 1])
        self.assertNotEqual(expected, [3, 2, 1, 0])

    def test_legacy_ddim_still_passes_the_loop_index(self):
        model = _sampler_model(
            num_steps=20,
            ddim=True,
            sample_steps=4,
            sample_method="linear",
            sampler_fix=False,
        )
        recorder = _StepRecorder()
        model.diffmodel = recorder
        _run_impute(model)
        self.assertEqual(recorder.steps, [3, 2, 1, 0])

    def test_economy_quad_schedule_noise_level_is_not_the_loop_index(self):
        model = _sampler_model(
            num_steps=300,
            beta_start=0.0001,
            beta_end=0.6,
            schedule="quad",
            ddim=True,
            sample_steps=20,
            sample_method="quad",
            sampler_fix=True,
        )
        time_steps, _ = model._build_ddim_schedule()
        first_tau = int(time_steps[-1])
        loop_index = model.sample_steps - 1
        self.assertEqual(first_tau, 241)
        self.assertNotEqual(first_tau, loop_index)
        alpha_told = float(model.alpha[loop_index])
        alpha_true = float(model.alpha[first_tau])
        self.assertGreater(alpha_told, 0.97)
        self.assertLess(alpha_true, 1e-15)
        recorder = _StepRecorder()
        model.diffmodel = recorder
        _run_impute(model)
        self.assertEqual(recorder.steps[0], first_tau)
        self.assertEqual(recorder.steps, [int(time_steps[t]) for t in range(19, -1, -1)])


class DdpmEndpointTests(unittest.TestCase):
    def test_ddpm_returns_the_t0_x0_prediction(self):
        torch.manual_seed(0)
        model = _sampler_model(num_steps=4, ddim=False, sampler_fix=True, noise_esti=False)
        recorder = _StepRecorder(value=4.0)
        model.diffmodel = recorder
        observed, cond_mask, samples = _run_impute(model)
        self.assertEqual(recorder.steps, [3, 2, 1, 0])
        restored = _invert_revin(observed, cond_mask, samples)
        self.assertTrue(torch.allclose(restored, torch.full_like(restored, 4.0)))

    def test_legacy_ddpm_does_not_return_the_t0_prediction(self):
        torch.manual_seed(0)
        model = _sampler_model(num_steps=4, ddim=False, sampler_fix=False, noise_esti=False)
        model.diffmodel = _StepRecorder(value=4.0)
        observed, cond_mask, samples = _run_impute(model)
        restored = _invert_revin(observed, cond_mask, samples)
        self.assertFalse(torch.allclose(restored, torch.full_like(restored, 4.0), atol=1e-4))


class DdimCoefficientTests(unittest.TestCase):
    def test_x0_ddim_direction_matches_song_eta_zero(self):
        model = _sampler_model(num_steps=4, ddim=True, sample_steps=2, sampler_fix=True)
        alpha_prev, alpha_tau = 0.4, 0.1
        fixed = model._ddim_x0_direction_coeff(alpha_prev, alpha_tau)
        self.assertAlmostEqual(fixed, math.sqrt((1.0 - alpha_prev) / (1.0 - alpha_tau)))
        self.assertAlmostEqual(model._ddim_x0_direction_coeff(0.25, 0.25), 1.0)
        model.sampler_fix = False
        legacy = model._ddim_x0_direction_coeff(alpha_prev, alpha_tau)
        self.assertAlmostEqual(legacy, (1.0 - alpha_prev) / math.sqrt(1.0 - alpha_tau))
        self.assertNotAlmostEqual(legacy, fixed)
        self.assertAlmostEqual(model._ddim_x0_direction_coeff(0.25, 0.25), math.sqrt(0.75))


def _feature_dropout_p(block):
    layer = block.feature_layer.layers[0]
    dropouts = [module.p for module in layer.modules() if isinstance(module, torch.nn.Dropout)]
    return dropouts[0]


class TowerDropoutTests(unittest.TestCase):
    def _block(self, text_mode, tower_dropout, dropout=0.0):
        return ResidualBlock(
            side_dim=4,
            channels=32,
            diffusion_embedding_dim=16,
            nheads=4,
            dropout=dropout,
            pred_len=4,
            use_tv=False,
            with_text=text_mode != "off",
            context_dim=32,
            text_mode=text_mode,
            tower_dropout=tower_dropout,
        )

    def test_explicit_zero_overrides_the_legacy_point_one(self):
        self.assertEqual(_feature_dropout_p(self._block("legacy", 0.0)), 0.0)
        self.assertEqual(_feature_dropout_p(self._block("legacy", None)), 0.1)
        self.assertEqual(_feature_dropout_p(self._block("gated", None, dropout=0.0)), 0.0)
        self.assertEqual(_feature_dropout_p(self._block("gated", 0.0)), 0.0)

    def test_a1_and_t1_configs_share_tower_dropout(self):
        with open("config/xattn/agriculture_36_12_xattn.yaml") as handle:
            a1 = yaml.safe_load(handle)
        with open("config/xattn/ablation/agriculture_T1_router_text.yaml") as handle:
            t1 = yaml.safe_load(handle)
        with open("config/xattn/ablation/agriculture_T0_numeric_only.yaml") as handle:
            t0 = yaml.safe_load(handle)
        self.assertTrue(a1["diffusion"]["sampler_fix"])
        self.assertEqual(float(a1["diffusion"]["tower_dropout"]), 0.0)
        self.assertEqual(float(t1["diffusion"]["tower_dropout"]), 0.0)
        self.assertEqual(float(t0["diffusion"]["tower_dropout"]), 0.0)
        self.assertEqual(t1["model"]["text_injection"], "off")
        self.assertEqual(a1["model"]["text_injection"], "gated")
        self.assertEqual(float(a1["model"]["guide_w_default"]), 1.0)


class ScheduleClosedFormTests(unittest.TestCase):
    def test_linear_schedule_matches_the_stride_formula(self):
        model = _sampler_model(num_steps=20, ddim=True, sample_steps=4, sample_method="linear")
        time_steps, time_steps_prev = model._build_ddim_schedule()
        np.testing.assert_array_equal(time_steps, np.array([1, 6, 11, 16]))
        np.testing.assert_array_equal(time_steps_prev, np.array([0, 1, 6, 11]))
