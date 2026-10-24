"""Tests for the computation-trace follow-up.

Each case is the check that would have failed on the previous code.
"""

import math
import unittest

import torch
import yaml

from data_provider.data_loader import Dataset_Custom
from tests.synthetic_factory import make_batch, make_model, xattn_config
from main_model import CSDI_Forecasting
from utils.text_presence import text_is_absent


class DeterministicValidTests(unittest.TestCase):
    def _predict_from_noise(self, model):
        masks = []

        def _run(total_input, side_info, t, cfg_mask, **kwargs):
            masks.append(None if cfg_mask is None else cfg_mask.detach().clone())
            noisy = total_input[0][:, 1] if isinstance(total_input, list) else total_input[:, 1]
            return noisy, None

        model._run_diffusion_model = _run
        return masks

    def test_validation_loss_repeats_and_keeps_text(self):
        model = make_model(layers=1, num_steps=2, decomp=False)
        model.c_mask_prob = 1.0
        model.eval()
        batch = make_batch(model)
        model.deterministic_valid = True
        fixed_masks = self._predict_from_noise(model)
        first = model(batch, is_train=0)
        second = model(batch, is_train=0)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(all(torch.equal(mask, torch.ones_like(mask)) for mask in fixed_masks))

        model.deterministic_valid = False
        random_masks = self._predict_from_noise(model)
        third = model(batch, is_train=0)
        fourth = model(batch, is_train=0)
        self.assertFalse(torch.equal(third, fourth))
        self.assertTrue(all(torch.equal(mask, torch.zeros_like(mask)) for mask in random_masks))

    def test_xattn_selects_on_deterministic_main_loss(self):
        with open("config/xattn/agriculture_36_12_xattn.yaml") as handle:
            config = yaml.safe_load(handle)
        self.assertTrue(config["train"]["deterministic_valid"])
        self.assertTrue(config["train"]["valid_loss_main_only"])


class WindowNormTests(unittest.TestCase):
    def test_fewer_than_two_observations_keep_unit_scale(self):
        model = make_model(layers=1, num_steps=2)
        observed = torch.tensor([[[5.0, 0.0, 0.0, 9.0]]])
        mask = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        model.estimable_window_norm = False
        legacy, _, legacy_std = model._standardize_observed(observed, mask)
        self.assertAlmostEqual(float(legacy_std), math.sqrt(1e-5), places=6)
        self.assertGreater(float(legacy.abs().max()), 100.0)

        model.estimable_window_norm = True
        fixed, means, stdev = model._standardize_observed(observed, mask)
        self.assertTrue(torch.allclose(stdev, torch.ones_like(stdev)))
        self.assertAlmostEqual(float(means), 5.0)
        self.assertTrue(torch.allclose(fixed, observed - 5.0))
        self.assertLess(float(fixed.abs().max()), 10.0)

        empty_mask = torch.zeros_like(mask)
        untouched, empty_mean, empty_std = model._standardize_observed(observed, empty_mask)
        self.assertTrue(torch.allclose(empty_std, torch.ones_like(empty_std)))
        self.assertAlmostEqual(float(empty_mean), 0.0)
        self.assertTrue(torch.allclose(untouched, observed))

    def test_two_observations_match_the_legacy_scale(self):
        model = make_model(layers=1, num_steps=2)
        observed = torch.tensor([[[1.0, 3.0, 0.0, 8.0]]])
        mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
        model.estimable_window_norm = False
        legacy, _, _ = model._standardize_observed(observed, mask)
        model.estimable_window_norm = True
        fixed, _, _ = model._standardize_observed(observed, mask)
        self.assertTrue(torch.allclose(legacy, fixed))

    def test_xattn_configs_enable_estimable_window_norm(self):
        with open("config/xattn/agriculture_36_12_xattn.yaml") as handle:
            config = yaml.safe_load(handle)
        self.assertTrue(config["diffusion"]["estimable_window_norm"])


class PlaceholderMarkTests(unittest.TestCase):
    def test_na_explanation_is_absent_only_when_enabled(self):
        placeholder = "NA (No relevant information found for this window)"
        self.assertFalse(text_is_absent(placeholder, placeholder_absent=False))
        self.assertTrue(text_is_absent(placeholder, placeholder_absent=True))
        self.assertTrue(text_is_absent("NA", placeholder_absent=False))
        self.assertTrue(text_is_absent("  na  ", placeholder_absent=False))
        self.assertFalse(text_is_absent("Napa valley prices rose", placeholder_absent=True))
        self.assertFalse(text_is_absent("wheat prices rose", placeholder_absent=True))
        self.assertTrue(text_is_absent("", placeholder_absent=True))

    def test_loader_and_model_marks_agree_for_raw_and_retrieved(self):
        class Holder:
            text_placeholder_absent = True

        normalize = Dataset_Custom._normalize_text
        self.assertEqual(normalize(Holder(), "NA (No relevant information found)"), "")
        self.assertEqual(normalize(Holder(), "NA"), "")
        self.assertEqual(normalize(Holder(), "prices rose"), "prices rose")
        Holder.text_placeholder_absent = False
        self.assertEqual(normalize(Holder(), "NA (No relevant information found)"), "NA (No relevant information found)")

        config = xattn_config(layers=1, num_steps=2)
        config["model"]["text_placeholder_absent"] = True
        model = CSDI_Forecasting(config, torch.device("cpu"), 2, [12, 4])
        texts = [
            "NA",
            "NA (No relevant information found for this window)",
            "prices rose",
        ]
        marks = model._mark_from_text_batch(texts, 3, torch.device("cpu"))
        self.assertTrue(torch.equal(marks, torch.tensor([0.0, 0.0, 1.0])))
        model.text_placeholder_absent = False
        legacy = model._mark_from_text_batch(texts, 3, torch.device("cpu"))
        self.assertTrue(torch.equal(legacy, torch.tensor([0.0, 1.0, 1.0])))

    def test_xattn_configs_mark_placeholders_absent(self):
        with open("config/xattn/economy_36_12_xattn.yaml") as handle:
            a1 = yaml.safe_load(handle)
        with open("config/xattn/ablation/traffic_T0_numeric_only.yaml") as handle:
            t0 = yaml.safe_load(handle)
        self.assertTrue(a1["model"]["text_placeholder_absent"])
        self.assertTrue(t0["model"]["text_placeholder_absent"])


class _DropoutEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = torch.nn.Dropout(p=0.5)


class TextEncoderEvalTests(unittest.TestCase):
    def _model(self, enabled):
        config = xattn_config(layers=1, num_steps=2)
        config["model"]["text_encoder_eval"] = enabled
        model = CSDI_Forecasting(config, torch.device("cpu"), 2, [12, 4])
        model.text_encoder = _DropoutEncoder()
        return model

    def test_train_leaves_the_text_encoder_in_eval_when_enabled(self):
        model = self._model(True)
        model.train()
        self.assertTrue(model.training)
        self.assertFalse(model.text_encoder.training)
        self.assertFalse(model.text_encoder.drop.training)
        model.eval()
        model.train()
        self.assertFalse(model.text_encoder.training)

    def test_legacy_train_still_enables_text_encoder_dropout(self):
        model = self._model(False)
        model.train()
        self.assertTrue(model.text_encoder.training)
        self.assertTrue(model.text_encoder.drop.training)

    def test_cache_is_per_string_and_cleared_on_load(self):
        model = self._model(True)
        text = " ".join("w%d" % index for index in range(30))
        first, _, _ = model._encode_text_tokens([text, text], truncation_side="left")
        self.assertEqual(len(model._text_encode_cache), 1)
        second, _, _ = model._encode_text_tokens([text], truncation_side="left")
        self.assertEqual(len(model._text_encode_cache), 1)
        self.assertTrue(torch.equal(first[0].cpu(), second[0].cpu()))
        other_side, _, _ = model._encode_text_tokens([text], truncation_side="right")
        self.assertEqual(len(model._text_encode_cache), 2)
        self.assertFalse(torch.equal(first[0].cpu(), other_side[0].cpu()))
        model.load_state_dict(model.state_dict())
        self.assertEqual(model._text_encode_cache, {})

    def test_xattn_configs_enable_the_flag(self):
        with open("config/xattn/agriculture_36_12_xattn.yaml") as handle:
            a1 = yaml.safe_load(handle)
        with open("config/xattn/ablation/agriculture_T1_router_text.yaml") as handle:
            t1 = yaml.safe_load(handle)
        self.assertTrue(a1["model"]["text_encoder_eval"])
        self.assertTrue(t1["model"]["text_encoder_eval"])
        legacy = xattn_config()
        legacy["model"].pop("text_encoder_eval", None)
        self.assertFalse(CSDI_Forecasting(legacy, torch.device("cpu"), 2, [12, 4]).text_encoder_eval)
