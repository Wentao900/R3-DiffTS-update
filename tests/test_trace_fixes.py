"""Tests for the computation-trace follow-up.

Each case is the check that would have failed on the previous code.
"""

import unittest

import torch
import yaml

from tests.synthetic_factory import make_model, xattn_config
from main_model import CSDI_Forecasting


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
