"""One behavior check for each protocol switch added after a90347e.

C1–C17 and D1–D2 are behind flags. Legacy configs omit the keys. The xattn
YAMLs turn the main-method switches on and leave cyclic shift and the sampling
validation schedule off.
"""

import importlib.util
import io
import math
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from data_provider.data_loader import Dataset_Custom
from utils.infer_override import apply_infer_override, parse_infer_override
from main_model import CSDI_Forecasting
from tests.synthetic_factory import make_batch, make_model, xattn_config
from tests.test_review_fixes import _invert_revin, _run_impute, _sampler_model
from utils.SelfAttention_Family import FullAttention
from utils.rag_cot import RAGCoTConfig, RAGCoTPipeline
from utils.text_protocol import (
    bert_content_budget,
    cyclic_source_shift,
    decide_text_sources,
    drop_future_year_rows,
    fill_token_budget,
    strip_na_tokens,
)
from utils.utils import evaluate
from utils.xattn_config import apply_entrypoint_overrides


LEGACY_SWITCHES = (
    "sampler_fix",
    "text_encoder_eval",
    "text_placeholder_absent",
    "idf_fit_train_only",
    "deterministic_valid",
    "valid_loss_main_only",
    "estimable_window_norm",
)


def _write_toy(root, n=24, reports=None, searches=None):
    num_dir = Path(root) / "numerical" / "Agriculture"
    text_dir = Path(root) / "textual" / "Agriculture"
    num_dir.mkdir(parents=True)
    text_dir.mkdir(parents=True)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    pd.DataFrame({
        "date": dates,
        "start_date": dates,
        "end_date": dates,
        "OT": np.linspace(1.0, 2.0, n),
    }).to_csv(num_dir / "Agriculture.csv", index=False)
    if reports is None:
        reports = pd.DataFrame({
            "start_date": dates,
            "end_date": dates,
            "fact": ["report fact %d wheat" % index for index in range(n)],
        })
    reports.to_csv(text_dir / "Agriculture_report.csv", index=False)
    if searches is None:
        searches = pd.DataFrame({
            "start_date": dates,
            "end_date": dates,
            "fact": ["search fact %d maize" % index for index in range(n)],
        })
    searches.to_csv(text_dir / "Agriculture_search.csv", index=False)
    return dates


def _dataset(root, **kwargs):
    settings = dict(
        root_path=str(root),
        flag="train",
        size=[4, 2],
        features="S",
        data_path="Agriculture/Agriculture.csv",
        target="OT",
        scale=False,
        timeenc=1,
        freq="m",
        text_len=4,
        text_features_mode="none",
        use_rag_cot=False,
        skip_cot_generation=True,
    )
    settings.update(kwargs)
    return Dataset_Custom(**settings)


def _legacy_model(**model_flags):
    config = xattn_config(decomp=False, lookback=8, pred=4, num_steps=4, layers=1, channels=32)
    config["model"]["force_legacy_entrypoint"] = True
    config["model"]["pattern_residual_diffusion"] = False
    config["model"]["text_injection"] = "off"
    config["model"].update(model_flags)
    apply_entrypoint_overrides(config, freq="m")
    return config


class TokenBudgetTests(unittest.TestCase):
    def test_fill_uses_the_budget_and_no_document_cap(self):
        segments = ["doc%d token" % index for index in range(10)]
        filled = fill_token_budget(segments, 7)
        self.assertEqual(len(filled.split()), 7)
        self.assertTrue(filled.startswith("doc0"))
        self.assertIn("doc3", filled)
        self.assertEqual(bert_content_budget(192), 190)
        self.assertEqual(bert_content_budget(160), 158)

    def test_time_aligned_concat_fills_the_domain_budget(self):
        with tempfile.TemporaryDirectory() as root:
            dates = pd.date_range("2000-01-01", periods=24, freq="MS")
            searches = pd.DataFrame({
                "start_date": dates,
                "end_date": dates,
                "fact": ["outside"] * 24,
            })
            searches.loc[0, "fact"] = "old1 old2"
            searches.loc[1, "fact"] = "mid1 mid2"
            searches.loc[2, "fact"] = "NA"
            searches.loc[3, "fact"] = "new1 new2 new3"
            searches.loc[4, "fact"] = "future1 future2"
            _write_toy(root, searches=searches)
            aligned = _dataset(root, ext_text_mode="time_aligned", text_max_length=6)
            item = aligned[0]
            self.assertEqual(item["text_ret"], "new1 new2 new3 mid1")
            self.assertNotIn("old1", item["text_ret"])
            self.assertNotIn("future1", item["text_ret"])
            self.assertEqual(float(item["text_ret_mark"]), 1.0)
            legacy = _dataset(root, ext_text_mode="tfidf")
            self.assertEqual(legacy[0]["text_ret"], "")

    def test_time_aligned_does_not_call_tfidf(self):
        with tempfile.TemporaryDirectory() as root:
            _write_toy(root)
            dataset = _dataset(
                root,
                ext_text_mode="time_aligned",
                text_max_length=8,
                use_rag_cot=True,
                rag_topk=1,
            )

            def boom(*args, **kwargs):
                raise AssertionError("TF-IDF query ran on the time-aligned path")

            dataset.rag_cot.build_guidance_text = boom
            item = dataset[0]
            self.assertNotEqual(item["text_ret"], "")


class FutureYearFilterTests(unittest.TestCase):
    def test_year_rule_drops_later_years_and_writes_the_audit(self):
        with tempfile.TemporaryDirectory() as root:
            dates = pd.date_range("2000-01-01", periods=24, freq="MS")
            searches = pd.DataFrame({
                "start_date": dates,
                "end_date": dates,
                "fact": ["kept maize"] * 24,
            })
            searches.loc[5, "fact"] = "outlook 2025 prices"
            reports = pd.DataFrame({
                "start_date": dates,
                "end_date": dates,
                "fact": ["kept wheat"] * 24,
            })
            reports.loc[2, "fact"] = "code 2024 noted"
            _write_toy(root, reports=reports, searches=searches)
            dataset = _dataset(root, ext_future_year_filter=True)
            facts = set(dataset.search_df["fact"])
            self.assertNotIn("outlook 2025 prices", facts)
            self.assertIn("kept maize", facts)
            self.assertNotIn("code 2024 noted", set(dataset.txt_report["fact"]))
            path = Path(root) / "removed.csv"
            dataset.write_future_year_audit(path)
            audit = pd.read_csv(path)
            self.assertEqual(set(audit["source"]), {"report", "search"})
            self.assertTrue(audit["fact"].str.contains("2025").any())
            self.assertTrue(audit["fact"].str.contains("2024").any())
            untouched = _dataset(root, ext_future_year_filter=False)
            self.assertIn("outlook 2025 prices", set(untouched.search_df["fact"]))


class RawPlaceholderTests(unittest.TestCase):
    def test_placeholder_only_window_has_raw_mark_zero(self):
        with tempfile.TemporaryDirectory() as root:
            dates = pd.date_range("2000-01-01", periods=24, freq="MS")
            facts = ["real wheat report"] * 24
            facts[0] = "NA"
            facts[1] = "NA (no reading)"
            facts[2] = "NA"
            facts[3] = "NA (placeholder)"
            reports = pd.DataFrame({"start_date": dates, "end_date": dates, "fact": facts})
            _write_toy(root, reports=reports)
            filtered = _dataset(root, raw_placeholder_filter=True)
            first = filtered[0]
            self.assertEqual(float(first["text_raw_mark"]), 0.0)
            self.assertNotIn("historical reporting", first["text_raw"].lower())
            later = filtered[4]
            self.assertEqual(float(later["text_raw_mark"]), 1.0)
            self.assertIn("real wheat report", later["text_raw"])
            legacy = _dataset(root, raw_placeholder_filter=False)
            self.assertEqual(float(legacy[0]["text_raw_mark"]), 1.0)
            self.assertIn("historical reporting", legacy[0]["text_raw"].lower())


class SourceDecisionTests(unittest.TestCase):
    def test_unused_source_is_dropped_and_constant_beta_folds_into_b(self):
        decision = decide_text_sources([0, 0, 0], [1, 1, 1])
        self.assertEqual(decision["dropped"], ["raw"])
        self.assertEqual(decision["beta_keep"], [False, False])
        self.assertEqual(decision["n_gate_parameters"], 2)
        mixed = decide_text_sources([1, 0, 1], [1, 1, 1])
        self.assertEqual(mixed["dropped"], [])
        self.assertEqual(mixed["beta_keep"], [True, False])
        config = xattn_config(layers=1, num_steps=2)
        config["model"]["text_sources"] = "auto_from_train"
        config["model"]["gate_beta"] = "auto_identifiable"
        model = CSDI_Forecasting(config, torch.device("cpu"), 2, [12, 4])
        before = model.diffmodel.text_source_emb.num_embeddings
        beta_count = int(model.text_gate.beta.numel())
        info = model.apply_source_decision(mixed)
        self.assertEqual(before, 2)
        self.assertEqual(model.diffmodel.text_source_emb.num_embeddings, before)
        self.assertEqual(beta_count, 2)
        self.assertEqual(info["gate_parameters"], 3)
        self.assertEqual(info["dropped"], [])
        self.assertEqual(model.text_gate.beta_mask.tolist(), [1.0, 0.0])
        dropped = model.apply_source_decision(decision)
        self.assertEqual(dropped["dropped"], ["raw"])
        self.assertEqual(dropped["gate_parameters"], 2)
        raw, ret, _ = model._apply_source_masks(torch.ones(2), torch.ones(2), torch.zeros(2))
        self.assertTrue(torch.equal(raw, torch.zeros(2)))
        self.assertTrue(torch.equal(ret, torch.ones(2)))
        model.text_sources_mode = "configured"
        kept = model.apply_source_decision(decision)
        self.assertEqual(kept["dropped"], [])


class TextShiftTests(unittest.TestCase):
    def test_shift_gap_is_the_history_length_and_short_pairs_stay(self):
        short = cyclic_source_shift([0, 1, 2], 10)
        self.assertEqual(short["unshifted"], 3)
        self.assertEqual(short["donor"], [0, 1, 2])
        wide = cyclic_source_shift([0, 10, 20], 8)
        self.assertEqual(wide["unshifted"], 0)
        self.assertGreaterEqual(wide["min_gap"], 8)

    def test_loader_shifts_content_and_keeps_marks(self):
        with tempfile.TemporaryDirectory() as root:
            _write_toy(root)
            plain = _dataset(root, text_shift_mode="none")
            shifted = _dataset(
                root,
                text_shift_mode="cyclic_per_source",
                text_shift_splits=["train"],
            )
            report = shifted.text_shift_report["raw"]
            self.assertEqual(report["history_length"], shifted.seq_len)
            self.assertGreaterEqual(report["min_gap"], shifted.seq_len)
            self.assertEqual(report["unshifted"], 0)
            self.assertNotEqual(shifted[0]["text_raw"], plain[0]["text_raw"])
            self.assertEqual(float(shifted[0]["text_raw_mark"]), float(plain[0]["text_raw_mark"]))
            held = _dataset(
                root,
                flag="valid",
                text_shift_mode="cyclic_per_source",
                text_shift_splits=["train"],
            )
            self.assertEqual(held.text_shift_report, {})


class GateModeTests(unittest.TestCase):
    def test_source_only_drops_the_similarity_term(self):
        config = xattn_config(layers=1, num_steps=2)
        config["model"]["text_gate_mode"] = "source_only"
        model = CSDI_Forecasting(config, torch.device("cpu"), 2, [12, 4])
        model.text_gate.a.data.fill_(5.0)
        model.text_gate.b.data.fill_(0.0)
        similarity = torch.tensor([1.0, 0.0])
        mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        values = model._text_gate_value(similarity, mask)
        self.assertTrue(torch.allclose(values, torch.full((2,), 0.5)))
        model.text_gate_mode = "learned"
        learned = model._text_gate_value(similarity, mask)
        self.assertGreater(float(learned[0]), float(learned[1]))

    def test_constant_gate_uses_the_last_epoch_mean(self):
        config = xattn_config(layers=1, num_steps=2)
        config["model"]["text_gate_mode"] = "constant"
        config["model"]["gate_constant_source"] = "last_epoch"
        model = CSDI_Forecasting(config, torch.device("cpu"), 2, [12, 4])
        model.train()
        self.assertAlmostEqual(float(model._current_gate_constant()), 0.5)
        model._update_gate_label_mean(torch.tensor([0.2, 0.4]), torch.tensor([True, True]))
        self.assertAlmostEqual(float(model._current_gate_constant()), 0.5)
        model.close_gate_label_epoch()
        self.assertAlmostEqual(float(model._current_gate_constant()), 0.3, places=6)
        model._update_gate_label_mean(torch.tensor([1.0]), torch.tensor([True]))
        self.assertAlmostEqual(float(model._current_gate_constant()), 0.3, places=6)
        cumulative = (0.2 + 0.4 + 1.0) / 3.0
        self.assertNotAlmostEqual(float(model._current_gate_constant()), cumulative, places=3)
        model.close_gate_label_epoch()
        self.assertAlmostEqual(float(model._current_gate_constant()), 1.0, places=6)
        other = CSDI_Forecasting(config, torch.device("cpu"), 2, [12, 4])
        other.load_state_dict(model.state_dict())
        self.assertAlmostEqual(float(other._current_gate_constant()), 1.0, places=6)
        legacy = make_model(layers=1, num_steps=2)
        self.assertNotIn("gate_epoch_sum", legacy.state_dict())
        legacy.close_gate_label_epoch()
        legacy._update_gate_label_mean(torch.tensor([0.2]), torch.tensor([True]))
        self.assertAlmostEqual(float(legacy._current_gate_constant()), 0.2, places=6)


class SharedDropoutTests(unittest.TestCase):
    def test_time_attention_dropout_defaults_to_point_one_and_replays(self):
        model = make_model(layers=1, num_steps=2)
        self.assertAlmostEqual(model.time_attn_dropout, 0.1)
        layers = [module for module in model.modules() if isinstance(module, FullAttention) and module.time_axis]
        self.assertTrue(layers)
        self.assertAlmostEqual(layers[0].dropout.p, 0.1)
        config = xattn_config(layers=1, num_steps=2)
        config["diffusion"]["time_attn_dropout"] = 0.0
        changed = CSDI_Forecasting(config, torch.device("cpu"), 2, [12, 4])
        changed_layers = [
            module for module in changed.modules() if isinstance(module, FullAttention) and module.time_axis
        ]
        self.assertAlmostEqual(changed_layers[0].dropout.p, 0.0)

        attention = FullAttention(mask_flag=False, attention_dropout=0.5, output_attention=False)
        attention.time_axis = True
        attention.train()
        queries = torch.randn(2, 4, 2, 8)
        attention._dropout_mode = "record"
        torch.manual_seed(0)
        first, _ = attention(queries, queries, queries, None)
        attention._dropout_mode = "replay"
        second, _ = attention(queries, queries, queries, None)
        self.assertTrue(torch.equal(first, second))
        attention._dropout_mode = None
        torch.manual_seed(1)
        third, _ = attention(queries, queries, queries, None)
        torch.manual_seed(2)
        fourth, _ = attention(queries, queries, queries, None)
        self.assertFalse(torch.equal(third, fourth))

    def test_reference_forward_reuses_the_recorded_mask(self):
        config = xattn_config(layers=1, num_steps=2)
        config["model"]["gate_ref_shared_dropout"] = True
        config["diffusion"]["time_attn_dropout"] = 0.5
        model = CSDI_Forecasting(config, torch.device("cpu"), 2, [12, 4])
        model.train()
        batch = make_batch(model)
        model(batch, is_train=1)
        armed = [
            module for module in model.modules()
            if isinstance(module, FullAttention) and module.time_axis
        ]
        self.assertTrue(all(module._dropout_mode is None for module in armed))
        self.assertTrue(any(module._dropout_masks for module in armed))


class InferenceOverrideTests(unittest.TestCase):
    def test_evaluate_loads_the_checkpoint_argument(self):
        model = make_model(layers=1, num_steps=2)
        model.text_gate.b.data.fill_(0.0)
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "trained" / "model.pth"
            checkpoint.parent.mkdir()
            torch.save(model.state_dict(), checkpoint)
            model.text_gate.b.data.fill_(4.0)
            other = Path(root) / "newrun"
            other.mkdir()
            torch.save({"not": "a state"}, other / "model.pth")

            class Empty:
                def __iter__(self):
                    return iter([])

                def __len__(self):
                    return 0

            with self.assertRaises(Exception):
                evaluate(
                    model,
                    Empty(),
                    nsample=1,
                    foldername=str(other) + "/",
                    checkpoint=str(checkpoint),
                    reload_weights=True,
                )
            self.assertAlmostEqual(float(model.text_gate.b.detach()), 0.0, places=6)

    def test_infer_override_sets_w_gate_shift_and_raw_mask(self):
        model = make_model(layers=1, num_steps=2)
        model.log_control_calls = True
        model.control_log = {}

        class Dataset:
            split_name = "test"
            seq_len = 4

            def __init__(self):
                self.text_shift_mode = "none"
                self.text_shift_splits = []
                self.calls = 0

            def _prepare_text_shift(self):
                self.calls += 1

        class Loader:
            def __init__(self, dataset):
                self.dataset = dataset

        dataset = Dataset()
        spec = parse_infer_override(
            "guide_w=0,text_gate_mode=one,mask_sources=raw,text_shift_mode=cyclic_per_source"
        )
        apply_infer_override(model, [Loader(dataset)], spec)
        self.assertEqual(model.infer_guide_w, 0.0)
        self.assertEqual(model.text_gate_mode, "one")
        self.assertEqual(model.infer_mask_sources, ["raw"])
        self.assertEqual(dataset.calls, 1)
        self.assertEqual(dataset.text_shift_splits, ["test"])
        self.assertEqual(model.control_log["infer_override"], 1)
        raw, ret, _ = model._apply_source_masks(torch.ones(1), torch.ones(1), torch.zeros(1))
        self.assertEqual(float(raw), 0.0)
        self.assertEqual(float(ret), 1.0)
        values = model._text_gate_value(torch.tensor([0.2]), torch.ones(1, 2))
        self.assertAlmostEqual(float(values), 1.0)
        apply_infer_override(model, [], {"gate_constant": "0.25"})
        self.assertEqual(model.text_gate_mode, "constant")
        constant = model._text_gate_value(torch.ones(2), torch.ones(2, 2))
        self.assertTrue(torch.allclose(constant, torch.full((2,), 0.25)))

        captured = {}

        def fake_impute(observed_data, cond_mask, side_info, n_samples, guide_w=None, **kwargs):
            captured["guide_w"] = guide_w
            batch, channels, length = observed_data.shape
            return torch.zeros(batch, n_samples, channels, length)

        model.impute = fake_impute
        model.infer_guide_w = 0.0
        model.evaluate(make_batch(model, batch_size=1), 1, 7.0)
        self.assertEqual(captured["guide_w"], 0.0)


class _FixedEvaluator(torch.nn.Module):
    def __init__(self, draw=False):
        super().__init__()
        self.draw = draw
        self._last_gate = torch.tensor([[[0.3]]])

    def evaluate(self, batch, nsample, guide_w):
        noise = torch.randn(3) if self.draw else torch.tensor([0.0, 0.0, 9.0])
        samples = torch.zeros(1, 3, 1, 4)
        samples[0, :, 0, -1] = noise
        target = torch.zeros(1, 1, 4)
        points = torch.zeros(1, 1, 4)
        points[0, 0, -1] = 1.0
        observed = torch.ones(1, 1, 4)
        times = torch.zeros(1, 4)
        return samples, target, points, observed, times


class EvalMetricTests(unittest.TestCase):
    def _batch(self):
        return {
            "text_raw_mark": torch.tensor([1.0]),
            "text_ret_mark": torch.tensor([1.0]),
            "text_cot_mark": torch.tensor([0.0]),
            "window_index": torch.tensor([7]),
            "history_end": ["2001-01-01"],
        }

    def test_reseed_makes_two_evaluations_match(self):
        model = _FixedEvaluator(draw=True)
        with tempfile.TemporaryDirectory() as root:
            folder = str(Path(root)) + "/"
            first = evaluate(model, [self._batch()], nsample=1, foldername=folder, reload_weights=False, reseed=2025)
            second = evaluate(model, [self._batch()], nsample=1, foldername=folder, reload_weights=False, reseed=2025)
            third = evaluate(model, [self._batch()], nsample=1, foldername=folder, reload_weights=False, reseed=None)
            self.assertAlmostEqual(first["MSE"], second["MSE"])
            self.assertNotAlmostEqual(second["MSE"], third["MSE"])

    def test_window_file_records_seed_and_source_combo(self):
        model = _FixedEvaluator(draw=False)
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / "window_metrics_seed2025_test.csv")
            folder = str(Path(root)) + "/"
            evaluate(
                model,
                [self._batch()],
                nsample=1,
                foldername=folder,
                reload_weights=False,
                window_metrics_path=path,
                seed=2025,
                split_name="test",
            )
            frame = pd.read_csv(path)
            self.assertEqual(list(frame.columns), [
                "seed", "split", "window_index", "mse_mean", "mae_median", "crps",
                "m_raw", "m_ret", "m_cot", "source_combo", "history_end", "g",
            ])
            self.assertEqual(int(frame.loc[0, "seed"]), 2025)
            self.assertEqual(int(frame.loc[0, "window_index"]), 7)
            self.assertEqual(frame.loc[0, "source_combo"], "raw+ext")
            self.assertEqual(frame.loc[0, "history_end"], "2001-01-01")

    def test_mse_uses_the_mean_and_mae_uses_the_median(self):
        model = _FixedEvaluator(draw=False)
        with tempfile.TemporaryDirectory() as root:
            folder = str(Path(root)) + "/"
            paired = evaluate(
                model, [self._batch()], nsample=1, foldername=folder,
                reload_weights=False, point_metric_pairing=True,
            )
            plain = evaluate(
                model, [self._batch()], nsample=1, foldername=folder,
                reload_weights=False, point_metric_pairing=False,
            )
        self.assertAlmostEqual(paired["MSE"], 9.0, places=5)
        self.assertAlmostEqual(paired["MAE"], 0.0, places=5)
        self.assertAlmostEqual(plain["MAE"], 3.0, places=5)
        self.assertEqual(paired["point_estimator"], "mean_mse_median_mae")


class ControlLogTests(unittest.TestCase):
    def test_counters_move_only_when_logging_is_on(self):
        quiet = make_model(layers=1, num_steps=2)
        self.assertIsNone(quiet.control_log)
        config = xattn_config(layers=1, num_steps=2)
        config["model"]["log_control_calls"] = True
        model = CSDI_Forecasting(config, torch.device("cpu"), 2, [12, 4])
        model.train()
        loss = model(make_batch(model), is_train=1)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.control_log.get("gate_calls", 0), 0)
        self.assertGreater(model.control_log.get("xattn_calls", 0), 0)
        model.eval()
        model.evaluate(make_batch(model, batch_size=1), 1, 1.0)
        self.assertGreater(model.control_log.get("cfg_combines", 0), 0)
        model.infer_mask_sources = ["ret"]
        model._apply_source_masks(torch.ones(1), torch.ones(1), torch.zeros(1))
        self.assertGreater(model.control_log.get("mask_ret", 0), 0)


class SeedLoopTests(unittest.TestCase):
    def test_protocol_seeds_and_dry_run_cover_eight_domains(self):
        seed_text = Path("config/xattn/protocol_seeds.yaml").read_text()
        self.assertNotIn("n_min", seed_text)
        self.assertNotIn("2027", seed_text)
        with open("config/xattn/protocol_seeds.yaml") as handle:
            protocol = yaml.safe_load(handle)
        self.assertEqual(protocol["seeds"], [2024, 2025, 2026])
        env = os.environ.copy()
        env.pop("SEEDS", None)
        env.update({
            "DRY_RUN": "1",
            "DOMAIN": "all",
            "ABLATION": "A1",
        })
        output = subprocess.check_output(["bash", "scripts/run_xattn_ablation.sh"], env=env, text=True)
        commands = [line for line in output.splitlines() if "exe_forecasting.py" in line]
        self.assertEqual(len(commands), 24)
        for domain in ("agriculture", "climate", "economy", "energy", "environment", "health", "socialgood", "traffic"):
            matched = [line for line in commands if "--domain %s" % domain in line]
            self.assertEqual(len(matched), 3)
        self.assertTrue(all("--config xattn/unified.yaml" in line for line in commands))
        self.assertTrue(all("--variant A1" in line for line in commands))
        for seed in ("2024", "2025", "2026"):
            self.assertTrue(any("--seed %s" % seed in line for line in commands))
        source = Path("exe_forecasting.py").read_text()
        self.assertIn("unified_output_dir", source)
        self.assertIn('foldername = "./save/%s_seed%s/" % (config_stem, args.seed)', source)


class SamplingScheduleTests(unittest.TestCase):
    def test_sampling_schedule_is_opt_in(self):
        model = make_model(layers=1, num_steps=4)
        model.ddim = True
        model.sample_steps = 2
        model.sample_method = "linear"
        seen = []

        def fake_calc_loss(*args, **kwargs):
            seen.append(kwargs["set_t"])
            return torch.zeros(())

        model.calc_loss = fake_calc_loss
        zeros = torch.zeros(1, 1, 4)
        model.valid_t_schedule = "all"
        model.calc_loss_valid(zeros, zeros, zeros, None, 0)
        self.assertEqual(seen, [0, 1, 2, 3])
        seen.clear()
        model.valid_t_schedule = "sampling"
        model.calc_loss_valid(zeros, zeros, zeros, None, 0)
        schedule, _ = model._build_ddim_schedule()
        self.assertEqual(seen, [int(value) for value in schedule])
        self.assertLess(len(seen), model.num_steps)
        self.assertEqual(make_model(layers=1, num_steps=2).valid_t_schedule, "all")


class GuideSelectionInterfaceTests(unittest.TestCase):
    def test_script_calls_this_repo_on_the_validation_split(self):
        spec = importlib.util.spec_from_file_location("select_guide_w", "scripts/select_guide_w.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        captured = {}

        def fake_call(command):
            captured["command"] = list(command)
            return 0

        module.subprocess.call = fake_call
        argv = [
            "select_guide_w.py",
            "--config", "xattn/agriculture_36_12_xattn.yaml",
            "--modelfolder", "agriculture_36_12_xattn_seed2025",
            "--guide_list", "0,1",
            "--", "--seq_len", "36",
        ]
        previous = sys.argv
        sys.argv = argv
        try:
            with self.assertRaises(SystemExit) as ctx:
                module.main()
        finally:
            sys.argv = previous
        self.assertEqual(ctx.exception.code, 0)
        command = captured["command"]
        self.assertIn("exe_forecasting.py", command[2])
        self.assertEqual(command[command.index("--guide_w") + 1], "-1")
        self.assertEqual(command[command.index("--guide_list") + 1], "0,1")
        self.assertIn("agriculture_36_12_xattn_seed2025", command)
        self.assertIn("--seq_len", command)
        self.assertNotIn("mcd", " ".join(command).lower())
        self.assertNotIn("MCD-TSF", Path("scripts/select_guide_w.py").read_text().split("Example:")[1])


class LegacySwitchTests(unittest.TestCase):
    def test_mainline_yaml_omits_the_seven_correction_switches(self):
        with open("config/agriculture_36_12_mainline.yaml") as handle:
            config = yaml.safe_load(handle)
        blob = yaml.safe_dump(config)
        for name in LEGACY_SWITCHES:
            self.assertNotIn(name, blob)

    def test_sampler_fix_changes_the_legacy_ddim_index(self):
        def build(enabled):
            config = _legacy_model()
            config["diffusion"].update({
                "ddim": True,
                "sample_steps": 4,
                "sample_method": "linear",
                "num_steps": 20,
                "sampler_fix": enabled,
            })
            model = CSDI_Forecasting(config, torch.device("cpu"), 2, [8, 4])
            self.assertTrue(model.force_legacy_entrypoint if hasattr(model, "force_legacy_entrypoint") else True)
            return model

        fixed = build(True)
        self.assertTrue(fixed.sampler_fix)
        steps, _ = fixed._build_ddim_schedule()
        self.assertEqual(fixed._denoiser_step_index(3, steps), int(steps[3]))
        legacy = build(False)
        self.assertFalse(legacy.sampler_fix)
        self.assertEqual(legacy._denoiser_step_index(3, steps), 3)
        self.assertNotEqual(int(steps[3]), 3)

    def test_text_encoder_eval_holds_on_the_legacy_entrypoint(self):
        enabled = _legacy_model(text_encoder_eval=True)
        disabled = _legacy_model(text_encoder_eval=False)
        self.assertTrue(enabled["model"]["force_legacy_entrypoint"])
        self.assertTrue(enabled["model"]["text_encoder_eval"])
        self.assertFalse(disabled["model"]["text_encoder_eval"])
        on = CSDI_Forecasting(enabled, torch.device("cpu"), 2, [8, 4])
        off = CSDI_Forecasting(disabled, torch.device("cpu"), 2, [8, 4])
        on.text_encoder = torch.nn.Dropout(0.5)
        off.text_encoder = torch.nn.Dropout(0.5)
        on.train()
        off.train()
        self.assertFalse(on.text_encoder.training)
        self.assertTrue(off.text_encoder.training)

    def test_placeholder_and_window_norm_switches_change_legacy_values(self):
        config = _legacy_model(text_placeholder_absent=True)
        config["diffusion"]["estimable_window_norm"] = True
        self.assertTrue(config["model"]["force_legacy_entrypoint"])
        model = CSDI_Forecasting(config, torch.device("cpu"), 2, [8, 4])
        self.assertTrue(model.estimable_window_norm)
        placeholder = "NA (No relevant information found)"
        self.assertEqual(float(model._mark_from_text_batch([placeholder], 1, torch.device("cpu"))), 0.0)
        model.text_placeholder_absent = False
        self.assertEqual(float(model._mark_from_text_batch([placeholder], 1, torch.device("cpu"))), 1.0)
        observed = torch.tensor([[[5.0, 0.0, 0.0, 9.0]]])
        mask = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        _, _, stdev = model._standardize_observed(observed, mask)
        self.assertTrue(torch.allclose(stdev, torch.ones_like(stdev)))
        model.estimable_window_norm = False
        _, _, legacy_std = model._standardize_observed(observed, mask)
        self.assertAlmostEqual(float(legacy_std), math.sqrt(1e-5), places=6)

    def test_deterministic_valid_and_main_loss_change_the_legacy_objective(self):
        config = _legacy_model(deterministic_valid=True, valid_loss_main_only=True)
        config["train"]["multi_res_loss_weight"] = 1.0
        model = CSDI_Forecasting(config, torch.device("cpu"), 2, [8, 4])
        model.multi_res_horizons = [1]
        model.c_mask_prob = 1.0
        model.eval()
        batch = make_batch(model, batch_size=1)
        masks = []

        def _run(total_input, side_info, t, cfg_mask, **kwargs):
            masks.append(cfg_mask.detach().clone())
            noisy = total_input[:, 1]
            return noisy, None

        model._run_diffusion_model = _run
        model.deterministic_valid = True
        first = model(batch, is_train=0)
        second = model(batch, is_train=0)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(all(torch.equal(mask, torch.ones_like(mask)) for mask in masks))
        model.deterministic_valid = False
        masks.clear()
        third = model(batch, is_train=0)
        fourth = model(batch, is_train=0)
        self.assertFalse(torch.equal(third, fourth))
        model.deterministic_valid = True
        model.valid_loss_main_only = False
        with_aux = model(batch, is_train=0)
        model.valid_loss_main_only = True
        main_only = model(batch, is_train=0)
        self.assertFalse(torch.allclose(with_aux, main_only))

    def test_idf_cutoff_changes_the_legacy_vocabulary(self):
        frame = pd.DataFrame({
            "fact": ["apple banana", "apple citrus", "apple zebra"],
            "start_date": pd.to_datetime(["2020-01-01", "2020-02-01", "2021-06-01"]),
            "end_date": pd.to_datetime(["2020-01-31", "2020-02-28", "2021-06-30"]),
        })
        config = RAGCoTConfig(use_retrieval=True, top_k=1, cot_model=None)
        fitted = RAGCoTPipeline("synthetic", frame, "desc", 4, 2, config, idf_fit_end_date=pd.Timestamp("2020-03-01"))
        leaked = RAGCoTPipeline("synthetic", frame, "desc", 4, 2, config, idf_fit_end_date=None)
        self.assertNotIn("zebra", fitted.retriever["vectorizer"].vocabulary_)
        self.assertIn("zebra", leaked.retriever["vectorizer"].vocabulary_)
        held = apply_entrypoint_overrides(_legacy_model(idf_fit_train_only=True))
        self.assertTrue(held["model"]["idf_fit_train_only"])
        self.assertTrue(held["model"]["force_legacy_entrypoint"])


class QueryHistoryTests(unittest.TestCase):
    def test_tfidf_query_uses_the_unaugmented_history_and_drops_na(self):
        self.assertEqual(strip_na_tokens("wheat NA prices NA"), "wheat prices")
        with tempfile.TemporaryDirectory() as root:
            _write_toy(root)
            dataset = _dataset(
                root,
                use_rag_cot=True,
                rag_topk=1,
                rag_query_unaugmented=True,
                ext_text_mode="tfidf",
                aug_noise_std=5.0,
            )
            self.assertTrue(dataset.rag_cot.config.query_drop_na)
            captured = {}

            def wrapped(numeric_history, start_date, end_date, base_text, target_index=None):
                captured["history"] = np.array(numeric_history, copy=True)
                captured["query"] = dataset.rag_cot._build_query("level NA units", "NA wheat")
                return {
                    "composed_text": base_text,
                    "cot_text": "",
                    "retrieved_text": "maize prices rose",
                    "retrieved_records": [],
                }

            dataset.rag_cot.build_guidance_text = wrapped
            np.random.seed(0)
            dataset[0]
            original = dataset.data_x[0:dataset.seq_len]
            self.assertTrue(np.allclose(captured["history"], original))
            self.assertNotIn(" NA ", " " + captured["query"] + " ")
            self.assertNotIn("NA", captured["query"].split())
            aligned = _dataset(
                root,
                use_rag_cot=True,
                rag_topk=1,
                rag_query_unaugmented=True,
                ext_text_mode="time_aligned",
                text_max_length=8,
            )
            self.assertFalse(aligned.rag_cot.config.query_drop_na)


class DdimEndpointTests(unittest.TestCase):
    def test_last_ddim_step_returns_the_x0_prediction(self):
        from tests.test_review_fixes import _StepRecorder

        torch.manual_seed(0)
        model = _sampler_model(
            num_steps=20, ddim=True, sample_steps=4, sample_method="linear",
            sampler_fix=True, noise_esti=False, ddim_x0_endpoint=True,
        )
        model.diffmodel = _StepRecorder(value=4.0)
        observed, cond_mask, samples = _run_impute(model)
        restored = _invert_revin(observed, cond_mask, samples)
        self.assertTrue(torch.allclose(restored, torch.full_like(restored, 4.0)))
        model.ddim_x0_endpoint = False
        model.diffmodel = _StepRecorder(value=4.0)
        _, _, other = _run_impute(model)
        other_restored = _invert_revin(observed, cond_mask, other)
        self.assertFalse(torch.allclose(other_restored, torch.full_like(other_restored, 4.0), atol=1e-4))

    def test_unique_steps_drop_repeated_tau_and_log_the_count(self):
        model = _sampler_model(
            num_steps=300, beta_start=0.0001, beta_end=0.6, schedule="quad",
            ddim=True, sample_steps=20, sample_method="quad", sampler_fix=True,
        )
        duplicated, _ = model._build_ddim_schedule()
        self.assertEqual(len(duplicated), 20)
        self.assertGreater(len(duplicated), len(set(int(value) for value in duplicated)))
        model.ddim_unique_steps = True
        model._ddim_steps_logged = False
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            unique, previous = model._build_ddim_schedule()
        self.assertEqual(len(unique), len(set(int(value) for value in unique)))
        self.assertLess(len(unique), 20)
        self.assertIn("effective steps %d (requested 20)" % len(unique), buffer.getvalue())
        self.assertEqual(int(previous[0]), 0)
        self.assertEqual(list(unique), list(dict.fromkeys(int(value) for value in duplicated)))


class XattnYamlTests(unittest.TestCase):
    def test_xattn_configs_enable_the_protocol_and_legacy_configs_omit_it(self):
        required = {
            "ext_text_mode": "time_aligned",
            "ext_future_year_filter": True,
            "raw_placeholder_filter": True,
            "text_sources": "auto_from_train",
            "gate_beta": "auto_identifiable",
            "text_shift_mode": "none",
            "gate_constant_source": "last_epoch",
            "gate_ref_shared_dropout": True,
            "rag_query_unaugmented": True,
            "log_control_calls": True,
        }
        train_required = {
            "eval_reseed": True,
            "save_window_metrics": True,
            "point_metric_pairing": True,
            "valid_t_schedule": "all",
            "named_output_dir": True,
        }
        xattn_files = list(Path("config/xattn").glob("*.yaml")) + list(Path("config/xattn/ablation").glob("*.yaml"))
        self.assertGreaterEqual(len(xattn_files), 8)
        for path in xattn_files:
            if path.name == "protocol_seeds.yaml":
                continue
            with path.open() as handle:
                config = yaml.safe_load(handle)
            for key, value in required.items():
                self.assertEqual(config["model"][key], value, msg="%s %s" % (path, key))
            for key, value in train_required.items():
                self.assertEqual(config["train"][key], value, msg="%s %s" % (path, key))
            self.assertEqual(config["diffusion"]["time_attn_dropout"], 0.1)
            self.assertTrue(config["diffusion"]["ddim_x0_endpoint"])
            self.assertTrue(config["diffusion"]["ddim_unique_steps"])
        with open("config/agriculture_36_12_mainline.yaml") as handle:
            legacy = yaml.safe_load(handle)
        blob = yaml.safe_dump(legacy)
        for key in (
            "ext_text_mode", "ddim_unique_steps", "ddim_x0_endpoint", "gate_ref_shared_dropout",
            "point_metric_pairing", "eval_reseed", "raw_placeholder_filter",
        ):
            self.assertNotIn(key, blob)


class YearHelperTests(unittest.TestCase):
    def test_four_digit_token_after_the_row_year_is_removed(self):
        frame = pd.DataFrame({
            "start_date": pd.to_datetime(["2000-01-01", "2000-02-01"]),
            "end_date": pd.to_datetime(["2000-01-31", "2000-02-28"]),
            "fact": ["prices in 1999", "prices in 2025"],
        })
        filtered, removed = drop_future_year_rows(frame, "search")
        self.assertEqual(list(filtered["fact"]), ["prices in 1999"])
        self.assertEqual(len(removed), 1)


if __name__ == "__main__":
    unittest.main()
