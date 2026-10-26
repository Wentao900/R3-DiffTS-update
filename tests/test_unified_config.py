"""The eight datasets share config/xattn/unified.yaml."""

import os
import subprocess
import unittest
from pathlib import Path

import torch
import yaml

from main_model import CSDI_Forecasting
from tests.synthetic_factory import xattn_config
from utils.xattn_config import (
    load_protocol_seeds,
    load_unified_config,
    plan_unified_runs,
    resolve_unified_config,
    unified_output_dir,
)


DOMAINS = (
    "agriculture",
    "climate",
    "economy",
    "energy",
    "environment",
    "health",
    "socialgood",
    "traffic",
)
SHARED_KEYS = (
    ("diffusion", "layers", 4),
    ("diffusion", "channels", 64),
    ("diffusion", "nheads", 8),
    ("diffusion", "num_steps", 50),
    ("diffusion", "beta_end", 0.5),
    ("diffusion", "schedule", "quad"),
    ("diffusion", "ddim", False),
    ("diffusion", "sample_steps", 50),
    ("diffusion", "c_mask_prob", 0.1),
    ("train", "epochs", 100),
    ("train", "batch_size", 8),
    ("train", "lr", 0.001),
    ("train", "n_samples", 15),
    ("train", "pattern_expert_weight", 0.05),
    ("train", "multi_res_loss_weight", 0.1),
    ("train", "multi_res_huber_delta", 1.0),
    ("model", "text_max_length", 512),
    ("model", "text_len", 36),
    ("model", "decomp", False),
)


class UnifiedConfigTests(unittest.TestCase):
    def test_eight_datasets_resolve_and_share_the_model(self):
        payload = load_unified_config()
        self.assertEqual(list(payload["domains"]), list(DOMAINS))
        resolved_models = []
        expected = {
            "agriculture": ("Agriculture/Agriculture.csv", "m", 36, 12, 12, 3),
            "climate": ("Climate/Climate.csv", "w", 96, 12, 52, 6),
            "economy": ("Economy/Economy.csv", "m", 36, 12, 12, 3),
            "energy": ("Energy/Energy.csv", "w", 96, 12, 52, 8),
            "environment": ("Environment/Environment.csv", "d", 336, 48, 7, 2),
            "health": ("Health_US/Health_US.csv", "w", 96, 12, 52, 7),
            "socialgood": ("SocialGood/SocialGood.csv", "m", 36, 12, 12, 1),
            "traffic": ("Traffic/Traffic.csv", "m", 36, 12, 12, 1),
        }
        for domain in DOMAINS:
            resolved, runtime = resolve_unified_config(payload, domain, "A1")
            self.assertNotIn("domains", resolved)
            self.assertNotIn("protocol_notes", resolved)
            self.assertNotIn("documented_numeric_columns", resolved["model"])
            self.assertTrue(resolved["dataset"]["use_all_numeric_features"])
            self.assertIsNone(resolved["train"]["multi_res_horizons"])
            self.assertFalse(resolved["train"]["multi_res_use_stat_horizons"])
            self.assertEqual(runtime["target"], "OT")
            path, freq, seq_len, pred_len, season, columns = expected[domain]
            self.assertEqual(runtime["data_path"], path)
            self.assertEqual(runtime["freq"], freq)
            self.assertEqual(runtime["seq_len"], seq_len)
            self.assertEqual(runtime["pred_len"], pred_len)
            self.assertEqual(runtime["season_period"], season)
            self.assertEqual(resolved["model"]["season_period"], season)
            self.assertEqual(runtime["documented_numeric_columns"], columns)
            self.assertEqual(resolved["model"]["cot_cache_dir"], ".rag_cot_cache/%s" % domain)
            for section, key, value in SHARED_KEYS:
                self.assertEqual(resolved[section][key], value, msg="%s %s" % (domain, key))
            resolved_models.append(resolved["diffusion"]["layers"])
        self.assertEqual(set(resolved_models), {4})
        health, health_runtime = resolve_unified_config(payload, "Health_US", "A1")
        self.assertEqual(health_runtime["domain"], "health")
        self.assertEqual(health["run"]["variant"], "A1")

    def test_overlays_cover_the_controls_on_every_domain(self):
        payload = load_unified_config()
        for domain in DOMAINS:
            t0, _ = resolve_unified_config(payload, domain, "T0_numeric_only")
            self.assertEqual(t0["model"]["text_drop_prob"], 1.0)
            self.assertEqual(t0["diffusion"]["layers"], 4)
            e6, _ = resolve_unified_config(payload, domain, "E6_source_only")
            self.assertEqual(e6["model"]["text_gate_mode"], "source_only")
            e4, _ = resolve_unified_config(payload, domain, "E4")
            self.assertEqual(e4["model"]["text_gate_mode"], "one")
            e7, _ = resolve_unified_config(payload, domain, "E7_cyclic_shift")
            self.assertEqual(e7["model"]["text_shift_mode"], "cyclic_per_source")
            self.assertEqual(e7["model"]["text_shift_splits"], ["train", "valid", "test"])
            e8, _ = resolve_unified_config(payload, domain, "E8_ext_only")
            self.assertEqual(e8["model"]["mask_sources"], ["raw"])
            e3, runtime = resolve_unified_config(payload, domain, "E3_gate_constant")
            self.assertEqual(e3["model"]["text_gate_mode"], "learned")
            self.assertEqual(e3["run"]["checkpoint_variant"], "A1")
            self.assertEqual(e3["run"]["infer_override"], "text_gate_mode=constant")
            self.assertNotIn("gate_constant=", e3["run"]["infer_override"])
            self.assertEqual(runtime["domain"], domain)

    def test_old_domain_files_remain(self):
        with open("config/xattn/agriculture_36_12_xattn.yaml") as handle:
            agriculture = yaml.safe_load(handle)
        self.assertEqual(agriculture["diffusion"]["num_steps"], 50)
        self.assertEqual(agriculture["model"]["text_max_length"], 192)
        self.assertTrue(agriculture["model"]["decomp"])
        with open("config/xattn/traffic_36_12_xattn.yaml") as handle:
            traffic = yaml.safe_load(handle)
        self.assertEqual(traffic["diffusion"]["sample_steps"], 120)
        self.assertNotIn("use_all_numeric_features", traffic.get("dataset", {}))
        legacy = {"model": {"force_legacy_entrypoint": True}, "train": {}, "diffusion": {}}
        returned, runtime = resolve_unified_config(legacy, domain="")
        self.assertIs(returned, legacy)
        self.assertIsNone(runtime)

    def test_output_dir_and_default_command_count(self):
        self.assertEqual(
            unified_output_dir("traffic", "E8_ext_only", 2024),
            "./save/traffic_E8_ext_only_seed2024/",
        )
        self.assertEqual(load_protocol_seeds(), [2024, 2025, 2026])
        runs = plan_unified_runs(domain="all", ablation="A1", seeds=None)
        self.assertEqual(len(runs), 24)
        self.assertTrue(all(argv[argv.index("--config") + 1] == "xattn/unified.yaml" for argv in runs))
        health = plan_unified_runs(domain="health", ablation="E3", seeds=[2024])
        self.assertEqual(len(health), 5)
        joined = [" ".join(argv) for argv in health]
        self.assertTrue(any("E3_w0" in line and "guide_w=0" in line and "health_A1_seed2024" in line for line in joined))
        self.assertTrue(all("--domain health" in line for line in joined))
        env = os.environ.copy()
        env.pop("SEEDS", None)
        env.update({"DRY_RUN": "1", "DOMAIN": "all", "ABLATION": "T0_numeric_only"})
        output = subprocess.check_output(["bash", "scripts/run_xattn_ablation.sh"], env=env, text=True)
        commands = [line for line in output.splitlines() if "exe_forecasting.py" in line]
        self.assertEqual(len(commands), 24)
        self.assertTrue(all("T0_numeric_only" in line for line in commands))

    def test_training_mask_drops_only_the_named_source(self):
        config = xattn_config(layers=1, num_steps=2)
        plain = CSDI_Forecasting(config, torch.device("cpu"), 2, [12, 4])
        self.assertEqual(plain.mask_sources, [])
        config["model"]["mask_sources"] = ["raw"]
        masked = CSDI_Forecasting(config, torch.device("cpu"), 2, [12, 4])
        raw = torch.ones(2)
        retrieved = torch.ones(2)
        cot = torch.zeros(2)
        out_raw, out_retrieved, out_cot = masked._apply_source_masks(raw, retrieved, cot)
        self.assertEqual(float(out_raw.sum()), 0.0)
        self.assertEqual(float(out_retrieved.sum()), 2.0)
        self.assertEqual(float(out_cot.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
