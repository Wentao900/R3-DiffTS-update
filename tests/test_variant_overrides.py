"""Variants are overrides on each domain YAML, not a shared config."""

import os
import subprocess
import unittest
from pathlib import Path

import torch
import yaml

from main_model import CSDI_Forecasting
from tests.synthetic_factory import xattn_config
from utils.xattn_config import (
    ALL_VARIANTS,
    DOMAIN_CONFIGS,
    DOMAIN_ORDER,
    YAML_BACKED_VARIANTS,
    apply_variant,
    load_protocol_seeds,
    plan_variant_runs,
    variant_output_dir,
)


PRESERVED = (
    ("diffusion", "layers"),
    ("diffusion", "channels"),
    ("diffusion", "nheads"),
    ("diffusion", "beta_end"),
    ("diffusion", "num_steps"),
    ("diffusion", "sample_steps"),
    ("diffusion", "c_mask_prob"),
    ("diffusion", "ddim"),
    ("train", "epochs"),
    ("train", "batch_size"),
    ("train", "lr"),
    ("train", "lr_warmup_epochs"),
    ("train", "max_grad_norm"),
    ("model", "text_max_length"),
    ("model", "decomp"),
    ("model", "season_period"),
    ("model", "cot_cache_dir"),
)


def _load(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def _without_run(config):
    copied = dict(config)
    copied.pop("run", None)
    return copied


class VariantOverrideTests(unittest.TestCase):
    def test_eight_domains_times_every_variant_keep_domain_values(self):
        self.assertEqual(len(DOMAIN_ORDER), 8)
        self.assertFalse(Path("config/xattn/unified.yaml").exists())
        for domain in DOMAIN_ORDER:
            base = _load("config/" + DOMAIN_CONFIGS[domain])
            self.assertNotIn("domains", base)
            for variant in ALL_VARIANTS:
                resolved = apply_variant(base, variant, domain=domain)
                self.assertEqual(resolved["run"]["domain"], domain)
                self.assertEqual(resolved["run"]["variant"], variant)
                for section, key in PRESERVED:
                    if key == "pattern_expert_weight":
                        continue
                    if variant == "N2_no_expert_loss" and key == "pattern_expert_weight":
                        continue
                    self.assertEqual(
                        resolved[section][key],
                        base[section][key],
                        msg="%s %s %s" % (domain, variant, key),
                    )
                if "use_all_numeric_features" not in base.get("dataset", {}):
                    self.assertNotIn("use_all_numeric_features", resolved.get("dataset", {}))
                else:
                    self.assertEqual(
                        resolved["dataset"]["use_all_numeric_features"],
                        base["dataset"]["use_all_numeric_features"],
                    )

    def test_overrides_match_agriculture_and_traffic_ablation_yamls(self):
        for domain in ("agriculture", "traffic"):
            base = _load("config/" + DOMAIN_CONFIGS[domain])
            for variant in YAML_BACKED_VARIANTS:
                resolved = _without_run(apply_variant(base, variant))
                expected = _load("config/xattn/ablation/%s_%s.yaml" % (domain, variant))
                self.assertEqual(resolved, expected, msg="%s %s" % (domain, variant))
            a1 = _without_run(apply_variant(base, "A1", domain=domain))
            self.assertEqual(a1, base)

    def test_control_overrides_and_aliases(self):
        base = _load("config/" + DOMAIN_CONFIGS["economy"])
        t0 = apply_variant(base, "T0_numeric_only", domain="economy")
        self.assertEqual(t0["model"]["text_drop_prob"], 1.0)
        self.assertEqual(t0["diffusion"]["layers"], base["diffusion"]["layers"])
        e6 = apply_variant(base, "E6_source_only")
        self.assertEqual(e6["model"]["text_gate_mode"], "source_only")
        e4 = apply_variant(base, "E4", domain="economy")
        self.assertEqual(e4["run"]["variant"], "T2_gate_one")
        self.assertEqual(e4["model"]["text_gate_mode"], "one")
        self.assertFalse(e4["model"]["text_validity_loss"])
        e5 = apply_variant(base, "E5")
        self.assertEqual(e5["model"]["text_gate_mode"], "constant")
        e7 = apply_variant(base, "E7_cyclic_shift")
        self.assertEqual(e7["model"]["text_shift_mode"], "cyclic_per_source")
        self.assertEqual(e7["model"]["text_shift_splits"], ["train", "valid", "test"])
        e8 = apply_variant(base, "E8_ext_only")
        self.assertEqual(e8["model"]["mask_sources"], ["raw"])
        e3 = apply_variant(base, "E3_gate_constant", domain="Health_US")
        self.assertEqual(e3["run"]["domain"], "health")
        self.assertEqual(e3["model"]["text_gate_mode"], base["model"]["text_gate_mode"])
        self.assertEqual(e3["run"]["checkpoint_variant"], "A1")
        self.assertEqual(e3["run"]["infer_override"], "text_gate_mode=constant")
        self.assertNotIn("gate_constant=", e3["run"]["infer_override"])
        s1 = apply_variant(base, "S1_raw_only")
        self.assertFalse(s1["model"]["use_rag_cot"])
        s4 = apply_variant(base, "S4_cot_source")
        self.assertTrue(s4["model"]["cot_as_source"])
        self.assertEqual(s4["model"]["n_text_sources"], 3)
        self.assertFalse(s4["model"]["skip_cot_generation"])

    def test_old_domain_files_remain(self):
        agriculture = _load("config/xattn/agriculture_36_12_xattn.yaml")
        self.assertEqual(agriculture["diffusion"]["num_steps"], 50)
        self.assertEqual(agriculture["model"]["text_max_length"], 192)
        self.assertTrue(agriculture["model"]["decomp"])
        traffic = _load("config/xattn/traffic_36_12_xattn.yaml")
        self.assertEqual(traffic["diffusion"]["sample_steps"], 120)
        self.assertNotIn("use_all_numeric_features", traffic.get("dataset", {}))
        legacy = {"model": {"force_legacy_entrypoint": True}, "train": {}, "diffusion": {}}
        returned = apply_variant(legacy, variant="A1", domain="")
        self.assertIs(returned, legacy)

    def test_output_dir_and_commands_use_the_domain_yaml(self):
        self.assertEqual(
            variant_output_dir("traffic", "E8_ext_only", 2024),
            "./save/traffic_E8_ext_only_seed2024/",
        )
        self.assertEqual(load_protocol_seeds(), [2024, 2025, 2026])
        runs = plan_variant_runs(domain="all", ablation="A1", seeds=None)
        self.assertEqual(len(runs), 24)
        configs = {argv[argv.index("--config") + 1] for argv in runs}
        self.assertEqual(configs, set(DOMAIN_CONFIGS.values()))
        self.assertTrue(all("unified.yaml" not in argv[argv.index("--config") + 1] for argv in runs))
        self.assertTrue(all("ablation/" not in argv[argv.index("--config") + 1] for argv in runs))
        health = plan_variant_runs(domain="health", ablation="E3", seeds=[2024])
        self.assertEqual(len(health), 5)
        joined = [" ".join(argv) for argv in health]
        self.assertTrue(all("xattn/health_96_12_xattn.yaml" in line for line in joined))
        self.assertTrue(any("E3_w0" in line and "guide_w=0" in line and "health_A1_seed2024" in line for line in joined))
        self.assertTrue(all("--domain health" in line for line in joined))
        environment = plan_variant_runs(domain="environment", ablation="T0_numeric_only", seeds=[2026])
        text = " ".join(environment[0])
        self.assertIn("--seq_len 336", text)
        self.assertIn("--pred_len 48", text)
        self.assertIn("--freq d", text)
        self.assertIn("xattn/environment_336_48_xattn.yaml", text)
        every = plan_variant_runs(domain="climate", ablation="all", seeds=[2024])
        self.assertEqual(len(every), len(ALL_VARIANTS))
        self.assertEqual(
            {argv[argv.index("--variant") + 1] for argv in every},
            set(ALL_VARIANTS),
        )
        env = os.environ.copy()
        env.pop("SEEDS", None)
        env.update({"DRY_RUN": "1", "DOMAIN": "all", "ABLATION": "T0_numeric_only"})
        output = subprocess.check_output(["bash", "scripts/run_xattn_ablation.sh"], env=env, text=True)
        commands = [line for line in output.splitlines() if "exe_forecasting.py" in line]
        self.assertEqual(len(commands), 24)
        self.assertTrue(all("T0_numeric_only" in line for line in commands))
        self.assertTrue(all("unified.yaml" not in line and "ablation/" not in line for line in commands))

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
