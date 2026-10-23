"""Instantiate legacy and xattn models and print trainable parameter counts.

Frozen BERT is not loaded. ``llm: hash`` keeps the same context width and adds
no trainable parameters, so the count matches a model whose BERT is frozen.
K is the numeric feature count passed as target_dim. It is not in the YAML;
counts below use K=1 unless --target-dim is set.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from main_model import CSDI_Forecasting
from utils.xattn_config import FREQ_SEASON_PERIOD, apply_entrypoint_overrides
PAIRS = [
    ("Agriculture", "agriculture_36_12_mainline.yaml", "agriculture_36_12_xattn.yaml", 36, 12, "m"),
    ("Economy", "economy_36_12_mainline.yaml", "economy_36_12_xattn.yaml", 36, 12, "m"),
    ("SocialGood", "socialgood_36_12_mainline.yaml", "socialgood_36_12_xattn.yaml", 36, 12, "m"),
    ("Traffic", "traffic_36_12_mainline.yaml", "traffic_36_12_xattn.yaml", 36, 12, "m"),
    ("Climate", "climate_96_12_mainline.yaml", "climate_96_12_xattn.yaml", 96, 12, "w"),
    ("Energy", "energy_96_12_mainline.yaml", "energy_96_12_xattn.yaml", 96, 12, "w"),
    ("Health", "health_96_12_mainline.yaml", "health_96_12_xattn.yaml", 96, 12, "w"),
    ("Environment", "environment_336_48_mainline.yaml", "environment_336_48_xattn.yaml", 336, 48, "d"),
]


def prepare(path, seq_len, pred_len, freq, legacy):
    with open(path) as handle:
        config = yaml.safe_load(handle)
    if legacy:
        config.setdefault("model", {})["force_legacy_entrypoint"] = True
    apply_entrypoint_overrides(config, freq=freq)
    config["model"]["llm"] = "hash"
    config["model"]["context_dim"] = 768
    config["model"]["lookback_len"] = int(seq_len)
    config["model"]["pred_len"] = int(pred_len)
    config["model"]["domain"] = "Agriculture"
    config["model"]["save_attn"] = False
    config["model"]["save_token"] = False
    config["model"]["is_unconditional"] = 0
    config["model"]["llm_layers"] = int(config["model"].get("llm_layers", 0))
    config["diffusion"]["dropout"] = float(config["diffusion"].get("dropout", 0.0))
    config["diffusion"]["attn_drop"] = float(config["diffusion"].get("attn_drop", 0.0))
    config["diffusion"]["time_weight"] = float(config["diffusion"].get("time_weight", 0.1))
    extra = 2
    freq_dim = {"m": 1, "w": 2, "d": 3}.get(freq, 1)
    if config["model"].get("timestep_branch") or config["model"].get("timestep_emb_cat"):
        config["model"]["timestep_dim"] = freq_dim + extra
    else:
        config["model"]["timestep_dim"] = 0
    if "season_period" not in config["model"]:
        config["model"]["season_period"] = FREQ_SEASON_PERIOD.get(freq, 12)
    return config


def trainable_count(model):
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad and "text_encoder" not in name:
            count += param.numel()
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dim", type=int, default=1)
    args = parser.parse_args()
    device = torch.device("cpu")
    print(f"{'domain':<14} {'legacy':>12} {'xattn':>12} {'delta':>12}  K={args.target_dim}")
    for domain, legacy_name, xattn_name, seq_len, pred_len, freq in PAIRS:
        legacy_cfg = prepare(ROOT / "config" / legacy_name, seq_len, pred_len, freq, legacy=True)
        xattn_cfg = prepare(ROOT / "config" / "xattn" / xattn_name, seq_len, pred_len, freq, legacy=False)
        legacy_model = CSDI_Forecasting(legacy_cfg, device, args.target_dim, [seq_len, pred_len])
        xattn_model = CSDI_Forecasting(xattn_cfg, device, args.target_dim, [seq_len, pred_len])
        legacy_n = trainable_count(legacy_model)
        xattn_n = trainable_count(xattn_model)
        print(f"{domain:<14} {legacy_n:12d} {xattn_n:12d} {xattn_n - legacy_n:12d}")
        del legacy_model, xattn_model

    print("ablation trainable counts (K=%d), relative to that domain's A1" % args.target_dim)
    for domain, seq_len, pred_len, freq in (
        ("agriculture", 36, 12, "m"),
        ("traffic", 36, 12, "m"),
    ):
        a1_path = ROOT / "config" / "xattn" / f"{domain}_36_12_xattn.yaml"
        a1_model = CSDI_Forecasting(
            prepare(a1_path, seq_len, pred_len, freq, legacy=False),
            device,
            args.target_dim,
            [seq_len, pred_len],
        )
        a1_n = trainable_count(a1_model)
        del a1_model
        ablation_dir = ROOT / "config" / "xattn" / "ablation"
        for path in sorted(ablation_dir.glob(f"{domain}_*.yaml")):
            model = CSDI_Forecasting(
                prepare(path, seq_len, pred_len, freq, legacy=False),
                device,
                args.target_dim,
                [seq_len, pred_len],
            )
            count = trainable_count(model)
            print(f"{path.stem:<32} {count:12d} {count - a1_n:12d}")
            del model


if __name__ == "__main__":
    main()
