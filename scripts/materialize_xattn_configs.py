"""Write the gated cross-attention configs and the minimal ablation set.

A1 is produced for every mainline domain. Single-factor ablations are written
for Agriculture (decomposition) and Traffic (no decomposition).
"""

import math
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAINLINE = {
    "agriculture": ("agriculture_36_12_mainline.yaml", 12, 12),
    "economy": ("economy_36_12_mainline.yaml", 12, 12),
    "socialgood": ("socialgood_36_12_mainline.yaml", 12, 12),
    "traffic": ("traffic_36_12_mainline.yaml", 12, 12),
    "climate": ("climate_96_12_mainline.yaml", 52, 12),
    "energy": ("energy_96_12_mainline.yaml", 52, 12),
    "health": ("health_96_12_mainline.yaml", 52, 12),
    "environment": ("environment_336_48_mainline.yaml", 7, 48),
}


def fixed_horizons(pred_len):
    values = [1, int(math.ceil(pred_len / 4.0)), int(math.ceil(pred_len / 2.0)), int(pred_len)]
    horizons = []
    for value in values:
        if value not in horizons and 1 <= value <= pred_len:
            horizons.append(value)
    return horizons


def apply_a1(config, season_period, pred_len):
    config = deepcopy(config)
    train = config.setdefault("train", {})
    train.update({
        "multi_res_horizons": fixed_horizons(pred_len),
        "multi_res_use_stat_horizons": False,
        "multi_res_dynamic": False,
        "multi_res_dynamic_by_t": False,
        "multi_res_dynamic_by_epoch": False,
        "multi_res_dynamic_by_trend": False,
        "multi_res_progressive": False,
        "multi_res_difficulty_weight": 0.0,
        "multi_res_huber_delta": 1.0,
        "multi_res_huber_deltas": None,
        "multi_res_huber_delta_mode": "fallback_uniform",
        "multi_res_segment_loss": False,
        "auxiliary_loss_max_ratio": 0.0,
        "pattern_consistency_weight": 0.0,
        "pattern_expert_weight": 0.05,
        "coarse_forecast_weight": 0.0,
        "uncertainty_forecast_weight": 0.0,
        "valid_loss_main_only": True,
        "forecast_calibrator": False,
    })
    diffusion = config.setdefault("diffusion", {})
    diffusion["mask_noisy_target"] = True
    diffusion["sampler_fix"] = True
    # Same numeric-tower dropout for A1, T0, and T1. 0 keeps the text and null
    # passes identical. Legacy configs omit the key and stay at 0.1.
    diffusion["tower_dropout"] = 0.0
    model = config.setdefault("model", {})
    model.update({
        "force_legacy_entrypoint": False,
        "timestep_branch": False,
        "timestep_emb_cat": True,
        "timestep_outer_mix": "none",
        "with_texts": True,
        "text_injection": "gated",
        "pattern_expert_set": "fpp4",
        "pattern_router_mode": "linear_history",
        "baseline_form": "fpp4",
        "season_period": int(season_period),
        "pattern_text_evidence": False,
        "pattern_reliability": False,
        "pattern_aux_reliability": False,
        "use_forecast_policy_controller": False,
        "use_aux_forecast_heads": False,
        "use_coarse_forecast_head": False,
        "use_uncertainty_head": False,
        "coarse_forecast_blend": 0.0,
        "guide_mode": "manual",
        "final_method": False,
        "guide_w_default": 1.0,
        "text_validity_gate": True,
        "text_gate_mode": "learned",
        "text_validity_loss": True,
        "cfg_output_combine": True,
        "text_features_mode": "none",
        "skip_cot_generation": True,
        "numeric_summary_target_only": True,
        "cot_as_source": False,
        "n_text_sources": 2,
        "use_two_stage_rag": False,
        "use_rag_cot": True,
        "shuffle_text_in_batch": False,
        "text_drop_prob": 0.0,
        "idf_fit_train_only": True,
        "text_encoder_eval": True,
        "text_placeholder_absent": True,
    })
    config.setdefault("dataset", {})["text_features_mode"] = "none"
    return config


def ablation_patches():
    return {
        "T0_numeric_only": {"model": {"text_drop_prob": 1.0}},
        "T1_router_text": {
            "model": {
                "text_injection": "off",
                "pattern_router_mode": "linear_history_text",
                "text_validity_gate": False,
                "text_validity_loss": False,
                "cfg_output_combine": False,
            }
        },
        "T2_gate_one": {"model": {"text_gate_mode": "one", "text_validity_loss": False}},
        "T3_gate_constant": {"model": {"text_gate_mode": "constant", "text_validity_loss": False}},
        "T4a_shuffle_text": {"model": {"shuffle_text_in_batch": True}},
        "S1_raw_only": {"model": {"use_rag_cot": False}},
        "S4_cot_source": {
            "model": {
                "cot_as_source": True,
                "n_text_sources": 3,
                "skip_cot_generation": False,
            }
        },
        "N1_naive": {"model": {"baseline_form": "naive"}},
        "N1_dlinear": {"model": {"baseline_form": "dlinear"}},
        "N1_none": {"model": {"baseline_form": "none", "pattern_residual_diffusion": False}},
    }


def merge(config, patch):
    merged = deepcopy(config)
    for section, values in patch.items():
        merged.setdefault(section, {}).update(values)
    return merged


def dump(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Gated cross-attention config generated by scripts/materialize_xattn_configs.py\n"
        "# A1 is the main method. Ablation files change one factor relative to A1.\n"
    )
    body = yaml.safe_dump(config, sort_keys=False)
    path.write_text(header + body)


def main():
    out_dir = ROOT / "config" / "xattn"
    a1 = {}
    for name, (filename, season, pred_len) in MAINLINE.items():
        with open(ROOT / "config" / filename) as handle:
            base = yaml.safe_load(handle)
        config = apply_a1(base, season, pred_len)
        stem = filename.replace("_mainline.yaml", "_xattn.yaml")
        dump(out_dir / stem, config)
        a1[name] = config
    patches = ablation_patches()
    for domain in ("agriculture", "traffic"):
        for ablation, patch in patches.items():
            filename = f"{domain}_{ablation}.yaml"
            dump(out_dir / "ablation" / filename, merge(a1[domain], patch))
    print(f"wrote configs under {out_dir}")


if __name__ == "__main__":
    main()
