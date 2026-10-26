"""Entrypoint flags for the legacy pipeline and the gated cross-attention method.

Legacy configs keep the historical forced switches. The xattn configs set
``force_legacy_entrypoint: false`` and read the YAML values instead.
"""

from copy import deepcopy
from pathlib import Path

import yaml

FREQ_SEASON_PERIOD = {
    "m": 12,
    "w": 52,
    "d": 7,
    "b": 5,
    "h": 24,
    "t": 96,
    "s": 60,
}


def apply_entrypoint_overrides(config, freq="m"):
    model_cfg = config.setdefault("model", {})
    force_legacy = bool(model_cfg.get("force_legacy_entrypoint", True))
    model_cfg["force_legacy_entrypoint"] = force_legacy
    if force_legacy:
        model_cfg["with_texts"] = True
        model_cfg["timestep_emb_cat"] = True
        model_cfg["timestep_branch"] = True
    else:
        model_cfg["with_texts"] = bool(model_cfg.get("with_texts", True))
        model_cfg["timestep_emb_cat"] = bool(model_cfg.get("timestep_emb_cat", True))
        model_cfg["timestep_branch"] = bool(model_cfg.get("timestep_branch", False))

    model_cfg["text_injection"] = str(model_cfg.get("text_injection", "legacy")).lower()
    model_cfg["timestep_outer_mix"] = str(model_cfg.get("timestep_outer_mix", "legacy")).lower()
    model_cfg["pattern_expert_set"] = str(model_cfg.get("pattern_expert_set", "legacy5")).lower()
    model_cfg["pattern_router_mode"] = str(model_cfg.get("pattern_router_mode", "legacy")).lower()
    model_cfg["baseline_form"] = str(model_cfg.get("baseline_form", "router")).lower()
    model_cfg["text_gate_mode"] = str(model_cfg.get("text_gate_mode", "learned")).lower()
    model_cfg["text_features_mode"] = str(model_cfg.get("text_features_mode", "legacy")).lower()
    model_cfg["n_text_sources"] = int(model_cfg.get("n_text_sources", 3 if model_cfg.get("cot_as_source", False) else 2))
    model_cfg["cot_as_source"] = bool(model_cfg.get("cot_as_source", False))
    model_cfg["text_validity_gate"] = bool(model_cfg.get("text_validity_gate", False))
    model_cfg["text_validity_loss"] = bool(model_cfg.get("text_validity_loss", model_cfg["text_validity_gate"]))
    model_cfg["cfg_output_combine"] = bool(model_cfg.get("cfg_output_combine", False))
    model_cfg["shuffle_text_in_batch"] = bool(model_cfg.get("shuffle_text_in_batch", False))
    model_cfg["skip_cot_generation"] = bool(model_cfg.get("skip_cot_generation", False))
    model_cfg["numeric_summary_target_only"] = bool(model_cfg.get("numeric_summary_target_only", False))
    if "season_period" not in model_cfg or model_cfg.get("season_period") in (None, ""):
        model_cfg["season_period"] = int(FREQ_SEASON_PERIOD.get(str(freq), 12))
    else:
        model_cfg["season_period"] = int(model_cfg["season_period"])
    model_cfg["benefit_eps"] = float(model_cfg.get("benefit_eps", 1e-8))
    model_cfg["idf_fit_train_only"] = bool(model_cfg.get("idf_fit_train_only", False))
    model_cfg["text_encoder_eval"] = bool(model_cfg.get("text_encoder_eval", False))
    model_cfg["text_placeholder_absent"] = bool(model_cfg.get("text_placeholder_absent", False))
    model_cfg["ext_text_mode"] = str(model_cfg.get("ext_text_mode", "tfidf")).lower()
    model_cfg["ext_future_year_filter"] = bool(model_cfg.get("ext_future_year_filter", False))
    model_cfg["raw_placeholder_filter"] = bool(model_cfg.get("raw_placeholder_filter", False))
    model_cfg["text_sources"] = str(model_cfg.get("text_sources", "configured")).lower()
    model_cfg["gate_beta"] = str(model_cfg.get("gate_beta", "all")).lower()
    model_cfg["text_shift_mode"] = str(model_cfg.get("text_shift_mode", "none")).lower()
    model_cfg["gate_constant_source"] = str(model_cfg.get("gate_constant_source", "cumulative")).lower()
    model_cfg["gate_ref_shared_dropout"] = bool(model_cfg.get("gate_ref_shared_dropout", False))
    model_cfg["rag_query_unaugmented"] = bool(model_cfg.get("rag_query_unaugmented", False))
    model_cfg["log_control_calls"] = bool(model_cfg.get("log_control_calls", False))

    diffusion_cfg = config.setdefault("diffusion", {})
    diffusion_cfg["mask_noisy_target"] = bool(diffusion_cfg.get("mask_noisy_target", False))

    train_cfg = config.setdefault("train", {})
    train_cfg["valid_loss_main_only"] = bool(train_cfg.get("valid_loss_main_only", False))
    train_cfg["deterministic_valid"] = bool(train_cfg.get("deterministic_valid", False))
    train_cfg["eval_reseed"] = bool(train_cfg.get("eval_reseed", False))
    train_cfg["save_window_metrics"] = bool(train_cfg.get("save_window_metrics", False))
    train_cfg["point_metric_pairing"] = bool(train_cfg.get("point_metric_pairing", False))
    train_cfg["valid_t_schedule"] = str(train_cfg.get("valid_t_schedule", "all")).lower()
    train_cfg["named_output_dir"] = bool(train_cfg.get("named_output_dir", False))
    if model_cfg["baseline_form"] == "none":
        model_cfg["pattern_residual_diffusion"] = False
    diffusion_cfg["sampler_fix"] = bool(diffusion_cfg.get("sampler_fix", False))
    diffusion_cfg["estimable_window_norm"] = bool(diffusion_cfg.get("estimable_window_norm", False))
    diffusion_cfg["time_attn_dropout"] = float(diffusion_cfg.get("time_attn_dropout", 0.1))
    diffusion_cfg["ddim_x0_endpoint"] = bool(diffusion_cfg.get("ddim_x0_endpoint", False))
    diffusion_cfg["ddim_unique_steps"] = bool(diffusion_cfg.get("ddim_unique_steps", False))
    return config


def variant_output_dir(domain, variant, seed):
    """Directory name carries the dataset, the variant, and the seed."""
    return "./save/%s_%s_seed%s/" % (domain, variant, seed)


def resolve_xattn_guide_w(config, guide_w, guide_list=""):
    """Keep the legacy 20-value sweep. xattn defaults to guide_w_default (1).

    A negative guide_w with an explicit guide_list still sweeps that list.
    A negative guide_w with an empty list sweeps the built-in table only when
    force_legacy_entrypoint is on.
    """
    guide_w = float(guide_w)
    if guide_w >= 0 or str(guide_list or "").strip():
        return guide_w
    model_cfg = config.get("model", {})
    if bool(model_cfg.get("force_legacy_entrypoint", True)):
        return guide_w
    return float(model_cfg.get("guide_w_default", 1.0))


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SEEDS = ROOT / "config" / "xattn" / "protocol_seeds.yaml"

# One existing xattn YAML per dataset. Ablations do not get their own files.
# The command line below is fixed by the Time-MMD file and the MCD-TSF short
# protocol (run.sh): it is not a model hyperparameter.
DOMAIN_CONFIGS = {
    "agriculture": "xattn/agriculture_36_12_xattn.yaml",
    "climate": "xattn/climate_96_12_xattn.yaml",
    "economy": "xattn/economy_36_12_xattn.yaml",
    "energy": "xattn/energy_96_12_xattn.yaml",
    "environment": "xattn/environment_336_48_xattn.yaml",
    "health": "xattn/health_96_12_xattn.yaml",
    "socialgood": "xattn/socialgood_36_12_xattn.yaml",
    "traffic": "xattn/traffic_36_12_xattn.yaml",
}
DOMAIN_CLI = {
    "agriculture": {"data_path": "Agriculture/Agriculture.csv", "seq_len": 36, "pred_len": 12, "freq": "m", "text_len": 36},
    "climate": {"data_path": "Climate/Climate.csv", "seq_len": 96, "pred_len": 12, "freq": "w", "text_len": 36},
    "economy": {"data_path": "Economy/Economy.csv", "seq_len": 36, "pred_len": 12, "freq": "m", "text_len": 36},
    "energy": {"data_path": "Energy/Energy.csv", "seq_len": 96, "pred_len": 12, "freq": "w", "text_len": 36},
    "environment": {"data_path": "Environment/Environment.csv", "seq_len": 336, "pred_len": 48, "freq": "d", "text_len": 36},
    "health": {"data_path": "Health_US/Health_US.csv", "seq_len": 96, "pred_len": 12, "freq": "w", "text_len": 36},
    "socialgood": {"data_path": "SocialGood/SocialGood.csv", "seq_len": 36, "pred_len": 12, "freq": "m", "text_len": 36},
    "traffic": {"data_path": "Traffic/Traffic.csv", "seq_len": 36, "pred_len": 12, "freq": "m", "text_len": 36},
}
DOMAIN_ORDER = (
    "agriculture",
    "climate",
    "economy",
    "energy",
    "environment",
    "health",
    "socialgood",
    "traffic",
)

# Names the shell accepts. E4 and E5 are the plan's names for T2 and T3.
VARIANT_ALIASES = {
    "E4": "T2_gate_one",
    "E4_gate_one": "T2_gate_one",
    "E5": "T3_gate_constant",
    "E5_gate_constant": "T3_gate_constant",
}
DOMAIN_ALIASES = {
    "healthus": "health",
    "health_us": "health",
    "social_good": "socialgood",
}
# Overrides applied on top of that domain's xattn YAML. Keys here are the only
# difference from A1. E3 does not retrain: it loads the A1 checkpoint and sets
# an inference override after load_state_dict.
VARIANT_SPECS = {
    "A1": {},
    "T0_numeric_only": {"model": {"text_drop_prob": 1.0}},
    "T1_router_text": {"model": {
        "text_injection": "off",
        "pattern_router_mode": "linear_history_text",
        "text_validity_gate": False,
        "text_validity_loss": False,
        "cfg_output_combine": False,
    }},
    "T2_gate_one": {"model": {"text_gate_mode": "one", "text_validity_loss": False}},
    "T3_gate_constant": {"model": {"text_gate_mode": "constant", "text_validity_loss": False}},
    "T4a_shuffle_text": {"model": {"shuffle_text_in_batch": True}},
    "S1_raw_only": {"model": {"use_rag_cot": False}},
    "S4_cot_source": {"model": {
        "cot_as_source": True,
        "n_text_sources": 3,
        "skip_cot_generation": False,
    }},
    "E6_source_only": {"model": {"text_gate_mode": "source_only"}},
    "E7_cyclic_shift": {"model": {
        "text_shift_mode": "cyclic_per_source",
        "text_shift_splits": ["train", "valid", "test"],
    }},
    "E8_ext_only": {"model": {"mask_sources": ["raw"]}},
    "N1_naive": {"model": {"baseline_form": "naive"}},
    "N1_dlinear": {"model": {"baseline_form": "dlinear"}},
    "N1_none": {"model": {"baseline_form": "none", "pattern_residual_diffusion": False}},
    "N2_no_expert_loss": {"train": {"pattern_expert_weight": 0.0}},
    "N3_no_multires_loss": {"train": {"multi_res_loss_weight": 0.0}},
    "C1_w1": {},
    "C1_sweep": {},
    "E3_w0": {"checkpoint_variant": "A1", "infer_override": "guide_w=0"},
    "E3_gate_one": {"checkpoint_variant": "A1", "infer_override": "text_gate_mode=one"},
    "E3_gate_constant": {"checkpoint_variant": "A1", "infer_override": "text_gate_mode=constant"},
    "E3_cyclic_shift": {"checkpoint_variant": "A1", "infer_override": "text_shift_mode=cyclic_per_source"},
    "E3_mask_raw": {"checkpoint_variant": "A1", "infer_override": "mask_sources=raw"},
}
RESERVED_VARIANT_KEYS = ("checkpoint_variant", "infer_override")
TRAINING_VARIANTS = (
    "A1",
    "T0_numeric_only",
    "T1_router_text",
    "T2_gate_one",
    "T3_gate_constant",
    "T4a_shuffle_text",
    "S1_raw_only",
    "S4_cot_source",
    "E6_source_only",
    "E7_cyclic_shift",
    "E8_ext_only",
    "N1_naive",
    "N1_dlinear",
    "N1_none",
    "N2_no_expert_loss",
    "N3_no_multires_loss",
    "C1_w1",
    "C1_sweep",
)
E3_VARIANTS = (
    "E3_w0",
    "E3_gate_one",
    "E3_gate_constant",
    "E3_cyclic_shift",
    "E3_mask_raw",
)
ALL_VARIANTS = TRAINING_VARIANTS + E3_VARIANTS
# Agriculture and Traffic still have these files under config/xattn/ablation/.
YAML_BACKED_VARIANTS = (
    "T0_numeric_only",
    "T1_router_text",
    "T2_gate_one",
    "T3_gate_constant",
    "T4a_shuffle_text",
    "S1_raw_only",
    "S4_cot_source",
    "N1_naive",
    "N1_dlinear",
    "N1_none",
)


def _normalize_domain(name):
    key = str(name or "").strip().lower().replace("-", "_")
    compact = key.replace("_", "")
    if compact in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[compact]
    if key in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[key]
    return key


def _deep_merge(base, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def load_protocol_seeds(path=None):
    with open(path or PROTOCOL_SEEDS) as handle:
        payload = yaml.safe_load(handle)
    return [int(seed) for seed in payload["seeds"]]


def canonical_variant(variant):
    name = VARIANT_ALIASES.get(str(variant or "A1"), str(variant or "A1"))
    if name not in VARIANT_SPECS:
        known = ", ".join(ALL_VARIANTS)
        raise ValueError("unknown variant %r. Known variants: %s" % (variant, known))
    return name


def domain_cli(domain):
    key = _normalize_domain(domain)
    if key not in DOMAIN_CLI:
        known = ", ".join(DOMAIN_ORDER)
        raise ValueError("unknown domain %r. Known datasets: %s" % (domain, known))
    return key, dict(DOMAIN_CLI[key])


def apply_variant(config, variant="A1", domain=""):
    """Return the domain config with one variant's overrides merged in.

    A1 with an empty domain returns the same object. Every other call copies
    the config and changes only the keys in ``VARIANT_SPECS``. Dataset
    hyperparameters stay on the values stored in that domain's YAML.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a mapping")
    variant_key = canonical_variant(variant)
    spec = VARIANT_SPECS[variant_key]
    domain_key = _normalize_domain(domain) if str(domain or "").strip() else ""
    if domain_key and domain_key not in DOMAIN_CONFIGS:
        known = ", ".join(DOMAIN_ORDER)
        raise ValueError("unknown domain %r. Known datasets: %s" % (domain, known))
    patch = {key: value for key, value in spec.items() if key not in RESERVED_VARIANT_KEYS}
    needs_run = bool(domain_key) or bool(patch) or bool(spec.get("checkpoint_variant")) or bool(spec.get("infer_override")) or variant_key != "A1"
    if not needs_run:
        return config
    resolved = deepcopy(config)
    if patch:
        _deep_merge(resolved, patch)
    resolved["run"] = {
        "domain": domain_key,
        "variant": variant_key,
        "checkpoint_variant": spec.get("checkpoint_variant"),
        "infer_override": spec.get("infer_override") or "",
    }
    return resolved


def variant_names(ablation):
    name = str(ablation or "A1")
    if name == "all":
        return list(ALL_VARIANTS)
    if name == "E3":
        return list(E3_VARIANTS)
    return [canonical_variant(name)]


def plan_variant_runs(domain="all", ablation="A1", seeds=None, root_path="../Time-MMD-main", device="cpu", guide_list="0,0.5,1.0,1.5,2.0", nsample=None):
    """Argv lists for one variant on the requested datasets and seeds.

    Each command loads that dataset's own xattn YAML. The variant is a
    ``--variant`` override, not a second config file. ``seeds`` left empty
    uses config/xattn/protocol_seeds.yaml.
    """
    if seeds is None or seeds == "" or seeds == []:
        seed_list = load_protocol_seeds()
    elif isinstance(seeds, str):
        seed_list = [int(item) for item in seeds.split(",") if item.strip()]
    else:
        seed_list = [int(item) for item in seeds]
    if str(domain) == "all":
        domains = list(DOMAIN_ORDER)
    else:
        domains = [domain_cli(domain)[0]]
    commands = []
    for variant_name in variant_names(ablation):
        spec = VARIANT_SPECS[variant_name]
        for domain_name in domains:
            cli = DOMAIN_CLI[domain_name]
            for seed in seed_list:
                guide_w = "1"
                extra_guide = None
                if variant_name == "C1_sweep":
                    guide_w = "-1"
                    extra_guide = guide_list
                argv = [
                    "python3", "-u", "exe_forecasting.py",
                    "--config", DOMAIN_CONFIGS[domain_name],
                    "--domain", domain_name,
                    "--variant", variant_name,
                    "--root_path", root_path,
                    "--data_path", cli["data_path"],
                    "--seq_len", str(cli["seq_len"]),
                    "--pred_len", str(cli["pred_len"]),
                    "--freq", cli["freq"],
                    "--text_len", str(cli["text_len"]),
                    "--device", device,
                    "--num_workers", "0",
                    "--seed", str(seed),
                    "--guide_w", guide_w,
                ]
                if extra_guide:
                    argv.extend(["--guide_list", extra_guide])
                if nsample is not None and str(nsample) != "":
                    argv.extend(["--nsample", str(nsample)])
                checkpoint = spec.get("checkpoint_variant")
                if checkpoint:
                    argv.extend(["--modelfolder", "%s_%s_seed%s" % (domain_name, checkpoint, seed)])
                override = spec.get("infer_override") or ""
                if override:
                    argv.extend(["--infer_override", override])
                    if override.startswith("guide_w="):
                        argv[argv.index("--guide_w") + 1] = override.split("=", 1)[1]
                commands.append(argv)
    return commands
