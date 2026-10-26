"""Entrypoint flags for the legacy pipeline and the gated cross-attention method.

Legacy configs keep the historical forced switches. The xattn configs set
``force_legacy_entrypoint: false`` and read the YAML values instead.
"""

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
