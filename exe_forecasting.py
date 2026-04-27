import argparse
import torch
import datetime
import json
import yaml
import os
import numpy as np
import random
import math

from main_model import CSDI_Forecasting
from dataset_forecasting import get_dataloader
from utils.utils import train, evaluate, fit_forecast_calibrator

parser = argparse.ArgumentParser(description="MCD-TSF")
parser.add_argument("--config", type=str, default="economy_36_18.yaml")
parser.add_argument("--datatype", type=str, default="multimodal")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=15)
parser.add_argument("--data", type=str, default="custom")
parser.add_argument("--embed", type=str, default="timeF")
parser.add_argument('--root_path', type=str, default='Time-MMD-main', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='Economy/Economy.csv', help='data file')
parser.add_argument('--seq_len', type=int, default=36, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=18, help='prediction sequence length')
parser.add_argument('--text_len', type=int, default=36, help='context length in time series freq')
parser.add_argument('--max_text_tokens', type=int, default=256, help='max tokens kept per text window after cleanup')
parser.add_argument('--text_drop_prob', type=float, default=0.0, help='probability to drop text during training/eval for robustness')
parser.add_argument('--use_rag_cot', action='store_true', help='enable retrieval-augmented CoT guidance text')
parser.add_argument('--cot_only', action='store_true', help='disable retrieval; only generate CoT guidance text')
parser.add_argument('--rag_topk', type=int, default=3, help='number of retrieved evidence snippets for RAG')
parser.add_argument('--use_two_stage_rag', action='store_true', help='enable two-stage retrieval for RAG guidance text')
parser.add_argument('--rag_stage1_topk', type=int, default=-1, help='stage-1 topk for two-stage RAG (-1 for auto)')
parser.add_argument('--rag_stage2_topk', type=int, default=-1, help='stage-2 topk for two-stage RAG (-1 for auto)')
parser.add_argument('--two_stage_gate', action='store_true', default=True, help='enable safety gate for two-stage RAG')
parser.add_argument('--trend_slope_eps', type=float, default=1e-3, help='slope epsilon for two-stage RAG gating')
parser.add_argument('--cot_model', type=str, default=None, help='local causal LM id/path for CoT generation (set None to use template)')
parser.add_argument('--cot_max_new_tokens', type=int, default=96, help='max new tokens for CoT generator')
parser.add_argument('--cot_temperature', type=float, default=0.7, help='sampling temperature for CoT generator')
parser.add_argument('--cot_cache_size', type=int, default=1024, help='cache size for generated CoT strings')
parser.add_argument('--cot_cache_dir', type=str, default=None, help='disk cache directory for generated RAG/CoT guidance')
parser.add_argument('--cot_device', type=str, default=None, help='device for CoT generator, e.g., cuda:0 or cpu')
parser.add_argument('--cot_load_in_8bit', action='store_true', help='load CoT model in 8-bit (requires bitsandbytes)')
parser.add_argument('--cot_load_in_4bit', action='store_true', help='load CoT model in 4-bit (requires bitsandbytes)')
parser.add_argument('--guide_w', type=float, default=-1, help='override guidance weight when cfg is enabled; negative to use default sweep')
parser.add_argument('--guide_list', type=str, default="", help='comma-separated guide weights for cfg sweep; overrides the built-in sweep when guide_w is negative')
parser.add_argument('--trend_cfg', action='store_true', help='enable trend-aware CFG modulation from CoT')
parser.add_argument('--trend_cfg_power', type=float, default=1.0, help='power for trend CFG time schedule')
parser.add_argument('--trend_cfg_random', action='store_true', help='replace trend prior with random draws')
parser.add_argument('--trend_strength_scale', type=float, default=1.0, help='affine mix for trend strength: 1 + scale*(strength-1)')
parser.add_argument('--trend_volatility_scale', type=float, default=1.0, help='scale for trend volatility in 1/(1+v*vol) penalty')
parser.add_argument('--trend_time_floor', type=float, default=0.0, help='minimum value added to trend time schedule')
parser.add_argument('--save_trend_prior', action='store_true', help='save per-sample trend priors during evaluation')
parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--freq', type=str, default='m', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--num_workers', type=int, default=16, help='data loader num workers')
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--attn_drop', type=float, default=0.)
parser.add_argument('--init', type=str, default='None')
parser.add_argument('--valid_interval', type=int, default=1)
parser.add_argument('--time_weight', type=float, default=0.1)
parser.add_argument('--c_mask_prob', type=float, default=-1)
parser.add_argument('--beta_end', type=float, default=-1)
parser.add_argument('--lr', type=float, default=-1)
parser.add_argument('--sample_steps_override', type=int, default=-1, help='override diffusion sample steps for fast testing')
parser.add_argument('--save_attn', type=bool, default=False)
parser.add_argument('--save_token', type=bool, default=False)


args = parser.parse_args()
print(args)


def default_multi_res_horizons(pred_len):
    pred_len = int(pred_len)
    if pred_len <= 0:
        return []
    if pred_len <= 4:
        return list(range(1, pred_len + 1))
    return sorted(
        set(
            [
                1,
                int(math.ceil(pred_len / 4.0)),
                int(math.ceil(pred_len / 2.0)),
                pred_len,
            ]
        )
    )


def sanitize_multi_res_horizons(horizons, pred_len):
    if horizons is None:
        return []
    if isinstance(horizons, int):
        horizons = [horizons]
    sanitized = []
    for horizon in horizons:
        try:
            horizon = int(horizon)
        except (TypeError, ValueError):
            continue
        if 1 <= horizon <= int(pred_len):
            sanitized.append(horizon)
    return sorted(set(sanitized))


def ordered_unique_horizons(candidates, pred_len, limit=5):
    result = []
    seen = set()
    for candidate in candidates:
        sanitized = sanitize_multi_res_horizons([candidate], pred_len)
        if not sanitized:
            continue
        horizon = sanitized[0]
        if horizon in seen:
            continue
        result.append(horizon)
        seen.add(horizon)
        if limit is not None and len(result) >= limit:
            break
    return sorted(result)


def get_horizon_bucket(horizon, pred_len):
    horizon = int(horizon)
    short_end = max(1, int(math.ceil(pred_len / 4.0)))
    mid_end = max(short_end + 1, int(math.ceil(pred_len / 2.0)))
    if horizon <= short_end:
        return "short"
    if horizon <= mid_end:
        return "mid"
    return "long"


def build_balanced_horizons(pred_len, candidates, max_count=5):
    ordered = ordered_unique_horizons(candidates, pred_len, limit=None)
    if not ordered:
        return []

    bucketed = {"short": [], "mid": [], "long": []}
    for horizon in ordered:
        bucketed[get_horizon_bucket(horizon, pred_len)].append(horizon)

    selected = []
    if bucketed["short"]:
        selected.append(bucketed["short"][0])
    if bucketed["mid"]:
        selected.append(bucketed["mid"][0])
    if bucketed["long"]:
        if int(pred_len) in bucketed["long"]:
            selected.append(int(pred_len))
        else:
            selected.append(bucketed["long"][-1])

    for horizon in ordered:
        if horizon not in selected:
            selected.append(horizon)
        if len(selected) >= max_count:
            break
    if int(pred_len) in ordered and int(pred_len) not in selected:
        if len(selected) >= max_count:
            selected = [h for h in selected if h != max(selected)]
        selected.append(int(pred_len))
    return sorted(selected[:max_count])


def _sigmoid(value):
    return 1.0 / (1.0 + math.exp(-float(value)))


def compute_horizon_reliabilities(horizons, stats, threshold=0.2, temperature=0.05, reference_horizons=None):
    if not horizons:
        return []
    if not stats:
        return [1.0 for _ in horizons]
    robust_acf = stats.get("robust_acf_values") or []
    signed_acf = stats.get("robust_signed_acf_values") or stats.get("acf_values") or stats.get("acf_head") or []
    if robust_acf:
        ref_horizons = reference_horizons or horizons
        ref_values = [
            float(robust_acf[int(h)])
            for h in ref_horizons
            if 0 <= int(h) < len(robust_acf)
        ]
        if not ref_values:
            ref_values = [float(x) for x in robust_acf[1:] if np.isfinite(x)]
        if not ref_values:
            return [1.0 for _ in horizons]
        tau = float(np.median(ref_values))
        mad = float(np.median(np.abs(np.asarray(ref_values, dtype=np.float64) - tau)))
        temp = max(mad, 1e-6)
        phase_anchors = [int(x) for x in stats.get("phase_anchors", []) if 0 < int(x) < len(robust_acf)]
        peak_lag = stats.get("peak_lag")
        peak_value = stats.get("peak_value")
        if peak_lag is not None and 0 <= int(peak_lag) < len(robust_acf):
            peak_score = float(robust_acf[int(peak_lag)])
        elif peak_value is not None:
            peak_score = float(peak_value)
        else:
            peak_score = max(ref_values)
        r_peak = _sigmoid((peak_score - tau) / temp)
        sigma_points = sorted(set(phase_anchors + [int(h) for h in ref_horizons if int(h) > 0]))
        if len(sigma_points) > 1:
            sigma_p = float(np.median(np.diff(np.asarray(sigma_points, dtype=np.float64))))
        else:
            sigma_p = 1.0
        sigma_p = max(sigma_p, 1e-6)
        decay_lag = stats.get("decay_lag")
        if decay_lag is None or int(decay_lag) <= 0:
            decay_lag = max(1, min(len(robust_acf) - 1, max(int(max(ref_horizons or [1])), 1)))
        short_end = max(1, min(int(decay_lag), len(robust_acf) - 1))
        short_acf_strength = float(np.max(np.asarray(robust_acf[1:short_end + 1], dtype=np.float64))) if short_end >= 1 else 0.0
        reliabilities = []
        for horizon in horizons:
            h = int(horizon)
            score = float(robust_acf[h]) if 0 <= h < len(robust_acf) else 0.0
            r_point = _sigmoid((score - tau) / temp)
            if phase_anchors:
                phase_score = max(math.exp(-((h - anchor) ** 2) / (2.0 * sigma_p ** 2)) for anchor in phase_anchors)
                r_phase = r_peak * phase_score
            else:
                r_phase = 0.0
            r_persist = short_acf_strength * math.exp(-float(max(h, 0)) / max(float(decay_lag), 1e-6))
            rel = max(r_point, r_phase, r_persist)
            reliabilities.append(float(min(max(rel, 0.0), 1.0)))
        return reliabilities

    acf_values = signed_acf or []
    peak_lag = stats.get("peak_lag")
    peak_value = stats.get("peak_value")
    reliabilities = []
    temp = max(float(temperature), 1e-6)
    for horizon in horizons:
        score = None
        h = int(horizon)
        if 0 <= h < len(acf_values):
            score = float(acf_values[h])
        elif peak_lag is not None and int(peak_lag) == h and peak_value is not None:
            score = float(peak_value)
        if score is None:
            score = 1.0 if not acf_values else 0.0
        score = max(score, 0.0)
        rel = _sigmoid((score - float(threshold)) / temp)
        reliabilities.append(float(min(max(rel, 0.0), 1.0)))
    return reliabilities


def compute_segment_reliabilities(horizons, stats):
    if not horizons:
        return []
    max_horizon = max(int(h) for h in horizons)
    lag_reliabilities = compute_horizon_reliabilities(
        list(range(1, max_horizon + 1)),
        stats,
        reference_horizons=horizons,
    )
    rel_by_lag = {lag: lag_reliabilities[lag - 1] for lag in range(1, max_horizon + 1)}
    segment_reliabilities = []
    prev_h = 0
    for horizon in horizons:
        h = int(horizon)
        values = [rel_by_lag[lag] for lag in range(prev_h + 1, h + 1) if lag in rel_by_lag]
        if not values:
            segment_reliabilities.append(1.0)
        else:
            values_arr = np.asarray(values, dtype=np.float64)
            mean_value = float(values_arr.mean())
            std_value = float(values_arr.std())
            lam = std_value / (mean_value + std_value + 1e-8)
            segment_reliabilities.append(float((1.0 - lam) * mean_value + lam * float(values_arr.max())))
        prev_h = h
    return segment_reliabilities


def resolve_multi_res_horizons(train_cfg, train_dataset, pred_len):
    explicit_horizons = sanitize_multi_res_horizons(
        train_cfg.get("multi_res_horizons"),
        pred_len,
    )
    if explicit_horizons:
        segment_reliabilities = [1.0 for _ in explicit_horizons]
        return {
            "horizons": explicit_horizons,
            "source": "explicit",
            "stats": {"horizon_reliabilities": [1.0 for _ in explicit_horizons], "segment_reliabilities": segment_reliabilities},
            "horizon_reliabilities": [1.0 for _ in explicit_horizons],
            "segment_reliabilities": segment_reliabilities,
            "fallback_used": False,
        }

    fallback_horizons = default_multi_res_horizons(pred_len)
    anchor_horizons = ordered_unique_horizons(
        [
            1,
            int(math.ceil(pred_len / 4.0)),
            int(math.ceil(pred_len / 2.0)),
            int(pred_len),
        ],
        pred_len,
        limit=4,
    )
    if not bool(train_cfg.get("multi_res_use_stat_horizons", True)):
        reliabilities = [1.0 for _ in fallback_horizons]
        segment_reliabilities = [1.0 for _ in fallback_horizons]
        return {
            "horizons": fallback_horizons,
            "source": "ratio_fallback",
            "stats": {"anchor_horizons": anchor_horizons, "horizon_reliabilities": reliabilities, "segment_reliabilities": segment_reliabilities},
            "horizon_reliabilities": reliabilities,
            "segment_reliabilities": segment_reliabilities,
            "fallback_used": True,
        }

    if train_dataset is None or not hasattr(train_dataset, "estimate_horizon_statistics"):
        reliabilities = [1.0 for _ in fallback_horizons]
        segment_reliabilities = [1.0 for _ in fallback_horizons]
        return {
            "horizons": fallback_horizons,
            "source": "ratio_fallback",
            "stats": {"reason": "training dataset does not expose ACF statistics", "horizon_reliabilities": reliabilities, "segment_reliabilities": segment_reliabilities},
            "horizon_reliabilities": reliabilities,
            "segment_reliabilities": segment_reliabilities,
            "fallback_used": True,
        }

    try:
        stats = train_dataset.estimate_horizon_statistics(
            drop_threshold=train_cfg.get("multi_res_acf_drop_threshold", 0.5),
            zero_threshold=train_cfg.get("multi_res_acf_zero_threshold", 0.1),
            max_lag=train_cfg.get("multi_res_acf_max_lag"),
            num_samples=train_cfg.get("multi_res_acf_num_samples", 128),
        )
        candidate_order = [
            1,
            stats.get("decay_lag"),
            stats.get("zero_lag"),
            stats.get("peak_lag"),
            int(math.ceil(pred_len / 4.0)),
            int(math.ceil(pred_len / 2.0)),
            int(pred_len),
        ]
        horizons = build_balanced_horizons(pred_len, candidate_order, max_count=5)
        if len(horizons) < 3:
            horizons = build_balanced_horizons(pred_len, candidate_order + anchor_horizons + fallback_horizons, max_count=5)
        if len(horizons) == 0:
            raise RuntimeError("no valid horizons generated from training ACF")
        stats["anchor_horizons"] = anchor_horizons
        reliabilities = compute_horizon_reliabilities(
            horizons,
            stats,
            threshold=train_cfg.get("multi_res_acf_reliability_threshold", 0.2),
            temperature=train_cfg.get("multi_res_acf_reliability_temperature", 0.05),
        )
        segment_reliabilities = compute_segment_reliabilities(horizons, stats)
        stats["horizon_reliabilities"] = reliabilities
        stats["segment_reliabilities"] = segment_reliabilities
        return {
            "horizons": horizons,
            "source": "train_acf",
            "stats": stats,
            "horizon_reliabilities": reliabilities,
            "segment_reliabilities": segment_reliabilities,
            "fallback_used": False,
        }
    except Exception as exc:
        reliabilities = [1.0 for _ in fallback_horizons]
        segment_reliabilities = [1.0 for _ in fallback_horizons]
        return {
            "horizons": fallback_horizons,
            "source": "ratio_fallback",
            "stats": {"reason": str(exc), "horizon_reliabilities": reliabilities, "segment_reliabilities": segment_reliabilities},
            "horizon_reliabilities": reliabilities,
            "segment_reliabilities": segment_reliabilities,
            "fallback_used": True,
        }


def write_run_summary(foldername, config, horizon_info, metrics=None, guide_sweep=None, extra=None):
    summary = {
        "config": config,
        "multi_res": horizon_info,
    }
    if metrics is not None:
        summary["metrics"] = metrics
    if guide_sweep is not None:
        summary["guide_sweep"] = guide_sweep
    if extra is not None:
        summary.update(extra)
    with open(os.path.join(foldername, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.text_len == 0:
    args.text_len = args.seq_len

timestep_dim_dict = {
    'd': 3,
    'w': 2,
    'm': 1
}
extra_timestep_dims = 2 if args.data == 'custom' else 0
context_dim_dict = {
    'bert': 768,
    'llama': 4096,
    'gpt2': 768
}
path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)
# Force unified TAA + TTF path: always use texts, timestep embeddings, and timestep branch
config["model"]["with_texts"] = True
config["model"]["timestep_emb_cat"] = True
config["model"]["timestep_branch"] = True
args.use_rag_cot = config["model"].get("use_rag_cot", args.use_rag_cot)
args.cot_only = config["model"].get("cot_only", args.cot_only)
args.use_two_stage_rag = config["model"].get("use_two_stage_rag", args.use_two_stage_rag)
args.rag_stage1_topk = config["model"].get("rag_stage1_topk", args.rag_stage1_topk)
args.rag_stage2_topk = config["model"].get("rag_stage2_topk", args.rag_stage2_topk)
args.two_stage_gate = config["model"].get("two_stage_gate", args.two_stage_gate)
args.trend_slope_eps = config["model"].get("trend_slope_eps", args.trend_slope_eps)
if args.cot_only:
    args.use_rag_cot = True
    args.rag_topk = 0
args.rag_topk = config["model"].get("rag_topk", args.rag_topk)
args.cot_model = config["model"].get("cot_model", args.cot_model)
args.cot_max_new_tokens = config["model"].get("cot_max_new_tokens", args.cot_max_new_tokens)
args.cot_temperature = config["model"].get("cot_temperature", args.cot_temperature)
args.cot_cache_size = config["model"].get("cot_cache_size", args.cot_cache_size)
args.cot_cache_dir = config["model"].get("cot_cache_dir", args.cot_cache_dir)
args.cot_device = config["model"].get("cot_device", args.cot_device)
args.cot_load_in_8bit = config["model"].get("cot_load_in_8bit", args.cot_load_in_8bit)
args.cot_load_in_4bit = config["model"].get("cot_load_in_4bit", args.cot_load_in_4bit)
args.trend_cfg = config["diffusion"].get("trend_cfg", args.trend_cfg)
args.trend_cfg_power = config["diffusion"].get("trend_cfg_power", args.trend_cfg_power)
args.trend_cfg_random = config["diffusion"].get("trend_cfg_random", args.trend_cfg_random)
args.trend_strength_scale = config["diffusion"].get("trend_strength_scale", args.trend_strength_scale)
args.trend_volatility_scale = config["diffusion"].get("trend_volatility_scale", args.trend_volatility_scale)
args.trend_time_floor = config["diffusion"].get("trend_time_floor", args.trend_time_floor)
args.save_trend_prior = config["model"].get("save_trend_prior", args.save_trend_prior)
if args.embed == 'timeF':
    if config["model"]["timestep_branch"] or config["model"]["timestep_emb_cat"]:
        config["model"]["timestep_dim"] = timestep_dim_dict[args.freq] + extra_timestep_dims
    else:
        config["model"]["timestep_dim"] = 0
else:
    config["model"]["timestep_dim"] = 4 + extra_timestep_dims
config["model"]["context_dim"] = context_dim_dict[config["model"]["llm"]] if config["model"]["with_texts"] else 0

if args.datatype == 'electricity':
    target_dim = 370
    args.seq_len = 168
    args.pred_len = 24
else:
    target_dim = 1

config["model"]["is_unconditional"] = args.unconditional
config["model"]["lookback_len"] = args.seq_len
config["model"]["pred_len"] = args.pred_len
config["model"]["domain"] = args.data_path.split('/')[0]
config["model"]["text_len"] = args.text_len
config["model"]["save_attn"] = args.save_attn
config["model"]["save_token"] = args.save_token
config["diffusion"]["dropout"] = args.dropout
config["diffusion"]["attn_drop"] = args.attn_drop
config["diffusion"]["time_weight"] = args.time_weight
config["model"]["rag_topk"] = config["model"].get("rag_topk", 1)
config["model"]["cot_temperature"] = config["model"].get("cot_temperature", 0.55)
config["model"]["use_rag_cot"] = args.use_rag_cot
config["model"]["cot_only"] = args.cot_only
config["model"]["rag_topk"] = args.rag_topk
config["model"]["use_two_stage_rag"] = args.use_two_stage_rag
config["model"]["rag_stage1_topk"] = args.rag_stage1_topk
config["model"]["rag_stage2_topk"] = args.rag_stage2_topk
config["model"]["two_stage_gate"] = args.two_stage_gate
config["model"]["trend_slope_eps"] = args.trend_slope_eps
config["model"]["cot_model"] = args.cot_model
config["model"]["cot_max_new_tokens"] = args.cot_max_new_tokens
config["model"]["cot_temperature"] = args.cot_temperature
config["model"]["cot_cache_size"] = args.cot_cache_size
config["model"]["cot_cache_dir"] = args.cot_cache_dir
config["model"]["cot_device"] = args.cot_device
config["model"]["cot_load_in_8bit"] = args.cot_load_in_8bit
config["model"]["cot_load_in_4bit"] = args.cot_load_in_4bit
config["model"]["save_trend_prior"] = args.save_trend_prior
config["model"]["pattern_residual_diffusion"] = bool(config["model"].get("pattern_residual_diffusion", True))
config["model"]["guide_mode"] = str(config["model"].get("guide_mode", "manual")).lower()
config["model"]["final_method"] = bool(config["model"].get("final_method", config["model"]["guide_mode"] == "auto"))
config["model"]["pattern_text_evidence"] = bool(config["model"].get("pattern_text_evidence", True))
config["model"]["pattern_hidden_dim"] = int(config["model"].get("pattern_hidden_dim", 64))
config["model"]["pattern_router_temperature"] = float(config["model"].get("pattern_router_temperature", 1.0))
config["model"]["pattern_reliability"] = bool(config["model"].get("pattern_reliability", True))
config["model"]["pattern_reliability_threshold"] = float(config["model"].get("pattern_reliability_threshold", 0.35))
config["model"]["pattern_reliability_temperature"] = float(config["model"].get("pattern_reliability_temperature", 0.1))
config["model"]["pattern_reliability_min"] = float(config["model"].get("pattern_reliability_min", 0.05))
config["model"]["pattern_reliability_max"] = float(config["model"].get("pattern_reliability_max", 1.0))
config["model"]["pattern_aux_reliability"] = bool(config["model"].get("pattern_aux_reliability", True))
config["model"]["pattern_text_drop_prob"] = float(config["model"].get("pattern_text_drop_prob", config["diffusion"].get("c_mask_prob", 0.0)))
config["train"]["pattern_consistency_weight"] = float(config["train"].get("pattern_consistency_weight", 0.03))
config["train"]["pattern_expert_weight"] = float(config["train"].get("pattern_expert_weight", 0.05))
config["train"]["multi_res_segment_loss"] = bool(config["train"].get("multi_res_segment_loss", True))
config["train"]["multi_res_reliability_weight"] = float(config["train"].get("multi_res_reliability_weight", 1.0))
config["train"]["multi_res_difficulty_inverse"] = bool(config["train"].get("multi_res_difficulty_inverse", True))
if "multi_res_difficulty_gamma" in config["train"]:
    config["train"]["multi_res_difficulty_gamma"] = float(config["train"]["multi_res_difficulty_gamma"])
config["train"]["multi_res_acf_reliability_threshold"] = float(config["train"].get("multi_res_acf_reliability_threshold", 0.2))
config["train"]["multi_res_acf_reliability_temperature"] = float(config["train"].get("multi_res_acf_reliability_temperature", 0.05))
config["train"]["forecast_point_estimator"] = str(config["train"].get("forecast_point_estimator", "mean")).lower()
config["train"]["forecast_calibrator"] = bool(config["train"].get("forecast_calibrator", False))
config["train"]["forecast_calibrator_ridge"] = float(config["train"].get("forecast_calibrator_ridge", 1e-3))
config["train"]["forecast_calibrator_min_gain"] = float(config["train"].get("forecast_calibrator_min_gain", 0.0))
config["train"]["forecast_calibrator_max_strength"] = float(config["train"].get("forecast_calibrator_max_strength", 1.0))
config["train"]["forecast_calibrator_max_batches"] = int(config["train"].get("forecast_calibrator_max_batches", 0))
config["train"]["forecast_calibrator_holdout_fraction"] = float(config["train"].get("forecast_calibrator_holdout_fraction", 0.35))
config["train"]["forecast_calibrator_residual_clip_quantile"] = float(config["train"].get("forecast_calibrator_residual_clip_quantile", 0.95))
config["train"]["forecast_calibrator_include_timestamp"] = bool(config["train"].get("forecast_calibrator_include_timestamp", False))
legacy_model_keys = [
    "pattern_adaptive",
    "pattern_disable_legacy_text_gates",
    "text_quality_gate",
    "text_quality_min_scale",
    "text_use_ret_in_context",
    "text_use_cot_in_context",
    "text_trend_only_guidance",
    "text_trend_ret_scale",
    "text_trend_cot_scale",
    "text_trend_raw_weight",
    "text_trend_ret_weight",
    "text_trend_cot_weight",
    "text_numeric_align_gamma",
    "multi_res_trend_source",
    "text_aug_max_ratio",
    "coverage_power",
    "coverage_cfg_boost",
    "reliability_min",
    "guide_reliability_power",
    "semantic_dim",
    "event_quality_dim",
    "event_quality_beta",
    "event_text_max_length",
    "use_gate_min",
    "strength_gate_min",
    "strength_use_mix_floor",
    "horizon_strength_bias",
    "text_context_ratio_min",
    "text_context_ratio_max",
    "text_context_max_base",
    "text_context_max_boost",
    "text_context_horizon_bias",
    "text_guide_ratio_max",
    "text_guide_max_base",
    "text_guide_max_boost",
    "text_guide_quality_power",
    "text_guide_step_low",
    "text_guide_step_high",
    "text_guide_step_k",
    "use_gate_warmup_epochs",
    "strength_gate_warmup_epochs",
    "text_benefit_hidden_dim",
    "text_aug_hidden_dim",
    "reliability_hidden_dim",
    "event_source_embed_dim",
    "aux_basis_dim",
    "use_calendar_residual",
    "calendar_residual_scale",
]
legacy_train_keys = [
    "text_consistency_weight",
    "text_use_weight",
    "text_use_margin",
    "text_aug_benefit_weight",
    "text_aug_reg_weight",
    "text_notext_fallback_weight",
    "text_strength_weight",
    "text_strength_tau",
    "text_context_benefit_weight",
    "text_context_benefit_tau",
    "text_positive_benefit_margin",
    "text_positive_benefit_tau",
    "detach_text_baselines",
    "aux_residual_mag_weight",
    "aux_residual_smooth_weight",
]
for key in legacy_model_keys:
    config["model"].pop(key, None)
for key in legacy_train_keys:
    config["train"].pop(key, None)
config["diffusion"]["trend_cfg"] = args.trend_cfg
config["diffusion"]["trend_cfg_power"] = args.trend_cfg_power
config["diffusion"]["trend_cfg_random"] = args.trend_cfg_random
config["diffusion"]["trend_strength_scale"] = args.trend_strength_scale
config["diffusion"]["trend_volatility_scale"] = args.trend_volatility_scale
config["diffusion"]["trend_time_floor"] = args.trend_time_floor

if args.c_mask_prob > 0:
    config["diffusion"]["c_mask_prob"] = args.c_mask_prob

if args.beta_end > 0:
    config["diffusion"]["beta_end"] = args.beta_end
if args.sample_steps_override > 0:
    config["diffusion"]["sample_steps"] = args.sample_steps_override

if args.lr > 0:
    config["train"]["lr"] = args.lr

dataset_cfg = {}
dataset_cfg.update(config.get("data", {}))
dataset_cfg.update(config.get("dataset", {}))
args.aug_noise_std = float(dataset_cfg.get("aug_noise_std", 0.0))
args.aug_time_warp_prob = float(dataset_cfg.get("aug_time_warp_prob", 0.0))
args.aug_segment_scale_std = float(dataset_cfg.get("aug_segment_scale_std", 0.1))
args.use_all_numeric_features = bool(dataset_cfg.get("use_all_numeric_features", False))
args.covariate_columns = dataset_cfg.get("covariate_columns", None)
args.exclude_numeric_features = dataset_cfg.get("exclude_numeric_features", None)
args.adaptive_noise_scale = float(config.get("train", {}).get("adaptive_noise_scale", 0.0))
args.text_quality_coverage_mix = float(config["model"].get("text_quality_coverage_mix", 0.5))
args.text_recency_tau_days = float(dataset_cfg.get("text_recency_tau_days", 14.0))
args.text_coverage_kappa = float(dataset_cfg.get("text_coverage_kappa", 3.0))
args.text_quality_weights = config["model"].get("text_quality_weights", [0.5, 0.3, 0.2])
args.text_trust_ret = float(config["model"].get("text_trust_ret", 0.75))
args.text_trust_cot = float(config["model"].get("text_trust_cot", 0.5))
args.text_quality_drop_threshold = float(config["model"].get("text_quality_drop_threshold", 0.3))
args.text_quality_mid_threshold = float(config["model"].get("text_quality_mid_threshold", 0.6))
args.max_text_events = int(dataset_cfg.get("max_text_events", 12))
args.num_workers = int(dataset_cfg.get("num_workers", args.num_workers))

args.batch_size = config["train"]["batch_size"]

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/forecasting_" + args.data_path.split('/')[0] + '_' + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    datatype=args.datatype,
    device= args.device,
    batch_size=config["train"]["batch_size"],
    args=args
)

if hasattr(train_loader, "dataset"):
    dataset_feature_dim = int(getattr(train_loader.dataset, "feature_dim", target_dim))
    target_dim = dataset_feature_dim
    config["model"]["target_feature_index"] = int(getattr(train_loader.dataset, "target_index", 0))
    config["model"]["feature_dim"] = dataset_feature_dim

horizon_info = resolve_multi_res_horizons(
    config["train"],
    getattr(train_loader, "dataset", None),
    args.pred_len,
)
config["train"]["multi_res_horizons"] = horizon_info["horizons"]
config["train"]["multi_res_horizon_source"] = horizon_info["source"]
config["train"]["multi_res_horizon_stats"] = horizon_info["stats"]
config["train"]["multi_res_horizon_reliabilities"] = horizon_info.get(
    "horizon_reliabilities",
    horizon_info.get("stats", {}).get("horizon_reliabilities", []),
)
config["train"]["multi_res_segment_reliabilities"] = horizon_info.get(
    "segment_reliabilities",
    horizon_info.get("stats", {}).get("segment_reliabilities", []),
)
print("resolved multi_res_horizons:", horizon_info["horizons"])
print("multi_res source:", horizon_info["source"])
print("multi_res reliabilities:", config["train"]["multi_res_horizon_reliabilities"])
print("multi_res segment reliabilities:", config["train"]["multi_res_segment_reliabilities"])
print(json.dumps(config, indent=4))
with open(foldername + "config_results.json", "w") as f:
    json.dump(config, f, indent=4)
write_run_summary(foldername, config, horizon_info)

model = CSDI_Forecasting(config, args.device, target_dim, window_lens=[args.seq_len, args.pred_len]).to(args.device)
write_run_summary(
    foldername,
    config,
    {
        **horizon_info,
        "model_state": model.get_multi_res_debug_state(),
    },
)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
        valid_epoch_interval=args.valid_interval
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
model.target_dim = target_dim
guide_sweep_metrics = []
forecast_calibrator = None
if config["train"].get("forecast_calibrator", False):
    forecast_calibrator = fit_forecast_calibrator(
        model,
        valid_loader,
        nsample=args.nsample,
        foldername=foldername,
        guide_w=0,
        ridge_alpha=config["train"].get("forecast_calibrator_ridge", 1e-3),
        min_gain=config["train"].get("forecast_calibrator_min_gain", 0.0),
        max_strength=config["train"].get("forecast_calibrator_max_strength", 1.0),
        max_batches=config["train"].get("forecast_calibrator_max_batches", 0),
        holdout_fraction=config["train"].get("forecast_calibrator_holdout_fraction", 0.35),
        residual_clip_quantile=config["train"].get("forecast_calibrator_residual_clip_quantile", 0.95),
        include_timestamp=config["train"].get("forecast_calibrator_include_timestamp", False),
    )
    if forecast_calibrator is not None:
        printable_calibrator = {k: v for k, v in forecast_calibrator.items() if k != "coefficients"}
        print("forecast calibrator:", json.dumps(printable_calibrator, indent=4))
    else:
        print("forecast calibrator: unavailable")
if config["diffusion"]["cfg"] and config["model"].get("guide_mode") != "auto":
    best_mse = 10e10
    best_metrics = None
    best_guide_w = None
    selection_loader = valid_loader if valid_loader is not None else test_loader
    selection_split = "valid" if valid_loader is not None else "test"
    if args.guide_w >= 0:
        guide_list = [args.guide_w]
    elif args.guide_list.strip():
        guide_list = [float(value) for value in args.guide_list.split(",") if value.strip()]
    else:
        guide_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 4.5, 5.0]
    for guide_w in guide_list:
        metrics = evaluate(
            model,
            selection_loader,
            nsample=args.nsample,
            scaler=scaler,
            mean_scaler=mean_scaler,
            foldername=foldername,
            window_lens=[args.seq_len, args.pred_len],
            guide_w=guide_w,
            save_attn=args.save_attn,
            save_token=args.save_token,
            save_trend_prior=args.save_trend_prior,
            point_estimator=config["train"].get("forecast_point_estimator", "mean"),
            forecast_calibrator=forecast_calibrator,
        )
        metrics = {**metrics, "selection_split": selection_split}
        guide_sweep_metrics.append(metrics)
        if metrics["MSE"] < best_mse:
            best_mse = metrics["MSE"]
            best_metrics = metrics
            best_guide_w = guide_w
    if best_guide_w is not None:
        best_metrics = evaluate(
            model,
            test_loader,
            nsample=args.nsample,
            scaler=scaler,
            mean_scaler=mean_scaler,
            foldername=foldername,
            window_lens=[args.seq_len, args.pred_len],
            guide_w=best_guide_w,
            save_attn=args.save_attn,
            save_token=args.save_token,
            save_trend_prior=args.save_trend_prior,
            point_estimator=config["train"].get("forecast_point_estimator", "mean"),
            forecast_calibrator=forecast_calibrator,
        )
        best_metrics["selected_guide_w"] = best_guide_w
        best_metrics["selection_split"] = selection_split
else:
    best_metrics = evaluate(
            model,
            test_loader,
            nsample=args.nsample,
            scaler=scaler,
            mean_scaler=mean_scaler,
            foldername=foldername,
            window_lens=[args.seq_len, args.pred_len],
            save_attn=args.save_attn,
            save_token=args.save_token,
            save_trend_prior=args.save_trend_prior,
            point_estimator=config["train"].get("forecast_point_estimator", "mean"),
            forecast_calibrator=forecast_calibrator,
        )
    guide_sweep_metrics.append(best_metrics)

if config["diffusion"]["cfg"] and best_metrics is None:
    best_metrics = {"MSE": best_mse}

write_run_summary(
    foldername,
    config,
    {
        **horizon_info,
        "model_state": model.get_multi_res_debug_state(),
    },
    metrics=best_metrics,
    guide_sweep=None if config["model"].get("guide_mode") == "auto" else guide_sweep_metrics,
    extra={"selected_metric": "MSE"},
)
