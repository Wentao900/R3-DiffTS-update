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
from utils.utils import train, evaluate

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


def resolve_multi_res_horizons(train_cfg, train_dataset, pred_len):
    explicit_horizons = sanitize_multi_res_horizons(
        train_cfg.get("multi_res_horizons"),
        pred_len,
    )
    if explicit_horizons:
        return {
            "horizons": explicit_horizons,
            "source": "explicit",
            "stats": {},
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
        return {
            "horizons": fallback_horizons,
            "source": "ratio_fallback",
            "stats": {"anchor_horizons": anchor_horizons},
            "fallback_used": True,
        }

    if train_dataset is None or not hasattr(train_dataset, "estimate_horizon_statistics"):
        return {
            "horizons": fallback_horizons,
            "source": "ratio_fallback",
            "stats": {"reason": "training dataset does not expose ACF statistics"},
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
        return {
            "horizons": horizons,
            "source": "train_acf",
            "stats": stats,
            "fallback_used": False,
        }
    except Exception as exc:
        return {
            "horizons": fallback_horizons,
            "source": "ratio_fallback",
            "stats": {"reason": str(exc)},
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
args.adaptive_noise_scale = float(config.get("train", {}).get("adaptive_noise_scale", 0.0))
args.text_quality_gate = bool(config["model"].get("text_quality_gate", True))
args.text_quality_min_scale = float(config["model"].get("text_quality_min_scale", 0.0))
args.text_quality_coverage_mix = float(config["model"].get("text_quality_coverage_mix", 0.5))
args.text_recency_tau_days = float(dataset_cfg.get("text_recency_tau_days", 14.0))
args.text_coverage_kappa = float(dataset_cfg.get("text_coverage_kappa", 3.0))
args.text_quality_weights = config["model"].get("text_quality_weights", [0.5, 0.3, 0.2])
args.text_trust_ret = float(config["model"].get("text_trust_ret", 0.75))
args.text_trust_cot = float(config["model"].get("text_trust_cot", 0.5))
args.text_quality_drop_threshold = float(config["model"].get("text_quality_drop_threshold", 0.3))
args.text_quality_mid_threshold = float(config["model"].get("text_quality_mid_threshold", 0.6))
args.text_trend_ret_scale = float(config["model"].get("text_trend_ret_scale", 0.5))
args.text_trend_cot_scale = float(config["model"].get("text_trend_cot_scale", 0.3))
args.text_trend_raw_weight = float(config["model"].get("text_trend_raw_weight", 1.0))
args.text_trend_ret_weight = float(config["model"].get("text_trend_ret_weight", 0.35))
args.text_trend_cot_weight = float(config["model"].get("text_trend_cot_weight", 0.15))
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

horizon_info = resolve_multi_res_horizons(
    config["train"],
    getattr(train_loader, "dataset", None),
    args.pred_len,
)
config["train"]["multi_res_horizons"] = horizon_info["horizons"]
config["train"]["multi_res_horizon_source"] = horizon_info["source"]
config["train"]["multi_res_horizon_stats"] = horizon_info["stats"]
print("resolved multi_res_horizons:", horizon_info["horizons"])
print("multi_res source:", horizon_info["source"])
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
if config["diffusion"]["cfg"]:
    best_mse = 10e10
    best_metrics = None
    guide_list = [args.guide_w] if args.guide_w >= 0 else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 4.5, 5.0]
    for guide_w in guide_list:
        metrics = evaluate(
            model,
            test_loader,
            nsample=args.nsample,
            scaler=scaler,
            mean_scaler=mean_scaler,
            foldername=foldername,
            window_lens=[args.seq_len, args.pred_len],
            guide_w=guide_w,
            save_attn=args.save_attn,
            save_token=args.save_token,
            save_trend_prior=args.save_trend_prior
        )
        guide_sweep_metrics.append(metrics)
        if metrics["MSE"] < best_mse:
            best_mse = metrics["MSE"]
            best_metrics = metrics
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
            save_trend_prior=args.save_trend_prior
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
    guide_sweep=guide_sweep_metrics,
)
