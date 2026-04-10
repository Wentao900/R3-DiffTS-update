import argparse
import torch
import datetime
import json
import yaml
import os
import numpy as np
import random

from main_model import CSDI_Forecasting
from dataset_forecasting import get_dataloader
from utils.utils import train, evaluate


def _parse_int_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parsed = []
        for item in value:
            try:
                parsed.append(int(item))
            except (TypeError, ValueError):
                continue
        return [item for item in parsed if item > 0]
    text = str(value).strip()
    if not text:
        return []
    parsed = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parsed.append(int(item))
        except ValueError:
            continue
    return [item for item in parsed if item > 0]


def _parse_float_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parsed = []
        for item in value:
            try:
                parsed.append(float(item))
            except (TypeError, ValueError):
                continue
        return parsed
    text = str(value).strip()
    if not text:
        return []
    parsed = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parsed.append(float(item))
        except ValueError:
            continue
    return parsed

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
parser.add_argument('--rag_consistency', action='store_true', help='compute evidence consistency from retrieved snippets')
parser.add_argument('--consistency_unknown_penalty', type=float, default=1.0, help='penalty strength for unknown evidence stances')
parser.add_argument('--consistency_conflict_penalty', type=float, default=0.5, help='penalty strength for conflicting evidence stances')
parser.add_argument('--cot_model', type=str, default=None, help='local causal LM id/path for CoT generation (set None to use template)')
parser.add_argument('--cot_max_new_tokens', type=int, default=96, help='max new tokens for CoT generator')
parser.add_argument('--cot_temperature', type=float, default=0.7, help='sampling temperature for CoT generator')
parser.add_argument('--cot_cache_size', type=int, default=1024, help='cache size for generated CoT strings')
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
parser.add_argument('--step_guidance', action='store_true', help='anneal CFG strength across reverse diffusion steps')
parser.add_argument('--step_guidance_power', type=float, default=1.0, help='power for step-aware CFG schedule')
parser.add_argument('--step_guidance_floor', type=float, default=0.0, help='minimum value added to step-aware CFG schedule')
parser.add_argument('--save_trend_prior', action='store_true', help='save per-sample trend priors during evaluation')
parser.add_argument('--use_scale_router', action='store_true', help='enable heuristic sample-level scale routing')
parser.add_argument('--scale_route_horizons', type=str, default='', help='comma-separated horizon endpoints for scale routing; empty uses train.multi_res_horizons or auto')
parser.add_argument('--scale_window_candidates', type=str, default='', help='comma-separated candidate text windows; empty uses evenly spaced windows up to text_len')
parser.add_argument('--scale_route_temperature', type=float, default=0.20, help='temperature for heuristic scale-routing soft assignment')
parser.add_argument('--scale_guidance', action='store_true', help='modulate sample-level CFG strength with scale routing during inference')
parser.add_argument('--scale_guidance_alpha', type=str, default='', help='comma-separated guidance multipliers aligned with scale bins, e.g. 0.9,1.0,1.1,1.2')
parser.add_argument('--consistency_guidance', action='store_true', help='multiply inference guidance by evidence consistency')
parser.add_argument('--consistency_threshold', type=float, default=0.0, help='zero-out consistency guidance below this threshold')
parser.add_argument('--multi_res_partition_mode', type=str, default='', help='override multi-res partition mode: cumulative or disjoint')
parser.add_argument('--multi_res_use_scale_router', action='store_true', help='weight multi-res bins with sample-level scale routing')
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
args.use_rag_cot = config["model"].get("use_rag_cot", args.use_rag_cot)
args.cot_only = config["model"].get("cot_only", args.cot_only)
args.use_two_stage_rag = config["model"].get("use_two_stage_rag", args.use_two_stage_rag)
args.rag_stage1_topk = config["model"].get("rag_stage1_topk", args.rag_stage1_topk)
args.rag_stage2_topk = config["model"].get("rag_stage2_topk", args.rag_stage2_topk)
args.two_stage_gate = config["model"].get("two_stage_gate", args.two_stage_gate)
args.trend_slope_eps = config["model"].get("trend_slope_eps", args.trend_slope_eps)
args.rag_consistency = config["model"].get("rag_consistency", args.rag_consistency)
args.consistency_unknown_penalty = config["model"].get("consistency_unknown_penalty", args.consistency_unknown_penalty)
args.consistency_conflict_penalty = config["model"].get("consistency_conflict_penalty", args.consistency_conflict_penalty)
if args.cot_only:
    args.use_rag_cot = True
    args.rag_topk = 0
args.rag_topk = config["model"].get("rag_topk", args.rag_topk)
args.cot_model = config["model"].get("cot_model", args.cot_model)
args.cot_max_new_tokens = config["model"].get("cot_max_new_tokens", args.cot_max_new_tokens)
args.cot_temperature = config["model"].get("cot_temperature", args.cot_temperature)
args.cot_cache_size = config["model"].get("cot_cache_size", args.cot_cache_size)
args.cot_device = config["model"].get("cot_device", args.cot_device)
args.cot_load_in_8bit = config["model"].get("cot_load_in_8bit", args.cot_load_in_8bit)
args.cot_load_in_4bit = config["model"].get("cot_load_in_4bit", args.cot_load_in_4bit)
args.trend_cfg = config["diffusion"].get("trend_cfg", args.trend_cfg)
args.trend_cfg_power = config["diffusion"].get("trend_cfg_power", args.trend_cfg_power)
args.trend_cfg_random = config["diffusion"].get("trend_cfg_random", args.trend_cfg_random)
args.trend_strength_scale = config["diffusion"].get("trend_strength_scale", args.trend_strength_scale)
args.trend_volatility_scale = config["diffusion"].get("trend_volatility_scale", args.trend_volatility_scale)
args.trend_time_floor = config["diffusion"].get("trend_time_floor", args.trend_time_floor)
args.step_guidance = config["diffusion"].get("step_guidance", args.step_guidance)
args.step_guidance_power = config["diffusion"].get("step_guidance_power", args.step_guidance_power)
args.step_guidance_floor = config["diffusion"].get("step_guidance_floor", args.step_guidance_floor)
args.save_trend_prior = config["model"].get("save_trend_prior", args.save_trend_prior)
args.use_scale_router = config["model"].get("use_scale_router", args.use_scale_router)
args.scale_route_temperature = config["model"].get("scale_route_temperature", args.scale_route_temperature)
args.scale_window_candidates = _parse_int_list(
    config["model"].get("scale_window_candidates", args.scale_window_candidates)
)
args.scale_guidance = config["diffusion"].get("scale_guidance", args.scale_guidance)
args.scale_guidance_alpha = _parse_float_list(
    config["diffusion"].get("scale_guidance_alpha", args.scale_guidance_alpha)
)
args.consistency_guidance = config["diffusion"].get("consistency_guidance", args.consistency_guidance)
args.consistency_threshold = config["diffusion"].get("consistency_threshold", args.consistency_threshold)
args.scale_route_horizons = _parse_int_list(
    config["train"].get("scale_route_horizons", args.scale_route_horizons)
)
if len(args.scale_route_horizons) == 0:
    args.scale_route_horizons = _parse_int_list(config["train"].get("multi_res_horizons", []))
args.multi_res_partition_mode = config["train"].get(
    "multi_res_partition_mode",
    args.multi_res_partition_mode or "cumulative",
)
args.multi_res_use_scale_router = config["train"].get(
    "multi_res_use_scale_router",
    args.multi_res_use_scale_router,
)
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
config["model"]["rag_consistency"] = args.rag_consistency
config["model"]["consistency_unknown_penalty"] = args.consistency_unknown_penalty
config["model"]["consistency_conflict_penalty"] = args.consistency_conflict_penalty
config["model"]["cot_model"] = args.cot_model
config["model"]["cot_max_new_tokens"] = args.cot_max_new_tokens
config["model"]["cot_temperature"] = args.cot_temperature
config["model"]["cot_cache_size"] = args.cot_cache_size
config["model"]["cot_device"] = args.cot_device
config["model"]["cot_load_in_8bit"] = args.cot_load_in_8bit
config["model"]["cot_load_in_4bit"] = args.cot_load_in_4bit
config["model"]["save_trend_prior"] = args.save_trend_prior
config["model"]["use_scale_router"] = args.use_scale_router
config["model"]["scale_window_candidates"] = args.scale_window_candidates
config["model"]["scale_route_temperature"] = args.scale_route_temperature
config["diffusion"]["trend_cfg"] = args.trend_cfg
config["diffusion"]["trend_cfg_power"] = args.trend_cfg_power
config["diffusion"]["trend_cfg_random"] = args.trend_cfg_random
config["diffusion"]["trend_strength_scale"] = args.trend_strength_scale
config["diffusion"]["trend_volatility_scale"] = args.trend_volatility_scale
config["diffusion"]["trend_time_floor"] = args.trend_time_floor
config["diffusion"]["step_guidance"] = args.step_guidance
config["diffusion"]["step_guidance_power"] = args.step_guidance_power
config["diffusion"]["step_guidance_floor"] = args.step_guidance_floor
config["diffusion"]["scale_guidance"] = args.scale_guidance
config["diffusion"]["scale_guidance_alpha"] = args.scale_guidance_alpha
config["diffusion"]["consistency_guidance"] = args.consistency_guidance
config["diffusion"]["consistency_threshold"] = args.consistency_threshold
config["train"]["scale_route_horizons"] = args.scale_route_horizons
config["train"]["multi_res_partition_mode"] = args.multi_res_partition_mode
config["train"]["multi_res_use_scale_router"] = args.multi_res_use_scale_router

if args.c_mask_prob > 0:
    config["diffusion"]["c_mask_prob"] = args.c_mask_prob

if args.beta_end > 0:
    config["diffusion"]["beta_end"] = args.beta_end
if args.sample_steps_override > 0:
    config["diffusion"]["sample_steps"] = args.sample_steps_override

if args.lr > 0:
    config["train"]["lr"] = args.lr

args.batch_size = config["train"]["batch_size"]

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/forecasting_" + args.data_path.split('/')[0] + '_' + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config_results.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    datatype=args.datatype,
    device= args.device,
    batch_size=config["train"]["batch_size"],
    args=args
)

model = CSDI_Forecasting(config, args.device, target_dim, window_lens=[args.seq_len, args.pred_len]).to(args.device)

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
if config["diffusion"]["cfg"]:
    best_mse = 10e10
    guide_list = [args.guide_w] if args.guide_w >= 0 else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 4.5, 5.0]
    for guide_w in guide_list:
        mse = evaluate(
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
else:
    evaluate(
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
