import argparse
import json
from json import JSONDecoder
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_forecasting import get_dataloader
from main_model import CSDI_Forecasting
from utils.utils import evaluate


def load_first_json_object(path: Path) -> dict:
    """
    config_results.json stores multiple JSON objects sequentially.
    The first object is the config; subsequent ones are metric dicts.
    """
    text = path.read_text()
    dec = JSONDecoder()
    idx = 0
    while idx < len(text) and text[idx].isspace():
        idx += 1
    obj, _ = dec.raw_decode(text, idx)
    if not isinstance(obj, dict):
        raise ValueError(f"First JSON object in {path} is not a dict.")
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Existing run directory under save/, e.g. save/forecasting_Health_US_20260414_153010",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--datatype", type=str, default="multimodal")
    parser.add_argument("--root_path", type=str, default="/root/autodl-tmp/Time-MMD-main")
    parser.add_argument("--data", type=str, default="custom")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--freq", type=str, default="w")
    parser.add_argument("--features", type=str, default="S")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--nsample", type=int, default=20)
    parser.add_argument("--guide_w", type=float, default=0.4)
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="Only compute and write text_reliability stats from the dataloader (no diffusion sampling).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"{run_dir} not found")
    cfg_path = run_dir / "config_results.json"
    model_path = run_dir / "model.pth"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found")
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found")

    config = load_first_json_object(cfg_path)
    if "model" not in config:
        raise ValueError("config_results.json first object missing 'model' section")

    # Build an args-like object expected by get_dataloader/data_provider.
    class A:
        pass

    a = A()
    a.config = ""
    a.datatype = args.datatype
    a.device = args.device
    a.seed = 2025
    a.unconditional = False
    a.modelfolder = ""
    a.nsample = args.nsample
    a.data = args.data
    a.embed = args.embed
    a.root_path = args.root_path
    a.data_path = f"{config['model'].get('domain')}/{config['model'].get('domain')}.csv"
    # Fix up Health_US naming to match Time-MMD folder layout.
    domain = str(config["model"].get("domain"))
    if domain and domain.endswith("_US"):
        a.data_path = "Health_US/Health_US.csv"
    a.seq_len = int(config["model"].get("lookback_len", 96))
    a.pred_len = int(config["model"].get("pred_len", 12))
    a.text_len = int(config["model"].get("text_len", a.seq_len))
    a.max_text_tokens = 256
    a.text_drop_prob = 0.0

    # RAG/CoT args consumed by data_provider
    a.use_rag_cot = bool(config["model"].get("use_rag_cot", False))
    a.cot_only = bool(config["model"].get("cot_only", False))
    a.rag_topk = int(config["model"].get("rag_topk", 3))
    a.use_two_stage_rag = bool(config["model"].get("use_two_stage_rag", False))
    a.rag_stage1_topk = int(config["model"].get("rag_stage1_topk", -1))
    a.rag_stage2_topk = int(config["model"].get("rag_stage2_topk", -1))
    a.two_stage_gate = bool(config["model"].get("two_stage_gate", True))
    a.trend_slope_eps = float(config["model"].get("trend_slope_eps", 1e-3))
    a.cot_model = config["model"].get("cot_model", None)
    a.cot_max_new_tokens = int(config["model"].get("cot_max_new_tokens", 96))
    a.cot_temperature = float(config["model"].get("cot_temperature", 0.7))
    a.cot_cache_size = int(config["model"].get("cot_cache_size", 1024))
    a.cot_device = config["model"].get("cot_device", None)
    a.cot_load_in_8bit = bool(config["model"].get("cot_load_in_8bit", False))
    a.cot_load_in_4bit = bool(config["model"].get("cot_load_in_4bit", False))
    a.guide_w = args.guide_w
    a.trend_cfg = bool(config.get("diffusion", {}).get("trend_cfg", False))
    a.trend_cfg_power = float(config.get("diffusion", {}).get("trend_cfg_power", 1.0))
    a.trend_cfg_random = bool(config.get("diffusion", {}).get("trend_cfg_random", False))
    a.trend_strength_scale = float(config.get("diffusion", {}).get("trend_strength_scale", 1.0))
    a.trend_volatility_scale = float(config.get("diffusion", {}).get("trend_volatility_scale", 1.0))
    a.trend_time_floor = float(config.get("diffusion", {}).get("trend_time_floor", 0.0))
    a.save_trend_prior = True
    a.features = args.features
    a.freq = args.freq
    a.target = args.target
    a.num_workers = args.num_workers
    a.dropout = 0.0
    a.attn_drop = 0.0
    a.init = "None"
    a.valid_interval = 1
    a.time_weight = float(config.get("diffusion", {}).get("time_weight", 0.1))
    a.c_mask_prob = -1
    a.beta_end = -1
    a.lr = -1
    a.sample_steps_override = -1
    a.save_attn = False
    a.save_token = False
    a.llm = config["model"].get("llm", "bert")
    a.batch_size = int(args.batch_size)

    # Rebuild loaders and scalers
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        datatype=args.datatype,
        device=args.device,
        batch_size=a.batch_size,
        args=a,
    )

    foldername = str(run_dir)
    if not foldername.endswith("/"):
        foldername += "/"

    if args.stats_only:
        # Fast path: compute distribution stats directly from batches.
        import numpy as np

        rels = []
        marks = []
        for batch in test_loader:
            if isinstance(batch, dict) and "text_reliability" in batch:
                rels.append(batch["text_reliability"].detach().cpu().numpy().reshape(-1))
            if isinstance(batch, dict) and "text_mark" in batch:
                marks.append(batch["text_mark"].detach().cpu().numpy().reshape(-1))
        if rels:
            rel = np.concatenate(rels, axis=0).reshape(-1)
        else:
            rel = np.zeros((0,), dtype=np.float32)
        if marks:
            mk = np.concatenate(marks, axis=0).reshape(-1)
        else:
            mk = None

        qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        summ = {
            "count": int(rel.shape[0]),
            "mean": float(np.mean(rel)) if rel.size else 0.0,
            "std": float(np.std(rel)) if rel.size else 0.0,
            "min": float(np.min(rel)) if rel.size else 0.0,
            "max": float(np.max(rel)) if rel.size else 0.0,
            "quantiles": {str(q): float(np.quantile(rel, q)) for q in qs} if rel.size else {},
        }
        if mk is not None and mk.size == rel.size:
            has = mk > 0
            summ["text_present"] = {
                "count": int(has.sum()),
                "mean": float(np.mean(rel[has])) if has.any() else 0.0,
            }
            summ["text_absent"] = {
                "count": int((~has).sum()),
                "mean": float(np.mean(rel[~has])) if (~has).any() else 0.0,
            }

        out_path = run_dir / "trend_text_reliability_summary.json"
        out_path.write_text(json.dumps(summ, indent=2))
        print("\n[ok] wrote", out_path)
        print(out_path.read_text())
        return

    model = CSDI_Forecasting(config, args.device, target_dim=1, window_lens=[a.seq_len, a.pred_len]).to(args.device)
    if str(args.device).startswith("cuda"):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Slow path: run full diffusion evaluation (also writes stats).
    evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
        window_lens=[a.seq_len, a.pred_len],
        guide_w=float(args.guide_w),
        save_attn=False,
        save_token=False,
        save_trend_prior=True,
    )

    summary = run_dir / "trend_text_reliability_summary.json"
    if summary.exists():
        print("\n[ok] wrote", summary)
        print(summary.read_text())
    else:
        print("\n[warn] summary not written (missing trend_prior/text_reliability in batches?)")


if __name__ == "__main__":
    main()

