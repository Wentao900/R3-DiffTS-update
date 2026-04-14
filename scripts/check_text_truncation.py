import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_provider.data_factory import data_provider  # noqa: E402


def _safe_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _snip(s: str, n: int = 220) -> str:
    s = str(s or "")
    s = " ".join(s.split())
    if len(s) <= n:
        return s
    return s[: n - 3].rstrip() + "..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to yaml config, e.g. config/economy_36_12.yaml")
    parser.add_argument("--root_path", type=str, default="Time-MMD-main")
    parser.add_argument("--data_path", type=str, default="Economy/Economy.csv")
    parser.add_argument("--seq_len", type=int, default=36)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--text_len", type=int, default=36)
    parser.add_argument("--max_text_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--freq", type=str, default="m")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--n_batches", type=int, default=2)
    parser.add_argument("--n_print", type=int, default=6)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Minimal args object expected by data_provider
    class A:
        pass

    a = A()
    a.data = "custom"
    a.root_path = args.root_path
    a.data_path = args.data_path
    a.seq_len = args.seq_len
    a.pred_len = args.pred_len
    a.text_len = args.text_len
    a.max_text_tokens = args.max_text_tokens
    a.text_drop_prob = 0.0
    a.use_rag_cot = bool(cfg.get("model", {}).get("use_rag_cot", False))
    a.cot_only = bool(cfg.get("model", {}).get("cot_only", False))
    a.rag_topk = _safe_int(cfg.get("model", {}).get("rag_topk", 3), 3)
    a.use_two_stage_rag = bool(cfg.get("model", {}).get("use_two_stage_rag", False))
    a.rag_stage1_topk = _safe_int(cfg.get("model", {}).get("rag_stage1_topk", -1), -1)
    a.rag_stage2_topk = _safe_int(cfg.get("model", {}).get("rag_stage2_topk", -1), -1)
    a.two_stage_gate = bool(cfg.get("model", {}).get("two_stage_gate", True))
    a.trend_slope_eps = float(cfg.get("model", {}).get("trend_slope_eps", 1e-3))
    a.cot_model = cfg.get("model", {}).get("cot_model", None)
    a.cot_max_new_tokens = _safe_int(cfg.get("model", {}).get("cot_max_new_tokens", 96), 96)
    a.cot_temperature = float(cfg.get("model", {}).get("cot_temperature", 0.7))
    a.cot_cache_size = _safe_int(cfg.get("model", {}).get("cot_cache_size", 1024), 1024)
    a.cot_device = cfg.get("model", {}).get("cot_device", None)
    a.cot_load_in_8bit = bool(cfg.get("model", {}).get("cot_load_in_8bit", False))
    a.cot_load_in_4bit = bool(cfg.get("model", {}).get("cot_load_in_4bit", False))
    a.trend_cfg = bool(cfg.get("diffusion", {}).get("trend_cfg", False))
    a.features = cfg.get("data", {}).get("features", "S") if isinstance(cfg.get("data", {}), dict) else "S"
    a.target = cfg.get("data", {}).get("target", "OT") if isinstance(cfg.get("data", {}), dict) else "OT"
    a.freq = args.freq
    a.embed = args.embed
    a.batch_size = args.batch_size
    a.num_workers = args.num_workers
    a.llm = cfg.get("model", {}).get("llm", "bert")

    # Build loader (train split is enough for truncation sanity)
    _, loader = data_provider(a, flag="train")

    # Load tokenizer to compute actual token length (same as dataset's intended llm tokenizer)
    llm = str(a.llm).lower().strip()
    if llm == "bert":
        tok = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
        max_len = 512
    elif llm == "gpt2":
        tok = AutoTokenizer.from_pretrained("openai-community/gpt2", local_files_only=True)
        max_len = 1024
    elif llm == "llama":
        tok = AutoTokenizer.from_pretrained("huggyllama/llama-7b", local_files_only=True)
        max_len = 2048
    else:
        tok = None
        max_len = None

    print(f"[check] llm={a.llm} max_text_tokens(ws)={a.max_text_tokens} tokenizer_max_len={max_len}")

    printed = 0
    for b_idx, batch in enumerate(loader):
        texts = batch.get("texts", None)
        if texts is None:
            continue
        # texts is a list of strings after collate
        for i, t in enumerate(texts):
            ws_len = len(str(t).split())
            tok_len = None
            over = None
            if tok is not None:
                ids = tok(str(t), add_special_tokens=True, truncation=False)["input_ids"]
                tok_len = len(ids)
                over = tok_len > int(max_len)
            print(f"\n[b{b_idx} i{i}] ws_len={ws_len} tok_len={tok_len} over_max={over}")
            print(" head:", _snip(str(t)[:600], 240))
            print(" tail:", _snip(str(t)[-900:], 240))
            printed += 1
            if printed >= args.n_print:
                return
        if b_idx + 1 >= args.n_batches:
            break


if __name__ == "__main__":
    main()

