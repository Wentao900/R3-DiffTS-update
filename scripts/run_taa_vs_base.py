import argparse
import subprocess
import tempfile
import uuid
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple


def load_cfg(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def save_cfg(cfg: Dict[str, Any], path: Path) -> None:
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def build_cfgs(
    base_cfg: Dict[str, Any],
    seq_len: int,
    pred_len: int,
    text_len: int,
    use_rag_cot: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (baseline_cfg, taa_cfg)."""
    def common_updates(cfg: Dict[str, Any]) -> Dict[str, Any]:
        cfg = yaml.safe_load(yaml.safe_dump(cfg))  # deep copy
        cfg["model"]["lookback_len"] = seq_len
        cfg["model"]["pred_len"] = pred_len
        cfg["model"]["text_len"] = text_len
        return cfg

    # baseline: 禁用文本和 TAA 拼接分支
    base = common_updates(base_cfg)
    base["model"]["with_texts"] = False
    base["model"]["timestep_emb_cat"] = False
    base["model"]["timestep_branch"] = False
    base["model"]["use_rag_cot"] = False
    base["model"]["cot_only"] = False
    base["model"]["rag_topk"] = 0

    # TAA: 启用文本和时间戳拼接
    taa = common_updates(base_cfg)
    taa["model"]["with_texts"] = True
    taa["model"]["timestep_emb_cat"] = True
    taa["model"]["use_rag_cot"] = use_rag_cot
    if not use_rag_cot:
        taa["model"]["cot_only"] = False
        taa["model"]["rag_topk"] = taa["model"].get("rag_topk", 1)

    return base, taa


def run_job(name: str, cfg_path: Path, args) -> None:
    cmd = [
        "python",
        "-u",
        "exe_forecasting.py",
        "--config",
        str(cfg_path.name),
        "--root_path",
        args.root_path,
        "--data_path",
        args.data_path,
        "--seq_len",
        str(args.seq_len),
        "--pred_len",
        str(args.pred_len),
        "--text_len",
        str(args.text_len),
        "--freq",
        args.freq,
        "--device",
        args.device,
        "--nsample",
        str(args.nsample),
        "--valid_interval",
        str(args.valid_interval),
    ]
    if args.sample_steps_override > 0:
        cmd.extend(["--sample_steps_override", str(args.sample_steps_override)])
    print(f"[{name}] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run TAA (文本+时间戳融合) vs baseline (纯数值) 训练对比脚本"
    )
    parser.add_argument("--config", required=True, help="基础配置文件路径，如 config/traffic_36_12.yaml")
    parser.add_argument("--root_path", required=True, help="数据根目录，如 ../Time-MMD-main")
    parser.add_argument("--data_path", required=True, help="数据文件相对路径，如 Traffic/Traffic.csv")
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--text_len", type=int, default=36)
    parser.add_argument("--freq", type=str, default="m")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--nsample", type=int, default=5)
    parser.add_argument("--valid_interval", type=int, default=1)
    parser.add_argument("--sample_steps_override", type=int, default=-1)
    parser.add_argument("--use_rag_cot", action="store_true", help="在 TAA 版本中启用 RAG/CoT")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    base_cfg = load_cfg(cfg_path)

    base_cfg_mod, taa_cfg_mod = build_cfgs(
        base_cfg=base_cfg,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        text_len=args.text_len,
        use_rag_cot=args.use_rag_cot,
    )

    tmp_id = uuid.uuid4().hex[:8]
    cfg_dir = cfg_path.parent
    base_tmp = cfg_dir / f"{cfg_path.stem}_base_{tmp_id}.yaml"
    taa_tmp = cfg_dir / f"{cfg_path.stem}_taa_{tmp_id}.yaml"

    save_cfg(base_cfg_mod, base_tmp)
    save_cfg(taa_cfg_mod, taa_tmp)

    try:
        run_job("BASE", base_tmp, args)
        run_job("TAA", taa_tmp, args)
    finally:
        # 清理临时配置
        for p in [base_tmp, taa_tmp]:
            if p.exists():
                p.unlink()


if __name__ == "__main__":
    main()
