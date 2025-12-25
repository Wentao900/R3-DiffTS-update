import argparse
import uuid
from pathlib import Path

from run_taa_vs_base import load_cfg, save_cfg, build_cfgs, run_job


def main():
    parser = argparse.ArgumentParser(
        description="单次运行基线或 TAA（文本+时间戳）版本，可选启用 RAG/CoT。"
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
    parser.add_argument(
        "--mode",
        choices=["baseline", "taa", "both"],
        default="taa",
        help="baseline: 纯数值; taa: 文本+时间戳; both: 先跑基线再跑 TAA",
    )
    parser.add_argument(
        "--ttf",
        action="store_true",
        help="在 TAA 运行中启用 timestep_branch（TTF 分支）。不加则仅用 TAA 融合时间、时间戳和文本。",
    )
    parser.add_argument(
        "--text_ttf_only",
        action="store_true",
        help="文本只在 TTF 阶段使用，主干 TAA 不融合文本（对照实验：TAA=时间+时间戳，文本通过 TTF 融合）。",
    )
    parser.add_argument(
        "--no_texts",
        action="store_true",
        help="关闭文本分支，仅融合时间与时间戳（TAA 无文本）。",
    )
    parser.add_argument("--use_rag_cot", action="store_true", help="TAA 时启用 RAG/CoT")
    parser.add_argument("--cot_only", action="store_true", help="TAA 仅使用 CoT，不检索")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    base_cfg = load_cfg(cfg_path)

    base_cfg_mod, taa_cfg_mod = build_cfgs(
        base_cfg=base_cfg,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        text_len=args.text_len,
        use_rag_cot=args.use_rag_cot or args.cot_only,
    )

    if args.cot_only:
        taa_cfg_mod["model"]["cot_only"] = True
        taa_cfg_mod["model"]["use_rag_cot"] = True
        taa_cfg_mod["model"]["rag_topk"] = 0

    # 文本分支控制：默认开；--no_texts 关闭
    if args.no_texts:
        taa_cfg_mod["model"]["with_texts"] = False
        taa_cfg_mod["model"]["text_in_taa"] = False
        taa_cfg_mod["model"]["text_for_ttf"] = False
    elif args.text_ttf_only:
        # 文本不进 TAA，只在 TTF 使用
        taa_cfg_mod["model"]["with_texts"] = True
        taa_cfg_mod["model"]["text_in_taa"] = False
        taa_cfg_mod["model"]["text_for_ttf"] = True
    else:
        # 默认：文本进 TAA，也可供 TTF
        taa_cfg_mod["model"]["text_in_taa"] = taa_cfg_mod["model"].get("text_in_taa", True)
        taa_cfg_mod["model"]["text_for_ttf"] = taa_cfg_mod["model"].get("text_for_ttf", True)

    # TTF 控制：默认 False；若需要 TAA+TTF，就加 --ttf
    taa_cfg_mod["model"]["timestep_branch"] = bool(args.ttf)

    tmp_id = uuid.uuid4().hex[:8]
    cfg_dir = cfg_path.parent
    base_tmp = cfg_dir / f"{cfg_path.stem}_base_{tmp_id}.yaml"
    taa_tmp = cfg_dir / f"{cfg_path.stem}_taa_{tmp_id}.yaml"

    try:
        if args.mode in ("baseline", "both"):
            save_cfg(base_cfg_mod, base_tmp)
            run_job("BASE", base_tmp, args)

        if args.mode in ("taa", "both"):
            save_cfg(taa_cfg_mod, taa_tmp)
            run_job("TAA", taa_tmp, args)
    finally:
        for p in (base_tmp, taa_tmp):
            if p.exists():
                p.unlink()


if __name__ == "__main__":
    main()
