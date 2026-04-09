import argparse
import copy
import csv
import json
import subprocess
import sys
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
SAVE_DIR = REPO_ROOT / "save"
LOG_DIR = REPO_ROOT / "logs"


CASE_PRESETS = OrderedDict(
    [
        (
            "no_multires",
            {
                "description": "Disable multi-resolution auxiliary loss; keep the tuned RAG setup.",
                "updates": {
                    "train": {
                        "multi_res_loss_weight": 0.0,
                        "multi_res_partition_mode": "cumulative",
                        "multi_res_use_scale_router": False,
                    },
                    "model": {
                        "use_scale_router": False,
                    },
                },
            },
        ),
        (
            "cum_base",
            {
                "description": "Current tuned baseline: cumulative multi-res bins, no scale router.",
                "updates": {
                    "train": {
                        "multi_res_partition_mode": "cumulative",
                        "multi_res_use_scale_router": False,
                    },
                    "model": {
                        "use_scale_router": False,
                    },
                },
            },
        ),
        (
            "disjoint_only",
            {
                "description": "Switch multi-res supervision from cumulative prefixes to disjoint horizon bins.",
                "updates": {
                    "train": {
                        "multi_res_partition_mode": "disjoint",
                        "multi_res_use_scale_router": False,
                    },
                    "model": {
                        "use_scale_router": False,
                    },
                },
            },
        ),
        (
            "router_window_only",
            {
                "description": "Enable heuristic scale routing for dynamic text windows only.",
                "updates": {
                    "train": {
                        "multi_res_partition_mode": "disjoint",
                        "multi_res_use_scale_router": False,
                        "scale_route_horizons": [1, 3, 6, 12],
                    },
                    "model": {
                        "use_scale_router": True,
                        "scale_window_candidates": [9, 18, 27, 36],
                        "scale_route_temperature": 0.20,
                    },
                },
            },
        ),
        (
            "router_loss_only",
            {
                "description": "Use scale-router-weighted disjoint loss while fixing text windows to the max length.",
                "updates": {
                    "train": {
                        "multi_res_partition_mode": "disjoint",
                        "multi_res_use_scale_router": True,
                        "scale_route_horizons": [1, 3, 6, 12],
                    },
                    "model": {
                        "use_scale_router": True,
                        "scale_window_candidates": [36, 36, 36, 36],
                        "scale_route_temperature": 0.20,
                    },
                },
            },
        ),
        (
            "router_full",
            {
                "description": "Full weekly integration: disjoint multi-res bins plus routed text windows and routed loss weights.",
                "updates": {
                    "train": {
                        "multi_res_partition_mode": "disjoint",
                        "multi_res_use_scale_router": True,
                        "scale_route_horizons": [1, 3, 6, 12],
                    },
                    "model": {
                        "use_scale_router": True,
                        "scale_window_candidates": [9, 18, 27, 36],
                        "scale_route_temperature": 0.20,
                    },
                },
            },
        ),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Economy scale-router ablations and write a summary manifest."
    )
    parser.add_argument(
        "--base-config",
        default="config/economy_36_12_multires_rag_tuned.yaml",
        help="Base YAML to clone for each ablation case.",
    )
    parser.add_argument("--root_path", default="../Time-MMD-main")
    parser.add_argument("--data_path", default="Economy/Economy.csv")
    parser.add_argument("--seq_len", type=int, default=36)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--text_len", type=int, default=36)
    parser.add_argument("--freq", default="m")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nsample", type=int, default=15)
    parser.add_argument("--valid_interval", type=int, default=1)
    parser.add_argument("--sample_steps_override", type=int, default=-1)
    parser.add_argument("--guide_w", type=float, default=-1.0)
    parser.add_argument("--epochs", type=int, default=-1, help="Optional override for all ablation runs.")
    parser.add_argument("--batch_size", type=int, default=-1, help="Optional override for all ablation runs.")
    parser.add_argument("--lr", type=float, default=-1.0, help="Optional override for all ablation runs.")
    parser.add_argument(
        "--cases",
        default=",".join(CASE_PRESETS.keys()),
        help=f"Comma-separated case list. Available: {', '.join(CASE_PRESETS.keys())}",
    )
    parser.add_argument(
        "--label",
        default="economy_scale_router_ablation",
        help="Prefix for temp configs and summary files.",
    )
    parser.add_argument(
        "--keep-temp-configs",
        action="store_true",
        help="Keep generated YAML files under config/ for inspection.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and configs without executing training.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: Path) -> None:
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def load_json_sequence(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text()
    decoder = json.JSONDecoder()
    idx = 0
    objs: List[Dict[str, Any]] = []
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        objs.append(obj)
        idx = end
    return objs


def merge_nested(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict):
            node = base.setdefault(key, {})
            if not isinstance(node, dict):
                node = {}
                base[key] = node
            merge_nested(node, value)
        else:
            base[key] = value
    return base


def list_economy_run_dirs() -> List[Path]:
    if not SAVE_DIR.exists():
        return []
    return sorted(
        [p for p in SAVE_DIR.glob("forecasting_Economy_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )


def detect_new_run_dir(before: Iterable[Path], after: Iterable[Path]) -> Path:
    before_set = {p.resolve() for p in before}
    candidates = [p for p in after if p.resolve() not in before_set]
    if not candidates:
        raise RuntimeError("No new Economy run directory was detected under save/.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config_results.json"
    objs = load_json_sequence(cfg_path)
    if not objs:
        raise RuntimeError(f"No JSON objects found in {cfg_path}")
    config = objs[0]
    metrics = [o for o in objs[1:] if isinstance(o, dict) and "MSE" in o and "MAE" in o]
    if not metrics:
        raise RuntimeError(f"No metric objects found in {cfg_path}")
    best_mse = min(metrics, key=lambda m: m["MSE"])
    best_mae = min(metrics, key=lambda m: m["MAE"])
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    return {
        "run_dir": run_dir.name,
        "best_mse": float(best_mse["MSE"]),
        "best_mse_gw": best_mse.get("guide_w"),
        "best_mae": float(best_mae["MAE"]),
        "best_mae_gw": best_mae.get("guide_w"),
        "use_scale_router": bool(model_cfg.get("use_scale_router", False)),
        "scale_window_candidates": model_cfg.get("scale_window_candidates", []),
        "multi_res_partition_mode": train_cfg.get("multi_res_partition_mode", "cumulative"),
        "multi_res_use_scale_router": bool(train_cfg.get("multi_res_use_scale_router", False)),
        "multi_res_loss_weight": train_cfg.get("multi_res_loss_weight", 0.0),
        "scale_route_horizons": train_cfg.get("scale_route_horizons", []),
    }


def build_case_config(base_cfg: Dict[str, Any], case_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg["model"]["lookback_len"] = args.seq_len
    cfg["model"]["pred_len"] = args.pred_len
    cfg["model"]["text_len"] = args.text_len
    cfg["model"]["domain"] = args.data_path.split("/")[0]
    if args.epochs > 0:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size > 0:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr > 0:
        cfg["train"]["lr"] = args.lr
    merge_nested(cfg, CASE_PRESETS[case_name]["updates"])
    return cfg


def write_summary(rows: List[Dict[str, Any]], label: str) -> Tuple[Path, Path]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest_id = uuid.uuid4().hex[:8]
    json_path = LOG_DIR / f"{label}_{manifest_id}.json"
    csv_path = LOG_DIR / f"{label}_{manifest_id}.csv"
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=True)
    fieldnames = [
        "case",
        "description",
        "run_dir",
        "best_mse",
        "best_mse_gw",
        "best_mae",
        "best_mae_gw",
        "use_scale_router",
        "scale_window_candidates",
        "multi_res_partition_mode",
        "multi_res_use_scale_router",
        "multi_res_loss_weight",
        "scale_route_horizons",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return json_path, csv_path


def main() -> None:
    args = parse_args()
    base_config_path = Path(args.base_config)
    if not base_config_path.is_absolute():
        base_config_path = REPO_ROOT / base_config_path
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    selected_cases = [case.strip() for case in args.cases.split(",") if case.strip()]
    invalid_cases = [case for case in selected_cases if case not in CASE_PRESETS]
    if invalid_cases:
        raise ValueError(f"Unknown ablation cases: {', '.join(invalid_cases)}")

    base_cfg = load_yaml(base_config_path)
    tmp_id = uuid.uuid4().hex[:8]
    rows: List[Dict[str, Any]] = []
    temp_paths: List[Path] = []

    try:
        for case_name in selected_cases:
            case_cfg = build_case_config(base_cfg, case_name, args)
            temp_cfg_path = CONFIG_DIR / f"_{args.label}_{case_name}_{tmp_id}.yaml"
            save_yaml(case_cfg, temp_cfg_path)
            temp_paths.append(temp_cfg_path)

            cmd = [
                sys.executable,
                "-u",
                "exe_forecasting.py",
                "--config",
                temp_cfg_path.name,
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
                "--guide_w",
                str(args.guide_w),
            ]
            if args.sample_steps_override > 0:
                cmd.extend(["--sample_steps_override", str(args.sample_steps_override)])

            print(f"\n=== {case_name} ===")
            print(CASE_PRESETS[case_name]["description"])
            print("Command:", " ".join(cmd))

            if args.dry_run:
                continue

            before = list_economy_run_dirs()
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
            after = list_economy_run_dirs()
            run_dir = detect_new_run_dir(before, after)
            summary = summarize_run(run_dir)
            row = {
                "case": case_name,
                "description": CASE_PRESETS[case_name]["description"],
                **summary,
            }
            rows.append(row)
            print(
                f"[{case_name}] run={row['run_dir']} "
                f"best_mse={row['best_mse']:.4f} (gw={row['best_mse_gw']}) "
                f"best_mae={row['best_mae']:.4f} (gw={row['best_mae_gw']})"
            )

        if args.dry_run:
            return

        json_path, csv_path = write_summary(rows, args.label)
        print("\n=== Summary ===")
        for row in rows:
            print(
                f"{row['case']}: "
                f"MSE={row['best_mse']:.4f} (gw={row['best_mse_gw']}), "
                f"MAE={row['best_mae']:.4f} (gw={row['best_mae_gw']}), "
                f"partition={row['multi_res_partition_mode']}, "
                f"scale_router={row['use_scale_router']}, "
                f"scale_loss={row['multi_res_use_scale_router']}"
            )
        print(f"Summary JSON: {json_path}")
        print(f"Summary CSV: {csv_path}")
    finally:
        if not args.keep_temp_configs:
            for path in temp_paths:
                if path.exists():
                    path.unlink()


if __name__ == "__main__":
    main()
