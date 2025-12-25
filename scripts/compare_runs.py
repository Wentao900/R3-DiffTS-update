import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any


def load_json_sequence(path: Path) -> List[Dict[str, Any]]:
    """Read a file that stores multiple JSON objects sequentially."""
    text = path.read_text()
    decoder = json.JSONDecoder()
    idx = 0
    objs = []
    while idx < len(text):
        # skip whitespace
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        objs.append(obj)
        idx = end
    return objs


def summarize_run(run_dir: Path) -> Tuple[str, Dict[str, Any]]:
    cfg_path = run_dir / "config_results.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found")

    objs = load_json_sequence(cfg_path)
    if not objs:
        raise ValueError(f"No JSON objects decoded from {cfg_path}")

    config = objs[0]
    metrics = [o for o in objs[1:] if "MSE" in o and "MAE" in o]
    if not metrics:
        raise ValueError(f"No metric objects found in {cfg_path}")

    best_mse = min(metrics, key=lambda m: m["MSE"])
    best_mae = min(metrics, key=lambda m: m["MAE"])

    return run_dir.name, {
        "domain": config["model"].get("domain"),
        "pred_len": config["model"].get("pred_len"),
        "lookback_len": config["model"].get("lookback_len"),
        "with_texts": config["model"].get("with_texts"),
        "use_rag_cot": config["model"].get("use_rag_cot"),
        "cot_only": config["model"].get("cot_only"),
        "best_mse": best_mse,
        "best_mae": best_mae,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare TAA vs baseline runs via config_results.json."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="Paths to run directories (e.g., save/forecasting_Traffic_20251205_180256)",
    )
    args = parser.parse_args()

    rows = []
    for run in args.runs:
        rows.append(summarize_run(Path(run)))

    print("name | domain | lookback->pred | text | RAG/CoT | best_mse(guide_w) | best_mae(guide_w)")
    print("-" * 90)
    for name, info in rows:
        look_pred = f"{info['lookback_len']}->{info['pred_len']}"
        text_flag = "on" if info["with_texts"] else "off"
        rag_flag = "on" if info["use_rag_cot"] else "off"
        cot_flag = "cot_only" if info["cot_only"] else rag_flag
        best_mse = info["best_mse"]
        best_mae = info["best_mae"]
        print(
            f"{name} | {info['domain']} | {look_pred} | {text_flag} | {cot_flag} | "
            f"{best_mse['MSE']:.4f} (gw={best_mse.get('guide_w')}) | "
            f"{best_mae['MAE']:.4f} (gw={best_mae.get('guide_w')})"
        )


if __name__ == "__main__":
    main()
