"""CPU smoke for rule and learned text-validity modes.

Builds a tiny Time-MMD-shaped folder and runs exe_forecasting.py for one
training batch plus evaluation. BERT is the local stub encoder.
"""

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def write_synthetic_dataset(root: Path) -> None:
    num_dir = root / "numerical" / "Economy"
    text_dir = root / "textual" / "Economy"
    num_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    rows = 48
    dates = pd.date_range("2018-01-01", periods=rows, freq="MS")
    values = np.linspace(1.0, 3.0, rows) + 0.1 * np.sin(np.arange(rows))
    cov = np.linspace(10.0, 0.0, rows)
    frame = pd.DataFrame(
        {
            "date": dates,
            "start_date": dates,
            "end_date": dates + pd.offsets.MonthEnd(0),
            "OT": values,
            "COV": cov,
        }
    )
    frame.to_csv(num_dir / "Economy.csv", index=False)
    reports = []
    search = []
    for idx, end in enumerate(frame["end_date"]):
        start = frame["start_date"].iloc[idx]
        direction = "down" if idx % 2 == 0 else "up"
        reports.append(
            {
                "start_date": start,
                "end_date": end,
                "fact": f"Monthly report {idx}: prices move {direction} after rainfall and supply comments.",
            }
        )
        search.append(
            {
                "start_date": start,
                "end_date": end,
                "fact": f"Retrieved note {idx}: exports {direction}.",
            }
        )
    pd.DataFrame(reports).to_csv(text_dir / "Economy_report.csv", index=False)
    pd.DataFrame(search).to_csv(text_dir / "Economy_search.csv", index=False)


def run_mode(root: Path, config: str, work: Path) -> Path:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONHASHSEED"] = "0"
    command = [
        sys.executable,
        str(ROOT / "exe_forecasting.py"),
        "--config",
        config,
        "--data_path",
        "Economy/Economy.csv",
        "--seq_len",
        "8",
        "--pred_len",
        "2",
        "--text_len",
        "8",
        "--freq",
        "m",
        "--root_path",
        str(root),
        "--device",
        "cpu",
        "--nsample",
        "1",
        "--num_workers",
        "0",
        "--valid_interval",
        "1",
        "--guide_w",
        "1.0",
        "--seed",
        "2025",
    ]
    print("running", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=str(ROOT), env=env, check=False, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"{config} failed with code {completed.returncode}")
    saves = sorted((ROOT / "save").glob("forecasting_Economy_*"))
    if not saves:
        raise SystemExit(f"{config} did not write a save folder")
    return saves[-1]


def assert_log(folder: Path, mode: str) -> None:
    path = folder / "text_validity_gates.csv"
    if not path.exists():
        raise SystemExit(f"missing gate log in {folder}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty gate log in {folder}")
    scales = [float(row["guidance_scale"]) for row in rows]
    if mode == "learned":
        gates = [float(row["sample_gate"]) for row in rows]
        if any(not (0.0 < gate < 1.0) for gate in gates):
            raise SystemExit(f"learned gates out of (0, 1): {gates}")
        if any(abs(gate - 0.05) < 1e-6 for gate in gates) and len(set(np.round(gates, 5))) == 1:
            raise SystemExit("learned gates collapsed to the old 0.05 floor")
    if any(scale < 0.0 or scale > 1.0 + 1e-5 for scale in scales):
        raise SystemExit(f"guidance scales outside [0, 1] for guide_w=1: {scales}")
    print(f"{mode}: {len(rows)} logged rows, guidance mean {float(np.mean(scales)):.4f}", flush=True)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="timemmd-smoke-") as tmp:
        root = Path(tmp)
        write_synthetic_dataset(root)
        rule_folder = run_mode(root, "text_validity/smoke_rule.yaml", root)
        assert_log(rule_folder, "rule")
        learned_folder = run_mode(root, "text_validity/smoke_learned.yaml", root)
        assert_log(learned_folder, "learned")
    print("smoke ok", flush=True)


if __name__ == "__main__":
    main()
