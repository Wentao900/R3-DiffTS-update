#!/usr/bin/env python3
"""Pick a guidance weight on the validation split of this repository.

MCD-TSF's own validation-set sweep stays in that external fork. This script
only calls this repository's evaluator. Pass the caller's grid with
--guide_list. The selection split is the validation loader, which
exe_forecasting.py already uses when a validation set exists.

Example:
  python3 scripts/select_guide_w.py \\
    --config xattn/agriculture_36_12_xattn.yaml \\
    --modelfolder agriculture_36_12_xattn_seed2025 \\
    --guide_list 0,0.5,1,1.5,2 \\
    --data_path Agriculture/Agriculture.csv \\
    --seq_len 36 --pred_len 12 --freq m --text_len 36
"""

import argparse
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Select guide_w on the validation split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--modelfolder", required=True)
    parser.add_argument("--guide_list", required=True, help="Comma-separated grid. The caller supplies the values.")
    parser.add_argument("--select_split", default="valid", choices=["valid"])
    parser.add_argument("forwarded", nargs=argparse.REMAINDER, help="Remaining exe_forecasting.py arguments, without a leading --.")
    args = parser.parse_args()
    command = [
        sys.executable,
        "-u",
        "exe_forecasting.py",
        "--config",
        args.config,
        "--modelfolder",
        args.modelfolder,
        "--guide_w",
        "-1",
        "--guide_list",
        args.guide_list,
    ]
    extra = list(args.forwarded)
    if extra and extra[0] == "--":
        extra = extra[1:]
    command.extend(extra)
    print("select_split", args.select_split)
    print("command", " ".join(shlex.quote(part) for part in command))
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
