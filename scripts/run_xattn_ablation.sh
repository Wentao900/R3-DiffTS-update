#!/usr/bin/env bash
# One variant on the unified config.
# With no arguments this runs A1 on all 8 datasets and the 3 protocol seeds:
#   bash scripts/run_xattn_ablation.sh
#   bash scripts/run_xattn_ablation.sh E7_cyclic_shift
#   DOMAIN=health ABLATION=T0_numeric_only bash scripts/run_xattn_ablation.sh
# Seeds default to config/xattn/protocol_seeds.yaml (2024, 2025, 2026).
# Set SEEDS=2024 to run one seed. DRY_RUN=1 prints the commands.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -ge 1 ]]; then
  ABLATION="$1"
fi
export DOMAIN="${DOMAIN:-all}"
export ABLATION="${ABLATION:-A1}"
export SEEDS="${SEEDS-}"
export DRY_RUN="${DRY_RUN:-0}"
export ROOT_PATH="${ROOT_PATH:-../Time-MMD-main}"
export DEVICE="${DEVICE:-cpu}"
export GUIDE_LIST="${GUIDE_LIST:-0,0.5,1.0,1.5,2.0}"
export NSAMPLE="${NSAMPLE-}"
python3 - << 'PY'
import os
import shlex
import subprocess

from utils.xattn_config import plan_unified_runs

runs = plan_unified_runs(
    domain=os.environ.get("DOMAIN", "all"),
    ablation=os.environ.get("ABLATION", "A1"),
    seeds=os.environ.get("SEEDS", ""),
    root_path=os.environ.get("ROOT_PATH", "../Time-MMD-main"),
    device=os.environ.get("DEVICE", "cpu"),
    guide_list=os.environ.get("GUIDE_LIST", "0,0.5,1.0,1.5,2.0"),
    nsample=os.environ.get("NSAMPLE") or None,
)
dry = os.environ.get("DRY_RUN", "0") == "1"
for argv in runs:
    print(" ".join(shlex.quote(part) for part in argv))
    if not dry:
        subprocess.check_call(argv)
PY
