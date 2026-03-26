#!/bin/bash
# R3-DiffTS Ablation Experiments
# 6 experiments + 1 control across 3 representative datasets
# Usage: bash run_ablation.sh
#   Or run specific experiment: bash run_ablation.sh e2
#   Or specific dataset:       bash run_ablation.sh e2 economy

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ROOT_PATH="${ROOT_PATH:-../Time-MMD-main}"
EXP_FILTER="${1:-all}"    # e0/e1/e2/e3/e4/e5/e2r/all
DS_FILTER="${2:-all}"     # economy/traffic/energy/all

# ─── Dataset definitions ───
declare -A DS_PATH DS_SEQ DS_PRED DS_FREQ DS_TEXT
DS_PATH[economy]="Economy/Economy.csv";       DS_SEQ[economy]=36;  DS_PRED[economy]=12; DS_FREQ[economy]=m;  DS_TEXT[economy]=36
DS_PATH[traffic]="Traffic/Traffic.csv";       DS_SEQ[traffic]=36;  DS_PRED[traffic]=12; DS_FREQ[traffic]=m;  DS_TEXT[traffic]=36
DS_PATH[energy]="Energy/Energy.csv";          DS_SEQ[energy]=96;   DS_PRED[energy]=24;  DS_FREQ[energy]=w;   DS_TEXT[energy]=36

# ─── Experiment IDs ───
ALL_EXPS="e0 e1 e2 e3 e4 e5 e2r"

is_valid_dataset() {
    local ds=$1
    [[ "$ds" == "economy" || "$ds" == "traffic" || "$ds" == "energy" ]]
}

is_valid_exp() {
    local exp=$1
    [[ "$exp" == "e0" || "$exp" == "e1" || "$exp" == "e2" || "$exp" == "e3" || "$exp" == "e4" || "$exp" == "e5" || "$exp" == "e2r" ]]
}

# ─── Run function ───
run_exp() {
    local ds=$1 exp=$2
    local seq=${DS_SEQ[$ds]} pred=${DS_PRED[$ds]} freq=${DS_FREQ[$ds]}
    local text=${DS_TEXT[$ds]} data=${DS_PATH[$ds]}
    local config="${ds}_${seq}_${pred}_abl_${exp}.yaml"

    if [[ ! -f "config/${config}" ]]; then
        echo "Missing config: config/${config}" >&2
        exit 1
    fi

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  [${exp^^}] ${ds} (seq=${seq}, pred=${pred})"
    echo "  Config: ${config}"
    echo "════════════════════════════════════════════════════"

    python -u exe_forecasting.py \
        --root_path "${ROOT_PATH}" \
        --data_path "${data}" \
        --config "${config}" \
        --seq_len ${seq} \
        --pred_len ${pred} \
        --text_len ${text} \
        --freq ${freq}
}

# ─── Main loop ───
echo "╔══════════════════════════════════════════════════╗"
echo "║     R3-DiffTS Ablation Experiments               ║"
echo "║     Experiments: ${EXP_FILTER}                   ║"
echo "║     Datasets:    ${DS_FILTER}                    ║"
echo "╚══════════════════════════════════════════════════╝"

DATASETS="economy traffic energy"
EXPS="${ALL_EXPS}"

if [ "$DS_FILTER" != "all" ]; then
    if ! is_valid_dataset "$DS_FILTER"; then
        echo "Invalid dataset filter: ${DS_FILTER}" >&2
        echo "Expected one of: economy traffic energy all" >&2
        exit 1
    fi
    DATASETS="${DS_FILTER}"
fi
if [ "$EXP_FILTER" != "all" ]; then
    if ! is_valid_exp "$EXP_FILTER"; then
        echo "Invalid experiment filter: ${EXP_FILTER}" >&2
        echo "Expected one of: e0 e1 e2 e3 e4 e5 e2r all" >&2
        exit 1
    fi
    EXPS="${EXP_FILTER}"
fi

for ds in ${DATASETS}; do
    for exp in ${EXPS}; do
        run_exp "$ds" "$exp"
    done
done

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     All ablation experiments completed!          ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Results saved in save/ directory."
echo "Check config_results.json in each run folder for MSE/MAE."
