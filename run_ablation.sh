#!/bin/bash
# R3-DiffTS Ablation Experiments
# 6 experiments + 1 control across 3 representative datasets
# Usage: bash run_ablation.sh
#   Or run specific experiment: bash run_ablation.sh e2
#   Or specific dataset:       bash run_ablation.sh e2 economy

set -e

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
EXP_LABELS="E0:Baseline E1:+R E2:+R+D E3:Full(R+D+S) E4:R+S(w/o_D) E5:D+S(w/o_R) E2R:+R+D_random"

# ─── Run function ───
run_exp() {
    local ds=$1 exp=$2
    local seq=${DS_SEQ[$ds]} pred=${DS_PRED[$ds]} freq=${DS_FREQ[$ds]}
    local text=${DS_TEXT[$ds]} data=${DS_PATH[$ds]}
    local config="${ds}_${seq}_${pred}_abl_${exp}.yaml"

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
    DATASETS="${DS_FILTER}"
fi
if [ "$EXP_FILTER" != "all" ]; then
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
