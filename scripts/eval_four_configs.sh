#!/usr/bin/env bash
# Evaluate four configs on an existing checkpoint (no retraining):
# baseline / full (RAG+CoT) / CoT-only / RAG-only.
# Usage:
#   # preferred: pin to a specific folder
#   MODEL_FOLDER=forecasting_Traffic_xxx bash scripts/eval_four_configs.sh
#   # optional: if you really want auto-pick latest folder that has model.pth
#   USE_LATEST=1 bash scripts/eval_four_configs.sh
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
DATA_PATH=${DATA_PATH:-Traffic/Traffic.csv}
CONFIG=${CONFIG:-traffic_36_12.yaml}
SEQ_LEN=${SEQ_LEN:-36}
PRED_LEN=${PRED_LEN:-12}
TEXT_LEN=${TEXT_LEN:-36}
FREQ=${FREQ:-m}
NSAMPLE=${NSAMPLE:-5}
SAMPLE_STEPS=${SAMPLE_STEPS:-120}
GUIDE_SWEEP=${GUIDE_SWEEP:-"0.9 1.0"}  # quick sweep; override if you want denser scan
COT_MODEL=${COT_MODEL:-./Qwen2.5-7B-Instruct}
COT_TEMPERATURE=${COT_TEMPERATURE:-0.55}
COT_MAX_NEW_TOKENS=${COT_MAX_NEW_TOKENS:-96}
# RAG settings per config (let them differ per experiment)
RAG_TOPK_FULL=${RAG_TOPK_FULL:-1}
RAG_TOPK_COTONLY=${RAG_TOPK_COTONLY:-0}
RAG_TOPK_RAGONLY=${RAG_TOPK_RAGONLY:-1}
LABEL=${LABEL:-fast_eval_four}

# choose checkpoint: explicit MODEL_FOLDER required unless USE_LATEST=1 is set
if [[ -n "${MODEL_FOLDER:-}" ]]; then
  MODEL_FOLDER="${MODEL_FOLDER}"
elif [[ "${USE_LATEST:-0}" == "1" ]]; then
  MODEL_FOLDER=$(find save -maxdepth 2 -type f -name model.pth -printf '%h\n' | xargs -n1 basename | sort -r | head -n1 || true)
  if [[ -z "${MODEL_FOLDER}" ]]; then
    echo "No checkpoint found under save/ (model.pth missing). Please train first or set MODEL_FOLDER explicitly." >&2
    exit 1
  fi
else
  echo "Please set MODEL_FOLDER to a directory under save/. (To auto-pick latest, export USE_LATEST=1.)" >&2
  exit 1
fi

echo "Using checkpoint: ${MODEL_FOLDER}"
CONFIG_NAME=$(basename "${CONFIG}")

run_eval() {
  local label=$1
  shift
  for gw in ${GUIDE_SWEEP}; do
    log_file="logs/${LABEL}_${label}_gw${gw}.log"
    echo "-- ${label}, gw=${gw} --"
    python -u exe_forecasting.py \
      --modelfolder "${MODEL_FOLDER}" \
      --root_path "${ROOT_PATH}" \
      --data_path "${DATA_PATH}" \
      --config "${CONFIG_NAME}" \
      --seq_len "${SEQ_LEN}" \
      --pred_len "${PRED_LEN}" \
      --text_len "${TEXT_LEN}" \
      --freq "${FREQ}" \
      --nsample "${NSAMPLE}" \
      --sample_steps_override "${SAMPLE_STEPS}" \
      --guide_w "${gw}" \
      "$@" | tee "${log_file}"
  done
}

mkdir -p logs

echo "== Baseline (no text/RAG) =="
run_eval baseline

echo "== Full (text + RAG+CoT) =="
run_eval full --use_rag_cot --rag_topk "${RAG_TOPK_FULL}" --cot_model "${COT_MODEL}" --cot_temperature "${COT_TEMPERATURE}" --cot_max_new_tokens "${COT_MAX_NEW_TOKENS}"

echo "== CoT-only (text + CoT, no RAG) =="
run_eval cot_only --use_rag_cot --cot_only --rag_topk "${RAG_TOPK_COTONLY}" --cot_model "${COT_MODEL}" --cot_temperature "${COT_TEMPERATURE}" --cot_max_new_tokens "${COT_MAX_NEW_TOKENS}"

echo "== RAG-only (text + RAG, no CoT) =="
run_eval rag_only --use_rag_cot --cot_only --rag_topk "${RAG_TOPK_RAGONLY}" --cot_model "" --cot_temperature 0.0

echo "Done. Logs in logs/${LABEL}_*.log"
