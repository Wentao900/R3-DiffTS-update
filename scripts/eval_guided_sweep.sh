#!/usr/bin/env bash
# Evaluate a trained checkpoint across a broad guide_w sweep (no retraining).
# Usage: MODEL_FOLDER=forecasting_Traffic_xxx bash scripts/eval_guided_sweep.sh
set -euo pipefail

MODEL_FOLDER=${MODEL_FOLDER:?set MODEL_FOLDER to a directory under save/}
ROOT_PATH=../Time-MMD-main
DATA_PATH=Traffic/Traffic.csv
CONFIG=config/traffic_36_12.yaml
SEQ_LEN=36
PRED_LEN=12
TEXT_LEN=36
FREQ=m
NSAMPLE=5
SAMPLE_STEPS=120
NUM_WORKERS=4
USE_RAG_COT=1
COT_ONLY=0
RAG_TOPK=1
COT_MODEL=./Qwen2.5-7B-Instruct
COT_TEMPERATURE=0.55
COT_MAX_NEW_TOKENS=96
GUIDE_SWEEP="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 3.0 4.0 5.0"
LABEL=eval

CONFIG_NAME=$(basename "${CONFIG}")

for gw in ${GUIDE_SWEEP}; do
  log_file="logs/${LABEL}_gw${gw}.log"
  echo "=== Evaluating gw=${gw} (log: ${log_file}) ==="
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
    --num_workers "${NUM_WORKERS}" \
    --use_rag_cot \
    --rag_topk "${RAG_TOPK}" \
    --cot_model "${COT_MODEL}" \
    --cot_temperature "${COT_TEMPERATURE}" \
    --cot_max_new_tokens "${COT_MAX_NEW_TOKENS}" \
    --guide_w "${gw}" | tee "${log_file}"
done

echo "Done. Logs saved to logs/${LABEL}_gw*.log"
