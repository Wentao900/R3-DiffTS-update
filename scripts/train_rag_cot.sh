#!/usr/bin/env bash
# Train RAG+CoT on Traffic 36->12. No evaluation run here.
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
RAG_TOPK=${RAG_TOPK:-1}
COT_MODEL=${COT_MODEL:-./Qwen2.5-7B-Instruct}
COT_TEMPERATURE=${COT_TEMPERATURE:-0.55}
COT_MAX_NEW_TOKENS=${COT_MAX_NEW_TOKENS:-96}

python -u exe_forecasting.py \
  --root_path "${ROOT_PATH}" \
  --data_path "${DATA_PATH}" \
  --config "${CONFIG}" \
  --seq_len "${SEQ_LEN}" \
  --pred_len "${PRED_LEN}" \
  --text_len "${TEXT_LEN}" \
  --freq "${FREQ}" \
  --nsample "${NSAMPLE}" \
  --sample_steps_override "${SAMPLE_STEPS}" \
  --use_rag_cot \
  --rag_topk "${RAG_TOPK}" \
  --cot_model "${COT_MODEL}" \
  --cot_temperature "${COT_TEMPERATURE}" \
  --cot_max_new_tokens "${COT_MAX_NEW_TOKENS}"
