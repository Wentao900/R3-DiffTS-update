#!/usr/bin/env bash
# Train on Traffic 36->12 and then evaluate a guide_w sweep using the trained checkpoint.
# Defaults reflect best current settings (rag_topk=1, temp=0.55, bs=8, sample_steps=120).
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
DATA_PATH=${DATA_PATH:-Traffic/Traffic.csv}
CONFIG=${CONFIG:-traffic_36_12.yaml}
SEQ_LEN=${SEQ_LEN:-36}
PRED_LEN=${PRED_LEN:-12}
TEXT_LEN=${TEXT_LEN:-36}
FREQ=${FREQ:-m}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-8}
SAMPLE_STEPS=${SAMPLE_STEPS:-120}
NSAMPLE=${NSAMPLE:-5}
GUIDE_SWEEP=${GUIDE_SWEEP:-"0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 3.0 4.0 5.0"}
RAG_TOPK=${RAG_TOPK:-1}
COT_MODEL=${COT_MODEL:-./Qwen2.5-7B-Instruct}
COT_TEMPERATURE=${COT_TEMPERATURE:-0.55}
COT_MAX_NEW_TOKENS=${COT_MAX_NEW_TOKENS:-96}
LABEL=${LABEL:-train_eval}

mkdir -p logs

echo "=== Training ==="
python -u exe_forecasting.py \
  --root_path "${ROOT_PATH}" \
  --data_path "${DATA_PATH}" \
  --config "${CONFIG}" \
  --seq_len "${SEQ_LEN}" \
  --pred_len "${PRED_LEN}" \
  --text_len "${TEXT_LEN}" \
  --freq "${FREQ}" \
  --sample_steps_override "${SAMPLE_STEPS}" \
  --nsample "${NSAMPLE}" \
  --use_rag_cot \
  --rag_topk "${RAG_TOPK}" \
  --cot_model "${COT_MODEL}" \
  --cot_temperature "${COT_TEMPERATURE}" \
  --cot_max_new_tokens "${COT_MAX_NEW_TOKENS}" \
  --guide_w 1.2 \
  --lr 0.0001 \
  --valid_interval 1 \
  --dropout 0.0 \
  --attn_drop 0.0 \
  --c_mask_prob 0.2 \
  --beta_end 0.7 \
  --sample_steps_override "${SAMPLE_STEPS}" \
  --save_attn false \
  --save_token false \
  --time_weight 0.1

LATEST_FOLDER=$(ls -td save/forecasting_Traffic_* 2>/dev/null | head -n1 | xargs -r basename)
if [[ -z "${LATEST_FOLDER}" ]]; then
  echo "No trained folder found under save/." >&2
  exit 1
fi

echo "=== Evaluating sweep on ${LATEST_FOLDER} ==="
for gw in ${GUIDE_SWEEP}; do
  log_file="logs/${LABEL}_gw${gw}.log"
  echo "-- gw=${gw} (log: ${log_file}) --"
  python -u exe_forecasting.py \
    --modelfolder "${LATEST_FOLDER}" \
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
    --cot_max_new_tokens "${COT_MAX_NEW_TOKENS}" \
    --guide_w "${gw}" | tee "${log_file}"
done

echo "Done. Logs saved to logs/${LABEL}_gw*.log"
