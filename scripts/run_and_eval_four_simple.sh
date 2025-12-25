#!/usr/bin/env bash
# Run like the original run.sh style: train once, then immediately evaluate four modes
# (baseline / RAG+CoT / CoT-only / RAG-only) and print MAE/MSE.
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
GUIDE_SWEEP=${GUIDE_SWEEP:-"0.9 1.0 1.2"}
COT_MODEL=${COT_MODEL:-./Qwen2.5-7B-Instruct}
COT_TEMPERATURE=${COT_TEMPERATURE:-0.55}
COT_MAX_NEW_TOKENS=${COT_MAX_NEW_TOKENS:-96}
LABEL=${LABEL:-run_like_run}
RUN_EVAL=${RUN_EVAL:-1}

echo "=== Training (single run, will use newest model for eval) ==="
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
  --rag_topk 1 \
  --cot_model "${COT_MODEL}" \
  --cot_temperature "${COT_TEMPERATURE}" \
  --cot_max_new_tokens "${COT_MAX_NEW_TOKENS}"

# pick the newest folder that now has model.pth (should be the one just trained)
MODEL_FOLDER=$(find save -maxdepth 2 -type f -name model.pth -printf '%T@ %h\n' | sort -nr | head -n1 | awk '{print $2}' | xargs -n1 basename)
if [[ -z "${MODEL_FOLDER}" ]]; then
  echo "No trained folder with model.pth found under save/." >&2
  exit 1
fi
echo "Using checkpoint: ${MODEL_FOLDER}"

if [[ "${RUN_EVAL}" != "1" ]]; then
  echo "RUN_EVAL=0, skip evaluation. Model saved under save/${MODEL_FOLDER}"
  exit 0
fi

run_eval() {
  local label=$1
  shift
  for gw in ${GUIDE_SWEEP}; do
    echo "-- ${label}, gw=${gw} --"
    python -u exe_forecasting.py \
      --modelfolder "${MODEL_FOLDER}" \
      --root_path "${ROOT_PATH}" \
      --data_path "${DATA_PATH}" \
      --config "${CONFIG}" \
      --seq_len "${SEQ_LEN}" \
      --pred_len "${PRED_LEN}" \
      --text_len "${TEXT_LEN}" \
      --freq "${FREQ}" \
      --nsample "${NSAMPLE}" \
      --sample_steps_override "${SAMPLE_STEPS}" \
      --guide_w "${gw}" \
      "$@"
  done
}

echo "=== Evaluating (MAE/MSE printed per run) ==="
echo "== Baseline (no RAG/CoT) =="
run_eval baseline

echo "== RAG+CoT =="
run_eval rag_cot --use_rag_cot --rag_topk 1 --cot_model "${COT_MODEL}" --cot_temperature "${COT_TEMPERATURE}" --cot_max_new_tokens "${COT_MAX_NEW_TOKENS}"

echo "== CoT-only (no retrieval) =="
run_eval cot_only --use_rag_cot --cot_only --rag_topk 0 --cot_model "${COT_MODEL}" --cot_temperature "${COT_TEMPERATURE}" --cot_max_new_tokens "${COT_MAX_NEW_TOKENS}"

echo "== RAG-only (no CoT) =="
run_eval rag_only --use_rag_cot --cot_only --rag_topk 1 --cot_model "" --cot_temperature 0.0

echo "Done."
