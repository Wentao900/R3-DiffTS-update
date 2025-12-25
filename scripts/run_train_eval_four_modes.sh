#!/usr/bin/env bash
# Train once (or reuse a given checkpoint), then evaluate four modes:
# baseline / RAG+CoT / CoT-only / RAG-only. MAE/MSE printed to terminal.
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
GUIDE_SWEEP=${GUIDE_SWEEP:-"0.9 1.0 1.2 1.4"}

# Training settings
SKIP_TRAIN=${SKIP_TRAIN:-0}           # set to 1 to skip training and only evaluate
RAG_TOPK_TRAIN=${RAG_TOPK_TRAIN:-1}   # retriever during training (if enabled)
COT_MODEL=${COT_MODEL:-./Qwen2.5-7B-Instruct}
COT_TEMPERATURE=${COT_TEMPERATURE:-0.55}
COT_MAX_NEW_TOKENS=${COT_MAX_NEW_TOKENS:-96}

# Eval RAG settings per mode
RAG_TOPK_FULL=${RAG_TOPK_FULL:-1}
RAG_TOPK_COTONLY=${RAG_TOPK_COTONLY:-0}
RAG_TOPK_RAGONLY=${RAG_TOPK_RAGONLY:-1}

LABEL=${LABEL:-run_four_modes}
RUN_EVAL=${RUN_EVAL:-1}               # set to 0 to only train
MODEL_FOLDER=${MODEL_FOLDER:-}        # if set, reuse this checkpoint

mkdir -p logs

if [[ "${SKIP_TRAIN}" != "1" && -z "${MODEL_FOLDER}" ]]; then
  echo "=== Training once ==="
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
    --rag_topk "${RAG_TOPK_TRAIN}" \
    --cot_model "${COT_MODEL}" \
    --cot_temperature "${COT_TEMPERATURE}" \
    --cot_max_new_tokens "${COT_MAX_NEW_TOKENS}"
fi

# Resolve checkpoint
if [[ -n "${MODEL_FOLDER}" ]]; then
  if [[ ! -f "save/${MODEL_FOLDER}/model.pth" ]]; then
    echo "model.pth not found at save/${MODEL_FOLDER}/model.pth" >&2
    exit 1
  fi
else
  MODEL_FOLDER=$(find save -maxdepth 2 -type f -name model.pth -printf '%T@ %h\n' | sort -nr | head -n1 | awk '{print $2}' | xargs -n1 basename)
  if [[ -z "${MODEL_FOLDER}" ]]; then
    echo "No checkpoint found under save/. Train first or set MODEL_FOLDER." >&2
    exit 1
  fi
fi
echo "Using checkpoint: ${MODEL_FOLDER}"

if [[ "${RUN_EVAL}" != "1" ]]; then
  echo "RUN_EVAL=0, skip evaluation."
  exit 0
fi

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
      --config "${CONFIG}" \
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

echo "=== Evaluating (MAE/MSE shown per run) ==="
echo "== Baseline =="
run_eval baseline

echo "== RAG+CoT =="
run_eval rag_cot --use_rag_cot --rag_topk "${RAG_TOPK_FULL}" --cot_model "${COT_MODEL}" --cot_temperature "${COT_TEMPERATURE}" --cot_max_new_tokens "${COT_MAX_NEW_TOKENS}"

echo "== CoT-only =="
run_eval cot_only --use_rag_cot --cot_only --rag_topk "${RAG_TOPK_COTONLY}" --cot_model "${COT_MODEL}" --cot_temperature "${COT_TEMPERATURE}" --cot_max_new_tokens "${COT_MAX_NEW_TOKENS}"

echo "== RAG-only =="
run_eval rag_only --use_rag_cot --cot_only --rag_topk "${RAG_TOPK_RAGONLY}" --cot_model "" --cot_temperature 0.0

echo "Done. Logs in logs/${LABEL}_*.log"
