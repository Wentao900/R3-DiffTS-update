#!/usr/bin/env bash
# Train once on Traffic 36->12, then evaluate four configs (baseline/full/CoT-only/RAG-only) over a quick gw sweep.
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
DATA_PATH=${DATA_PATH:-Traffic/Traffic.csv}
BASE_CONFIG=${BASE_CONFIG:-config/traffic_36_12.yaml}
TMP_CONFIG=${TMP_CONFIG:-config/_four_cfg_tmp.yaml}
SEQ_LEN=${SEQ_LEN:-36}
PRED_LEN=${PRED_LEN:-12}
TEXT_LEN=${TEXT_LEN:-36}
FREQ=${FREQ:-m}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-8}
SAMPLE_STEPS=${SAMPLE_STEPS:-120}
NSAMPLE=${NSAMPLE:-5}
GUIDE_SWEEP=${GUIDE_SWEEP:-"0.9 1.0"}
COT_MODEL=${COT_MODEL:-./Qwen2.5-7B-Instruct}
COT_TEMPERATURE=${COT_TEMPERATURE:-0.55}
COT_MAX_NEW_TOKENS=${COT_MAX_NEW_TOKENS:-96}
# Rag settings per config
RAG_TOPK_FULL=${RAG_TOPK_FULL:-1}
RAG_TOPK_COTONLY=${RAG_TOPK_COTONLY:-0}
RAG_TOPK_RAGONLY=${RAG_TOPK_RAGONLY:-1}
LABEL=${LABEL:-train_eval_four}

python - <<'PY'
import os, yaml
base = os.environ.get('BASE_CONFIG', 'config/traffic_36_12.yaml')
tmp = os.environ.get('TMP_CONFIG', 'config/_four_cfg_tmp.yaml')
epochs = int(os.environ.get('EPOCHS', '20'))
batch_size = int(os.environ.get('BATCH_SIZE', '8'))
sample_steps = int(os.environ.get('SAMPLE_STEPS', '120'))
with open(base, 'r') as f:
    cfg = yaml.safe_load(f)
cfg['train']['epochs'] = epochs
cfg['train']['batch_size'] = batch_size
cfg['diffusion']['sample_steps'] = sample_steps
with open(tmp, 'w') as f:
    yaml.safe_dump(cfg, f)
print(f"Written tmp config {tmp} (epochs={epochs}, batch_size={batch_size}, sample_steps={sample_steps})")
PY

mkdir -p logs

echo "=== Training (single run) ==="
python -u exe_forecasting.py \
  --root_path "${ROOT_PATH}" \
  --data_path "${DATA_PATH}" \
  --config "$(basename ${TMP_CONFIG})" \
  --seq_len "${SEQ_LEN}" \
  --pred_len "${PRED_LEN}" \
  --text_len "${TEXT_LEN}" \
  --freq "${FREQ}" \
  --sample_steps_override "${SAMPLE_STEPS}" \
  --nsample "${NSAMPLE}" \
  --use_rag_cot \
  --rag_topk "${RAG_TOPK_FULL}" \
  --cot_model "${COT_MODEL}" \
  --cot_temperature "${COT_TEMPERATURE}" \
  --cot_max_new_tokens "${COT_MAX_NEW_TOKENS}"

# pick latest folder that actually has model.pth
LATEST_FOLDER=$(find save -maxdepth 2 -type f -name model.pth -printf '%h
' | xargs -n1 basename | sort -r | head -n1)
if [[ -z "${LATEST_FOLDER}" ]]; then
  echo "No trained folder with model.pth found under save/." >&2
  exit 1
fi

CONFIG_NAME=$(basename "${TMP_CONFIG}")

eval_cfg() {
  local label=$1; shift
  for gw in ${GUIDE_SWEEP}; do
    log_file="logs/${LABEL}_${label}_gw${gw}.log"
    echo "-- ${label}, gw=${gw} --"
    python -u exe_forecasting.py \
      --modelfolder "${LATEST_FOLDER}" \
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

echo "=== Evaluating ${LATEST_FOLDER} ==="
# baseline
eval_cfg baseline
# full (RAG+CoT)
eval_cfg full --use_rag_cot --rag_topk "${RAG_TOPK_FULL}" --cot_model "${COT_MODEL}" --cot_temperature "${COT_TEMPERATURE}" --cot_max_new_tokens "${COT_MAX_NEW_TOKENS}"
# CoT-only
eval_cfg cot_only --use_rag_cot --cot_only --rag_topk "${RAG_TOPK_COTONLY}" --cot_model "${COT_MODEL}" --cot_temperature "${COT_TEMPERATURE}" --cot_max_new_tokens "${COT_MAX_NEW_TOKENS}"
# RAG-only (no CoT)
eval_cfg rag_only --use_rag_cot --cot_only --rag_topk "${RAG_TOPK_RAGONLY}" --cot_model "" --cot_temperature 0.0

echo "Done. Logs: logs/${LABEL}_*.log"
rm -f "${TMP_CONFIG}"
