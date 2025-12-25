#!/usr/bin/env bash
# 单纯思维链文本引导的便捷运行脚本（不做检索）
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
DATA_PATH=${DATA_PATH:-Traffic/Traffic.csv}
CONFIG=${CONFIG:-config/traffic_36_12.yaml}
SEQ_LEN=${SEQ_LEN:-36}
PRED_LEN=${PRED_LEN:-12}
TEXT_LEN=${TEXT_LEN:-36}
FREQ=${FREQ:-m}
NSAMPLE=${NSAMPLE:-15}
SAMPLE_STEPS=${SAMPLE_STEPS:-60}
GUIDE_W=${GUIDE_W:-}
COT_MODEL=${COT_MODEL:-}  # 可选：本地因果 LM 路径/ID；留空则用模板 CoT
COT_MAX_NEW_TOKENS=${COT_MAX_NEW_TOKENS:-96}
COT_TEMPERATURE=${COT_TEMPERATURE:-0.7}
COT_CACHE_SIZE=${COT_CACHE_SIZE:-1024}

CONFIG_NAME=$(basename "${CONFIG}")

cmd=(
  python -u exe_forecasting.py
  --root_path "${ROOT_PATH}"
  --data_path "${DATA_PATH}"
  --config "${CONFIG_NAME}"
  --seq_len "${SEQ_LEN}"
  --pred_len "${PRED_LEN}"
  --text_len "${TEXT_LEN}"
  --freq "${FREQ}"
  --nsample "${NSAMPLE}"
  --use_rag_cot
  --cot_only
  --rag_topk 0
  --cot_max_new_tokens "${COT_MAX_NEW_TOKENS}"
  --cot_temperature "${COT_TEMPERATURE}"
  --cot_cache_size "${COT_CACHE_SIZE}"
)

if [[ -n "${COT_MODEL}" ]]; then
  cmd+=(--cot_model "${COT_MODEL}")
fi
cmd+=(--sample_steps_override "${SAMPLE_STEPS}")
if [[ -n "${GUIDE_W}" ]]; then
  cmd+=(--guide_w "${GUIDE_W}")
fi

echo "Running CoT-only guidance:"
echo "CONFIG=${CONFIG_NAME}, DATA_PATH=${DATA_PATH}, SEQ_LEN=${SEQ_LEN}, PRED_LEN=${PRED_LEN}, TEXT_LEN=${TEXT_LEN}, SAMPLE_STEPS=${SAMPLE_STEPS}, GUIDE_W=${GUIDE_W:-auto}"
"${cmd[@]}"
