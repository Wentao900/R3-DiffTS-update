#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
DATA_PATH=${DATA_PATH:-Traffic/Traffic.csv}
CONFIG=${CONFIG:-traffic_36_12.yaml}
SEQ_LEN=${SEQ_LEN:-36}
PRED_LEN=${PRED_LEN:-12}
TEXT_LEN=${TEXT_LEN:-36}
FREQ=${FREQ:-m}
GUIDE_LIST=${GUIDE_LIST:-"3 4 5"}
EXTRA_ARGS=${EXTRA_ARGS:-""}

python -u exe_forecasting.py \
  --root_path "${ROOT_PATH}" \
  --data_path "${DATA_PATH}" \
  --config "${CONFIG}" \
  --seq_len "${SEQ_LEN}" --pred_len "${PRED_LEN}" --text_len "${TEXT_LEN}" --freq "${FREQ}" \
  ${EXTRA_ARGS}

domain="${DATA_PATH%%/*}"
latest_dir=$(ls -dt "save/forecasting_${domain}_"* 2>/dev/null | head -n 1 || true)
if [[ -z "${latest_dir}" ]]; then
  echo "No model folder found under save/forecasting_${domain}_*" >&2
  exit 1
fi

model_folder=$(basename "${latest_dir}")
for gw in ${GUIDE_LIST}; do
  python -u exe_forecasting.py \
    --root_path "${ROOT_PATH}" \
    --data_path "${DATA_PATH}" \
    --config "${CONFIG}" \
    --seq_len "${SEQ_LEN}" --pred_len "${PRED_LEN}" --text_len "${TEXT_LEN}" --freq "${FREQ}" \
    --modelfolder "${model_folder}" \
    --guide_w "${gw}" \
    ${EXTRA_ARGS}
done
