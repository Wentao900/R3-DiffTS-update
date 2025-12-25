#!/usr/bin/env bash
# Quick validation script: runs two lightweight experiments for Traffic (36/6)
# with small epochs to compare sampling steps and text dropout settings.
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
DATA_PATH=${DATA_PATH:-Traffic/Traffic.csv}
BASE_CONFIG=${BASE_CONFIG:-config/traffic_36_6.yaml}
TMP_CONFIG=${TMP_CONFIG:-config/_quick_tmp.yaml}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-8}
NSAMPLE=${NSAMPLE:-5}
SEQ_LEN=${SEQ_LEN:-36}
PRED_LEN=${PRED_LEN:-6}
TEXT_LEN=${TEXT_LEN:-36}
FREQ=${FREQ:-m}

mkdir -p logs

BASE_CONFIG="$BASE_CONFIG" TMP_CONFIG="$TMP_CONFIG" EPOCHS="$EPOCHS" BATCH_SIZE="$BATCH_SIZE" python - <<'PY'
import os, yaml
base = os.environ.get("BASE_CONFIG", "config/traffic_36_6.yaml")
tmp = os.environ.get("TMP_CONFIG", "config/_quick_tmp.yaml")
epochs = int(os.environ.get("EPOCHS", "5"))
batch_size = int(os.environ.get("BATCH_SIZE", "8"))
with open(base, "r") as f:
    cfg = yaml.safe_load(f)
cfg["train"]["epochs"] = epochs
cfg["train"]["batch_size"] = batch_size
with open(tmp, "w") as f:
    yaml.safe_dump(cfg, f)
print(f"Written quick config to {tmp} (epochs={epochs}, batch_size={batch_size})")
PY

RUNS=(
  "name=fast30 sample_steps=30 text_drop=0.3"
  "name=default60 sample_steps=60 text_drop=0.0"
)

for run in "${RUNS[@]}"; do
  IFS=' ' read -r -a parts <<< "$run"
  name="${parts[0]#name=}"
  sample_steps="${parts[1]#sample_steps=}"
  text_drop="${parts[2]#text_drop=}"
  log_file="logs/quick_${name}.log"
  echo "=== Running ${name}: sample_steps=${sample_steps}, text_drop_prob=${text_drop} (log: ${log_file}) ==="
  python -u exe_forecasting.py \
    --root_path "${ROOT_PATH}" \
    --data_path "${DATA_PATH}" \
    --config "$(basename "${TMP_CONFIG}")" \
    --seq_len "${SEQ_LEN}" \
    --pred_len "${PRED_LEN}" \
    --text_len "${TEXT_LEN}" \
    --freq "${FREQ}" \
    --sample_steps_override "${sample_steps}" \
    --text_drop_prob "${text_drop}" \
    --max_text_tokens 256 \
    --nsample "${NSAMPLE}" \
    --num_workers 4 | tee "${log_file}"
done

rm -f "${TMP_CONFIG}"
echo "Done. Check logs/quick_*.log for metrics."
