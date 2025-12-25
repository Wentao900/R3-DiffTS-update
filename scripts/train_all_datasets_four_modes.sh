#!/usr/bin/env bash
# Train four modes (baseline / RAG+CoT / CoT-only / RAG-only) on all datasets listed below.
# Only trains; no evaluation runs. Models + config_results.json will be saved under save/.
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
NSAMPLE=${NSAMPLE:-5}
SAMPLE_STEPS=${SAMPLE_STEPS:-120}
EPOCHS=${EPOCHS:-}         # optional override; if set, writes a tmp config per dataset
BATCH_SIZE=${BATCH_SIZE:-} # optional override

# Dataset list: name data_path config seq_len pred_len text_len freq
DATASETS=(
  "SocialGood SocialGood/SocialGood.csv socialgood_36_12.yaml 36 12 36 m"

)

for entry in "${DATASETS[@]}"; do
  read -r NAME DATA_PATH CONFIG SEQ_LEN PRED_LEN TEXT_LEN FREQ <<< "${entry}"
  echo "==== ${NAME} | ${CONFIG} (${SEQ_LEN}->${PRED_LEN}, freq=${FREQ}) ===="

  # If EPOCHS/BATCH_SIZE overrides are provided, write a temp config in config/_tmp_<config>
  CFG_TO_USE="${CONFIG}"
  if [[ -n "${EPOCHS}" || -n "${BATCH_SIZE}" ]]; then
    TMP_CFG="_tmp_${CONFIG}"
    CONFIG="${CONFIG}" TMP_CFG="${TMP_CFG}" EPOCHS="${EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" python - <<'PY'
import os, yaml
cfg_name = os.environ["CONFIG"]
tmp_name = os.environ["TMP_CFG"]
epochs = os.environ.get("EPOCHS", "")
batch_size = os.environ.get("BATCH_SIZE", "")
base_path = os.path.join("config", cfg_name)
tmp_path = os.path.join("config", tmp_name)
with open(base_path, "r") as f:
    cfg = yaml.safe_load(f)
if epochs:
    cfg["train"]["epochs"] = int(epochs)
if batch_size:
    cfg["train"]["batch_size"] = int(batch_size)
with open(tmp_path, "w") as f:
    yaml.safe_dump(cfg, f)
print(f"Written {tmp_path} (epochs={cfg['train']['epochs']}, batch_size={cfg['train']['batch_size']})")
PY
    CFG_TO_USE="${TMP_CFG}"
  fi

  echo "-- baseline --"
  ROOT_PATH="${ROOT_PATH}" DATA_PATH="${DATA_PATH}" CONFIG="${CFG_TO_USE}" \
  SEQ_LEN="${SEQ_LEN}" PRED_LEN="${PRED_LEN}" TEXT_LEN="${TEXT_LEN}" FREQ="${FREQ}" \
  NSAMPLE="${NSAMPLE}" SAMPLE_STEPS="${SAMPLE_STEPS}" \
    bash scripts/train_baseline.sh

  echo "-- RAG+CoT --"
  ROOT_PATH="${ROOT_PATH}" DATA_PATH="${DATA_PATH}" CONFIG="${CFG_TO_USE}" \
  SEQ_LEN="${SEQ_LEN}" PRED_LEN="${PRED_LEN}" TEXT_LEN="${TEXT_LEN}" FREQ="${FREQ}" \
  NSAMPLE="${NSAMPLE}" SAMPLE_STEPS="${SAMPLE_STEPS}" \
    bash scripts/train_rag_cot.sh

  echo "-- CoT-only --"
  ROOT_PATH="${ROOT_PATH}" DATA_PATH="${DATA_PATH}" CONFIG="${CFG_TO_USE}" \
  SEQ_LEN="${SEQ_LEN}" PRED_LEN="${PRED_LEN}" TEXT_LEN="${TEXT_LEN}" FREQ="${FREQ}" \
  NSAMPLE="${NSAMPLE}" SAMPLE_STEPS="${SAMPLE_STEPS}" \
    bash scripts/train_cot_only.sh

  echo "-- RAG-only --"
  ROOT_PATH="${ROOT_PATH}" DATA_PATH="${DATA_PATH}" CONFIG="${CFG_TO_USE}" \
  SEQ_LEN="${SEQ_LEN}" PRED_LEN="${PRED_LEN}" TEXT_LEN="${TEXT_LEN}" FREQ="${FREQ}" \
  NSAMPLE="${NSAMPLE}" SAMPLE_STEPS="${SAMPLE_STEPS}" \
    bash scripts/train_rag_only.sh
done

echo "All datasets trained (four modes each). Models saved under save/forecasting_*."
