#!/usr/bin/env bash
# Long-run CoT-only with best-performing guide_w (found from experiments).
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
DATA_PATH=${DATA_PATH:-Traffic/Traffic.csv}
CONFIG=${CONFIG:-config/traffic_36_12.yaml}
SEQ_LEN=${SEQ_LEN:-36}
PRED_LEN=${PRED_LEN:-12}
TEXT_LEN=${TEXT_LEN:-36}
FREQ=${FREQ:-m}
NSAMPLE=${NSAMPLE:-5}
SAMPLE_STEPS=${SAMPLE_STEPS:-120}
EPOCHS=${EPOCHS:-20}
# BATCH_SIZE left configurable; defaults to config value if unset
# BATCH_SIZE=${BATCH_SIZE:-8}
RAG_TOPK=${RAG_TOPK:-1}
GUIDE_W=${GUIDE_W:-1.2}  # recommended after experiments
COT_MODEL=${COT_MODEL:-gpt2-medium}  # default to stronger causal LM; override with local path if needed
COT_MAX_NEW_TOKENS=${COT_MAX_NEW_TOKENS:-96}
COT_TEMPERATURE=${COT_TEMPERATURE:-0.55}
COT_CACHE_SIZE=${COT_CACHE_SIZE:-1024}
NUM_WORKERS=${NUM_WORKERS:-4}

CONFIG_NAME=$(basename "${CONFIG}")

python - <<'PY'
import os, yaml
base = os.environ.get("CONFIG", "config/traffic_36_12.yaml")
tmp = "config/_cot_best_tmp.yaml"
epochs = int(os.environ.get("EPOCHS", "20"))
sample_steps = int(os.environ.get("SAMPLE_STEPS", "120"))
batch_size = os.environ.get("BATCH_SIZE", None)
with open(base, "r") as f:
    cfg = yaml.safe_load(f)
cfg["train"]["epochs"] = epochs
cfg["diffusion"]["sample_steps"] = sample_steps
if batch_size is not None:
    cfg["train"]["batch_size"] = int(batch_size)
with open(tmp, "w") as f:
    yaml.safe_dump(cfg, f)
print(f"Written best-run config to {tmp} (epochs={epochs}, sample_steps={sample_steps}, batch_size={cfg['train']['batch_size']})")
PY

python -u exe_forecasting.py \
  --root_path "${ROOT_PATH}" \
  --data_path "${DATA_PATH}" \
  --config "_cot_best_tmp.yaml" \
  --seq_len "${SEQ_LEN}" \
  --pred_len "${PRED_LEN}" \
  --text_len "${TEXT_LEN}" \
  --freq "${FREQ}" \
  --nsample "${NSAMPLE}" \
  --use_rag_cot \
  --rag_topk "${RAG_TOPK}" \
  --sample_steps_override "${SAMPLE_STEPS}" \
  --guide_w "${GUIDE_W}" \
  --cot_model "${COT_MODEL}" \
  --cot_max_new_tokens "${COT_MAX_NEW_TOKENS}" \
  --cot_temperature "${COT_TEMPERATURE}" \
  --cot_cache_size "${COT_CACHE_SIZE}" \
  --num_workers "${NUM_WORKERS}"

rm -f config/_cot_best_tmp.yaml
