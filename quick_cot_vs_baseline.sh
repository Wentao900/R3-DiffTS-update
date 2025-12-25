#!/usr/bin/env bash
# Quick validation: baseline vs CoT-only (no retrieval). Supports guide_w sweeping and optional long-run profile.
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
DATA_PATH=${DATA_PATH:-Traffic/Traffic.csv}
BASE_CONFIG=${BASE_CONFIG:-config/traffic_36_12.yaml}
TMP_CONFIG=${TMP_CONFIG:-config/_cot_quick_tmp.yaml}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-8}
SEQ_LEN=${SEQ_LEN:-36}
PRED_LEN=${PRED_LEN:-12}
TEXT_LEN=${TEXT_LEN:-36}
FREQ=${FREQ:-m}
NSAMPLE=${NSAMPLE:-5}
SAMPLE_STEPS=${SAMPLE_STEPS:-60}
NUM_WORKERS=${NUM_WORKERS:-4}
COT_MODEL=${COT_MODEL:-gpt2-medium} # default to stronger local/ cached causal LM; override if needed
COT_TEMPERATURE=${COT_TEMPERATURE:-0.55}
COT_MAX_NEW_TOKENS=${COT_MAX_NEW_TOKENS:-96}
GUIDE_W_BASE=${GUIDE_W_BASE:-}
GUIDE_W_COT=${GUIDE_W_COT:-}
GUIDE_SWEEP_BASE=${GUIDE_SWEEP_BASE:-"0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 3.0 4.0 5.0"}
GUIDE_SWEEP_COT=${GUIDE_SWEEP_COT:-"0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 3.0 4.0 5.0"}
LONG_RUN=${LONG_RUN:-1}
LONG_TMP_CONFIG=${LONG_TMP_CONFIG:-config/_cot_long_tmp.yaml}
LONG_EPOCHS=${LONG_EPOCHS:-20}
LONG_SAMPLE_STEPS=${LONG_SAMPLE_STEPS:-120}
LONG_NSAMPLE=${LONG_NSAMPLE:-5}

mkdir -p logs

# write tmp config
BASE_CONFIG="$BASE_CONFIG" TMP_CONFIG="$TMP_CONFIG" EPOCHS="$EPOCHS" BATCH_SIZE="$BATCH_SIZE" SAMPLE_STEPS="$SAMPLE_STEPS" python - <<'PY'
import os, yaml
base = os.environ.get("BASE_CONFIG", "config/traffic_36_12.yaml")
tmp = os.environ.get("TMP_CONFIG", "config/_cot_quick_tmp.yaml")
epochs = int(os.environ.get("EPOCHS", "5"))
batch_size = int(os.environ.get("BATCH_SIZE", "8"))
sample_steps = int(os.environ.get("SAMPLE_STEPS", "60"))
with open(base, "r") as f:
    cfg = yaml.safe_load(f)
cfg["train"]["epochs"] = epochs
cfg["train"]["batch_size"] = batch_size
cfg["diffusion"]["sample_steps"] = sample_steps
with open(tmp, "w") as f:
    yaml.safe_dump(cfg, f)
print(f"Written quick config to {tmp} (epochs={epochs}, batch_size={batch_size}, sample_steps={sample_steps})")
PY

RUNS=(
  "baseline"
  "cot_only --use_rag_cot --cot_only --rag_topk 0"
)

run_suite() {
  local config_path="$1"
  local sample_steps_override="$2"
  local nsample_override="$3"
  local log_suffix="$4"
  local config_name
  config_name=$(basename "${config_path}")

  for run in "${RUNS[@]}"; do
    IFS=' ' read -r -a parts <<< "${run}"
    label="${parts[0]}"
    flags=("${parts[@]:1}")
    guides=()
    if [[ "${label}" == "baseline" && -n "${GUIDE_SWEEP_BASE}" ]]; then
      guides=(${GUIDE_SWEEP_BASE})
    elif [[ "${label}" == "cot_only" && -n "${GUIDE_SWEEP_COT}" ]]; then
      guides=(${GUIDE_SWEEP_COT})
    elif [[ "${label}" == "baseline" && -n "${GUIDE_W_BASE}" ]]; then
      guides=("${GUIDE_W_BASE}")
    elif [[ "${label}" == "cot_only" && -n "${GUIDE_W_COT}" ]]; then
      guides=("${GUIDE_W_COT}")
    else
      guides=("")
    fi

    for gw in "${guides[@]}"; do
      suffix=""
      gw_args=()
      if [[ -n "${gw}" ]]; then
        gw_args=(--guide_w "${gw}")
        suffix="_gw${gw}"
      fi
      log_file_with_gw="logs/quick_cot_${label}${suffix}${log_suffix}.log"
      cmd=(
        python -u exe_forecasting.py
        --root_path "${ROOT_PATH}"
        --data_path "${DATA_PATH}"
        --config "${config_name}"
        --seq_len "${SEQ_LEN}"
        --pred_len "${PRED_LEN}"
        --text_len "${TEXT_LEN}"
        --freq "${FREQ}"
        --nsample "${nsample_override}"
        --num_workers "${NUM_WORKERS}"
        --sample_steps_override "${sample_steps_override}"
      )
      if [[ ${#flags[@]} -gt 0 ]]; then
        cmd+=("${flags[@]}")
      fi
      cmd+=("${gw_args[@]}")
      if [[ -n "${COT_MODEL}" ]]; then
        cmd+=(--cot_model "${COT_MODEL}")
      fi
      echo "=== Running ${label}${log_suffix} (log: ${log_file_with_gw}) ==="
      echo "Config=${config_name}, data=${DATA_PATH}, seq=${SEQ_LEN}, pred=${PRED_LEN}, text_len=${TEXT_LEN}, sample_steps=${sample_steps_override}, guide_w=${gw:-auto}"
      "${cmd[@]}" | tee "${log_file_with_gw}"
    done
  done
}

run_suite "${TMP_CONFIG}" "${SAMPLE_STEPS}" "${NSAMPLE}" ""

if [[ "${LONG_RUN}" == "1" ]]; then
  BASE_CONFIG="$BASE_CONFIG" TMP_CONFIG="$LONG_TMP_CONFIG" EPOCHS="$LONG_EPOCHS" BATCH_SIZE="$BATCH_SIZE" SAMPLE_STEPS="$LONG_SAMPLE_STEPS" python - <<'PY'
import os, yaml
base = os.environ.get("BASE_CONFIG", "config/traffic_36_12.yaml")
tmp = os.environ.get("TMP_CONFIG", "config/_cot_long_tmp.yaml")
epochs = int(os.environ.get("EPOCHS", "15"))
batch_size = int(os.environ.get("BATCH_SIZE", "8"))
sample_steps = int(os.environ.get("SAMPLE_STEPS", "100"))
with open(base, "r") as f:
    cfg = yaml.safe_load(f)
cfg["train"]["epochs"] = epochs
cfg["train"]["batch_size"] = batch_size
cfg["diffusion"]["sample_steps"] = sample_steps
with open(tmp, "w") as f:
    yaml.safe_dump(cfg, f)
print(f"Written long config to {tmp} (epochs={epochs}, batch_size={batch_size}, sample_steps={sample_steps})")
PY
  run_suite "${LONG_TMP_CONFIG}" "${LONG_SAMPLE_STEPS}" "${LONG_NSAMPLE}" "_long"
fi

rm -f "${TMP_CONFIG}"
if [[ "${LONG_RUN}" == "1" ]]; then
  rm -f "${LONG_TMP_CONFIG}"
fi
echo "Done. Logs saved to logs/quick_cot_<label>[_gwX][_long].log."
