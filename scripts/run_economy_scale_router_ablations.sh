#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
BASE_CONFIG=${BASE_CONFIG:-config/economy_36_12_multires_rag_tuned.yaml}
DATA_PATH=${DATA_PATH:-Economy/Economy.csv}
SEQ_LEN=${SEQ_LEN:-36}
PRED_LEN=${PRED_LEN:-12}
TEXT_LEN=${TEXT_LEN:-36}
FREQ=${FREQ:-m}
DEVICE=${DEVICE:-cuda:0}
NSAMPLE=${NSAMPLE:-15}
VALID_INTERVAL=${VALID_INTERVAL:-1}
SAMPLE_STEPS=${SAMPLE_STEPS:--1}
GUIDE_W=${GUIDE_W:--1}
EPOCHS=${EPOCHS:--1}
BATCH_SIZE=${BATCH_SIZE:--1}
LR=${LR:--1}
CASES=${CASES:-no_multires,cum_base,disjoint_only,router_window_only,router_loss_only,router_full,router_guidance}
LABEL=${LABEL:-economy_scale_router_ablation}
EXTRA_ARGS=${EXTRA_ARGS:-}

cmd=(
  python -u scripts/run_economy_scale_router_ablations.py
  --base-config "${BASE_CONFIG}"
  --root_path "${ROOT_PATH}"
  --data_path "${DATA_PATH}"
  --seq_len "${SEQ_LEN}"
  --pred_len "${PRED_LEN}"
  --text_len "${TEXT_LEN}"
  --freq "${FREQ}"
  --device "${DEVICE}"
  --nsample "${NSAMPLE}"
  --valid_interval "${VALID_INTERVAL}"
  --guide_w "${GUIDE_W}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --cases "${CASES}"
  --label "${LABEL}"
)

if [[ "${SAMPLE_STEPS}" -gt 0 ]]; then
  cmd+=(--sample_steps_override "${SAMPLE_STEPS}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_ARGS} )
  cmd+=("${extra[@]}")
fi

"${cmd[@]}"
