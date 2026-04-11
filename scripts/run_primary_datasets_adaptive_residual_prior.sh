#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
GUIDE_W=${GUIDE_W:-1.0}
NSAMPLE=${NSAMPLE:-15}
DEVICE=${DEVICE:-cuda:0}

RESIDUAL_PRIOR_CANDIDATES=${RESIDUAL_PRIOR_CANDIDATES:-none,last,linear,seasonal}
RESIDUAL_PRIOR_BACKTEST_LEN=${RESIDUAL_PRIOR_BACKTEST_LEN:--1}
RESIDUAL_PRIOR_LAG=${RESIDUAL_PRIOR_LAG:-6}
RESIDUAL_PRIOR_SEASONAL_LAG=${RESIDUAL_PRIOR_SEASONAL_LAG:--1}
RESIDUAL_PRIOR_SLOPE_CLIP=${RESIDUAL_PRIOR_SLOPE_CLIP:-1.0}
RESIDUAL_PRIOR_MAX_WEIGHT=${RESIDUAL_PRIOR_MAX_WEIGHT:-1.0}
RESIDUAL_PRIOR_MIN_GAIN=${RESIDUAL_PRIOR_MIN_GAIN:-0.0}
SAMPLE_STEPS_OVERRIDE=${SAMPLE_STEPS_OVERRIDE:--1}

# name data_path config seq_len pred_len text_len freq
DATASETS=(
  "Traffic Traffic/Traffic.csv traffic_36_12.yaml 36 12 36 m"
  "SocialGood SocialGood/SocialGood.csv socialgood_36_12.yaml 36 12 36 m"
  "Health_US Health_US/Health_US.csv health_96_12.yaml 96 12 36 w"
  "Environment Environment/Environment.csv environment_336_48.yaml 336 48 36 d"
  "Energy Energy/Energy.csv energy_96_12.yaml 96 12 36 w"
  "Economy Economy/Economy.csv economy_36_12_scale_router.yaml 36 12 36 m"
  "Climate Climate/Climate.csv climate_96_12.yaml 96 12 36 w"
  "Agriculture Agriculture/Agriculture.csv agriculture_36_12_scale_router.yaml 36 12 36 m"
)

for entry in "${DATASETS[@]}"; do
  read -r NAME DATA_PATH CONFIG SEQ_LEN PRED_LEN TEXT_LEN FREQ <<< "${entry}"
  echo "==== ${NAME} | ${CONFIG} (${SEQ_LEN}->${PRED_LEN}, freq=${FREQ}) ===="

  cmd=(
    python -u exe_forecasting.py
    --root_path "${ROOT_PATH}"
    --data_path "${DATA_PATH}"
    --config "${CONFIG}"
    --seq_len "${SEQ_LEN}"
    --pred_len "${PRED_LEN}"
    --text_len "${TEXT_LEN}"
    --freq "${FREQ}"
    --device "${DEVICE}"
    --nsample "${NSAMPLE}"
    --guide_w "${GUIDE_W}"
    --adaptive_residual_prior
    --residual_prior_candidates "${RESIDUAL_PRIOR_CANDIDATES}"
    --residual_prior_backtest_len "${RESIDUAL_PRIOR_BACKTEST_LEN}"
    --residual_prior_lag "${RESIDUAL_PRIOR_LAG}"
    --residual_prior_seasonal_lag "${RESIDUAL_PRIOR_SEASONAL_LAG}"
    --residual_prior_slope_clip "${RESIDUAL_PRIOR_SLOPE_CLIP}"
    --residual_prior_max_weight "${RESIDUAL_PRIOR_MAX_WEIGHT}"
    --residual_prior_min_gain "${RESIDUAL_PRIOR_MIN_GAIN}"
  )
  if [[ "${SAMPLE_STEPS_OVERRIDE}" -gt 0 ]]; then
    cmd+=(--sample_steps_override "${SAMPLE_STEPS_OVERRIDE}")
  fi
  "${cmd[@]}"
done
