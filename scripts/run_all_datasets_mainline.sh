#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
GUIDE_W=${GUIDE_W:-1.0}

COMMON_ARGS=(
  --root_path "${ROOT_PATH}"
  --guide_w "${GUIDE_W}"
)

run_case() {
  local data_path="$1"
  local config="$2"
  local seq_len="$3"
  local pred_len="$4"
  local freq="$5"
  local text_len="${6:-}"
  local seed="${7:-}"

  local cmd=(
    python -u exe_forecasting.py
    --data_path "${data_path}"
    --config "${config}"
    --seq_len "${seq_len}"
    --pred_len "${pred_len}"
    --freq "${freq}"
    "${COMMON_ARGS[@]}"
  )
  if [[ -n "${text_len}" ]]; then
    cmd+=(--text_len "${text_len}")
  fi
  if [[ -n "${seed}" ]]; then
    cmd+=(--seed "${seed}")
  fi
  "${cmd[@]}"
}

run_case "Traffic/Traffic.csv" "traffic_36_12_mainline.yaml" 36 12 m 36
run_case "SocialGood/SocialGood.csv" "socialgood_36_12_mainline.yaml" 36 12 m
run_case "Health_US/Health_US.csv" "health_96_12_mainline.yaml" 96 12 w
run_case "Environment/Environment.csv" "environment_336_48_mainline.yaml" 336 48 d "" 2021
run_case "Energy/Energy.csv" "energy_96_12_mainline.yaml" 96 12 w 36
run_case "Economy/Economy.csv" "economy_36_12_mainline.yaml" 36 12 m 36
run_case "Climate/Climate.csv" "climate_96_12_mainline.yaml" 96 12 w
run_case "Agriculture/Agriculture.csv" "agriculture_36_12_mainline.yaml" 36 12 m 36
