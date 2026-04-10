#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
COT_MODEL=${COT_MODEL:-./Qwen2.5-7B-Instruct}
GUIDE_W=${GUIDE_W:--1}
STEP_GUIDANCE_POWER=${STEP_GUIDANCE_POWER:-1.0}
STEP_GUIDANCE_FLOOR=${STEP_GUIDANCE_FLOOR:-0.35}
USE_TWO_STAGE_RAG=${USE_TWO_STAGE_RAG:-1}

COMMON_ARGS=(
  --root_path "${ROOT_PATH}"
  --use_rag_cot
  --cot_model "${COT_MODEL}"
  --step_guidance
  --step_guidance_power "${STEP_GUIDANCE_POWER}"
  --step_guidance_floor "${STEP_GUIDANCE_FLOOR}"
  --guide_w "${GUIDE_W}"
)

if [[ "${USE_TWO_STAGE_RAG}" -eq 1 ]]; then
  COMMON_ARGS+=(--use_two_stage_rag)
fi

run_case() {
  local data_path="$1"
  local config="$2"
  local seq_len="$3"
  local pred_len="$4"
  local freq="$5"
  local text_len="${6:-}"

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
  "${cmd[@]}"
}

# Traffic
run_case "Traffic/Traffic.csv" "traffic_36_6.yaml" 36 6 m 36
run_case "Traffic/Traffic.csv" "traffic_36_12.yaml" 36 12 m 36
run_case "Traffic/Traffic.csv" "traffic_36_18.yaml" 36 18 m 36

# SocialGood
run_case "SocialGood/SocialGood.csv" "socialgood_36_6.yaml" 36 6 m
run_case "SocialGood/SocialGood.csv" "socialgood_36_12.yaml" 36 12 m
run_case "SocialGood/SocialGood.csv" "socialgood_36_18.yaml" 36 18 m

# Health_US
run_case "Health_US/Health_US.csv" "health_96_12.yaml" 96 12 w
run_case "Health_US/Health_US.csv" "health_96_24.yaml" 96 24 w
run_case "Health_US/Health_US.csv" "health_96_48.yaml" 96 48 w

# Environment
run_case "Environment/Environment.csv" "environment_336_48.yaml" 336 48 d
run_case "Environment/Environment.csv" "environment_336_96.yaml" 336 96 d
run_case "Environment/Environment.csv" "environment_336_192.yaml" 336 192 d

# Energy
run_case "Energy/Energy.csv" "energy_96_12.yaml" 96 12 w 36
run_case "Energy/Energy.csv" "energy_96_24.yaml" 96 24 w 36
run_case "Energy/Energy.csv" "energy_96_48.yaml" 96 48 w 36

# Economy
run_case "Economy/Economy.csv" "economy_36_6.yaml" 36 6 m 36
run_case "Economy/Economy.csv" "economy_36_12.yaml" 36 12 m 36
run_case "Economy/Economy.csv" "economy_36_18.yaml" 36 18 m 36

# Climate
run_case "Climate/Climate.csv" "climate_96_12.yaml" 96 12 w
run_case "Climate/Climate.csv" "climate_96_24.yaml" 96 24 w
run_case "Climate/Climate.csv" "climate_96_48.yaml" 96 48 w

# Agriculture
run_case "Agriculture/Agriculture.csv" "agriculture_36_6.yaml" 36 6 m 36
run_case "Agriculture/Agriculture.csv" "agriculture_36_12.yaml" 36 12 m 36
run_case "Agriculture/Agriculture.csv" "agriculture_36_18.yaml" 36 18 m 36
