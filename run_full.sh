#!/bin/bash
# R3-DiffTS full pipeline: run all datasets with RAG+CoT, Two-Stage RAG,
# Trend-aware CFG, Multi-resolution Loss, and Scale Router.
# Usage: bash run_full.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ROOT_PATH="${ROOT_PATH:-../Time-MMD-main}"

run_case() {
  local data_path=$1
  local config=$2
  local seq_len=$3
  local pred_len=$4
  local text_len=$5
  local freq=$6
  local extra_args=("${@:7}")

  if [[ ! -f "config/${config}" ]]; then
    echo "Missing config: config/${config}" >&2
    exit 1
  fi

  python -u exe_forecasting.py \
    --root_path "${ROOT_PATH}" \
    --data_path "${data_path}" \
    --config "${config}" \
    --seq_len "${seq_len}" \
    --pred_len "${pred_len}" \
    --text_len "${text_len}" \
    --freq "${freq}" \
    "${extra_args[@]}"
}

echo "========================================"
echo "  R3-DiffTS Full Pipeline - All Datasets"
echo "========================================"

# --- Monthly datasets (seq_len=36, freq=m) ---

# Economy
for pred_len in 6 12 18; do
  echo "[Economy] seq=36 pred=${pred_len}"
  run_case "Economy/Economy.csv" "economy_36_${pred_len}_full.yaml" 36 "${pred_len}" 36 m
done

# Traffic
for pred_len in 6 12 18; do
  echo "[Traffic] seq=36 pred=${pred_len}"
  run_case "Traffic/Traffic.csv" "traffic_36_${pred_len}_full.yaml" 36 "${pred_len}" 36 m
done

# Agriculture
for pred_len in 6 12 18; do
  echo "[Agriculture] seq=36 pred=${pred_len}"
  run_case "Agriculture/Agriculture.csv" "agriculture_36_${pred_len}_full.yaml" 36 "${pred_len}" 36 m
done

# SocialGood
for pred_len in 6 12 18; do
  echo "[SocialGood] seq=36 pred=${pred_len}"
  run_case "SocialGood/SocialGood.csv" "socialgood_36_${pred_len}_full.yaml" 36 "${pred_len}" 36 m
done

# --- Weekly datasets (seq_len=96, freq=w) ---

# Energy
for pred_len in 12 24 48; do
  echo "[Energy] seq=96 pred=${pred_len}"
  run_case "Energy/Energy.csv" "energy_96_${pred_len}_full.yaml" 96 "${pred_len}" 36 w
done

# Health
for pred_len in 12 24 48; do
  echo "[Health] seq=96 pred=${pred_len}"
  run_case "Health_US/Health_US.csv" "health_96_${pred_len}_full.yaml" 96 "${pred_len}" 36 w
done

# Climate
for pred_len in 12 24 48; do
  echo "[Climate] seq=96 pred=${pred_len}"
  run_case "Climate/Climate.csv" "climate_96_${pred_len}_full.yaml" 96 "${pred_len}" 36 w
done

# --- Daily dataset (seq_len=336, freq=d) ---

# Environment
for pred_len in 48 96 192; do
  echo "[Environment] seq=336 pred=${pred_len}"
  run_case "Environment/Environment.csv" "environment_336_${pred_len}_full.yaml" 336 "${pred_len}" 36 d --seed 2021
done

echo "========================================"
echo "  All experiments completed!"
echo "========================================"
