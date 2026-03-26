#!/bin/bash
# R3-DiffTS full pipeline: run all datasets with RAG+CoT, Two-Stage RAG,
# Trend-aware CFG, Multi-resolution Loss, and Scale Router.
# Usage: bash run_full.sh

set -e

ROOT_PATH="../Time-MMD-main"

echo "========================================"
echo "  R3-DiffTS Full Pipeline - All Datasets"
echo "========================================"

# --- Monthly datasets (seq_len=36, freq=m) ---

# Economy
for pred_len in 6 12 18; do
  echo "[Economy] seq=36 pred=${pred_len}"
  python -u exe_forecasting.py \
    --root_path ${ROOT_PATH} \
    --data_path Economy/Economy.csv \
    --config economy_36_${pred_len}_full.yaml \
    --seq_len 36 --pred_len ${pred_len} --text_len 36 --freq m
done

# Traffic
for pred_len in 6 12 18; do
  echo "[Traffic] seq=36 pred=${pred_len}"
  python -u exe_forecasting.py \
    --root_path ${ROOT_PATH} \
    --data_path Traffic/Traffic.csv \
    --config traffic_36_${pred_len}_full.yaml \
    --seq_len 36 --pred_len ${pred_len} --text_len 36 --freq m
done

# Agriculture
for pred_len in 6 12 18; do
  echo "[Agriculture] seq=36 pred=${pred_len}"
  python -u exe_forecasting.py \
    --root_path ${ROOT_PATH} \
    --data_path Agriculture/Agriculture.csv \
    --config agriculture_36_${pred_len}_full.yaml \
    --seq_len 36 --pred_len ${pred_len} --text_len 36 --freq m
done

# SocialGood
for pred_len in 6 12 18; do
  echo "[SocialGood] seq=36 pred=${pred_len}"
  python -u exe_forecasting.py \
    --root_path ${ROOT_PATH} \
    --data_path SocialGood/SocialGood.csv \
    --config socialgood_36_${pred_len}_full.yaml \
    --seq_len 36 --pred_len ${pred_len} --freq m
done

# --- Weekly datasets (seq_len=96, freq=w) ---

# Energy
for pred_len in 12 24 48; do
  echo "[Energy] seq=96 pred=${pred_len}"
  python -u exe_forecasting.py \
    --root_path ${ROOT_PATH} \
    --data_path Energy/Energy.csv \
    --config energy_96_${pred_len}_full.yaml \
    --seq_len 96 --pred_len ${pred_len} --text_len 36 --freq w
done

# Health
for pred_len in 12 24 48; do
  echo "[Health] seq=96 pred=${pred_len}"
  python -u exe_forecasting.py \
    --root_path ${ROOT_PATH} \
    --data_path Health_US/Health_US.csv \
    --config health_96_${pred_len}_full.yaml \
    --seq_len 96 --pred_len ${pred_len} --freq w
done

# Climate
for pred_len in 12 24 48; do
  echo "[Climate] seq=96 pred=${pred_len}"
  python -u exe_forecasting.py \
    --root_path ${ROOT_PATH} \
    --data_path Climate/Climate.csv \
    --config climate_96_${pred_len}_full.yaml \
    --seq_len 96 --pred_len ${pred_len} --freq w
done

# --- Daily dataset (seq_len=336, freq=d) ---

# Environment
for pred_len in 48 96 192; do
  echo "[Environment] seq=336 pred=${pred_len}"
  python -u exe_forecasting.py \
    --root_path ${ROOT_PATH} \
    --data_path Environment/Environment.csv \
    --config environment_336_${pred_len}_full.yaml \
    --seq_len 336 --pred_len ${pred_len} --seed 2021 --freq d
done

echo "========================================"
echo "  All experiments completed!"
echo "========================================"
