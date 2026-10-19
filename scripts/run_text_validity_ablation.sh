#!/usr/bin/env bash
# Text-validity ablations. Same seed. guide_w is chosen on the validation split
# by the existing manual sweep (exe_forecasting.py), then the chosen value is
# evaluated once on test. This script does not tune on the test set.
#
# Rule mode is the unchanged controller path:
#   config/economy_36_12_mainline.yaml
#   config/socialgood_36_12_scorehack.yaml   (auto mode; no guide sweep)
#
# Learned-mode configs live in config/text_validity/.

set -euo pipefail

SEED="${SEED:-2025}"
GUIDE_LIST="${GUIDE_LIST:-0,0.5,1.0,1.5,2.0}"
DEVICE="${DEVICE:-cuda:0}"
NSAMPLE="${NSAMPLE:-15}"
ROOT_PATH="${ROOT_PATH:-../Time-MMD-main}"

run_manual() {
  local config="$1"
  local data_path="$2"
  local seq_len="$3"
  local pred_len="$4"
  local freq="$5"
  python -u exe_forecasting.py \
    --config "${config}" \
    --data_path "${data_path}" \
    --seq_len "${seq_len}" \
    --pred_len "${pred_len}" \
    --freq "${freq}" \
    --root_path "${ROOT_PATH}" \
    --device "${DEVICE}" \
    --nsample "${NSAMPLE}" \
    --seed "${SEED}" \
    --guide_w -1 \
    --guide_list "${GUIDE_LIST}" \
    --valid_interval 1 \
    --text_len "${seq_len}"
}

run_auto_rule() {
  local config="$1"
  local data_path="$2"
  python -u exe_forecasting.py \
    --config "${config}" \
    --data_path "${data_path}" \
    --seq_len 36 \
    --pred_len 12 \
    --freq m \
    --root_path "${ROOT_PATH}" \
    --device "${DEVICE}" \
    --nsample "${NSAMPLE}" \
    --seed "${SEED}" \
    --valid_interval 1 \
    --text_len 36
}

# Economy, seq 36, pred 12, monthly.
run_manual economy_36_12_mainline.yaml Economy/Economy.csv 36 12 m
run_manual text_validity/economy_learned.yaml Economy/Economy.csv 36 12 m
run_manual text_validity/economy_fixed.yaml Economy/Economy.csv 36 12 m
run_manual text_validity/economy_fixed_mean.yaml Economy/Economy.csv 36 12 m
run_manual text_validity/economy_notext.yaml Economy/Economy.csv 36 12 m
run_manual text_validity/economy_shuffle.yaml Economy/Economy.csv 36 12 m

# SocialGood. The historical scorehack config is guide_mode=auto.
run_auto_rule socialgood_36_12_scorehack.yaml SocialGood/SocialGood.csv
run_manual text_validity/socialgood_learned.yaml SocialGood/SocialGood.csv 36 12 m
run_manual text_validity/socialgood_fixed.yaml SocialGood/SocialGood.csv 36 12 m
run_manual text_validity/socialgood_fixed_mean.yaml SocialGood/SocialGood.csv 36 12 m
run_manual text_validity/socialgood_notext.yaml SocialGood/SocialGood.csv 36 12 m
run_manual text_validity/socialgood_shuffle.yaml SocialGood/SocialGood.csv 36 12 m
