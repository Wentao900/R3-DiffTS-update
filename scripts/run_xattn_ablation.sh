#!/usr/bin/env bash
# Single-factor ablations for the gated cross-attention method.
# numeric_only (T0) changes only text_drop_prob. It does not turn off the router,
# retrieval code path's parameters, the controller, or the multi-resolution loss.
# A1, T0, and T1 share diffusion.tower_dropout 0 (set when the configs are
# generated). T1 turns text injection off, which used to fall back to the
# legacy 0.1 feature-tower dropout.
#
# Examples:
#   DRY_RUN=1 bash scripts/run_xattn_ablation.sh
#   DOMAIN=traffic ABLATION=A1 bash scripts/run_xattn_ablation.sh
#   ABLATION=C1_sweep GUIDE_LIST=0,0.5,1.0,1.5,2.0 bash scripts/run_xattn_ablation.sh
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
DEVICE=${DEVICE:-cpu}
NSAMPLE=${NSAMPLE:-15}
DRY_RUN=${DRY_RUN:-0}
DOMAIN=${DOMAIN:-agriculture}
ABLATION=${ABLATION:-all}
GUIDE_W=${GUIDE_W:-1.0}
GUIDE_LIST=${GUIDE_LIST:-0,0.5,1.0,1.5,2.0}
# R4: n_min = 6 because 2 * 0.5^6 = 0.03125 <= 0.05. The values are the
# existing default seed 2025 and the next five integers. Leave SEEDS empty
# to run once on exe_forecasting.py's default seed.
SEEDS=${SEEDS:-}

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

stem_for() {
  case "$1" in
    agriculture|economy|socialgood|traffic) echo "${1}_36_12" ;;
    climate|energy|health) echo "${1}_96_12" ;;
    environment) echo "environment_336_48" ;;
    *) echo "unsupported" ;;
  esac
}

config_for() {
  local domain="$1"
  local ablation="$2"
  local stem
  stem="$(stem_for "${domain}")"
  case "${ablation}" in
    A1|S2|C1_w1|C1_sweep)
      echo "xattn/${stem}_xattn.yaml"
      ;;
    *)
      echo "xattn/ablation/${domain}_${ablation}.yaml"
      ;;
  esac
}

run_one() {
  local domain="$1"
  local ablation="$2"
  local seed="${3:-}"
  local config
  config="$(config_for "${domain}" "${ablation}")"
  local -a cmd=(
    python3 -u exe_forecasting.py
    --config "${config}"
    --root_path "${ROOT_PATH}"
    --device "${DEVICE}"
    --nsample "${NSAMPLE}"
    --num_workers 0
  )
  case "${domain}" in
    agriculture)
      cmd+=(--data_path "Agriculture/Agriculture.csv" --seq_len 36 --pred_len 12 --freq m --text_len 36)
      ;;
    traffic)
      cmd+=(--data_path "Traffic/Traffic.csv" --seq_len 36 --pred_len 12 --freq m --text_len 36)
      ;;
    economy)
      cmd+=(--data_path "Economy/Economy.csv" --seq_len 36 --pred_len 12 --freq m --text_len 36)
      ;;
    socialgood)
      cmd+=(--data_path "SocialGood/SocialGood.csv" --seq_len 36 --pred_len 12 --freq m --text_len 36)
      ;;
    climate)
      cmd+=(--data_path "Climate/Climate.csv" --seq_len 96 --pred_len 12 --freq w --text_len 36)
      ;;
    energy)
      cmd+=(--data_path "Energy/Energy.csv" --seq_len 96 --pred_len 12 --freq w --text_len 36)
      ;;
    health)
      cmd+=(--data_path "Health_US/Health_US.csv" --seq_len 96 --pred_len 12 --freq w --text_len 36)
      ;;
    environment)
      cmd+=(--data_path "Environment/Environment.csv" --seq_len 336 --pred_len 48 --freq d --text_len 36)
      ;;
    *)
      echo "unsupported domain ${domain}" >&2
      return 1
      ;;
  esac
  if [[ -n "${seed}" ]]; then
    cmd+=(--seed "${seed}")
  fi
  case "${ablation}" in
    C1_w1)
      cmd+=(--guide_w 1)
      ;;
    C1_sweep)
      cmd+=(--guide_w -1 --guide_list "${GUIDE_LIST}")
      ;;
    *)
      cmd+=(--guide_w "${GUIDE_W}")
      ;;
  esac
  echo "# ${domain} ${ablation}"
  run_cmd "${cmd[@]}"
}

ablations=(
  A1
  T0_numeric_only
  T1_router_text
  T2_gate_one
  T3_gate_constant
  T4a_shuffle_text
  C1_w1
  C1_sweep
  S1_raw_only
  S4_cot_source
  N1_naive
  N1_dlinear
  N1_none
)

domains=("${DOMAIN}")
if [[ "${DOMAIN}" == "all" ]]; then
  domains=(agriculture climate economy energy environment health socialgood traffic)
fi

run_domain() {
  local domain="$1"
  local ablation="$2"
  if [[ -z "${SEEDS}" ]]; then
    run_one "${domain}" "${ablation}"
    return
  fi
  local seed
  IFS=',' read -r -a seed_list <<< "${SEEDS}"
  for seed in "${seed_list[@]}"; do
    run_one "${domain}" "${ablation}" "${seed}"
  done
}

for domain in "${domains[@]}"; do
  if [[ "${ABLATION}" == "all" ]]; then
    for name in "${ablations[@]}"; do
      run_domain "${domain}" "${name}"
    done
  else
    run_domain "${domain}" "${ABLATION}"
  fi
done
