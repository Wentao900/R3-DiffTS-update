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

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

config_for() {
  local domain="$1"
  local ablation="$2"
  case "${ablation}" in
    A1|S2|C1_w1|C1_sweep)
      echo "xattn/${domain}_36_12_xattn.yaml"
      ;;
    *)
      echo "xattn/ablation/${domain}_${ablation}.yaml"
      ;;
  esac
}

run_one() {
  local domain="$1"
  local ablation="$2"
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
    *)
      echo "unsupported domain ${domain}; add it beside agriculture/traffic" >&2
      return 1
      ;;
  esac
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

if [[ "${ABLATION}" == "all" ]]; then
  for name in "${ablations[@]}"; do
    run_one "${DOMAIN}" "${name}"
  done
else
  run_one "${DOMAIN}" "${ABLATION}"
fi
