#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
DEVICE=${DEVICE:-cuda:0}
NSAMPLE=${NSAMPLE:-15}
VALID_INTERVAL=${VALID_INTERVAL:-1}
CONFIGS=${CONFIGS:-"socialgood_36_12_plain.yaml socialgood_36_12_notext.yaml socialgood_36_12_scorehack.yaml"}
SEEDS=${SEEDS:-"2021 2025 2029"}
LOG_DIR=${LOG_DIR:-logs/socialgood_scorehack}
DEFAULT_TEXT_DROP_PROB=${DEFAULT_TEXT_DROP_PROB:-}
NOTEXT_TEXT_DROP_PROB=${NOTEXT_TEXT_DROP_PROB:-1.0}
EXTRA_ARGS=${EXTRA_ARGS:-}
NOTEXT_EXTRA_ARGS=${NOTEXT_EXTRA_ARGS:-}

mkdir -p "${LOG_DIR}"

for config in ${CONFIGS}; do
  config_stem=${config%.yaml}
  text_drop_prob=${DEFAULT_TEXT_DROP_PROB}
  extra_args=${EXTRA_ARGS}
  if [[ "${config}" == *"notext"* ]]; then
    text_drop_prob=${NOTEXT_TEXT_DROP_PROB}
    if [[ -n "${NOTEXT_EXTRA_ARGS}" ]]; then
      extra_args="${extra_args} ${NOTEXT_EXTRA_ARGS}"
    fi
  fi

  for seed in ${SEEDS}; do
    log_file="${LOG_DIR}/${config_stem}_seed_${seed}.log"
    echo "[RUN] config=${config} seed=${seed} text_drop_prob=${text_drop_prob:-default} log=${log_file}"
    cmd=(
      python -u exe_forecasting.py
      --root_path "${ROOT_PATH}"
      --data_path SocialGood/SocialGood.csv
      --config "${config}"
      --seq_len 36
      --pred_len 12
      --freq m
      --device "${DEVICE}"
      --nsample "${NSAMPLE}"
      --valid_interval "${VALID_INTERVAL}"
      --seed "${seed}"
    )
    if [[ -n "${text_drop_prob}" ]]; then
      cmd+=(--text_drop_prob "${text_drop_prob}")
    fi
    if [[ -n "${extra_args}" ]]; then
      # shellcheck disable=SC2206
      extra_parts=(${extra_args})
      cmd+=("${extra_parts[@]}")
    fi
    "${cmd[@]}" | tee "${log_file}"
  done
done

echo "[DONE] Completed SocialGood sweep. Logs in ${LOG_DIR}"
