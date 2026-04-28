#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
DEVICE=${DEVICE:-cuda:0}
NSAMPLE=${NSAMPLE:-15}
VALID_INTERVAL=${VALID_INTERVAL:-1}
SAMPLE_STEPS=${SAMPLE_STEPS:-}
CONFIG_DIR=${CONFIG_DIR:-config/_final_method}
LOG_DIR=${LOG_DIR:-logs/final_method_pairs}
DRY_RUN=${DRY_RUN:-0}

# Semicolon-separated batches. Each batch is a whitespace-separated dataset list.
# Default batches are arranged as two parallel runs per batch, then sequentially
# proceed to the next batch after both complete.
BATCHES=${BATCHES:-"Energy Agriculture;Traffic SocialGood;Climate Health_US;Environment"}

mkdir -p "${LOG_DIR}"

launch_dataset() {
  local dataset="$1"
  local stamp
  stamp="$(date -u +%Y%m%d_%H%M%S)"
  local log_file="${LOG_DIR}/${dataset}_${stamp}.log"

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY][%s] ROOT_PATH=%q DATASETS=%q DEVICE=%q NSAMPLE=%q VALID_INTERVAL=%q SAMPLE_STEPS=%q CONFIG_DIR=%q bash scripts/run_final_method.sh\n' \
      "${dataset}" "${ROOT_PATH}" "${dataset}" "${DEVICE}" "${NSAMPLE}" "${VALID_INTERVAL}" "${SAMPLE_STEPS}" "${CONFIG_DIR}"
    return 0
  fi

  printf '[LAUNCH] %s -> %s\n' "${dataset}" "${log_file}" >&2
  ROOT_PATH="${ROOT_PATH}" \
  DATASETS="${dataset}" \
  DEVICE="${DEVICE}" \
  NSAMPLE="${NSAMPLE}" \
  VALID_INTERVAL="${VALID_INTERVAL}" \
  SAMPLE_STEPS="${SAMPLE_STEPS}" \
  CONFIG_DIR="${CONFIG_DIR}" \
  bash scripts/run_final_method.sh > "${log_file}" 2>&1 &
  LAUNCH_PID="$!"
  LAUNCH_DATASET="${dataset}"
  LAUNCH_LOG_FILE="${log_file}"
}

trim() {
  local text="$1"
  text="${text#"${text%%[![:space:]]*}"}"
  text="${text%"${text##*[![:space:]]}"}"
  printf '%s' "${text}"
}

IFS=';' read -r -a batch_list <<< "${BATCHES}"

batch_index=0
for raw_batch in "${batch_list[@]}"; do
  batch="$(trim "${raw_batch}")"
  [[ -z "${batch}" ]] && continue
  batch_index=$((batch_index + 1))
  printf '\n[BATCH %d] %s\n' "${batch_index}" "${batch}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    read -r -a datasets <<< "${batch}"
    for dataset in "${datasets[@]}"; do
      launch_dataset "${dataset}"
    done
    continue
  fi

  read -r -a datasets <<< "${batch}"
  pids=()
  names=()
  logs=()
  for dataset in "${datasets[@]}"; do
    launch_dataset "${dataset}"
    pids+=("${LAUNCH_PID}")
    names+=("${LAUNCH_DATASET}")
    logs+=("${LAUNCH_LOG_FILE}")
  done

  batch_failed=0
  for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    name="${names[$idx]}"
    log_file="${logs[$idx]}"
    if wait "${pid}"; then
      printf '[DONE] %s (%s)\n' "${name}" "${log_file}"
    else
      printf '[FAIL] %s (%s)\n' "${name}" "${log_file}" >&2
      batch_failed=1
    fi
  done

  if [[ "${batch_failed}" -ne 0 ]]; then
    printf '[STOP] Batch %d failed, aborting subsequent batches.\n' "${batch_index}" >&2
    exit 1
  fi
done

printf '\n[ALL DONE] Completed all configured batches.\n'
