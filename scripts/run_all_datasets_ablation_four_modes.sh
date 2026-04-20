#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
NSAMPLE=${NSAMPLE:-5}
SAMPLE_STEPS=${SAMPLE_STEPS:-120}
GUIDE_W=${GUIDE_W:-1.0}
DEVICE=${DEVICE:-cuda:0}
VALID_INTERVAL=${VALID_INTERVAL:-1}
DRY_RUN=${DRY_RUN:-0}

DATASET_FILTER=${DATASET_FILTER:-}
MODE_FILTER=${MODE_FILTER:-}
EPOCHS=${EPOCHS:-}
BATCH_SIZE=${BATCH_SIZE:-}

RAG_TOPK=${RAG_TOPK:-1}
COT_MODEL=${COT_MODEL:-./Qwen2.5-7B-Instruct}
COT_TEMPERATURE=${COT_TEMPERATURE:-0.55}
COT_MAX_NEW_TOKENS=${COT_MAX_NEW_TOKENS:-96}

contains_word() {
  local needle="$1"
  local haystack="$2"
  [[ -z "${haystack}" ]] && return 0
  for word in ${haystack}; do
    [[ "${word}" == "${needle}" ]] && return 0
  done
  return 1
}

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

write_mode_config() {
  local dataset_key="$1"
  local mode="$2"
  local base_config="$3"
  local tmp_config="$4"

  DATASET_KEY="${dataset_key}" \
  MODE="${mode}" \
  BASE_CONFIG="${base_config}" \
  TMP_CONFIG="${tmp_config}" \
  EPOCHS="${EPOCHS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  RAG_TOPK="${RAG_TOPK}" \
  COT_MODEL="${COT_MODEL}" \
  COT_TEMPERATURE="${COT_TEMPERATURE}" \
  COT_MAX_NEW_TOKENS="${COT_MAX_NEW_TOKENS}" \
  DRY_RUN="${DRY_RUN}" \
  python - <<'PY'
import os
from pathlib import Path

import yaml

mode = os.environ["MODE"]
base_path = Path("config") / os.environ["BASE_CONFIG"]
tmp_path = Path("config") / os.environ["TMP_CONFIG"]

with base_path.open("r") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("train", {})
cfg.setdefault("model", {})

if os.environ.get("EPOCHS"):
    cfg["train"]["epochs"] = int(os.environ["EPOCHS"])
if os.environ.get("BATCH_SIZE"):
    cfg["train"]["batch_size"] = int(os.environ["BATCH_SIZE"])

rag_topk = int(os.environ["RAG_TOPK"])
cot_temperature = float(os.environ["COT_TEMPERATURE"])
cot_max_new_tokens = int(os.environ["COT_MAX_NEW_TOKENS"])
cot_model = os.environ["COT_MODEL"]

if mode == "baseline":
    cfg["model"].update(
        use_rag_cot=False,
        cot_only=False,
        rag_topk=0,
        use_two_stage_rag=False,
    )
elif mode == "rag_cot":
    cfg["model"].update(
        use_rag_cot=True,
        cot_only=False,
        rag_topk=rag_topk,
        rag_stage1_topk=cfg["model"].get("rag_stage1_topk", 3),
        rag_stage2_topk=cfg["model"].get("rag_stage2_topk", 1),
        use_two_stage_rag=True,
        cot_model=cot_model,
        cot_temperature=cot_temperature,
        cot_max_new_tokens=cot_max_new_tokens,
    )
elif mode == "cot_only":
    cfg["model"].update(
        use_rag_cot=True,
        cot_only=True,
        rag_topk=0,
        use_two_stage_rag=False,
        cot_model=cot_model,
        cot_temperature=cot_temperature,
        cot_max_new_tokens=cot_max_new_tokens,
    )
elif mode == "rag_only":
    cfg["model"].update(
        use_rag_cot=True,
        cot_only=True,
        rag_topk=rag_topk,
        use_two_stage_rag=False,
        cot_model="",
        cot_temperature=0.0,
        cot_max_new_tokens=cot_max_new_tokens,
    )
else:
    raise SystemExit(f"unknown ablation mode: {mode}")

tmp_path.write_text("#type: args\n\n" + yaml.safe_dump(cfg, sort_keys=False))
if os.environ.get("DRY_RUN") != "1":
    print(f"wrote {tmp_path} for {os.environ['DATASET_KEY']}:{mode}")
PY
}

run_case() {
  local dataset_key="$1"
  local data_path="$2"
  local base_config="$3"
  local seq_len="$4"
  local pred_len="$5"
  local freq="$6"
  local text_len="$7"
  local seed="$8"

  contains_word "${dataset_key}" "${DATASET_FILTER}" || return 0

  local modes=(baseline rag_cot cot_only rag_only)
  for mode in "${modes[@]}"; do
    contains_word "${mode}" "${MODE_FILTER}" || continue

    local tmp_config="_tmp_ablation_${dataset_key}_${mode}.yaml"
    write_mode_config "${dataset_key}" "${mode}" "${base_config}" "${tmp_config}"

    local cmd=(
      python -u exe_forecasting.py
      --root_path "${ROOT_PATH}"
      --data_path "${data_path}"
      --config "${tmp_config}"
      --seq_len "${seq_len}"
      --pred_len "${pred_len}"
      --freq "${freq}"
      --nsample "${NSAMPLE}"
      --sample_steps_override "${SAMPLE_STEPS}"
      --guide_w "${GUIDE_W}"
      --device "${DEVICE}"
      --valid_interval "${VALID_INTERVAL}"
    )
    if [[ -n "${text_len}" ]]; then
      cmd+=(--text_len "${text_len}")
    fi
    if [[ -n "${seed}" ]]; then
      cmd+=(--seed "${seed}")
    fi

    if [[ "${DRY_RUN}" != "1" ]]; then
      echo "=== ${dataset_key} | ${mode} | ${base_config} ==="
    fi
    run_cmd "${cmd[@]}"
  done
}

run_case traffic "Traffic/Traffic.csv" "traffic_36_12_mainline.yaml" 36 12 m 36 ""
run_case socialgood "SocialGood/SocialGood.csv" "socialgood_36_12_mainline.yaml" 36 12 m "" ""
run_case health "Health_US/Health_US.csv" "health_96_12_mainline.yaml" 96 12 w "" ""
run_case environment "Environment/Environment.csv" "environment_336_48_mainline.yaml" 336 48 d "" 2021
run_case energy "Energy/Energy.csv" "energy_96_12_mainline.yaml" 96 12 w 36 ""
run_case economy "Economy/Economy.csv" "economy_36_12_mainline.yaml" 36 12 m 36 ""
run_case climate "Climate/Climate.csv" "climate_96_12_mainline.yaml" 96 12 w "" ""
run_case agriculture "Agriculture/Agriculture.csv" "agriculture_36_12_mainline.yaml" 36 12 m 36 ""
