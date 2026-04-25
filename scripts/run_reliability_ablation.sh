#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
DEVICE=${DEVICE:-cuda:0}
NSAMPLE=${NSAMPLE:-15}
SAMPLE_STEPS=${SAMPLE_STEPS:-}
VALID_INTERVAL=${VALID_INTERVAL:-1}
GUIDE_W=${GUIDE_W:--1}
GUIDE_LIST=${GUIDE_LIST:-"0,0.5,1.0,1.5,2.0"}
DATASETS=${DATASETS:-"Traffic SocialGood Economy Agriculture"}
MODES=${MODES:-"E0 E1 E2 E3 E4 E5"}
DRY_RUN=${DRY_RUN:-0}
TMP_DIR=${TMP_DIR:-config/_tmp_reliability_ablation}

mkdir -p "${TMP_DIR}" logs

make_config() {
  local base_config="$1"
  local mode="$2"
  local out_config="$3"
  python - "$base_config" "$mode" "$out_config" <<'PY'
import sys
import yaml

base_config, mode, out_config = sys.argv[1:4]
with open("config/" + base_config, "r") as f:
    cfg = yaml.safe_load(f)

train = cfg.setdefault("train", {})
model = cfg.setdefault("model", {})
diff = cfg.setdefault("diffusion", {})

train.setdefault("multi_res_acf_reliability_threshold", 0.2)
train.setdefault("multi_res_acf_reliability_temperature", 0.05)
model.setdefault("pattern_reliability_threshold", 0.35)
model.setdefault("pattern_reliability_temperature", 0.1)
model.setdefault("pattern_reliability_min", 0.05)
model.setdefault("pattern_reliability_max", 1.0)

if mode == "E0":
    train["multi_res_segment_loss"] = False
    train["multi_res_reliability_weight"] = 0.0
    train["multi_res_difficulty_inverse"] = False
    model["pattern_reliability"] = False
    model["pattern_aux_reliability"] = False
    model["pattern_text_drop_prob"] = 0.0
elif mode == "E1":
    train["multi_res_segment_loss"] = False
    train["multi_res_reliability_weight"] = 1.0
    train["multi_res_difficulty_inverse"] = True
    train["multi_res_difficulty_gamma"] = 0.5
    model["pattern_reliability"] = False
    model["pattern_aux_reliability"] = False
    model["pattern_text_drop_prob"] = 0.0
elif mode == "E2":
    train["multi_res_segment_loss"] = True
    train["multi_res_reliability_weight"] = 1.0
    train["multi_res_difficulty_inverse"] = True
    train["multi_res_difficulty_gamma"] = 0.5
    model["pattern_reliability"] = False
    model["pattern_aux_reliability"] = False
    model["pattern_text_drop_prob"] = 0.0
elif mode == "E3":
    train["multi_res_segment_loss"] = True
    train["multi_res_reliability_weight"] = 1.0
    train["multi_res_difficulty_inverse"] = True
    train["multi_res_difficulty_gamma"] = 0.5
    model["pattern_reliability"] = True
    model["pattern_aux_reliability"] = False
    model["pattern_text_drop_prob"] = 0.0
elif mode == "E4":
    train["multi_res_segment_loss"] = True
    train["multi_res_reliability_weight"] = 1.0
    train["multi_res_difficulty_inverse"] = True
    train["multi_res_difficulty_gamma"] = 0.5
    model["pattern_reliability"] = True
    model["pattern_aux_reliability"] = True
    model["pattern_text_drop_prob"] = 0.0
elif mode == "E5":
    train["multi_res_segment_loss"] = True
    train["multi_res_reliability_weight"] = 1.0
    train["multi_res_difficulty_inverse"] = True
    train["multi_res_difficulty_gamma"] = 0.5
    model["pattern_reliability"] = True
    model["pattern_aux_reliability"] = True
    model["pattern_text_drop_prob"] = float(diff.get("c_mask_prob", 0.0))
else:
    raise SystemExit(f"unknown ablation mode: {mode}")

with open(out_config, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_case() {
  local dataset="$1"
  local mode="$2"
  local data_path config seq_len pred_len freq text_len seed
  seed=""
  case "${dataset}" in
    Traffic) data_path="Traffic/Traffic.csv"; config="traffic_36_12_mainline.yaml"; seq_len=36; pred_len=12; freq=m; text_len=36 ;;
    SocialGood) data_path="SocialGood/SocialGood.csv"; config="socialgood_36_12_mainline.yaml"; seq_len=36; pred_len=12; freq=m; text_len="" ;;
    Economy) data_path="Economy/Economy.csv"; config="economy_36_12_mainline.yaml"; seq_len=36; pred_len=12; freq=m; text_len=36 ;;
    Agriculture) data_path="Agriculture/Agriculture.csv"; config="agriculture_36_12_mainline.yaml"; seq_len=36; pred_len=12; freq=m; text_len=36 ;;
    *) echo "unknown dataset: ${dataset}" >&2; return 1 ;;
  esac

  local tmp_config="${TMP_DIR}/${dataset}_${mode}.yaml"
  make_config "${config}" "${mode}" "${tmp_config}"
  local cmd=(
    python -u exe_forecasting.py
    --root_path "${ROOT_PATH}"
    --data_path "${data_path}"
    --config "${tmp_config#config/}"
    --seq_len "${seq_len}"
    --pred_len "${pred_len}"
    --freq "${freq}"
    --guide_w "${GUIDE_W}"
    --guide_list "${GUIDE_LIST}"
    --nsample "${NSAMPLE}"
    --device "${DEVICE}"
    --valid_interval "${VALID_INTERVAL}"
  )
  if [[ -n "${SAMPLE_STEPS}" ]]; then
    cmd+=(--sample_steps_override "${SAMPLE_STEPS}")
  fi
  if [[ -n "${text_len}" ]]; then
    cmd+=(--text_len "${text_len}")
  fi
  if [[ -n "${seed}" ]]; then
    cmd+=(--seed "${seed}")
  fi
  local log_file="logs/reliability_${dataset}_${mode}.log"
  echo "=== ${dataset} ${mode} -> ${log_file} ==="
  if [[ "${DRY_RUN}" == "1" ]]; then
    run_cmd "${cmd[@]}"
  else
    "${cmd[@]}" | tee "${log_file}"
  fi
}

for dataset in ${DATASETS}; do
  for mode in ${MODES}; do
    run_case "${dataset}" "${mode}"
  done
done
