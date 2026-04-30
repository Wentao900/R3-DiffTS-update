#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
DEVICE=${DEVICE:-cuda:0}
NSAMPLE=${NSAMPLE:-15}
SAMPLE_STEPS=${SAMPLE_STEPS:-}
VALID_INTERVAL=${VALID_INTERVAL:-1}
GUIDE_W=${GUIDE_W:--1}
GUIDE_LIST=${GUIDE_LIST:-"0,0.5,1.0,1.5,2.0"}
DRY_RUN=${DRY_RUN:-0}
TMP_DIR=${TMP_DIR:-config/_method_ablation}
DATASETS=${DATASETS:-"Traffic SocialGood Economy Agriculture Energy Climate Environment Health_US"}
ABLATIONS=${ABLATIONS:-"full numeric_only raw_text_only one_stage_rag_cot two_stage_rag_cot wop_pattern_residual wop_pattern_text_evidence wop_controller wop_reliability wop_multires wop_aux_heads wop_calibrator"}
EPOCHS=${EPOCHS:-}
BATCH_SIZE=${BATCH_SIZE:-}
TEXT_DROP_PROB=${TEXT_DROP_PROB:-}

mkdir -p "${TMP_DIR}" logs

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

make_config() {
  local base_config="$1"
  local ablation="$2"
  local out_config="$3"

  python - "$base_config" "$ablation" "$out_config" "$EPOCHS" "$BATCH_SIZE" <<'PY'
import sys
import yaml

base_config, ablation, out_config, epochs_arg, batch_size_arg = sys.argv[1:6]

with open("config/" + base_config, "r") as f:
    cfg = yaml.safe_load(f)

model = cfg.setdefault("model", {})
train = cfg.setdefault("train", {})
diff = cfg.setdefault("diffusion", {})
dataset = cfg.setdefault("dataset", {})

if epochs_arg:
    train["epochs"] = int(epochs_arg)
if batch_size_arg:
    train["batch_size"] = int(batch_size_arg)

# Normalize defaults so every ablation starts from a comparable "full" setting.
model["use_rag_cot"] = bool(model.get("use_rag_cot", True))
model["cot_only"] = bool(model.get("cot_only", False))
model["use_two_stage_rag"] = bool(model.get("use_two_stage_rag", True))
model["pattern_residual_diffusion"] = bool(model.get("pattern_residual_diffusion", True))
model["pattern_text_evidence"] = bool(model.get("pattern_text_evidence", True))
model["use_forecast_policy_controller"] = bool(model.get("use_forecast_policy_controller", True))
model["use_aux_forecast_heads"] = bool(model.get("use_aux_forecast_heads", True))
model["use_coarse_forecast_head"] = bool(model.get("use_coarse_forecast_head", True))
model["use_uncertainty_head"] = bool(model.get("use_uncertainty_head", True))
model["pattern_reliability"] = bool(model.get("pattern_reliability", True))
model["pattern_aux_reliability"] = bool(model.get("pattern_aux_reliability", True))

train["multi_res_loss_weight"] = float(train.get("multi_res_loss_weight", 0.1))
train["multi_res_segment_loss"] = bool(train.get("multi_res_segment_loss", True))
train["multi_res_reliability_weight"] = float(train.get("multi_res_reliability_weight", 1.0))
train["coarse_forecast_weight"] = float(train.get("coarse_forecast_weight", 0.05))
train["uncertainty_forecast_weight"] = float(train.get("uncertainty_forecast_weight", 0.02))
train["forecast_calibrator"] = bool(train.get("forecast_calibrator", False))
train["multi_res_dynamic_by_trend"] = bool(train.get("multi_res_dynamic_by_trend", True))

diff["trend_cfg"] = bool(diff.get("trend_cfg", False))

dataset["ablation_name"] = ablation

if ablation == "full":
    pass
elif ablation == "numeric_only":
    model["use_rag_cot"] = False
    model["cot_only"] = False
    model["use_two_stage_rag"] = False
    model["pattern_text_evidence"] = False
    model["use_forecast_policy_controller"] = False
    model["pattern_reliability"] = False
    model["pattern_aux_reliability"] = False
    train["multi_res_dynamic_by_trend"] = False
    diff["trend_cfg"] = False
elif ablation == "raw_text_only":
    model["use_rag_cot"] = False
    model["cot_only"] = False
    model["use_two_stage_rag"] = False
    diff["trend_cfg"] = False
elif ablation == "one_stage_rag_cot":
    model["use_rag_cot"] = True
    model["cot_only"] = False
    model["use_two_stage_rag"] = False
elif ablation == "two_stage_rag_cot":
    model["use_rag_cot"] = True
    model["cot_only"] = False
    model["use_two_stage_rag"] = True
elif ablation == "wop_pattern_residual":
    model["pattern_residual_diffusion"] = False
elif ablation == "wop_pattern_text_evidence":
    model["pattern_text_evidence"] = False
elif ablation == "wop_controller":
    model["use_forecast_policy_controller"] = False
elif ablation == "wop_reliability":
    model["use_forecast_policy_controller"] = False
    model["pattern_reliability"] = False
    model["pattern_aux_reliability"] = False
    train["multi_res_segment_loss"] = False
    train["multi_res_reliability_weight"] = 0.0
    train["multi_res_dynamic_by_trend"] = False
elif ablation == "wop_multires":
    train["multi_res_loss_weight"] = 0.0
    train["multi_res_segment_loss"] = False
    train["multi_res_reliability_weight"] = 0.0
    train["multi_res_dynamic_by_trend"] = False
elif ablation == "wop_aux_heads":
    model["use_aux_forecast_heads"] = False
    model["use_coarse_forecast_head"] = False
    model["use_uncertainty_head"] = False
    train["coarse_forecast_weight"] = 0.0
    train["uncertainty_forecast_weight"] = 0.0
elif ablation == "wop_calibrator":
    train["forecast_calibrator"] = False
elif ablation == "trend_cfg_on":
    diff["trend_cfg"] = True
elif ablation == "wop_trend_cfg":
    diff["trend_cfg"] = False
else:
    raise SystemExit(f"unknown ablation: {ablation}")

with open(out_config, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

resolve_dataset() {
  local dataset="$1"
  case "${dataset}" in
    Traffic)
      DATA_PATH="Traffic/Traffic.csv"
      BASE_CONFIG="traffic_36_12_mainline.yaml"
      SEQ_LEN=36
      PRED_LEN=12
      FREQ=m
      TEXT_LEN=36
      SEED=""
      ;;
    SocialGood)
      DATA_PATH="SocialGood/SocialGood.csv"
      BASE_CONFIG="socialgood_36_12_mainline.yaml"
      SEQ_LEN=36
      PRED_LEN=12
      FREQ=m
      TEXT_LEN=""
      SEED=""
      ;;
    Economy)
      DATA_PATH="Economy/Economy.csv"
      BASE_CONFIG="economy_36_12_mainline.yaml"
      SEQ_LEN=36
      PRED_LEN=12
      FREQ=m
      TEXT_LEN=36
      SEED=""
      ;;
    Agriculture)
      DATA_PATH="Agriculture/Agriculture.csv"
      BASE_CONFIG="agriculture_36_12_mainline.yaml"
      SEQ_LEN=36
      PRED_LEN=12
      FREQ=m
      TEXT_LEN=36
      SEED=""
      ;;
    Energy)
      DATA_PATH="Energy/Energy.csv"
      BASE_CONFIG="energy_96_12_mainline.yaml"
      SEQ_LEN=96
      PRED_LEN=12
      FREQ=w
      TEXT_LEN=36
      SEED=""
      ;;
    Climate)
      DATA_PATH="Climate/Climate.csv"
      BASE_CONFIG="climate_96_12_mainline.yaml"
      SEQ_LEN=96
      PRED_LEN=12
      FREQ=w
      TEXT_LEN=""
      SEED=""
      ;;
    Environment)
      DATA_PATH="Environment/Environment.csv"
      BASE_CONFIG="environment_336_48_mainline.yaml"
      SEQ_LEN=336
      PRED_LEN=48
      FREQ=d
      TEXT_LEN=""
      SEED=2021
      ;;
    Health_US)
      DATA_PATH="Health_US/Health_US.csv"
      BASE_CONFIG="health_96_12_mainline.yaml"
      SEQ_LEN=96
      PRED_LEN=12
      FREQ=w
      TEXT_LEN=""
      SEED=""
      ;;
    *)
      echo "unknown dataset: ${dataset}" >&2
      return 1
      ;;
  esac
}

run_case() {
  local dataset="$1"
  local ablation="$2"
  resolve_dataset "${dataset}"

  local tmp_config="${TMP_DIR}/${dataset}_${ablation}.yaml"
  make_config "${BASE_CONFIG}" "${ablation}" "${tmp_config}"

  local cmd=(
    python -u exe_forecasting.py
    --root_path "${ROOT_PATH}"
    --data_path "${DATA_PATH}"
    --config "${tmp_config#config/}"
    --seq_len "${SEQ_LEN}"
    --pred_len "${PRED_LEN}"
    --freq "${FREQ}"
    --guide_w "${GUIDE_W}"
    --guide_list "${GUIDE_LIST}"
    --nsample "${NSAMPLE}"
    --device "${DEVICE}"
    --valid_interval "${VALID_INTERVAL}"
  )

  if [[ -n "${SAMPLE_STEPS}" ]]; then
    cmd+=(--sample_steps_override "${SAMPLE_STEPS}")
  fi
  if [[ -n "${TEXT_LEN}" ]]; then
    cmd+=(--text_len "${TEXT_LEN}")
  fi
  if [[ -n "${SEED}" ]]; then
    cmd+=(--seed "${SEED}")
  fi
  if [[ -n "${TEXT_DROP_PROB}" ]]; then
    cmd+=(--text_drop_prob "${TEXT_DROP_PROB}")
  elif [[ "${ablation}" == "numeric_only" ]]; then
    cmd+=(--text_drop_prob "1.0")
  else
    cmd+=(--text_drop_prob "0.0")
  fi

  local log_file="logs/method_ablation_${dataset}_${ablation}.log"
  echo "=== ${dataset} | ${ablation} ==="
  if [[ "${DRY_RUN}" == "1" ]]; then
    run_cmd "${cmd[@]}"
  else
    "${cmd[@]}" | tee "${log_file}"
  fi
}

for dataset in ${DATASETS}; do
  for ablation in ${ABLATIONS}; do
    run_case "${dataset}" "${ablation}"
  done
done
