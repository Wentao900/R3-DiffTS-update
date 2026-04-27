#!/usr/bin/env bash
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
DATASETS=${DATASETS:-"Traffic SocialGood Economy Agriculture Energy Climate Environment Health_US"}
NSAMPLE=${NSAMPLE:-15}
SAMPLE_STEPS=${SAMPLE_STEPS:-}
DEVICE=${DEVICE:-cuda:0}
VALID_INTERVAL=${VALID_INTERVAL:-1}
DRY_RUN=${DRY_RUN:-0}
CONFIG_DIR=${CONFIG_DIR:-config/_final_method}

mkdir -p "${CONFIG_DIR}"

COMMON_ARGS=(
  --root_path "${ROOT_PATH}"
  --nsample "${NSAMPLE}"
  --device "${DEVICE}"
  --valid_interval "${VALID_INTERVAL}"
)

if [[ -n "${SAMPLE_STEPS}" ]]; then
  COMMON_ARGS+=(--sample_steps_override "${SAMPLE_STEPS}")
fi

make_final_config() {
  local base_config="$1"
  local final_config="$2"
  python - "$base_config" "$final_config" <<'PY'
import os
import sys
import yaml

base_path, final_path = sys.argv[1], sys.argv[2]
with open(base_path, "r") as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("model", {})
cfg.setdefault("train", {})
cfg.setdefault("dataset", {})
cfg["model"]["guide_mode"] = "auto"
cfg["model"]["final_method"] = True
cfg["model"]["pattern_residual_diffusion"] = True
cfg["model"]["pattern_text_evidence"] = True
cfg["model"]["use_forecast_policy_controller"] = True
cfg["model"]["use_aux_forecast_heads"] = True
cfg["model"]["use_coarse_forecast_head"] = True
cfg["model"]["use_uncertainty_head"] = True
cfg["model"]["coarse_forecast_blend"] = 0.15
cfg["model"]["coarse_forecast_factor"] = 4
cfg["model"]["aux_forecast_hidden_dim"] = 128
cfg["model"]["forecast_policy_controller"] = {
    "enabled": True,
    "fine_topk_ratio": 0.35,
    "max_fine_ratio": 0.6,
    "min_fine_points": 1,
    "rag_invalid_scale": 0.25,
    "unclear_trend_scale": 0.5,
    "sample_budget_floor": 0.5,
    "sample_budget_ceiling": 1.0,
    "turning_center": 0.35,
    "turning_width": 0.18,
}
cfg["train"]["multi_res_segment_loss"] = True
cfg["train"]["multi_res_use_stat_horizons"] = True
cfg["train"].pop("multi_res_difficulty_gamma", None)
cfg["train"]["coarse_forecast_weight"] = 0.05
cfg["train"]["uncertainty_forecast_weight"] = 0.02
cfg["train"]["forecast_point_estimator"] = "auto"
cfg["train"]["forecast_calibrator"] = True
cfg["train"]["forecast_calibrator_ridge"] = 1.0
cfg["train"]["forecast_calibrator_min_gain"] = 0.02
cfg["train"]["forecast_calibrator_max_strength"] = 0.25
cfg["train"]["forecast_calibrator_max_batches"] = 0
cfg["train"]["forecast_calibrator_holdout_fraction"] = 0.35
cfg["train"]["forecast_calibrator_residual_clip_quantile"] = 0.95
cfg["train"]["forecast_calibrator_include_timestamp"] = False
cfg["train"]["forecast_calibrator_season_acf_threshold"] = 0.3
cfg["train"]["forecast_calibrator_season_max_periods"] = 8
cfg["train"]["forecast_calibrator_block_count"] = 4
cfg["train"]["forecast_calibrator_block_min_pos_ratio"] = 0.5
cfg["train"]["forecast_calibrator_reliability_floor"] = 0.5
cfg["train"]["forecast_calibrator_reliability_threshold"] = 0.4
cfg["train"]["forecast_calibrator_reliability_temperature"] = 0.15
cfg["train"]["forecast_calibrator_solver"] = "nnls"
cfg["train"]["forecast_calibrator_fused_smoothing"] = 0.25
cfg["train"]["forecast_calibrator_force_mean_when_off"] = True
domain = os.path.basename(base_path).split("_")[0].lower()
if domain in {"economy", "energy", "agriculture"}:
    cfg["dataset"]["use_all_numeric_features"] = True
os.makedirs(os.path.dirname(final_path), exist_ok=True)
with open(final_path, "w") as f:
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
  local data_path="$2"
  local base_config="$3"
  local seq_len="$4"
  local pred_len="$5"
  local freq="$6"
  local text_len="${7:-}"
  local seed="${8:-}"
  local final_config="${CONFIG_DIR}/${base_config}"

  make_final_config "config/${base_config}" "${final_config}"
  local cmd=(
    python -u exe_forecasting.py
    --data_path "${data_path}"
    --config "_final_method/${base_config}"
    --seq_len "${seq_len}"
    --pred_len "${pred_len}"
    --freq "${freq}"
    "${COMMON_ARGS[@]}"
  )
  if [[ -n "${text_len}" ]]; then
    cmd+=(--text_len "${text_len}")
  fi
  if [[ -n "${seed}" ]]; then
    cmd+=(--seed "${seed}")
  fi
  printf '[FINAL] %s\n' "${dataset}"
  run_cmd "${cmd[@]}"
}

for dataset in ${DATASETS}; do
  case "${dataset}" in
    Traffic)
      run_case "Traffic" "Traffic/Traffic.csv" "traffic_36_12_mainline.yaml" 36 12 m 36
      ;;
    SocialGood)
      run_case "SocialGood" "SocialGood/SocialGood.csv" "socialgood_36_12_mainline.yaml" 36 12 m
      ;;
    Economy)
      run_case "Economy" "Economy/Economy.csv" "economy_36_12_mainline.yaml" 36 12 m 36
      ;;
    Agriculture)
      run_case "Agriculture" "Agriculture/Agriculture.csv" "agriculture_36_12_mainline.yaml" 36 12 m 36
      ;;
    Energy)
      run_case "Energy" "Energy/Energy.csv" "energy_96_12_mainline.yaml" 96 12 w 36
      ;;
    Climate)
      run_case "Climate" "Climate/Climate.csv" "climate_96_12_mainline.yaml" 96 12 w
      ;;
    Environment)
      run_case "Environment" "Environment/Environment.csv" "environment_336_48_mainline.yaml" 336 48 d "" 2021
      ;;
    Health_US)
      run_case "Health_US" "Health_US/Health_US.csv" "health_96_12_mainline.yaml" 96 12 w
      ;;
    Health_AFR)
      printf '[SKIP] Health_AFR is intentionally excluded from FINAL runs.\n'
      ;;
    *)
      printf 'Unknown dataset: %s\n' "${dataset}" >&2
      exit 2
      ;;
  esac
done
