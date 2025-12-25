#!/usr/bin/env bash
# Train + guide_w sweep for trend_cfg modulation settings.
# Default is a 3-point quick probe; set USE_FULL_GRID=1 for full grid.
set -euo pipefail

ROOT_PATH=${ROOT_PATH:-../Time-MMD-main}
DATA_PATH=${DATA_PATH:-Traffic/Traffic.csv}
CONFIG=${CONFIG:-traffic_36_12.yaml}
SEQ_LEN=${SEQ_LEN:-36}
PRED_LEN=${PRED_LEN:-12}
TEXT_LEN=${TEXT_LEN:-36}
FREQ=${FREQ:-m}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-8}
SAMPLE_STEPS=${SAMPLE_STEPS:-120}
NSAMPLE=${NSAMPLE:-5}
GUIDE_W=${GUIDE_W:--1}
RAG_TOPK=${RAG_TOPK:-1}
COT_MODEL=${COT_MODEL:-./Qwen2.5-7B-Instruct}
COT_TEMPERATURE=${COT_TEMPERATURE:-0.55}
COT_MAX_NEW_TOKENS=${COT_MAX_NEW_TOKENS:-96}

TREND_POWER=${TREND_POWER:-1.0}
USE_FULL_GRID=${USE_FULL_GRID:-0}
TREND_PRESETS=${TREND_PRESETS:-"2.0,0.5,0.1 2.5,0.5,0.2 3.0,0.3,0.2"}
TREND_STRENGTH_SCALES=${TREND_STRENGTH_SCALES:-"1.0 2.0 3.0"}
TREND_VOLATILITY_SCALES=${TREND_VOLATILITY_SCALES:-"0.3 0.5 0.7 1.0"}
TREND_TIME_FLOORS=${TREND_TIME_FLOORS:-"0.0 0.1 0.2"}
SAVE_TREND_PRIOR=${SAVE_TREND_PRIOR:-1}

LABEL=${LABEL:-trendcfg_grid}
LOG_DIR=${LOG_DIR:-logs}

mkdir -p "${LOG_DIR}"
RUN_LIST="${LOG_DIR}/${LABEL}_runs.csv"
echo "label,folder,strength_scale,volatility_scale,time_floor,trend_power" > "${RUN_LIST}"

trend_prior_flag=""
if [[ "${SAVE_TREND_PRIOR}" -eq 1 ]]; then
  trend_prior_flag="--save_trend_prior"
fi

run_once() {
  local s_scale="$1"
  local v_scale="$2"
  local t_floor="$3"
  local label="s${s_scale}_v${v_scale}_f${t_floor}_p${TREND_POWER}"
  local log_file="${LOG_DIR}/${LABEL}_${label}.log"
  echo "=== ${label} ===" | tee "${log_file}"
  python -u exe_forecasting.py \
    --root_path "${ROOT_PATH}" \
    --data_path "${DATA_PATH}" \
    --config "${CONFIG}" \
    --seq_len "${SEQ_LEN}" \
    --pred_len "${PRED_LEN}" \
    --text_len "${TEXT_LEN}" \
    --freq "${FREQ}" \
    --sample_steps_override "${SAMPLE_STEPS}" \
    --nsample "${NSAMPLE}" \
    --use_rag_cot \
    --rag_topk "${RAG_TOPK}" \
    --cot_model "${COT_MODEL}" \
    --cot_temperature "${COT_TEMPERATURE}" \
    --cot_max_new_tokens "${COT_MAX_NEW_TOKENS}" \
    --guide_w "${GUIDE_W}" \
    --lr 0.0001 \
    --valid_interval 1 \
    --dropout 0.0 \
    --attn_drop 0.0 \
    --c_mask_prob 0.2 \
    --beta_end 0.7 \
    --time_weight 0.1 \
    --trend_cfg \
    --trend_cfg_power "${TREND_POWER}" \
    --trend_strength_scale "${s_scale}" \
    --trend_volatility_scale "${v_scale}" \
    --trend_time_floor "${t_floor}" \
    ${trend_prior_flag} | tee -a "${log_file}"
  latest=$(ls -td save/forecasting_Traffic_* 2>/dev/null | head -n1 | xargs -r basename)
  echo "${label},${latest},${s_scale},${v_scale},${t_floor},${TREND_POWER}" >> "${RUN_LIST}"
}

if [[ "${USE_FULL_GRID}" -eq 1 ]]; then
  for s_scale in ${TREND_STRENGTH_SCALES}; do
    for v_scale in ${TREND_VOLATILITY_SCALES}; do
      for t_floor in ${TREND_TIME_FLOORS}; do
        run_once "${s_scale}" "${v_scale}" "${t_floor}"
      done
    done
  done
else
  for preset in ${TREND_PRESETS}; do
    IFS=',' read -r s_scale v_scale t_floor <<< "${preset}"
    run_once "${s_scale}" "${v_scale}" "${t_floor}"
  done
fi

echo "Done. Run list: ${RUN_LIST}"
