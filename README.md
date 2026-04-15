# MCD-TSF
Implementation for our paper "Multimodal Conditioned Diffusive Time Series Forecasting".

## Requirements
```bash
pip install -r requirements.txt
```

## Datasets
Public benchmark: [Time-MMD](https://github.com/adityalab/time-mmd)

## Experiment
```bash
bash ./run.sh
```

## Retrieval-augmented / CoT guidance
- `--use_rag_cot`: enable text guidance with TF-IDF retrieval + CoT generation.
- `--cot_only`: disable retrieval; generate CoT from numeric summary + raw text only.
- Common knobs: `--rag_topk`, `--cot_model`, `--cot_max_new_tokens`, `--cot_temperature`,
  `--cot_cache_size`, `--cot_device`.
- Default behavior: CoT text is concatenated with raw reports and fed into the text encoder.
- When `--trend_cfg` is enabled, CoT is parsed into a trend prior and no longer concatenated
  into text; it is used to modulate CFG weights along the diffusion path.
- Optional quantized loading: `--cot_load_in_8bit` / `--cot_load_in_4bit` (requires bitsandbytes).

## Multi-resolution auxiliary loss (latest mainline)
Introduce a lightweight, multi-horizon supervision term inspired by multi-resolution forecasting.
The model is encouraged to fit several horizons within the prediction window, with the final
horizon set resolved in this order:
1. explicit `train.multi_res_horizons`
2. training-split ACF statistics
3. ratio fallback based on `pred_len`

This adds training signal without changing the model architecture or the original
train / val / test workflow.
- Config keys (under `train`):
  - `multi_res_horizons`: explicit horizons to supervise (clipped by `pred_len`).
  - `multi_res_use_stat_horizons`: when `multi_res_horizons` is null, derive horizons from training-set ACF.
  - `multi_res_acf_drop_threshold`: first ACF decay threshold.
  - `multi_res_acf_zero_threshold`: first near-zero ACF threshold.
  - `multi_res_acf_max_lag`: max lag used for ACF analysis; `null` means auto.
  - `multi_res_acf_num_samples`: number of sampled training windows for ACF estimation.
  - `multi_res_loss_weight`: weight for the auxiliary loss (set to 0 to disable).
  - `multi_res_use_huber`: use Huber loss (recommended for stability).
  - `multi_res_huber_delta`: uniform fallback delta for Huber loss.
  - `multi_res_huber_deltas`: per-horizon delta list; falls back safely if lengths mismatch.
  - `multi_res_dynamic`: enable dynamic multi-horizon weighting.
  - `multi_res_dynamic_by_t`: adapt horizon weights by diffusion step.
  - `multi_res_dynamic_by_epoch`: keep epoch-aware confidence in the dynamic weighting mix.
  - `multi_res_dynamic_by_trend`: adapt horizon weights by trend prior strength/volatility.
  - `multi_res_dynamic_min_weight`: lower bound for each dynamic horizon weight.
  - `multi_res_progressive`: progressively unlock horizons across training epochs.
  - `multi_res_ema_alpha`: EMA update factor for per-horizon difficulty.
  - `multi_res_difficulty_weight`: mixing weight between confidence-based weights and EMA difficulty weights.
  - `lr_warmup_epochs`: number of warmup epochs before the original `MultiStepLR` schedule takes over.
  - `max_grad_norm`: gradient clipping threshold; disable when `<= 0`.
  - `adaptive_noise_scale`: optional interface for augmentation noise scaling; keep `0.0` for the mainline run.
- Config keys (under `dataset` or `data`):
  - `aug_noise_std`: light Gaussian noise for training lookback augmentation.
  - `aug_time_warp_prob`: kept for backward compatibility; currently triggers local segment scaling, not strict time warping.
  - `aug_segment_scale_std`: std used by the local segment scaling augmentation.
- Example (YAML):
  ```yaml
  train:
    multi_res_horizons: null
    multi_res_use_stat_horizons: true
    multi_res_acf_drop_threshold: 0.5
    multi_res_acf_zero_threshold: 0.1
    multi_res_acf_max_lag: null
    multi_res_acf_num_samples: 128
    multi_res_loss_weight: 0.1
    multi_res_use_huber: true
    multi_res_huber_delta: 1.0
    multi_res_huber_deltas: [0.5, 0.8, 1.0, 1.5]
    multi_res_dynamic: true
    multi_res_dynamic_by_t: true
    multi_res_dynamic_by_epoch: true
    multi_res_dynamic_by_trend: true
    multi_res_dynamic_min_weight: 0.2
    multi_res_progressive: true
    multi_res_ema_alpha: 0.05
    multi_res_difficulty_weight: 0.5
    lr_warmup_epochs: 10
    max_grad_norm: 1.0
    adaptive_noise_scale: 0.0

  dataset:
    aug_noise_std: 0.01
    aug_time_warp_prob: 0.3
    aug_segment_scale_std: 0.1
  ```

## Economy mainline config
- Latest single-domain mainline config: `config/economy_36_12_mainline.yaml`
- Default target:
  - domain: `Economy`
  - full latest mainline on: adaptive horizons, progressive curriculum, EMA difficulty,
    per-horizon Huber delta, gradient clipping, LR warmup, lookback augmentation
  - `adaptive_noise_scale`: kept off by default

## Two-stage RAG (minimal change enhancement)
- Switch: `--use_two_stage_rag` (off by default to preserve one-shot behavior).
- Stage-1: reuse the original one-shot query to retrieve E0.
- Trend hypothesis: generated by CoT; on failure, uses a numeric template.
- Stage-2: convert the trend hypothesis into a natural language query, retrieve E1,
  then merge E1 with E0 (deduplicate, keep top-k).
- Stability: gate/fallback to one-shot if raw text is empty or retrieval fails.
- Output template is fixed:
  `[RAW TEXT] / [NUMERICAL SUMMARY] / [TREND HYPOTHESIS] / [RETRIEVED EVIDENCE - REFINED]`.
- Tunables: `--rag_stage1_topk`, `--rag_stage2_topk`, `--two_stage_gate`, `--trend_slope_eps`.
- Numeric stats (`slope/std/mean`) are appended to `key_factors` to improve Stage-2 retrieval.

## Trend-aware CFG (CoT -> trend prior)
CoT is promoted from a text condition to a diffusion-path modulation signal.
- Switch: `--trend_cfg`
- Time schedule: `--trend_cfg_power`, `--trend_time_floor`
- Trend mapping:
  - strength: `1 + trend_strength_scale * (strength - 1)`
  - volatility: `1 / (1 + trend_volatility_scale * volatility)`
- Tunables: `--trend_strength_scale`, `--trend_volatility_scale`, `--trend_time_floor`,
  `--trend_cfg_random`
- Save priors: `--save_trend_prior` outputs `trend_priors.npy` and `trend_text_marks.npy`

## Guide weight sweep
- `--guide_w -1` triggers the built-in sweep list (includes `4.5`).
- To override, pass a fixed `--guide_w` or run your own loop.

## Debug
Use `debug_two_stage_rag.py` to inspect Q1/E0/z0/Q2/E1 and the composed text preview.

## Full command example
```bash
python -u exe_forecasting.py \
  --root_path ../Time-MMD-main \
  --data_path Traffic/Traffic.csv \
  --config traffic_36_12.yaml \
  --seq_len 36 --pred_len 12 --text_len 36 --freq m \
  --use_rag_cot --use_two_stage_rag \
  --trend_cfg --trend_cfg_power 1.0 \
  --trend_strength_scale 0.35 --trend_volatility_scale 1.0 --trend_time_floor 0.30 \
  --guide_w -1
```

## Scripts
- Full run with trend CFG: `scripts/run_all_datasets_trendcfg.sh`
- Trend CFG grid search: `scripts/train_trendcfg_grid.sh`

## Acknowledgements
Codes are based on:
[CSDI](https://github.com/ermongroup/CSDI),
[Time-LLM](https://github.com/KimMeen/Time-LLM/tree/main),
[MM-TSF](https://github.com/adityalab/time-mmd),
[Autoformer](https://github.com/thuml/Autoformer)
