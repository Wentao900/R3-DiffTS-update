# MCD-TSF（中文版）
本仓库实现论文 **“Multimodal Conditioned Diffusive Time Series Forecasting”**。

## 环境依赖
```bash
pip install -r requirements.txt
```

## 数据集
公共基准：Time-MMD（可参考原项目仓库）。

## 训练 / 评测
```bash
bash ./run.sh
```

## RAG / CoT 文本引导
- `--use_rag_cot`：启用 TF-IDF 检索 + CoT 生成。
- `--cot_only`：关闭检索，仅用数值摘要 + 原始文本生成 CoT。
- 常用参数：`--rag_topk`, `--cot_model`, `--cot_max_new_tokens`, `--cot_temperature`,
  `--cot_cache_size`, `--cot_device`。
- 默认行为：将 CoT 文本与原始报告拼接后输入文本编码器。
- 开启 `--trend_cfg` 后：CoT 被解析为趋势先验，不再拼接进文本；用于调制扩散路径上的 CFG 权重。
- 可选量化加载：`--cot_load_in_8bit` / `--cot_load_in_4bit`（需要 bitsandbytes）。

## 两阶段 RAG（最小改动增强）
- 开关：`--use_two_stage_rag`（默认关闭以保持单阶段行为）。
- Stage-1：复用原始 query 检索 E0。
- 趋势假设：由 CoT 生成；失败则使用数值模板。
- Stage-2：将趋势假设转为自然语言 query 检索 E1，合并 E1 与 E0（去重、保留 top-k）。
- 稳定性：若原始文本为空或检索失败，自动回退到单阶段。
- 输出模板固定：
  `[RAW TEXT] / [NUMERICAL SUMMARY] / [TREND HYPOTHESIS] / [RETRIEVED EVIDENCE - REFINED]`
- 可调参数：`--rag_stage1_topk`, `--rag_stage2_topk`, `--two_stage_gate`, `--trend_slope_eps`
- 数值统计（`slope/std/mean`）会追加到 `key_factors`，提升 Stage-2 检索效果。

## Trend-aware CFG（CoT -> 趋势先验）
将 CoT 从“文本条件”提升为“扩散路径调制信号”。
- 开关：`--trend_cfg`
- 时间调度：`--trend_cfg_power`, `--trend_time_floor`
- 趋势映射：
  - strength：`1 + trend_strength_scale * (strength - 1)`
  - volatility：`1 / (1 + trend_volatility_scale * volatility)`
- 可调参数：`--trend_strength_scale`, `--trend_volatility_scale`, `--trend_time_floor`,
  `--trend_cfg_random`
- 保存趋势先验：`--save_trend_prior` 输出 `trend_priors.npy` 与 `trend_text_marks.npy`

## 多分辨率辅助损失（最小改动增强）
引入轻量多视距监督（如 {1,3,6,12}），在**不改模型结构**的前提下增强训练信号。
- 配置项（位于 `train`）：
  - `multi_res_horizons`：监督的预测视距列表（自动裁剪到 `pred_len`）
  - `multi_res_loss_weight`：辅助损失权重（设为 0 即关闭）
  - `multi_res_use_huber`：是否使用 Huber（推荐）
  - `multi_res_huber_delta`：Huber 的 delta
  - `multi_res_dynamic`：是否启用 A 版动态多视距加权
  - `multi_res_dynamic_by_t`：按 diffusion step 动态调整 horizon 权重
  - `multi_res_dynamic_by_epoch`：按训练 epoch 做 curriculum
  - `multi_res_dynamic_by_trend`：按 `trend_prior` 的强度/波动度调整 horizon 权重
  - `multi_res_dynamic_min_weight`：动态权重下限，避免某个 horizon 被完全压掉
- 示例（YAML）：
  ```yaml
  train:
    multi_res_horizons: [1, 3, 6, 12]
    multi_res_loss_weight: 0.1
    multi_res_use_huber: true
    multi_res_huber_delta: 1.0
    multi_res_dynamic: true
    multi_res_dynamic_by_t: true
    multi_res_dynamic_by_epoch: true
    multi_res_dynamic_by_trend: true
    multi_res_dynamic_min_weight: 0.2
  ```

## Guide weight 扫描
- `--guide_w -1` 会使用内置列表自动扫描（包含 `4.5`）。
- 如需固定某个值，直接传 `--guide_w`。

## 调试
使用 `debug_two_stage_rag.py` 查看 Q1/E0/z0/Q2/E1 及文本拼接预览。

## 完整命令示例
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
- 全量跑 trend CFG：`scripts/run_all_datasets_trendcfg.sh`
- trend CFG 网格搜索：`scripts/train_trendcfg_grid.sh`

## 致谢
代码基于：
CSDI、Time-LLM、MM-TSF、Autoformer
