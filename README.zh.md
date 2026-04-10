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
  - `multi_res_partition_mode`：`cumulative`（原前缀累计模式）或 `disjoint`（互不重叠区间，如 `[1] [2-3] [4-6] [7-12]`）
  - `multi_res_use_scale_router`：是否使用样本级尺度路由权重来加权 multi-res 各区间
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
    multi_res_partition_mode: disjoint
    multi_res_use_scale_router: true
  ```

## 频域辅助损失
可选地在预测段上增加一个轻量 FFT 幅度损失。
- 配置项（位于 `train`）：
  - `freq_loss_weight`：FFT 辅助损失权重
  - `freq_loss_low_bins`：仅保留最低频的非 DC bin；`<=0` 表示使用全部 bin
  - `freq_loss_exclude_dc`：比较频谱时是否去掉 DC 分量
  - `freq_loss_normalize`：比较前是否对幅度谱做归一化
- 推荐用法：
  - 保持较小权重，和现有 pointwise loss 配合使用
  - 对月频或周频数据优先只看低频 bin

## 启发式尺度路由（吸收本周周报）
从数值历史中估计样本级尺度偏好 `r_i`，并用它统一驱动文本窗口和可选的 multi-res 区间权重。
- 开关：`--use_scale_router`
- 路由区间：`train.scale_route_horizons`
  - 若未显式设置，优先复用 `train.multi_res_horizons`，再自动生成
- 文本窗口候选：`model.scale_window_candidates`
  - 若未设置，则在 `text_len` 范围内自动均匀生成
- 软分配温度：`model.scale_route_temperature`
- 当前接入位置：
  - dataset 侧动态文本窗口
  - 可选的 disjoint multi-res loss 加权
- 当前实现是启发式版本，不引入新的可训练 router，风险较低，便于先做消融。

## 尺度引导 CFG
可选地把 `scale_route` 转成推理期的样本级 guidance 倍数。
- 开关：`--scale_guidance`
- 各尺度倍率：`--scale_guidance_alpha`
  - 例如：`0.9,1.0,1.1,1.2`
- 行为：
  - 关闭时，仍使用原来的全局 `guide_w`
  - 开启时，每个样本使用 `guide_w_i = guide_w * <scale_route, alpha>`
  - 若同时开启 `trend_cfg`，该尺度倍数会继续乘到 trend-aware guidance 上

## Step-aware CFG
可选地在反向扩散采样过程中，对 CFG 强度做按步调度。
- 开关：`--step_guidance`
- 调度参数：
  - `--step_guidance_power`
  - `--step_guidance_floor`
- 行为：
  - 关闭时，仍使用原来的全局 `guide_w`
  - 开启后，每个反向步使用 `guide_w_t = guide_w_i * s(t)`
  - `s(t)` 会从设定的 floor 单调增加到 `1.0`，因此后期去噪步骤得到更强约束
  - 若同时开启 `trend_cfg`，该按步系数会继续乘到 trend-aware guidance 上

## 证据一致性引导
可选地对检索到的证据做一致性打分，并在推理时压低不可靠证据对应的 guidance。
- 检索侧开关：
  - `--rag_consistency`
  - `--consistency_unknown_penalty`
  - `--consistency_conflict_penalty`
- 扩散侧开关：
  - `--consistency_guidance`
  - `--consistency_threshold`
- 行为：
  - 每条 evidence 会被启发式映射到 `up/down/flat/unknown`
  - 再根据主导立场、unknown 比例、立场冲突计算 `[0,1]` 的 `consistency_score`
  - 开启后，推理期使用 `guide_w_i = guide_w_i * consistency_score`
  - 阈值较高时，可对低一致性样本直接把 guidance 置零

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
  --step_guidance --step_guidance_power 1.0 --step_guidance_floor 0.35 \
  --guide_w -1
```

## Scripts
- 全量跑 trend CFG：`scripts/run_all_datasets_trendcfg.sh`
- 全量跑 step-aware CFG：`scripts/run_all_datasets_stepcfg.sh`
- trend CFG 网格搜索：`scripts/train_trendcfg_grid.sh`
- Economy 消融：
  - 入口脚本：`scripts/run_economy_scale_router_ablations.sh`
  - 默认 case：`no_multires`, `cum_base`, `disjoint_only`, `router_window_only`, `router_loss_only`, `router_full`, `router_guidance`, `router_guidance_freq`, `router_consistency`

## 致谢
代码基于：
CSDI、Time-LLM、MM-TSF、Autoformer
