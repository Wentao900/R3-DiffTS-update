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

## 多分辨率辅助监督（当前实现）
当前实现分两层：
1) 基础多分辨率监督（等权）  
2) 可选 Conservative 自适应加权（`multi_res_adapt_mode: conservative`）

### 1) 总损失
设主损失为 \(L_{\text{main}}\)，多分辨率辅助损失为 \(L_{\text{aux}}\)，则：

\[
L = L_{\text{main}} + \lambda_{\text{mr}} L_{\text{aux}},
\]

其中 \(\lambda_{\text{mr}}=\texttt{multi\_res\_loss\_weight}\)。

### 2) 单尺度损失
对任意 horizon \(h\in\{1,\dots,\texttt{pred\_len}\}\)，定义：

\[
M_h = M_{\text{target}} \odot \mathbf{1}[t_0 \le t < t_0 + h],\quad
R_h = (\mathbf{x}-\hat{\mathbf{x}})\odot M_h,
\]

其中 \(t_0=\texttt{lookback\_len}\)。

若使用 Huber（\(\delta=\texttt{multi\_res\_huber\_delta}\)）：

\[
\rho_\delta(r)=
\begin{cases}
\frac12 r^2, & |r|\le\delta\\
\delta |r|-\frac12\delta^2, & |r|>\delta
\end{cases}
\]

\[
L_h = \frac{1}{N_h}\sum \rho_\delta(R_h),\quad N_h=\sum M_h.
\]

基础等权辅助损失：

\[
L_{\text{base}}=\frac{1}{m}\sum_{h\in\mathcal{H}}L_h.
\]

### 3) Conservative 自适应（可选）
仅当 `multi_res_adapt_mode: conservative` 开启时生效；否则 \(L_{\text{aux}}=L_{\text{base}}\)。

#### 3.1 EMA 难度统计（训练阶段更新）
\[
\bar{L}_h^{(k)}=\beta \bar{L}_h^{(k-1)} + (1-\beta)L_h^{(k)},
\]
其中 \(\beta=\texttt{multi\_res\_adapt\_beta}\)。  
若步数 \(k < k_{\text{warmup}}\)（`multi_res_adapt_warmup_steps`），直接使用 \(L_{\text{base}}\)。

#### 3.2 动态权重
\[
z_h=
\begin{cases}
\bar{L}_h/T, & \texttt{focus=hard}\\
-\bar{L}_h/T, & \texttt{focus=easy}
\end{cases}
\quad,\quad
w_h^{dyn}=\text{softmax}(z_h),
\]
其中 \(T=\texttt{multi\_res\_adapt\_temp}\)。

定义均匀权重 \(w_h^u=1/m\)，先做强度混合：
\[
\tilde{w}_h=(1-s)w_h^u + s\,w_h^{dyn},
\]
其中 \(s=\texttt{multi\_res\_adapt\_strength}\)。

再做“地板回拉”：
\[
w_h=(1-f)\tilde{w}_h+f w_h^u,
\]
其中 \(f=\texttt{multi\_res\_adapt\_weight\_floor}\)。

得到加权损失：
\[
L_{\text{weighted}}=\sum_{h\in\mathcal{H}} w_h L_h.
\]

最终辅助损失为保守融合：
\[
L_{\text{aux}}=(1-b)L_{\text{base}} + bL_{\text{weighted}},
\]
其中 \(b=\texttt{multi\_res\_adapt\_blend}\)。

### 4) 二阶段冲突抑制（可选）
若开启 `multi_res_conflict_guard: true`，会基于各尺度梯度代理的余弦冲突比例，计算缩放系数 \(c\in[c_{min},1]\)，并对 \(s,b\) 做缩放（冲突越强，越回退到均匀/保守）：

\[
s \leftarrow c s,\quad b \leftarrow c b.
\]

### 5) 常用配置（`train`）
```yaml
train:
  multi_res_horizons: [1, 3, 6, 12]
  multi_res_loss_weight: 0.1
  multi_res_use_huber: true
  multi_res_huber_delta: 1.0

  # conservative adaptive
  multi_res_adapt_mode: conservative   # off | conservative
  multi_res_adapt_focus: hard          # hard | easy
  multi_res_adapt_beta: 0.95
  multi_res_adapt_temp: 1.0
  multi_res_adapt_strength: 0.5
  multi_res_adapt_weight_floor: 0.1
  multi_res_adapt_warmup_steps: 300
  multi_res_adapt_blend: 0.4

  # optional conflict guard
  multi_res_conflict_guard: false
  multi_res_conflict_tau: 0.05
  multi_res_conflict_power: 0.8
  multi_res_conflict_min_scale: 0.4
```

## Guide weight 扫描
- `--guide_w -1` 会使用内置列表自动扫描（包含 `4.5`）。
- 如需固定某个值，直接传 `--guide_w`。

## 当前 Economy V2 基线
- 当前使用配置：`config/economy_36_12_scale_router_guide.yaml`
- 任务设置：`seq_len=36`、`pred_len=12`、`text_len=36`、`freq=m`
- 当前报告只记录验证集上的 `guide_w` 选参结果，不写测试集指标。
- 验证集选参文件：`save/forecasting_Economy_20260312_194619/selected_guide_w.json`
- 验证集最终选中：`guide_w = 1.4`
- 最优验证集 `MSE = 0.14063129197983515`

验证集 guide 扫描结果：
- `0.4 -> 0.1933452288309733`
- `0.5 -> 0.18269025257655552`
- `0.6 -> 0.17350386664980932`
- `0.7 -> 0.16563423247564407`
- `0.8 -> 0.15907128197806222`
- `0.9 -> 0.15369917097545804`
- `1.0 -> 0.1493416014171782`
- `1.2 -> 0.14333132335117885`
- `1.4 -> 0.14063129197983515`

推荐的 Economy 命令：
```bash
python -u exe_forecasting.py \
  --root_path ../Time-MMD-main \
  --data_path Economy/Economy.csv \
  --config economy_36_12_scale_router_guide.yaml \
  --seq_len 36 \
  --pred_len 12 \
  --text_len 36 \
  --freq m
```

说明：
- `root_path` 应使用 `../Time-MMD-main`
- `data_path` 应使用 `Economy/Economy.csv`
- 不要传 `Time-MMD-main/numerical/Economy/Economy.csv`
- 详细方法说明见 `REPORT_V2_ECONOMY.md`

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
