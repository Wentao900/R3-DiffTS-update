# R3-DiffTS 科研实验路线

## 研究背景

多模态时间序列预测面临三大核心挑战：
1. **文本信息噪声大**：新闻报告等文本信号稀疏、低频、质量参差，直接融入模型可能引入噪声
2. **多尺度预测困难**：不同预测horizon上的误差模式差异显著，单一损失函数难以兼顾
3. **文本条件静态注入**：现有方法将文本嵌入作为静态条件，无法根据扩散去噪阶段动态调整

R3-DiffTS 通过 **Retrieval → Reasoning → Routing** 三阶段渐进式增强，系统性地解决上述问题。

---

## 实验阶段设计

### Phase 1：Baseline 复现（纯数值 + 文本条件）

**目标**：建立基准性能，验证扩散框架的基本能力

**内容**：
- 在 8 个 Time-MMD 数据集上训练 MCD-TSF baseline
- 使用 BERT 编码原始文本，通过 cross-attention 融入扩散模型
- 启用 Classifier-Free Guidance (CFG)，通过验证集选择最优 `guide_w`

**运行命令**：
```bash
bash scripts/train_baseline.sh
```

**关键指标**：MSE / MAE on test split

**预期结果**：
- 文本条件应比无条件 baseline 有提升
- 但提升幅度可能有限（文本噪声问题）

---

### Phase 2：RAG + CoT 文本引导

**目标**：通过检索和推理提升文本信息质量

**内容**：
- **RAG**：基于 TF-IDF 从历史文本库检索与当前窗口最相关的 top-k 证据
- **CoT**：通过模板或 LLM 生成结构化推理文本（数值摘要 + 趋势判断 + 原因分析）
- 组合文本格式：`[原始文本] / [数值摘要] / [检索证据] / [CoT推理]`

**实验变体**：
| 实验 | RAG | CoT | 命令脚本 |
|------|-----|-----|----------|
| RAG only | ✅ | ❌ | `scripts/train_rag_only.sh` |
| CoT only | ❌ | ✅ | `scripts/train_cot_only.sh` |
| RAG + CoT | ✅ | ✅ | `scripts/train_rag_cot.sh` |

**关键参数**：
- `--rag_topk 3`：检索证据数量
- `--cot_model`：CoT 生成模型（`None` 使用模板）
- `--cot_temperature 0.7`：生成温度

**预期结果**：
- RAG + CoT 组合应显著优于单独使用
- CoT 提供的结构化趋势信息对预测特别有帮助

---

### Phase 3：Two-Stage RAG（两阶段检索）

**目标**：通过趋势假设引导的二次检索，提升证据相关性

**内容**：
- **Stage-1**：标准 one-shot 检索，获得初始证据 E0
- **趋势假设生成**：由 CoT 生成趋势假设（direction/strength/volatility）
- **Stage-2**：将趋势假设转为自然语言查询，检索补充证据 E1
- 合并 E0 + E1（去重，保留 top-k）
- 安全门控：文本为空或检索失败时回退到 one-shot

**关键参数**：
- `--use_two_stage_rag`：启用两阶段检索
- `--rag_stage1_topk / --rag_stage2_topk`：各阶段检索数量
- `--two_stage_gate`：安全门控
- `--trend_slope_eps`：斜率阈值

**消融对比**：
- One-shot RAG vs Two-stage RAG
- 有/无安全门控

**预期结果**：
- Two-stage RAG 在趋势明确的数据集上（Economy、Energy）应有更大提升
- 安全门控对波动大的数据集（Traffic）很重要

---

### Phase 4：Trend-aware CFG（趋势感知扩散路径调制）

**目标**：将 CoT 从静态文本条件提升为动态扩散路径控制信号

**核心创新**：
- CoT 被解析为显式趋势先验 `z_trend = [direction, strength, volatility]`
- 趋势先验动态调制 CFG 权重：
  - **时间调制**：前期去噪（大噪声）文本影响弱，后期去噪（小噪声）文本影响强
  - **趋势调制**：强趋势 → 更强引导；高波动 → 抑制引导

**运行命令**：
```bash
bash scripts/run_all_datasets_trendcfg.sh
```

**关键参数**：
- `--trend_cfg`：启用趋势感知 CFG
- `--trend_cfg_power 1.0`：时间调制的幂次
- `--trend_strength_scale 0.35`：趋势强度缩放
- `--trend_volatility_scale 1.0`：波动性缩放
- `--trend_time_floor 0.30`：时间调制下限

**消融实验**：
| 对比条件 | 说明 |
|----------|------|
| 固定 CFG vs Trend-aware CFG | 核心贡献验证 |
| CoT 有/无趋势调制 | 证明 CoT 升级为控制信号的价值 |
| 随机趋势 vs CoT 趋势 | 排除随机调制的可能性 |
| 不同 `trend_cfg_power` | 时间调制曲线敏感性 |
| 不同 `trend_time_floor` | 下限选择的影响 |

**预期结果**：
- Trend-aware CFG 应一致优于固定 CFG
- 随机趋势应显著劣于 CoT 趋势（排除偶然性）
- 对趋势明确的数据集提升更大

---

### Phase 5：Multi-resolution Loss + Scale Router

**目标**：引入多尺度监督信号，让模型在不同预测窗口内自适应关注不同频率特征

**内容**：

#### 5a. Multi-resolution Auxiliary Loss
- 在预测窗口内划分多个 band（如 `[1-3, 4-6, 7-12]`）
- 对每个 band 分别计算损失
- 使用 Huber loss 提升稳定性

#### 5b. Scale Router
- 轻量 MLP 网络，根据时间序列统计特征（斜率、波动率、加速度等）+ 趋势先验 + 文本可用性
- 自适应地为不同 band 分配损失权重
- 支持 teacher 引导的 warmup 策略

**配置示例**（YAML）：
```yaml
train:
  multi_res_band_boundaries: [3, 6, 12]
  multi_res_loss_weight: 0.1
  multi_res_use_huber: true
  multi_res_huber_delta: 1.0
  use_scale_router: true
  scale_router_hidden_dim: 32
  scale_router_entropy_weight: 0.001
```

**消融实验**：
| 对比条件 | 说明 |
|----------|------|
| 无 multi-res vs 有 multi-res | 多尺度监督的价值 |
| 固定权重 vs Scale Router | 自适应路由的价值 |
| 有/无 teacher warmup | warmup 策略的必要性 |
| 不同 band 划分 | band 边界敏感性 |

---

### Phase 6：全模块集成与消融总表

**实验矩阵**：

| # | Baseline | RAG+CoT | Two-Stage | Trend CFG | Multi-Res | Scale Router | MSE ↓ | MAE ↓ |
|---|----------|---------|-----------|-----------|-----------|--------------|-------|-------|
| 1 | ✅ | | | | | | — | — |
| 2 | ✅ | ✅ | | | | | — | — |
| 3 | ✅ | ✅ | ✅ | | | | — | — |
| 4 | ✅ | ✅ | ✅ | ✅ | | | — | — |
| 5 | ✅ | ✅ | ✅ | ✅ | ✅ | | — | — |
| 6 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | — |

在每个 Phase 完成后填入 8 个数据集的平均 MSE/MAE。

---

## 数据集与评估

### 评估指标
- **MSE**（Mean Squared Error）：主指标
- **MAE**（Mean Absolute Error）：辅助指标

### 数据划分
使用 Time-MMD 默认的 train/valid/test 划分。验证集用于：
- 早停（early stopping）
- CFG guide weight 自动选择

### 跨数据集对比
- **月频数据集**（4个）：Economy / Traffic / Agriculture / SocialGood
- **周频数据集**（3个）：Energy / Health / Climate
- **日频数据集**（1个）：Environment

每个数据集提供 3 个预测长度设置。

---

## 实验执行建议

### 硬件需求
- GPU：≥ 1× NVIDIA A100 40GB（推荐）
- 使用 BF16/FP16 可在 24GB GPU 上运行
- CoT 生成可在 CPU 或独立 GPU 上运行

### 实验顺序建议
1. 先在 **Economy** 数据集上完整走通所有 Phase（月频，样本量适中）
2. 确认 pipeline 稳定后，扩展到全部 8 个数据集
3. 每个 Phase 完成后保存 checkpoint 和 config，方便回溯

### 超参搜索优先级
1. `guide_w`（验证集自动选择）
2. `trend_strength_scale` 和 `trend_time_floor`（Trend CFG 核心参数）
3. `multi_res_loss_weight`（Multi-res 损失权重）
4. `rag_topk` 和 `cot_temperature`（文本质量相关）

---

## 论文写作对应

| 实验阶段 | 论文章节 | 论文贡献点 |
|----------|----------|-----------|
| Phase 1-2 | Method §3.1-3.2 | 多模态条件扩散 + RAG/CoT 增强 |
| Phase 3 | Method §3.3 | 两阶段趋势引导检索 |
| Phase 4 | Method §3.4 | **核心创新**：趋势感知扩散路径调制 |
| Phase 5 | Method §3.5 | 多分辨率自适应损失 |
| Phase 6 | Experiment §4 | 完整消融 + SOTA 对比 |
