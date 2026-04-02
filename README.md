# R3-DiffTS

**R**etrieval · **R**easoning · **R**outing Enhanced **Diff**usion for **T**ime **S**eries Forecasting

---

## Overview

在多模态时间序列预测任务中，文本信息（新闻、宏观报告、事件描述）与数值序列的有效融合是核心挑战。现有方法通常将文本作为静态条件输入，忽略了两个关键问题：**文本质量参差不齐**、**文本影响应随去噪阶段动态变化**。

R3-DiffTS 提出三阶段渐进增强策略：

```
        ┌─────────────────────────────────────────────────────┐
        │              R3-DiffTS Pipeline                      │
        │                                                      │
        │   Historical Series + Raw Text                       │
        │         │                                            │
        │         ▼                                            │
        │   ┌──────────┐    ┌──────────┐    ┌──────────┐      │
        │   │Retrieval │───▶│Reasoning │───▶│ Routing  │      │
        │   │(R)       │    │(R)       │    │(R)       │      │
        │   │          │    │          │    │          │      │
        │   │ TF-IDF   │    │ CoT →    │    │ Scale    │      │
        │   │ Two-Stage│    │ Trend    │    │ Router + │      │
        │   │ RAG      │    │ Prior →  │    │ Multi-Res│      │
        │   │          │    │ Dynamic  │    │ Loss     │      │
        │   │          │    │ CFG      │    │          │      │
        │   └──────────┘    └──────────┘    └──────────┘      │
        │         │                │               │           │
        │         └────────────────┴───────────────┘           │
        │                         │                            │
        │                         ▼                            │
        │              CSDI Diffusion Backbone                 │
        │         (TAA + TTF + CFG Denoising)                  │
        │                         │                            │
        │                         ▼                            │
        │                   Prediction ŷ                       │
        └─────────────────────────────────────────────────────┘
```

| 阶段 | 模块 | 核心思想 |
|------|------|---------|
| **Retrieval** | Two-Stage RAG | 两阶段检索：先获取初始证据，再用 CoT 生成的趋势假设做二次精准检索 |
| **Reasoning** | CoT → Trend-aware CFG | 将 CoT 推理结果从"静态文本条件"提升为"扩散去噪路径的动态控制信号" |
| **Routing** | Multi-Res Loss + Scale Router | 轻量路由网络根据序列特征自适应分配多分辨率损失权重 |

---

## Method Details

### 1. Retrieval — Two-Stage RAG

```
Stage-1: TF-IDF(query=数值摘要) → 初始证据 E₀
                ↓
         CoT → 趋势假设 z_trend
                ↓
Stage-2: TF-IDF(query=趋势假设) → 补充证据 E₁
                ↓
         Merge(E₀, E₁) → 精选 top-k 证据
```

- 安全门控：文本为空或检索失败时自动回退到 one-shot 模式

### 2. Reasoning — Trend-aware Diffusion Control

CoT 输出被解析为结构化趋势先验：

```
z_trend = [direction ∈ {-1, 0, +1},  strength ∈ ℝ⁺,  volatility ∈ ℝ⁺]
```

该先验动态调制 Classifier-Free Guidance 权重：

```
ŷ_k = y_uncond + w_trend(k, z_trend) · (y_cond − y_uncond)

其中：
  w_trend(k) = α · g(k) · h(z_trend)
  g(k) = (1 − k/K)^p + floor        ← 时间调制：后期去噪时文本影响更强
  h(z) = (1 + s·strength) / (1 + v·volatility)  ← 趋势调制：强趋势→强引导，高波动→抑制引导
```

### 3. Routing — Scale-aware Multi-resolution Supervision

将预测窗口划分为多个 band（如 `[1-3, 4-6, 7-12]`），Scale Router 根据序列特征自适应分配各 band 权重：

```
Router 输入: [slope, volatility, diff_std, accel, mean_abs, log_scale, trend_prior, text_mask]
       ↓
    MLP → Softmax → band weights w₁, w₂, ..., wₙ
       ↓
    L_multi_res = Σᵢ wᵢ · Huber(pred_bandᵢ − gt_bandᵢ)
```

---

## Installation

```bash
pip install -r requirements.txt
```

**核心依赖**: PyTorch ≥ 2.5, Transformers ≥ 4.51, scikit-learn, pandas, linear-attention-transformer

---

## Dataset

基准测试集：[Time-MMD](https://github.com/adityalab/time-mmd)（8 个多模态时间序列数据集）

| 数据集 | 频率 | `seq_len` | `pred_len` | `freq` |
|--------|------|-----------|------------|--------|
| Economy | 月 | 36 | 6 / 12 / 18 | m |
| Traffic | 月 | 36 | 6 / 12 / 18 | m |
| Agriculture | 月 | 36 | 6 / 12 / 18 | m |
| SocialGood | 月 | 36 | 6 / 12 / 18 | m |
| Energy | 周 | 96 | 12 / 24 / 48 | w |
| Health_US | 周 | 96 | 12 / 24 / 48 | w |
| Climate | 周 | 96 | 12 / 24 / 48 | w |
| Environment | 日 | 336 | 48 / 96 / 192 | d |

```bash
# 将数据集放置在仓库同级目录
git clone https://github.com/adityalab/time-mmd ../Time-MMD-main
```

---

## Quick Start

### Main Config（全部 8 个数据集）

```bash
bash run.sh
```

### Full Pipeline（全部模块开启）

```bash
bash run_full.sh
```

### 单个数据集

```bash
# Main config
python -u exe_forecasting.py \
  --root_path ../Time-MMD-main \
  --data_path Economy/Economy.csv \
  --config economy_36_12.yaml \
  --seq_len 36 --pred_len 12 --text_len 36 --freq m

# Full pipeline
python -u exe_forecasting.py \
  --root_path ../Time-MMD-main \
  --data_path Economy/Economy.csv \
  --config economy_36_12_full.yaml \
  --seq_len 36 --pred_len 12 --text_len 36 --freq m
```

---

## Ablation Experiments

提供 6 组消融实验 + 1 组对照，验证 3 大创新点的独立贡献：

| ID | 配置 | R (文本增强) | D (趋势调制) | S (多尺度) |
|----|------|:---:|:---:|:---:|
| E0 | Baseline | ❌ | ❌ | ❌ |
| E1 | +R | ✅ | ❌ | ❌ |
| E2 | +R+D | ✅ | ✅ | ❌ |
| E3 | Full | ✅ | ✅ | ✅ |
| E4 | w/o D | ✅ | ❌ | ✅ |
| E5 | w/o R | ❌ | ✅ | ✅ |
| E2' | +R+D_random | ✅ | 🎲 | ❌ |

```bash
# 跑全部消融
bash run_ablation.sh

# 跑指定实验
bash run_ablation.sh e3              # 全部数据集的 Full
bash run_ablation.sh e2 economy      # Economy 上的 +R+D
bash run_ablation.sh all traffic     # Traffic 全部消融
```

---

## Configuration

### Config 文件分类

```
config/
├── {dataset}_{seq}_{pred}.yaml          # 每个任务的主实验默认配置
├── {dataset}_{seq}_{pred}_full.yaml     # 扩展 / 对照用全模块配置
└── {dataset}_{seq}_{pred}_abl_{id}.yaml # 消融配置
```

### 关键参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_rag_cot` | False | 启用 RAG + CoT 文本引导 |
| `--use_two_stage_rag` | False | 启用两阶段检索 |
| `--trend_cfg` | False | 启用趋势感知 CFG 调制 |
| `--trend_strength_scale` | 1.0 | 趋势强度缩放因子 |
| `--trend_time_floor` | 0.0 | 时间调制下限 |
| `--guide_w` | -1 | CFG 权重（-1 为验证集自动选择） |

### YAML 中的多尺度配置

```yaml
train:
  multi_res_horizons: [1, 3, 6, 12]       # 辅助监督窗口
  multi_res_band_boundaries: [1, 3, 6, 12]
  multi_res_loss_weight: 0.03             # 调轻后的辅助损失权重
  multi_res_mode: dynamic_band            # 动态 band 加权
  multi_res_weight_floor: 0.3             # 权重下限
  use_scale_router: true                  # 启用 Scale Router
  scale_router_use_trend_prior: false     # 关闭 trend prior 路由偏置
```

---

## Project Structure

```
R3-DiffTS/
├── main_model.py            # 核心模型 (CSDI_base, ScaleRouter, 多尺度损失)
├── diff_models.py           # 扩散骨架 (ResidualBlock, diff_CSDI)
├── exe_forecasting.py       # 训练 & 评估入口
├── dataset_forecasting.py   # 数据集调度
├── run.sh                   # Baseline 全数据集运行
├── run_full.sh              # Full pipeline 全数据集运行
├── run_ablation.sh          # 消融实验运行
├── EXPERIMENT_ROADMAP.md    # 科研实验路线
├── requirements.txt
├── LICENSE
├── config/                  # YAML 配置 (base / full / ablation)
├── data_provider/           # 数据加载与预处理
│   ├── data_factory.py
│   └── data_loader.py
├── utils/
│   ├── utils.py             # 训练/评估循环
│   ├── rag_cot.py           # RAG 检索 + CoT 生成
│   ├── trend_prior.py       # 趋势先验抽取
│   ├── prepare4llm.py       # LLM 加载工具
│   ├── SelfAttention_Family.py
│   ├── timefeatures.py
│   └── masking.py
├── scripts/                 # 模式训练脚本
│   ├── train_baseline.sh
│   ├── train_rag_cot.sh
│   ├── train_rag_only.sh
│   ├── train_cot_only.sh
│   └── run_all_datasets_trendcfg.sh
└── docs/                    # 架构图 (SVG)
```

---

## Acknowledgements

- [CSDI](https://github.com/ermongroup/CSDI) — Conditional Score-based Diffusion
- [Time-LLM](https://github.com/KimMeen/Time-LLM) — LLM for Time Series
- [Time-MMD](https://github.com/adityalab/time-mmd) — Multimodal TSF Benchmark
- [Autoformer](https://github.com/thuml/Autoformer) — Decomposition Transformers

## License

MIT License — see [LICENSE](LICENSE) for details.
