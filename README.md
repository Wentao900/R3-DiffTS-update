# R3-DiffTS

**Retrieval-Reasoning-Routing Enhanced Diffusion for Time Series Forecasting**

R3-DiffTS 是一个基于扩散模型的多模态时间序列预测框架。它将文本信息（新闻、报告、事件描述等）与数值时间序列深度融合，通过检索增强生成（RAG）、思维链推理（CoT）和趋势感知扩散路径调制等机制，实现更准确、更可解释的预测。

> R3-DiffTS is a multimodal diffusion-based time series forecasting framework. It deeply integrates textual information (news, reports, event descriptions) with numerical time series via Retrieval-Augmented Generation (RAG), Chain-of-Thought (CoT) reasoning, and trend-aware diffusion path modulation for more accurate and interpretable forecasting.

<p align="center">
  <img src="docs/paper_style_scale_aware_architecture.svg" width="80%" alt="R3-DiffTS Architecture"/>
</p>

---

## ✨ Key Features

| Module | Description |
|--------|-------------|
| **Multimodal Conditioned Diffusion** | CSDI-based diffusion backbone with classifier-free guidance (CFG), fusing time series and text modalities via cross-attention |
| **RAG + CoT Guidance** | TF-IDF retrieval of relevant text evidence + Chain-of-Thought generation for structured trend reasoning |
| **Two-Stage RAG** | Stage-1 retrieves initial evidence; Stage-2 refines retrieval using CoT-generated trend hypotheses |
| **Trend-aware CFG** | CoT is parsed into a structured trend prior (direction/strength/volatility) that dynamically modulates CFG weights along the diffusion path |
| **Multi-resolution Loss** | Auxiliary multi-horizon supervision within the prediction window for better scale awareness |
| **Scale Router** | Lightweight learned router that adaptively weights multi-resolution loss bands based on time series characteristics |
| **Timestamp-Assisted Attention (TAA)** | Joint self-attention between time series and timestamp features with learnable weighting |
| **Text-to-Time Fusion (TTF)** | Cross-attention from text embeddings to time series representations |

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

**Dependencies**: PyTorch ≥ 2.5, Transformers ≥ 4.51, scikit-learn, pandas, numpy, linear-attention-transformer

---

## 📊 Dataset

We use the [Time-MMD](https://github.com/adityalab/time-mmd) benchmark, which contains 8 multimodal time series datasets:

| Domain | Dataset | Frequency | Recommended `seq_len` |
|--------|---------|-----------|----------------------|
| Economy | Economy | Monthly | 36 |
| Traffic | Traffic | Monthly | 36 |
| Agriculture | Agriculture | Monthly | 36 |
| Social Good | SocialGood | Monthly | 36 |
| Energy | Energy | Weekly | 96 |
| Health | Health_US | Weekly | 96 |
| Climate | Climate | Weekly | 96 |
| Environment | Environment | Daily | 336 |

Download [Time-MMD](https://github.com/adityalab/time-mmd) and place it at `../Time-MMD-main` relative to this repository.

---

## 🚀 Quick Start

### Train & Evaluate (all datasets, baseline)

```bash
bash run.sh
```

### Train & Evaluate (all datasets, full pipeline)

```bash
bash run_full.sh
```

> This uses `*_full.yaml` configs which enable all modules: RAG+CoT, Two-Stage RAG, Trend-aware CFG, Multi-resolution Loss, and Scale Router.

### Single dataset example

```bash
python -u exe_forecasting.py \
  --root_path ../Time-MMD-main \
  --data_path Economy/Economy.csv \
  --config economy_36_12.yaml \
  --seq_len 36 --pred_len 12 --text_len 36 --freq m
```

### With RAG + CoT guidance

```bash
python -u exe_forecasting.py \
  --root_path ../Time-MMD-main \
  --data_path Economy/Economy.csv \
  --config economy_36_12.yaml \
  --seq_len 36 --pred_len 12 --text_len 36 --freq m \
  --use_rag_cot
```

### With Trend-aware CFG

```bash
python -u exe_forecasting.py \
  --root_path ../Time-MMD-main \
  --data_path Economy/Economy.csv \
  --config economy_36_12.yaml \
  --seq_len 36 --pred_len 12 --text_len 36 --freq m \
  --use_rag_cot --trend_cfg \
  --trend_cfg_power 1.0 --trend_strength_scale 0.35 \
  --trend_volatility_scale 1.0 --trend_time_floor 0.30
```

---

## ⚙️ Key Configuration

### YAML Config (`config/*.yaml`)

```yaml
train:
  epochs: 150
  batch_size: 32
  lr: 0.0025

diffusion:
  layers: 6
  channels: 64
  nheads: 8
  num_steps: 300
  cfg: True            # Enable classifier-free guidance
  c_mask_prob: 0.05    # Unconditional dropout probability

model:
  llm: "bert"          # Text encoder: bert / gpt2 / llama
  with_texts: True     # Enable text modality
  timestep_emb_cat: True
  timestep_branch: True
```

### Command-line Arguments (selected)

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_rag_cot` | `False` | Enable RAG + CoT text guidance |
| `--cot_only` | `False` | Use CoT without retrieval |
| `--use_two_stage_rag` | `False` | Enable two-stage retrieval |
| `--trend_cfg` | `False` | Enable trend-aware CFG modulation |
| `--trend_cfg_power` | `1.0` | Power for time schedule |
| `--trend_strength_scale` | `1.0` | Scale for trend strength |
| `--trend_volatility_scale` | `1.0` | Scale for volatility penalty |
| `--trend_time_floor` | `0.0` | Floor for time schedule |
| `--guide_w` | `-1` | CFG weight (`-1` for auto sweep) |

---

## 📁 Project Structure

```
R3-DiffTS/
├── main_model.py          # Core model: CSDI_base, ScaleRouter, multi-res loss
├── diff_models.py         # Diffusion backbone: ResidualBlock, diff_CSDI
├── exe_forecasting.py     # Training & evaluation entry point
├── dataset_forecasting.py # Dataset loading dispatcher
├── run.sh                 # Run all datasets
├── requirements.txt       # Python dependencies
├── config/                # YAML configs for each dataset × horizon
│   ├── economy_36_12.yaml
│   ├── traffic_36_12.yaml
│   └── ...
├── data_provider/         # Data loading & preprocessing
│   ├── data_factory.py
│   └── data_loader.py
├── utils/
│   ├── utils.py           # Train/evaluate loops
│   ├── rag_cot.py         # RAG retrieval + CoT generation
│   ├── trend_prior.py     # Trend prior extraction from CoT
│   ├── prepare4llm.py     # LLM loading utilities
│   ├── SelfAttention_Family.py
│   ├── timefeatures.py
│   └── masking.py
├── scripts/               # Training scripts for different modes
│   ├── train_baseline.sh
│   ├── train_rag_cot.sh
│   ├── train_rag_only.sh
│   ├── train_cot_only.sh
│   └── run_all_datasets_trendcfg.sh
├── docs/                  # Architecture diagrams (SVG)
└── LICENSE
```

---

## 🙏 Acknowledgements

This codebase builds upon:
- [CSDI](https://github.com/ermongroup/CSDI) — Conditional Score-based Diffusion Model for Imputation
- [Time-LLM](https://github.com/KimMeen/Time-LLM) — Large Language Models for Time Series
- [MM-TSF / Time-MMD](https://github.com/adityalab/time-mmd) — Multimodal Time Series Forecasting Benchmark
- [Autoformer](https://github.com/thuml/Autoformer) — Decomposition Transformers

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
