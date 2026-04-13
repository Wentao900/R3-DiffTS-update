# Codex 精准改造指令（贴合 MCD-TSF 现状）：实现 Two-stage RAG（最小改动、低风险）

> 目标：在 **不改动扩散模型 / CFG / main_model.py** 的前提下，让现有 `utils/rag_cot.py` 的 `RAGCoTPipeline.build_guidance_text(...)` 支持 **Two-stage RAG**。  
> 原则：**默认行为完全不变**（不开启新 flag 时结果应与当前一致），所有新逻辑必须可回退、不崩溃、不引入重依赖。

---

## 0) 你必须先读的现有代码位置（按现状对齐）

请 Codex 先打开并理解：

1. `utils/rag_cot.py`
   - 已存在 `class RAGCoTPipeline`
   - 已存在 TF-IDF 检索（`TfidfVectorizer` / cosine similarity）
   - 已存在数值摘要（例如 `_summarize_num(...)`）
   - 已存在生成 CoT（例如 `_generate_cot(...)`，可能用 `cot_model` 或 fallback 模板）
   - **主入口**：`build_guidance_text(...)`（Dataset 在这里拿到最终 `texts`）

2. `data_provider/data_loader.py`
   - `Dataset_Custom.__init__` 中创建 `self.rag_cot = RAGCoTPipeline(...)`
   - `Dataset_Custom.__getitem__` 中调用 `self.rag_cot.build_guidance_text(...)` 得到 `seq_x_txt` 并写入返回 dict（字段 `texts` / `text_mark`）

3. `exe_forecasting.py`
   - argparse 已有 `--use_rag_cot`、`--rag_topk`、`--cot_only`、`--cot_model`、`--cot_max_new_tokens` 等参数
   - Dataset 初始化时通过 args 传入 dataloader

> 不需要改 `main_model.py`：它只消费 dataloader 的 `texts` 字段并编码。

---

## 1) 最小改动策略（核心要求）

### ✅ 要做
- 在现有 `RAGCoTPipeline` 内部加一个分支：one-shot（原逻辑） vs two-stage（新逻辑）
- 通过新参数 `--use_two_stage_rag` 控制启用（默认 False）
- two-stage 只影响 **检索与最终文本拼接**，不动扩散、不动 CFG

### ❌ 不要做
- 不要重写检索器，不要引入向量数据库
- 不要改动 `build_guidance_text(...)` 的对外函数签名（Dataset 已经调用了它，改签名风险大）
- 不要让训练速度显著下降：two-stage 必须默认关闭

---

## 2) 需要新增的 CLI 参数（只在 exe_forecasting.py 改）

在 `exe_forecasting.py` argparse 中新增（保持默认值保证不影响旧行为）：

- `--use_two_stage_rag`（action='store_true'，默认 False）
- `--rag_stage1_topk`（int，默认 `-1` 表示自动）
- `--rag_stage2_topk`（int，默认 `-1` 表示自动，通常等于现有 `--rag_topk`）
- `--two_stage_gate`（action='store_true'，默认 True 或 False 均可；推荐默认 True）
- `--trend_slope_eps`（float，默认 1e-3 或根据数据 scale 微调）

> 说明：如果你不想加太多参数，至少要有 `--use_two_stage_rag`，其余用内部默认即可。

---

## 3) Dataset 侧最小改动（data_provider/data_loader.py）

在 `Dataset_Custom.__init__` 创建 `RAGCoTPipeline(...)` 时，把新增 args 原样透传进去。

要求：
- 不改变 Dataset 输出字段结构
- `__getitem__` 仍旧只调用 `build_guidance_text(...)` 得到一个字符串 `seq_x_txt`

如果现在 `RAGCoTPipeline(...)` 构造参数是 `RAGCoTPipeline(args, ...)` 或显式列出参数，按现有风格追加即可：

- `use_two_stage_rag=args.use_two_stage_rag`
- `rag_stage1_topk=args.rag_stage1_topk`
- `rag_stage2_topk=args.rag_stage2_topk`
- `two_stage_gate=args.two_stage_gate`
- `trend_slope_eps=args.trend_slope_eps`

---

## 4) rag_cot.py：按现状“精准加功能”，尽量少动原逻辑

### 4.1 在 `RAGCoTPipeline.__init__` 增加字段（默认不影响旧行为）

新增成员变量（建议）：

- `self.use_two_stage_rag: bool`
- `self.rag_stage1_topk: int`
- `self.rag_stage2_topk: int`
- `self.two_stage_gate: bool`
- `self.trend_slope_eps: float`

并设置**安全默认**：

- 若 `rag_stage2_topk <= 0`：`stage2_topk = self.rag_topk`（复用现有 rag_topk）
- 若 `rag_stage1_topk <= 0`：`stage1_topk = min(20, max(self.rag_topk * 3, self.rag_topk))`

> 注意：`self.rag_topk` 当前已存在（来自 args.rag_topk），不要破坏它。

---

### 4.2 保持 `build_guidance_text(...)` 的签名不变，只加内部分支

伪结构（Codex 按现有变量名对齐）：

```python
def build_guidance_text(...):
    if not self.use_two_stage_rag:
        # 走你现在的原逻辑（one-shot），保持输出完全一致
        return original_text

    # 否则走 two-stage
    return two_stage_text
```

## 5) Two-stage 逻辑：直接复用你现有的内部函数（关键：少自由发挥）

### 5.1 Stage-1：用“原来的检索 query”做一次检索得到 E0

你现在 one-shot 的 query 基本是：

- domain prompt（如果有）
- 数值摘要 `S_num`（由 `_summarize_num(...)` 产出）
- 原始文本片段（来自 dataset 的 `seq_x_txt` 或报告文本）

**Stage-1 Query（Q1）**：就用当前 one-shot 的 query 生成方式（复制那段字符串拼接逻辑即可）

然后调用你现有的检索函数（比如 `_retrieve_evidence(query, topk)`）：

- `E0 = _retrieve_evidence(Q1, topk=stage1_topk)`

> 这里不要发明新检索器：复用 TF-IDF 那套。

------

### 5.2 生成趋势假设（z0）：尽量复用 `_generate_cot(...)`

要求：趋势假设**短**、**稳定**、便于写回 query。

实现方式（两种都行，推荐 A）：

**A. 结构化 JSON（推荐）**
 让 `_generate_cot` 的 prompt 明确要求输出 JSON（direction/strength/volatility/key_factors）。

输出 `z0_json_str`，若解析失败则当普通字符串使用。

**B. 简短趋势摘要文本（fallback）**
 若没有 `cot_model` 或生成失败，用模板基于 `slope/std` 生成一句话：

- “likely upward / downward / flat”
- “volatility low/med/high”

> 重点：z0 不是最终预测，只是给 Stage-2 检索提供“趋势语义锚点”。

------

### 5.3 Gate/回退（必须有，防止 MSE/MAE 崩）

在 two-stage 开始前，做一个 gate：

- 如果原始文本为空 AND `abs(slope) < trend_slope_eps`：
  - 直接回退为 one-shot（或只用 Stage-1）
- 如果 `E0` 为空：
  - 回退 one-shot（或直接返回原始文本）

推荐逻辑：

```
if gate触发:
   return 原one-shot输出（最安全）
```

> 注意：回退必须走“原one-shot”，保证稳定且可对比。

------

### 5.4 Stage-2：把 z0 写回 query，再检索得到 E1

**Stage-2 Query（Q2）** 在 Q1 基础上追加：

```
[TREND HYPOTHESIS]
{z0}

Retrieve evidence that best supports/explains this trend hypothesis and is useful for forecasting.
```

然后检索：

- `E1 = _retrieve_evidence(Q2, topk=stage2_topk)`

若 `E1` 为空，回退为 `E0`（或回退 one-shot）。

------

### 5.5 输出 final_text：用固定模板拼接（不要让 Codex随意改格式）

输出 `final_text` 的模板固定为：

```
[RAW TEXT]
{raw_text_or_NA}

[NUMERICAL SUMMARY]
{S_num}

[TREND HYPOTHESIS]
{z0}

[RETRIEVED EVIDENCE - REFINED]
1) {e1}
2) {e2}
...
```

要求：

- 每条 evidence 可以截断（例如最多 300~500 字符）防止文本过长
- 保持结构一致（利于 BERT/LLM 编码稳定）

------

## 6) Debug 能力（强烈建议，但默认关闭）

在 `RAGCoTPipeline` 加一个 `debug=False`（或使用已有日志方式），当 debug 打开时返回更多信息：

- Q1 / E0
- z0
- Q2 / E1

但**Dataset 的 `texts` 仍旧只返回 final_text**，debug 信息可打印或写到 `batch` 的额外字段（可选，默认不启用，避免训练 I/O）。

------

## 7) 验收标准（Codex 必须满足）

### 7.1 向后兼容

- 不启用 `--use_two_stage_rag`：输出 `texts` 与当前版本一致（允许极小差异但建议完全一致）

### 7.2 可运行

- 启用 `--use_two_stage_rag`：训练/测试至少能跑过一个 batch，不崩溃

### 7.3 稳定回退

- raw_text 缺失、E0/E1 为空、cot_model 不可用时：必须能回退，不影响主流程

------

## 8) 极简测试脚本（Codex 可以加一个 debug 文件）

新增 `debug_two_stage_rag.py`（可选）：

- 随机取一个样本 index
- 打印 Q1/E0/z0/Q2/E1
- 打印 final_text 前 600 字

不影响训练代码。

------

## 9) 最重要的实现纪律（防止 Codex “自由发挥改坏”）

- ✅ **只在 `rag_cot.py` 内实现 two-stage 逻辑**（核心）
- ✅ Dataset 仅透传新 args，不重构 data pipeline
- ✅ `build_guidance_text` 的对外接口不改
- ✅ one-shot 原逻辑尽量不动（复制/封装都可，但输出必须一致）
- ❌ 不改任何 diffusion / model forward / CFG 相关代码