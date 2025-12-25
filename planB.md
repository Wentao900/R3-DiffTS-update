## 方案 B：将 TAA + TTF “真正”合并为一次注意力（Single-block Multi-source Attention）

### 目标
把原来每层里的两步：
- **TAA**：时间序列与时间戳/辅助序列的 self-attention  
- **TTF**：文本 → 时间序列的 cross-attention  
合并为 **一次多源注意力（Multi-source Attention）**，在结构上成为一个统一模块（一个 Attention + 若干线性层/门控）。

---

### 记号与输入输出
- 时间序列 token：\(s_{l-1}\)
- 时间戳/辅助序列 token：\(u_{l-1}\)
- 文本 token（由预训练 LM 得到）：\(e\)
- 拼接后的“时间侧”token：  
  \[
  v_{l-1} = [\, s_{l-1};\ \lambda u_{l-1} \,]
  \]
  其中 \(\lambda\) 为可学习标量或超参，用于控制时间戳信息的强度。:contentReference[oaicite:0]{index=0}

模块输入：\((s_{l-1}, u_{l-1}, e)\)  
模块输出：\((s_l, u_l)\)

---

### 核心思想：一次 Attention 同时“看”时间侧与文本侧
将 **Key/Value** 扩展为两部分：  
- 来自时间侧（自身）：\(v_{l-1}\)  
- 来自文本侧：\(e\)

并把它们拼接成一个统一的记忆库：
\[
K = [\, W_k v_{l-1};\ W_k^e e \,],\quad
V = [\, W_v v_{l-1};\ W_v^e e \,]
\]

Query 仍来自时间侧：
\[
Q = W_q v_{l-1}
\]

单次注意力更新：
\[
\Delta v = \mathrm{Attn}(Q,\ K,\ V)
\]
残差更新：
\[
v_l = v_{l-1} + \Delta v
\]
最后拆回两路输出：
\[
[s_l;\ u_l] = \mathrm{split}(v_l)
\]

---

### 门控/结构约束（强烈建议）
#### 1) 门控文本注入强度（降低噪声文本影响）
将注意力输出拆分为“来自时间侧”和“来自文本侧”的贡献（概念上）：
\[
\Delta v = \Delta v_{\text{time}} + \Delta v_{\text{text}}
\]
引入门控 \(g \in [0,1]\)：
\[
v_l = v_{l-1} + \Delta v_{\text{time}} + g\cdot \Delta v_{\text{text}}
\]
其中 \(g\) 可由
- learnable scalar
- 或 \(g=\sigma(\mathrm{MLP}(\mathrm{pool}(v_{l-1})))\)
得到，用于自适应抑制不可靠的文本条件（与 CFG/无条件分支思路相容）。:contentReference[oaicite:1]{index=1}

#### 2) 保持与原设计一致的“只增强 s，不直接增强 u”
原框架中，文本 cross-attention 的 Query 是时间序列表示 \(s'_l\)，文本主要增强 \(s\)，而不是增强 \(u\)。:contentReference[oaicite:2]{index=2}  
为了不改变 inductive bias，建议增加结构约束之一：
- **Mask**：禁止 \(u\) token attend 到文本 token（\(u \nrightarrow e\)）
- 或者：在输出时只对 \(s\) 注入文本分量，\(u\) 仅来自时间侧 self-attn

---

### 伪代码（单模块）
```text
# Inputs: s, u, e
v = concat(s, λ*u)

Q = Wq(v)
K = concat(Wk(v), Wk_e(e))
V = concat(Wv(v), Wv_e(e))

Δv = Attention(Q, K, V)

# optional: split Δv into time/text parts, apply gate g on text part
v = v + Δv_time + g * Δv_text

s, u = split(v)
return s, u