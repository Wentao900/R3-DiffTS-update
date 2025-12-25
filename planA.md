## 方案 A：将 TAA + TTF 封装为一个模块（保持机制不变，最稳）

### 目标
不改变原框架的计算逻辑与信息流，只把每层的两步融合 **封装成一个统一模块**（外观/代码层面合并）：
- **先 TAA**：时间序列与时间戳/辅助序列的 self-attention
- **再 TTF**：文本 → 时间序列的 cross-attention

这样在网络结构图里只画一个“融合模块”，实现上也只维护一个类/函数，但行为与原设计等价。:contentReference[oaicite:0]{index=0}

---

### 输入输出与记号
- 时间序列 token：\(s_{l-1}\)
- 时间戳/辅助序列 token：\(u_{l-1}\)
- 文本 token（由预训练 LM 得到）：\(e\)

模块输入：\((s_{l-1}, u_{l-1}, e)\)  
模块输出：\((s_l, u_l)\)

---

### 模块内部流程（严格保持 “TAA → TTF”）
#### Step 1：TAA（Timestamp-Assisted Attention）
把时间序列与时间戳/辅助序列拼起来做自注意力：
\[
v_{l-1} = [\, s_{l-1};\ \lambda u_{l-1} \,]
\]
其中 \(\lambda\) 是可学习标量或超参，用于控制时间戳信息强度。:contentReference[oaicite:1]{index=1}

对 \(v_{l-1}\) 做多头自注意力（MSA）得到：
\[
v_l = [\, s'_l;\ u_l \,]
\]
此时得到更新后的时间戳分支 \(u_l\)，以及中间的时间序列表示 \(s'_l\)。:contentReference[oaicite:2]{index=2}

---

#### Step 2：TTF（Text-to-Time Fusion）
用文本 token \(e\) 作为 Key/Value，使用 \(s'_l\) 作为 Query 做 cross-attention，增强时间序列表示：
\[
s_l = s'_l + \mathrm{MCA}(Q=s'_l,\ K=e,\ V=e)
\]
其中 MCA 表示多头交叉注意力。:contentReference[oaicite:3]{index=3}

> 这一步的关键是：**文本只用来增强时间序列分支 \(s\)**，而非直接增强时间戳分支 \(u\)