# 方案 B：基于 CoT 的扩散去噪路径调制（Trend-aware Hard CFG）

## 1. 设计动机（Motivation）

在多模态扩散时间序列预测中，文本信息（如新闻、报告、事件描述）往往具有**低频、弱监督、高噪声**的特点。
 现有方法通常将文本嵌入作为静态条件输入扩散模型，但这种方式存在两个关键问题：

1. **文本条件的影响在整个扩散过程中是静态的**，无法根据去噪阶段动态调整；
2. **文本推理（如 CoT）只作为语义增强信号**，并未真正参与扩散路径的建模。

为此，我们提出一种 **基于 Chain-of-Thought 的趋势感知扩散路径调制机制（Trend-aware Hard CFG）**，
 使 CoT 不再仅作为文本嵌入，而是**直接参与扩散去噪动力学的控制**。

------

## 2. 核心思想（Key Idea）

> **将 CoT 从“文本条件”提升为“去噪路径调制信号”，
>  使其动态控制扩散模型在不同去噪阶段对趋势信息的依赖强度。**

具体而言：

- CoT 不直接输入扩散模型；
- CoT 被解析为显式趋势先验；
- 趋势先验用于 **调制 classifier-free guidance（CFG）在不同 diffusion step 的权重**。

------

## 3. 总体结构概览

```
Historical Time Series + Text
        ↓
   RAG + CoT Generator
        ↓
   Structured Trend Prior z_trend
        ↓
 Trend-aware CFG Scheduler
        ↓
 Diffusion Denoising Process
```

------

## 4. CoT → 趋势先验建模

### 4.1 CoT 结构化输出约束

CoT 生成阶段采用结构化提示，要求模型输出如下格式：

```
{
  "direction": "up | down | flat",
  "strength": "weak | moderate | strong",
  "volatility": "low | medium | high",
  "reasoning": "natural language explanation"
}
```

其中：

- `reasoning` 仅用于可解释性分析；
- `direction / strength / volatility` 用于后续扩散路径调制。

------

### 4.2 趋势先验表示

将趋势信息映射为数值向量：

```
z_trend = [
  d_dir,        # ∈ {-1, 0, +1}
  s_strength,   # ∈ ℝ⁺
  v_volatility  # ∈ ℝ⁺
]
```

该向量不作为文本嵌入，而是作为**扩散调制信号**。

------

## 5. Trend-aware Hard CFG（核心修改点）

### 5.1 原始 CFG 机制（回顾）

标准 classifier-free guidance：

y^k=yuncond+w⋅(ycond−yuncond)\hat{y}_k = y_{\text{uncond}} + w \cdot (y_{\text{cond}} - y_{\text{uncond}})y^k=yuncond+w⋅(ycond−yuncond)

其中：

- www 为常数超参数；
- 文本影响在所有 diffusion step 中固定。

------

### 5.2 趋势感知 CFG（本文方法）

我们将固定权重 www 替换为**趋势与时间相关的动态权重函数**：

y^k=yuncond+wtrend(k,ztrend)⋅(ytrend−yuncond)\hat{y}_k = y_{\text{uncond}} + w_{\text{trend}}(k, z_{\text{trend}}) \cdot \left(y_{\text{trend}} - y_{\text{uncond}}\right)y^k=yuncond+wtrend(k,ztrend)⋅(ytrend−yuncond)

其中：

- kkk 为当前 diffusion step；
- ztrendz_{\text{trend}}ztrend 为 CoT 生成的趋势先验。

------

### 5.3 动态权重函数设计

一种可实现的简单形式为：

wtrend(k)=α⋅g(k)⋅h(ztrend)w_trend(k) = α · g(k) · h(z_trend) wtrend(k)=α⋅g(k)⋅h(ztrend)

- **时间调制项** g(k)g(k)g(k)：

  - 前期去噪（大噪声）：文本影响弱
  - 后期去噪（小噪声）：文本影响强
     例如：

  g(k)=1−kKg(k) = 1 - \frac{k}{K}g(k)=1−Kk

- **趋势调制项** h(ztrend)h(z_{\text{trend}})h(ztrend)：

  - 强趋势 → 更强引导
  - 高波动 → 抑制引导
     例如：

  h(z)=sstrength⋅exp⁡(−vvolatility)h(z) = s_{\text{strength}} \cdot \exp(-v_{\text{volatility}})h(z)=sstrength⋅exp(−vvolatility)

------

## 6. 与原方法的关键差异

| 维度       | 原始文本条件 | 本文方案 B                     |
| ---------- | ------------ | ------------------------------ |
| CoT 作用   | 文本增强     | 去噪路径调制                   |
| 文本影响   | 静态         | 动态（随 diffusion step 变化） |
| 条件位置   | embedding 层 | diffusion 动力学               |
| 可解释性   | 弱           | 强（趋势级）                   |
| 机制层影响 | 无           | 有                             |

------

## 7. 优势分析（Why This Is Stronger）

1. **机制级创新**
    CoT 直接影响扩散去噪轨迹，而非仅作为条件输入。
2. **鲁棒性提升**
    文本影响被限制在合适的去噪阶段，降低错误文本的破坏性。
3. **趋势一致性约束**
    扩散模型在生成过程中被显式引导遵循趋势先验。
4. **可解释性增强**
    每条预测路径都可对应一个明确的趋势调制策略。

------

## 8. 消融实验建议（用于支撑方案 B）

- 固定 CFG vs Trend-aware CFG
- 使用 CoT 但不调制 CFG
- 随机趋势 vs CoT 生成趋势
- 不同 diffusion step 调制曲线对比

------

## 9. 小结

> 本方案将 Chain-of-Thought 从“文本生成工具”提升为“扩散路径控制信号”，
>  通过趋势感知的动态 CFG 机制，使文本信息在扩散过程中以更合理、可控的方式发挥作用。