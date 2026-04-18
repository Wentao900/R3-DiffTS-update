# Counterfactual Utility-Gated Guidance 方法方案

## 1. 当前问题定义

当前模型已经证明文本信息对预测是有用的，但文本的两条使用路径表现不同：

```text
text context path: 有稳定收益
sampling guidance path: 不稳定，容易无效或伤害指标
```

最新 Economy 实验显示：

```text
动态 context_gate 版 MSE: 0.25517
动态 guide_gate 版 MSE:   0.24802
```

这说明文本 context 分支继续带来收益。但同时当前 `guide_gate.mean = 0.00124`，`guide_w` 从 `0` 扫到 `5` 几乎没有指标变化，说明 sampling guidance 基本被安全关闭。

因此，当前根本问题不是继续微调 guidance 强度参数，而是 guidance 缺少一个可靠的使用依据：

```text
文本质量高 != 文本条件分支一定能降低当前样本的预测误差
```

方法上应从 quality-gated guidance 升级为 utility-gated guidance。

## 2. 已完成工作

### 2.1 文本检索和 CoT 缓存

已经加入持久化 RAG/CoT 缓存，避免每个 epoch 重复生成文本，降低训练耗时。

核心目标：

```text
相同数据窗口 + 相同检索文本 + 相同数值历史 -> 复用 CoT 输出
```

该改动不改变训练、验证、测试流程，只减少重复文本生成成本。

### 2.2 事件级文本质量建模

当前文本证据向量为：

```text
e_i = [q_i, d_i, f_i, n_i, a_i, r_i, v_i]
```

其中：

```text
q_i: text quality
d_i: report/retrieval density
f_i: freshness
n_i: novelty
a_i: multi-source agreement
r_i: numeric regime shift
v_i: event trigger score
```

这使文本质量判断从单一文本存在性，升级为样本级、多证据维度的细粒度判断。

### 2.3 动态 context_gate

已经把训练/推理中的文本上下文注入改成动态门控：

```text
g_ctx(i,h) = min(s_i,h * rho_ctx(e_i), m_ctx(e_i))
```

其中：

```text
s_i,h: strength_gate
rho_ctx(e_i): 基于文本质量、密度、新鲜度、一致性、事件强度的动态折扣
m_ctx(e_i): 动态上限
```

作用：

```text
高质量文本 -> 允许进入 context
低质量文本 -> 自动降低注入强度
```

实验结果表明该方向有效，Economy 和 Agriculture 都得到改善。

### 2.4 动态 guide_gate 和 step_gate

已经把 sampling guidance 从直接使用 `strength_gate` 改为：

```text
guide_gate = f(strength_gate, evidence, trend_align)
```

并加入 diffusion step gate：

```text
step_gate(t) =
sigmoid(k * (tau - low)) * sigmoid(k * (high - tau))
```

其中：

```text
tau = t / (T - 1)
```

作用：

```text
early noisy steps: 降低 guidance
middle steps: 允许 guidance
late denoising steps: 降低 guidance
```

该改动解决了 guidance 伤害的问题，但当前 gate 太保守，导致 guidance 基本失效。

## 3. 当前方法的不足

当前 guide_gate 的依据仍然主要是文本证据质量：

```text
quality / freshness / density / agreement / event_score / trend_align
```

这些指标只能说明文本“看起来可信”，不能说明：

```text
有文本条件分支 pred_cond 是否真的比无文本分支 pred_uncond 更可靠
```

因此会出现两种极端：

```text
gate 放宽: guidance 可能伤害 Agriculture
gate 收紧: guidance 对 Economy/Agriculture 都几乎无作用
```

根本解决方向是让模型在采样时进行反事实效用判断。

## 4. Counterfactual Utility-Gated Guidance

### 4.1 核心思想

在 CFG sampling 中，模型本来就会得到：

```text
pred_cond(i,t): 有文本条件的预测
pred_uncond(i,t): 无文本条件的预测
```

不要直接使用：

```text
pred = pred_uncond + w * (pred_cond - pred_uncond)
```

而是先判断文本条件分支是否真的更好。

关键原则：

```text
如果 pred_cond 在已知 lookback 区间上比 pred_uncond 更一致，则文本 guidance 有正效用。
如果 pred_cond 连已知历史都解释不好，则降低或关闭文本 guidance。
```

该判断只使用输入窗口中的已观测 lookback，不使用未来标签，因此不改变训练、验证、测试流程，也不引入测试泄漏。

### 4.2 反事实历史效用

设已知 lookback 观测区域为：

```text
Omega_i
```

有文本条件分支误差：

```text
E_c(i,t) =
mean_{Omega_i} || pred_cond(i,t) - x_obs(i) ||^2
```

无文本条件分支误差：

```text
E_u(i,t) =
mean_{Omega_i} || pred_uncond(i,t) - x_obs(i) ||^2
```

定义历史效用：

```text
U_past(i,t) =
ReLU((E_u(i,t) - E_c(i,t)) / (E_u(i,t) + eps))
```

含义：

```text
U_past > 0: 文本条件分支更好，允许 guidance
U_past = 0: 文本条件分支没有优势，关闭或降低 guidance
```

### 4.3 前瞻事件效用

只用 lookback consistency 可能会误杀未来事件文本。例如 Economy 中，文本可能描述的是未来政策、冲击或趋势变化，而不是当前历史形态。

因此定义事件效用：

```text
U_event(i) =
q_i * f_i * a_i * v_i * r_i
```

其中：

```text
q_i: 文本质量
f_i: 新鲜度
a_i: 多来源一致性
v_i: 事件触发强度
r_i: 数值 regime shift
```

该项表示文本是否具有可信的前瞻事件信息。

### 4.4 Horizon-aware 融合

不同预测步对文本的依赖不同。近端预测应更依赖历史一致性，远端预测可以更多使用事件文本。

定义 horizon 位置：

```text
lambda_h = h / (H - 1)
```

最终效用：

```text
U(i,h,t) =
(1 - lambda_h) * U_past(i,t)
+ lambda_h * max(U_past(i,t), U_event(i) * trend_align(i,h))
```

含义：

```text
near horizon: 主要由 U_past 控制
far horizon: 允许高质量事件文本发挥作用
trend_align: 约束文本趋势和数值趋势的一致性
```

### 4.5 最终 guidance 权重

保留已有 strength gate 和 step gate：

```text
w_eff(i,h,t) =
guide_w * strength_gate(i,h) * U(i,h,t) * step_gate(t)
```

其中：

```text
step_gate(t) =
sigmoid(k * (tau - low)) * sigmoid(k * (high - tau))
```

最终 CFG：

```text
pred(i,h,t) =
pred_uncond(i,h,t)
+ w_eff(i,h,t) * [pred_cond(i,h,t) - pred_uncond(i,h,t)]
```

## 5. 实现方案

### 5.1 不改变流程

保持当前训练、验证、测试入口不变：

```text
train()
evaluate()
guide_w sweep
run_summary.json 输出
```

只修改模型内部 sampling guidance 权重。

### 5.2 新增函数

建议在 `main_model.py` 中新增：

```python
def _compute_counterfactual_utility(
    self,
    pred_cond,
    pred_uncond,
    observed_data,
    cond_mask,
    evidence_vec,
    trend_align,
):
    ...
```

输出：

```text
utility_gate: [B, pred_len]
```

### 5.3 修改 impute

当前 CFG 位置：

```python
predicted_cond, predicted_uncond = predicted[:B], predicted[B:]
```

在这里计算：

```text
U_past
U_event
U(i,h,t)
```

再用于：

```text
effective_guide_w_t = guide_w * strength_gate * U * step_gate
```

### 5.4 Debug 输出

run_summary 中继续输出：

```text
use_gate
strength_gate
context_gate
guide_gate
```

建议新增：

```text
utility_gate
past_utility
event_utility
```

这样下一轮实验可以判断：

```text
guidance 是因为文本没效用而关闭，
还是因为 gate 设计过严而关闭。
```

## 6. 实验判断标准

下一步实验不应只看 best MSE，还应看 guidance 是否有可解释行为。

成功标准：

```text
1. Economy 和 Agriculture 的 guide_w > 0 不再系统性伤害 MSE。
2. guide_w sweep 开始有可测差异，而不是完全平。
3. utility_gate 在高质量样本上非零，在低质量样本上接近零。
4. context_gate 继续保持当前收益。
5. 不需要按数据集手动设置不同参数。
```

如果最终结果仍然显示：

```text
guide_w = 0 最优
utility_gate 经常接近 0
```

则说明当前文本主要适合作为 learned context，不适合做 sampling CFG。此时方法应定位为：

```text
Text as learned context: 主路径
Utility-verified guidance: 安全可选增强路径
```

## 7. 当前结论

当前工作已经解决了两个问题：

```text
1. 文本 context 过强导致 Economy loss/metric 变差的问题。
2. sampling guidance 过强导致 guide_w > 0 伤害指标的问题。
```

但新的问题是：

```text
guide_gate 过于保守，guidance 基本失去作用。
```

因此下一步不应继续做参数微调，而应引入 Counterfactual Utility-Gated Guidance：

```text
用 pred_cond 和 pred_uncond 在已知历史上的反事实表现，动态判断文本 guidance 是否有用。
```

这是方法论上的改进，而不是数据集级别调参。
