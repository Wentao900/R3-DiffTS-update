# Evidence Consistency Score（一致性评分）→ 接入 Trend-aware Hard CFG（方案 B）

> 目标：让 RAG 不仅输出 Top-K 证据文本 `E`，还输出一个量化的一致性分数 `consistency ∈ [0,1]`，用于调制扩散过程中的动态引导权重 `w_trend(k)`，从而降低“错证据强引导”导致的扩散路径偏移风险。

---

## 1. 输出接口（RAG 的升级输出）

**原始 RAG 输出：**

- `E = [e₁, e₂, ..., e_K]`（Top-K 证据文本）

**升级后的 RAG 输出：**

- `E = [e₁, e₂, ..., e_K]`
- `stance = [s₁, s₂, ..., s_K]`，其中 `sᵢ ∈ {up, down, flat, unknown}`
- `stance_probs = {p_up, p_down, p_flat, p_unknown}`
- `consistency ∈ [0,1]`（核心指标）
- （可选）`conflict_flag ∈ {0,1}`

---

## 2. Evidence Stance 判别（证据立场识别）

小型 LLM 判别

------

## 3. 一致性评分（Consistency）计算方式

### 3.1 基础一致性（主导立场比例）

统计：

```
n_up, n_down, n_flat, n_unknown
K = len(E)

p_up = n_up / K
p_down = n_down / K
p_flat = n_flat / K
p_unknown = n_unknown / K
```

定义一致性分数：

```
consistency = max(p_up, p_down, p_flat)
```

解释：

- `consistency → 1`：证据高度同向
- `consistency ≈ 1/3`：证据严重冲突
- `unknown` 增多会自然拉低一致性

------

### 3.2 带惩罚项的一致性（可选增强）

```
consistency = max(p_up, p_down, p_flat) * (1 - p_unknown)
```

或加入冲突惩罚：

```
conflict = min(p_up + p_down, 1.0)
consistency = max(p_up, p_down, p_flat)
              * (1 - p_unknown)
              * (1 - λ * conflict)
```

其中 `λ ∈ [0,1]` 为冲突惩罚系数。

------

## 4. 一致性接入 Trend-aware Hard CFG

### 4.1 原始 Hard CFG 权重

wtrend(k)=α⋅g(k)⋅h(ztrend)w_trend(k) = α · g(k) · h(z_trend) wtrend(k)=α⋅g(k)⋅h(ztrend)

### 4.2 加入一致性调制（推荐）

wtrend(k)=α⋅g(k)⋅h(ztrend)⋅consistencyw_trend(k) = α · g(k) · h(z_trend) · consistency wtrend(k)=α⋅g(k)⋅h(ztrend)⋅consistency

含义：

- 证据一致 → 强引导
- 证据冲突 / 不确定 → 自动弱引导
- 防止错误 RAG 证据放大趋势偏差

------

## 5. 与趋势先验 `z_trend` 的联动（可选但很强）

### 5.1 强度收缩（趋势置信度）

```
s_strength ← s_strength * consistency
```

### 5.2 波动增强（证据冲突）

```
v_volatility ← v_volatility + (1 - consistency)
```

形成闭环：

```
RAG evidence → stance → consistency
        ↘                ↙
     z_trend update & w_trend(k) modulation
```

------

## 6. 训练与推理阶段使用建议

- **训练阶段**：默认开启一致性调制，提高鲁棒性

- **推理阶段**：强烈建议开启，防止坏证据放大

- **极端情况处理**：

  - 若 `consistency < τ`（如 0.4）：

    ```
    w_trend(k) = 0
    ```

    直接退化为无条件扩散路径

------

## 7. 消融实验设计建议

| 实验设置                | 描述                   | 预期现象         |
| ----------------------- | ---------------------- | ---------------- |
| Base                    | RAG + CoT + 固定 CFG   | 易被错误证据带偏 |
| +Consistency            | 一致性调制 CFG         | 鲁棒性明显提升   |
| +Consistency + z_update | 一致性同时影响 z_trend | 趋势更保守、稳定 |
| Random Consistency      | 随机一致性             | 性能无系统提升   |

------

## 8. 模块级修改清单（落点）

### 8.1 `utils/rag_cot.py`

- 新增：
  - `stance_list`
  - `consistency_score`
- 随 RAG 输出一并返回

### 8.2 `data_provider/data_loader.py`

- `__getitem__` 中增加：
  - `consistency_score`

### 8.3 扩散模型（Hard CFG）

- 在计算 `w_trend(k)` 时乘以 `consistency`
- 或将 `consistency` 作为 side_info 参与调制