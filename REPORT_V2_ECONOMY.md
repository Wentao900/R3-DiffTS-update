# Economy V2 固定版报告

## 1. 结论

当前仓库已固定为 `Economy V2` 主线版本，`V3 / V3.1` 的文本候选路由方案已从可运行主路径中移除。

当前推荐配置文件：

- `config/economy_36_12_scale_router_guide.yaml`

当前评测设置：

- `seq_len = 36`
- `pred_len = 12`
- `text_len = 36`
- `guide_w` 由验证集从候选集合中自动选择

当前最新结果：

- Checkpoint: `save/forecasting_Economy_20260312_194619`
- Metric file: `save/forecasting_Economy_20260312_194619/eval_metrics_guide_1p4.json`
- 验证集选出的 `guide_w = 1.4`
- `MSE = 0.2496992787744245`
- `MAE = 0.3801771555191431`

最终建议：`Economy` 任务后续全部以 `V2` 为基线，不再继续 `V3 / V3.1` 方向。

## 2. V2 方法概述

最终保留的 `V2` 由四部分组成：

1. 数据侧动态文本时间窗选择
2. 动态分段的多时间尺度辅助损失
3. `scale router` 的分段加权
4. 基于 router 的自适应 CFG 引导强度

这里不再使用文本候选集路由，也不再进行路由后文本回退。

## 3. 数学公式与修改方法

### 3.1 数据构造

对每个样本，输入历史窗口记为：

$$
\mathbf{x}_{1:L}
$$

预测目标记为：

$$
\mathbf{x}_{L+1:L+H}
$$

其中：

- $L = 36$
- $H = 12$

数据集先从历史数值序列中估计粗粒度时间尺度，再从候选文本窗中选择一个长度：

$$
\mathcal{W} = \{6, 18, 36\}
$$

该步骤输出：

- `texts`
- `text_mark`
- `trend_prior`
- `scale_code`
- `text_window_len`

其中趋势先验向量为：

$$
\mathbf{p} = [p_{\text{dir}}, p_{\text{strength}}, p_{\text{vol}}]
$$

含义分别是趋势方向、趋势强度、波动强度。

### 3.2 基础预测损失

原始主损失保持为预测区间上的去噪误差：

$$
\mathcal{L}_{\text{base}} =
\frac{\sum \left((\hat{\mathbf{x}} - \mathbf{x}) \odot \mathbf{m}\right)^2}{\sum \mathbf{m}}
$$

其中：

- $\hat{\mathbf{x}}$ 为模型预测
- $\mathbf{x}$ 为标准化后的目标
- $\mathbf{m}$ 为有效位置掩码

### 3.3 多时间尺度辅助损失

将预测区间切成 4 个时间段：

$$
\mathcal{B} = \{[1], [2,3], [4,6], [7,12]\}
$$

对每一段 $b$ 单独计算误差：

$$
\mathcal{L}_b =
\frac{\sum \ell(\hat{\mathbf{x}}_b - \mathbf{x}_b)}{\sum \mathbf{m}_b}
$$

其中 $\ell(\cdot)$ 使用 Huber 损失：

$$
\ell(r) =
\begin{cases}
\frac{1}{2}r^2, & |r| \le \delta \\
\delta |r| - \frac{1}{2}\delta^2, & |r| > \delta
\end{cases}
$$

本实验中：

$$
\delta = 1.0
$$

训练目标改为：

$$
\mathcal{L} =
\mathcal{L}_{\text{base}} + \lambda_{\text{mr}} \mathcal{L}_{\text{mr}}
$$

其中：

$$
\lambda_{\text{mr}} = 0.1
$$

也就是说，`V2` 的第一处核心修改是：

- 在原始 diffusion forecasting 主损失外，增加一个多时间尺度辅助项
- 该辅助项不再对整个 horizon 一刀切，而是按短期到长期分段建模

### 3.4 Scale Router 加权

为了避免所有 horizon band 被同等看待，`V2` 引入 `scale router`。其输入为历史统计特征、趋势先验和文本可用性标记：

$$
\mathbf{f} =
[f_{\text{slope}}, f_{\text{abs-slope}}, f_{\text{vol}}, f_{\text{diff-std}}, f_{\text{accel}}, f_{\log |x|}, \mathbf{p}, m_{\text{text}}]
$$

router 输出 band logits：

$$
\mathbf{z} = g_{\theta}(\mathbf{f})
$$

再经 softmax 得到 band 权重：

$$
\mathbf{w} = \text{softmax}\left(\frac{\mathbf{z}}{\tau}\right)
$$

本实验中：

$$
\tau = 1.0
$$

最终多尺度损失写成：

$$
\mathcal{L}_{\text{mr}} =
\sum_{b=1}^{B} \tilde{w}_b \mathcal{L}_b
$$

这里的 $\tilde{w}_b$ 不是单纯的 router 输出，而是三部分的稳定化融合：

- 样本级 router 权重
- 全局 EMA 难度权重
- 一个 uniform floor

因此，`V2` 的第二处核心修改是：

- 不再手工固定每个 horizon 段的权重
- 改为让 router 对不同样本自适应分配短期/中期/长期关注度

### 3.5 Teacher 正则

数据侧动态文本窗会同步给出 `scale_code`，它可作为一个弱监督教师信号。训练 warmup 阶段加入 KL 正则：

$$
\mathcal{L}_{\text{teacher}} =
\alpha_{\text{teacher}} \cdot
\mathrm{KL}(\mathbf{w} \parallel \mathbf{w}^{\text{teacher}})
$$

其中：

$$
\alpha_{\text{teacher}} = 0.1
$$

这一步的作用是：

- 让 router 初期不要完全自由漂移
- 先与数据侧尺度判断保持一致，再逐步学到更细的加权策略

### 3.6 Router-aware CFG

标准 classifier-free guidance 为：

$$
\hat{\epsilon}_{\text{cfg}} =
\hat{\epsilon}_{\text{uncond}} +
g \cdot (\hat{\epsilon}_{\text{cond}} - \hat{\epsilon}_{\text{uncond}})
$$

`V2` 不再对所有样本使用固定的 $g$，而是先根据 router 权重计算一个样本级尺度得分：

$$
s = \sum_{b=1}^{B} w_b c_b
$$

其中 $c_b$ 是第 $b$ 个 horizon band 的中心位置。

然后将其映射成引导比例：

$$
r = \mathrm{clip}\left(1 + \alpha (s - 0.5), r_{\min}, r_{\max}\right)
$$

其中：

- $\alpha = 0.6$
- $r_{\min} = 0.5$
- $r_{\max} = 1.5$

于是样本级 guidance 变为：

$$
g_i = r_i \cdot g_0
$$

当前这次最新实验中，验证候选集合为：

$$
\mathcal{G} = \{0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4\}
$$

最终选中的基准引导系数为：

$$
g_0 = 1.4
$$

最终采样公式变成：

$$
\hat{\epsilon}_{\text{cfg}}^{(i)} =
\hat{\epsilon}_{\text{uncond}}^{(i)} +
g_i \cdot
\left(\hat{\epsilon}_{\text{cond}}^{(i)} - \hat{\epsilon}_{\text{uncond}}^{(i)}\right)
$$

这一步是 `V2` 的第三处核心修改：

- 把原先固定的 CFG 强度改成样本自适应
- 短期主导和长期主导的样本，可以自动使用不同的引导幅度

## 4. 代码修改位置

本次固定 `V2` 主要涉及以下文件：

- `data_provider/data_loader.py`
- `data_provider/data_factory.py`
- `main_model.py`
- `exe_forecasting.py`
- `utils/utils.py`

对应关系如下：

### 4.1 数据侧改动

`data_provider/data_loader.py`

- 增加 `dynamic_text_len / dynamic_text_lens / scale_aware_rag`
- 基于数值历史估计尺度
- 生成 `scale_code` 与 `text_window_len`
- 保留动态文本窗口，但移除 `V3/V3.1` 的文本候选返回逻辑

`data_provider/data_factory.py`

- 仅训练集使用 `shuffle=True, drop_last=True`
- `valid/test` 改为 `shuffle=False, drop_last=False`
- 避免评测阶段 dataloader 空批次

### 4.2 模型侧改动

`main_model.py`

- 增加 `ScaleRouter`
- 增加 band 切分与多尺度损失计算
- 增加 teacher regularization
- 增加 router-aware CFG
- 删除 `V3/V3.1` 的文本选择路由分支

关键函数：

- `main_model.py::_compute_scale_router_weights`
- `main_model.py::_calc_multi_res_loss`
- `main_model.py::_compute_router_guidance`
- `main_model.py::get_scale_router_diagnostics`

### 4.3 训练与评测侧改动

`exe_forecasting.py`

- 优先从 YAML 读取 `seq_len / pred_len / text_len`
- 固定 `guide_w_candidates`
- `cfg` 模式下先在 `valid` 上选 guide，再在 `test` 上评测
- 避免再次出现误跑成 `36/18`

`utils/utils.py`

- 增加空 dataloader 保护
- 支持按 `split` 保存评测结果
- 增加 horizon、band、router 诊断统计

## 5. 固定配置

当前使用配置文件：

- `config/economy_36_12_scale_router_guide.yaml`

该配置当前关键项为：

- `guide_w_default: 0.8`
- `guide_w_candidates: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4]`
- `seq_len: 36`
- `pred_len: 12`
- `text_len: 36`

实际运行时，本次实验最终由验证集自动选择 `guide_w = 1.4`。

## 6. 复现实验命令

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

注意：

- 当前命令使用相对路径 `../Time-MMD-main`
- `data_path` 应写为 `Economy/Economy.csv`
- 不要再写成 `Time-MMD-main/numerical/Economy/Economy.csv`

## 7. 删除的分支

以下实验分支已从当前主线路径移除：

- `V3`: router 驱动的文本候选选择
- `V3.1`: 基于置信度门控的文本候选回退

删除原因：

- `MSE` 和 `MAE` 相比 `V2` 变差
- 文本路由缺少稳定监督
- 引入了额外不确定性，但没有带来收益

## 8. 最终建议

`Economy` 数据集后续可以按下面顺序推进：

1. 以当前 `V2` 作为唯一基线版本
2. 只在 `guide_w=0.5` 固定条件下继续做细化 ablation
3. 后续若要继续优化，优先动 `router` 和 `multi-resolution loss`，不要再回到文本候选路由
