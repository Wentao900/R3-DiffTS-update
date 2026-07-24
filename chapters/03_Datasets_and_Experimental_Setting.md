# 3 数据集与实验设置

## 3.1 章节定位与实验范围说明

本章用于给出本文实验部分的可复现背景，内容严格对应当前代码分支中的脚本、配置与评估实现。为了避免叙述层级混淆，本章将实验范围拆分为两个互补层面。一个层面是基准覆盖层面，即仓库主线脚本已经支持的跨数据域评测设置，它反映本文方法在 Time-MMD 基准上的统一运行方式。另一个层面是当前分支的重点分析层面，即围绕 SocialGood 数据域组织的关键对照与消融组合，它承载第 4 章将展开的细粒度讨论。两者的关系是“全域设置给出方法适用边界，重点设置给出机制分析入口”，而不是以单一局部实验替代整个基准结论。

从工程入口看，实验统一通过 `exe_forecasting.py` 执行。该脚本并非只负责读取 YAML，而是把命令行参数、配置文件参数和数据侧统计结果融合为一次完整运行所需的最终配置，并将该配置写入运行目录下的 `config_results.json` 与 `run_summary.json`。这意味着本文描述的实验设置不是静态模板，而是“配置输入—数据统计—运行时归一化—训练评估”的闭环。后续各节都以这一闭环为依据，不引入仓库中不存在的流程，也不将尚未执行的结果写成结论。

## 3.2 基准覆盖与任务设置

仓库的主线覆盖由 `scripts/run_all_datasets_mainline.sh` 与 README/README.zh 中的说明共同界定。该脚本以同一入口循环运行 8 个领域数据，分别是 Traffic、SocialGood、Health_US、Environment、Energy、Economy、Climate 与 Agriculture。每个数据域在脚本中显式绑定 `data_path`、配置文件、历史窗口长度 `seq_len`、预测窗口长度 `pred_len` 与时间频率 `freq`，并通过同一组公共参数传递 `root_path`、`guide_w`、`nsample`、`device` 与 `valid_interval`。在这一组织方式下，跨领域比较的核心约束来自“统一执行框架+数据域特定窗口”，而不是每个数据集各自独立的训练代码。

具体任务形态遵循多模态时间序列预测定义。模型输入由历史窗口中的数值观测与时间相关文本共同构成，输出为未来窗口的条件生成样本及其点预测。`dataset_forecasting.py` 在 `datatype=multimodal` 路径下调用 `data_provider` 构建 train/valid/test 三个数据加载器，并把训练数据的均值和标准差回传给评估模块用于反标准化。对跨域实验而言，脚本中的窗口配置体现了任务时间尺度差异，例如月频场景采用 `36->12`，周频场景采用 `96->12`，环境场景采用 `336->48`，这与 README 中“按数据域保持原始窗口尺度”的说明一致。

在运行时，`exe_forecasting.py` 还会对模型输入维度进行归一化处理。脚本根据频率设置时间特征维度，并将 `lookback_len`、`pred_len`、`text_len`、目标域名等信息回写到 `config["model"]`。模型构建前还会读取训练集暴露的 `feature_dim` 与 `target_index`，从而确定目标维度与索引位置。该路径保证了“同一脚本适配多域”时的输入一致性，也避免把维度信息硬编码到模型内部。

需要强调的是，仓库主线可以报告基准层面的统一设置，但当前分支的关键比较并非平均铺开在 8 个域，而是集中到 SocialGood 的配置组上开展机制分析。这个区分在脚本层同样清晰：一个脚本负责全域覆盖，另一个脚本专门负责 SocialGood 的多配置多种子扫描。

## 3.3 训练实现与实现细节

训练过程由 `utils/utils.py` 中的 `train(...)` 实现。优化器采用 Adam，学习率来自配置 `train.lr`，并固定 `weight_decay=1e-6`。学习率调度采用 MultiStepLR，里程碑按总 epoch 的 75% 与 90% 自动计算，衰减系数为 0.1。若配置了 `lr_warmup_epochs`，训练前期执行线性 warmup，warmup 结束后再回到原始调度。该实现和 README 中“warmup 与原调度共存”的描述一致。

梯度稳定化通过 `max_grad_norm` 控制，当该值大于 0 时启用 `clip_grad_norm_`。每个 epoch 的最大迭代批次数受 `itr_per_epoch` 限制，代码中取 `min(len(train_loader), itr_per_epoch)`，因此能够在大数据加载器下控制每轮训练成本。验证集评估按 `valid_epoch_interval` 触发，若验证损失下降则覆盖保存 `model.pth`。若没有可用验证集或未触发“最佳更新”，训练结束时也会写出最终权重，避免运行目录缺少可评估模型。

实验配置中的多分辨率监督同样在训练前由 `exe_forecasting.py` 解析。`multi_res_horizons` 的解析顺序是显式配置优先，再用训练集 ACF 统计自动生成，仍无法确定时回退到基于 `pred_len` 的比例默认集合。该解析过程不仅给出 horizon 列表，还会计算 lag 级与区段级可靠性，并将 `multi_res_horizon_reliabilities` 与 `multi_res_segment_reliabilities` 回写到训练配置。模型训练据此接收“监督边界+可靠性先验”的组合输入，形成与数据统计一致的多尺度优化目标。

数据侧增强参数通过 `dataset` 或 `data` 子配置注入，包括 `aug_noise_std`、`aug_time_warp_prob`、`aug_segment_scale_std` 等。文本相关统计阈值与权重也在脚本中统一解析，如 `text_quality_weights`、`text_trust_ret`、`text_trust_cot`、`text_quality_drop_threshold` 与 `text_quality_mid_threshold`。这些参数属于实验设置与训练稳健化机制，不在本文中作为独立创新点陈述。

## 3.4 评估流程、点预测协议与评价指标

评估流程由 `utils/utils.py` 的 `evaluate(...)` 与 `fit_forecast_calibrator(...)` 组成，并由 `exe_forecasting.py` 进行调度。模型在评估前加载当前运行目录的 `model.pth`，随后对测试批次生成多样本预测。生成样本会构造成候选点预测集合，候选包括样本均值、样本中位数、末端观测延拓、线性趋势外推、历史均值回归、事件衰减以及多周期季节复制等规则项。该候选机制用于统一点预测选择与校准输入，不改变扩散生成本身。

点预测选择遵循 `forecast_point_estimator`。当设置为 `mean` 或 `median` 时直接采用对应候选；当设置为 `auto` 或 `valid_auto` 且存在校准信息时，会依据校准器给出的 `base_estimator` 选择均值或中位数。评估同时记录 `MSE_mean`、`MSE_median` 与 `MSE_calibrated`，用于区分不同点预测策略下的误差行为。

概率评价使用 CRPS，实现位于 `calc_quantile_CRPS(...)`。代码以 0.05 到 0.95 的分位点网格计算分位损失并归一化求均值。点预测评价输出 MSE 与 MAE，其中 `evaluate(...)` 内部以归一化空间累积误差并最终除以有效预测点数，结果与 CRPS 一并写入 `metrics.json`。因此，本文实验协议中 CRPS 表示分布预测质量，MSE/MAE 表示点预测偏差，三者角色互补而非替代。

引导权重的选择也在评估阶段完成。若扩散配置启用 CFG 且 `guide_mode` 不是 `auto`，脚本会在验证集优先、测试集回退的规则下扫描一组 `guide_w`，并以 MSE 选择最佳权重，再用该权重重跑测试评估。若用户显式指定 `--guide_w` 或 `--guide_list`，扫描集合会被对应参数覆盖。该流程属于测试时超参数选择协议，不应与模型结构贡献混写。

预测校准器是可选后处理。`fit_forecast_calibrator(...)` 在验证集候选预测上按 horizon 拟合岭回归系数，保留截距与各候选权重，再依据 holdout 子集估计增益与应用强度。代码中用 `min_gain` 与 `max_strength` 控制是否启用与启用幅度，同时可用分位裁剪约束残差。校准器仅作用于点预测组合层，不参与主模型训练，不改写主模型的扩散生成路径。本章将其定义为评估支持机制，而不是本文方法创新主体。

## 3.5 关键比较与消融设计

当前分支用于重点分析的脚本是 `scripts/run_socialgood_scorehack_sweep.sh`。该脚本固定数据为 `SocialGood/SocialGood.csv`，固定窗口为 `36->12` 与月频 `m`，默认配置集合为 `socialgood_36_12_plain.yaml`、`socialgood_36_12_notext.yaml` 与 `socialgood_36_12_scorehack.yaml`，默认种子为 2021、2025、2029，并将各次运行日志保存到 `logs/socialgood_scorehack/`。脚本还提供文本丢弃概率覆盖参数，对 `notext` 配置默认使用 `NOTEXT_TEXT_DROP_PROB=1.0`，确保无文本对照成立。这一设计使得关键比较在同域同窗口同采样预算下进行，便于隔离机制差异。

结合配置文件可以得到本章采用的比较语义。`socialgood_36_12_plain.yaml` 对应不使用文本、也不启用结构化残差扩散主干的参考基线，可视为较为朴素的数值预测设置。`socialgood_36_12_notext.yaml` 则进一步保留结构化残差扩散主干与自动控制路径，但关闭文本证据通道，并通过极高文本丢弃确保文本不发挥作用，用于考察“仅保留主干结构、移除文本证据”时的行为。`socialgood_36_12_scorehack.yaml` 在 SocialGood 条件下同时保留结构化残差扩散与结构化文本证据，并启用校准相关设置，是当前重点分析中的完整配置。`socialgood_36_12_backbone_coarse.yaml` 与 `socialgood_36_12_conservative.yaml` 可作为补充分析配置，前者用于观察无文本条件下加入 coarse 辅助头后的骨干表现，后者用于观察更保守文本使用与更强辅助机制并存时的稳健性边界。

本章对这些配置的解释遵循主次贡献边界。结构化残差扩散对应主线比较，结构化文本证据对应次线比较，两条线形成第 4 章的核心消融框架。scorehack 与 conservative 这类策略组合只作为补充分析维度，用于展示不同风险偏好或预算约束下的设置差异，不作为贡献定义层面的决定性实验。控制器、校准器、辅助头在叙述中保持“可选机制”定位，避免把工程策略误写为理论创新。

从可复现角度看，SocialGood 重点组与基准全域组是并行存在的。前者用于解释方法机制，后者用于说明本文方法在多域任务上的统一可运行性。章节组织若混淆这两类证据，会导致读者误解“局部深入分析”与“全域覆盖设置”的关系。本章因此明确采用“全域设置先交代，重点配置再展开”的结构，使后续结果章节能够在同一术语体系下同时报告总体表现与重点机制观察。

## 3.6 本章小结

本章给出了与代码实现一致的数据集与实验设置。基准层面，仓库通过统一入口覆盖 Time-MMD 的 8 个数据域，并在数据域特定窗口与频率上运行主线配置。实现层面，训练采用 Adam、warmup 与多阶段学习率调度，并结合梯度裁剪和多分辨率 horizon 解析。评估层面，CRPS 与 MSE/MAE 共同构成分布与点预测指标体系，点预测通过候选集合选择，校准器仅作为可选后处理机制。重点分析层面，当前分支以 SocialGood 的 plain/notext/scorehack 为关键比较组，backbone_coarse 与 conservative 作为补充策略组。通过这一分层设置，本文能够在不虚构结果的前提下，为后续实验分析提供清晰、可追溯且与方法主次贡献一致的证据框架。
