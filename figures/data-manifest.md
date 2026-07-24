# Figure Data Manifest

| Figure | Purpose | Data source | Script | Status |
|---|---|---|---|---|
| Figure 1 | 方法总框架图：展示“结构化文本证据 → 模式基线 → 残差扩散 → 最终预测”的整体流程 | `chapters/02_Methodology.md`, `PPT_outline_multimodal_forecasting.md`, `main_model.py`, `data_provider/data_loader.py`, `utils/rag_cot.py` | 待创建（建议 diagram 脚本或 draw.io / PPT 重绘） | recommended |
| Figure 2 | 文本证据构建图：展示 raw text、检索、趋势假设、质量评估、窗口证据向量的形成路径 | `chapters/02_Methodology.md`, `data_provider/data_loader.py`, `utils/rag_cot.py` | 待创建（建议 diagram 脚本） | recommended |
| Figure 3 | SocialGood 关键配置组均值误差对比图：比较 plain / notext / scorehack / backbone_coarse / conservative 的 CRPS、MSE、MAE 均值 | `tables/socialgood_grouped_results.csv` | `figures/results/socialgood_group_mean_bars.py` | generated |
| Figure 4 | SocialGood 均值-最优对照图：对 plain / notext / scorehack 展示均值与最佳单次运行的差异，强调“均值优势”与“单次最优”不完全一致 | `tables/socialgood_grouped_results.csv` | `figures/results/socialgood_mean_best_gap.py` | generated |
| Figure 5 | SocialGood 误差区间图：对各组展示 min-mean-max 区间，辅助说明 conservative 波动大、scorehack 区间更集中 | `tables/socialgood_grouped_results.csv` | `figures/results/socialgood_range_plot.py` | generated |

## 已生成文件

- `figures/results/socialgood_group_mean_bars.png`
- `figures/results/socialgood_group_mean_bars.svg`
- `figures/results/socialgood_mean_best_gap.png`
- `figures/results/socialgood_mean_best_gap.svg`
- `figures/results/socialgood_range_plot.png`
- `figures/results/socialgood_range_plot.svg`

## 出图优先级建议

1. 必做：Figure 1（方法总框架图）
2. 必做：Figure 3（关键配置组均值误差对比图）
3. 建议做：Figure 4（均值-最优对照图）
4. 若版面允许：Figure 2（文本证据构建图）
5. 可选：Figure 5（误差区间图）

## 当前可直接出图的数据基础

当前已经具备可直接驱动实验图的数据文件：
- `tables/socialgood_real_results.csv`
- `tables/socialgood_grouped_results.csv`

其中 `tables/socialgood_grouped_results.csv` 已支持以下真实统计：
- scorehack：count=4，CRPS_mean=0.147703898580，MSE_mean=0.962174203824，MAE_mean=0.544939111370
- notext：count=5，CRPS_mean=0.172124734678，MSE_mean=1.173905884480，MAE_mean=0.639917091991
- plain：count=6，CRPS_mean=0.180450866097，MSE_mean=1.378459949903，MAE_mean=0.670870706401
- backbone_coarse：count=1，CRPS_mean=0.274744887101，MSE_mean=1.507823173390，MAE_mean=0.956032217011
- conservative：count=2，CRPS_mean=0.311060014524，MSE_mean=3.346510331760，MAE_mean=1.217857122421

## 图文对应关系

- Figure 1 对应第 2 章方法主线
- Figure 2 对应第 2.3 节结构化文本证据
- Figure 3 对应第 4.2 节总体比较
- Figure 4 对应第 4.3 与第 4.4 节主贡献/次贡献分析
- Figure 5 对应第 4.2 与第 4.5 节的波动性解释

说明：当前阶段实验图已基于真实 CSV 生成 PNG/SVG。方法图仍更适合使用 diagram 类工具按论文术语重绘。