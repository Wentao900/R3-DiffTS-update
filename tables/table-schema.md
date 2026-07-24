# Table Schema

| Table | Purpose | Rows | Metrics | Data source | Replacement owner |
|---|---|---|---|---|---|
| Table 1 | SocialGood 关键配置比较 | plain / notext / scorehack / backbone_coarse / conservative | CRPS, MSE, MAE, point estimator | `save/forecasting_SocialGood_*/metrics.json` 与 `run_summary.json` 聚合结果 | Claude |
| Table 2 | 配置开关与模块映射 | 各 SocialGood 配置 | 文本、结构化残差扩散、辅助头、校准器开关 | 配置 YAML | Claude |
| Table 3 | 全域 benchmark 运行覆盖说明 | 8 个领域 | seq_len, pred_len, freq, config | `scripts/run_all_datasets_mainline.sh` | Claude |

说明：若某个表尚未完成多种子聚合，只能展示已存在的真实运行结果或写成结构说明，不得虚构均值±方差。
