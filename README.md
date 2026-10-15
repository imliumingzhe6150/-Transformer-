# 建筑用电负荷预测：计算实训第一阶段

当前问题：用过去 168 小时的用电记录和日期特征，预测未来 24 小时的用电量。
先验证预测流程；节能、降低电费、削峰的具体调度目标等待导师确认。

已完成首轮实验，见[2026-10-15 进度记录](docs/progress_2026-10-15.md)。单栋建筑测试期 MAE 从 24.374 降到 23.757 kWh，RMSE 略差；这些是预测指标，不是节能指标。

## 第一阶段做什么

1. 获取真实建筑的小时级电表数据，检查数据完整性。
2. 实现“上周同一小时”预测，作为最简单的参照。
3. 训练一个小型 Transformer，比较同一批日期上的预测误差。
4. 保存预测曲线、实验配置和误差结果，形成第一次进度汇报。

这不是节能效果验证。当前没有调度器，没有设备控制，也没有节电率结果。

## 数据来源与边界

- 来源：[Building Data Genome 2](https://github.com/buds-lab/building-data-genome-project-2)。
- 数据论文：Miller et al. (2020), [Scientific Data, 7, 368](https://doi.org/10.1038/s41597-020-00712-x)。
- 使用 `data/meters/raw/electricity.csv`，保留原始文件；电表文件 SHA-256 固定在 `prepare_data.py` 中。
- 单位为每小时电量 kWh，不是瞬时功率 kW。时间戳沿用原数据的当地时间标签，不转成北京时间。
- 单位及清洗说明：[官方文档](https://github.com/buds-lab/building-data-genome-project-2/wiki/Meters-data-features)。
- 数据授权和再分发以[上游 LICENSE](https://github.com/buds-lab/building-data-genome-project-2/blob/master/LICENSE)为准；本项目不把原始数据提交到 Git。

建筑只根据 **2016 年训练数据**选取：教育或办公建筑，缺失率不超过 0.5%，非正值比例不超过 1%，均值和标准差大于零；再按缺失率、非正值比例、建筑 ID 排序取第一栋。此规则用于获得一个易于入门的样例，不代表所有建筑。

负值和非有限值记为缺失，不插值、不根据测试误差删除异常值，零值保留并统计。只保留历史输入和未来目标全部有效的窗口。被排除的窗口数写入实验配置，评估结论仅针对保留的窗口。

## 实验设置

| 项目 | 设置 |
|---|---|
| 训练目标日期 | 2016 年 |
| 验证目标日期 | 2017 年 1—6 月 |
| 测试目标日期 | 2017 年 7—12 月 |
| 历史长度 | 168 小时 |
| 预测长度 | 24 小时，一次输出 |
| 训练窗口 | 默认每 6 小时取一个窗口 |
| 验证与测试 | 每天 00:00 预测当天 24 小时，目标不重叠 |
| 特征 | 历史用电量、小时和星期的 sin/cos 编码 |
| 基线 | 上周同一天各小时的用电量 |
| Transformer | 2 层 Encoder、宽度 32、4 个注意力头、回归输出头 |
| 训练 | AdamW，学习率 0.0003，最多 15 轮，验证 MAE 连续 5 轮未改善则停止 |
| 评估 | MAE、RMSE（kWh）；WAPE（%）；分预测步长的误差 |

标准化只用训练期数据。每个窗口的全部预测目标必须位于同一数据划分内；验证和测试允许使用预测时刻之前已经观测到的历史，包括前一数据划分的数据。预测日期逐日推进，但不会用验证或测试目标更新模型权重。Encoder 只看到历史，因此无需语言模型那样的因果 mask。推理时将负预测截为零。

模型按验证 MAE 选择 checkpoint，最后统一评估测试集。不能依据测试误差反复挑模型，再把同一测试集当作未见数据。首轮仅一个建筑、一个随机种子、一个季节划分，不能用于泛化或统计显著性结论。

## 运行

推荐 Python 3.12 和 uv。从本目录运行：

```bash
uv sync --locked
uv run prepare_data.py --download
uv run run_experiment.py --output-dir outputs/first_run
uv run plot_results.py --run-dir outputs/first_run
```

默认使用 CPU，限制为 4 个线程；也可明确传入 `--device mps` 或 `--device cuda`。不同设备的软件算子和数值误差可能导致结果差异。

已有 `.venv` 时可离线运行，不需要再次解析依赖：

```bash
.venv/bin/python prepare_data.py
.venv/bin/python run_experiment.py --output-dir outputs/another_run
.venv/bin/python plot_results.py --run-dir outputs/another_run
```

每次训练使用新的输出目录，防止覆盖已有结果。数据准备脚本会重新生成处理后的数据文件。

## 文件说明

| 文件 | 作用 |
|---|---|
| `prepare_data.py` | 下载、核验、选择建筑和划分数据 |
| `forecasting.py` | 时间窗口、特征、Transformer 和指标 |
| `run_experiment.py` | 训练、验证选模和最终测试 |
| `plot_results.py` | 首批测试日期预测曲线，以及整个测试期每日误差 |
| `data/processed/data_summary.json` | 数据来源、建筑信息和各划分缺失情况 |
| `data/processed/training_building_quality.csv` | 仅根据训练期计算的建筑选择依据 |
| `outputs/<run>/config.json` | 软件版本、超参数、标准化参数、窗口排除数 |
| `outputs/<run>/metrics.csv` | 基线与 Transformer 的验证、测试指标 |
| `outputs/<run>/predictions.csv` | 每次预测的发出时间、目标时间、真实值和预测值 |
| `outputs/<run>/training_history.csv` | 各轮训练损失、验证误差和用时 |
| `outputs/<run>/best_model.pt` | 验证 MAE 最低的模型权重 |

## 怎样理解结果

MAE 是每个小时平均预测差多少度电，越小越好；RMSE 更重视较大误差；WAPE 是总绝对误差相对于总实际电量的比例。

Transformer 暂时不如简单基线，也应如实记录。后续可先检查数据、工作日/周末、预测步长、季节变化和验证集表现，再决定是否加入天气或调整模型。

待导师确认后，再决定连接储能充放电调度，还是带温度和舒适度约束的空调控制。真实建筑历史数据上的预测实验，不能直接证明真实设备的节能收益。
