# 弹性网络 (Elastic Net)

- UID: `CS-0094`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `207`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0207-弹性网络_(Elastic_Net)`

## R01

弹性网络（Elastic Net）是把 `L1`（稀疏化）和 `L2`（稳定化）同时加入线性回归目标的正则化方法。它解决了两个常见痛点：
- 纯 Lasso（仅 `L1`）在强相关特征下可能“随机选一个、抛弃其他”；
- 纯 Ridge（仅 `L2`）虽然稳定，但不会产生稀疏模型。

本条目给出一个可运行、可审计的最小 MVP：
- 用 NumPy 手写循环坐标下降（cyclic coordinate descent）训练 Elastic Net；
- 用 sklearn 的 `ElasticNet` 做数值对照验证；
- 输出收敛轨迹、泛化指标、系数恢复情况与内置质量检查。

## R02

问题定义：
- 输入：特征矩阵 `X in R^(n*p)`、目标向量 `y in R^n`。
- 输出：系数 `w in R^p` 和截距 `b`。
- 目标：在保持预测误差低的同时控制模型复杂度，并在相关特征组上保持更稳定的选择行为。

该 MVP 使用合成的“组内强相关”回归数据，验证 Elastic Net 的基本行为，并与 sklearn 实现对齐。

## R03

优化目标为：

`min_{w,b}  (1/(2n)) * ||y - Xw - b||_2^2 + alpha * l1_ratio * ||w||_1 + (alpha*(1-l1_ratio)/2) * ||w||_2^2`

其中：
- `alpha > 0`：总正则强度；
- `l1_ratio in [0,1]`：`L1` 与 `L2` 的混合比例；
- `l1_ratio=1` 退化为 Lasso；`l1_ratio=0` 退化为 Ridge 风格正则。

对第 `j` 个坐标的更新使用软阈值：
- `rho_j = (1/n) * x_j^T (r + x_j * w_j)`
- `w_j <- S(rho_j, alpha*l1_ratio) / (z_j + alpha*(1-l1_ratio))`
- `z_j = (1/n) * ||x_j||_2^2`
- `S(t, lam)=sign(t)*max(|t|-lam,0)`

## R04

`demo.py` 的高层流程：

1. 生成带相关结构的合成回归数据集并划分训练/测试集。
2. 调用手写 `elastic_net_coordinate_descent` 训练模型。
3. 在同一数据上训练 sklearn `ElasticNet` 作为参考实现。
4. 对比两者 `train/test MSE` 与 `R2`。
5. 输出训练轨迹快照（目标函数、最大坐标更新量、非零系数数目）。
6. 输出系数绝对值 Top10，观察信号恢复情况。
7. 执行内置质量检查：单调性、与 sklearn 的数值一致性、信号特征覆盖率。

## R05

核心数据结构：
- `EpochRecord`：每轮迭代的 `epoch/objective/max_delta/nnz`。
- `ElasticNetResult`：训练结果容器（系数、截距、是否收敛、迭代轮数、历史轨迹、最终目标值）。
- `SyntheticData`：合成数据包（`X/y/true_coef/feature_names`）。

这些结构确保输出可审计，不是只给最终系数的黑盒脚本。

## R06

正确性关键点：
- 每次坐标更新都在固定其他变量下精确求解一维子问题；
- 通过残差增量维护（`residual += old`, `residual -= new`）避免重复计算 `Xw`，同时保证数值一致性；
- `L1` 的软阈值负责稀疏化，`L2` 负责抑制系数爆炸与共线不稳定；
- 与 sklearn 同参对照可以直接检查实现是否偏离标准算法。

## R07

复杂度分析（`n` 样本数，`p` 特征数，`T` 迭代轮数）：
- 单次坐标更新成本约 `O(n)`（一次内积 + 一次残差更新）。
- 单轮扫描全部特征成本 `O(n*p)`。
- 总时间复杂度 `O(T*n*p)`。
- 额外空间复杂度 `O(n + p)`（残差、参数向量与统计量）。

## R08

边界与异常处理：
- `X` 必须二维、`y` 必须一维且样本数一致；
- 输入必须全为有限值，拒绝 `NaN/Inf`；
- `alpha<=0`、`tol<=0`、`max_epochs<=0`、`l1_ratio` 越界会直接报错；
- 若特征列退化导致分母 `z_j + alpha*(1-l1_ratio)` 近零，会抛异常防止 silent failure。

## R09

MVP 取舍：
- 保留：坐标下降核心数学、收敛轨迹、与参考实现对照。
- 省略：并行坐标更新、稀疏矩阵高性能实现、正则路径（多 alpha 网格）与交叉验证。
- 原则：优先小而完整、可审计、可复现实验，而不是工程级大框架。

## R10

`demo.py` 主要函数职责：
- `validate_inputs`：输入合法性检查。
- `soft_threshold`：实现 `L1` 近端算子。
- `elastic_net_objective`：计算完整目标函数值。
- `elastic_net_coordinate_descent`：手写循环坐标下降核心。
- `make_synthetic_regression`：构造相关特征组数据。
- `support_metrics`：计算支持集恢复精度/召回。
- `history_snapshot`：提取训练历史关键帧。
- `run_quality_checks`：执行断言式质量门槛。
- `main`：串联训练、对比、输出与验收。

## R11

运行方式：

```bash
cd Algorithms/计算机-机器学习／深度学习-0207-弹性网络_(Elastic_Net)
uv run python demo.py
```

脚本无需交互输入，会自动生成数据、训练并输出结果。

## R12

输出字段说明：
- `custom_converged`：手写坐标下降是否在 `max_epochs` 内收敛。
- `epochs`：实际迭代轮数。
- `final_objective`：最终目标函数值。
- `[Metrics Comparison]`：手写实现与 sklearn 的 `train_mse/test_mse/test_r2` 对照。
- `[Support Recovery]`：按阈值统计系数支持集的 `precision/recall`。
- `[Training History Snapshot]`：若干代表性迭代点的 `objective/max_delta/nnz`。
- `[Top 10 |custom_coef|]`：系数绝对值最大的特征及真值对照。
- `All checks passed.`：内置验收断言全部通过。

## R13

内置最小测试与质量门槛：
1. 最终目标函数必须是有限值；
2. 训练历史长度必须大于 1；
3. 目标函数需单调非增（允许极小数值容差）；
4. 手写实现与 sklearn 在测试集 MSE 的差异必须很小；
5. 手写实现系数与 sklearn 系数向量距离必须足够小；
6. 系数绝对值 Top8 中需覆盖至少 5 个真实信号特征。

## R14

关键参数与调参建议：
- `alpha`：总正则强度，越大越稀疏但偏差可能增加。
- `l1_ratio`：`L1/L2` 混合比例，越大越接近 Lasso，越小越接近 Ridge。
- `tol`：坐标收敛阈值，越小越严格但训练更慢。
- `max_epochs`：最大轮数，防止未收敛时无限迭代。

实践建议：先固定 `tol/max_epochs`，再调 `alpha` 与 `l1_ratio` 找预测误差和稀疏度平衡点。

## R15

与相关方法对比：
- 对比 Lasso：Elastic Net 对强相关特征更稳定，不易“只留一个”。
- 对比 Ridge：Elastic Net 能产生稀疏解，便于解释和特征筛选。
- 对比直接黑盒调用：本实现保留训练轨迹与公式级更新，更适合教学和审计。

## R16

典型应用场景：
- 高维线性回归且特征存在明显共线性；
- 需要“可解释 + 稳定”折中的建模任务；
- 作为更复杂模型前的强基线（baseline）；
- 特征工程阶段的初步筛选与鲁棒回归建模。

## R17

可扩展方向：
- 增加 K 折交叉验证自动选 `alpha/l1_ratio`；
- 增加正则路径绘图（系数随 `alpha` 变化）；
- 支持稀疏矩阵输入与更大规模数据；
- 引入 warm start 和屏蔽规则（screening）加速迭代；
- 增加真实数据集基准评测与可视化报告。

## R18

`demo.py` 的源码级算法流程（8 步）如下：

1. `make_synthetic_regression` 生成组内相关特征矩阵 `X`，并注入已知稀疏真系数，得到 `y`。  
2. `main` 将数据划分为训练集与测试集，固定 `alpha` 与 `l1_ratio`。  
3. `elastic_net_coordinate_descent` 先调用 `validate_inputs` 做形状、数值和参数检查。  
4. 对 `X/y` 做中心化（用于截距解耦），初始化 `coef=0` 与 `residual=y_centered`。  
5. 每个 epoch 逐坐标执行：先把旧系数贡献加回残差，计算 `rho_j`，再做软阈值更新 `w_j`。  
6. 用 `residual -= x_j * w_j_new` 增量维护残差，记录 `max_delta`，并计算完整 Elastic Net 目标值。  
7. 若 `max_delta < tol` 则判定收敛，返回 `ElasticNetResult`（含历史轨迹与最终目标值）。  
8. `main` 再训练 sklearn 版本进行 MSE/系数对照，并执行 `run_quality_checks` 输出验收结论。  
