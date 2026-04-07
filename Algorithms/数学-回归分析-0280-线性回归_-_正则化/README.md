# 线性回归 - 正则化

- UID: `MATH-0280`
- 学科: `数学`
- 分类: `回归分析`
- 源序号: `280`
- 目标目录: `Algorithms/数学-回归分析-0280-线性回归_-_正则化`

## R01

线性回归正则化的核心目标是：
在最小二乘拟合误差的基础上加入参数惩罚项，抑制过拟合与多重共线性导致的参数不稳定。

本目录 MVP 覆盖两类典型正则：
- `Ridge (L2)`：连续收缩系数，改善数值条件。
- `Lasso (L1)`：可将部分系数压到 0，得到稀疏模型。

并使用 `OLS`（无正则）作为对照基线。

## R02

要解决的问题：给定回归数据 `X in R^(n*p), y in R^n`，比较三种参数估计方式在同一数据上的表现：

- OLS：`min_w (1/(2n)) ||y - Xw||_2^2`
- Ridge：`min_w (1/(2n)) ||y - Xw||_2^2 + (alpha/2)||w||_2^2`
- Lasso：`min_w (1/(2n)) ||y - Xw||_2^2 + alpha||w||_1`

输出指标包括：
- 训练/测试 `MSE`
- 系数 `L2` 范数与相对真值误差
- 稀疏度（非零系数数）
- Lasso 的收敛与目标单调性审计

## R03

选择该问题的原因：
- 回归分析中“正则化”是最基础、最高频的稳定化手段之一；
- 多重共线性会使 `OLS` 系数方差变大，便于展示正则化收益；
- `Ridge + Lasso` 互补：一个强调稳定收缩，一个强调变量筛选；
- 可以在一个小型脚本内完整实现，不依赖黑盒训练框架。

## R04

关键数学形式：

1. Ridge 闭式解（在标准化特征与中心化标签下）：
`w_ridge = (X^T X + alpha I)^(-1) X^T y`

2. Lasso 坐标下降单坐标更新：
- `z_j = (1/n)||x_j||_2^2`
- `rho_j = (1/n) x_j^T (r + x_j w_j)`
- 软阈值 `S(t, lam)=sign(t)*max(|t|-lam, 0)`
- 更新 `w_j <- S(rho_j, alpha)/z_j`

3. 预测形式：
- 训练阶段先中心化 `y`，求得 `w` 后用 `intercept = mean(y_train)`
- `y_hat = X_std w + intercept`

## R05

整体流程：

1. 生成带多重共线性的合成回归数据。  
2. 划分训练/测试集。  
3. 用训练集统计量对特征做标准化。  
4. 训练 OLS 闭式解。  
5. 训练 Ridge 闭式解（固定 `alpha=5.0`）。  
6. 训练 Lasso 坐标下降（`alpha=0.08`，含收敛判据）。  
7. 统一评估三者的误差、系数规模、稀疏度。  
8. 打印全局检查结果（收缩、稀疏化、收敛与单调性）。

## R06

正确性依据：
- OLS 与 Ridge 的闭式解直接由一阶最优条件得到；
- Ridge 在 `X^T X` 上加 `alpha I`，提高矩阵可逆性与条件数稳定性；
- Lasso 坐标下降在凸目标上进行逐坐标精确子问题更新，目标函数应近似非增；
- 脚本中对 Lasso 额外做 `objective_monotone_check`，用于实现级别审计。

## R07

复杂度分析（`n` 样本数，`p` 特征数，`T` 为 Lasso 迭代轮数）：

- OLS（`lstsq`）：典型代价约 `O(np^2 + p^3)`。
- Ridge（闭式解）：构造与求解线性系统约 `O(np^2 + p^3)`。
- Lasso（坐标下降）：
  - 单坐标更新 `O(n)`
  - 每轮 `p` 坐标，`O(np)`
  - 总计 `O(Tnp)`
- 空间复杂度：主要为 `X`、残差和系数，`O(np + n + p)`。

## R08

边界与异常处理：
- `X` 必须是二维、`y` 必须是一维且样本数一致；
- 输入必须为有限数值（拒绝 `nan/inf`）；
- `test_ratio` 必须在 `(0,1)`；
- 标准化时若某列方差近零会报错；
- Ridge/Lasso 的 `alpha` 必须大于 0；
- Lasso 的 `tol/max_epochs` 也做了正值检查。

## R09

MVP 设计取舍：
- 仅依赖 `numpy`，避免外部黑盒优化器；
- 不实现交叉验证、正则路径或并行加速，保持脚本可读；
- 使用合成数据而非文件输入，保证可复现与无交互；
- 通过 OLS/Ridge/Lasso 三模型并列输出，优先体现“正则化效果”而非工程封装。

## R10

`demo.py` 主要函数职责：
- `validate_dataset`：输入维度和数值合法性检查。
- `train_test_split`：固定随机种子切分数据。
- `standardize_from_train`：按训练集统计量标准化。
- `fit_ols_closed_form`：无正则闭式拟合。
- `fit_ridge_closed_form`：`L2` 正则闭式拟合。
- `fit_lasso_coordinate_descent`：`L1` 正则坐标下降。
- `objective_monotone_check`：检测 Lasso 目标是否近似单调下降。
- `summarize_result`：统一输出训练/测试误差与系数统计。
- `main`：串联数据生成、训练、评估、全局检查。

## R11

运行方式：

```bash
cd Algorithms/数学-回归分析-0280-线性回归_-_正则化
uv run python demo.py
```

脚本为非交互模式，直接输出全部结果。

## R12

输出字段说明：
- `condition_number(X^T X)`：未正则时法方程矩阵条件数。
- `condition_number(X^T X + 5I)`：Ridge 正则后的条件数变化。
- `train_mse/test_mse`：训练与测试均方误差。
- `coef_l2_norm`：系数向量范数（收缩强弱指标）。
- `coef_l2_error_vs_true`：与合成真值系数的距离。
- `nnz(|coef|>1e-3)`：非零系数个数。
- `objective_monotone`：仅 Lasso 输出，表示目标函数是否近似单调下降。
- `global_checks_pass`：全局检查是否通过。

## R13

当前最小验证覆盖：
- 固定随机种子（`2026`），结果可复现；
- 在同一数据集上比较 OLS / Ridge / Lasso；
- 检查 Ridge 的收缩效果（`coef_l2_norm` 下降）；
- 检查 Lasso 的泛化改进与稀疏化（`test_mse` 与 `nnz`）；
- 检查 Lasso 收敛和目标函数单调性。

## R14

关键超参数：
- `Ridge alpha=5.0`：增大可提升稳定性但可能引入偏差；
- `Lasso alpha=0.08`：增大后更稀疏，但可能损失拟合精度；
- `tol=1e-8`：Lasso 收敛阈值；
- `max_epochs=3000`：Lasso 最大迭代轮次；
- `test_ratio=0.25`：训练/测试划分比例。

调参建议：
- 先固定数据与切分种子，只调 `alpha`；
- 观察 `test_mse`、`nnz`、`coef_l2_norm` 的三方折中；
- 若 Lasso 不收敛，可先放宽 `tol` 或增大 `max_epochs`。

## R15

方法对比：
- OLS：无偏但对共线性敏感，系数可能剧烈波动。
- Ridge：通过 `L2` 收缩提升稳定性，通常保留全部特征。
- Lasso：通过 `L1` 促稀疏，兼具拟合与特征选择能力。

与库调用对比：
- 工程上可直接使用 `scikit-learn` 的 `Ridge/Lasso`；
- 本目录聚焦算法透明性，关键更新步骤完全手写，便于审计与教学。

## R16

典型应用场景：
- 特征间高度相关的线性回归任务；
- 需要抑制过拟合的小样本高维回归；
- 需要可解释稀疏模型的特征筛选场景；
- 作为更复杂模型前的可解释基线。

## R17

可扩展方向：
- 增加 `Elastic Net (L1 + L2)`；
- 对 `alpha` 做交叉验证和正则路径搜索；
- 引入真实数据集与特征工程流程；
- 为 Lasso 增加 warm-start 与筛选规则（screening）；
- 扩展到稀疏矩阵输入和更大规模数据。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `make_collinear_regression` 构造存在强相关特征的数据，并生成带噪声标签。  
2. `train_test_split` 在固定种子下划分训练/测试集，`standardize_from_train` 用训练统计量标准化特征。  
3. `fit_ols_closed_form` 对中心化标签调用 `np.linalg.lstsq` 获得 OLS 系数作为无正则基线。  
4. `fit_ridge_closed_form` 显式构造 `(X^T X + alpha I)` 并 `np.linalg.solve`，完成 Ridge 闭式求解。  
5. `fit_lasso_coordinate_descent` 初始化 `coef=0/residual=y_centered`，预计算 `z_j=(1/n)||x_j||^2`。  
6. Lasso 每轮对每个坐标计算 `rho_j`，再经 `soft_threshold` 得到新系数，用增量方式维护残差。  
7. 每轮记录 `(epoch, objective, max_delta, nnz)`，并以 `max_delta < tol` 判断收敛；随后 `objective_monotone_check` 审计单调性。  
8. `summarize_result` 汇总三模型指标，`main` 输出系数对照与 `global_checks_pass` 完成端到端验证。
