# 岭回归 (Ridge Regression)

- UID: `CS-0093`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `205`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0205-岭回归_(Ridge_Regression)`

## R01

岭回归（Ridge Regression）是在线性回归损失上加入 `L2` 正则项的模型，用于缓解多重共线性、抑制参数过大、提升泛化稳定性。

核心直觉：
- OLS 只最小化残差，参数可能因特征相关性而剧烈波动；
- Ridge 在目标函数里增加参数范数惩罚，使解更平滑；
- 代价是引入偏差（bias），换取更低方差（variance）。

## R02

本条目要解决的问题：
- 输入：`X in R^(n x p)`、`y in R^n`、正则系数 `alpha > 0`；
- 输出：回归系数 `w` 与截距 `b`，并评估测试集误差；
- 目标：在保持可解释线性模型的同时，让参数估计在相关特征下更稳定。

本目录 MVP 同时提供：
- 自实现闭式解 Ridge（主实现）；
- `sklearn` Ridge（对照实现）；
- OLS（无正则基线）。

## R03

优化目标（不正则化截距）：

`min_{w,b} ||y - Xw - b||_2^2 + alpha * ||w||_2^2`

对中心化后的变量 `Xc = X - mean(X)`、`yc = y - mean(y)`，可写为：

`min_w ||yc - Xc w||_2^2 + alpha * ||w||_2^2`

一阶最优条件得到：

`(Xc^T Xc + alpha I) w = Xc^T yc`

随后恢复截距：

`b = mean(y) - mean(X)^T w`

## R04

算法路线（本实现）：
1. 对训练集特征做标准化（`StandardScaler`），降低数值尺度差异。  
2. 对标准化后的训练矩阵做中心化推导（代码中显式减均值）。  
3. 构造 Gram 矩阵 `G = Xc^T Xc` 与右端向量 `r = Xc^T yc`。  
4. 解线性系统 `(G + alpha I)w = r` 得到 `w`。  
5. 计算截距 `b`。  
6. 在测试集上计算 `MSE/RMSE/R2`，并与 OLS、`sklearn` Ridge 对照。

## R05

`demo.py` 关键数据结构：
- `RidgeClosedFormModel`（dataclass）
  - `alpha`：正则系数；
  - `coef_`：系数向量；
  - `intercept_`：截距；
  - `fit/predict`：训练与预测接口。
- `numpy.ndarray`：训练与预测主数据结构。
- `pandas.DataFrame`：指标表、系数表、预测样例表。

## R06

正确性与稳定性要点：
- `alpha > 0` 时，`X^T X + alpha I` 相比 `X^T X` 更接近正定，求解更稳定；
- 截距单独恢复，避免把偏置项也正则化；
- 用与 `sklearn` Ridge 的系数/截距差做数值对照，验证自实现正确性；
- 打印条件数 `cond(X^T X)` 与 `cond(X^T X + alpha I)`，直接观察稳定性提升。

## R07

复杂度分析（`n` 样本，`p` 特征）：
- 构造 `X^T X`：`O(n p^2)`；
- 构造 `X^T y`：`O(n p)`；
- 求解 `p x p` 线性系统：`O(p^3)`；
- 总体：`O(n p^2 + p^3)`；
- 空间复杂度：`O(n p + p^2)`。

在 `p` 不是特别大的表格数据中，闭式解速度通常足够快。

## R08

边界与异常处理：
- `alpha <= 0`：抛 `ValueError`；
- `X` 不是二维或 `y` 不是一维：抛 `ValueError`；
- `X` 与 `y` 样本数不一致：抛 `ValueError`；
- 未 `fit` 直接 `predict`：抛 `RuntimeError`。

## R09

MVP 取舍说明：
- 选择“闭式解 + 对照验证”，而非大而全训练框架；
- 不引入 CLI 参数解析，保证 `uv run python demo.py` 一键运行；
- 使用可复现合成数据，避免外部数据集下载依赖；
- 重点展示 Ridge 的核心机制与数值行为，不扩展到复杂特征工程流水线。

## R10

`demo.py` 函数职责：
- `make_correlated_regression_data`：构造具有共线性的回归数据；
- `RidgeClosedFormModel.fit`：执行闭式解训练；
- `RidgeClosedFormModel.predict`：输出预测；
- `evaluate_regression`：计算 `MSE/RMSE/R2`；
- `condition_number`：计算矩阵条件数；
- `main`：组织训练、对照、打印与结果汇总。

## R11

运行方式（无交互输入）：

```bash
cd Algorithms/计算机-机器学习／深度学习-0205-岭回归_(Ridge_Regression)
uv run python demo.py
```

## R12

输出字段说明：
- `n_train / n_test / n_features`：数据规模；
- `alpha`：Ridge 正则强度；
- `cond(X^T X)` 与 `cond(X^T X + alpha I)`：正则前后数值条件；
- `Test Metrics`：三种模型在测试集上的 `mse/rmse/r2`；
- `L2(coef_custom - coef_sklearn)`：自实现与 `sklearn` 系数差异；
- `Coefficient Snapshot`：真实系数、OLS、自实现 Ridge、`sklearn` Ridge 的对照；
- `Prediction Preview`：测试集样本预测与误差预览。

## R13

建议验证项（本脚本已覆盖）：
1. 可复现性：固定随机种子后多次运行输出一致。  
2. 数值一致性：自实现 Ridge 与 `sklearn` Ridge 参数差应很小。  
3. 稳定性：`cond(X^T X + alpha I)` 应明显低于 `cond(X^T X)`。  
4. 效果性：在共线特征下，Ridge 相比 OLS 通常更平稳（系数幅值更收缩）。

## R14

关键超参数：
- `alpha`：正则强度，越大收缩越强；
- `noise_std`：合成数据噪声强度；
- `test_size/random_state`：训练测试划分策略；
- `n_samples/n_features/latent_dim`：数据规模与共线性强度。

调参建议：
- 先在对数尺度尝试 `alpha in {1e-3, 1e-2, ..., 1e2}`；
- 若系数过于震荡，提高 `alpha`；
- 若欠拟合明显，减小 `alpha`。

## R15

方法对比：
- OLS：无正则，偏差小但方差可能大；
- Ridge：`L2` 连续收缩，不会产生稀疏解，但稳定性好；
- Lasso：`L1` 可做特征选择（稀疏），但在强相关特征下选择不稳定；
- Elastic Net：折中 `L1+L2`，常用于高维相关特征场景。

## R16

典型应用：
- 金融风控中的线性评分卡基线建模；
- 广告/推荐中的可解释回归基线；
- 工业过程控制中的多传感器回归；
- 科研中高相关特征的小中规模回归问题。

## R17

可扩展方向：
- 加入 `KFold` 交叉验证选择 `alpha`（如 `RidgeCV`）；
- 支持样本权重、鲁棒损失或分位数回归；
- 增加 Torch 版本梯度下降实现，与闭式解做一致性验证；
- 加入真实数据集（如 Boston 替代数据）并输出实验报告。

## R18

`demo.py` 与 `scikit-learn` 的源码级流程（8 步）：
1. `make_correlated_regression_data` 先生成潜变量，再线性混合成高相关特征，最后按线性模型加噪声得到 `y`。  
2. `main` 使用 `train_test_split` 和 `StandardScaler` 得到训练/测试矩阵，降低尺度差异对求解器的影响。  
3. 自实现 `RidgeClosedFormModel.fit` 对 `X,y` 做中心化，构造 `G=Xc^T Xc` 与 `r=Xc^T yc`。  
4. 调用 `np.linalg.solve(G + alpha I, r)` 得到 `w`，再按 `b = mean(y) - mean(X)^T w` 恢复截距。  
5. `main` 同时调用 `sklearn.linear_model.Ridge(..., solver="cholesky")` 训练对照模型，并计算与自实现的参数差。  
6. 在本环境的 `sklearn/linear_model/_ridge.py` 中，`_BaseRidge.fit` 先做 `_preprocess_data`，再路由到 `_ridge_regression(...)`。  
7. 当 solver 为 `cholesky` 且 `n_features <= n_samples` 时，`_ridge_regression` 调用 `_solve_cholesky`，其核心是解 `linalg.solve(X^T X + alpha I, X^T y)`。  
8. `main` 汇总 `MSE/RMSE/R2`、条件数与样例预测，形成“数据生成 -> 训练 -> 对照 -> 评估”的完整闭环。
