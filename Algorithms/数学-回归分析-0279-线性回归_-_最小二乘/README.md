# 线性回归 - 最小二乘

- UID: `MATH-0279`
- 学科: `数学`
- 分类: `回归分析`
- 源序号: `279`
- 目标目录: `Algorithms/数学-回归分析-0279-线性回归_-_最小二乘`

## R01

本条目实现“线性回归 - 最小二乘”的最小可运行 MVP：
- 在合成数据上用法方程（Normal Equation）估计线性模型参数；
- 输出参数、误差与拟合优度，验证结果可解释；
- 代码保持短小、可复现、可直接运行。

## R02

问题定义：
- 输入：
  - 特征矩阵 `X in R^(n x d)`；
  - 目标向量 `y in R^n`；
  - 线性模型 `y_hat = beta0 + beta1*x1 + ... + betad*xd`。
- 目标：
  - 求参数向量 `beta in R^(d+1)`，最小化平方误差和
    `min_beta ||X_tilde beta - y||_2^2`，其中 `X_tilde=[1, X]`。
- 输出：
  - 参数估计 `beta`；
  - 预测值 `y_hat`；
  - 指标 `MSE`、`R^2`、残差范数。

## R03

数学基础：

1. 最小二乘目标函数
- `J(beta) = (X_tilde beta - y)^T (X_tilde beta - y)`。

2. 一阶最优条件（法方程）
- 对 `beta` 求导并令其为 0：
  `X_tilde^T X_tilde beta = X_tilde^T y`。

3. 闭式解
- 若 `X_tilde^T X_tilde` 可逆：
  `beta = (X_tilde^T X_tilde)^(-1) X_tilde^T y`。
- 若不可逆或病态：
  使用伪逆 `pinv` 求最小范数解。

4. 指标
- `MSE = mean((y - y_hat)^2)`；
- `R^2 = 1 - SSE/SST`。

## R04

算法总览（MVP）：
1. 构造可复现的合成回归数据集（固定随机种子）。
2. 校验输入维度、样本数和有限值约束。
3. 给 `X` 增广常数列，得到 `X_tilde`。
4. 计算 `X_tilde^T X_tilde` 与 `X_tilde^T y`。
5. 用 `solve` 或 `pinv` 计算最小二乘参数 `beta`。
6. 计算训练集预测值与回归指标。
7. 打印数据预览和拟合结果。

## R05

核心数据结构：
- `numpy.ndarray`
  - `X`：形状 `(n, d)`；
  - `y`：形状 `(n,)`；
  - `beta`：形状 `(d+1,)`；
  - `y_pred`：形状 `(n,)`。
- `RegressionResult`（`dataclass`）
  - `beta`、`y_pred`、`mse`、`r2`、`residual_norm_l2`。

## R06

正确性要点：
- 目标函数是凸二次函数，法方程给出全局最优解；
- 代码先尝试精确线性求解（满秩），否则回退到伪逆，保证一般情况下可得最小二乘解；
- 通过 `MSE`、`R^2` 和参数误差（相对合成数据真值）交叉验证结果合理性。

## R07

复杂度分析：
- 设样本数 `n`、特征数 `d`。
- 构造法方程：
  - `X_tilde^T X_tilde` 约 `O(n d^2)`；
  - `X_tilde^T y` 约 `O(n d)`。
- 求解线性系统：
  - 约 `O(d^3)`。
- 总体：`O(n d^2 + d^3)`；
- 空间复杂度：`O(n d + d^2)`。

## R08

边界与异常处理：
- `X` 不是二维或 `y` 不是一维时报错；
- 样本数与标签数不一致时报错；
- 样本数小于 `d+1` 时提示拟合不稳定并报错；
- `X/y` 含 `NaN/Inf` 时直接报错；
- 矩阵退化时自动使用伪逆而非崩溃退出。

## R09

MVP 取舍：
- 采用 `numpy` 手写法方程，避免把回归过程完全交给黑盒 API；
- 使用合成数据（已知真参数）而不是外部数据文件，确保目录自包含；
- 仅做训练集拟合演示，不引入复杂的数据拆分/可视化框架。

## R10

`demo.py` 函数职责：
- `generate_synthetic_data`：构造可复现线性数据与真参数；
- `validate_xy`：检查输入合法性；
- `add_intercept_column`：给特征矩阵添加截距列；
- `solve_least_squares_normal_equation`：执行最小二乘求解并返回指标；
- `print_dataset_preview`：输出前几行样本；
- `print_result_summary`：输出参数、误差与预测预览；
- `main`：串联完整流程。

## R11

运行方式：

```bash
cd Algorithms/数学-回归分析-0279-线性回归_-_最小二乘
uv run python demo.py
```

脚本无需交互输入。

## R12

输出解读：
- `true beta`：合成数据的真实参数；
- `estimated beta`：最小二乘估计参数；
- `beta error`：估计参数减真值；
- `MSE`：均方误差，越低越好；
- `R2`：拟合优度，接近 1 代表拟合好；
- `cond(X^T X)`：法方程矩阵条件数，用于观察数值稳定性。

## R13

建议最小测试集：
- 默认主流程：`n=120, d=3, noise_std=0.30`；
- 低噪声测试：`noise_std=0.05`，参数应更接近真值；
- 高噪声测试：`noise_std=1.0`，`MSE` 增大但流程应稳定；
- 异常测试：传入维度错误或 `NaN`，应触发明确异常。

## R14

可调参数：
- `n_samples`：样本数；
- `seed`：随机种子（控制可复现性）；
- `noise_std`：噪声强度；
- 可修改 `beta_true` 与特征分布，构造不同难度的数据。

## R15

方法对比：
- 最小二乘法方程
  - 优点：闭式、实现简单、在中小维度很高效；
  - 缺点：`X^T X` 可能病态，数值稳定性依赖数据条件。
- 梯度下降回归
  - 优点：适合超大规模数据；
  - 缺点：需学习率、迭代轮数等超参。
- 正则化回归（Ridge/Lasso）
  - 优点：缓解共线性或做特征选择；
  - 缺点：引入额外超参和偏差。

## R16

应用场景：
- 连续值预测基线模型；
- 解释型建模（系数可读）；
- 更复杂模型（GLM、树模型、神经网络）前的对照基线。

## R17

后续扩展方向：
- 增加训练/验证划分与泛化误差评估；
- 支持加权最小二乘（WLS）和鲁棒回归；
- 增加岭回归对病态矩阵的稳健性对比；
- 与 `scikit-learn` 的 `LinearRegression`、`Ridge` 做批量实验对齐。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `main` 调用 `generate_synthetic_data` 生成 `X, y` 与真实参数 `beta_true`。  
2. `solve_least_squares_normal_equation` 首先通过 `validate_xy` 检查维度、样本规模与有限值。  
3. `add_intercept_column` 把常数列拼接到 `X`，构造增广矩阵 `X_tilde`。  
4. 计算法方程两侧：`xtx = X_tilde^T X_tilde`、`xty = X_tilde^T y`。  
5. 若 `xtx` 满秩，使用 `np.linalg.solve(xtx, xty)`；否则使用 `np.linalg.pinv(xtx) @ xty`。  
6. 用 `y_pred = X_tilde @ beta` 得到预测，并计算残差、`MSE`、`R^2`、`||residual||_2`。  
7. `print_dataset_preview` 输出输入样本切片，验证数据分布和量纲。  
8. `print_result_summary` 对比真参数与估计参数并打印指标，形成完整最小二乘求解闭环。  
