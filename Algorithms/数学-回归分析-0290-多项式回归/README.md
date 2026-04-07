# 多项式回归

- UID: `MATH-0290`
- 学科: `数学`
- 分类: `回归分析`
- 源序号: `290`
- 目标目录: `Algorithms/数学-回归分析-0290-多项式回归`

## R01

多项式回归（Polynomial Regression）是把原始特征通过幂次展开后，再做线性回归的经典方法。  
在一维输入场景下，模型形式为：

`y ≈ beta0 + beta1*x + beta2*x^2 + ... + betad*x^d`

虽然参数是线性的，但对 `x` 呈现非线性拟合能力，能表达弯曲关系。  
本目录的 MVP 使用 `numpy` 手写：
- 多项式设计矩阵构造；
- 带轻量 `L2` 稳定项的闭式最小二乘求解；
- 训练/验证/测试切分与多项式阶数选择；
- RMSE 与 `R^2` 评估。

## R02

问题定义（本实现）：
- 输入：
  - 一维样本 `x in R^n`，目标 `y in R^n`；
  - 候选阶数集合（默认 `1..8`）；
  - 正则系数 `l2_reg`（默认 `1e-8`）；
  - 固定随机种子用于数据切分与复现。
- 输出：
  - 选择得到的最佳阶数 `d*`（以验证集 RMSE 最小为准）；
  - 最终模型系数 `beta`（按 `x^0` 到 `x^d` 排列）；
  - 训练/测试集上的 RMSE、`R^2`；
  - 测试样本预测明细（部分展示）。

## R03

核心数学关系：

1. 设计矩阵（Vandermonde 形式）  
   `X = [1, x, x^2, ..., x^d]`，其中 `X in R^(n x (d+1))`。
2. 正则化最小二乘目标  
   `min_beta ||X beta - y||_2^2 + lambda ||beta_(1:)||_2^2`  
   其中偏置项 `beta0` 不做正则化。
3. 闭式解（法方程）  
   `(X^T X + lambda*R) beta = X^T y`，`R=diag(0,1,1,...,1)`。
4. 预测  
   `y_hat = X beta`。
5. 评价指标  
   - `RMSE = sqrt(mean((y - y_hat)^2))`
   - `R^2 = 1 - SS_res / SS_tot`

## R04

算法流程（高层）：
1. 生成可复现的一维合成数据（含噪声），并构造训练/验证/测试划分。  
2. 对每个候选阶数 `d`：
   - 构造设计矩阵 `X_d`；
   - 用闭式解拟合系数；
   - 计算训练与验证 RMSE。  
3. 选择验证 RMSE 最小的阶数 `d*`（并列时选更小阶数）。  
4. 使用 `train+valid` 重新拟合 `d*` 阶模型。  
5. 在测试集上计算 RMSE、`R^2`，并打印样例预测。

## R05

核心数据结构：
- `PolynomialRegressionModel`（dataclass）：
  - `degree: int`，多项式阶数；
  - `coefficients: np.ndarray`，长度为 `degree+1` 的参数向量。
- `records: list[(degree, train_rmse, valid_rmse)]`：阶数搜索日志。
- 数据向量：`x_train/y_train/x_valid/y_valid/x_test/y_test`，均为 `1D numpy` 数组。

## R06

正确性要点：
- 目标函数是凸二次（给定阶数后），法方程解是全局最优解。  
- 加入轻量 `L2` 后，`X^T X + lambda R` 数值稳定性更好，降低病态矩阵风险。  
- 使用验证集选择阶数，避免只看训练误差导致高阶过拟合。  
- 输出同时给出训练与测试指标，便于观察泛化能力与是否欠/过拟合。

## R07

复杂度分析（`n` 样本、最大阶数 `d_max`）：
- 构造单个阶数的设计矩阵：`O(n*d)`。  
- 计算法方程并求解：
  - `X^T X`：`O(n*d^2)`；
  - 线性方程求解：`O(d^3)`。  
- 阶数搜索总成本（`K` 个候选阶数）：
  - 时间复杂度约 `O(K*(n*d_max^2 + d_max^3))`；
  - 空间复杂度约 `O(n*d_max + d_max^2)`。

## R08

边界与异常处理：
- `x/y` 不是一维或含 `nan/inf`：抛 `ValueError`。  
- `x` 与 `y` 长度不一致：抛 `ValueError`。  
- `degree < 1`、`l2_reg < 0`：抛 `ValueError`。  
- 数据量过少导致无法合理切分训练/验证/测试：抛 `ValueError`。  
- 若法方程直接求解失败，回退 `pinv` 伪逆求解，避免脚本崩溃。

## R09

MVP 取舍：
- 只实现一维输入场景，突出多项式回归核心机制。  
- 采用闭式解，不引入迭代优化器。  
- 数据使用可控合成样本，保证示例可复现。  
- 不引入 CLI 参数系统，保持单文件直接运行。  
- 不调用 `sklearn` 黑盒拟合器，便于源码级审计。

## R10

`demo.py` 主要函数职责：
- `polynomial_design_matrix`：构造 `[1, x, ..., x^d]` 设计矩阵。  
- `fit_polynomial_regression`：求解闭式回归系数（含 `L2` 稳定项）。  
- `predict_polynomial`：根据模型与输入输出预测值。  
- `rmse` / `r2_score`：计算常用回归指标。  
- `split_train_valid_test`：确定性随机切分数据集。  
- `select_best_degree`：遍历候选阶数并按验证 RMSE 选最优。  
- `print_degree_search` / `print_prediction_samples`：输出可读日志。  
- `main`：组织端到端实验与汇总结果。

## R11

运行方式：

```bash
cd Algorithms/数学-回归分析-0290-多项式回归
uv run python demo.py
```

脚本无需任何交互输入。

## R12

输出字段说明：
- `Dataset split`：训练/验证/测试样本数。  
- `degree search table`：
  - `degree`：候选阶数；
  - `train_rmse`：训练集 RMSE；
  - `valid_rmse`：验证集 RMSE。  
- `Selected degree`：验证集上最优阶数。  
- `Final model coefficients`：最终模型参数（`x^0` 到 `x^d`）。  
- `Metrics`：`train+valid` 与 `test` 的 RMSE、`R^2`。  
- `Sample predictions`：测试集中部分点的真实值、预测值、绝对误差。  
- `Summary`：选定阶数与阈值判定结果。

## R13

建议最小测试集（当前脚本已包含）：
- 一维三次多项式生成数据 + 高斯噪声（用于验证可恢复非线性关系）。  
- 候选阶数 `1..8`（用于验证模型选择逻辑）。

建议补充测试：
- `noise_std=0`（低噪声下应接近真实参数）；  
- 增大噪声（验证过拟合趋势与阶数选择变化）；  
- 极小样本场景（触发切分与稳定性保护分支）。

## R14

关键参数：
- `candidate_degrees`：候选阶数（默认 `1..8`）。  
- `l2_reg`：法方程稳定项（默认 `1e-8`）。  
- `train_ratio/valid_ratio`：数据切分比例（默认 `0.6/0.2`）。  
- `seed`：随机种子，保证可复现实验。  
- `noise_std`：合成数据噪声强度。

调参建议：
- 高噪声数据可适当减小最大阶数或增大 `l2_reg`；  
- 若欠拟合明显，可提升候选最大阶数；  
- 观察验证误差曲线而非只看训练误差。

## R15

方法对比：
- 对比线性回归（仅 `d=1`）：
  - 线性回归只能拟合直线；
  - 多项式回归可拟合弯曲关系。  
- 对比样条回归：
  - 多项式回归实现简单；
  - 样条在高阶或局部复杂关系上更稳。  
- 对比树模型（如随机森林）：
  - 树模型非参数、表达力强；
  - 多项式回归参数可解释性更好，训练成本更低。

## R16

典型应用场景：
- 物理/工程中低维非线性标定曲线拟合。  
- 趋势建模：销量-价格、温度-能耗等二次或三次关系。  
- 作为更复杂非线性建模前的可解释基线模型。  
- 教学场景中演示偏差-方差权衡与模型选择。

## R17

可扩展方向：
- 扩展到多维输入（交叉项与高阶项）。  
- 使用 `k-fold` 交叉验证替代单验证集。  
- 增加 AIC/BIC 等模型复杂度惩罚准则。  
- 引入稳健损失（Huber）应对异常点。  
- 增加可视化（拟合曲线、误差曲线、残差图）。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 调用 `generate_synthetic_data` 构造一维非线性回归数据（真实三次多项式 + 噪声）。  
2. `split_train_valid_test` 用固定种子打乱并切分为 train/valid/test 三部分。  
3. `select_best_degree` 遍历候选阶数；每个阶数都执行 `fit_polynomial_regression`。  
4. `fit_polynomial_regression` 内部先用 `polynomial_design_matrix` 生成 `[1, x, ..., x^d]`，再组装法方程。  
5. 通过 `solve`（失败则 `pinv` 回退）求系数，形成 `PolynomialRegressionModel`。  
6. `select_best_degree` 对每个阶数计算训练/验证 RMSE，并选择验证 RMSE 最小的阶数。  
7. `main` 使用选出的阶数在 `train+valid` 上重拟合最终模型，并在测试集上用 `predict_polynomial` 得到预测。  
8. 计算并打印 RMSE、`R^2`、系数和样例预测，最后给出 `Summary` 与通过标记。
