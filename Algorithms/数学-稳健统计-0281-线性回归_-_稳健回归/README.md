# 线性回归 - 稳健回归

- UID: `MATH-0281`
- 学科: `数学`
- 分类: `稳健统计`
- 源序号: `281`
- 目标目录: `Algorithms/数学-稳健统计-0281-线性回归_-_稳健回归`

## R01

稳健回归（Robust Regression）用于在存在离群点（outliers）时，仍能稳定估计线性关系。  
普通最小二乘（OLS）最小化平方误差，离群点会被二次放大，导致参数偏移明显。  
本目录 MVP 采用 Huber 损失的 M-估计，并用 IRLS（Iteratively Reweighted Least Squares）求解。

核心目标：
- 在训练数据被污染（响应变量含异常值）时，降低参数估计偏差；
- 在干净测试集上取得更稳健的泛化误差；
- 给出可审计、非黑盒的最小可运行实现。

## R02

问题定义（本实现）：
- 输入：
  - 特征矩阵 `X in R^(n x d)`；
  - 目标向量 `y in R^n`；
  - Huber 阈值 `delta`（可自动估计）；
  - IRLS 参数：`max_iter`、`tol`、`l2_reg`。
- 输出：
  - OLS 参数向量 `theta_ols`；
  - Huber-IRLS 参数向量 `theta_huber`；
  - 训练集（污染标签）与测试集（干净标签）上的 RMSE/MAE；
  - IRLS 收敛日志（目标值、相对参数变化、最小权重）。

## R03

数学形式（带偏置的线性模型）：
- 记增广设计矩阵 `A = [1, X]`，参数 `theta = [b, w]`，预测为 `y_hat = A theta`。

OLS 目标：
- `min_theta sum_i (y_i - a_i^T theta)^2`

Huber 损失：
- `rho_delta(r) = 0.5*r^2`，当 `|r| <= delta`
- `rho_delta(r) = delta*(|r| - 0.5*delta)`，当 `|r| > delta`

稳健回归目标：
- `min_theta (1/n) * sum_i rho_delta(r_i) + (l2_reg/2)*||w||_2^2`

IRLS 等价权重：
- `w_i = 1`，当 `|r_i| <= delta`
- `w_i = delta/|r_i|`，当 `|r_i| > delta`

每轮求解加权最小二乘：
- `theta <- argmin_theta sum_i w_i * (y_i - a_i^T theta)^2 + l2_reg*||w||_2^2`

## R04

算法流程（高层）：
1. 生成可复现的线性合成数据，并划分训练/测试。  
2. 在训练标签中注入一部分大幅离群扰动。  
3. 拟合 OLS 作为基线。  
4. 用 OLS 初始化 Huber-IRLS，并自动估计 `delta`。  
5. 迭代执行“残差 -> 权重 -> 加权最小二乘”直到收敛。  
6. 在训练（污染）和测试（干净）上比较 OLS 与 Huber 的 RMSE/MAE。  
7. 输出参数误差、收敛日志和样例预测。

## R05

核心数据结构：
- `LinearRegressionModel`（dataclass）：
  - `theta: np.ndarray`，增广参数 `[bias, w1, ..., wd]`；
  - `method: str`，模型类型（`OLS` / `Huber-IRLS`）；
  - `delta: Optional[float]`，Huber 阈值。
- `history: list[(iter, obj, rel_change, min_weight)]`：IRLS 每轮日志。
- `dataset: dict[str, np.ndarray]`：包含训练/测试、干净/污染标签、真实参数、离群标记。

## R06

正确性要点：
- Huber 在小残差区是二次损失、在大残差区是线性增长，降低离群点影响。  
- IRLS 使用 `delta/|r|` 权重自动下调大残差样本贡献。  
- 每轮子问题是带 `L2` 稳定项的加权线性最小二乘，可直接线性代数求解。  
- `delta` 通过 MAD 尺度估计初始化，减少手工阈值敏感性。  
- 求解失败时回退伪逆，避免病态矩阵导致脚本中断。

## R07

复杂度分析（`n` 样本、`d` 特征、`T` 轮 IRLS）：
- 单轮构造并求解加权法方程：
  - 计算 `A^T W A`：`O(n*d^2)`；
  - 解 `(d+1)x(d+1)` 线性系统：`O(d^3)`。
- 总时间复杂度：`O(T*(n*d^2 + d^3))`。  
- 空间复杂度：`O(n*d + d^2)`。

对本 MVP（小维度线性回归）足够轻量，通常几十轮内收敛。

## R08

边界与异常处理：
- `x` 不是二维或 `y` 不是一维：抛 `ValueError`。  
- 样本数不匹配、含 `nan/inf`：抛 `ValueError`。  
- `l2_reg < 0`、`tol <= 0`、`max_iter < 1`：抛 `ValueError`。  
- 训练/测试切分比例过极端导致样本过少：抛 `ValueError`。  
- 矩阵求解失败时使用 `pinv` 回退分支。

## R09

MVP 取舍：
- 仅做线性回归，不扩展到广义线性模型。  
- 仅实现 Huber M-估计，不额外实现 Tukey/Cauchy 等损失。  
- 重点强调“离群响应污染”场景，测试集保持干净用于观察泛化恢复。  
- 不引入外部黑盒稳健回归器，算法步骤全部在源码中显式实现。

## R10

`demo.py` 主要函数职责：
- `build_synthetic_dataset`：生成线性数据并注入训练离群点。  
- `fit_ols`：基线普通最小二乘拟合。  
- `choose_huber_delta` / `mad_scale`：鲁棒尺度估计与阈值设定。  
- `huber_weights` / `huber_objective`：Huber 权重与目标计算。  
- `fit_huber_irls`：IRLS 主循环（加权重拟合 + 收敛判定）。  
- `predict`、`rmse`、`mae`：预测与评估指标。  
- `summarize_history` / `print_metric_table`：输出收敛与对比结果。  
- `main`：组织端到端实验流程。

## R11

运行方式：

```bash
cd Algorithms/数学-稳健统计-0281-线性回归_-_稳健回归
uv run python demo.py
```

脚本无需任何交互输入。

## R12

输出字段说明：
- `dataset`：训练/测试样本数、特征维度、训练离群点数量。  
- `train contamination mean abs shift`：训练标签污染幅度均值。  
- `huber delta`：自动估计得到的 Huber 阈值。  
- `IRLS iteration log`：迭代号、目标值、相对参数变化、最小样本权重。  
- `True vs estimated parameters`：真实参数与两种估计参数。  
- `Metrics`：训练(污染)/测试(干净)上的 RMSE 与 MAE。  
- `Sample predictions`：测试集部分样本预测细节。  
- `Summary`：稳健回归是否优于 OLS 的判定标记。

## R13

建议最小测试（脚本已覆盖）：
- 中等噪声线性数据 + 训练集 18% 标签离群点。  
- 对比 OLS 与 Huber 在干净测试集误差。  
- 检查 Huber 参数估计与真实参数距离是否更小。

建议补充测试：
- `outlier_fraction=0`（两者应接近）。  
- 极高离群比例（检验 `delta` 与收敛稳定性）。  
- 多随机种子批量运行（观察稳健收益分布）。

## R14

关键参数：
- `outlier_fraction`：训练标签离群比例。  
- `delta`：Huber 阈值（`None` 时自动 MAD 估计）。  
- `max_iter`、`tol`：IRLS 迭代上限与收敛阈值。  
- `l2_reg`：数值稳定项，默认很小。  
- `noise_std`：底层高斯噪声强度。

调参建议：
- 离群更重时，适当减小 `delta` 会更保守。  
- 若收敛慢，可增大 `max_iter` 或放宽 `tol`。  
- 若数值不稳定，可稍增 `l2_reg`。

## R15

方法对比：
- OLS：实现简单、效率高，但对离群点高度敏感。  
- Huber-IRLS：在轻中度污染下通常有更稳健参数与更低测试误差。  
- RANSAC：对大比例离群点强，但会引入采样随机性和更多超参。  
- Theil-Sen：更鲁棒但在高维或大样本时计算成本更高。

## R16

典型应用：
- 传感器读数偶发尖峰导致的线性标定。  
- 财务/运营数据中存在异常记录时的趋势建模。  
- 工业日志与质量控制中的鲁棒基线回归。  
- 作为复杂模型前的“可解释稳健基线”。

## R17

可扩展方向：
- 增加更多稳健损失（Tukey biweight、Cauchy）。  
- 用交叉验证选择 `delta` 与正则项。  
- 扩展到多输出回归与加权样本场景。  
- 增加残差分布和权重分布可视化。  
- 与 `sklearn` 的 `HuberRegressor` 做一致性验证。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 调用 `build_synthetic_dataset` 生成线性真值数据，并在训练标签注入离群点，得到 `x_train/y_train_contaminated` 与干净测试集。  
2. `fit_ols` 先在污染训练集上求 OLS 解，作为基线和 Huber 初始化。  
3. `fit_huber_irls` 内部用 `initial_residual = y - A@theta_ols`，并通过 `choose_huber_delta -> mad_scale` 估计阈值 `delta`。  
4. 每轮 IRLS 先用当前残差调用 `huber_weights` 计算样本权重：大残差样本权重按 `delta/|r|` 衰减。  
5. `solve_weighted_least_squares` 解加权法方程 `(A^T W A + lambda R) theta = A^T W y`，更新参数。  
6. 同轮使用 `huber_objective` 记录目标值，并计算 `rel_change` 判断是否达到 `tol` 收敛。  
7. 收敛后用 `predict` 在训练与测试上产生 OLS/Huber 预测，并通过 `rmse`、`mae` 汇总指标。  
8. `main` 打印参数误差、IRLS 日志、预测样例和 `Summary`，形成完整“建模-求解-评估”闭环。
