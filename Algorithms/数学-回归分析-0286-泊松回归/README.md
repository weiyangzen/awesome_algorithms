# 泊松回归

- UID: `MATH-0286`
- 学科: `数学`
- 分类: `回归分析`
- 源序号: `286`
- 目标目录: `Algorithms/数学-回归分析-0286-泊松回归`

## R01

泊松回归（Poisson Regression）用于建模“计数型目标变量”，例如事件发生次数、到访次数、故障数量等。  
本目录给出一个最小可运行 MVP：
- 手写 `IRLS`（Iteratively Reweighted Least Squares）求解，不把拟合完全交给黑盒库；
- 使用 `log` 链接函数，保证预测均值 `mu` 恒为正；
- 输出收敛信息、对数似然、偏差（deviance）和预测误差，形成可审计结果。

## R02

问题定义（本实现）：
- 输入：
  - 特征矩阵 `X in R^(n x p)`；
  - 计数标签 `y in N_0^n`（非负整数）；
  - 超参数：`max_iter`、`tol`、`l2_reg`。
- 模型：
  - `eta = beta0 + X beta`；
  - `mu = exp(eta)`；
  - `y_i ~ Poisson(mu_i)`。
- 输出：
  - 参数估计 `beta_hat`；
  - 预测均值 `mu_hat`；
  - 训练/测试评估指标（`log_likelihood`、`mean_deviance`、`MSE`、`MAE`、`McFadden R^2`）。

## R03

核心数学关系：

1. 泊松分布对数似然（忽略常数可写成）
- `L(beta) = sum_i [ y_i * eta_i - exp(eta_i) - log(y_i!) ]`，其中 `eta_i = x_i^T beta`。

2. 链接函数
- `mu_i = E[y_i|x_i] = exp(eta_i)`。

3. IRLS 的工作变量与权重（Poisson + log link）
- `W = diag(mu_i)`；
- `z_i = eta_i + (y_i - mu_i) / mu_i`。

4. 每次迭代的加权最小二乘子问题
- `beta_new = argmin_b ||W^(1/2) (z - X_aug b)||_2^2`；
- 对应法方程：`(X_aug^T W X_aug + lambda*R) beta_new = X_aug^T W z`。

其中 `R` 为正则矩阵（本实现不惩罚截距项）。

## R04

算法总流程（高层）：
1. 生成可复现合成计数数据（已知真参数）。
2. 划分训练/测试集。
3. 通过 IRLS 在训练集估计参数。
4. 用估计参数在训练/测试集预测 `mu_hat`。
5. 计算并打印收敛、参数误差和回归指标。
6. 输出部分样本预测，做可读性检查。

## R05

核心数据结构：
- `numpy.ndarray`
  - `x`: `(n, p)` 特征矩阵；
  - `y`: `(n,)` 计数标签；
  - `coef`: `(p+1,)`（含截距）。
- `PoissonIRLSResult`（`dataclass`）
  - `coef`、`fitted_mean`、`linear_predictor`；
  - `converged`、`n_iter`；
  - `log_likelihood`、`mean_deviance`；
  - `history[(iter, ll, delta_inf)]`。
- `EvalReport`（`dataclass`）
  - `log_likelihood`、`mean_deviance`、`mse`、`mae`、`mcfadden_r2`。

## R06

正确性要点：
- `mu = exp(eta)` 保证预测均值严格为正，满足泊松分布参数约束；
- IRLS 每轮解一个加权最小二乘问题，是 GLM 在该设定下的标准数值求解方式；
- 每轮通过 `||beta_new - beta||_inf < tol` 判定收敛；
- 计算对数似然与偏差，能够从概率建模角度校验拟合质量。

## R07

复杂度分析（`n` 样本数，`p` 特征数，`T` 迭代轮数）：
- 单轮 IRLS：
  - 构造 `X^T W X` 约 `O(n p^2)`；
  - 解 `(p+1)` 维线性系统约 `O(p^3)`。
- 总时间复杂度：`O(T * (n p^2 + p^3))`。
- 空间复杂度：`O(n p + p^2)`（主要来自数据矩阵与法方程矩阵）。

## R08

边界与异常处理：
- `x` 维度必须是 2D，`y` 必须是 1D，且样本数一致；
- `x/y` 含 `NaN/Inf` 直接报错；
- `y` 必须是非负整数（计数语义）；
- `max_iter/tol/l2_reg/test_ratio` 均做合法性检查；
- 法方程求解失败时回退 `pinv`，避免因矩阵病态崩溃。

## R09

MVP 设计取舍：
- 只实现最核心的 Poisson GLM（`log link` + IRLS），不叠加复杂工程封装；
- 采用合成数据，保证脚本自包含、可复现；
- 指标覆盖概率建模指标与误差指标，方便直观理解；
- 不引入命令行参数解析，保持 `uv run python demo.py` 一步运行。

## R10

`demo.py` 主要函数职责：
- `make_synthetic_poisson_data`：生成可复现计数数据与真参数；
- `validate_dataset`：数据合法性检查；
- `fit_poisson_regression_irls`：核心 IRLS 拟合过程；
- `predict_mean_count`：根据系数预测 `mu`；
- `poisson_log_likelihood` / `poisson_mean_deviance`：概率视角指标；
- `evaluate_poisson_regression`：汇总评估指标；
- `print_*` 函数：结果展示；
- `main`：串联端到端流程。

## R11

运行方式：

```bash
cd Algorithms/数学-回归分析-0286-泊松回归
uv run python demo.py
```

脚本无交互输入，直接输出完整结果。

## R12

输出字段说明：
- `converged`：是否达到迭代收敛阈值；
- `iterations`：实际迭代轮数；
- `true beta / estimated beta / beta error`：参数恢复情况；
- `log_likelihood`：越大通常越好；
- `mean_deviance`：越小通常越好；
- `MSE / MAE`：计数预测误差；
- `McFadden R^2`：相对空模型（仅截距）改进程度；
- `pass`：用于 MVP 运行时快速自检的布尔标记。

## R13

最小验证覆盖：
- 固定随机种子生成数据与切分，保证复现；
- 验证 IRLS 收敛状态与迭代轮数；
- 训练/测试双集评估，避免只看训练结果；
- 对比真参数与估计参数，检查方向与量级合理性。

建议补充测试：
- 调大/调小样本数，观察参数稳定性；
- 调整 `tol` 与 `max_iter`，观察收敛速度；
- 在高计数或稀疏计数场景下观察数值稳定性。

## R14

关键可调参数：
- `max_iter`：IRLS 最大迭代次数；
- `tol`：收敛阈值（参数变化的无穷范数）；
- `l2_reg`：轻量正则，改善病态矩阵下稳定性；
- `max_eta`：线性预测裁剪阈值，防止 `exp` 溢出；
- `test_ratio`：训练/测试划分比例；
- `seed`：数据生成与划分随机种子。

## R15

与相关方法对比：
- 与线性回归：
  - 线性回归适合连续目标；
  - 泊松回归面向计数目标，且天然保证 `mu > 0`。
- 与负二项回归：
  - 泊松回归假设 `Var(y|x)=E(y|x)`；
  - 若明显过度离散，负二项模型通常更稳健。
- 与黑盒库调用：
  - 工程中可直接用 `sklearn`/`statsmodels`；
  - 本实现强调算法透明，可逐步审计每轮更新。

## R16

典型应用场景：
- 用户行为计数（点击次数、访问次数）；
- 设备/系统故障计数；
- 交通、医疗、运维中的事件发生频次建模；
- 作为计数建模基线，给更复杂模型提供可解释对照。

## R17

可扩展方向：
- 引入 offset/exposure（暴露量）处理；
- 增加过度离散诊断并扩展到负二项回归；
- 加入 L1/L2 更完整正则路径搜索；
- 支持稳健标准误与置信区间估计；
- 扩展到零膨胀泊松（ZIP）等稀疏计数模型。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 调用 `make_synthetic_poisson_data` 构造 `X, y` 与 `beta_true`，再做 `train_test_split`。  
2. `fit_poisson_regression_irls` 先执行 `validate_dataset`，并构造增广矩阵 `X_aug=[1, X]`。  
3. 以 `log(mean(y))` 初始化截距，每轮先算 `eta = X_aug @ beta`，再经 `mu = exp(eta)` 得到当前均值。  
4. 根据泊松 GLM 推导构造工作变量：`w=mu`、`z=eta+(y-mu)/mu`。  
5. 组装加权法方程 `(X^T W X + reg) beta_new = X^T W z`，优先 `solve`，失败回退 `pinv`。  
6. 用 `delta_inf = max|beta_new-beta|` 判断是否收敛，并把 `(iter, log_likelihood, delta_inf)` 记录到 `history`。  
7. 训练完成后通过 `predict_mean_count` 在训练/测试集生成 `mu_pred`。  
8. `evaluate_poisson_regression` 计算 `log_likelihood`、`mean_deviance`、`MSE`、`MAE`、`McFadden R^2`，最终打印摘要与 `pass` 标记。  
