# 广义线性模型 (GLM)

- UID: `MATH-0282`
- 学科: `数学`
- 分类: `回归分析`
- 源序号: `282`
- 目标目录: `Algorithms/数学-回归分析-0282-广义线性模型_(GLM)`

## R01

广义线性模型（Generalized Linear Model, GLM）把“线性预测子”与“非高斯响应分布”统一起来：

- 线性预测子：`eta = X * beta`
- 链接函数：`g(mu) = eta`
- 条件均值：`mu = E[y | X]`

它把线性回归、逻辑回归、泊松回归放到同一框架中。  
本目录 MVP 同时演示两种经典 GLM：
- Bernoulli + Logit（二分类）
- Poisson + Log（计数回归）

## R02

本实现的问题定义：

- 输入：
  - 特征矩阵 `X in R^(n x p)`；
  - 标签 `y in R^n`；
  - 分布族（Bernoulli 或 Poisson）；
  - IRLS 超参数 `max_iter / tol / l2_reg`。
- 输出：
  - 拟合系数 `beta`（含截距）；
  - 是否收敛、迭代轮数；
  - 测试集指标：
    - Bernoulli: `accuracy`, `logloss`
    - Poisson: `RMSE`, `mean deviance`
  - 真实系数与估计系数对比、样例预测。

## R03

统一模型形式：

1. 线性预测子：`eta_i = x_i^T beta`
2. 均值：`mu_i = g^{-1}(eta_i)`
3. 方差函数：`Var(y_i | x_i) = V(mu_i)`

本脚本对应两族：

- Bernoulli-Logit：
  - `mu = sigmoid(eta)`
  - `V(mu) = mu * (1 - mu)`
- Poisson-Log：
  - `mu = exp(eta)`
  - `V(mu) = mu`

## R04

优化算法使用 IRLS（Iteratively Reweighted Least Squares）：

- 在第 `t` 轮，构造工作响应 `z` 与权重 `w`：
  - `w_i = (dmu/deta)^2 / V(mu_i)`
  - `z_i = eta_i + (y_i - mu_i) / (dmu/deta)`
- 解加权最小二乘子问题：
  - `beta_{t+1} = argmin ||W^(1/2)(z - X beta)||_2^2 + l2_reg * ||beta_(1:)||_2^2`
- 等价线性方程：
  - `(X^T W X + lambda R) beta = X^T W z`
  - `R` 对角第一项为 0（不正则化截距）。

## R05

高层流程：

1. 生成可复现合成数据（logistic 与 poisson 各一份）。
2. 划分 train/test。
3. 根据族类型检查标签合法性（0/1 或非负整数）。
4. 初始化参数（基于 `mean(y)` 的截距 warm-start）。
5. 迭代执行 IRLS 更新，直到收敛或达到 `max_iter`。
6. 在测试集上预测均值/概率。
7. 计算指标并打印系数恢复效果。

## R06

核心数据结构：

- `GLMModel`（dataclass）
  - `family`: 分布族名称
  - `link`: 链接函数名称
  - `coefficients`: 参数向量（截距 + 特征系数）
  - `n_iter`: 实际迭代轮数
  - `converged`: 是否满足收敛阈值
- `BernoulliLogitFamily` / `PoissonLogFamily`
  - `validate_target`
  - `inv_link`
  - `variance`
  - `dmu_deta`
  - `clip_mu`

## R07

复杂度分析（`n` 样本，`p` 特征，`k` 次迭代）：

- 每次 IRLS 主成本：
  - 构造加权矩阵与法方程约 `O(n p^2)`
  - 解 `(p+1)x(p+1)` 线性系统约 `O(p^3)`
- 总时间复杂度：`O(k * (n p^2 + p^3))`
- 空间复杂度：`O(n p + p^2)`

当 `n >> p` 时，主耗时通常在 `X^T W X` 的构造。

## R08

边界与异常处理：

- 输入校验：
  - `X` 必须 2D、`y` 必须 1D；
  - 行数一致；
  - 不允许 `nan/inf`；
  - 最少样本数限制。
- 标签约束：
  - Bernoulli 仅允许 `0/1`；
  - Poisson 要求非负且“整数型”。
- 超参数约束：
  - `max_iter > 0`，`tol > 0`，`l2_reg >= 0`。
- 数值稳定：
  - 对 `mu`、`w`、`dmu` 做下界裁剪；
  - 线性求解失败时自动回退 `pinv`。

## R09

MVP 取舍说明：

- 只实现最核心的两类 GLM（Bernoulli/Poisson），不扩展到 Gamma/Inverse-Gaussian。  
- 不调用 `statsmodels`/`sklearn` 的一键 GLM 拟合接口，保证算法透明。  
- 使用合成数据做演示，确保仓库内可复现、可直接运行。  
- 暂不包含正则路径搜索、交叉验证与置信区间估计。

## R10

`demo.py` 主要函数职责：

- `validate_inputs`：输入形状与数值合法性检查。
- `add_intercept`：添加截距列。
- `train_test_split`：固定随机种子划分数据。
- `irls_fit`：核心 GLM 拟合器（IRLS）。
- `predict_mean`：输出 `E[y|x]`。
- `binary_accuracy` / `binary_logloss`：二分类指标。
- `poisson_rmse` / `poisson_mean_deviance`：计数回归指标。
- `generate_logistic_data` / `generate_poisson_data`：构造两类可控实验数据。
- `run_logistic_demo` / `run_poisson_demo`：端到端实验入口。

## R11

运行方式：

```bash
cd Algorithms/数学-回归分析-0282-广义线性模型_(GLM)
uv run python demo.py
```

脚本不需要交互输入。

## R12

输出字段说明：

- `converged`：IRLS 是否收敛。
- `n_iter`：收敛或停止时的迭代轮数。
- `test_accuracy` / `test_logloss`：Bernoulli-Logit 测试指标。
- `test_rmse` / `test_mean_deviance`：Poisson-Log 测试指标。
- `coefficient comparison`：真实系数与估计系数逐项对比。
- `Sample probabilities/rates`：部分测试样本预测值。

## R13

最小验证建议（当前脚本已覆盖）：

- 功能性：
  - Bernoulli 与 Poisson 两条流程都可运行；
  - 都能输出收敛信息和指标。
- 数值合理性：
  - 系数估计应接近合成数据真值；
  - Logistic 的 `accuracy` 应明显高于随机猜测；
  - Poisson 的偏差指标应在合理范围。
- 异常路径：
  - 非法标签（如 Bernoulli 中出现 2）应抛 `ValueError`。

## R14

关键参数与调参建议：

- `max_iter`：最大迭代次数。若未收敛可增大。  
- `tol`：收敛阈值。越小越精确但迭代可能更多。  
- `l2_reg`：轻量稳定正则。病态数据可适当增大。  
- 数据生成规模 `n_samples`：更大样本一般系数估计更稳定。

实践建议：
- 收敛慢时先检查特征尺度，必要时做标准化；
- Poisson 任务若存在过度离散，可考虑负二项回归（超出本 MVP 范围）。

## R15

方法对比：

- 对比普通线性回归：
  - GLM 能处理非高斯标签（0/1、计数）；
  - 通过链接函数保证预测落在合法范围（概率、非负率）。
- 对比黑盒实现：
  - 黑盒调用更简洁；
  - 本实现更适合教学和源码审计，便于理解 IRLS 细节。
- 对比非线性深度模型：
  - 深度模型表达力更强；
  - GLM 可解释性更高、样本效率更好、部署更轻量。

## R16

典型应用场景：

- Bernoulli-Logit：
  - 点击率/转化率预测
  - 疾病有无风险建模
- Poisson-Log：
  - 呼叫中心到达数预测
  - 故障次数、事故次数等计数建模
- 作为复杂模型前的高可解释 baseline。

## R17

可扩展方向：

- 增加更多分布族：Gamma、Inverse Gaussian。  
- 支持 offset/exposure（计量统计常见需求）。  
- 增加标准误、Wald 检验、似然比检验。  
- 增加 L1/L2 正则路径与交叉验证。  
- 与稳健标准误（sandwich estimator）结合。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 依次触发 `run_logistic_demo` 与 `run_poisson_demo`，分别构造二分类和计数任务。  
2. 每条任务先通过 `generate_*_data` 生成带真系数的合成数据，再用 `train_test_split` 固定随机切分。  
3. 调用 `irls_fit` 后，先执行输入与标签校验，并为参数向量加截距列初始化。  
4. `irls_fit` 每轮先算 `eta = X beta`、`mu = g^{-1}(eta)`，然后由族对象给出 `V(mu)` 与 `dmu/deta`。  
5. 根据 GLM 理论构造 IRLS 的 `w` 与 `z`，把原问题转为一轮加权最小二乘。  
6. 通过 `(X^T W X + lambda R) beta = X^T W z` 求新参数；若 `solve` 失败回退 `pinv`，保证鲁棒性。  
7. 以参数增量 `||beta_new - beta||` 判断收敛，未收敛继续迭代直到 `max_iter`。  
8. 拟合完成后，`predict_mean` 输出概率/率，最后计算并打印 accuracy、logloss、RMSE、deviance 与系数对照。
