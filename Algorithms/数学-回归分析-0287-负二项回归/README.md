# 负二项回归

- UID: `MATH-0287`
- 学科: `数学`
- 分类: `回归分析`
- 源序号: `287`
- 目标目录: `Algorithms/数学-回归分析-0287-负二项回归`

## R01

负二项回归（Negative Binomial Regression）是针对计数数据（`y = 0,1,2,...`）的广义线性模型，常用于“方差显著大于均值”的过度离散场景。它可看作 Poisson 回归的扩展：当数据比 Poisson 更“散”时，用额外离散参数吸收这部分方差。

本目录 MVP 采用常见的 NB2 参数化：
- 条件均值：`E[Y|X]=mu`
- 条件方差：`Var(Y|X)=mu + alpha*mu^2`
- 链接函数：`log(mu)=X*beta`

## R02

问题定义（本实现）：
- 输入：
  - 特征矩阵 `X in R^(n x p)`；
  - 非负整数响应 `y in N^n`；
  - 正则参数 `l2_reg` 与优化迭代上限。
- 输出：
  - 回归系数 `beta`（含截距）；
  - 离散参数 `alpha > 0`；
  - 训练过程收敛信息、训练负对数似然、测试集指标。

## R03

NB2 的概率模型写为：
- 令 `r = 1/alpha`
- `P(Y=y|mu,alpha) = Gamma(y+r)/(Gamma(r)*y!) * (r/(r+mu))^r * (mu/(r+mu))^y`

对应单样本对数似然：
`log p = gammaln(y+r) - gammaln(r) - gammaln(y+1) + r*(log r - log(r+mu)) + y*(log mu - log(r+mu))`

其中 `mu = exp(x^T beta)`，保证预测均值始终为正。

## R04

参数估计采用最大似然（MLE），优化目标是：
`min_{beta,alpha}  -sum_i log p(y_i|x_i;beta,alpha) + (lambda/2)*||beta_(1:)||_2^2`

实现细节：
- 用 `theta = [beta, log(alpha)]` 做无约束参数化，自动保证 `alpha>0`；
- 使用手写解析梯度，不依赖自动微分；
- 使用 `scipy.optimize.minimize(method="L-BFGS-B")` 做数值优化，并给定参数边界提升稳定性。

## R05

核心数据结构：
- `NegativeBinomialRegression`（dataclass）
  - `coef_`: 含截距的参数向量；
  - `alpha_`: 估计离散参数；
  - `n_iter_`: 优化迭代次数；
  - `converged_`: 是否收敛；
  - `train_nll_`: 训练目标函数值。
- `numpy.ndarray`：`X`、`y`、预测均值 `mu_hat`。
- `pandas.DataFrame`：用于打印预测样例表。

## R06

正确性要点：
- 通过 `mu=exp(Xbeta)` 保证均值合法（正数）。
- `log(alpha)` 重参数化避免优化过程中出现非法 `alpha<=0`。
- 对数似然使用 `gammaln` 而非阶乘，避免大计数数值溢出。
- 梯度按解析式推导，优化速度和稳定性优于纯数值差分。
- 合成数据使用 Gamma-Poisson 混合采样，与 NB2 假设一致。

## R07

复杂度分析（`n` 样本、`p` 特征）：
- 单次目标/梯度评估：
  - `Xbeta` 与梯度主成本约 `O(n*p)`；
  - 特殊函数（`gammaln/digamma`）逐样本计算约 `O(n)`。
- 若优化迭代 `T` 次，总体约 `O(T*n*p)`。
- 空间复杂度约 `O(n*p)`（存储数据与中间向量）。

## R08

边界与异常处理：
- `X` 非二维、`y` 非一维、样本数不一致会抛 `ValueError`。
- 若 `y` 含负值会抛 `ValueError`（计数数据必须非负）。
- `test_ratio` 不在 `(0,1)` 会抛 `ValueError`。
- 若优化器未收敛，会抛 `RuntimeError`，避免静默失败。

## R09

MVP 取舍：
- 仅做最小可运行的 NB2 回归，不扩展零膨胀、层级随机效应等复杂变体。
- 数据使用可复现合成样本，避免外部数据依赖。
- 不封装 CLI 参数系统，保持 `uv run python demo.py` 一键运行。
- 不调用现成统计黑盒回归器（如 `statsmodels`），而是手写目标与梯度。

## R10

`demo.py` 主要函数职责：
- `make_synthetic_nb_data`：生成符合 NB2 假设的计数数据。
- `train_test_split`：固定种子划分训练/测试集。
- `nb2_nll_and_grad`：计算 NB2 负对数似然与解析梯度。
- `fit_nb2_regression`：执行 L-BFGS-B 训练并返回模型对象。
- `nb2_mean_nll`：评估测试集平均负对数似然。
- `main`：串联数据生成、训练、评估和结果打印。

## R11

运行方式：

```bash
cd Algorithms/数学-回归分析-0287-负二项回归
uv run python demo.py
```

脚本无交互输入，直接输出参数估计与误差指标。

## R12

输出字段说明：
- `beta_true / alpha_true`：合成数据真实参数。
- `beta_hat / alpha_hat`：模型估计参数。
- `optimizer_converged, n_iter`：优化收敛状态和迭代步数。
- `train_nll`：训练目标值（含 L2 项）。
- `train_rmse/test_rmse`、`train_mae/test_mae`：回归误差。
- `test_mean_nll (NB2)`：测试集平均负对数似然。
- `Sample predictions`：测试集中若干样本的真实值、预测均值、绝对误差。

## R13

建议最小验证项（当前脚本已覆盖）：
- 过度离散计数数据（由 Gamma-Poisson 混合生成）。
- 固定随机种子下的可复现实验。
- 参数恢复：`beta_hat`、`alpha_hat` 应与真实参数同量级。

建议扩展测试：
- 变更 `true_alpha`（弱离散/强离散）观察估计偏差变化。
- 增加样本量检查一致性。
- 构造极端特征尺度，验证数值稳定边界。

## R14

关键超参数：
- `l2_reg`：系数正则强度（默认 `1e-4`），用于抑制过大系数。
- `max_iter`：优化最大迭代次数（默认 `600`）。
- `bounds`：参数边界，避免优化跑到极端数值区。
- 数据生成参数：`n_samples`、`seed`、`true_beta`、`true_alpha`。

调参建议：
- 收敛慢时先提高 `max_iter`；
- 估计不稳定时适当增大 `l2_reg`；
- 对极端离散数据可放宽 `log(alpha)` 上界以提升拟合能力。

## R15

方法对比（概念）：
- 对比 Poisson 回归：
  - Poisson 假设 `Var=mu`；
  - NB2 允许 `Var=mu+alpha*mu^2`，更适合过度离散。
- 对比零膨胀模型：
  - NB2 只建模离散度，不显式建“结构性零”；
  - 零值异常多时可考虑 ZINB。
- 对比树模型回归：
  - 树模型预测灵活但统计解释较弱；
  - NB2 参数可解释性更强，适合计量建模基线。

## R16

典型应用场景：
- 保险理赔次数建模（索赔次数常过度离散）。
- 医疗服务使用次数（门诊次数、住院次数）。
- 交通/事故计数建模。
- 内容平台的互动次数（评论数、转发数）建模。

## R17

可扩展方向：
- 加入偏置项 `offset`（如曝光量 `log(exposure)`）。
- 增加类别特征编码与交互项。
- 扩展零膨胀负二项（ZINB）。
- 引入交叉验证和信息准则（AIC/BIC）做模型选择。
- 输出参数标准误、置信区间与显著性检验。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `make_synthetic_nb_data` 先生成高斯特征，再按 `mu=exp(Xbeta)` 与 `alpha` 构造 Gamma-Poisson 混合采样，得到 NB2 计数标签。  
2. `train_test_split` 用固定种子打乱并切分训练集/测试集。  
3. `fit_nb2_regression` 用 `log(y+0.5)` 线性回归做 `beta` 热启动，并用矩估计初始化 `alpha`。  
4. 训练时，`nb2_nll_and_grad` 根据 NB2 对数似然公式计算目标值：`-sum log p(y|mu,alpha) + L2`。  
5. 同一函数里手工推导并计算梯度：一部分对 `beta`，一部分经 `r=1/alpha` 链式法则回传到 `log(alpha)`。  
6. `scipy.optimize.minimize(..., method="L-BFGS-B")` 重复调用第 4-5 步，在边界约束下迭代更新参数直至收敛。  
7. 得到 `beta_hat` 与 `alpha_hat` 后，`predict_mean` 用 `exp(Xbeta_hat)` 输出计数均值预测。  
8. `main` 计算 RMSE/MAE/NLL，并用 `pandas` 打印预测样例表，形成可审计的端到端 MVP 输出。
