# Metropolis-Hastings算法

- UID: `MATH-0261`
- 学科: `数学`
- 分类: `计算统计`
- 源序号: `261`
- 目标目录: `Algorithms/数学-计算统计-0261-Metropolis-Hastings算法`

## R01

Metropolis-Hastings（MH）算法是最经典的马尔可夫链蒙特卡洛（MCMC）方法之一，用于从难以直接采样的目标分布 `π(x)` 生成样本。

它的关键思想是：
- 构造一个马尔可夫链，让平稳分布就是目标分布 `π(x)`；
- 每步先按提议分布 `q(x'|x)` 产生候选 `x'`；
- 再用接受-拒绝机制修正提议偏差，使长期样本分布收敛到 `π(x)`。

本目录 MVP 特别选择了**非对称提议分布**（对数正态随机游走），显式体现 Hastings 修正项，而不是只做对称 Metropolis 特例。

## R02

本实现解决的具体问题：

- 目标：从泊松率参数 `lambda` 的后验分布采样；
- 数据模型：
  - `y_i ~ Poisson(lambda)`；
  - 先验 `lambda ~ Gamma(alpha0, beta0)`（`beta` 为 rate 参数）；
- 后验（解析可得）：
  - `lambda | y ~ Gamma(alpha0 + sum(y), beta0 + n)`。

为什么选这个任务：
- 有解析后验，便于验证 MH 样本质量；
- 支持“只需未归一化目标密度”的 MH 核心特性展示；
- 参数是一维，代码最小但算法步骤完整。

## R03

核心数学关系：

1. 后验未归一化对数密度：
   - `log pi(lambda) = (alpha-1)log(lambda) - beta*lambda + C`，`lambda>0`。
2. 提议分布（非对称）：
   - `log(lambda') ~ N(log(lambda_t), sigma^2)`；
   - 即 `q(lambda'|lambda_t)` 是对数正态分布。
3. MH 接受率：
   - `a = min(1, [pi(lambda') q(lambda_t|lambda')] / [pi(lambda_t) q(lambda'|lambda_t)])`。
4. 对数形式（实现中使用）：
   - `log r = log pi(lambda') - log pi(lambda_t) + log q(lambda_t|lambda') - log q(lambda'|lambda_t)`；
   - 若 `log(u) < log r`（`u~Uniform(0,1)`）则接受。

该公式中 `q` 比值就是 Hastings 校正项，专门修复非对称提议带来的偏差。

## R04

算法流程（MVP）：

1. 读取先验参数与观测计数，计算后验 `Gamma(alpha_post, beta_post)` 参数。  
2. 设定初值 `lambda_0` 与提议尺度 `sigma`。  
3. 在第 `t` 步，从 `q(lambda'|lambda_t)` 生成候选 `lambda'`（log-normal random walk）。  
4. 计算 `log pi(lambda')` 与 `log pi(lambda_t)`。  
5. 计算 `log q(lambda_t|lambda')` 与 `log q(lambda'|lambda_t)`。  
6. 组合成 `log r`，执行接受-拒绝更新状态。  
7. 重复 `n_steps` 次得到完整链。  
8. 丢弃 burn-in 并按 thin 抽稀，得到近似后验样本并输出诊断。

## R05

核心数据结构：

- `MHConfig(dataclass)`：
  - `n_steps, burn_in, thin, proposal_sigma, init_lambda, seed`；
- `MHResult(dataclass)`：
  - `samples`：burn-in+thin 后样本；
  - `acceptance_rate`：接受率；
  - `full_chain`：完整马尔可夫链。

这些结构使实验参数和输出结果清晰分离，方便复用与扩展。

## R06

正确性要点：

- 接受率使用完整 MH 比率，包含 Hastings 校正，非对称提议下仍满足详细平衡；
- 目标分布只用未归一化对数密度，避免显式归一化常数；
- 对数域计算（`log_accept_ratio`）提升数值稳定性；
- `lambda>0` 约束由 log-normal 提议天然保证；
- 使用解析后验（Gamma）对样本矩、分位数、CDF 误差做交叉校验。

## R07

复杂度分析：

设总迭代数为 `T=n_steps`。

- 时间复杂度：`O(T)`，每步仅常数次标量运算；
- 空间复杂度：
  - 全链存储 `O(T)`；
  - 保留样本约 `O((T-burn_in)/thin)`。

本任务是一维参数采样，因此单步代价极低，主要成本来自迭代步数本身。

## R08

边界与异常处理：

- `n_steps <= burn_in`、`thin<=0`、`proposal_sigma<=0`、`init_lambda<=0` 会抛 `ValueError`；
- 输入观测为空、存在负计数或维度不对会抛 `ValueError`；
- 先验参数非正会抛 `ValueError`；
- 如果 burn-in/thin 导致无保留样本，会抛 `RuntimeError`。

## R09

MVP 取舍：

- 只做一维 `lambda`，不做多维参数块更新；
- 只实现一种提议分布（log-normal random walk）；
- 不做自适应调参（如动态调 `sigma`）；
- 不接入现成 MCMC 框架（PyMC/NumPyro），保证流程透明可追踪。

目标是“小而诚实”：核心 MH 逻辑完整、验证闭环完整、代码短且可读。

## R10

`demo.py` 主要函数职责：

- `validate_config`：检查采样配置合法性；
- `generate_poisson_data`：生成演示用泊松观测；
- `posterior_gamma_params`：计算解析后验参数；
- `log_gamma_target_unnormalized`：后验目标的未归一化对数密度；
- `log_lognormal_q`：log-normal 提议密度；
- `metropolis_hastings_gamma`：MH 主循环；
- `estimate_ess`：基于自相关截断的 ESS 粗估；
- `empirical_cdf`：经验 CDF 计算；
- `main`：组织实验、对照解析解、输出表格与检查结果。

## R11

运行方式：

```bash
cd Algorithms/数学-计算统计-0261-Metropolis-Hastings算法
uv run python demo.py
```

脚本无需任何交互输入。

## R12

输出说明：

- 基础信息：
  - 观测样本数 `n_obs`、观测和/均值；
  - 解析后验 `Gamma(alpha, rate)` 参数；
  - 采样配置（步数、burn-in、thin、提议尺度）。
- 采样诊断：
  - `acceptance_rate`；
  - `kept_samples`；
  - `ESS` 粗估。
- 后验对照表：
  - `mean/std/q2.5/q50/q97.5` 的样本值、理论值与绝对误差。
- 额外检查：
  - `cdf_mse`（经验 CDF 与理论 CDF 的均方误差）；
  - `torch_mean_diff`、`torch_std_diff`（与 NumPy 的一致性检查）；
  - `pass_loose_check`（宽松通过标志）。

## R13

最小测试建议：

1. 修改 `proposal_sigma` 为更小值（如 `0.1`）与更大值（如 `1.0`），比较接受率和 ESS 变化；
2. 修改 `n_steps`（如 `4000` 与 `30000`）观察统计误差收敛；
3. 改变先验强度（`alpha_prior, beta_prior`）验证后验均值移动；
4. 改变 `lambda_true` 与 `n_obs` 观察数据量对后验方差的影响。

## R14

关键参数调优建议：

- `proposal_sigma`：
  - 太小：接受率高但链移动慢、相关性强；
  - 太大：接受率低、拒绝多；
  - 实践上常把接受率调到约 `0.2 ~ 0.6`。
- `burn_in`：
  - 初值偏离目标分布时需要更长 burn-in；
- `thin`：
  - 增大 thin 可降自相关，但会减少样本数量；
- `n_steps`：
  - 预算允许时优先增加总步数，再考虑轻度 thin。

## R15

方法对比：

- 对比直接采样：
  - 本例有解析 Gamma 后验，可直接采样；
  - MH 的价值在于推广到“无法直接采样但可计算相对密度”的问题。
- 对比拒绝采样：
  - 拒绝采样依赖易采样包络且高维效率差；
  - MH 通过马尔可夫链逐步探索复杂分布。
- 对比 HMC/NUTS：
  - HMC/NUTS 在高维连续空间通常更高效；
  - MH 实现更简单、依赖更少，适合教学和基线。

## R16

典型应用场景：

- 贝叶斯后验推断（尤其是只知道未归一化后验）；
- 物理统计中的玻尔兹曼分布采样；
- 复杂似然模型参数估计；
- 作为更高级 MCMC（如自适应 MH、并行链）的基线实现。

## R17

可扩展方向：

- 多维参数与块更新；
- 自适应提议协方差（Adaptive MH）；
- 多链并行与 `R-hat` 诊断；
- 更稳健的 ESS/MCSE 估计；
- 与 HMC/NUTS 做同任务对比基准。

## R18

`demo.py` 源码级算法流（8 步）：

1. `main` 先调用 `generate_poisson_data` 生成观测，再用 `posterior_gamma_params` 得到 `alpha_post, beta_post`。  
2. 组装 `MHConfig`，进入 `metropolis_hastings_gamma`。  
3. `metropolis_hastings_gamma` 初始化 `current=lambda_0`，并计算当前 `log_gamma_target_unnormalized(current)`。  
4. 每轮用 `proposal = current * exp(sigma * N(0,1))` 生成对数正态候选。  
5. 分别计算目标差 `log pi(proposal)-log pi(current)` 与 Hastings 校正 `log q(current|proposal)-log q(proposal|current)`，得到 `log_accept_ratio`。  
6. 用 `log(u) < log_accept_ratio` 做接受-拒绝；接受则状态跳到 `proposal`，否则保留 `current`，并写入 `full_chain`。  
7. 迭代结束后按 `burn_in` 与 `thin` 截取 `samples`，并计算 `acceptance_rate` 返回 `MHResult`。  
8. `main` 使用 `scipy.stats.gamma` 计算理论矩/分位数，使用 `pandas` 生成对照表、`sklearn` 计算 `cdf_mse`、`torch` 做统计交叉校验并打印最终结果。

该流程完整展开了 MH 的核心：提议、Hastings 修正、接受-拒绝、样本诊断，没有调用黑盒 MCMC 采样器。
