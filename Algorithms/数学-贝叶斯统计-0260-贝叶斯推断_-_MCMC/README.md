# 贝叶斯推断 - MCMC

- UID: `MATH-0260`
- 学科: `数学`
- 分类: `贝叶斯统计`
- 源序号: `260`
- 目标目录: `Algorithms/数学-贝叶斯统计-0260-贝叶斯推断_-_MCMC`

## R01

贝叶斯推断（Bayesian Inference）把未知参数视为随机变量，通过先验分布与观测数据结合得到后验分布：

`p(theta | D) ∝ p(D | theta) * p(theta)`

当后验分布无法解析求积分时，常用 MCMC（Markov Chain Monte Carlo）从后验中采样，用样本统计量近似后验均值、区间与预测分布。

本目录实现一个最小可审计 MVP：
- 场景：贝叶斯逻辑回归（二分类）
- 推断器：随机游走 Metropolis（Random-Walk Metropolis）
- 输出：参数后验均值/标准差/95% 可信区间、链诊断（接受率、R-hat、ESS）

## R02

本 MVP 要解决的问题：

给定数据 `X in R^(n*p)`、标签 `y in {0,1}^n`，设模型：

`Pr(y_i=1 | x_i, beta) = sigmoid(x_i^T beta)`

并给 `beta` 设正态先验 `N(0, sigma_prior^2 I)`，目标是通过 MCMC 近似后验 `p(beta|X,y)`，并回答：
- 参数后验中心位置（均值）与不确定性（标准差、可信区间）
- 链是否混合良好（R-hat 是否接近 1）
- 样本有效数量是否足够（ESS）

## R03

选择该问题的原因：
- 逻辑回归后验在常见先验下没有闭式解，天然需要数值近似；
- 参数维度低（含截距共 4 维）便于用手写 MCMC 透明展示；
- 相比直接调用黑盒采样库，手写提议-接受流程更容易验证细节正确性。

## R04

模型定义与目标密度：

1. 似然：
`log p(y|X,beta) = Σ_i [ y_i * eta_i - log(1 + exp(eta_i)) ]`，`eta_i = x_i^T beta`

2. 先验：
`beta_j ~ N(0, sigma_prior^2)`（独立同分布）

3. 后验对数密度（忽略与 beta 无关常数也可）：
`log p(beta|D) = log p(y|X,beta) + log p(beta)`

4. 提议分布（对称随机游走）：
`beta' = beta + epsilon,  epsilon ~ N(0, s^2 I)`

5. Metropolis 接受概率：
`alpha = min(1, exp(logpost(beta') - logpost(beta)))`

## R05

算法流程（多链 Random-Walk Metropolis）：

1. 生成可复现二分类数据，并构造带截距设计矩阵。
2. 设置先验尺度、总步数、burn-in、thin、链数。
3. 每条链从不同初始点出发，循环执行：
   - 采样提议参数 `beta'`
   - 计算 `logpost(beta')` 与 `logpost(beta)` 差值
   - 按 Metropolis 规则接受或拒绝
4. burn-in 阶段每固定窗口自适应调整提议步长（目标接受率附近）。
5. burn-in 后按 thin 间隔保留样本。
6. 合并各链后计算后验统计量与预测准确率。
7. 对每个参数计算 R-hat 与 ESS，检查采样质量。

## R06

正确性要点：
- 提议分布对称（`q(a->b)=q(b->a)`），接受率只需后验比值；
- 马尔可夫链以目标后验为平稳分布（满足详细平衡）；
- burn-in 用于削弱初值影响，thin 降低序列相关性；
- 多链并行可通过 R-hat 观察链间/链内方差是否一致；
- ESS 用自相关衰减估计“等价独立样本数”。

## R07

复杂度分析：

记：
- `n` 为样本数，`p` 为参数维度，`T` 为每链总步数，`m` 为链数。

单次提议需要一次 `X @ beta` 级别计算，成本约 `O(n*p)`。
总成本约：
`O(m * T * n * p)`。

空间复杂度：
- 存储链样本 `O(m * T_keep * p)`，`T_keep ≈ (T-burn_in)/thin`
- 数据矩阵 `O(n*p)`。

## R08

边界与异常处理：
- 检查 `X` 为二维、`y` 为一维且样本数匹配；
- 检查 `y` 仅包含 `{0,1}`；
- 检查输入中无 `nan/inf`；
- 检查 `n_steps > burn_in >= 0`、`thin > 0`、`n_chains > 1`；
- 若 burn-in 后无保留样本，直接报错并提示调整参数。

## R09

MVP 取舍：
- 仅依赖 `numpy` 与标准库，减少环境复杂度；
- 选择随机游走 Metropolis，不引入 HMC/NUTS 等复杂实现；
- 自适应步长只用于 burn-in，采样阶段固定步长，避免破坏平稳性假设；
- 采用合成数据演示，不做数据文件读写与交互输入。

## R10

`demo.py` 主要函数职责：
- `sigmoid`：稳定计算逻辑函数。
- `validate_dataset`：输入合法性检查。
- `log_posterior`：计算目标后验对数密度。
- `run_rw_metropolis_chain`：单链 MCMC（含 burn-in 自适应步长）。
- `rhat_per_dimension`：按维度计算 Gelman-Rubin R-hat。
- `ess_per_dimension`：按维度估计有效样本数。
- `make_synthetic_logistic_data`：生成可复现实验数据。
- `summarize_posterior`：输出后验均值、标准差、95% 区间。
- `main`：组织完整实验并打印诊断与结果。

## R11

运行方式：

```bash
cd Algorithms/数学-贝叶斯统计-0260-贝叶斯推断_-_MCMC
uv run python demo.py
```

脚本无交互输入，运行后直接打印完整结果。

## R12

输出字段说明：
- `acceptance_rate`：每条链提议被接受的比例。
- `final_proposal_scale`：burn-in 自适应后保留下来的步长。
- `posterior mean/std`：参数后验一阶、二阶统计。
- `95% CI`：参数后验 2.5% 与 97.5% 分位数区间。
- `R-hat`：收敛诊断指标，接近 1 越好。
- `ESS`：有效样本数估计，越大越稳定。
- `posterior predictive accuracy`：后验均值预测概率在训练集上的分类准确率。

## R13

最小测试覆盖（脚本内已执行）：
- 固定随机种子，确保可复现；
- 4 条链不同初始点独立采样；
- 输出每条链接受率，检查是否过低/过高；
- 输出每个参数 R-hat 与 ESS；
- 对比后验均值与合成数据真值，观察是否大体贴合。

## R14

关键参数建议：
- `n_steps`：总迭代步数，过小会导致后验估计方差大。
- `burn_in`：丢弃前期样本，降低初值偏差。
- `thin`：抽稀间隔，减弱自相关但会减少保留样本。
- `proposal_scale`：提议步长，过小移动慢，过大拒绝率高。
- `prior_std`：先验强度，越小越强正则。

经验上可先把接受率调到约 `0.2 ~ 0.5`，再增加 `n_steps` 提升稳定性。

## R15

与其他贝叶斯推断方法对比：
- 对比解析后验：解析法速度快但只适用于共轭/可积模型；本问题一般无闭式。
- 对比变分推断（VI）：VI 更快但有近似偏差；MCMC 更接近“真后验”但计算更慢。
- 对比 HMC/NUTS：HMC 在高维常更高效，但实现复杂度和调参难度更高；本 MVP 优先透明可审计。

## R16

典型应用场景：
- 小中等维度分类任务中需要参数不确定性评估；
- 风险敏感决策（不仅要点估计，也要区间）；
- 教学或审计场景，需展示“先验 + 似然 -> 后验采样”的全过程。

## R17

可扩展方向：
- 用块更新或自适应协方差提议替代球形随机游走；
- 扩展到分层先验（hierarchical priors）；
- 引入 HMC/NUTS 以提升高维效率；
- 增加后验预测校准指标（Brier score、对数损失）；
- 支持真实数据加载与结果持久化（CSV/JSON）。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 调用 `make_synthetic_logistic_data` 生成标准化特征、二分类标签和真值参数。
2. 组装 `MCMCConfig`（链数、总步数、burn-in、thin、先验尺度、初始步长）。
3. 对每条链调用 `run_rw_metropolis_chain`，以不同随机种子和初值独立采样。
4. 在单步采样里，用高斯随机游走生成 `proposal`，并分别计算 `log_posterior(proposal)` 与当前点后验。
5. 依据 `log(u) < log_accept_ratio` 执行接受/拒绝；若在 burn-in 阶段，按窗口接受率微调步长。
6. burn-in 之后按 `thin` 间隔保存样本，得到每条链的后验样本矩阵。
7. 将各链样本堆叠后，`summarize_posterior` 计算均值、标准差和 95% 可信区间，并与真值并列打印。
8. 对链样本调用 `rhat_per_dimension` 与 `ess_per_dimension` 做收敛诊断，再计算后验预测概率与准确率作为任务端验证。
