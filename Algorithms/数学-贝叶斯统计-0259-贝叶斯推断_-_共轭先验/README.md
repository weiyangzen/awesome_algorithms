# 贝叶斯推断 - 共轭先验

- UID: `MATH-0259`
- 学科: `数学`
- 分类: `贝叶斯统计`
- 源序号: `259`
- 目标目录: `Algorithms/数学-贝叶斯统计-0259-贝叶斯推断_-_共轭先验`

## R01

共轭先验（Conjugate Prior）指的是：在给定似然模型下，先验分布与后验分布属于同一分布族。  
这使贝叶斯更新从“数值积分问题”变成“参数代数更新问题”。

本条目选择最经典的 Bernoulli/Binomial + Beta 组合：

- 数据模型：`x_i ~ Bernoulli(p)` 或 `k ~ Binomial(n, p)`
- 先验：`p ~ Beta(alpha, beta)`
- 后验：`p | data ~ Beta(alpha + k, beta + n - k)`

核心价值是更新快、可解释、可在线迭代。

## R02

为什么共轭先验重要：

- 工程上：后验有闭式解，不需要 MCMC 就能得到可用结论；
- 教学上：能清晰展示“先验 + 证据 -> 后验”的数学机制；
- 计算上：批量更新和流式（逐条）更新完全一致，便于在线系统；
- 解释上：`alpha-1` 与 `beta-1` 常被解释为“先验成功/失败伪计数”。

因此共轭模型常作为贝叶斯系统的 baseline。

## R03

本 MVP 解决的问题：

- 输入：0/1 观测序列（例如点击/未点击、正例/负例）；
- 任务：在 Beta 先验下更新 Bernoulli 参数 `p` 的后验；
- 输出：
  - 后验参数 `alpha_post, beta_post`
  - 后验均值、方差、置信区间（可信区间）
  - 下一次成功概率 `P(x_next=1|data)`
  - 未来 `m` 次试验中成功 `k` 次的后验预测概率（Beta-Binomial）。

`demo.py` 会给出批量更新、序列更新、网格校验三重验证。

## R04

数学形式（Beta-Binomial 共轭）：

设样本有 `n` 次试验，成功数 `k`，失败数 `n-k`。

- 先验：`p(p) ∝ p^(alpha-1) * (1-p)^(beta-1)`
- 似然：`p(data|p) ∝ p^k * (1-p)^(n-k)`
- 后验：
  `p(p|data) ∝ p^(alpha+k-1) * (1-p)^(beta+n-k-1)`
  即
  `p|data ~ Beta(alpha+k, beta+n-k)`

后验均值：
`E[p|data] = (alpha+k)/(alpha+beta+n)`

后验方差：
`Var[p|data] = (a*b)/((a+b)^2*(a+b+1))`，其中 `a=alpha+k, b=beta+n-k`。

## R05

最小算法流程（与 `demo.py` 一致）：

1. 读取先验参数 `alpha, beta` 和 0/1 数据。
2. 统计 `k = sum(x)` 与 `n = len(x)`。
3. 计算后验参数：`alpha_post = alpha + k`，`beta_post = beta + n - k`。
4. 从后验 Beta 分布计算均值、方差、可信区间。
5. 计算后验预测：
   - 一步预测成功概率 `alpha_post / (alpha_post + beta_post)`；
   - `m` 步预测用 `Beta-Binomial(m, alpha_post, beta_post)`。
6. 用序列更新和网格近似再做一致性验证。

## R06

手算例子：

- 先验：`Beta(2,2)`
- 观测：`n=10`，其中成功 `k=7`

则后验为：

- `alpha_post = 2 + 7 = 9`
- `beta_post = 2 + (10-7) = 5`
- 后验：`Beta(9,5)`

后验均值：
`E[p|data] = 9/(9+5) = 9/14 ≈ 0.6429`

可见相较样本频率 `0.7`，后验均值因先验而向中间收缩。

## R07

后验预测（Posterior Predictive）：

- 单步成功概率：
  `P(x_next=1|data) = E[p|data] = alpha_post/(alpha_post+beta_post)`
- 未来 `m` 次试验成功 `k_future` 次概率：
  `P(k_future|data) = C(m,k_future) * B(k_future+alpha_post, m-k_future+beta_post) / B(alpha_post,beta_post)`
  这正是 Beta-Binomial 分布。

在业务上，它能直接回答“下一段流量里会出现多少正例”的概率问题。

## R08

复杂度分析：

- 批量更新：
  - 时间复杂度 `O(n)`（一次遍历计数成功数）；
  - 空间复杂度 `O(1)`（仅保存若干标量）。
- 序列更新（流式）：
  - 每到一条样本 `O(1)` 更新；
  - 全部 `n` 条仍为 `O(n)`。
- 网格校验（用于验证，不是必须）：
  - 若网格点数为 `G`，则 `O(G)`。

## R09

正确性验证点（本 MVP 自动检查）：

- 批量后验参数 == 序列逐条更新后参数；
- 后验均值与一步预测概率一致；
- 网格法构建的归一化后验密度与解析 Beta 后验的 `L1` 误差很小；
- 未来 `m` 次成功数的 Beta-Binomial 概率和约等于 1。

这些检查可防止“公式写错但程序可运行”的伪正确。

## R10

数值稳定性与边界处理：

- 输入只接受 `{0,1}`，避免把连续值误传给 Bernoulli 模型；
- 要求 `alpha>0, beta>0`，否则 Beta 分布非法；
- 网格法中 `p` 避开 0 和 1（如 `1e-6` 到 `1-1e-6`）防止 `log(0)`；
- 计算未归一化后验时采用 log-space，再做指数平移归一化，减小上/下溢风险。

## R11

`demo.py` 主要函数职责：

- `generate_bernoulli_data`：生成可复现实验数据；
- `validate_binary_data`：输入合法性检查；
- `batch_beta_posterior_update`：闭式批量更新；
- `sequential_beta_posterior_update`：逐样本在线更新并记录轨迹；
- `beta_posterior_summary`：计算均值、方差、区间；
- `beta_binomial_predictive_pmf`：计算未来成功次数的预测分布；
- `grid_vs_closed_form_l1_error`：网格近似与解析后验对照；
- `main`：执行实验、打印并断言。

## R12

输出字段说明：

- `n, k`：样本总数与成功数；
- `posterior alpha/beta`：后验参数；
- `posterior mean/var`：后验矩；
- `credible interval`：后验参数区间（默认 95%）；
- `P(next_success|data)`：下一次观测为 1 的预测概率；
- `predictive(k)`：未来 `m` 次中成功 `k` 次的概率；
- `grid-vs-closed L1 error`：网格后验与解析后验差异。

## R13

运行方式：

```bash
cd Algorithms/数学-贝叶斯统计-0259-贝叶斯推断_-_共轭先验
uv run python demo.py
```

依赖：

- `numpy`
- `scipy`
- `pandas`

无需任何交互输入，执行后直接输出结果和自动校验结论。

## R14

超参数影响（先验强度）：

- 当 `alpha+beta` 较大时，先验更“硬”，后验更难被少量数据改变；
- 当 `alpha=beta=1` 时是均匀先验，几乎完全由数据驱动；
- 非对称先验（如 `alpha>beta`）会将后验均值向更大概率方向拉动。

工程建议：

- 冷启动时可用温和先验（如 `alpha=2,beta=2`）；
- 有历史统计时可把历史成功/失败折算进先验。

## R15

常见失败模式：

- 把“无信息先验”误认为总是 `Beta(1,1)`；
- 把后验均值当成后验众数（两者不同）；
- 忘记检查数据是否二值；
- 误把置信区间（频率学）和可信区间（贝叶斯）混用。

本 MVP 通过显式输入校验和结果字段命名来降低这些风险。

## R16

同类共轭对照（扩展视角）：

- `Poisson` 似然 + `Gamma` 先验 -> `Gamma` 后验；
- 高斯均值未知（方差已知）+ 高斯先验 -> 高斯后验；
- 多项分布 + Dirichlet 先验 -> Dirichlet 后验。

共同模式：

- 后验参数 = 先验参数 + “样本充分统计量”。

这也是共轭先验在在线学习中高效的根本原因。

## R17

MVP 局限与可扩展：

- 仅覆盖单参数 Bernoulli 场景；
- 未处理分层贝叶斯或时间漂移；
- 可信区间与预测分布均是静态批量结果。

可扩展方向：

- 扩展为 Beta-Binomial 多组 A/B 测试比较；
- 引入折扣因子做“遗忘”以适应非平稳数据；
- 升级到分层模型（共享先验）并用 MCMC/VI 近似。

## R18

`demo.py` 源码级流程拆解（8 步）：

1. `main` 设定真值 `p_true`、先验 `alpha,beta` 和样本量，调用 `generate_bernoulli_data` 生成 0/1 数据。  
2. `validate_binary_data` 检查数据维度与取值，只允许一维且元素属于 `{0,1}`。  
3. `batch_beta_posterior_update` 一次性统计成功数 `k` 并计算 `alpha_post, beta_post`。  
4. `sequential_beta_posterior_update` 逐条样本更新后验参数，构造迭代轨迹表，最后与批量结果对比。  
5. `beta_posterior_summary` 调用 Beta 分布公式/函数得到后验均值、方差和 95% 可信区间。  
6. `beta_binomial_predictive_pmf` 计算未来 `m` 次试验中每个成功次数 `k` 的后验预测概率。  
7. `grid_vs_closed_form_l1_error` 在 `p` 网格上用 `log prior + log likelihood` 构建数值后验，并与解析 Beta 后验做 `L1` 误差比较。  
8. `main` 打印关键结果并执行断言：批量=序列、预测概率归一、网格误差足够小，最终给出“运行成功”结论。  
