# 最大后验估计 (MAP)

- UID: `MATH-0258`
- 学科: `数学`
- 分类: `贝叶斯统计`
- 源序号: `258`
- 目标目录: `Algorithms/数学-贝叶斯统计-0258-最大后验估计_(MAP)`

## R01

最大后验估计（Maximum A Posteriori, MAP）是在后验分布上做点估计：

- 后验：`p(theta|D) ∝ p(D|theta) p(theta)`
- MAP 定义：`theta_map = argmax_theta p(theta|D)`
- 等价形式：`theta_map = argmax_theta [log p(D|theta) + log p(theta)]`

与 MLE 相比，MAP 在似然之外显式引入先验信息，因此可理解为“带先验正则化”的参数估计。

## R02

背景与定位（简要）：

- MLE 只看当前样本，MAP 同时结合历史知识（先验）；
- 在小样本、噪声大、参数可辨识性弱的场景，MAP 往往比纯 MLE 更稳定；
- 在贝叶斯框架里，MAP 是“完整后验推断”（MCMC/VI）与“纯频率学点估计”之间的折中方案。

## R03

本条目 MVP 选择共轭的 Beta-Bernoulli 模型：

- 数据：`x_i in {0,1}`，`x_i ~ Bernoulli(theta)`；
- 先验：`theta ~ Beta(alpha,beta)`；
- 目标：估计 `theta` 的 MLE、后验均值、MAP，并比较差异；
- 实现要求：
  - 给出闭式 MAP；
  - 从零实现对数后验梯度上升（非黑盒）；
  - 用数值梯度检查与闭式解对照验证正确性。

## R04

Beta-Bernoulli 的关键闭式公式：

设样本中 1 的个数为 `k=sum x_i`，样本量 `n`。

- 似然：`p(D|theta) ∝ theta^k (1-theta)^(n-k)`
- 先验：`p(theta) ∝ theta^(alpha-1) (1-theta)^(beta-1)`
- 后验：`theta|D ~ Beta(alpha+k, beta+n-k)`

由此得到：

- `theta_mle = k/n`
- `theta_post_mean = (alpha+k)/(alpha+beta+n)`
- 当 `alpha+k>1` 且 `beta+n-k>1` 时，
  `theta_map = (alpha+k-1)/(alpha+beta+n-2)`

## R05

复杂度分析：

- 闭式统计（一次计数）：
  - 时间复杂度 `O(n)`
  - 空间复杂度 `O(1)`（不计输入存储）
- 梯度上升（迭代 `T` 轮）：
  - 时间复杂度 `O(T)`（本实现把数据压缩为 `k,n` 后每轮是常数操作）
  - 空间复杂度 `O(T)`（若保存目标函数历史），不保存历史则 `O(1)`

## R06

手算示例：`n=10` 次投掷，`k=7` 次正面，先验 `Beta(3,3)`。

1. `theta_mle = 7/10 = 0.7`
2. 后验 `Beta(3+7,3+3)=Beta(10,6)`
3. 后验均值 `theta_post_mean = 10/(10+6)=0.625`
4. MAP `theta_map = (10-1)/(10+6-2)=9/14≈0.6429`

可以看到：`theta_map` 相对 `theta_mle` 被先验拉向 0.5 附近，体现了先验收缩效应。

## R07

MAP 的优势与局限：

- 优势：
  - 能引入先验知识，缓解小样本过拟合；
  - 形式上常等价于“似然 + 正则项”，工程上易理解；
  - 对很多共轭模型可得闭式解，计算高效。
- 局限：
  - 先验选取不当会引入系统偏差；
  - 只输出一个点，丢失后验不确定性信息；
  - 在强偏态或多峰后验中，单点估计可能误导。

## R08

本 MVP 梯度公式推导（对 `phi=logit(theta)` 优化）：

设

- `theta = sigmoid(phi)`
- `a = k + alpha - 1`
- `b = (n-k) + beta - 1`

去掉与 `theta` 无关常数后：

`L(phi) = a log(theta) + b log(1-theta)`

利用链式法则：

- `d theta / d phi = theta(1-theta)`
- `dL/dphi = a - (a+b)theta`

`demo.py` 直接实现这一解析梯度，并用 finite difference 做自动校验。

## R09

适用前提与边界条件：

- 数据可近似看作 i.i.d. Bernoulli；
- 先验参数需满足 `alpha>0,beta>0`；
- 若后验参数落在边界区域（如 `alpha_post<=1` 或 `beta_post<=1`），MAP 可能在 0 或 1 边界；
- 当样本极少时，结果对先验非常敏感，应在业务上解释清楚先验来源。

## R10

正确性验证（本实现内置）：

1. 数值梯度检查：解析梯度与中心差分误差需足够小；
2. 闭式 MAP 对照：梯度上升解应接近闭式解；
3. 目标改进：最终对数后验应高于初始值；
4. 先验影响实验：大样本下 `|MAP-MLE|` 应小于小样本下对应差距。

## R11

数值稳定性策略：

- 用 `phi=logit(theta)` 把 `theta in (0,1)` 约束转为无约束实数优化；
- `sigmoid` 采用分段实现，避免 `exp` 溢出；
- 计算 `log(theta)` 时做 `clip`（`[1e-12,1-1e-12]`）防止 `log(0)`；
- 固定随机种子，保证验证输出可复现。

## R12

调参与性能建议：

- 该共轭场景优先闭式解，梯度法用于教学和通用化模板；
- 学习率 `lr` 过大易震荡，过小收敛慢；
- `tol` 与 `max_iter` 共同决定精度/耗时平衡；
- 若扩展到多参数 MAP，可用同样的“重参数化 + 梯度检查”流程。

## R13

理论性质（简述）：

- 在正则条件下，随着 `n` 增大，MAP 与 MLE 会渐近一致；
- MAP 的偏差主要来自先验，方差通常较 MLE 更小（尤其小样本）；
- 这是偏差-方差权衡：以少量偏差换取更稳定估计。

## R14

常见失败模式与防护：

- 失败：把 MAP 当作“完整贝叶斯推断”。
  - 防护：明确 MAP 仅是后验众数，不含区间不确定性。
- 失败：忽视边界后验（mode 在 0 或 1）。
  - 防护：在代码中显式处理边界条件。
- 失败：先验参数拍脑袋，结论不可解释。
  - 防护：记录先验来源，并做敏感性分析。
- 失败：直接在 `theta` 空间做无约束优化。
  - 防护：改用 logit 重参数化。

## R15

`demo.py` 结构说明：

- `generate_bernoulli_samples`：生成可复现 Bernoulli 样本；
- `closed_form_beta_bernoulli`：计算 MLE、后验均值、闭式 MAP；
- `log_posterior_and_grad_phi`：返回对数后验与解析梯度；
- `gradient_ascent_map`：从零实现 MAP 梯度上升；
- `finite_difference_grad_check`：数值梯度校验；
- `main`：跑完整实验并执行断言。

## R16

相关方法关系：

- MLE：`argmax log p(D|theta)`，可视作 MAP 在均匀先验下的特例；
- 后验均值：`E[theta|D]`，通常比 MAP 更平滑；
- 完整贝叶斯推断：输出后验分布/区间，信息最完整但成本更高；
- 正则化学习：很多 L2/L1 惩罚可解释为 Gaussian/Laplace 先验下的 MAP。

## R17

运行方式（无交互）：

```bash
cd Algorithms/数学-贝叶斯统计-0258-最大后验估计_(MAP)
uv run python demo.py
```

依赖：

- `numpy`
- Python 标准库：`dataclasses`、`typing`

运行后会打印：小样本与大样本的 MLE/MAP 对比、梯度检查误差、收敛信息与自动断言结果。

## R18

`demo.py` 源码级算法流程拆解（8 步，非黑盒）：

1. `main` 固定随机种子，调用 `generate_bernoulli_samples` 产生 Bernoulli 观测。  
2. `closed_form_beta_bernoulli` 统计 `k,n` 并直接给出 `theta_mle`、后验均值与闭式 `theta_map`。  
3. 在初始 `phi0` 处调用 `finite_difference_grad_check`，对照解析梯度与数值梯度。  
4. `gradient_ascent_map` 初始化 `phi`，每轮调用 `log_posterior_and_grad_phi` 计算目标和梯度。  
5. 按 `phi <- phi + lr * grad` 更新参数，并记录对数后验历史。  
6. 当 `|phi_new-phi| < tol` 或达到 `max_iter` 时停止，得到 `phi_hat`。  
7. 通过 `theta_hat = sigmoid(phi_hat)` 把优化变量映射回概率空间，得到数值 MAP。  
8. `main` 比较“闭式 MAP vs 梯度 MAP”与“MAP vs MLE（小样本/大样本）”，并用断言验证实现正确。  

本 MVP 未调用第三方黑盒估计器；后验、梯度、优化迭代均在源码中显式实现。
