# 置信区间计算

- UID: `MATH-0267`
- 学科: `数学`
- 分类: `统计推断`
- 源序号: `267`
- 目标目录: `Algorithms/数学-统计推断-0267-置信区间计算`

## R01

置信区间（Confidence Interval, CI）用于在给定置信水平下给出参数可能取值范围。它的典型形式是：

- 点估计：`theta_hat`
- 区间估计：`[L(X), U(X)]`
- 置信水平：`P(theta in [L(X), U(X)]) = 1 - alpha`

本条目实现 4 种常用区间计算：

- 总体方差已知时的均值 Z 区间
- 总体方差未知时的均值 t 区间
- 二项比例 Wilson 区间
- 非参数 bootstrap 百分位区间

## R02

在统计推断中，点估计（例如样本均值）不足以表达不确定性。置信区间提供了“估计值 + 不确定范围”的统一表示，广泛用于：

- A/B 测试效果评估
- 质量控制参数区间判定
- 生物医学试验中的疗效与风险区间报告
- 机器学习模型指标（均值、比例）稳定性说明

区间方法的关键不是“给出一个数”，而是“给出可解释的误差范围”。

## R03

MVP 任务定义：

- 输入 1：实数样本 `x_1,...,x_n`（用于均值区间）
- 输入 2：`successes/trials`（用于比例区间）
- 输入 3：置信水平 `confidence`（默认 0.95）
- 输出：统一数据结构 `ConfidenceInterval(method, confidence, lower, upper, center, margin)`

MVP 目标：

- 给出可复现、可运行、无交互的 Python 脚本
- 展示区间计算结果与基本覆盖率验证
- 不依赖高层黑盒 `confint` API，而是显式写出公式与计算步骤

## R04

本实现使用的核心公式：

- 均值 Z 区间（已知 `sigma`）：
  - `x_bar +/- z_(1-alpha/2) * sigma / sqrt(n)`
- 均值 t 区间（未知 `sigma`）：
  - `x_bar +/- t_(1-alpha/2, n-1) * s / sqrt(n)`
- Wilson 比例区间：
  - `p_tilde = (p_hat + z^2/(2n)) / (1 + z^2/n)`
  - `half = z/(1+z^2/n) * sqrt(p_hat(1-p_hat)/n + z^2/(4n^2))`
  - 区间：`[p_tilde-half, p_tilde+half]`
- Bootstrap 百分位区间：
  - 重采样 `B` 次，得到统计量样本 `T*`
  - 取分位数 `q_(alpha/2), q_(1-alpha/2)`

## R05

复杂度分析（`n` 为样本量，`B` 为 bootstrap 重采样次数）：

- Z 区间：时间 `O(n)`，空间 `O(1)`（不计输入）
- t 区间：时间 `O(n)`，空间 `O(1)`
- Wilson 区间：时间 `O(1)`，空间 `O(1)`
- Bootstrap 百分位：时间 `O(B*n)`，空间 `O(B*n)`（当前实现向量化存储重采样索引与样本）

## R06

手算微型示例（均值 t 区间）：

- 样本：`[3, 4, 5, 6, 7]`，`n=5`
- `x_bar = 5`
- 样本标准差 `s = sqrt(2.5) ≈ 1.5811`
- 95% 置信下 `t_(0.975,4) ≈ 2.776`
- 边际误差：`2.776 * 1.5811 / sqrt(5) ≈ 1.963`
- 区间约为：`[3.037, 6.963]`

这反映了小样本下 t 临界值较大，区间会比大样本更宽。

## R07

方法优缺点（本 MVP 涵盖）：

- Z 区间优点：公式简单、解释直接；限制是通常需要已知总体方差
- t 区间优点：现实中更常用，适合未知方差；限制是依赖近似正态或样本量足够
- Wilson 区间优点：二项比例在小样本/极端比例时比 Wald 更稳健
- Bootstrap 优点：可迁移到均值以外统计量；限制是计算更重，且仍依赖样本代表性

## R08

关键推导思路：

- Z/t 区间都可写为“点估计 ± 临界值 × 标准误差”
- 差异在于临界值来源：
  - 已知方差用正态分布分位数 `z`
  - 未知方差用 t 分布分位数 `t`
- Wilson 区间可看作对比例估计做了中心与方差修正，避免朴素区间在边界处表现不稳定
- Bootstrap 则通过“数据驱动的经验分布”近似统计量采样分布

## R09

适用前提与边界条件：

- `confidence` 必须在 `(0,1)`
- t 区间要求样本量至少 2
- Z 区间要求 `sigma > 0`
- Wilson 要求 `0 <= successes <= trials` 且 `trials > 0`
- Bootstrap 要求样本非空且重采样次数足够（本实现要求 `>=100`）

若数据存在强依赖、显著重尾或采样偏差，区间覆盖率会偏离名义值。

## R10

正确性验证要点：

- 区间基本性质：`lower <= upper` 且 `margin >= 0`
- 均值区间中心应等于样本均值（Z/t）
- Wilson 区间应落在 `[0,1]`
- 通过 Monte Carlo 检查覆盖率应接近目标置信度（本脚本给出 1200 次实验）

`demo.py` 中用断言自动检查以上条件，确保无交互运行时即可完成最小验证。

## R11

数值稳定与实现细节：

- 统一使用 `float64`（NumPy 默认）
- 分位数只调用 `scipy.stats.norm.ppf` 和 `scipy.stats.t.ppf`，避免手写逆 CDF 误差
- Wilson 区间结果做 `[0,1]` 裁剪，防止浮点微小越界
- Bootstrap 采用向量化索引重采样，减少 Python 循环开销

## R12

性能与调参建议：

- 日常均值区间优先 Z/t 闭式公式，速度更快
- Bootstrap 的主要超参数是 `n_resamples`：
  - 较小（如 500）速度快但分位数噪声较大
  - 较大（如 5000+）更稳定但计算更慢
- 覆盖率模拟 `n_trials` 增大后，估计方差下降但运行时间线性增加

## R13

统计解释提醒：

- 95% 置信区间并不表示“参数有 95% 概率在这一个已算出的区间里”
- 频率学派含义是：重复抽样并重复构造区间时，约 95% 的区间会覆盖真值
- 单次实验的区间宽度由样本波动和样本量共同决定

## R14

常见错误与防护：

- 把 t 区间标准差写成 `ddof=0`
  - 防护：实现中明确 `np.std(x, ddof=1)`
- 对比例直接使用 Wald 区间导致小样本失真
  - 防护：MVP 采用 Wilson 区间
- 误把 bootstrap 当成“无需前提的万能方法”
  - 防护：README 明确样本代表性仍是前提
- 忽略输入校验导致无意义区间
  - 防护：每个函数都做边界检查并抛 `ValueError`

## R15

`demo.py` 模块结构：

- `ConfidenceInterval`：统一输出结构与 `contains` 判断
- `z_interval_mean_known_sigma`：均值 Z 区间
- `t_interval_mean_unknown_sigma`：均值 t 区间
- `wilson_interval_proportion`：二项比例 Wilson 区间
- `bootstrap_percentile_interval`：非参数百分位 bootstrap 区间
- `estimate_mean_interval_coverage`：Monte Carlo 覆盖率估计
- `main`：构造样本、打印结果、执行断言

## R16

相关方法对比：

- Wald 比例区间：形式简单但边界表现差（尤其小样本）
- Clopper-Pearson：更保守的精确二项区间
- Agresti-Coull：Wilson 的近似修正版本
- Bootstrap BCa：比简单百分位更精细，但实现更复杂
- 贝叶斯可信区间：解释语义不同于频率学派置信区间

## R17

运行方式（无交互）：

```bash
cd Algorithms/数学-统计推断-0267-置信区间计算
uv run python demo.py
```

依赖：

- `numpy`
- `scipy`
- Python 标准库：`dataclasses`、`typing`

运行后会打印：

- 均值的 Z/t/bootstrap 区间
- 比例的 Wilson 区间
- 1200 次 Monte Carlo 覆盖率估计

## R18

`demo.py` 源码级算法流程（9 步）：

1. `main` 固定随机种子，生成一组高斯样本与一组二项样本，确保结果可复现。  
2. 调用 `z_interval_mean_known_sigma`：先校验输入，再用 `norm.ppf` 取 `z` 分位数，按 `x_bar ± z*sigma/sqrt(n)` 计算上下界。  
3. 调用 `t_interval_mean_unknown_sigma`：计算 `ddof=1` 样本标准差，用 `t.ppf` 取临界值，按 `x_bar ± t*s/sqrt(n)` 得区间。  
4. 调用 `wilson_interval_proportion`：由 `successes/trials` 得到 `p_hat`，计算 Wilson 修正中心与半宽，再限制在 `[0,1]`。  
5. 调用 `bootstrap_percentile_interval`：生成 `B x n` 重采样索引矩阵，批量重采样并计算统计量向量。  
6. 对 bootstrap 统计量向量取 `alpha/2` 与 `1-alpha/2` 分位数，得到百分位区间上下界。  
7. `estimate_mean_interval_coverage` 做 1200 次 Monte Carlo：每轮生成样本，分别构造 Z/t 区间并累计是否覆盖真值。  
8. `main` 打印所有区间结果与覆盖率统计，形成最小可核对输出。  
9. 最后执行断言检查区间有序性、边界合法性和覆盖率合理范围，保证脚本可用于自动验证。  

说明：本实现仅使用 `scipy.stats` 获取分位数，区间构造公式、bootstrap 重采样和覆盖率统计都在源码中显式展开，不是第三方黑盒一键求解。
