# Jackknife方法

- UID: `MATH-0269`
- 学科: `数学`
- 分类: `统计推断`
- 源序号: `269`
- 目标目录: `Algorithms/数学-统计推断-0269-Jackknife方法`

## R01

Jackknife（删一法）是一种重抽样统计推断方法。核心思想是：对大小为 `n` 的样本，每次删去 1 个观测，得到 `n` 个“留一子样本”，在每个子样本上计算同一个统计量，然后利用这些“留一估计”来估计偏差和标准误。

设原始统计量为 `theta_hat`，第 `i` 个留一估计为 `theta_(i)`，其均值为 `theta_dot=(1/n)sum_i theta_(i)`，则：

- Jackknife 偏差估计：`bias_jack = (n-1)*(theta_dot - theta_hat)`
- Jackknife 偏差修正估计：`theta_jack = theta_hat - bias_jack = n*theta_hat - (n-1)*theta_dot`
- Jackknife 标准误：`se_jack = sqrt((n-1)/n * sum_i (theta_(i)-theta_dot)^2)`

## R02

历史背景（简要）：

- Jackknife 由 M. Quenouille 在 20 世纪中期提出，用于减少估计量偏差；
- J. Tukey 进一步推广并命名该方法，使其成为经典非参数重抽样工具；
- 在 Bootstrap 普及前，Jackknife 是实践中最常用的“通用标准误估计器”之一；
- 至今它仍常用于快速误差评估、影响点诊断，以及作为 Bootstrap 的轻量替代方案。

## R03

本条目 MVP 聚焦以下可验证任务：

- 输入：一维数值样本 `x`；
- 支持任意标量统计量 `statistic(x)`（由 Python 函数传入）；
- 输出：
  - 原始估计 `theta_hat`
  - `n` 个留一估计 `theta_(i)`
  - Jackknife 偏差估计、偏差修正估计、标准误
- 演示重点：
  - 用“有偏方差（分母 `n`）”作为目标统计量，展示 Jackknife 如何显著减小偏差；
  - 用“样本均值”展示 Jackknife 标准误与经典解析标准误的一致性。

## R04

算法定义与公式细节：

1. 计算全样本统计量：`theta_hat = t(x_1,...,x_n)`。
2. 对每个 `i=1..n`，构造删一子样本 `x^(-i)`，计算 `theta_(i)=t(x^(-i))`。
3. 计算留一均值：`theta_dot = (1/n)sum_i theta_(i)`。
4. 计算偏差估计：`bias_jack = (n-1)(theta_dot-theta_hat)`。
5. 给出偏差修正值：`theta_jack = theta_hat - bias_jack`。
6. 计算标准误估计：`se_jack = sqrt((n-1)/n * sum_i (theta_(i)-theta_dot)^2)`。

该方法不依赖分布参数化假设，但默认统计量在样本扰动下“足够平滑”。

## R05

复杂度分析（`n` 为样本量，`C_t(m)` 为统计量函数在 `m` 个样本上的计算成本）：

- 时间复杂度：`O(C_t(n) + n * C_t(n-1))`  
  对常见线性统计量（均值、方差）可近似看作 `O(n^2)`（直接实现）。
- 空间复杂度：`O(n)`（存储 `n` 个留一估计值）。

工程上若统计量可增量更新，可把时间成本从二次量级进一步降低。

## R06

微型手算示例（统计量取“样本均值”）：

样本 `x=[2,4,5,9]`，`n=4`。

1. 全样本均值：`theta_hat = (2+4+5+9)/4 = 5`。
2. 留一均值：
   - 去掉 2：`(4+5+9)/3 = 6`
   - 去掉 4：`(2+5+9)/3 = 16/3`
   - 去掉 5：`(2+4+9)/3 = 5`
   - 去掉 9：`(2+4+5)/3 = 11/3`
3. 留一均值平均：`theta_dot = 5`。
4. 偏差估计：`bias_jack = (4-1)*(5-5)=0`，故 `theta_jack=5`。
5. 标准误估计：
   `se_jack = sqrt(3/4 * ((1)^2 + (1/3)^2 + 0^2 + (-4/3)^2)) ≈ 1.472`。

该值与均值标准误 `s/sqrt(n)` 一致（`s` 为样本标准差，`ddof=1`）。

## R07

优点：

- 几乎不依赖参数分布模型；
- 适配面广，只要能计算统计量函数就可套用；
- 可同时给出偏差估计与标准误估计；
- 相比 Bootstrap，计算和实现更轻量。

局限：

- 对非平滑统计量（如分位数在小样本下）稳定性较弱；
- 对强依赖样本（时间序列、空间相关）需改造为 block-jackknife；
- 直接实现的时间复杂度较高（通常 `O(n^2)`）。

## R08

关键推导直觉（偏差修正）：

- 若统计量的有限样本偏差可展开为 `E[theta_hat]-theta = a/n + O(1/n^2)`；
- 留一估计的偏差近似为 `a/(n-1) + O(1/n^2)`；
- `theta_dot - theta_hat` 近似抓住了 `a/(n(n-1))` 量级差异；
- 乘以 `(n-1)` 后可得到 `a/n` 量级的偏差估计，从而做一阶偏差消除：
  `theta_jack = theta_hat - bias_jack`。

因此 Jackknife 常能将一阶偏差从 `O(1/n)` 降到 `O(1/n^2)`（在正则条件下）。

## R09

适用前提与边界：

- 样本近似 i.i.d.；
- 目标统计量对单点删减具有可解释的局部稳定性；
- 样本量不能过小（否则留一波动可能过大）；
- 若数据有分组/依赖结构，应采用分组删法（如 delete-group Jackknife）。

## R10

本 MVP 的正确性检查：

1. 对单次样本，验证“有偏方差的 Jackknife 修正”与 `ddof=1` 方差几乎一致；
2. 对均值统计量，验证 `se_jack` 与解析标准误 `std(x, ddof=1)/sqrt(n)` 一致；
3. 在 Monte Carlo 实验中比较偏差：
   - 原始有偏方差的平均偏差显著偏负；
   - Jackknife 修正后的平均偏差明显减小。

`demo.py` 对上述检查设置了自动断言。

## R11

数值与实现细节：

- 全部计算使用 `float64`，减小舍入误差；
- 输入检查：`x` 必须一维，且样本数至少为 2；
- 标准误计算中用 `max(var, 0.0)` 防止浮点误差造成负零下开方；
- 通过固定随机种子使演示结果可复现。

## R12

调参与性能建议：

- `n` 较小时，直接逐次删一最直观，便于教学与验证；
- Monte Carlo 的 `trials` 决定结果稳定度，通常 1000 以上更平滑；
- 对重复运行的统计量，可考虑向量化或缓存中间量降低常数开销；
- 若统计量计算昂贵，优先考虑 delete-d Jackknife 或并行化。

## R13

理论性质（简述）：

- Jackknife 对许多“平滑统计量”具有一阶偏差校正能力；
- 其标准误估计在大样本下通常一致；
- 对线性统计量（如样本均值）表现尤其稳健，常与经典解析公式一致；
- 对高阶非线性、非光滑统计量的有限样本表现需单独验证。

## R14

常见失败模式与防护：

- 失败：样本极小（如 `n<=5`）导致留一估计波动过大。  
  防护：增大样本或改用更稳健方法（Bootstrap/贝叶斯）。
- 失败：把相关样本当 i.i.d. 处理。  
  防护：采用 block-jackknife（按块删除）。
- 失败：统计量函数不稳定（含阈值、排序断点）。  
  防护：先做敏感性分析，必要时换用 Bootstrap 分位数区间。
- 失败：误把 Jackknife 当“万能无偏化器”。  
  防护：明确它主要消除一阶偏差，不保证任意统计量完全无偏。

## R15

`demo.py` 的模块结构：

- `JackknifeResult`：统一保存估计、偏差、标准误与留一序列；
- `biased_variance`：定义示例统计量（分母 `n` 的方差估计）；
- `jackknife`：核心删一法实现，返回完整诊断结果；
- `monte_carlo_variance_experiment`：多次重复比较偏差；
- `main`：执行单样本验证 + Monte Carlo 统计 + 断言。

## R16

相关方法对比：

- 与 Bootstrap：
  - Jackknife 更快、更简单；
  - Bootstrap 更通用，适用于更复杂统计量与区间估计。
- 与 Delta Method：
  - Delta 依赖可微解析近似；
  - Jackknife 直接基于重抽样，推导负担更低。
- 与交叉验证（CV）：
  - 都有“留一”结构；
  - CV 评估泛化误差，Jackknife 评估统计量不确定性。

## R17

运行方式（无交互）：

```bash
cd Algorithms/数学-统计推断-0269-Jackknife方法
uv run python demo.py
```

依赖：

- `numpy`
- Python 标准库：`dataclasses`、`typing`

运行后将打印：

- 单次样本上的原始估计、Jackknife 修正估计、标准误；
- Monte Carlo 下各方法平均估计与偏差；
- 自动断言通过信息。

## R18

`demo.py` 源码级流程拆解（8 步）：

1. `main` 固定随机种子，生成一次长度为 `n=12` 的正态样本 `x_single`。  
2. 调用 `jackknife(x_single, biased_variance)`：先算全样本有偏方差，再循环构造 `n` 个删一样本并计算 `theta_(i)`。  
3. 在 `jackknife` 内部计算 `theta_dot`、`bias_jack`、`theta_jack`、`se_jack`，并打包成 `JackknifeResult`。  
4. `main` 同时计算 `np.var(x_single, ddof=1)`，与 `theta_jack` 做数值对照。  
5. `main` 再对统计量 `np.mean` 调用 `jackknife`，并将其 `se_jack` 与解析标准误 `std/sqrt(n)` 比较。  
6. 调用 `monte_carlo_variance_experiment`，重复采样 `trials` 次，分别记录原始有偏方差与 Jackknife 修正方差。  
7. 计算两种估计的 Monte Carlo 平均值和对真方差的偏差，输出“偏差缩减倍数”。  
8. 执行断言：单样本一致性、均值标准误一致性、以及 Monte Carlo 下 Jackknife 偏差更小，全部通过后结束。  

该实现未调用第三方黑盒 Jackknife API，删一重抽样、偏差修正与标准误计算都在源码中显式展开。
