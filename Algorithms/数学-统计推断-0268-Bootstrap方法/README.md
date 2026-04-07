# Bootstrap方法

- UID: `MATH-0268`
- 学科: `数学`
- 分类: `统计推断`
- 源序号: `268`
- 目标目录: `Algorithms/数学-统计推断-0268-Bootstrap方法`

## R01

Bootstrap 方法是一类重采样（resampling）统计推断技术。核心思想是：
在真实总体分布未知时，把“观测样本”当作总体的经验近似，通过“有放回抽样”反复生成伪样本，从而近似目标统计量（如均值、中位数、均值差）的抽样分布。

本条目给出一个可运行 MVP，覆盖：
- 单样本 Bootstrap 置信区间（均值与中位数示例）；
- 双样本 Bootstrap 置信区间（均值差）；
- 手写 percentile/basic/normal 三种区间；
- 用 `scipy.stats.bootstrap` 做可选对照，而非黑盒依赖。

## R02

问题定义（本实现）：

1. 单样本场景
- 输入：样本 `x_1, ..., x_n` 与统计量 `T(x)`（如 `mean`、`median`）。
- 目标：估计参数 `theta = T(F)` 并构造 `(1-alpha)` 置信区间。

2. 双样本场景
- 输入：两组独立样本 `x_1, ..., x_n`、`y_1, ..., y_m`，统计量 `T(x, y)`（如均值差）。
- 目标：估计 `theta = T(F_x, F_y)`，并给出 Bootstrap 置信区间。

3. 输出
- 点估计 `theta_hat`；
- Bootstrap 偏差 `bias_boot = mean(theta*) - theta_hat`；
- Bootstrap 标准误 `SE_boot = sd(theta*)`；
- 各类区间边界。

## R03

Bootstrap 的价值：
- 分布无关：不需要显式写出总体分布密度；
- 适用广：对均值、分位数、中位数、模型指标都可用；
- 工程友好：依赖抽样和重复计算，容易并行、容易复现；
- 对非正态统计量友好：当解析近似不可靠时，Bootstrap 往往更稳健。

本条目将 Bootstrap 逻辑拆成“索引重采样 + 统计量映射 + 区间构造”三个可检查模块，便于验证正确性。

## R04

数学要点：

1. 经验分布
- 给定样本 `x_1,...,x_n`，经验分布记为 `F_n`。
- Bootstrap 样本来自 `F_n`，即从原样本中有放回抽取 `n` 次。

2. Bootstrap 复制
- 第 `b` 次重采样得到 `x^(b)`，计算 `theta^(b) = T(x^(b))`，`b=1,...,B`。
- `theta^(1),...,theta^(B)` 近似 `theta_hat` 的抽样分布。

3. 三种区间（双侧）
- Percentile：`[q_{alpha/2}, q_{1-alpha/2}]`
- Basic：`[2*theta_hat - q_{1-alpha/2}, 2*theta_hat - q_{alpha/2}]`
- Normal：`theta_hat ± z_{1-alpha/2} * SE_boot`

其中 `q_p` 为 Bootstrap 统计量分布的 `p` 分位点。

## R05

算法流程（MVP）：

1. 校验输入为一维有限数值数组。
2. 生成 `B` 组有放回抽样索引矩阵（避免 Python 循环逐条采样）。
3. 用索引矩阵得到 Bootstrap 样本批次。
4. 在批次维度上批量计算统计量，形成 `theta*` 数组。
5. 由 `theta*` 计算 `bias_boot` 与 `SE_boot`。
6. 计算 percentile/basic/normal 置信区间。
7. 可选调用 SciPy 的 `bootstrap(..., method="percentile")` 对照边界。
8. 汇总为 `pandas.DataFrame` 输出。

## R06

正确性直觉：
- 如果观测样本能代表总体，经验分布 `F_n` 就能近似真实分布 `F`；
- 从 `F_n` 反复抽样，相当于模拟“重复做实验”；
- 每次实验得到一个 `theta*`，其波动刻画估计不确定性；
- 区间覆盖率随样本量增大与重采样次数增加通常更稳定。

注意：Bootstrap 不是万能。若样本很小、强依赖、极端偏态或有厚尾异常值，区间质量会明显受影响。

## R07

复杂度分析（单样本）：
- 重采样索引生成：`O(B*n)`；
- 统计量批量计算：`O(B*n)`（均值/中位数常见实现）；
- 分位数计算：`O(B log B)` 或近似线性选择实现。
- 总体：`O(B*n)` 主导，空间 `O(B*n)`（矩阵化实现）。

双样本均值差类似，复杂度约 `O(B*(n+m))`。

在工程中可通过减小 `B`、分块处理、并行计算降低资源占用。

## R08

适用条件与边界：
- 假设观测样本近似独立同分布（i.i.d.）；
- 时间序列/空间相关数据不应直接用 i.i.d. bootstrap，应改用 block bootstrap 等方法；
- 离群点会显著影响重采样分布，建议搭配稳健统计量；
- 置信区间是“随机区间”，不是参数落在区间的后验概率声明。

本实现默认双侧区间，不包含单侧区间与假设检验 p 值估计。

## R09

MVP 取舍说明：
- 已实现：single/two sample、自定义统计量、三类区间、SciPy 对照、固定随机种子。
- 未实现：BCa 区间、studentized bootstrap、block bootstrap、并行/分布式加速。
- 取舍原则：优先把最核心的 Bootstrap 机制写清楚并可复现运行。

## R10

`demo.py` 关键函数职责：
- `validate_1d_finite`：输入合法性校验。
- `bootstrap_resample_indices`：批量生成有放回抽样索引。
- `bootstrap_statistic_1sample`：单样本统计量 Bootstrap 分布。
- `bootstrap_statistic_2sample`：双样本统计量 Bootstrap 分布。
- `percentile_interval` / `basic_interval` / `normal_interval`：三类区间构造。
- `run_one_sample_experiment` / `run_two_sample_experiment`：执行实验并输出记录。
- `main`：构造实验数据、汇总打印结果。

## R11

运行方式：

```bash
cd Algorithms/数学-统计推断-0268-Bootstrap方法
uv run python demo.py
```

脚本不需要任何交互输入，默认会打印单样本与双样本实验结果表。

## R12

输出解读指南：
- `estimate`：原样本点估计；
- `bootstrap_bias`：Bootstrap 偏差估计（接近 0 通常更理想）；
- `bootstrap_se`：Bootstrap 标准误；
- `pct_ci_*`：Percentile 区间；
- `basic_ci_*`：Basic 区间；
- `normal_ci_*`：基于 `SE_boot` 的正态近似区间；
- `scipy_pct_ci_*`：SciPy percentile 区间（若 SciPy 可用）；
- `percentile_CI_contains_0`：双样本均值差区间是否包含 0，可作为差异是否显著的直观提示。

## R13

内置实验：

1. 单样本均值（近正态）
- 数据：`N(2, 1.5^2)`；
- 目标：估计均值及区间；
- 预期：三类区间应大致一致。

2. 单样本中位数（偏态）
- 数据：`Exponential(1)`；
- 目标：估计中位数；
- 预期：percentile/basic 与 normal 可能出现可见差异，体现非对称性影响。

3. 双样本均值差（有差异）
- `X~N(0.5,1)` vs `Y~N(0,1)`；
- 预期：区间多数情况下不含 0。

4. 双样本均值差（无差异）
- `X~N(0,1)` vs `Y~N(0,1)`；
- 预期：区间通常包含 0。

## R14

参数建议：
- `n_resamples`：建议 `1000` 到 `10000`，MVP 默认 `4000`；
- `alpha`：常用 `0.05` 或 `0.01`；
- 样本量过小时，Bootstrap 区间波动会较大；
- 若统计量计算昂贵，可先小 `B` 调试，再逐步增大。

实践里可报告：
- 点估计；
- 区间端点；
- `B`、`alpha`、随机种子；
- 是否采用偏差校正（如 BCa）。

## R15

与经典解析法对比：
- 解析法（如 t 区间）依赖分布假设和公式推导；
- Bootstrap 用计算替代推导，适合“公式难写但可重复计算”的统计量；
- 在小样本且重尾分布下，普通正态近似可能失真，Bootstrap 往往更稳健；
- 但 Bootstrap 也依赖样本代表性，不能弥补系统偏差和采样偏差。

## R16

典型应用场景：
- A/B 测试中的转化率差异、均值差异区间估计；
- 风险管理中 VaR、ES 等复杂统计量不确定性估计；
- 机器学习评估指标（AUC、F1、MAE）置信区间；
- 医学与生物统计中中位数、分位数等稳健统计量推断。

## R17

可扩展方向：
- BCa（Bias-Corrected and accelerated）区间；
- Studentized Bootstrap（利用二级重采样估计标准化统计量）；
- Block Bootstrap（时序相关数据）；
- Wild Bootstrap（异方差回归残差）；
- 与并行框架结合（多核/GPU）提升大规模场景性能。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 固定随机种子、`alpha` 与 `n_resamples`，生成单样本和双样本实验数据。  
2. 进入 `run_one_sample_experiment` / `run_two_sample_experiment` 后，先用 `validate_1d_finite` 做输入校验。  
3. `bootstrap_resample_indices` 生成大小为 `(B, n)`（或 `(B, m)`）的有放回索引矩阵。  
4. 通过索引矩阵切片得到 Bootstrap 样本批次，再调用统计量函数（如 `stat_mean`、`stat_median`、`stat_mean_diff`）在 `axis=1` 上批量计算 `theta*`。  
5. 由 `theta*` 计算 `bootstrap_bias` 和 `bootstrap_se`，并得到点估计 `theta_hat`。  
6. 分别调用 `percentile_interval`、`basic_interval`、`normal_interval` 计算三类置信区间端点。  
7. 若 SciPy 可用，调用 `scipy.stats.bootstrap(..., method="percentile")` 获取对照区间，验证手写实现在同一重采样规模下的合理性。  
8. `main` 将所有实验结果汇总为 `pandas.DataFrame`，打印区间与“是否包含 0”提示，形成可复现的 Bootstrap 推断报告。  
