# Kolmogorov-Smirnov检验

- UID: `MATH-0275`
- 学科: `数学`
- 分类: `统计推断`
- 源序号: `275`
- 目标目录: `Algorithms/数学-统计推断-0275-Kolmogorov-Smirnov检验`

## R01

Kolmogorov-Smirnov 检验（简称 KS 检验）是一类基于经验分布函数（ECDF）的非参数检验方法，用于比较“样本分布”和“目标分布”是否一致，或比较“两组样本分布”是否一致。

它的核心统计量是两个分布函数之间的最大垂直距离：
- 单样本 KS：`D_n = sup_x |F_n(x) - F_0(x)|`
- 双样本 KS：`D_{n,m} = sup_x |F_n(x) - G_m(x)|`

本条目给出一个最小可运行 MVP：
- 手写单样本/双样本 KS 统计量计算；
- 使用渐近分布近似 p 值；
- 用 `scipy.stats.kstest` 与 `scipy.stats.ks_2samp` 做对照验证。

## R02

问题定义：

1. 单样本 KS（Goodness-of-fit）
- 输入：样本 `x_1,...,x_n` 与一个完全指定的连续分布 `F_0`。
- 原假设 `H0`：样本来自 `F_0`。
- 备择假设 `H1`：样本不来自 `F_0`（双侧场景）。

2. 双样本 KS（Two-sample）
- 输入：两组独立样本 `x_1,...,x_n` 与 `y_1,...,y_m`。
- 原假设 `H0`：两组样本来自同一连续分布。
- 备择假设 `H1`：两组样本分布不同（双侧场景）。

## R03

为什么 KS 检验适合作为统计推断入门算法：
- 非参数：不要求数据服从特定参数族（除了单样本需要一个给定参考分布）；
- 对整个分布敏感：不仅看均值/方差，也看形状差异；
- 输出直观：`D` 就是两条分布函数曲线最大差值；
- 可直接用于快速分布比较和模型诊断。

本条目没有把第三方函数当黑盒，而是先手写统计量，再做库函数对照。

## R04

核心数学对象：

1. 经验分布函数（ECDF）
- `F_n(x) = (1/n) * sum_{i=1}^n 1(x_i <= x)`

2. 单样本 KS 统计量
- `D_n = max(D_n^+, D_n^-)`
- `D_n^+ = max_i (i/n - F_0(x_(i)))`
- `D_n^- = max_i (F_0(x_(i)) - (i-1)/n)`
其中 `x_(i)` 是样本升序排列。

3. 双样本 KS 统计量
- `D_{n,m} = sup_x |F_n(x) - G_m(x)|`

4. 渐近 p 值近似（双侧）
- 记 `lambda = (sqrt(ne) + 0.12 + 0.11/sqrt(ne)) * D`
- 单样本时 `ne = n`，双样本时 `ne = n*m/(n+m)`
- `p ≈ Q_KS(lambda)`，在实现里用 `scipy.stats.kstwobign.sf(lambda)`。

## R05

算法流程（MVP）：

1. 校验输入样本为一维、非空、有限数值。
2. 单样本：排序后按公式计算 `D+`、`D-`、`D`。
3. 双样本：构造两组样本合并后的有序网格，用 `searchsorted` 计算两条 ECDF 并取最大差值。
4. 根据有效样本量 `ne` 计算渐近 p 值。
5. 若环境安装了 SciPy，则与 `kstest/ks_2samp` 对照，验证统计量与结论一致。
6. 打印结果表格与“是否拒绝 H0”判定（默认 `alpha=0.05`）。

## R06

正确性直觉：
- ECDF 是真实分布函数的样本近似；
- 若样本确实来自同一分布，两个分布函数整体会靠近，`D` 值通常较小；
- 若存在系统差异（位置、尺度、形状），某些区间 ECDF 会明显偏离，`D` 变大；
- p 值反映“在 H0 成立时观察到至少这么大偏差”的概率，越小越不支持 H0。

## R07

复杂度分析：

1. 单样本 KS
- 排序：`O(n log n)`
- 线性扫描求 `D+`/`D-`：`O(n)`
- 总时间：`O(n log n)`，空间约 `O(n)`。

2. 双样本 KS
- 两组排序：`O(n log n + m log m)`
- 网格比较（`searchsorted`）：约 `O((n+m) log(n+m))`
- 总体可写为 `O((n+m) log(n+m))` 级别。

## R08

边界与适用性说明：
- 样本必须是一维数值；
- 本实现假设独立同分布采样；
- 单样本 KS 的参考分布应当“预先指定”，若先用样本估计参数再检验，会改变理论分布（典型是 Lilliefors 情况）；
- 对离散分布存在并列值时，标准连续分布 KS 的 p 值会偏保守或失真，需要专门修正方法。

## R09

MVP 取舍：
- 已实现：单样本/双样本双侧 KS，手写统计量，渐近 p 值，SciPy 对照。
- 未实现：精确有限样本 p 值、单侧备择、离散分布修正、参数估计后修正检验。
- 取舍原则：优先可读、可运行、可验证。

## R10

`demo.py` 关键函数职责：
- `validate_1d_finite`：输入校验。
- `ks_statistic_one_sample`：手写单样本 KS 的 `D+`/`D-`/`D`。
- `ks_statistic_two_sample`：手写双样本 KS 的 `D`。
- `asymptotic_pvalue`：统一的双侧渐近 p 值近似。
- `run_one_sample_experiment`：执行单样本实验并与 SciPy 对照。
- `run_two_sample_experiment`：执行双样本实验并与 SciPy 对照。
- `main`：固定随机种子，组织输出，保证可复现实验。

## R11

运行方式：

```bash
cd Algorithms/数学-统计推断-0275-Kolmogorov-Smirnov检验
python3 demo.py
```

脚本无需任何交互输入。

## R12

输出解读：
- `manual_D` / `scipy_D`：手写与库函数统计量对比，应非常接近；
- `manual_p_asymptotic`：基于渐近分布的近似 p 值；
- `scipy_p`：SciPy 返回 p 值（可能使用更精细策略）；若未安装 SciPy，该列为 `NaN`；
- `reject_H0_at_0.05`：在 `alpha=0.05` 下的拒绝结论。

## R13

内置实验设计：

1. 单样本检验（两组）
- `N(0,1)` 样本 vs 标准正态：通常不拒绝 `H0`；
- `Uniform(-2.5,2.5)` 样本 vs 标准正态：通常会拒绝 `H0`。

2. 双样本检验（两组）
- `N(0,1)` vs `N(0,1)`：通常不拒绝 `H0`；
- `N(0,1)` vs `N(0.7,1)`：通常会拒绝 `H0`。

## R14

参数与实践建议：
- `alpha` 常取 `0.05` 或 `0.01`；
- 样本量越大，检验对微小差异越敏感；
- 只看 p 值不够，建议同时查看 `D`（效应规模）；
- 对模型拟合场景，建议配合 QQ-plot、直方图、残差分析联合判断。

## R15

与常见分布检验对比：
- Shapiro-Wilk：更偏向正态性检验，在小样本常有较强检出力；
- Anderson-Darling：更强调尾部偏差；
- KS：分布无关、通用性强，但对尾部极端差异不一定最敏感。

## R16

典型应用场景：
- 检查仿真输出是否匹配理论分布；
- 比较 A/B 两组指标分布是否发生整体漂移；
- 机器学习中的训练集/线上数据漂移检测（单特征层面）；
- 金融/风控中收益分布或风险指标分布的快速比较。

## R17

可扩展方向：
- 增加单侧 KS（`less` / `greater`）；
- 增加精确 p 值或置换法近似；
- 对离散数据引入 mid-rank / permutation 修正；
- 扩展到批量特征自动检验并输出多重检验校正结果（如 BH/FDR）。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 固定随机种子，分别生成“近似满足 H0”和“明显偏离 H0”的样本对。  
2. `run_one_sample_experiment` 调 `validate_1d_finite` 校验输入，然后调用 `ks_statistic_one_sample`。  
3. 在 `ks_statistic_one_sample` 中，样本先排序，再按序号 `i` 计算 `D+ = max(i/n - F0(x_(i)))` 与 `D- = max(F0(x_(i)) - (i-1)/n)`，最终 `D=max(D+,D-)`。  
4. 同一实验里调用 `asymptotic_pvalue(D, ne=n)`，通过 `kstwobign.sf` 得到双侧渐近 p 值。  
5. 若环境存在 SciPy，`run_one_sample_experiment` 会调用 `scipy.stats.kstest` 对照 `D` 与 p 值；否则跳过对照。  
6. `run_two_sample_experiment` 调 `ks_statistic_two_sample`：先排序两组样本，构造合并网格 `z`，用 `searchsorted(..., side='right')` 在 `z` 上计算两条 ECDF，再取最大绝对差。  
7. 双样本同样通过 `asymptotic_pvalue(D, ne=nm/(n+m))` 得到近似 p 值；若环境存在 SciPy，再与 `scipy.stats.ks_2samp` 对照。  
8. `main` 把各实验结果汇总为 `pandas.DataFrame`，打印统计量、p 值和 `alpha=0.05` 下的拒绝结论，形成可复现 MVP 报告。
