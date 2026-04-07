# 方差分析 (ANOVA)

- UID: `MATH-0273`
- 学科: `数学`
- 分类: `统计推断`
- 源序号: `273`
- 目标目录: `Algorithms/数学-统计推断-0273-方差分析_(ANOVA)`

## R01

方差分析（Analysis of Variance, ANOVA）用于比较多组样本均值是否存在显著差异。最常见的是单因素 ANOVA：

- 原假设 `H0`：`mu_1 = mu_2 = ... = mu_k`
- 备择假设 `H1`：至少有一组均值不同

其核心思想是把总波动拆成“组间波动”和“组内波动”，再用比值统计量 `F` 判断均值差异是否超过随机波动可解释的范围。

## R02

本条目 MVP 解决的是固定效应单因素 ANOVA：

- 输入：`k>=2` 个独立样本组，每组是一维数值向量；
- 输出：`SSB/SSW/SST`、自由度、均方、`F` 统计量、`p` 值与效应量 `eta^2`；
- 实现方式：
  - 手写 ANOVA 主流程（不把第三方库当黑盒）；
  - 使用 `scipy.stats.f_oneway` 仅做结果对照验证。

## R03

ANOVA 的价值：

- 比多次两两 t 检验更规范，避免显著性水平膨胀；
- 统计量结构清晰，便于教学和工程诊断；
- 可扩展到双因素、重复测量、线性模型框架；
- 与回归模型本质等价，可在统一框架下解释。

## R04

单因素 ANOVA 主要公式：

- 记第 `j` 组样本均值为 `xbar_j`，组大小为 `n_j`，总体均值为 `xbar`。
- 组间平方和：`SSB = sum_j n_j * (xbar_j - xbar)^2`
- 组内平方和：`SSW = sum_j sum_i (x_ij - xbar_j)^2`
- 总平方和：`SST = SSB + SSW`

自由度与均方：

- `df_between = k - 1`
- `df_within = N - k`
- `MSB = SSB / df_between`
- `MSW = SSW / df_within`
- `F = MSB / MSW`

在 `H0` 成立且满足 ANOVA 假设时，`F` 近似服从 `F(df_between, df_within)`，据此得到 `p` 值。

## R05

本实现算法流程：

1. 校验输入分组（维度、长度、有限值）；
2. 计算每组样本量、组均值、总体均值；
3. 计算 `SSB` 与 `SSW`，并得到 `SST`；
4. 计算自由度 `df` 与均方 `MS`；
5. 计算 `F = MSB/MSW`；
6. 用 `scipy.stats.f.sf` 计算右尾 `p` 值；
7. 计算效应量 `eta^2 = SSB/SST`；
8. 与 `scipy.stats.f_oneway` 对照并断言一致。

## R06

微型手算示例（3 组，每组 3 个样本）：

- `G1=[4,5,6]`, `G2=[5,6,7]`, `G3=[8,9,10]`
- 组均值分别为 `5,6,9`，总体均值 `xbar=60/9=6.6667`

计算：

- `SSB = 3*(5-6.6667)^2 + 3*(6-6.6667)^2 + 3*(9-6.6667)^2 = 26`
- `SSW = 2 + 2 + 2 = 6`
- `df_between=2`, `df_within=6`
- `MSB=13`, `MSW=1`
- `F=13`

`F` 值较大，通常对应较小 `p` 值，倾向拒绝“均值全相等”的原假设。

## R07

复杂度分析（`N` 为总样本数，`k` 为组数）：

- 时间复杂度：`O(N + k)`（均值与平方和统计都是线性）；
- 空间复杂度：`O(k)`（仅保存组级统计量，不计输入存储）；
- 若增加检验（Shapiro/Levene），复杂度仍以线性统计为主。

## R08

单因素 ANOVA 常见前提：

- 组间独立；
- 每组总体近似正态（或样本量足够大可放宽）；
- 方差齐性（各组方差接近）。

本 MVP 在 `demo.py` 中额外输出：

- 每组 Shapiro-Wilk 正态性 `p` 值；
- Levene 方差齐性 `p` 值（median-centered）。

这些是辅助诊断，不是机械的“通过/不通过开关”。

## R09

MVP 范围与边界：

- 已实现：单因素（one-way）固定效应 ANOVA、效应量、基础假设检查、SciPy 对照；
- 未实现：
  - Welch ANOVA（异方差稳健）；
  - 事后多重比较（Tukey HSD 等）；
  - 双因素/重复测量 ANOVA；
  - 非参数替代（Kruskal-Wallis）自动回退。

设计取舍：优先把“ANOVA 核心计算链条”写透明、可复现、可验证。

## R10

正确性保证（本实现）：

1. 手写 ANOVA 由平方和分解直接推导；
2. 对每个实验场景都计算 SciPy `f_oneway` 结果；
3. 代码中断言 `manual_F` 与 `scipy_F`、`manual_p` 与 `scipy_p` 数值一致；
4. 附加场景断言：
   - 均值相同场景应不拒绝 `H0`；
   - 均值明显错开场景应拒绝 `H0`。

## R11

数值与工程细节：

- 统一转为 `float` 并检查 `NaN/Inf`，避免脏数据污染统计量；
- 明确要求每组至少 2 个样本，避免残差自由度异常；
- `p` 值使用 `F` 分布生存函数 `sf`（右尾概率）计算；
- 输出 ANOVA 表、组统计与假设诊断，便于排查异常结果。

## R12

实践调参建议：

- 显著性水平常用 `alpha=0.05`，高风险场景可用更严格阈值；
- 若组大小极不平衡，建议额外做稳健性分析（如 Welch ANOVA）；
- 单看 `p` 值不够，建议同时报告效应量 `eta^2`；
- 样本量较小时，结论不稳定，建议结合置信区间与领域知识。

## R13

结果解释建议：

- 若 `p < alpha`：说明“至少一组均值不同”，但不直接告诉你哪两组不同；
- `eta^2 = SSB/SST` 表示组别解释的总变异比例；
- 经验阈值（仅粗略参考）：`0.01/0.06/0.14` 可对应小/中/大效应；
- 若要定位具体组差异，应继续做事后比较（本 MVP 未实现）。

## R14

常见失败模式与防护：

- 失败：把相关样本当独立样本。
  - 防护：确认实验设计，必要时改用重复测量模型。
- 失败：方差明显不齐却直接用普通 ANOVA。
  - 防护：查看 Levene 结果，必要时使用 Welch ANOVA。
- 失败：只看显著性，不看效应大小。
  - 防护：同步报告 `eta^2`。
- 失败：多次两两检验替代 ANOVA。
  - 防护：先做总体 ANOVA，再做多重比较并校正。

## R15

`demo.py` 结构：

- `validate_groups`：分组输入校验；
- `manual_one_way_anova`：手写 ANOVA 主计算；
- `group_summary_frame`：输出每组 `n/mean/std`；
- `make_anova_table`：格式化 ANOVA 分解表；
- `assumption_checks`：Shapiro 与 Levene 辅助检查；
- `compare_with_scipy_f_oneway`：调用 SciPy 对照；
- `run_experiment`：组织单场景运行与结果汇总；
- `main`：执行两个确定性场景并做断言验证。

## R16

相关方法关系：

- 两组均值比较时，one-way ANOVA 与独立样本 t 检验等价；
- one-way ANOVA 可视作线性回归中“类别变量回归”的 `F` 检验；
- 方差不齐时可转 Welch ANOVA；
- 非正态且异常值明显时可考虑 Kruskal-Wallis。

## R17

运行方式：

```bash
cd Algorithms/数学-统计推断-0273-方差分析_(ANOVA)
uv run python demo.py
```

依赖：

- `numpy`
- `pandas`
- `scipy`

脚本无交互输入，运行后会打印组统计、ANOVA 表、假设检查与手写/库函数对照结果。

## R18

`demo.py` 源码级流程拆解（8 步）：

1. `main` 定义两个确定性场景（`H0` 成立与 `H1` 成立），设置 `alpha=0.05`。  
2. 每个场景先经 `validate_groups` 校验：组数、每组样本数、数值合法性。  
3. `manual_one_way_anova` 计算每组均值与总体均值，并求 `SSB`（组间）和 `SSW`（组内）。  
4. 同一函数继续计算 `df_between/df_within`、`MSB/MSW`，得到 `F=MSB/MSW`。  
5. 用 `scipy.stats.f.sf(F, df_between, df_within)` 计算右尾 `p` 值，并给出 `eta^2=SSB/SST`。  
6. `make_anova_table` 和 `group_summary_frame` 生成可读表格；`assumption_checks` 输出 Shapiro 与 Levene `p` 值。  
7. `compare_with_scipy_f_oneway` 运行 SciPy ANOVA，对比手写 `F/p`；`run_experiment` 汇总差异字段。  
8. `main` 对全部场景执行断言：手写与 SciPy 数值一致，且 `H0/H1` 场景结论方向正确。  

这使得第三方库仅承担“交叉验证”角色，ANOVA 主算法链条在源码中完整可见。
