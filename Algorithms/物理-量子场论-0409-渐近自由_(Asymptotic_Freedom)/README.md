# 渐近自由 (Asymptotic Freedom)

- UID: `PHYS-0390`
- 学科: `物理`
- 分类: `量子场论`
- 源序号: `409`
- 目标目录: `Algorithms/物理-量子场论-0409-渐近自由_(Asymptotic_Freedom)`

## R01

渐近自由（Asymptotic Freedom）指的是在非阿贝尔规范理论中，耦合常数会随能标 `mu` 升高而减小。  
对 QCD（`SU(3)`）而言，这意味着高能短程过程里强相互作用趋弱，夸克近似“自由粒子”。

## R02

本条目要解决的核心问题是：  
如何用最小可运行数值脚本，直接验证“QCD 在 `N_f`（夸克味数）较小时具有渐近自由，而味数过大时会失去该性质”。

MVP 目标：

- 计算 QCD 一圈/二圈 beta 系数；
- 数值积分 `d alpha_s / d ln(mu)`，展示 `alpha_s(mu)` 随 `mu` 的流动；
- 比较一圈数值解与一圈解析解；
- 验证 `N_f=5` 与 `N_f=17` 的 beta 符号差异。

## R03

`demo.py` 的输入输出约定（无交互）：

- 输入（脚本内固定）：
1. 参考点：`mu_ref=2 GeV`，`alpha_ref=0.30`；
2. 主案例：`N_f=5`；
3. 对照案例：`N_f=17`（验证失去渐近自由）；
4. 能标网格：`mu_ref * geomspace(1, 500, 14)`。
- 输出：
1. `N_f` 与 `(beta0, beta1)` 系数表；
2. `alpha_s(mu)` 的一圈解析、一圈数值、二圈数值对照表；
3. beta 符号检查打印；
4. 断言通过后输出 `All checks passed.`。

## R04

采用的数学模型（`alpha_s` 记为 `alpha`）：

1. QCD beta 函数（常见规范）：
`mu d alpha / d mu = -beta0 * alpha^2/(2pi) - beta1 * alpha^3/(4pi^2) + ...`

2. 对 `SU(3)`：
`beta0 = 11 - 2N_f/3`
`beta1 = 102 - 38N_f/3`

3. 写成 `t=ln(mu)` 的常微分方程：
`d alpha / dt = -b0 alpha^2 - b1 alpha^3`
其中 `b0=beta0/(2pi)`，`b1=beta1/(4pi^2)`。

4. 一圈解析解：
`alpha(mu)=1 / [b0 ln(mu/Lambda)]`（`b0>0` 且 `mu>Lambda`）。

## R05

复杂度分析（网格点数 `M`）：

- beta 系数计算：`O(1)`；
- 一次 ODE 求解（固定维度一阶方程）：约 `O(M)` 到 `O(M * step_factor)`；
- 表格构建与误差统计：`O(M)`；
- 空间复杂度：`O(M)`。

这里的主成本来自 `solve_ivp` 在 `ln(mu)` 网格上的积分。

## R06

MVP 的算法闭环：

1. 由 `(mu_ref, alpha_ref)` 反推一圈 `Lambda_QCD`；
2. 在同一 `mu` 网格上计算一圈解析解；
3. 用 `solve_ivp` 分别积分一圈与二圈方程；
4. 生成对照表并计算一圈“数值 vs 解析”误差；
5. 通过断言验证“随 `mu` 增大 `alpha` 下降”；
6. 检查 `N_f=5` 与 `N_f=17` 的 beta 符号；
7. 通过后打印 `All checks passed.`。

## R07

优点：

- 直接对应量子场论 beta 方程，公式与代码可审计；
- 同时包含解析解比对与数值积分，不依赖单一路径；
- 通过不同 `N_f` 展示“是否渐近自由”的机制分界。

局限：

- 仅做微扰区的一圈/二圈演示，不覆盖强耦合非微扰区域；
- 默认 `SU(3)`，未泛化到任意规范群表示；
- 未接入实验数据拟合，仅做机制验证。

## R08

前置知识与环境：

- 量子场论中的重整化群与 beta 函数概念；
- `ln(mu)` 变量下的一阶 ODE 积分；
- Python `>=3.10`；
- 依赖：`numpy`, `pandas`, `scipy`。

## R09

适用场景：

- 量子场论课程里演示渐近自由；
- 检查 beta 系数符号与耦合流动方向；
- 作为后续更复杂 RG 流模拟的最小基线。

不适用场景：

- 需要高精度 `alpha_s` 现象学拟合；
- 低能强耦合、格点 QCD 或非微扰计算；
- 涉及阈值匹配、方案依赖细节的精细分析。

## R10

正确性直觉：

1. 当 `beta0>0` 时，小耦合区 `beta(alpha)<0`，所以 `mu` 增大时 `alpha` 减小；
2. QCD 的 `beta0 = 11 - 2N_f/3`，当 `N_f < 16.5` 才是渐近自由；
3. 一圈 ODE 数值解应与一圈解析解一致；
4. 二圈修正不改变高能趋弱的总体趋势（在本参数区间内）。

## R11

数值稳定策略：

- 在 `ln(mu)` 空间积分，减少跨尺度直接积分的不稳定性；
- 约束 `mu_grid` 严格递增且正值；
- 对 ODE 内部 `alpha` 设置极小正下界，避免浮点下溢到非物理负值；
- 使用较严格 `rtol/atol`，并显式检查 `solution.success`。

## R12

关键参数与影响：

- `N_f`：决定 `beta0` 符号，是是否渐近自由的主开关；
- `alpha_ref`：参考耦合强度，过大时微扰近似可信度下降；
- `mu_ref` 与 `mu_grid`：决定演示区间；
- `loops`：`1` 或 `2`，决定是否包含二圈项；
- ODE 容差 `rtol/atol`：影响一圈数值与解析比对误差。

调参建议：

- 需更稳定误差时先加密 `mu_grid`，再收紧容差；
- 如需更接近教材值，可替换参考点为常用 `M_Z` 条件。

## R13

- 近似比保证：N/A（非优化近似算法条目）。
- 随机成功率保证：N/A（全流程确定性，无随机采样）。

可验证保证（由断言给出）：

- 一圈数值解与一圈解析解相对误差低于阈值；
- `N_f=5` 情况下 `alpha(mu)` 随 `mu` 单调下降；
- `N_f=17` 的小耦合区 beta 为正，显示失去渐近自由。

## R14

常见失效模式：

1. `mu <= 0` 或 `mu_grid` 非递增，导致 `ln(mu)` 非法；
2. `mu` 靠近或低于推断出的 `Lambda_QCD`，一圈解析式发散；
3. 传入非物理 `alpha<=0`；
4. 把 `beta0,beta1` 归一化因子写错（`2pi/4pi^2` 系数错误）；
5. ODE 成功标志未检查，导致静默错误结果。

## R15

工程扩展方向：

- 加入三圈/四圈 beta 系数与方案切换；
- 加入夸克阈值匹配（分段 `N_f`）；
- 用 `pandas` 导出 CSV 做参数扫描；
- 对接实验输入点（例如 `alpha_s(M_Z)`）做反推与对比。

## R16

相关条目：

- 重整化群方程（RGE）；
- 非阿贝尔规范理论 beta 函数；
- QCD 跑动耦合常数 `alpha_s(mu)`；
- 深度非弹散射中的 Bjorken scaling violation。

## R17

`demo.py` 交付能力清单：

- 显式实现 `beta0,beta1` 与 `d alpha / d ln(mu)`；
- 提供一圈解析解与一圈/二圈数值积分；
- 输出结构化表格（`pandas`）；
- 内置断言验证渐近自由核心性质；
- 无交互，单命令可运行。

运行方式：

```bash
cd Algorithms/物理-量子场论-0409-渐近自由_(Asymptotic_Freedom)
uv run python demo.py
```

## R18

`demo.py` 的源码级算法流程（9 步）：

1. `qcd_beta_coefficients` 根据 `N_f` 计算 `beta0,beta1`，给出 QCD beta 系数。  
2. `reduced_beta_coefficients` 把系数转换到 `d alpha / d ln(mu) = -b0 alpha^2 - b1 alpha^3` 的数值积分形式。  
3. `beta_alpha` 在给定 `alpha`、`N_f`、圈数下返回局部流速，不把 ODE 求解器当黑盒。  
4. `one_loop_lambda_qcd` 用参考点 `(mu_ref, alpha_ref)` 反推一圈 `Lambda_QCD`。  
5. `alpha_one_loop_closed_form` 计算一圈解析跑动解，作为数值积分基准。  
6. `integrate_running_alpha` 把 `t=ln(mu)` 作为自变量，调用 `solve_ivp` 积分一圈或二圈方程并做成功性检查。  
7. `main` 同时得到一圈解析、一圈数值、二圈数值 `alpha(mu)`，组装为 `pandas` 表格。  
8. `main` 额外计算 `N_f=5` 与 `N_f=17` 在小耦合点的 beta 符号，验证渐近自由分界。  
9. 通过断言检查误差阈值与单调性，全部通过后输出 `All checks passed.`。  

说明：`scipy.integrate.solve_ivp` 只负责通用 ODE 积分；物理方程、系数归一化、边界条件、验证指标都在源码中显式定义。
