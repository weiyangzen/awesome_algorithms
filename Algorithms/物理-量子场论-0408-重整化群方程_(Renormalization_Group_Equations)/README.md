# 重整化群方程 (Renormalization Group Equations)

- UID: `PHYS-0389`
- 学科: `物理`
- 分类: `量子场论`
- 源序号: `408`
- 目标目录: `Algorithms/物理-量子场论-0408-重整化群方程_(Renormalization_Group_Equations)`

## R01

重整化群方程（RGE）描述“有效耦合常数如何随能标 `mu` 改变”。

在单耦合近似下，常写为：
`d g / d ln(mu) = beta(g)`。

它把“同一个理论在不同观察尺度下参数不同”变成可积的常微分方程问题，是量子场论连接高能与低能现象的核心工具。

## R02

本条目要解决的问题：

- 给出一个可运行、可审计的 RGE 最小数值框架；
- 在同一脚本内演示两类典型流动：
1. QCD 型：`beta < 0`，耦合随能标升高而减小（渐近自由）；
2. QED 型：`beta > 0`，耦合随能标升高而增大（Landau pole 倾向）；
- 用两圈 QCD (`N_f=16`) 计算非平凡固定点（Banks-Zaks）并做符号检查。

## R03

`demo.py` 的输入输出约定（无交互）：

- 输入（脚本内固定）：
1. 参考能标 `mu_ref=2 GeV`；
2. QCD 参考耦合 `alpha_qcd_ref=0.30`，`N_f=5`；
3. QED 参考耦合 `alpha_qed_ref=1/137`，`N_f=1`；
4. 能标网格 `mu_ref * geomspace(1, 1e4, 18)`；
5. 固定点分析使用 QCD 两圈 `N_f=16`。
- 输出：
1. QCD/QED beta 系数汇总表；
2. QCD/QED 跑动耦合对照表（解析 vs 数值）；
3. `N_f=16` Banks-Zaks 固定点及稳定性判定；
4. 断言通过后输出 `All checks passed.`。

## R04

本 MVP 使用的数学模型：

1. 统一写法（单耦合）：
`d g / d ln(mu) = c1 * g + c2 * g^2 + c3 * g^3`。

2. QCD（`SU(3)`）常用系数：
`beta0 = 11 - 2N_f/3`，`beta1 = 102 - 38N_f/3`，
`d alpha_s / d ln(mu) = -beta0 * alpha_s^2/(2pi) - beta1 * alpha_s^3/(4pi^2)`。

3. QED 一圈近似：
`d alpha / d ln(mu) = (2N_f / 3pi) * alpha^2`。

4. 若 `c1=c3=0`，一圈解析解：
`g(mu) = g_ref / [1 - c2 * g_ref * ln(mu/mu_ref)]`。

5. 固定点由 `beta(g*)=0` 给出；稳定性由 `beta'(g*)` 符号判定。

## R05

复杂度（网格点数为 `M`）：

- 单次 ODE 积分（固定维度一阶方程）：`O(M)` 到 `O(M * step_factor)`；
- 解析解与误差统计：`O(M)`；
- 固定点求解（低阶多项式）：`O(1)`；
- 空间复杂度：`O(M)`。

主开销是 `solve_ivp` 在 `ln(mu)` 网格上的积分。

## R06

算法流程：

1. 定义统一 beta 多项式模型 `c1,c2,c3`；
2. 构造 QCD 一圈/两圈、QED 一圈参数；
3. 在 `ln(mu)` 变量下用 `solve_ivp` 积分得到数值流；
4. 对可解析的一圈模型计算闭式解；
5. 计算“数值 vs 解析”误差；
6. 计算并分类固定点（特别是 `N_f=16` 两圈 QCD）；
7. 输出表格并用断言验证物理方向性。

## R07

优点：

- 一个统一框架覆盖 QCD 与 QED 两类 RG 行为；
- 数值积分与解析解互相校验，结果可审计；
- 直接包含固定点与稳定性分析。

局限：

- 仅是单耦合 MVP，未处理多耦合耦联 RG；
- 未加入阈值匹配与方案依赖高阶细节；
- 仅作机制演示，不替代高精度现象学拟合。

## R08

前置知识与环境：

- 量子场论中的 beta 函数、跑动耦合、固定点；
- 常微分方程数值积分；
- Python `>=3.10`；
- 依赖：`numpy`, `pandas`, `scipy`。

## R09

适用场景：

- 课程或组会中演示 RGE 的最小闭环；
- 快速检查给定 beta 系数的流动方向与固定点；
- 为更高阶或多耦合 RG 代码做 baseline。

不适用场景：

- 要求高精度实验拟合（例如完整 `alpha_s` 标准化流程）；
- 需要非微扰区域（格点、Schwinger-Dyson 等）结论；
- 需要完整阈值分段、scheme 变换的精细分析。

## R10

正确性直觉：

1. `beta(g)` 的符号直接决定 `mu` 增大时耦合的增减；
2. 一圈模型有闭式解，可作为数值积分基准；
3. 固定点满足 `beta(g*)=0`，其邻域稳定性由 `beta'(g*)` 决定；
4. QCD 的 `beta0>0` 对应小耦合区 `beta<0`（渐近自由），QED 一圈 `beta>0` 对应反向趋势。

## R11

数值稳定策略：

- 始终在 `t=ln(mu)` 空间积分，避免跨数量级直接积分；
- 校验 `mu_grid` 正值且严格递增；
- 在 RHS 中对耦合施加微小正下界，避免浮点下溢到非物理值；
- 显式检查 `solve_ivp.success`；
- 对一圈解析可用情形做相对误差阈值断言。

## R12

关键参数与影响：

- `N_f`：改变 QCD/QED beta 系数，并可能改变固定点结构；
- `alpha_ref`：决定从参考能标出发的轨道；
- `mu_ref` 与 `mu_grid`：决定观察区间；
- `c1,c2,c3`：决定流形形态与固定点位置；
- `rtol/atol`：影响数值精度与运行时间。

调参建议：

- 先加密 `mu_grid` 再收紧容差；
- 观察 Landau pole 趋势时避免把网格推进到分母过小区间；
- 固定点分析可优先检查 `c2,c3` 符号再做数值扫描。

## R13

- 近似比保证：N/A（非组合优化问题）。
- 随机成功率保证：N/A（全流程确定性）。

可验证保证：

- QCD 一圈数值解与一圈解析解一致（误差在阈值内）；
- QCD 耦合随 `mu` 升高单调减小；
- QED 一圈耦合随 `mu` 升高单调增大；
- `N_f=16` 两圈 QCD 给出正的 Banks-Zaks 固定点且 `beta(g*)≈0`。

## R14

常见失效模式：

1. `mu_grid` 非正或不递增导致 `ln(mu)` 不合法；
2. 一圈解析分母接近 0（Landau pole 邻域）导致数值爆炸；
3. beta 系数归一化写错（`2pi/4pi^2` 因子错误）；
4. ODE 失败却未检查 `success`；
5. 把固定点“存在性”与“稳定性”混淆。

## R15

可扩展方向：

- 多耦合 RG（例如 `g, y, lambda` 联立）；
- 阈值匹配与分段 `N_f` 跑动；
- 三圈/四圈 beta 系数；
- 增加参数扫描与 CSV 输出；
- 接入不确定度传播与误差带可视化。

## R16

相关条目：

- 渐近自由（Asymptotic Freedom）；
- Banks-Zaks 固定点；
- 跑动耦合常数；
- Callan-Symanzik 方程；
- 临界现象中的 RG 固定点与临界指数。

## R17

`demo.py` 交付能力：

- 显式实现单耦合 beta 多项式 RGE；
- 提供 QCD/QED 参数化示例；
- 对一圈模型给出解析解并与数值对照；
- 输出固定点与稳定性分析；
- 无交互、单命令可运行。

运行：

```bash
cd Algorithms/物理-量子场论-0408-重整化群方程_(Renormalization_Group_Equations)
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程（8 步）：

1. `qcd_coefficients` 与 `qed_one_loop_coefficient` 生成物理模型所需 beta 系数。  
2. `beta_polynomial` 按 `c1*g + c2*g^2 + c3*g^3` 计算局部 RG 流速。  
3. `integrate_rge` 把自变量改写成 `t=ln(mu)`，调用 `solve_ivp` 积分并检查求解状态。  
4. `one_loop_closed_form` 在 `c1=c3=0` 条件下给出闭式解，用于数值校验。  
5. `fixed_points` 通过解析求根得到全部实固定点（含 `g=0`）。  
6. `classify_fixed_point` 计算 `beta'(g*)`，给出 UV 方向的吸引/排斥判定。  
7. `main` 同步计算 QCD 与 QED 的数值轨道、解析轨道、误差和固定点汇总表。  
8. `main` 运行断言（单调性、误差阈值、Banks-Zaks 根条件），全部通过后输出 `All checks passed.`。  

说明：`scipy.solve_ivp` 仅承担通用 ODE 步进；物理系数、方程形式、固定点判定和验证标准都在源码中显式实现。
