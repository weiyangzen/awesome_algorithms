# 量子电动力学 (Quantum Electrodynamics, QED)

- UID: `PHYS-0062`
- 学科: `物理`
- 分类: `量子场论`
- 源序号: `62`
- 目标目录: `Algorithms/物理-量子场论-0062-量子电动力学_(Quantum_Electrodynamics,_QED)`

## R01

量子电动力学（QED）是描述带电轻子与电磁场相互作用的阿贝尔规范场论。  
在重整化群视角下，QED 耦合常数 `alpha(mu)` 会随着能标 `mu` 增大而缓慢上升，这与 QCD 的渐近自由相反。  
本条目聚焦“QED 一圈跑动耦合”的最小数值可验证实现。

## R02

本条目要解决的核心问题是：  
如何用最小可运行脚本，验证 QED 在一圈近似下的三个关键性质：

1. beta 函数为正，`alpha(mu)` 在紫外（更高能标）增大；
2. 一圈 ODE 数值积分与一圈解析解一致；
3. Landau pole 位置可由解析式直接估计，且在常见参考点下远高于演示能区。

MVP 目标：

- 实现 `d alpha / d ln(mu) = b1 * alpha^2`；
- 用 `solve_ivp` 进行数值积分，并与闭式解比对；
- 打印系数表、跑动表和 Landau pole 量级；
- 用断言检验趋势与误差。

## R03

`demo.py` 的输入输出约定（无交互）：

- 输入（脚本内固定）：
1. 参考点：`mu_ref = m_e = 0.00051099895 GeV`，`alpha_ref = 1/137.035999084`；
2. 基线场景：`sum_q2 = 1`（仅电子，toy model）；
3. 对照场景：`sum_q2 = 3`（三轻子全激活，toy model）；
4. 能标网格：`mu_ref * geomspace(1, 1e6, 16)`。
- 输出：
1. 不同场景的 `sum_q2, b1, log10(mu_pole/GeV)`；
2. 基线场景 `alpha(mu)` 的闭式解与数值解对照表；
3. 自动断言后输出 `All checks passed.`。

## R04

采用的一圈数学模型（自然单位，`hbar=c=1`）：

1. QED beta 函数（固定活跃费米子集合）：
`mu d alpha / d mu = b1 * alpha^2`，其中  
`b1 = (2/(3pi)) * sum_i Q_i^2`。

2. 写成 `t = ln(mu)` 的常微分方程：
`d alpha / dt = b1 * alpha^2`。

3. 对参考点 `(mu_ref, alpha_ref)` 的一圈闭式解：
`alpha(mu) = alpha_ref / (1 - alpha_ref*b1*ln(mu/mu_ref))`。

4. Landau pole 估计：
`mu_pole = mu_ref * exp(1/(alpha_ref*b1))`。

当 `mu -> mu_pole^-` 时分母趋零，微扰理论失效。

## R05

复杂度分析（网格点数 `M`）：

- 系数计算与 Landau pole 估计：`O(1)`；
- 一次一阶 ODE 积分：约 `O(M)`；
- 表格构造与误差统计：`O(M)`；
- 空间复杂度：`O(M)`。

主成本来自在 `ln(mu)` 网格上的 ODE 采样。

## R06

MVP 算法闭环：

1. 根据活跃费米子电荷平方和计算 `b1`；
2. 构造几何能标网格 `mu_grid`；
3. 用闭式公式计算 `alpha_closed(mu)`；
4. 在 `t=ln(mu)` 变量下数值积分 `alpha_numeric(mu)`；
5. 计算逐点误差与最大相对误差；
6. 估算 Landau pole `mu_pole`；
7. 与更大 `sum_q2` 场景做对照；
8. 断言趋势与误差，输出结果。

## R07

优点：

- 公式简单、代码短小，便于教学与审计；
- 同时提供解析解和数值解交叉验证；
- 通过 `sum_q2` 对照直观看出 beta 系数与 UV 增长速度关系。

局限：

- 仅一圈近似，未覆盖更高圈修正；
- 使用固定活跃味数 toy 模型，未做阈值匹配；
- 不直接替代精密电弱拟合。

## R08

前置知识与环境：

- 重整化群与 beta 函数基础；
- 一阶常微分方程与 `ln(mu)` 变量变换；
- Python `>=3.10`；
- 依赖：`numpy`, `pandas`, `scipy`。

## R09

适用场景：

- 量子场论课程中的 QED 跑动耦合演示；
- 快速验证“QED 非渐近自由（beta>0）”；
- 作为更复杂 RGE 代码的最小单元测试基线。

不适用场景：

- 需要高精度电弱参数拟合；
- 需要跨阈值严格处理（质量门槛、方案依赖）；
- 需要非微扰或强耦合区域结论。

## R10

正确性直觉：

1. 在一圈近似里 `d alpha / d ln(mu) = b1*alpha^2` 且 `b1>0`，故 `mu` 增大时 `alpha` 单调增大；
2. 该 ODE 可分离变量并得闭式解，因此数值解应与解析解贴合；
3. `sum_q2` 越大，`b1` 越大，耦合增长越快且 Landau pole 更早出现；
4. 若演示网格远小于 `mu_pole`，则计算应保持稳定且正值。

## R11

数值稳定策略：

- 在 `ln(mu)` 空间积分，避免跨多个数量级时直接以 `mu` 为自变量导致的精度损失；
- 限制 `mu_grid` 为正且严格递增；
- 对 ODE 状态设置正下界 `1e-14`，防止浮点噪声导致非物理值；
- 显式检查 `solve_ivp` 的 `success` 状态；
- 闭式解中若分母非正则立刻报错，阻止越过 Landau pole。

## R12

关键参数与影响：

- `sum_q2`：决定 `b1` 大小，是跑动速度主开关；
- `alpha_ref`：参考耦合，影响整体尺度与 Landau pole 位置；
- `mu_ref`：参考能标，决定 `mu_pole` 绝对量级；
- `mu_max_factor`：决定演示的 UV 范围；
- `rtol/atol`：影响数值积分与解析解比对精度。

调参建议：

- 做教学演示可增大 `mu_max_factor` 看更明显增长；
- 若误差不达标，优先收紧 `rtol/atol` 并适当加密网格；
- 若要更接近真实现象学，应加入阈值匹配和高圈修正。

## R13

- 近似比保证：N/A（非优化近似算法条目）。
- 随机成功率保证：N/A（全流程确定性计算）。

可验证保证（由断言实现）：

1. 一圈数值解与闭式解最大相对误差 `< 1e-6`；
2. 基线场景 `alpha(mu_end) > alpha(mu_start)`（UV 增长）；
3. 演示区间上限低于估算 Landau pole；
4. 更大 `sum_q2` 场景具有更快增长和更低 pole。

## R14

常见失效模式：

1. `mu_grid` 非正或非递增，`ln(mu)` 非法；
2. 输入 `alpha_ref<=0` 或空电荷集合，导致模型不成立；
3. 请求能区跨越 Landau pole，闭式解分母为零或负；
4. 忽略 `solve_ivp` 成功标记，可能掩盖积分失败；
5. 误把 `b1` 系数写错（漏掉 `2/(3pi)` 因子）。

## R15

工程扩展方向：

- 加入两圈及更高圈 QED beta 修正；
- 做分段阈值匹配（电子/缪子/τ 激活区间）；
- 与 `MSbar` 方案下文献曲线做数值比较；
- 输出 CSV/图像用于课程报告或自动化回归测试。

## R16

相关条目：

- 重整化群方程（RGE）；
- 真空极化（vacuum polarization）；
- Landau pole；
- QCD 渐近自由（对照：beta 符号相反）。

## R17

`demo.py` 交付能力清单：

- 实现一圈 QED beta 系数计算；
- 实现一圈闭式解与数值 ODE 积分；
- 输出结构化表格（`pandas`）；
- 比较两种 `sum_q2` 场景的 UV 跑动与 Landau pole；
- 无交互，单命令运行并内置断言。

运行方式：

```bash
cd Algorithms/物理-量子场论-0062-量子电动力学_(Quantum_Electrodynamics,_QED)
uv run python demo.py
```

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `qed_one_loop_beta_coefficient` 读取活跃费米子电荷平方和 `sum_q2`，计算一圈系数 `b1=(2/(3pi))*sum_q2`。  
2. `beta_alpha_one_loop` 给出局部流速 `d alpha / d ln(mu) = b1*alpha^2`，并约束输入为正。  
3. `alpha_one_loop_closed_form` 用闭式解直接计算网格上的 `alpha(mu)`，同时检查是否跨越 Landau pole。  
4. `integrate_running_alpha` 在 `t=ln(mu)` 空间构造 ODE 初值问题，调用 `solve_ivp` 获得数值解。  
5. `landau_pole_scale` 根据 `(mu_ref, alpha_ref, b1)` 给出 `mu_pole` 解析估计。  
6. `analyze_scenario` 组织单场景完整流程：建网格、求解析解、求数值解、统计误差、汇总指标。  
7. `main` 构造基线与对照场景（`sum_q2=1` 和 `sum_q2=3`），打印系数表和基线跑动表。  
8. `main` 通过断言检查误差、单调性与 Landau pole 关系，全部通过后输出 `All checks passed.`。  

说明：`scipy.integrate.solve_ivp` 仅用于通用积分；beta 方程、物理参数、误差判据与一致性检查均在源码中显式实现。
