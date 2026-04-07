# 黑洞热力学 (Black Hole Thermodynamics)

- UID: `PHYS-0377`
- 学科: `物理`
- 分类: `广义相对论`
- 源序号: `396`
- 目标目录: `Algorithms/物理-广义相对论-0396-黑洞热力学_(Black_Hole_Thermodynamics)`

## R01

黑洞热力学研究事件视界几何量与热力学量之间的一一对应关系。  
对非旋转、无电荷的 Schwarzschild 黑洞，质量 `M` 决定全部宏观态：

- 视界半径 `r_s = 2GM/c^2`；
- 视界面积 `A = 4πr_s^2`；
- Bekenstein-Hawking 熵 `S = k_B c^3 A/(4Għ)`；
- Hawking 温度 `T = ħ c^3/(8πGk_B M)`。

这使黑洞从“纯几何对象”变成“可热力学描述的物理系统”。

## R02

本条目要解决的问题是：如何用最小可运行脚本，数值验证黑洞热力学中的三个核心结论：

1. `T` 随 `M` 增大而降低（负热容特征）；
2. `S` 随 `M` 增大而增加；
3. 蒸发方程 `dM/dt = -alpha/M^2` 的数值积分与解析解一致。

MVP 目标：

- 计算 Schwarzschild 黑洞的 `r_s, A, T, S, C`；
- 检查第一定律 `d(Mc^2)=T dS`；
- 用 `solve_ivp` 积分 Hawking 蒸发并与解析曲线对照。

## R03

`demo.py` 的输入输出约定（无交互）：

- 输入（代码内固定）：
1. 一组黑洞质量（kg）；
2. 蒸发演示初始质量 `M0=1e8 kg`；
3. 蒸发时间采样点 `t/tau in [0, 0.9]`。
- 输出：
1. 黑洞热力学量表格（`pandas`）；
2. 第一定律相对误差表格；
3. 蒸发数值解 vs 解析解对照表；
4. 断言通过后打印 `All checks passed.`。

## R04

MVP 使用的数学模型（SI 单位）：

1. 事件视界半径：`r_s = 2GM/c^2`。
2. 视界面积：`A = 4πr_s^2 = 16πG^2M^2/c^4`。
3. Hawking 温度：`T = ħc^3/(8πGk_BM)`。
4. Bekenstein-Hawking 熵：`S = k_B c^3 A/(4Għ) = 4πk_BGM^2/(ħc)`。
5. Schwarzschild 热容：`C = dE/dT = -8πk_BG M^2/(ħc) < 0`。
6. Hawking 辐射功率近似：`P = ħc^6/(15360πG^2M^2)`。
7. 质量演化：`dM/dt = -P/c^2 = -alpha/M^2`，其中 `alpha = ħc^4/(15360πG^2)`。
8. 蒸发解析解：`M(t) = (M0^3 - 3alpha t)^(1/3)`，寿命 `tau = M0^3/(3alpha)`。

## R05

复杂度分析（采样点数记为 `N`）：

- 热力学量向量化计算：`O(N)`；
- 第一定律误差计算：`O(N)`；
- 蒸发 ODE 积分：约 `O(N * step_factor)`；
- 空间复杂度：`O(N)`。

MVP 成本主要来自 `solve_ivp` 时间积分。

## R06

算法闭环（本条目实现流程）：

1. 读入质量数组并计算 `r_s, A, T, S, C`；
2. 用中心差分估计 `dS/dM` 并检查 `T*dS/dM ≈ c^2`；
3. 构造蒸发常数 `alpha` 和寿命 `tau`；
4. 数值积分 `dM/dt = -alpha/M^2`；
5. 计算同一时间网格下解析解；
6. 对比相对误差并执行断言；
7. 输出可审计表格和检查结论。

## R07

优点：

- 方程与代码一一对应，公式透明；
- 同时有解析解和数值解，便于交叉验证；
- 直接展示黑洞负热容与蒸发特性。

局限：

- 只覆盖 Schwarzschild 黑洞，不含自旋/电荷；
- 辐射功率使用简化常数，忽略灰体因子与粒子自由度细节；
- 没有引入量子引力修正与末态模型。

## R08

前置知识与运行环境：

- 广义相对论中的 Schwarzschild 解；
- 热力学第一定律与导数关系；
- 常微分方程基础；
- Python `>=3.10`；
- 依赖：`numpy`, `pandas`, `scipy`。

## R09

适用场景：

- 黑洞热力学入门教学；
- 检查公式量纲与数量级；
- 作为更复杂 Kerr/Reissner-Nordstrom 模型的基线。

不适用场景：

- 需要高精度灰体谱或量子场论曲时空全计算；
- 研究蒸发末期的普朗克尺度动力学；
- 直接用于天体观测拟合。

## R10

正确性直觉：

1. `S ∝ M^2`，所以黑洞越重熵越大；
2. `T ∝ 1/M`，所以黑洞越重温度越低；
3. `C < 0` 表示“越辐射越热”，导致加速蒸发；
4. 蒸发方程是可分离变量方程，解析解可作为数值积分基准；
5. 第一定律在 Schwarzschild 情形退化为 `d(Mc^2)=T dS`。

## R11

数值稳定策略：

- 对 `M` 施加正值约束，避免出现非物理负质量；
- 蒸发只积分到 `0.9*tau`，避开 `M->0` 奇点；
- ODE 使用较严格 `rtol/atol` 并检查 `solution.success`；
- 误差使用相对误差并设置下界，避免除零放大。

## R12

关键参数与影响：

- `mass_samples_kg`：决定热力学量覆盖范围；
- `M0_evap_kg`：决定蒸发寿命 `tau ∝ M0^3`；
- `num_time_points`：蒸发曲线分辨率；
- `rtol/atol`：决定数值积分精度与耗时；
- `delta_frac`（中心差分步长）：影响第一定律数值校验误差。

调参建议：

- 想看更平滑蒸发曲线时先增加采样点；
- 想要更严格解析对照时收紧 `rtol/atol`；
- 若要避免极端数量级，可把质量范围限制在 `1e6~1e12 kg`。

## R13

- 近似比保证：N/A（非近似优化问题）。
- 随机成功率保证：N/A（确定性流程，无随机项）。

可验证保证：

- `T(M)` 单调下降；
- `S(M)` 单调上升；
- `C(M)` 全部为负；
- 蒸发 ODE 数值解与解析解相对误差在阈值内；
- 寿命比例满足 `tau(2M)/tau(M) ≈ 8`。

## R14

常见失效模式：

1. 误把 `ħ` 写成 `h`，导致数量级错误；
2. 把 `E=M` 与 `E=Mc^2` 混淆，第一定律校验失败；
3. 积分接近 `t=tau` 时步长控制不当导致误差放大；
4. 质量单位混用（kg 与太阳质量）导致结果失真；
5. 忽略 `solution.success` 直接使用 ODE 输出。

## R15

工程扩展方向：

- 扩展到 Kerr 黑洞：加入角速度项 `Omega dJ`；
- 扩展到带电黑洞：加入电势项 `Phi dQ`；
- 引入灰体因子和粒子物种计数，改进蒸发功率模型；
- 做参数扫描并导出 CSV/图像用于教学报告。

## R16

相关条目：

- Bekenstein bound；
- Hawking radiation；
- Unruh effect；
- Schwarzschild metric；
- 重力、热力学与信息悖论相关问题。

## R17

`demo.py` 交付能力清单：

- 显式实现 Schwarzschild 黑洞热力学基本量；
- 计算并验证第一定律数值一致性；
- 积分 Hawking 蒸发方程并对照解析解；
- 输出结构化表格用于快速审计；
- 无交互，单命令可运行。

运行方式：

```bash
cd Algorithms/物理-广义相对论-0396-黑洞热力学_(Black_Hole_Thermodynamics)
uv run python demo.py
```

## R18

`demo.py` 的源码级算法流程（9 步）：

1. `schwarzschild_radius` 计算 `r_s=2GM/c^2`，建立几何尺度。  
2. `horizon_area` 基于 `r_s` 计算视界面积 `A`。  
3. `hawking_temperature` 用 `T=ħc^3/(8πGk_BM)` 得到温度。  
4. `bekenstein_hawking_entropy` 用 `S=k_B c^3 A/(4Għ)` 得到熵。  
5. `schwarzschild_heat_capacity` 计算负热容 `C=dE/dT`。  
6. `first_law_relative_error` 用中心差分估计 `dS/dM`，检查 `T*dS/dM` 是否等于 `c^2`。  
7. `integrate_evaporation` 明确写出 `dM/dt=-alpha/M^2` 后调用 `solve_ivp` 积分（求解器不是黑盒物理模型）。  
8. `evaporation_mass_analytic` 计算解析解 `M(t)=(M0^3-3alpha t)^(1/3)`，与数值解逐点对照。  
9. `main` 汇总表格、执行单调性与误差断言，全部通过后输出 `All checks passed.`。

说明：`scipy.integrate.solve_ivp` 只负责通用 ODE 时间推进；关键物理关系、常数、验证指标均在源码中显式实现。
