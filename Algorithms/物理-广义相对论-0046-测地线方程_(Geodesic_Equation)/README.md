# 测地线方程 (Geodesic Equation)

- UID: `PHYS-0046`
- 学科: `物理`
- 分类: `广义相对论`
- 源序号: `46`
- 目标目录: `Algorithms/物理-广义相对论-0046-测地线方程_(Geodesic_Equation)`

## R01

问题定义：实现一个可运行、可验证的广义相对论测地线方程最小 MVP。

本条目聚焦史瓦西时空赤道面（`theta = pi/2`）下的类时测地线，显式写出克里斯托费尔符号并进行数值积分，而不是仅调用黑箱轨道模块。

## R02

物理背景：在广义相对论中，自由粒子满足

`d2x^mu/dtau2 + Gamma^mu_{alpha beta} (dx^alpha/dtau)(dx^beta/dtau) = 0`

其中 `tau` 为固有时，`Gamma^mu_{alpha beta}` 由度规导出。测地线方程把“引力”转化为“时空几何导致的惯性运动”。

## R03

MVP 模型设定：
- 度规：史瓦西度规；
- 单位：几何化单位 `G = c = 1`；
- 质量参数：`M = 1`；
- 运动平面：赤道面 `theta = pi/2`；
- 状态变量：`y = [t, r, phi, u^t, u^r, u^phi]`。

脚本对两个初值案例积分：
1. `angular_scale = 1.0`（精确圆轨道基线）；
2. `angular_scale = 0.97`（轻微偏离圆轨道，展示非平凡径向变化）。

## R04

史瓦西赤道面下使用的关键克氏符（`r > 2M`）：
- `Gamma^t_{tr} = Gamma^t_{rt} = M / (r (r - 2M))`
- `Gamma^r_{tt} = (1 - 2M/r) * M / r^2`
- `Gamma^r_{rr} = -M / (r (r - 2M))`
- `Gamma^r_{phi phi} = -(r - 2M)`
- `Gamma^phi_{r phi} = Gamma^phi_{phi r} = 1/r`

对应一阶系统：
- `dt/dtau = u^t`
- `dr/dtau = u^r`
- `dphi/dtau = u^phi`
- `du^mu/dtau = -Gamma^mu_{alpha beta} u^alpha u^beta`

## R05

初值构造与约束：
- 圆轨道参考角速度由解析式给出（`r0 > 3M`）：
  - `u^t_circ = 1/sqrt(1 - 3M/r0)`
  - `u^phi_circ = sqrt(M/r0^3) / sqrt(1 - 3M/r0)`
- 实际案例采用 `u^phi = angular_scale * u^phi_circ`；
- 用类时归一化约束求 `u^t`：
  - `g_{mu nu} u^mu u^nu = -1`

脚本默认 `r0 = 10M`，处于稳定圆轨道区（`r > 6M`）。

## R06

数值算法主流程：
1. 根据 `r0` 与 `angular_scale` 生成初始四速度；
2. 显式构造 geodesic RHS（不依赖黑箱轨道方程）；
3. 用 `solve_ivp(method="DOP853")` 积分 `tau in [0, tau_end]`；
4. 事件函数监控是否接近视界（`r -> 2M`）；
5. 在均匀网格上重采样解；
6. 计算守恒量与残差指标；
7. 输出表格与阈值检查，给出 `PASS/FAIL`。

## R07

复杂度分析：
- 设积分器实际步数为 `N_step`，后处理采样点为 `N_sample`，案例数为 `K`；
- 时间复杂度约 `O(K * (N_step + N_sample))`；
- 空间复杂度约 `O(N_sample)`（每个案例保留重采样轨迹用于诊断）。

在教学级和中小规模参数扫描下，该复杂度足够轻量。

## R08

数值稳定性与诊断策略：
- 使用高阶显式 RK：`DOP853`，容差 `rtol=1e-10, atol=1e-12`；
- 限制最大步长 `max_step=0.2`，抑制快速相位段误差；
- 设置视界事件终止，避免跨过坐标奇点；
- 双重质量检查：
  - 守恒量漂移（`E`、`L`、`u.u`）；
  - 测地方程加速度残差 RMS（有限差分 vs 模型 RHS）。

## R09

适用边界：
- 适用于：史瓦西背景、赤道面运动、单体中心场、教学与算法验证。
- 不适用于：
  - Kerr 自旋黑洞；
  - 多体引力场；
  - 真实观测拟合（需更完整天体模型与测量误差传播）。

## R10

MVP 技术栈：
- Python 3
- `numpy`：向量化与数值后处理
- `scipy`：`solve_ivp` 常微分方程积分
- `pandas`：结果表格输出
- 标准库 `dataclasses`

核心物理方程和克氏符均在源码中显式展开，不是黑箱封装。

## R11

运行方式：

```bash
cd Algorithms/物理-广义相对论-0046-测地线方程_(Geodesic_Equation)
uv run python demo.py
```

脚本无交互输入，直接打印案例结果和验证结论。

## R12

一次实测输出（默认参数）核心结果：
- `scale=1.000`（圆轨道）
  - `r_min = r_max = 10.0`（数值上保持圆轨道）
  - `geodesic_residual_rms ≈ 2.34e-16`
- `scale=0.970`（轻微偏离）
  - `r_min ≈ 7.990679, r_max = 10.0`
  - `energy_rel_drift ≈ 4.78e-15`
  - `ang_mom_rel_drift ≈ 6.18e-15`
  - `norm_abs_drift ≈ 9.77e-15`

阈值检查全部通过，输出 `Validation: PASS`。

## R13

正确性检查（脚本内置）：
1. 所有案例积分成功，且未触发视界事件；
2. 相对能量漂移 `< 1e-9`；
3. 相对角动量漂移 `< 1e-9`；
4. 四速度归一化误差 `max |u.u + 1| < 1e-9`；
5. 测地方程残差 `RMS < 2e-7`；
6. 圆轨道案例径向跨度 `< 2e-6`。

## R14

当前实现局限：
- 固定在赤道面，未演示 `theta` 自由度耦合；
- 使用史瓦西坐标，未切换到穿越视界更平滑的坐标系；
- 未做高阶物理效应（自旋、辐射反作用、非真空背景）；
- 残差检验中的二阶导来自有限差分，受采样密度影响。

## R15

可扩展方向：
- 扩展到 4D 全变量 `t,r,theta,phi`；
- 切换 Kerr 度规，研究自旋引发的轨道进动；
- 对多个初始条件批量扫描，绘制守恒漂移热图；
- 引入辛积分器，对长期轨道稳定性做更严格比较。

## R16

应用场景：
- 广义相对论课程中的测地线数值实验；
- 黑洞附近轨道动力学的原型验证；
- 引力透镜与时延计算前的几何运动基础模块；
- 数值相对论工具链中的“单粒子轨道 sanity check”。

## R17

本目录交付内容：
- `README.md`：R01-R18 全部完成；
- `demo.py`：可直接运行的测地线方程最小实现；
- `meta.json`：保持 `PHYS-0046 / 广义相对论 / source 46` 一致。

脚本默认参数可直接复现实验，不依赖命令行输入。

## R18

`demo.py` 源码级算法流拆解（9 步）：
1. `circular_orbit_rates` 给出圆轨道参考 `u^t_circ, u^phi_circ`，建立物理量纲基线。
2. `solve_ut_from_normalization` 用 `g_{mu nu}u^mu u^nu=-1` 反解初始 `u^t`，确保初值物理可行。
3. `christoffel_equatorial` 在每个 `r` 上显式计算所需克氏符，不依赖外部 GR 黑箱。
4. `geodesic_rhs` 把二阶测地线方程改写为 6 维一阶系统 `d/dtau [x,u]`。
5. `solve_ivp(DOP853)` 对一阶系统积分，并通过 `build_horizon_event` 监控 `r -> 2M` 风险。
6. `compute_invariants` 从轨迹计算 `E=f u^t`、`L=r^2 u^phi`、`kappa=g(u,u)` 作为守恒诊断。
7. `geodesic_residual_rms` 用有限差分近似 `du/dtau`，与模型加速度逐点比较得到 RMS 残差。
8. `validate_results` 统一执行守恒漂移、残差和圆轨道径向跨度阈值判定。
9. `main` 汇总为 `pandas` 表格并输出 `Validation: PASS/FAIL`，形成可审计最小闭环。

说明：虽然使用了 `scipy.solve_ivp` 作为积分器，但度规相关建模、克氏符构造、守恒验证与残差判定全部在源码层透明展开。
