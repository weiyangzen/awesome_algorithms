# 限制性三体问题 (Restricted Three-Body Problem)

- UID: `PHYS-0124`
- 学科: `物理`
- 分类: `天体力学`
- 源序号: `124`
- 目标目录: `Algorithms/物理-天体力学-0124-限制性三体问题_(Restricted_Three-Body_Problem)`

## R01

限制性三体问题研究两个大质量主天体与一个“可忽略质量”的第三天体的动力学。第三天体不反过来影响两主天体，因此模型比一般三体问题更可计算，但仍保留非线性、混沌、共振等关键现象。

本目录 MVP 采用平面圆限制性三体问题（Planar Circular RTBP, CR3BP）的旋转坐标系形式，完成：
- 两组初值轨迹传播；
- Jacobi 常数守恒漂移评估；
- 主天体最小距离与零速度关系检查；
- 非交互、可直接复现实验输出。

## R02

问题设定（归一化单位制）：
- 两主天体总质量归一化为 1；
- 两主天体距离归一化为 1；
- 坐标系角速度归一化为 1；
- 质量参数 `mu = m2/(m1+m2)`，其中 `0 < mu < 0.5`；
- 主天体位置固定为 `(-mu, 0)` 与 `(1-mu, 0)`。

第三天体状态向量为：
`[x, y, vx, vy]`。

目标是给定初值后，积分得到 `t in [0, t_final]` 的轨道，并输出动力学一致性指标。

## R03

为何采用该 MVP 形式：
- 旋转系下主天体静止，便于直接验证几何与守恒关系；
- 平面模型比三维模型更小，但已覆盖 CR3BP 的核心难点；
- 只依赖 `numpy + scipy + pandas`，便于复现与自动验收；
- 重点放在“方程显式实现 + 可审计指标”，而不是黑箱轨道库。

## R04

核心公式如下。

1. 两主天体距离：
`r1 = sqrt((x+mu)^2 + y^2)`，`r2 = sqrt((x-1+mu)^2 + y^2)`。

2. 有效势函数：
`Omega(x,y) = (1-mu)/r1 + mu/r2 + 0.5*(x^2+y^2)`。

3. 旋转系运动方程：
`x_dot = vx`
`y_dot = vy`
`vx_dot = 2*vy + dOmega/dx`
`vy_dot = -2*vx + dOmega/dy`

其中：
`dOmega/dx = x - (1-mu)(x+mu)/r1^3 - mu(x-1+mu)/r2^3`
`dOmega/dy = y - (1-mu)y/r1^3 - mu y/r2^3`

4. Jacobi 常数：
`C = 2*Omega - (vx^2 + vy^2)`。

## R05

算法流程（高层）：
1. 校验参数与初值合法性（`mu`、时间窗、积分容差、状态维度）。
2. 根据 CR3BP 方程构造 RHS（显式写出 `dOmega/dx, dOmega/dy`）。
3. 构造两主天体的碰撞保护事件（距离小于 guard 半径即终止）。
4. 调用 `solve_ivp(method="DOP853")` 对两组初值分别积分。
5. 计算每条轨迹的 `C(t)`、最小 `r1/r2`、速度上界、Poincare 风格过零计数。
6. 用 `2*Omega - C0` 与 `vx^2+vy^2` 的差值检查零速度关系一致性。
7. 汇总为表格并执行质量门禁（守恒漂移、最小距离、事件终止等）。
8. 通过则输出 `All checks passed.`。

## R06

正确性依据：
- 方程层面：RHS 与 CR3BP 标准旋转系模型一一对应；
- 守恒层面：Jacobi 常数应近似不变，漂移可衡量积分可信度；
- 几何层面：`r1/r2` 显式跟踪，可防止轨迹潜入奇点附近；
- 物理层面：`2*Omega - C0` 应与速度平方一致，数值误差应很小；
- 工程层面：失败直接抛异常，避免“看起来运行但结果无效”的静默问题。

## R07

设时间采样点为 `N`，自适应求解总步数为 `K`。

- 数值积分主成本近似与 `K` 成正比；
- 后处理（Jacobi、距离、统计）均为 `O(N)`；
- 存储整条轨迹状态为 `O(N)` 空间。

在本 MVP 规模（数千采样点、两条轨迹）下，运行成本很低，适合作为基线验证脚本。

## R08

边界与异常处理：
- `mu` 必须在 `(0, 0.5)`；
- 积分容差和时间长度必须为正；
- 初值必须是长度为 4 的有限实数向量；
- 若求解器失败或无输出点，立即抛出 `RuntimeError`；
- 若触发碰撞保护事件（`terminated_early=1`），在质量门禁阶段判为失败。

## R09

MVP 取舍：
- 仅实现平面圆限制性三体问题，不覆盖椭圆限制性三体或一般 N 体；
- 不做绘图，仅输出可审计的数值表；
- 不引入复杂任务设计（Halo/Lissajous 边值求解），先保证基础传播链路可靠；
- `solve_ivp` 仅作通用 ODE 数值器，动力学方程、守恒量和校验逻辑都在源码中显式实现。

## R10

`demo.py` 函数分工：
- `validate_config` / `validate_initial_state`：参数与输入检查；
- `distances_to_primaries` / `effective_potential`：几何与势函数计算；
- `grad_effective_potential`：计算 `dOmega/dx, dOmega/dy`；
- `planar_cr3bp_rhs`：封装状态导数；
- `make_collision_event`：构建碰撞守卫事件；
- `propagate_case`：执行单案例积分；
- `jacobi_constant` / `count_upward_y_crossings`：动力学诊断；
- `summarize_case` / `sample_trajectory_table`：汇总指标与采样表；
- `assert_quality`：验收门禁；
- `main`：组织案例、打印结果、触发最终通过/失败。

## R11

运行方式：

```bash
cd Algorithms/物理-天体力学-0124-限制性三体问题_(Restricted_Three-Body_Problem)
uv run python demo.py
```

脚本无交互输入，执行后会自动输出两条轨迹的摘要与采样表。

## R12

输出字段说明（`Summary metrics`）：
- `case`：案例名称；
- `C0`：初始 Jacobi 常数；
- `max_jacobi_drift`：`max |C(t)-C0|`；
- `min_r1`, `min_r2`：与两主天体最小距离；
- `max_speed`：轨迹最大速度；
- `min_zvc_margin`：`min(2*Omega-C0)`，理论上应非负；
- `max_kinetic_recon_error`：`max |(2*Omega-C0)-(vx^2+vy^2)|`；
- `upward_y_crossings`：从 `y<=0` 到 `y>0` 且 `vy>0` 的过零次数；
- `terminated_early`：是否触发碰撞终止事件；
- `steps`：实际输出步数。

## R13

最小验证项（脚本自动执行）：
- `max_jacobi_drift <= 1e-7`；
- `min_r1` 与 `min_r2` 均大于 `guard` 半径；
- `min_zvc_margin >= -1e-7`；
- `terminated_early == 0`。

若任意条件不满足，脚本抛出异常并返回失败。

## R14

关键参数与调参建议：
- `mu`：系统质量比，Earth-Moon 约 `0.0121505856`；
- `t_final`：积分时长，增大可观察更丰富动力学但漂移累计会增大；
- `samples`：输出采样点数，仅影响输出分辨率，不直接控制求解内部步长；
- `rtol/atol`：积分精度，越严格越稳但成本更高；
- `min_distance_guard`：奇点防护阈值，过大可能过早终止，过小可能放大数值风险。

## R15

与其他实现路径对比：
- 纯解析法：CR3BP 一般无闭式轨道解，不适用完整传播；
- 直接黑箱轨道库：开发快，但很难审计守恒与方程一致性；
- 本方案（显式方程 + 通用 ODE 求解 + 守恒验收）：
  - 优点：可解释、可诊断、工程上可回归；
  - 缺点：未利用辛结构，超长时间积分可能不如专用辛积分器保守。

## R16

应用场景：
- 拉格朗日点附近任务的初步轨迹筛选；
- 共振转移与低能转移的教学演示基线；
- 研究零速度曲线可达域的数值实验；
- 更复杂轨道设计工具前的“方程级 sanity check”。

## R17

可扩展方向：
- 增加三维 CR3BP（含 `z, vz`）及对应线性化分析；
- 增加 Poincare 截面采样输出与周期轨道初值搜索；
- 使用辛积分器对比长期 Jacobi 漂移表现；
- 接入 `L1/L2/L4/L5` 附近目标轨道（Lyapunov/Halo）求解与 continuation；
- 批量扫描不同 `mu` 与初值，输出稳定性统计数据集。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 创建 `CR3BPConfig`，调用 `validate_config` 保证参数在物理与数值上可用。  
2. `run_demo` 生成两组确定性初值（近 L4 与内区穿越型），逐条调用 `propagate_case`。  
3. `propagate_case` 构建 `t_eval`、碰撞事件函数，并把 `planar_cr3bp_rhs` 传给 `solve_ivp(DOP853)`。  
4. `planar_cr3bp_rhs` 每一步先由 `grad_effective_potential` 计算势函数梯度，再按 CR3BP 方程返回 `[x_dot,y_dot,vx_dot,vy_dot]`。  
5. 求解完成后，`summarize_case` 计算 `C(t)`、最小主星距离、速度上界和上穿越次数等核心诊断。  
6. 同时计算 `2*Omega-C0` 与 `vx^2+vy^2` 的最大差值，验证零速度关系在数值上闭合。  
7. `sample_trajectory_table` 从整条轨迹抽样成表，输出关键时刻状态与 `C` 值，便于人工审计。  
8. `assert_quality` 执行门禁阈值；若全部通过，脚本打印 `All checks passed.`，否则抛异常终止。  

第三方库没有被当作黑盒算法：`solve_ivp` 仅承担通用 ODE 步进，问题建模、方程构造、守恒量定义与验收判据都在源码中显式展开。
