# 科里奥利力 (Coriolis Force)

- UID: `PHYS-0107`
- 学科: `物理`
- 分类: `经典力学`
- 源序号: `107`
- 目标目录: `Algorithms/物理-经典力学-0107-科里奥利力_(Coriolis_Force)`

## R01

科里奥利力用于描述“在旋转参考系内观察运动”时出现的惯性力修正项。  
本条目给出一个最小可运行 MVP，目标是把公式

`a_cor = -2 * (Omega x v)`

直接落到可执行代码，并用两个固定发射案例展示偏转方向与量级。

MVP 重点：
- 在地球局部 ENU（East-North-Up）坐标中构造 `Omega`；
- 用同一个积分器分别计算“有/无科里奥利项”的轨迹；
- 输出终点偏移、全程最大偏移和 `v·a_cor` 数值检查（验证科里奥利力不做功）。

## R02

问题定义（本目录实现）：
- 输入：
  - 纬度 `latitude_deg`；
  - 时间步长 `dt`；
  - 步数 `steps`；
  - 初始位置 `r0=[x0,y0,z0]`（ENU，单位 m）；
  - 初始速度 `v0=[vx0,vy0,vz0]`（ENU，单位 m/s）。
- 动力学方程（简化模型）：
  - `dr/dt = v`
  - `dv/dt = g + a_cor`
  - `a_cor = -2 * (Omega x v)`
- 输出：
  - 时间序列 `t`；
  - 位置序列 `r(t)`；
  - 速度序列 `v(t)`；
  - 科里奥利加速度序列 `a_cor(t)`；
  - 有/无科里奥利两条轨迹差分。

说明：本 MVP 把离心项吸收到常量重力 `g=[0,0,-9.81]` 中，不做地球曲率与高度变化建模。

## R03

旋转参考系中的常见分解（`Omega` 常量时）：

`a_rot = a_inertial - 2(Omega x v_rot) - Omega x (Omega x r)`

其中：
- `-2(Omega x v_rot)` 是科里奥利项；
- `-Omega x (Omega x r)` 是离心项。

本条目只显式保留科里奥利项，理由是：
- 目标是隔离并演示“速度相关偏转”这一核心机制；
- 在局部近地问题中，可把重力与离心项合并为常量有效重力近似；
- 更容易与“无科里奥利对照组”做差分解释。

局部 ENU 坐标下地球角速度向量：

`Omega_enu = [0, Omega*cos(phi), Omega*sin(phi)]`

其中 `phi` 为纬度。

## R04

算法流程（高层）：
1. 检查输入：`dt > 0`、`steps > 0`、向量维度与有限性。
2. 由纬度生成局部 `Omega_enu`；若关闭科里奥利，则置零向量。
3. 构造状态 `s=[x,y,z,vx,vy,vz]` 与常量重力向量。
4. 在每步中计算右端项：
   - `dx/dt = v`
   - `dv/dt = g - 2(Omega x v)`
5. 用 RK4 执行一步积分，推进到下一时刻。
6. 保存整段轨迹的 `t/r/v`。
7. 同参数再跑一遍“无科里奥利”轨迹。
8. 输出两条轨迹的终点差、最大差、速度差与 `max|v·a_cor|`。

## R05

核心数据结构（`numpy.ndarray`）：
- `state: shape (6,)`，按 `[x, y, z, vx, vy, vz]` 存储单时刻状态。
- `states: shape (steps+1, 6)`，整段仿真状态历史。
- `t: shape (steps+1,)`，时间网格。
- `positions: shape (steps+1, 3)`，位置轨迹。
- `velocities: shape (steps+1, 3)`，速度轨迹。
- `coriolis_acc: shape (steps+1, 3)`，科里奥利加速度历史。
- `delta = pos_with - pos_without`，有/无科里奥利差分轨迹。

使用连续数组而非对象链表，便于向量化诊断和后续可视化扩展。

## R06

正确性与物理一致性检查：
- 符号检查：
  - 北半球、纯北向高速初速度时，东向偏移应为正（向右偏）。
  - 中纬度、纯东向高速初速度时，出现南向与上向分量偏转。
- 能量相关检查：
  - 连续理论中 `v·a_cor = 0`，科里奥利力不做功；
  - 代码输出 `max|v·a_cor|`，应接近 0（仅存在积分误差量级残差）。
- 对照检查：
  - 同参数下“有/无科里奥利”差分应随时间累积，不应恒为零。

该三类检查分别覆盖方向、机理、数值实现三层正确性。

## R07

复杂度（`N = steps`）：
- 时间复杂度：`O(N)`  
  每步执行常数次向量运算和 4 次 RHS 评估（RK4）。
- 空间复杂度：`O(N)`  
  保存全轨迹 `states/t` 以及派生 `positions/velocities`。

若只关心终点，可改成滚动状态把空间降到 `O(1)`，但会失去轨迹审计能力。

## R08

边界与异常处理：
- `dt <= 0`、`steps <= 0`：抛 `ValueError`。
- `gravity_mps2 <= 0`：抛 `ValueError`。
- `latitude_deg` 不在 `[-90, 90]`：抛 `ValueError`。
- 初始位置/速度不是 3 维向量或含 `nan/inf`：抛 `ValueError`。
- 迭代中若状态出现非有限值：抛 `RuntimeError`。

这些约束保证了模型参数和数值状态都在可解释范围内。

## R09

MVP 取舍：
- 仅依赖 `numpy`，不调用 `scipy.integrate.solve_ivp` 黑盒积分器。
- 只做局部 ENU、常量重力近似，不引入球面几何与大气阻力。
- 用两个固定案例替代交互输入，保证 `uv run python demo.py` 可复现。
- 同时跑“有/无科里奥利”对照，直接给出可解释差值，而不是只输出单条轨迹。

目标是小而诚实：先把核心力学项与数值流程讲清楚，再谈复杂扩展。

## R10

`demo.py` 函数职责：
- `check_finite_scalar`：标量有限性校验。
- `check_vector3`：三维向量形状与有限性校验。
- `earth_rotation_vector_enu`：由纬度构造 `Omega_enu`。
- `coriolis_acceleration`：计算 `-2(Omega x v)`。
- `rhs_state`：构造状态方程右端项。
- `rk4_step`：执行单步 RK4 积分。
- `simulate_trajectory`：运行完整轨迹仿真并返回 `t/r/v/a_cor`。
- `print_trajectory_sample`：打印轨迹前若干行做快速审计。
- `run_case`：执行单个案例，生成有/无科里奥利对照摘要。
- `main`：组织参数与两个固定案例并打印结果。

## R11

运行方式：

```bash
cd "Algorithms/物理-经典力学-0107-科里奥利力_(Coriolis_Force)"
uv run python demo.py
```

脚本无交互输入，启动后直接打印两组案例结果。

## R12

输出字段说明：
- `Final displacement difference (with - without Coriolis)`：
  - 终点 ENU 位移差（单位 m）。
- `Max absolute displacement difference over whole trajectory`：
  - 全程各轴最大绝对偏移差（单位 m）。
- `Max speed difference between trajectories`：
  - 两条轨迹速度模长差的最大值（单位 m/s）。
- `Max |v · a_coriolis|`：
  - 不做功数值检查指标（理论上应为 0）。
- `Trajectory sample`：
  - 若干时刻 `t, east, north, up, vx, vy, vz` 明细。

关注重点：偏移方向是否符合物理直觉，量级是否随时间合理累积。

## R13

建议最小测试集：
- 正常案例：
  - 北向初速度（验证右偏）；
  - 东向初速度（验证南/上分量偏转）。
- 参数异常：
  - `dt=0`、`steps=0`、`gravity_mps2<=0`。
- 边界纬度：
  - `latitude=0`（赤道）；
  - `latitude=90`（北极）；
  - `latitude=-90`（南极）。
- 非法值：
  - 初始向量含 `nan` 或 `inf`。

这些测试可覆盖主要物理分支与输入鲁棒性。

## R14

关键可调参数：
- `latitude_deg`：控制科里奥利强度方向（通过 `sin/cos(phi)`）。
- `dt`、`steps`：控制积分分辨率与仿真总时长。
- `initial_position_enu`、`initial_velocity_enu`：决定轨迹形态。
- `gravity_mps2`：有效重力近似。
- `include_coriolis`：快速切换对照组。

调参建议：
- 先固定 `dt`，增大 `steps` 观察偏转累积；
- 再减小 `dt` 做数值收敛检查（看指标稳定性）。

## R15

方法对比：
- 直接在旋转系积分（本实现）：
  - 优点：方程局部、直观，可直接观察惯性力项贡献；
  - 缺点：需要谨慎处理参考系约定与符号。
- 先在惯性系积分再坐标变换：
  - 优点：避免显式惯性力项；
  - 缺点：实现更重，对初学者不直观。
- `f` 平面二维近似（只保留 `f=2*Omega*sin(phi)`）：
  - 优点：公式简洁；
  - 缺点：忽略垂向耦合与 `cos(phi)` 相关项。

本条目采用 3D 局部模型，复杂度仍低，但比纯 2D `f` 平面更完整。

## R16

典型应用场景：
- 弹道学中的横向偏差估计（中短时局部模型）。
- 大气与海洋动力学中的旋转效应教学演示。
- 航迹预测中“是否需要考虑地球自转项”的量级评估。
- 物理仿真课程中惯性力与参考系切换的实验案例。

## R17

后续扩展方向：
- 加入空气阻力（速度平方阻力或线性阻力）。
- 增加地球曲率与位置相关重力，升级为球面模型。
- 引入可变纬度路径，实时更新 `Omega_enu`。
- 加入离心项单独开关，分析其与科里奥利项相对贡献。
- 输出 CSV/图形，便于批量参数扫描与报告生成。

## R18

`demo.py` 源码级算法流（8 步）：
1. `main` 固定地球角速度、纬度、时间网格和两组初速度案例。  
2. 每个案例由 `run_case` 分别调用 `simulate_trajectory(..., include_coriolis=True/False)`。  
3. `simulate_trajectory` 先做参数校验，再通过 `earth_rotation_vector_enu` 生成 `Omega_enu`。  
4. 将状态写成 `s=[x,y,z,vx,vy,vz]`，并构造常量重力向量 `g=[0,0,-9.81]`。  
5. 每一步调用 `rk4_step`，而 `rk4_step` 内部 4 次调用 `rhs_state` 计算斜率。  
6. `rhs_state` 按 `dv/dt = g - 2(Omega x v)` 生成加速度，完成科里奥利项离散化。  
7. 仿真结束后向量化计算 `a_cor`、位移差 `delta=pos_with-pos_without`、以及 `max|v·a_cor|`。  
8. `run_case` 输出终点偏移、全程最大偏移和轨迹采样表，用于方向与量级审计。  
