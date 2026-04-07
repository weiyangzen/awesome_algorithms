# 刚体运动学 (Rigid Body Kinematics)

- UID: `PHYS-0101`
- 学科: `物理`
- 分类: `经典力学`
- 源序号: `101`
- 目标目录: `Algorithms/物理-经典力学-0101-刚体运动学_(Rigid_Body_Kinematics)`

## R01

问题定义：刚体运动学研究“刚体如何运动”，不直接涉及受力求解。  
本条目的核心对象是一个刚体上固定点 `P`，其运动由以下量决定：
- 参考点 `O` 的平动：`x_O(t), v_O(t), a_O(t)`
- 刚体姿态：旋转矩阵 `R(t) in SO(3)`
- 角速度与角加速度：`omega(t), alpha(t)`

目标是在数值仿真中验证经典关系：
- `v_P = v_O + omega x r`
- `a_P = a_O + alpha x r + omega x (omega x r)`  
其中 `r = R(t) r_body` 为点 `P` 相对 `O` 的当前位置矢量。

## R02

物理背景与定位：
- 刚体运动通常分为“平动 + 转动”，是多体系统、机器人、航天姿态问题的基础模块；
- 仅做运动学时，不需要先解牛顿-欧拉动力学，也不依赖质量或力矩；
- 上述速度/加速度传递公式是经典力学、机械设计、控制工程中的高频公式。

本 MVP 聚焦“可验证实现”：用 ODE 积分姿态 + 解析公式 + 数值微分交叉校验。

## R03

本目录要解决的具体计算任务：
1. 设定时变角速度 `omega(t)` 与参考点平动轨迹；
2. 数值积分旋转矩阵 ODE：`R_dot = [omega]x R`；
3. 对刚体固定点 `P` 计算解析速度/加速度；
4. 对位置轨迹做数值微分，得到 `v_num, a_num`；
5. 比较解析值与数值值误差，并验证 `R^T R = I` 与 `det(R)=1`。

## R04

建模假设（最小可运行版本）：
- 刚体不可形变，点 `P` 在体坐标系中位置 `r_body` 恒定；
- 角速度 `omega(t)` 视为惯性系已知函数；
- 参考点 `O` 的平动轨迹给定为光滑解析函数；
- 不求解力学来源（力/力矩），只验证运动学关系。

因此这是“纯运动学一致性”实验，不是动力学反演。

## R05

核心公式（`demo.py` 直接实现）：

1. 旋转矩阵运动学方程  
`R_dot = [omega]x R`，其中
`[omega]x = [[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]`

2. 刚体点位置  
`x_P = x_O + R r_body`

3. 刚体点速度  
`v_P = v_O + omega x r`

4. 刚体点加速度  
`a_P = a_O + alpha x r + omega x (omega x r)`  
其中 `alpha = domega/dt`。

## R06

数值算法流程：
1. 构造 `omega(t)`、`alpha(t)` 与 `x_O, v_O, a_O`；
2. 设初始姿态 `R(0)=R0`，将其展平为 9 维向量；
3. 用 `scipy.integrate.solve_ivp(DOP853)` 积分 `R_dot=[omega]xR`；
4. 每个时刻对 `R` 做一次投影到 `SO(3)`（SVD 正交化）；
5. 计算 `r=Rr_body`，再计算 `x_P, v_P, a_P` 解析值；
6. 对 `x_P` 做数值微分得到 `v_num`，再微分得到 `a_num`；
7. 输出误差统计并执行断言。

## R07

复杂度分析（设采样点数 `N`）：
- 旋转 ODE 积分每步仅常数阶 3x3 运算，时间 `O(N)`；
- SVD 正交化每步常数规模（3x3），总计仍为 `O(N)`；
- 轨迹存储（`R, x, v, a`）为 `O(N)` 空间。

该问题瓶颈主要由采样密度与积分容差决定。

## R08

数值稳定性策略：
- 采用高阶显式积分器 `DOP853`，并设置较严 `rtol/atol`；
- 对每个积分得到的 `R` 用 SVD 投影到最近旋转矩阵，抑制长期漂移；
- 误差评估使用内点区间（去除边界差分噪声）；
- 同时监控三类误差：
  - `orthogonality_error = ||R^T R - I||`
  - `determinant_error = |det(R)-1|`
  - 解析/数值速度与加速度误差。

## R09

适用范围：
- 机器人连杆、机电装置、飞行器姿态中“已知角速度轨迹”的前向运动学；
- 教学演示中验证刚体点速度/加速度传递公式；
- 作为动力学模型上层的几何运动模块。

不适用范围：
- 需要由外力外矩反推 `omega` 的动力学求解；
- 包含大变形、柔性体、碰撞接触等非刚体情形。

## R10

正确性检查框架（脚本中有对应断言）：
1. ODE 求解必须成功；
2. `R` 的正交性误差和行列式误差须足够小；
3. `v_formula` 与 `v_num` 的 RMSE 低于阈值；
4. `a_formula` 与 `a_num` 的 RMSE 低于阈值；
5. `dr/dt` 与 `omega x r` 的 RMSE 低于阈值。

这对应“姿态正确 + 速度关系正确 + 加速度关系正确”的三层校验。

## R11

默认参数（见 `RigidBodyScenario`）：
- 角速度（惯性系）：
  - `wx(t)=0.8+0.15 cos(0.7t)`
  - `wy(t)=-0.3+0.1 sin(0.5t)`
  - `wz(t)=1.2+0.12 cos(0.9t)`
- 参考点平动：
  - `x_O(t)=[0.3t, -0.1t+0.05t^2, 0.2 sin(0.6t)]`
- 刚体点体坐标：`r_body=[0.4, -0.2, 0.35]`
- 仿真：`t_end=10.0 s`, `num_steps=2401`

参数选择兼顾“非平凡耦合”与“稳定可验证”。

## R12

一次典型运行输出会包含：
- `orthogonality_max_fro`（一般在 `1e-12 ~ 1e-10`）
- `determinant_max_abs_error`（一般在 `1e-13 ~ 1e-11`）
- `velocity_rmse`（通常 `1e-4` 量级）
- `acceleration_rmse`（通常 `1e-3 ~ 1e-2` 量级）
- `transport_rmse`，即 `dr/dt` 与 `omega x r` 的一致性误差

这些指标共同说明刚体运动学关系在离散数值中成立。

## R13

理论层面的正确性依据：
- 若 `R_dot=[omega]xR` 且 `R(0) in SO(3)`，连续系统上有 `R(t) in SO(3)`；
- 对刚体固定点 `r_body`，有 `r=Rr_body`，故 `dr/dt = omega x r`；
- 再对速度求导即可得到加速度传递公式。

本实现无法给出全局离散误差严格上界，但通过：
- 高精度积分；
- `SO(3)` 投影；
- 多指标误差诊断；
实现了工程可审计的正确性验证。

## R14

常见失败模式与修复：
- 失败：步长过大，`v/a` 数值微分噪声高。  
  修复：增大 `num_steps`，缩小差分间距。
- 失败：积分容差过松，姿态漂移导致误差升高。  
  修复：收紧 `rtol/atol`。
- 失败：不做旋转矩阵重投影，长时仿真偏离 `SO(3)`。  
  修复：启用 SVD 正交化。
- 失败：角速度变化过激引发采样不足。  
  修复：提高采样率或限制频率/幅值。

## R15

实现设计（对应 `demo.py`）：
- `RigidBodyScenario`：统一管理仿真参数；
- `omega_world` / `alpha_world`：角速度与角加速度解析表达；
- `origin_kinematics`：参考点平动的 `x,v,a`；
- `rotation_rhs`：`R_dot=[omega]xR` 的 ODE 右端；
- `project_to_so3`：SVD 投影，保证 `R` 的旋转矩阵性质；
- `simulate`：积分、后处理、误差统计一站式流程；
- `print_report`：用 `pandas` 输出可读诊断表。

## R16

相关算法与方法脉络：
- 姿态表示：旋转矩阵、四元数、旋转向量；
- 数值积分：显式 Runge-Kutta、Lie 群积分、辛积分；
- 动力学扩展：欧拉刚体方程、牛顿-欧拉递推；
- 工程扩展：IMU 融合、机械臂正逆运动学、轨迹优化。

本条目处于“动力学之前”的几何运动层。

## R17

运行方式：

```bash
cd Algorithms/物理-经典力学-0101-刚体运动学_(Rigid_Body_Kinematics)
uv run python demo.py
```

运行特性：
- 无交互输入；
- 直接打印参数与误差诊断表；
- 含断言自检，全部通过会输出 `All checks passed.`。

依赖：
- `numpy`
- `scipy`
- `pandas`

## R18

`demo.py` 源码级算法流程拆解（8 步）：
1. `RigidBodyScenario` 给出时变 `omega(t)`、参考点平动、体坐标点 `r_body` 与仿真时间网格。
2. `initial_rotation()` 通过轴角公式构造 `R0`，保证初值在 `SO(3)`。
3. `rotation_rhs()` 在每个时刻计算 `R_dot=[omega]xR`，将 3x3 矩阵展平/还原供 ODE 求解器使用。
4. `integrate_rotation()` 调用 `solve_ivp(DOP853)` 获得全时域 `R(t)`，随后逐帧执行 `project_to_so3()` 抑制正交漂移。
5. `point_kinematics()` 对每个采样点计算 `r=Rr_body`，并由运动学公式得到 `x_P, v_P, a_P`。
6. `finite_difference_kinematics()` 对 `x_P` 做两次数值微分，生成 `v_num, a_num` 作为独立参照。
7. `compute_diagnostics()` 汇总 `SO(3)` 误差、`v/a` 误差、以及 `dr/dt` 与 `omega x r` 的传输定理误差。
8. `main()` 打印 `pandas` 报表并执行阈值断言，确保实现可复现且可自动验证。

这里没有调用“刚体运动学黑盒函数”直接给结果；关键方程都在源码中逐步显式实现和检查。
