# 复摆 (Physical Pendulum)

- UID: `PHYS-0140`
- 学科: `物理`
- 分类: `经典力学`
- 源序号: `140`
- 目标目录: `Algorithms/物理-经典力学-0140-复摆_(Physical_Pendulum)`

## R01

复摆（Physical Pendulum）描述的是“刚体绕固定转轴在重力作用下做摆动”的动力学，而不是理想质点单摆。

本条目目标：
- 给出复摆最小可运行 ODE 模型；
- 数值积分得到 `theta(t), omega(t)`；
- 同时输出能量守恒与周期诊断；
- 对比小角近似周期与有限振幅精确周期。

## R02

复摆和单摆的核心区别在于转动惯量：
- 单摆只需长度 `L`；
- 复摆需要刚体相对支点的转动惯量 `I_p`。

动力学方程：

`I_p * theta_ddot + c * theta_dot + m * g * d * sin(theta) = 0`

其中：
- `m`：刚体质量；
- `d`：支点到质心距离；
- `c`：线性阻尼系数（本 MVP 默认 `0`）；
- `I_p = I_cm + m*d^2`（平行轴定理）。

## R03

本题采用二维角变量状态：

`y = [theta, omega]`，其中 `omega = dtheta/dt`。

一阶系统形式：
- `dtheta/dt = omega`
- `domega/dt = -(m*g*d/I_p) * sin(theta) - (c/I_p) * omega`

这是一个非线性自治系统，`sin(theta)` 项是非线性来源。

## R04

推导要点（简版）：
1. 对支点写转动方程：`sum(tau) = I_p * theta_ddot`。
2. 重力矩大小为 `m*g*d*sin(theta)`，方向与位移相反，故取负号。
3. 加入线性耗散矩 `tau_d = -c*theta_dot`。
4. 合并得到：
   `I_p*theta_ddot = -m*g*d*sin(theta) - c*theta_dot`。
5. 再转成一阶 ODE 便于数值积分。

## R05

`demo.py` 默认实验参数：
- `mass = 1.2 kg`
- `gravity = 9.81 m/s^2`
- `com_distance = 0.22 m`
- `inertia_cm = 0.075 kg*m^2`
- `damping = 0.0`
- 初值：`theta0 = 0.60 rad`（约 34.38°），`omega0 = 0`
- 仿真：`t ∈ [0, 30] s`，`num_points = 2400`
- 积分器：`solve_ivp(method="DOP853", rtol=1e-9, atol=1e-11)`

输出包括轨迹表、周期对比和能量漂移指标。

## R06

MVP 代码结构：
- `PhysicalPendulumConfig`：集中管理物理与数值参数
- `pivot_inertia`：平行轴定理计算 `I_p`
- `pendulum_rhs`：实现 ODE 右端
- `mechanical_energy`：计算机械能
- `estimate_upward_crossing_period`：基于上穿零点估计周期
- `exact_period_finite_amplitude`：椭圆积分计算有限振幅精确周期
- `simulate`：积分并汇总诊断
- `main`：打印 summary + 轨迹头尾 + 断言校验

## R07

伪代码：

```text
input: m, g, d, I_cm, c, theta0, omega0
I_p = I_cm + m*d^2

def rhs(theta, omega):
  dtheta = omega
  domega = -(m*g*d/I_p)*sin(theta) - (c/I_p)*omega
  return [dtheta, domega]

[theta(t), omega(t)] = solve_ivp(rhs)
E(t) = 0.5*I_p*omega^2 + m*g*d*(1-cos(theta))

T_est = mean(diff(upward_zero_crossings(theta)))
T_small = 2*pi*sqrt(I_p/(m*g*d))
T_exact = 4*sqrt(I_p/(m*g*d))*K(sin^2(theta_amp/2))

report: T_est, T_small, T_exact, period errors, energy drift
```

## R08

复杂度分析（采样点数为 `N`）：
- 时间复杂度：`O(N)`（每次 RHS 评估为常数开销）
- 空间复杂度：`O(N)`（保存 `t, theta, omega, energy` 轨迹）

该问题维度很低，计算瓶颈主要由积分精度和输出采样长度决定。

## R09

数值稳定性策略：
- 使用高阶自适应积分器 `DOP853`
- 设置严格容差 `rtol=1e-9, atol=1e-11`
- 用机械能相对漂移 `max_rel_energy_drift` 做数值健康检查
- 周期估计使用线性插值上穿零点，降低采样网格粗糙导致的偏差

若漂移偏大，可收紧容差或增加输出采样点。

## R10

与相关模型对比：
- 复摆（本题）：适用于刚体，关键参数是 `I_p` 与 `d`
- 单摆：适用于质点 + 细绳，参数更少但不含刚体转动细节
- 小角线性化模型：`sin(theta)≈theta`，便于解析但大角度会产生可见误差

本 MVP 同时给出 `T_small` 和 `T_exact`，明确体现“大角度非线性影响”。

## R11

参数建议：
- 若要验证非线性周期修正：设 `theta0` 在 `0.4~0.9 rad`
- 若要接近单摆线性行为：设 `theta0 < 0.15 rad`
- 若要观察衰减包络：设置 `damping > 0`
- 确保 `mass > 0, d > 0, I_p > 0`，否则模型物理上不成立

## R12

实现细节说明（`demo.py`）：
- `simulate` 内先做参数合法性检查，再调用 ODE 积分
- 轨迹保存为 `pandas.DataFrame`，便于后续审计/导出
- 周期估计并非依赖黑盒函数：
  - 逐段检测 `theta[i] < 0 <= theta[i+1]`
  - 对每个上穿点做线性插值求 `t_cross`
  - 对连续上穿点时间差求均值
- 有限振幅周期使用 `scipy.special.ellipk` 明确调用椭圆积分定义式

## R13

运行方式：

```bash
cd "Algorithms/物理-经典力学-0140-复摆_(Physical_Pendulum)"
uv run python demo.py
```

或在项目根目录直接运行：

```bash
uv run python Algorithms/物理-经典力学-0140-复摆_(Physical_Pendulum)/demo.py
```

脚本无交互输入，执行后直接打印诊断结果。

## R14

输出解读：
- `observed_amplitude_deg`：数值轨迹观测到的最大摆角
- `small_angle_period_s`：小角近似周期
- `finite_amplitude_period_s`：有限振幅理论周期（椭圆积分）
- `estimated_period_s`：由零点上穿估计的实测周期
- `period_rel_err_vs_exact`：数值周期与有限振幅理论的偏差
- `max_rel_energy_drift`：保守系统数值守恒诊断（越小越好）

通常应看到：
- `estimated_period_s` 更接近 `finite_amplitude_period_s`，而非 `small_angle_period_s`；
- 能量漂移维持在很小量级。

## R15

常见问题排查：
- 报 `ODE integration failed`：
  - 检查参数是否有 `nan/inf`；
  - 检查 `mass, gravity, com_distance, I_p` 是否为正。
- 能量漂移偏大：
  - 缩小 `rtol/atol`；
  - 增加 `num_points` 或缩短 `t_end`。
- 周期估计为 `nan`：
  - 仿真时间太短，未出现至少两次上穿零点；
  - 或系统进入旋转而非振荡（需减小初始能量）。

## R16

可扩展方向：
- 加入非线性阻尼（如二次阻尼）
- 用外力矩激励形成受迫复摆
- 参数辨识：由实验轨迹反推 `I_cm` 或阻尼系数
- 与辛积分方法对比长时能量表现
- 批量参数扫描生成周期-振幅关系曲线

## R17

适用边界与限制：
- 当前模型是“单自由度平面复摆”，不含三维姿态耦合
- 未建模支点摩擦、结构柔性、碰撞和气动力细节
- 在高能量跨越顶点（连续旋转）场景下，周期定义需改为转速统计
- 结果用于教学与算法验证，不直接替代高保真工程仿真

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main()` 创建 `PhysicalPendulumConfig`，固定物理参数和积分参数。  
2. `simulate(cfg)` 先用 `pivot_inertia()` 计算 `I_p = I_cm + m*d^2`，并做正值校验。  
3. `simulate` 调用 `solve_ivp`，把 `pendulum_rhs` 作为 RHS 回调。  
4. `pendulum_rhs` 在每次回调中按  
   `theta_ddot = -(m*g*d/I_p)sin(theta) - (c/I_p)omega`  
   返回 `[dtheta, domega]`。  
5. 积分完成后，`mechanical_energy` 逐时刻计算  
   `E = 0.5*I_p*omega^2 + m*g*d*(1-cos(theta))`。  
6. `estimate_upward_crossing_period` 扫描 `theta` 序列，提取每次“负到正”的零点并线性插值，得到数值周期估计。  
7. `exact_period_finite_amplitude` 用观测振幅 `theta_amp` 计算  
   `T_exact = 4*sqrt(I_p/(m*g*d))*K(sin^2(theta_amp/2))`，同时计算 `T_small`。  
8. `main` 打印 `summary` 与轨迹头尾，并对能量漂移和周期误差做断言，形成“建模-积分-诊断-验证”的完整闭环。  
