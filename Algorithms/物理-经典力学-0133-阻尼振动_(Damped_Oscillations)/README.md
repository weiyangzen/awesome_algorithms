# 阻尼振动 (Damped Oscillations)

- UID: `PHYS-0133`
- 学科: `物理`
- 分类: `经典力学`
- 源序号: `133`
- 目标目录: `Algorithms/物理-经典力学-0133-阻尼振动_(Damped_Oscillations)`

## R01

阻尼振动研究的是“系统在耗散作用下的振幅衰减运动”。

本条目采用最小但完整的单自由度模型：

`m x¨ + c x˙ + k x = 0`

并在同一脚本中实现三件事：
- 按阻尼比自动判别欠阻尼/临界阻尼/过阻尼；
- 给出对应解析解；
- 用 `scipy.solve_ivp` 数值积分交叉验证误差与能量衰减。

## R02

本目录 MVP 问题定义：

- 输入：
  - 物理参数 `m, c, k`；
  - 初值 `x0, v0`；
  - 时间区间 `t_end` 和采样点数 `num_points`；
  - 数值积分容差 `rtol, atol`。
- 输出：
  - 时序轨迹 `x(t), v(t)`（解析解与数值解）；
  - 最大绝对误差 `max_abs_x_err, max_abs_v_err`；
  - 机械能变化 `energy_end_relative_change`；
  - 由峰值对数减量估计的阻尼比 `zeta_est_log_dec`。

## R03

基础动力学方程：

`m x¨ + c x˙ + k x = 0`

定义参数：
- 无阻尼固有角频率：`omega_n = sqrt(k/m)`
- 衰减率：`alpha = c/(2m)`
- 阻尼比：`zeta = c/(2*sqrt(km))`

特征方程为：

`r^2 + (c/m) r + (k/m) = 0`

其根结构决定运动形态。

## R04

按阻尼比分三类：

- `zeta < 1`（欠阻尼）：复根，出现振荡衰减；
- `zeta = 1`（临界阻尼）：重根，最快无振荡回到平衡；
- `zeta > 1`（过阻尼）：双实根，无振荡但回稳较慢。

本实现使用 `zeta` 直接分支，不依赖人工指定模式。

## R05

三种解析解（`x(0)=x0`, `x˙(0)=v0`）：

1. 欠阻尼 `zeta<1`：

`x(t)=e^{-alpha t}(A cos(omega_d t)+B sin(omega_d t))`

其中 `omega_d = omega_n*sqrt(1-zeta^2)`，
`A=x0`，`B=(v0+alpha*x0)/omega_d`。

2. 临界阻尼 `zeta=1`：

`x(t)=(A+Bt)e^{-omega_n t}`，`A=x0`，`B=v0+omega_n*x0`。

3. 过阻尼 `zeta>1`：

`x(t)=C1 e^{s1 t}+C2 e^{s2 t}`，
`si=-omega_n(zeta \mp sqrt(zeta^2-1))`，常数由初值线性求解。

`demo.py` 同时给出 `x˙(t)` 的显式表达并用于误差验证。

## R06

数值路径采用一阶状态空间：

`y=[x,v]`，`y˙=[v, -(c/m)v-(k/m)x]`

然后调用 `solve_ivp(method="DOP853")`。

解析路径与数值路径在同一时间网格逐点比较：
- `abs_x_err = |x_num - x_ana|`
- `abs_v_err = |v_num - v_ana|`

这样可以把“公式实现错误”和“积分器设置错误”区分开。

## R07

伪代码：

```text
input: m,c,k,x0,v0,t_end,N,rtol,atol

t = linspace(0, t_end, N)
compute wn, alpha, zeta

if zeta<1: use underdamped closed form
elif zeta==1: use critical closed form
else: use overdamped closed form
-> (x_ana, v_ana)

solve_ivp on y=[x,v], rhs=[v, -(c/m)v-(k/m)x]
-> (x_num, v_num)

energy = 0.5*m*v_num^2 + 0.5*k*x_num^2
errors = abs(num - ana)
estimate zeta from peaks via log decrement

report summary + trajectory sample
assert error and physical checks
```

## R08

复杂度（时间采样点 `N`）：

- 解析解计算：`O(N)`；
- 数值积分（固定维度 2）：约 `O(N)`；
- 误差与能量统计：`O(N)`；
- 空间复杂度：保存轨迹 `O(N)`。

因为是单自由度模型，本 MVP 的性能开销很小。

## R09

数值稳定与诊断策略：

- 强制参数合法性检查：`m>0`, `k>0`, `c>=0`, 容差正值；
- `solve_ivp` 使用高阶 `DOP853` 和紧容差（默认 `1e-10/1e-12`）；
- 检查数值结果有限性（防 `nan/inf`）；
- 用物理量验证：阻尼系统总机械能应整体下降（末态低于初态）；
- 用解析-数值一致性验证实现正确性。

## R10

与相关方法对比：

- 纯解析法：
  - 优点：精确、快；
  - 局限：仅适用于已知闭式解的线性模型。
- 纯数值积分法：
  - 优点：易扩展到外力、非线性；
  - 局限：缺少基准时不易发现实现偏差。
- 本条目方案（解析 + 数值对照）：
  - 同时保留可解释性与可扩展性，适合作为后续复杂模型基线。

## R11

`demo.py` 默认参数：

- `m=1.0`, `c=0.6`, `k=16.0`
- `x0=0.10`, `v0=-0.05`
- `t_end=14.0`, `num_points=2200`
- `rtol=1e-10`, `atol=1e-12`

这组参数对应欠阻尼，能够明显看到振幅指数衰减并保留多个峰值，便于做对数减量估计。

## R12

实现结构说明：

- `DampedOscillationConfig`：集中管理所有参数；
- `validate_config`：输入合法性与有限值检查；
- `system_characteristics`：计算 `omega_n, zeta, alpha` 并判别阻尼区；
- `analytical_solution`：按三种阻尼分支生成 `x_ana, v_ana`；
- `integrate_numerically`：构造状态方程并调用 `solve_ivp`；
- `estimate_log_decrement`：基于峰值估计 `zeta` 和 `omega_d`；
- `simulate`：组织全流程，产出轨迹表与汇总表。

## R13

运行方式：

```bash
cd "Algorithms/物理-经典力学-0133-阻尼振动_(Damped_Oscillations)"
uv run python demo.py
```

或在仓库根目录直接运行：

```bash
uv run python Algorithms/物理-经典力学-0133-阻尼振动_(Damped_Oscillations)/demo.py
```

脚本无交互输入，直接打印 summary 和轨迹样本。

## R14

输出字段含义：

- `regime`：阻尼类别（under/critical/over）；
- `omega_n_rad_s`：无阻尼固有角频率；
- `zeta`：理论阻尼比；
- `alpha_1_over_s`：指数衰减率；
- `omega_d_rad_s`：欠阻尼角频率（其余情形为 `nan/0`）；
- `max_abs_x_err`, `max_abs_v_err`：解析解与数值解最大偏差；
- `energy_end_relative_change`：末态相对能量变化（应为负）；
- `zeta_est_log_dec`：由峰值对数减量估计的阻尼比。

## R15

常见问题排查：

- 误差断言失败：
  - 增加 `num_points`，或收紧 `rtol/atol`；
  - 检查是否误改了解析分支公式。
- 找不到足够峰值（`zeta_est_log_dec=nan`）：
  - 可能阻尼过大（临界/过阻尼）或仿真时间过短。
- 能量未下降：
  - 先确认 `c>0`；
  - 再确认 `rhs` 中阻尼项符号是否为 `-(c/m)*v`。

## R16

可扩展方向：

- 加入外力 `f(t)` 形成受迫阻尼振动；
- 扩展到多自由度 `M x¨ + C x˙ + K x = 0`；
- 加入噪声观测并做参数辨识（估计 `m,c,k`）；
- 对比不同积分器（`RK45`, `Radau`, `BDF`）在强阻尼下的表现。

## R17

适用边界与限制：

- 仅适用于线性弹簧与线性黏性阻尼；
- 不覆盖库仑摩擦、间隙碰撞、强非线性刚度；
- 单自由度模型主要用于教学、原型验证和单元测试基线；
- 真实工程系统通常需要更高维模型与参数标定。

## R18

`demo.py` 源码级算法流程（9 步）：

1. `main()` 构造 `DampedOscillationConfig`（物理参数、初值、时间网格、容差）。
2. `simulate(cfg)` 先调用 `validate_config` 做参数有限性与正值约束检查。
3. 在 `simulate` 中生成统一时间网格 `times`，保证解析解和数值解逐点可比。
4. `integrate_numerically` 把二阶方程改写为 `y=[x,v]` 一阶系统，`rhs` 显式计算 `a=-(c/m)v-(k/m)x`。
5. 同函数调用 `solve_ivp(DOP853)` 得到 `x_num, v_num`，并检查 `sol.success` 与有限值。
6. `analytical_solution` 通过 `system_characteristics` 计算 `omega_n/zeta/alpha`，按欠阻尼、临界、过阻尼三分支使用对应闭式公式构造 `x_ana, v_ana`。
7. `simulate` 计算 `abs_x_err/abs_v_err` 和 `energy=0.5*m*v^2+0.5*k*x^2`，得到误差上界与能量衰减指标。
8. `estimate_log_decrement` 用 `scipy.signal.find_peaks` 提取峰值，按对数减量公式估计 `zeta_est` 与 `omega_d_est`，形成额外物理一致性诊断。
9. `main` 打印 `summary` 与轨迹样本，并通过断言阈值完成自动验收（误差阈值、能量下降、阻尼比估计偏差）。
