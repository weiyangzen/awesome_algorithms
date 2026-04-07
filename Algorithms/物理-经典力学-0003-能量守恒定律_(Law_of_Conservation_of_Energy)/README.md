# 能量守恒定律 (Law of Conservation of Energy)

- UID: `PHYS-0003`
- 学科: `物理`
- 分类: `经典力学`
- 源序号: `3`
- 目标目录: `Algorithms/物理-经典力学-0003-能量守恒定律_(Law_of_Conservation_of_Energy)`

## R01

问题定义：实现一个可运行、可审计、非黑盒的最小 MVP，用一维无阻尼弹簧-质点系统验证经典力学中的机械能守恒。

目标不是做“复杂仿真平台”，而是把能量守恒定律落实为可执行检查：
- 在保守力系统中，`E = T + U` 应近似常量；
- 对比两种离散算法（显式 Euler 与 Velocity-Verlet）的能量行为；
- 输出周期误差、能量漂移、能量趋势斜率等量化指标。

## R02

物理背景：
- 对于仅受保守力作用的系统，机械能守恒；
- 线性弹簧系统是最小可验证模型，势能和动能表达都简单明确；
- 数值积分会引入离散误差，因此“守恒”应通过漂移上界来审计，而不是要求机器零误差。

本条目用同一物理系统对比两种积分器，展示“定律本身”与“数值方法误差”之间的关系。

## R03

输入/输出定义：
- 输入（脚本内固定，无交互）：
  - 质量 `m`、弹簧刚度 `k`；
  - 初始位移 `x0`、初始速度 `v0`；
  - 步长 `dt`、步数 `num_steps`。
- 输出：
  - `pandas` 汇总表：周期、相对误差、能量漂移、能量趋势斜率、log-log 拟合斜率、Torch 一致性；
  - 步长扫描表：不同 `dt` 下的能量漂移；
  - 轨迹快照表：Verlet/Euler 在若干时刻的 `x, v, E`。

## R04

建模假设（MVP 级）：
- 一维系统；
- 无阻尼、无外驱动；
- 胡克定律严格成立：`F = -k x`；
- 只考察机械能（动能 + 弹性势能）；
- 用双精度浮点数近似真实连续系统。

该模型是“最小可信的守恒演示”，不是包含摩擦、非线性弹簧或多自由度耦合的全量模型。

## R05

核心方程：
- 运动方程：
  - `x' = v`
  - `v' = -(k/m) * x`
- 机械能：
  - `E = 1/2 * m * v^2 + 1/2 * k * x^2`
- 理论周期：
  - `T = 2*pi*sqrt(m/k)`

离散更新（两种）：
- 显式 Euler：
  - `x_{n+1} = x_n + dt*v_n`
  - `v_{n+1} = v_n + dt*a(x_n)`
- Velocity-Verlet：
  - `x_{n+1} = x_n + dt*v_n + 0.5*dt^2*a_n`
  - `v_{n+1} = v_n + 0.5*dt*(a_n + a_{n+1})`

## R06

算法流程：
1. 读取 `OscillatorConfig` 并校验参数合法性；
2. 分别运行 `simulate_explicit_euler` 与 `simulate_velocity_verlet`；
3. 计算理论周期与峰值法估计周期（`scipy.signal.find_peaks`）；
4. 计算两种方法的能量相对漂移；
5. 用线性回归拟合 `E(t)` 的趋势斜率（`sklearn.LinearRegression`）；
6. 做步长扫描 `dt in [0.04, 0.02, 0.01, 0.005]`；
7. 对漂移做 log-log 拟合，提取经验斜率；
8. 用 `torch` 复算能量并与 `numpy` 对齐校验；
9. 打印汇总表、扫描表和轨迹快照；
10. 运行阈值断言，保证最小物理正确性。

## R07

复杂度分析（`N = num_steps`）：
- 单次仿真时间复杂度：`O(N)`；
- 单次仿真空间复杂度：`O(N)`（保存全时序）；
- 步长扫描 `M` 组参数时总时间复杂度：`O(M*N)`。

本案例 `M=4`，且每步仅常数次算术运算，执行非常轻量。

## R08

稳定性与鲁棒性策略：
- 参数校验：`m>0, k>0, dt>0, steps>0`；
- 周期估计前检查峰值数是否充足（避免误判）；
- 用相对漂移 `max|E-E0|/|E0|` 统一度量不同量纲；
- 引入两套积分器对照，防止把单一算法误差误解成“定律失效”；
- Torch/NumPy 交叉计算，避免单实现错误未被发现。

## R09

适用场景：
- 经典力学课程中的能量守恒演示；
- 数值积分方法入门（显式 Euler vs Verlet）；
- 小型 CI 回归检查中的物理一致性指标。

不适用场景：
- 含阻尼、驱动、碰撞、非线性势能的复杂系统；
- 多自由度刚柔耦合和真实工程结构动力学；
- 需要高精度长期轨道或辛几何误差分析的科研级任务。

## R10

脚本中的正确性检查点：
1. `verlet_period_rel_error <= 1e-2`；
2. `verlet_energy_rel_drift <= 5e-3`；
3. `euler_energy_rel_drift > verlet_energy_rel_drift`；
4. `torch_numpy_energy_max_abs_diff <= 1e-12`。

这四项分别覆盖：周期正确性、守恒质量、方法对照一致性、实现一致性。

## R11

默认参数（`OscillatorConfig`）：
- `mass_kg = 1.5`
- `spring_k_npm = 9.0`
- `x0_m = 0.2`
- `v0_mps = 0.0`
- `dt_s = 0.01`
- `num_steps = 6000`

参数意图：
- 系统周期约 2.565 s；
- 总时长 60 s，可覆盖多个周期并放大累计误差差异；
- `dt=0.01` 足够让 Verlet 保持很小能量漂移。

## R12

一次实际运行（`uv run python demo.py`）关键结果：
- `theory_period_s = 2.56509966e+00`
- `euler_period_s = 2.56545455e+00`
- `verlet_period_s = 2.56500000e+00`
- `euler_energy_rel_drift = 3.55587455e+01`
- `verlet_energy_rel_drift = 1.49999795e-04`
- `euler_energy_trend_slope = 8.90253284e-02 J/s`
- `verlet_energy_trend_slope = -4.60541859e-09 J/s`
- `torch_numpy_energy_max_abs_diff = 0.0`

步长扫描中，Verlet 漂移随 `dt` 明显下降（约 `O(dt^2)`），而 Euler 在该指标下出现显著漂移放大。

## R13

理论与验证口径：
- 能量守恒在连续系统层面成立；
- 数值离散后，守恒通过“漂移足够小”来工程化验证；
- Euler 的漂移放大并不否定定律，而是体现了积分器对长期守恒特性的差异；
- 因此本条目强调“物理定律 + 数值方法”双重视角。

## R14

常见失败模式与修复：
- 失败：`dt` 过大导致 Verlet 漂移也变大。
  - 修复：减小 `dt` 或缩短仿真时长再做收敛检查。
- 失败：峰值检测失败（周期估计报错）。
  - 修复：增加总时长或放宽 `find_peaks` 的阈值。
- 失败：误把 Euler 漂移解释成“能量不守恒”。
  - 修复：对照 Verlet 结果并检查理论方程是否保守。
- 失败：单位不一致（`k`、`m`、`x` 混用）。
  - 修复：统一 SI 单位并固定输出单位说明。

## R15

工程实践建议：
- 在物理仿真中把守恒量漂移做成固定 CI 指标；
- 总是保留至少一个对照积分器，避免单实现偏差；
- 汇总表与快照表同时输出，既有统计也能追踪轨迹；
- 跨库交叉验证（如 NumPy vs Torch）可快速发现实现错误。

## R16

方法脉络与扩展：
- 经典层：动能-势能分解与保守力判据；
- 数值层：Euler、RK、Verlet、辛积分器家族；
- 扩展层：加入阻尼与外驱动可转向“能量耗散/注入”分析；
- 高维层：可推广到多自由度系统并追踪哈密顿量漂移。

本目录定位于“守恒定律的最小可运行演示 + 数值误差认知”。

## R17

本目录交付：
- `demo.py`：可直接运行的 Python MVP；
- `README.md`：R01-R18 完整说明；
- `meta.json`：与任务元数据一致。

运行方式：

```bash
cd "Algorithms/物理-经典力学-0003-能量守恒定律_(Law_of_Conservation_of_Energy)"
uv run python demo.py
```

脚本无交互输入，运行后直接输出 3 张文本表。

## R18

`demo.py` 源码级算法流拆解（8 步）：
1. `main` 创建 `OscillatorConfig` 并调用 `validate_config`，先锁死参数边界。  
2. 分别调用 `simulate_explicit_euler` 和 `simulate_velocity_verlet`，两者都显式展开离散更新，不依赖黑盒 ODE 求解器。  
3. 在每条轨迹上调用 `mechanical_energy`，逐步生成 `E(t)`，并用 `relative_energy_drift` 计算最大相对漂移。  
4. 用 `estimate_period_seconds`（`find_peaks`）从位移峰间距估计数值周期，再与 `2*pi*sqrt(m/k)` 做误差对比。  
5. 用 `energy_trend_slope`（`LinearRegression`）拟合 `E` 对 `t` 的斜率，量化长期漂移方向与强度。  
6. 用 `dt_scan` 生成多步长漂移数据，再在 `loglog_convergence_order` 中拟合 `log(drift)`-`log(dt)` 斜率，审计步长敏感性。  
7. 用 `torch_energy_consistency` 在同一状态序列上复算能量，与 NumPy 结果做最大绝对差比较。  
8. 最后汇总到 `pandas` 表格打印，并执行断言门槛，确保结果在可接受的物理与数值范围。  

说明：第三方库仅用于峰值检测、回归拟合、表格展示和交叉复算；核心动力学方程与积分更新都在源码中逐行展开，可追踪且可复审。
