# 单摆周期公式 (Pendulum Period Formula)

- UID: `PHYS-0139`
- 学科: `物理`
- 分类: `经典力学`
- 源序号: `139`
- 目标目录: `Algorithms/物理-经典力学-0139-单摆周期公式_(Pendulum_Period_Formula)`

## R01

本条目实现“单摆周期公式”的可运行最小 MVP，不把公式当静态结论，而是做成可审计计算流程。

MVP 目标：
- 给出小角近似周期公式；
- 给出有限振幅下的级数修正公式与椭圆积分精确公式；
- 数值积分非线性单摆 ODE，直接估计真实周期；
- 用误差与能量漂移指标验证公式和数值结果的一致性。

## R02

问题背景：
- 理想单摆（无阻尼、无驱动、绳长不变、质点摆球）的非线性方程是 `theta'' + (g/L) sin(theta) = 0`；
- 常见教学公式 `T0 = 2*pi*sqrt(L/g)` 只在小角振幅下准确；
- 振幅变大时，真实周期会变长，必须用高阶修正或椭圆积分公式。

因此，本条目同时提供近似、修正、精确与数值四个口径，避免“只记结论不知边界”。

## R03

输入/输出定义（脚本内固定参数，无交互输入）：
- 输入参数：`L, g, theta0, omega0, t_span, num_points, rtol, atol`；
- 主要输出：
  - 小角周期 `small_angle_period_s`；
  - 级数修正周期 `series_period_s`；
  - 椭圆积分精确周期 `exact_period_s`；
  - 数值积分估计周期 `estimated_period_s`；
  - 相对误差、能量漂移、零点计数；
  - 不同振幅下公式误差对比表（`formula_sweep`）。

## R04

建模假设（MVP 级）：
- 理想单摆、平面运动；
- 摆线质量忽略，摆球当质点；
- 无空气阻力和轴摩擦；
- 无外部驱动力；
- `g` 常量，`L` 常量。

这些假设保证了“周期与振幅关系”这一核心机制可以被直接、干净地验证。

## R05

核心方程与公式：
- 动力学：`theta'' + (g/L) sin(theta) = 0`
- 小角周期：`T0 = 2*pi*sqrt(L/g)`
- 有限振幅精确周期：
  - `T_exact = 4*sqrt(L/g)*K(k^2)`
  - `k = sin(theta_amp/2)`，`K` 为第一类完全椭圆积分
- 有限振幅级数修正（到四阶）：
  - `T_series ~= T0 * (1 + theta^2/16 + 11*theta^4/3072)`

脚本里四个口径都显式计算，并进行一致性比较。

## R06

算法流程（高层）：
1. 初始化 `PendulumConfig`；
2. 用 `solve_ivp(DOP853)` 积分非线性 ODE；
3. 从轨迹提取 `theta(t), omega(t)` 并计算单位质量机械能；
4. 用上穿零点插值法估计数值周期；
5. 用同一振幅计算 `T0`、`T_series`、`T_exact`；
6. 统计相对误差和能量漂移；
7. 额外做振幅扫描，输出公式误差表；
8. 打印 summary、轨迹头尾并执行断言。

## R07

复杂度分析（`N = num_points`）：
- 时间复杂度：`O(N)`（积分后处理和零点扫描均为线性）；
- 空间复杂度：`O(N)`（存储时间序列和派生列）。

该问题状态维度固定为 2，计算量主要由采样长度与容差要求决定。

## R08

稳定性与可信度策略：
- 使用高阶自适应积分器 `DOP853`；
- 收紧容差 `rtol=1e-9, atol=1e-11`；
- 用能量相对漂移 `max_rel_energy_drift` 做积分健康检查；
- 周期估计采用零点线性插值，而不是直接读离散采样点；
- 增加“级数修正应优于小角公式”的逻辑断言，防止实现退化。

## R09

默认参数（`demo.py`）：
- `length_m = 1.20`
- `gravity_mps2 = 9.81`
- `theta0_rad = 0.90`（约 51.57°）
- `omega0_rad_s = 0.0`
- `t_end_s = 35.0`
- `num_points = 4200`

参数意图：
- 选较大振幅，让小角近似误差清晰可见；
- 仿真时长覆盖多个周期，便于稳健估计平均周期；
- 采样密度和容差足以把数值误差压到较低量级。

## R10

与相关模型对比：
- 单摆小角模型：公式最简，适合快速估算；
- 本条目非线性模型：保留 `sin(theta)`，更接近真实理想单摆；
- 只给椭圆积分公式：缺少轨迹级可审计证据；
- 本条目做法：解析公式 + 数值积分 + 诊断指标的组合验证。

这比“背公式”更适合工程与算法训练。

## R11

输出指标解释：
- `small_angle_period_s`：小角近似周期；
- `series_period_s`：四阶级数修正周期；
- `exact_period_s`：椭圆积分精确周期；
- `estimated_period_s`：由轨迹零点估计的数值周期；
- `num_vs_exact_rel_err`：数值周期相对精确周期误差；
- `small_vs_exact_rel_err`、`series_vs_exact_rel_err`：两种公式误差；
- `max_rel_energy_drift`：能量守恒误差诊断。

期望现象：`series` 和 `estimated` 都应明显优于 `small_angle`。

## R12

实现细节说明：
- `pendulum_rhs` 直接实现非线性方程右端，不依赖黑箱模型构造；
- `estimate_period_upward_crossings` 扫描 `theta[i] < 0 <= theta[i+1]`，并做线性插值求过零时刻；
- `specific_mechanical_energy` 使用单位质量能量：
  - `E/m = 0.5*L^2*omega^2 + g*L*(1-cos(theta))`；
- `make_formula_comparison_table` 扫描多个振幅，展示误差随振幅增大而上升。

## R13

运行方式：

```bash
cd "Algorithms/物理-经典力学-0139-单摆周期公式_(Pendulum_Period_Formula)"
uv run python demo.py
```

或在仓库根目录直接运行：

```bash
uv run python Algorithms/物理-经典力学-0139-单摆周期公式_(Pendulum_Period_Formula)/demo.py
```

脚本无交互输入，直接打印结果。

## R14

常见问题排查：
- `ODE integration failed`：检查 `L, g` 是否为正、参数是否有限；
- `estimated_period_s` 为 `nan`：仿真时长不足，或采样过稀导致过零点过少；
- 能量漂移偏大：收紧容差或提高采样点数；
- 级数修正未优于小角公式：检查级数系数与振幅单位（必须是弧度）。

## R15

适用边界：
- 适用于理想单摆、无阻尼、无外驱场景；
- 不适用于大阻尼、驱动摆、复摆、球摆或刚体摆；
- 当系统发生连续翻转（旋转而非振荡）时，“周期”定义需要改为转速统计；
- 输出用于教学、算法验证与口径对比，不直接替代高保真实验反演。

## R16

可扩展方向：
- 加入阻尼与外驱动，分析频率锁定和共振；
- 用辛积分器对比长期能量守恒；
- 做参数反演：由观测周期反推 `L` 或 `g`；
- 扫描振幅-周期曲线并拟合经验修正模型；
- 与复摆模型联动，比较 `I_p` 引入后的周期变化。

## R17

本目录交付内容：
- `README.md`：R01-R18 完整说明；
- `demo.py`：可直接运行的 Python MVP；
- `meta.json`：保持与 `PHYS-0139` 元数据一致。

验证口径：
- `README.md` 与 `demo.py` 不含待填占位符；
- `uv run python demo.py` 可直接完成运行并输出结果。

## R18

`demo.py` 源码级算法流程拆解（8 步）：
1. `main()` 创建 `PendulumConfig`，确定 `L, g, theta0` 和积分参数。  
2. `simulate(cfg)` 先校验参数合法性，再构造 `t_eval` 与初始状态 `[theta0, omega0]`。  
3. `simulate` 调用 `solve_ivp`，把 `pendulum_rhs` 作为 RHS 回调；该函数每步计算 `theta_ddot = -(g/L)sin(theta)`。  
4. 积分完成后，`specific_mechanical_energy` 逐时刻计算单位质量能量，得到能量漂移指标。  
5. `estimate_period_upward_crossings` 扫描角位移序列，提取每次上穿零点并线性插值，输出数值周期均值。  
6. 以观测振幅为输入，分别调用 `small_angle_period`、`finite_amplitude_period_series`、`finite_amplitude_period_exact` 计算三种公式周期。  
7. `make_formula_comparison_table` 对多个振幅重复第 6 步，形成误差对比表（小角 vs 级数 vs 精确）。  
8. `main` 打印 summary、公式扫描表与轨迹样本，并通过断言检查“数值与精确接近、级数优于小角、能量漂移足够小”。  

说明：`scipy.special.ellipk` 仅用于实现椭圆积分 `K(m)` 数值计算，周期公式本身和误差流程都在源码中显式展开，不是黑箱调用。
