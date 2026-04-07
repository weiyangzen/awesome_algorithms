# 共振理论 (Resonance Theory)

- UID: `PHYS-0135`
- 学科: `物理`
- 分类: `经典力学`
- 源序号: `135`
- 目标目录: `Algorithms/物理-经典力学-0135-共振理论_(Resonance_Theory)`

## R01

共振理论研究“外驱频率接近系统固有频率时，稳态响应显著放大”的机制。  
本条目使用单自由度阻尼受迫振子作为最小可验证模型：

`m*x¨ + c*x˙ + k*x = F0*cos(omega*t)`

目标是把以下内容串成一个可运行闭环：
- 解析频响曲线（振幅/相位）
- 理论共振频率预测
- 数值积分得到时域轨迹
- 稳态谐波拟合并与解析结果对比

## R02

本目录 MVP 问题定义：
- 输入：`m, c, k, F0` 与频率扫描区间 `[omega_min, omega_max]`，以及积分参数。
- 输出：
  - 频率响应表 `A(omega), phi(omega)`；
  - 理论共振频率 `omega_res`；
  - 近共振频点的数值稳态幅值/相位估计；
  - 理论-数值误差与自动断言结果。

脚本固定一组默认参数直接运行，无需交互输入。

## R03

线性阻尼受迫振子稳态解可写成：

`x_ss(t) = A(omega) * cos(omega*t - phi(omega))`

其中：

`A(omega) = F0 / sqrt((k - m*omega^2)^2 + (c*omega)^2)`

`phi(omega) = atan2(c*omega, k - m*omega^2)`

该公式给出了经典共振图：低频近同相、过共振后相位滞后接近 `pi`。

## R04

位移共振峰（阻尼不太大时）出现在：

`omega_res = sqrt(k/m - c^2/(2m^2))`

同时定义：
- 固有角频率：`omega0 = sqrt(k/m)`
- 阻尼比：`zeta = c / (2*sqrt(k*m))`

当 `k/m - c^2/(2m^2) <= 0` 时，不存在位移共振峰，本 MVP 会在参数检查阶段直接报错。

## R05

本实现采用“两条路径交叉验证”：
1. 解析路径：直接用频响公式扫描频率，找到离散网格上的峰值。  
2. 数值路径：对近共振三个频点执行 `solve_ivp`，截取稳态段做最小二乘谐波拟合，估计幅值和相位。  

两条路径一致时，说明“理论公式 + ODE 实现 + 稳态提取”是自洽的。

## R06

数值稳态提取策略：
- 先积分一段瞬态时间 `t_transient`，再分析 `t_sample`；
- 对稳态段拟合
  `x(t) = a*cos(omega*t) + b*sin(omega*t) + d`；
- 幅值 `A_est = sqrt(a^2+b^2)`，相位 `phi_est = atan2(b, a)`；
- 与解析 `A_theory, phi_theory` 比较相对误差与相位误差。

这样可避免只看峰-峰值导致的噪声敏感问题。

## R07

伪代码：

```text
input: cfg = {m,c,k,F0,omega_range,integrator_settings}
validate(cfg)

omega_grid = linspace(omega_min, omega_max, N)
A_grid = F0 / sqrt((k-m*omega_grid^2)^2 + (c*omega_grid)^2)
phi_grid = atan2(c*omega_grid, k-m*omega_grid^2)
omega_peak_grid = argmax(A_grid)
omega_res_theory = sqrt(k/m - c^2/(2m^2))

for omega in {0.9*omega_res_theory, omega_res_theory, 1.1*omega_res_theory}:
  solve_ivp for m*x'' + c*x' + k*x = F0*cos(omega*t)
  keep steady segment t >= t_transient
  least-squares fit x = a*cos(omega*t)+b*sin(omega*t)+d
  A_est, phi_est <- (a,b)
  compare with A_theory(omega), phi_theory(omega)

assert frequency peak alignment + amplitude/phase errors within thresholds
print summary and tables
```

## R08

复杂度（频率网格点数 `N`，稳态拟合点数 `M`，仿真频点数 `K=3`）：
- 解析频响扫描：`O(N)`；
- 每个频点拟合：`O(M)`（固定 3 列设计矩阵）；
- ODE 积分：与步数线性相关，整体近似 `O(K*steps)`。

本条目参数规模很小，运行开销主要来自三次 `solve_ivp`。

## R09

数值稳定性设计：
- `solve_ivp(method="DOP853")` 配合 `rtol=1e-8, atol=1e-10`；
- `max_step` 按每周期采样点数 `points_per_period` 限制，保证拟合精度；
- 瞬态段丢弃，降低初值影响；
- 相位误差采用包角差 `atan2(sin(d), cos(d))`，避免 `2pi` 跳变误判。

## R10

与常见替代方案对比：
- 只用解析频响：
  - 优点：快、公式清晰；
  - 局限：不能验证 ODE 实现正确性。
- 只做时域积分：
  - 优点：可扩展到更复杂非线性/非平稳外力；
  - 局限：难直接判断“是否符合共振理论”。
- 本实现（解析 + 数值 + 拟合）：
  - 兼顾可解释性与可验证性，适合教学与算法基线测试。

## R11

默认参数（`demo.py`）：
- `m=1.0`, `c=0.45`, `k=25.0`, `F0=1.0`
- 扫频：`omega in [1.0, 9.0]`，`161` 点
- 时域积分：`t_transient=35s`, `t_sample=18s`

该参数处于欠阻尼区，存在明显但有限的位移共振峰，适合稳定演示。

## R12

脚本输出包含三部分：
- `Summary metrics`：
  - `natural_frequency_rad_s`, `damping_ratio`
  - `theoretical_resonance_rad_s`, `grid_peak_rad_s`
  - 峰值频率偏差、幅值/相位最大误差、拟合 RMSE
- `Steady-state fit vs theory near resonance`：
  - 三个频点的理论值、估计值与误差
- `Top-5 amplitudes from analytic frequency sweep`：
  - 频响曲线中振幅最大的 5 个网格点

## R13

运行方式：

```bash
cd "Algorithms/物理-经典力学-0135-共振理论_(Resonance_Theory)"
uv run python demo.py
```

或在仓库根目录：

```bash
uv run python Algorithms/物理-经典力学-0135-共振理论_(Resonance_Theory)/demo.py
```

脚本为非交互执行，运行结束即给出结果并完成断言。

## R14

自动验收断言（`demo.py`）：
- 网格峰值频率与理论共振频率偏差不超过 `1.5 * grid_step`；
- 稳态幅值最大相对误差 `< 2%`；
- 稳态相位最大绝对误差 `< 0.08 rad`；
- 拟合 RMSE `< 0.003 m`；
- 中心频点（`omega_res`）估计幅值大于两侧频点。

这些阈值保证了“共振峰存在且被正确识别”。

## R15

常见问题排查：
- 报错 `too heavily damped`：阻尼过大，不满足位移共振峰条件。  
- 幅值误差偏大：增大 `t_transient` 或 `points_per_period`。  
- 相位误差偏大：检查相位定义是否与 `x=A*cos(omega*t-phi)` 一致。  
- 积分失败：检查参数是否有限、是否出现不合理单位尺度。

## R16

可扩展方向：
- 扩展到多自由度 `M x¨ + C x˙ + K x = F0*cos(omega t)` 并画模态共振峰；
- 增加外力包络（扫频 chirp）研究动态共振穿越；
- 在频域加入半功率带宽估计 `Q` 值；
- 将稳态拟合替换为 FFT 或锁相检测，比较估计鲁棒性。

## R17

MVP 已完成内容：
- 受迫阻尼振子共振的解析频响实现；
- 理论共振频率计算与离散扫频峰值定位；
- `solve_ivp` 时域积分与稳态谐波拟合；
- 幅值/相位误差自动校验；
- 一键非交互运行。

该版本是“共振理论”条目的最小、可复验实现。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 构造 `ResonanceConfig`，并在 `validate_config` 检查参数有限性、正值约束与“存在位移共振峰”条件。  
2. `frequency_response_table` 在 `omega` 网格上调用 `steady_state_amplitude/steady_state_phase_lag`，生成解析频响表。  
3. 在频响表中取 `amplitude_m` 最大值，得到离散峰值频率 `grid_peak_rad_s`。  
4. 基于公式 `omega_res = sqrt(k/m - c^2/(2m^2))` 计算理论共振频率，并生成三个近共振分析频点。  
5. `simulate_trajectory` 对每个分析频点调用 `solve_ivp(DOP853)`：内部按自适应步长反复调用 `rhs_forced_oscillator` 评估导数，并输出全时域轨迹。  
6. `fit_steady_harmonic` 仅在稳态时间窗上做线性最小二乘，拟合 `a*cos(wt)+b*sin(wt)+d`，显式求得 `A_est` 与 `phi_est`。  
7. `analyze_single_frequency` 将拟合结果与解析幅值/相位逐点比较，计算相对幅值误差、包角相位误差和拟合 RMSE。  
8. `main` 汇总指标打印表格，并执行断言，形成“理论预测 -> 时域仿真 -> 稳态反演 -> 误差验收”的完整闭环。
