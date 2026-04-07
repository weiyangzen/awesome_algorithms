# 受迫振动 (Forced Oscillations)

- UID: `PHYS-0134`
- 学科: `物理`
- 分类: `经典力学`
- 源序号: `134`
- 目标目录: `Algorithms/物理-经典力学-0134-受迫振动_(Forced_Oscillations)`

## R01

受迫振动研究“振子在外部周期驱动力作用下的响应”，是经典力学和振动工程中的核心问题。最基础模型是单自由度阻尼谐振子：

`m x'' + c x' + k x = F0 cos(omega_d t)`

本条目目标是做一个可运行的最小 MVP，覆盖：
- 时域数值积分（含阻尼与外力）；
- 稳态解析振幅/相位公式；
- 频率扫描下的共振响应与误差校验。

## R02

本目录实现的问题定义：
- 输入（脚本内置）：
  - 物理参数 `m, c, k, F0`；
  - 主频验证用驱动角频率 `omega_d`；
  - 扫频范围与采样数量；
  - 数值积分时间网格和容差。
- 输出：
  - 主频点的解析振幅/相位与数值拟合结果；
  - 扫频表（理论幅值 vs 数值幅值）；
  - 峰值频率、拟合误差、功率平衡误差等诊断指标；
  - `PASS/FAIL` 验证结论。

## R03

微分方程与参数定义：

`m x'' + c x' + k x = F0 cos(omega_d t)`

其中：
- `m > 0`：质量；
- `c >= 0`：线性阻尼系数；
- `k > 0`：弹簧刚度；
- `F0`：驱动力幅值；
- `omega_d`：驱动角频率。

系统固有角频率：

`omega0 = sqrt(k / m)`

阻尼比：

`zeta = c / (2 * sqrt(k m))`

## R04

稳态解写作：

`x_ss(t) = A(omega_d) cos(omega_d t - phi(omega_d))`

其中：

`A(omega_d) = F0 / sqrt((k - m omega_d^2)^2 + (c omega_d)^2)`

`phi(omega_d) = atan2(c omega_d, k - m omega_d^2)`

这两个公式是 demo 校验的核心基准：数值积分在足够长时间后应收敛到同样的振幅和相位。

## R05

轻阻尼时，位移幅值在共振附近出现峰值。理论峰值频率（位移响应）可写为：

`omega_peak ~= sqrt(omega0^2 - 2 * beta^2)`, `beta = c/(2m)`

等价于：

`omega_peak ~= sqrt(k/m - c^2/(2m^2))`

当阻尼过大时，上式可能无实根，响应不再出现明显共振峰。`demo.py` 会对该条件进行判断并输出。

## R06

本 MVP 采用“两条路径交叉验证”：
1. 解析路径：用上式直接算 `A_theory, phi_theory`。
2. 数值路径：`solve_ivp` 积分后，在尾段用最小二乘拟合
   `x(t) ~= a cos(omega t) + b sin(omega t) + bias`，
   再恢复 `A_num = sqrt(a^2+b^2)`、`phi_num = atan2(b,a)`。

最终比较两者差异，并检验稳态功率平衡：
- 输入平均功率 `mean(F(t) * v(t))`；
- 阻尼耗散功率 `mean(c v(t)^2)`。

## R07

伪代码：

```text
input: m,c,k,F0,omega_main,omega_grid

for omega in {omega_main and omega_grid}:
  integrate x'' = (F0*cos(omega t) - c x' - k x)/m
  take tail segment (after transient)
  fit x_tail = a*cos(omega t) + b*sin(omega t) + bias
  A_num = sqrt(a^2+b^2)
  phi_num = atan2(b,a)
  A_theory, phi_theory from closed-form formula
  record errors

on main omega:
  compute mean input power and mean dissipation power on tail

aggregate:
  resonance estimate, median/max amplitude relative error
  threshold checks -> PASS/FAIL
```

## R08

复杂度（`Nw` 为扫频点数，`Nt` 为每次积分采样点）：
- 每个频点一次 ODE 积分与一次线性最小二乘拟合；
- 总体约 `O(Nw * Nt)`（在本 1 自由度系统中，状态维数固定为 2）。

默认参数下 `Nw` 与 `Nt` 都较小，运行时间通常为秒级。

## R09

数值稳定性策略：
- 使用 `solve_ivp(method="DOP853")` 与严格容差；
- 仅在尾段（去除瞬态）拟合稳态幅值，避免初期过渡态污染；
- 相位误差用 `[-pi, pi)` 包裹，避免 `2pi` 跳变误判；
- 用拟合 `RMSE` 与功率平衡误差共同判断结果质量。

## R10

与其他实现方式对比：
- 仅公式计算：
  - 优点：快；
  - 局限：无法展示瞬态收敛过程，难做时域验证。
- 纯黑箱工具（直接频响 API）：
  - 优点：开发快；
  - 局限：不透明，不利于教学和审计。
- 本实现（公式 + 显式 ODE + 显式拟合）：
  - 在保持简洁的同时，保留完整可解释的算法链路。

## R11

`demo.py` 默认参数：
- `m=1.0, c=0.35, k=20.0, F0=1.5`
- 主频点 `omega_d=4.2 rad/s`（接近 `omega0=sqrt(20)`）
- 主仿真：`t in [0, 80]`, `num_points=5000`
- 扫频：`omega in [0.4*omega0, 1.8*omega0]`, `num_freqs=19`

调参建议：
- 减小 `c` 可观察更尖锐共振峰；
- 增大 `t_end` 可提升低频点稳态拟合准确性；
- 扫频点数可增大以获得更平滑频响曲线。

## R12

实现函数映射：
- `steady_state_amplitude_phase`：解析稳态振幅与相位；
- `integrate_forced_oscillator`：时域积分 `x, v`；
- `fit_harmonic_response`：尾段谐波拟合提取数值幅值与相位；
- `run_single_frequency_validation`：主频点误差与功率平衡诊断；
- `run_frequency_sweep`：整段频率响应表；
- `build_summary`：汇总峰值频率、统计误差与验证阈值。

## R13

运行方式：

```bash
cd "Algorithms/物理-经典力学-0134-受迫振动_(Forced_Oscillations)"
uv run python demo.py
```

或在仓库根目录：

```bash
uv run python Algorithms/物理-经典力学-0134-受迫振动_(Forced_Oscillations)/demo.py
```

脚本无交互输入，直接输出摘要与扫频结果表。

## R14

关键输出字段：
- `amp_theory` / `amp_numeric`：理论/数值稳态幅值；
- `phase_theory_rad` / `phase_numeric_rad`：理论/数值相位；
- `amp_rel_err`：幅值相对误差；
- `phase_abs_err_rad`：相位绝对误差；
- `fit_rmse`：尾段拟合残差；
- `mean_input_power`、`mean_dissipation_power`：功率平衡校验；
- `omega_peak_theory`、`omega_peak_numeric`：理论/数值共振峰频率。

## R15

常见问题排查：
- 幅值误差偏大：
  - 延长仿真时间，确保瞬态衰减后再拟合；
  - 增加 `num_points` 或收紧容差。
- 相位误差异常跳变：
  - 检查是否使用了相位包裹函数；
  - 低幅值区相位本身易受噪声影响。
- 功率平衡偏差大：
  - 可能尾段太短，或频率太低导致周期覆盖不足；
  - 适当提高 `tail_fraction` 或 `t_end`。

## R16

可扩展方向：
- 支持任意外力 `F(t)`（方波、脉冲、扫频激励）；
- 扩展到多自由度 `M x'' + C x' + K x = f(t)` 并做模态叠加；
- 对扫频结果做自动峰值检测与半功率带宽估计；
- 结合实验数据做参数辨识（估计 `m,c,k`）。

## R17

适用边界：
- 当前为线性单自由度模型，仅适用于小变形线性区；
- 阻尼采用粘性线性阻尼，未覆盖干摩擦等非线性耗散；
- 不包含刚度非线性（Duffing）、碰撞、间隙、时变参数；
- 是教学与原型级实现，不替代工程级高保真多体仿真。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `validate_config` 检查 `m,c,k,F0` 与积分网格合法性，避免无意义参数进入求解。
2. `steady_state_amplitude_phase` 根据解析公式计算给定驱动频率下的理论稳态振幅和相位。
3. `integrate_forced_oscillator` 将二阶方程改写为一阶状态 `y=[x,v]`，调用 `solve_ivp` 得到时域轨迹。
4. `fit_harmonic_response` 在轨迹尾段构造设计矩阵 `[cos(omega t), sin(omega t), 1]`，用 `numpy.linalg.lstsq` 拟合稳态谐波参数。
5. 由拟合系数恢复 `A_num` 与 `phi_num`，并与 `A_theory, phi_theory` 计算幅值/相位误差（相位先包裹到 `[-pi, pi)`）。
6. `run_single_frequency_validation` 额外计算尾段平均输入功率和阻尼耗散功率，检查稳态能流是否一致。
7. `run_frequency_sweep` 对一组驱动频率重复执行“积分+拟合+对比”，输出频响 `DataFrame`。
8. `build_summary` 汇总峰值频率、误差统计和阈值检查，在 `main` 中打印并给出 `PASS/FAIL`。
