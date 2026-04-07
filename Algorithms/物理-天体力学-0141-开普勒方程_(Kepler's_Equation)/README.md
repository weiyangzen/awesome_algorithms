# 开普勒方程 (Kepler's Equation)

- UID: `PHYS-0141`
- 学科: `物理`
- 分类: `天体力学`
- 源序号: `141`
- 目标目录: `Algorithms/物理-天体力学-0141-开普勒方程_(Kepler's_Equation)`

## R01

问题目标：实现并验证开普勒方程的数值求解器。

本条目聚焦椭圆轨道情形 `0 <= e < 1`，求解：
- `M = E - e sin(E)`（已知平近点角 `M` 与离心率 `e`，求偏近点角 `E`）。

MVP 验证三件事：
- 显式 Newton/Halley 迭代在多组 `e` 上稳定收敛；
- 方程残差 `|E - e sin(E) - M|` 达到高精度；
- 与 SciPy `brentq` 参考解的误差保持在机器精度量级。

## R02

任务定义（本目录）：
- 输入（脚本内固定参数）：
  - 离心率集合 `e_values=(0.0, 0.2, 0.5, 0.8, 0.9, 0.99)`；
  - 每组离心率对应 `n_mean_anomalies=360` 个 `M` 样本；
  - 迭代收敛阈值 `tol=1e-12`，最大迭代数 `max_iter=20`。
- 输出：
  - 每组离心率的收敛率、迭代次数、最大/平均残差；
  - 与 `brentq` 参考解的最大/平均绝对误差；
  - 求解耗时与参考方法耗时对比；
  - 轨道几何派生量示例（`e=0.8` 时的 `nu` 覆盖范围、`r_min/r_max`）。

脚本无交互输入，执行 `uv run python demo.py` 即完成计算与断言。

## R03

核心数学关系：

1. 开普勒方程（椭圆轨道）：
   - `M = E - e sin(E)`。
2. 残差函数：
   - `f(E) = E - e sin(E) - M`。
3. 一阶与二阶导数：
   - `f'(E) = 1 - e cos(E)`；
   - `f''(E) = e sin(E)`。
4. Halley 更新式：
   - `E_{k+1} = E_k - f / (f' - 0.5 * f * f'' / f')`。
5. 高偏心率下可退化为 Newton 步：
   - `E_{k+1} = E_k - f/f'`（分母过小保护）。
6. 轨道几何关系（用于物理解释）：
   - `r = a(1 - e cos(E))`；
   - `nu = 2 atan2(sqrt(1+e) sin(E/2), sqrt(1-e) cos(E/2))`。

## R04

算法流程（高层）：
1. 校验参数（`e` 范围、采样规模、阈值、最大迭代数）。
2. 生成 `M` 网格并加入极小抖动，覆盖 `[0, 2π)`。
3. 依据 `M` 与 `e` 构造初值 `E0`（低阶傅里叶近似 + 高 `e` 偏置）。
4. 执行向量化 Newton/Halley 迭代，记录每个样本迭代次数与收敛掩码。
5. 计算残差 `|f(E)|` 与收敛统计。
6. 用 `scipy.optimize.brentq` 逐点求参考解 `E_ref`。
7. 评估 `|E - E_ref|` 误差、耗时与加速比。
8. 对 `e=0.8` 额外计算 `nu` 和 `r`，确认轨道几何合理。
9. 汇总为 `pandas.DataFrame`，打印结果并执行阈值断言。

## R05

核心数据结构：
- `KeplerParams(dataclass)`：统一管理离心率集合、采样规模、精度阈值与轨道尺度。
- `mean_anomaly: np.ndarray`：平近点角样本。
- `e_fast: np.ndarray`：Newton/Halley 求得的偏近点角。
- `e_ref: np.ndarray`：`brentq` 参考解。
- `n_iters: np.ndarray`：每个样本迭代次数。
- `converged: np.ndarray[bool]`：收敛掩码。
- `detail: pandas.DataFrame`：分离心率诊断表。
- `summary: pandas.DataFrame`：全局最坏指标汇总表。

## R06

正确性检查点：
- 方程一致性：`max_abs_residual = max|E - e sin(E) - M|` 应足够小。
- 参考一致性：`max_abs_error_vs_brentq = max|E - E_ref|` 应足够小。
- 收敛一致性：所有样本都应在 `max_iter` 内收敛。
- 轨道几何一致性（示例 `e=0.8`）：
  - 近心距 `r_min ~= a(1-e)=0.2`；
  - 远心距 `r_max ~= a(1+e)=1.8`。

脚本内置断言：
- `global_min_converged_ratio >= 1.0`
- `global_worst_abs_residual <= 1e-10`
- `global_worst_abs_error_vs_brentq <= 1e-10`

## R07

复杂度分析：
- 主算法（向量迭代）：
  - 设样本数 `N`、迭代轮数 `K`，时间复杂度 `O(N*K)`，空间复杂度 `O(N)`。
- 参考解（`brentq`）：
  - 对每个样本做标量根求解，整体 `O(N*K_ref)`，常数显著更大。
- 本实现对每个 `e` 的规模很小（`N=360`），适合教学、验算与基准测试。

## R08

边界与异常处理：
- `e` 不在 `[0,1)`：抛 `ValueError`。
- `n_mean_anomalies < 64`：抛 `ValueError`（诊断稳定性不足）。
- `tol <= 0` 或 `max_iter < 3`：抛 `ValueError`。
- `semi_major_axis <= 0`：抛 `ValueError`。
- 迭代分母接近 0 时自动回退到 Newton 步，避免 Halley 分母奇异。
- `M` 统一规范到 `[0,2π)`，避免周期边界导致的误判。

## R09

MVP 取舍：
- 只覆盖椭圆轨道（`e < 1`），不实现抛物/双曲版本。
- 只做“给定 `M,e` 求 `E`”这一核心子问题，不扩展到完整轨道传播器框架。
- 参考解选用稳健的 `brentq`，用于校验，不作为主算法。
- 优先保证透明、可审计的算法流程，而非追求工程大而全。

## R10

`demo.py` 函数职责：
- `validate_params`：参数合法性校验。
- `normalize_mean_anomaly`：将 `M` 归一化到 `[0, 2π)`。
- `initial_guess`：构造迭代初值（低阶近似 + 高偏心修正）。
- `kepler_residual`：计算残差 `f(E)`。
- `solve_kepler_newton_halley`：向量化迭代求解主算法。
- `solve_kepler_reference_brentq`：逐点参考解（数值金标准）。
- `true_anomaly_from_eccentric`：`E -> nu` 转换。
- `orbital_radius`：由 `E,e,a` 计算半径。
- `build_report`：构造分组与全局诊断表。
- `main`：运行、打印并执行断言门槛。

## R11

运行方式：

```bash
cd Algorithms/物理-天体力学-0141-开普勒方程_(Kepler's_Equation)
uv run python demo.py
```

脚本不读取命令行参数，也不请求交互输入。

## R12

本地实测输出（2026-04-07）：

```text
=== Kepler Equation MVP (Elliptical Orbit, 0 <= e < 1) ===
params: KeplerParams(e_values=(0.0, 0.2, 0.5, 0.8, 0.9, 0.99), n_mean_anomalies=360, tol=1e-12, max_iter=20, random_seed=42, semi_major_axis=1.0)

[Per-e report]
 eccentricity_e    n_samples  converged_ratio  mean_iterations  max_iterations  max_abs_residual  mean_abs_residual  max_abs_error_vs_brentq  mean_abs_error_vs_brentq  solver_time_ms  reference_time_ms  speedup_vs_ref  nu_span_for_e=0.8  radius_min_for_e=0.8  radius_max_for_e=0.8
   0.000000e+00 3.600000e+02     1.000000e+00     1.000000e+00    1.000000e+00      0.000000e+00       0.000000e+00             4.440892e-16              3.222733e-17    2.666248e-01       5.420458e+00    2.032991e+01                NaN                   NaN                   NaN
   2.000000e-01 3.600000e+02     1.000000e+00     2.916667e+00    3.000000e+00      8.881784e-16       2.868076e-17             1.421085e-14              1.003272e-15    7.412490e-02       5.176750e+00    6.983821e+01                NaN                   NaN                   NaN
   5.000000e-01 3.600000e+02     1.000000e+00     2.994444e+00    3.000000e+00      8.881784e-16       7.316678e-17             1.243450e-14              7.015785e-16    6.150012e-02       5.420833e+00    8.814346e+01                NaN                   NaN                   NaN
   8.000000e-01 3.600000e+02     1.000000e+00     3.611111e+00    4.000000e+00      8.881784e-16       1.312800e-16             1.332268e-14              8.243407e-16    7.616612e-02       5.535791e+00    7.268049e+01       6.023460e+00          2.000000e-01          1.800000e+00
   9.000000e-01 3.600000e+02     1.000000e+00     3.708333e+00    5.000000e+00      8.881784e-16       1.197153e-16             1.243450e-14              9.095354e-16    8.995901e-02       5.752625e+00    6.394718e+01                NaN                   NaN                   NaN
   9.900000e-01 3.600000e+02     1.000000e+00     3.755556e+00    6.000000e+00      8.881784e-16       1.531772e-16             1.243450e-14              9.604851e-16    1.024581e-01       6.069500e+00    5.923888e+01                NaN                   NaN                   NaN

[Global summary]
                          metric        value
       global_worst_abs_residual 8.881784e-16
global_worst_abs_error_vs_brentq 1.421085e-14
           global_max_iterations 6.000000e+00
      global_min_converged_ratio 1.000000e+00
            n_eccentricity_cases 6.000000e+00
              n_samples_per_case 3.600000e+02
```

结论：在 `e=0.0~0.99` 范围内全部样本收敛，残差和参考误差均处于双精度舍入量级。

## R13

建议测试集：
- 基线：默认参数全量运行，检查断言通过。
- 高偏心极限：加入 `e=0.999`，观察 `max_iterations` 与误差变化。
- 低迭代压力测试：将 `max_iter` 暂降到 `4`，确认高 `e` 时可能触发失败。
- 精度敏感性：将 `tol` 从 `1e-12` 放宽到 `1e-8`，确认误差上升趋势。
- 样本密度敏感性：将 `n_mean_anomalies` 改为 `64/720/2048`，比较耗时和稳定性。

异常测试建议：
- 设置 `e=1.0`（应抛错）；
- 设置 `tol<=0`（应抛错）；
- 设置 `n_mean_anomalies=16`（应抛错）。

## R14

关键可调参数：
- `e_values`：覆盖的轨道偏心率范围。
- `n_mean_anomalies`：每组 `e` 的 `M` 样本量。
- `tol`：迭代步长收敛阈值。
- `max_iter`：最大迭代轮数。
- `semi_major_axis`：几何派生量中的轨道尺度。

调参建议：
- 若高偏心率收敛变慢，先提高 `max_iter`；
- 若要更快执行可降低 `n_mean_anomalies`；
- 若只关注工程精度，可放宽 `tol` 到 `1e-10` 或 `1e-9`。

## R15

方法对比：
- 本实现（Newton/Halley）：
  - 优点：公式显式、易向量化、速度快；
  - 风险：需要较好初值和分母保护。
- `brentq`（参考方法）：
  - 优点：单峰单根场景鲁棒性高；
  - 缺点：逐点标量求根，速度明显慢于向量迭代。
- 纯 Newton（不带 Halley）：
  - 实现更简单，但在高偏心率下迭代次数通常略高。

本条目采用 “Newton/Halley 主求解 + brentq 基准对照” 的折中方案。

## R16

应用场景：
- 天体力学/轨道力学课程中的“从平均近点角到轨道位置”计算链路演示。
- 卫星、行星历表中的相位推进基础模块。
- Lambert 问题、摄动模型前的核心子步骤验证。
- 轨道仿真系统中快速批量求解 `E(M,e)` 的前处理阶段。

## R17

可扩展方向：
- 增加双曲开普勒方程（`M = e sinh(H) - H, e>1`）。
- 增加抛物轨道近似（Barker 方程）并统一接口。
- 在批量任务中引入 NumPy/JAX/GPU 向量化版本。
- 增加与 Vallado/Battin 常见初值策略的对比基准。
- 输出轨道平面坐标 `(x,y)` 与速度 `(vx,vy)`，形成完整状态传播器。

## R18

`demo.py` 源码级算法流（8 步，非黑盒）：
1. `main` 创建 `KeplerParams`，`validate_params` 对离心率范围、采样规模、阈值做门禁校验。  
2. `build_report` 生成 `M` 样本并调用 `normalize_mean_anomaly` 映射到 `[0, 2π)`。  
3. `initial_guess` 计算显式初值（低阶展开 + 高偏心率偏置），避免把迭代起点交给黑盒。  
4. `solve_kepler_newton_halley` 在每轮显式计算 `f, f', f''`，按 Halley 更新；若分母不稳定回退到 Newton 步。  
5. 迭代内记录 `converged` 与 `n_iters`，达到阈值即停止；最终输出 `E`、迭代次数和收敛掩码。  
6. `solve_kepler_reference_brentq` 对同一批样本逐点求参考根 `E_ref`，作为数值正确性对照。  
7. `build_report` 计算残差、误差、耗时、加速比，并通过 `true_anomaly_from_eccentric`、`orbital_radius` 生成物理几何诊断。  
8. `main` 打印 `detail/summary` 表格并执行三条断言（收敛率、残差、参考误差），脚本运行即可自动验收。  
