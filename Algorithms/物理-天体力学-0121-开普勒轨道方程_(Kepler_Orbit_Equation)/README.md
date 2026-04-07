# 开普勒轨道方程 (Kepler Orbit Equation)

- UID: `PHYS-0121`
- 学科: `物理`
- 分类: `天体力学`
- 源序号: `121`
- 目标目录: `Algorithms/物理-天体力学-0121-开普勒轨道方程_(Kepler_Orbit_Equation)`

## R01

开普勒轨道方程用于连接“平均近点角”与“轨道几何位置”，是两体椭圆轨道传播中的核心方程之一：

`M = E - e sin(E)`

其中：
- `M` 是平均近点角（mean anomaly）；
- `E` 是偏近点角（eccentric anomaly）；
- `e` 是轨道离心率，椭圆轨道范围为 `0 <= e < 1`。

本目录 MVP 的目标是：给定 `M, e, a`（半长轴），数值求解 `E`，并进一步计算真近点角 `nu`、轨道半径 `r` 与轨道平面坐标 `(x, y)`。

## R02

MVP 要解决的具体计算任务：

输入：
- 一组 `M`（弧度制）；
- 单个椭圆离心率 `e`；
- 半长轴 `a`。

输出：
- 每个采样点的 `E`；
- 由 `E` 推导的 `nu`、`r`、`x`、`y`；
- 数值残差 `|E - e sin(E) - M|`；
- 与 `scipy.optimize.newton` 参考解的误差对比。

## R03

采用该设定的原因：
- 开普勒方程是超越方程，通常没有闭式初等解，必须数值迭代；
- Newton 迭代在本问题上高效且实现简单，适合作为可解释 MVP；
- 增加二分法兜底可提升高偏心率情况下的鲁棒性；
- 输出几何量（`nu/r/x/y`）可直接验证“方程求解”是否真正落到轨道状态上。

## R04

核心公式：

1. 开普勒方程

`f(E) = E - e sin(E) - M = 0`

2. Newton 更新

`E_{k+1} = E_k - f(E_k)/f'(E_k)`

`f'(E) = 1 - e cos(E)`

3. 真近点角

`tan(nu/2) = sqrt((1+e)/(1-e)) * tan(E/2)`

实现中使用更稳健的 `atan2` 形式：

`sin(nu) = sqrt(1-e^2) sin(E) / (1 - e cos(E))`

`cos(nu) = (cos(E)-e) / (1 - e cos(E))`

4. 轨道半径与轨道平面坐标

`r = a(1 - e cos(E))`

`x = a(cos(E)-e)`

`y = a sqrt(1-e^2) sin(E)`

## R05

算法流程（高层）：

1. 对输入 `M` 做角度归一化到 `[0, 2pi)`。
2. 依据离心率给 Newton 选择初值（低偏心用 `M`，高偏心用 `pi`）。
3. 执行向量化 Newton 迭代，直到 `|delta| <= tol` 或达到上限。
4. 对未收敛点逐个调用二分法（区间 `[0, 2pi]`）做兜底。
5. 得到 `E` 后计算 `nu`、`r`、`x`、`y`。
6. 计算方程残差并统计收敛迭代次数。
7. 用 SciPy 的 `newton` 逐点求参考解，检查两者一致性。
8. 打印结果表和验收指标。

## R06

正确性与实现一致性要点：
- 目标函数直接实现为 `f(E)=E-e sin(E)-M`，残差定义与理论一致；
- Newton 使用解析导数 `1-e cos(E)`，避免数值差分误差；
- 二分法依赖椭圆轨道单调性（`E-e sin(E)` 在 `e<1` 时严格递增）保证收敛；
- 通过 `max_residual` 与 `max_abs_diff_vs_scipy` 两个指标双重校验结果可信性。

## R07

复杂度分析（设样本数 `N`，Newton 迭代上限 `K`）：

- Newton 主流程：`O(N*K)`；
- 二分兜底（最坏每点 `B` 次）：`O(N*B)`，通常只影响少数难点；
- 轨道几何推导与残差计算：`O(N)`；
- SciPy 参考解（用于验证）：约 `O(N*K_ref)`。

整体为线性于采样点数量的数值流程，适合小到中规模轨道采样演示。

## R08

边界与异常处理：
- 对 `e` 做范围约束，超出 `0 <= e < 1` 直接报错（本 MVP 不覆盖抛物线/双曲线）；
- 对 `M` 与 `e` 检查有限值，拒绝 NaN/Inf；
- Newton 未收敛的点自动进入二分法兜底，避免静默失败；
- 所有角度统一弧度制，输出前保持一致单位，减少混用错误。

## R09

MVP 取舍说明：
- 聚焦二维轨道平面，不扩展到三维轨道根数转换（`i/Ω/ω`）；
- 仅实现椭圆轨道，不实现双曲线开普勒方程；
- 使用 SciPy 仅作“结果比对”，主算法仍由源码显式实现；
- 不引入复杂天体历元系统，优先保证核心数值链路清晰可运行。

## R10

`demo.py` 主要函数职责：
- `validate_inputs`：输入合法性检查；
- `normalize_angle_rad`：角度归一化；
- `kepler_residual`：计算方程残差；
- `solve_kepler_newton`：向量化 Newton 主求解器；
- `solve_kepler_bisection_scalar`：单点二分兜底；
- `eccentric_to_true_anomaly`：由 `E` 转 `nu`；
- `orbital_radius / perifocal_coordinates`：计算 `r/x/y`；
- `solve_kepler_scipy_reference`：SciPy 参考解；
- `build_orbit_table`：组装表格与诊断指标；
- `main`：运行两个测试轨道并做验收判断。

## R11

运行方式：

```bash
cd Algorithms/物理-天体力学-0121-开普勒轨道方程_(Kepler_Orbit_Equation)
uv run python demo.py
```

脚本无交互输入，执行后会自动输出两个轨道案例（低偏心与高偏心）的结果摘要与校验指标。

## R12

输出字段说明：
- `M_rad`：平均近点角；
- `E_rad`：偏近点角（数值解）；
- `nu_rad`：真近点角；
- `r`：轨道半径；
- `x, y`：轨道平面（perifocal）坐标；
- `residual_abs`：开普勒方程绝对残差；
- `iter`：该点迭代次数；
- `max_residual`：全样本最大残差；
- `max_abs_diff_vs_scipy`：与 SciPy 参考解最大角度差；
- `all_converged`：是否所有采样点都收敛。

## R13

最小测试与验收项（脚本内已自动执行）：
- 案例 1：`e=0.0167`（近圆轨道）；
- 案例 2：`e=0.90`（高偏心椭圆）；
- 检查 `max_residual <= 1e-10`；
- 检查 `max_abs_diff_vs_scipy <= 1e-10`；
- 检查 `all_converged=True`；
- 若任一失败，脚本抛出 `RuntimeError`，便于自动化验证。

## R14

关键参数与调参建议：
- `tol`：收敛阈值；越小精度越高但迭代更多；
- `max_iter`：Newton 最大迭代次数；高偏心率场景可适当增大；
- `max_iter`（二分法）：兜底收敛保障，可保持较大默认值；
- 初值策略：`e<0.8` 用 `E0=M` 通常很快，`e` 较大时 `E0=pi` 更稳；
- `n_samples`：采样密度，影响轨道离散精细程度与运行时间。

## R15

与其他方案对比：
- 纯二分法：稳定但收敛阶低，速度慢；
- 纯 Newton：速度快但在极端初值下可能不稳；
- Newton + 二分兜底（本 MVP）：在保持速度的同时提升鲁棒性；
- 直接调用库函数：实现短但可解释性弱，本目录避免黑箱化并保留算法细节。

## R16

典型应用场景：
- 两体轨道传播中的轨道位置计算；
- 卫星任务中的轨道采样与可视化预处理；
- 天体力学教学中的“平均近点角 -> 真位置”演示；
- 轨道机动前后的快速状态估计基线模型。

## R17

可扩展方向：
- 扩展到双曲线开普勒方程（`e>1`）与抛物线近似；
- 增加三维轨道根数到惯性系坐标变换；
- 按时间 `t` 与平均运动 `n` 自动生成 `M(t)`；
- 提供批量高性能实现（Numba/CUDA）；
- 接入观测数据反演轨道参数（轨道确定问题）。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 定义两个测试轨道（近圆与高偏心），并设置采样数量。
2. `build_orbit_table` 先生成等间隔 `M` 网格，再调用 `solve_kepler_newton` 求解 `E`。
3. `solve_kepler_newton` 对 `M` 归一化，构造初值，进入向量化 Newton 循环。
4. 每次循环计算 `residual=f(E)` 与导数 `f'(E)`，按 `E <- E - f/f'` 更新并记录收敛位。
5. 循环结束后，若存在未收敛点，逐点调用 `solve_kepler_bisection_scalar` 在 `[0,2pi]` 内继续求根。
6. 得到 `E` 后，调用 `eccentric_to_true_anomaly`、`orbital_radius`、`perifocal_coordinates` 推导 `nu/r/x/y`。
7. 计算 `residual_abs`，并用 `solve_kepler_scipy_reference` 生成参考 `E_ref`，比较最大误差。
8. `main` 打印结果表与指标，若残差或参考误差超阈值则抛异常，否则输出 `All checks passed.`。
