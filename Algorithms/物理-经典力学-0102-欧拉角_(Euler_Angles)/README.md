# 欧拉角 (Euler Angles)

- UID: `PHYS-0102`
- 学科: `物理`
- 分类: `经典力学`
- 源序号: `102`
- 目标目录: `Algorithms/物理-经典力学-0102-欧拉角_(Euler_Angles)`

## R01

欧拉角用于描述刚体在三维空间中的姿态，是经典力学中处理转动问题的基础参数化方法之一。
本条目选择常见的 `ZXZ` 内禀（intrinsic）欧拉角约定：
- `phi`：绕空间 `Z` 轴的第一次旋转；
- `theta`：绕新 `X` 轴的第二次旋转；
- `psi`：绕新 `Z` 轴的第三次旋转。

MVP 目标是构建一个可运行且可审计的最小实现，覆盖“表示-反解-动力学量”三件事。

## R02

MVP 解决的具体任务：

1. 给定 `phi, theta, psi`，计算旋转矩阵 `R`（手写公式，不仅依赖库）。
2. 给定旋转矩阵 `R`，反解回一组 `ZXZ` 欧拉角（含奇异位姿处理）。
3. 给定欧拉角及其时间导数，计算刚体坐标系下角速度 `omega`。
4. 对一个时间轨迹批量计算角速度与转动动能，输出结构化表格。

## R03

为什么选这个设定：

- `ZXZ` 是经典刚体动力学教材中的标准形式，能直接对应章动/进动语义。
- 仅做姿态变换不够，结合角速度与动能更贴近“经典力学算法”而非纯几何变换。
- 欧拉角存在万向节锁（gimbal lock）奇异性，适合展示边界处理与工程自检。

## R04

核心数学模型：

1. 旋转分解

`R = Rz(phi) * Rx(theta) * Rz(psi)`

其中：

`Rz(a) = [[cos a, -sin a, 0], [sin a, cos a, 0], [0, 0, 1]]`

`Rx(b) = [[1, 0, 0], [0, cos b, -sin b], [0, sin b, cos b]]`

2. 角速度映射（刚体坐标系分量）

`wx = phi_dot * sin(theta) * sin(psi) + theta_dot * cos(psi)`

`wy = phi_dot * sin(theta) * cos(psi) - theta_dot * sin(psi)`

`wz = phi_dot * cos(theta) + psi_dot`

3. 转动动能

`T = 0.5 * omega^T * I * omega`

其中 `I` 为主惯量对角矩阵。

## R05

算法流程（高层）：

1. 输入一组 `ZXZ` 欧拉角，按显式矩阵公式计算 `R`。
2. 对 `R` 执行反解得到 `(phi, theta, psi)`，并判断是否处于奇异区。
3. 使用同一组状态与角速度导数计算解析 `omega`。
4. 用中心差分近似 `Rdot`，通过 `R^T * Rdot` 提取数值 `omega`，与解析值比对。
5. 构造一段时间轨迹，批量计算 `omega` 与 `T`。
6. 将轨迹结果整理为 `pandas.DataFrame` 并输出统计摘要。

## R06

正确性与一致性检查：

- 正交性：检查 `||R^T R - I||` 是否接近 0；
- 行列式：检查 `det(R)` 是否接近 `1`；
- 库对照：手写矩阵与 `scipy.spatial.transform.Rotation.from_euler('ZXZ', ...)` 的差值；
- 反解回代：`R -> angles -> R_rebuild` 的矩阵误差；
- 角速度一致性：解析 `omega` 与数值微分提取 `omega` 的误差。

## R07

复杂度分析：

- 单次角度到矩阵：常数规模矩阵乘法，时间 `O(1)`，空间 `O(1)`。
- 单次矩阵到角度反解：常数规模三角函数与分支判断，时间 `O(1)`，空间 `O(1)`。
- 轨迹批处理（长度 `N`）：时间 `O(N)`，空间 `O(N)`。

该 MVP 的成本主要来自打印和表格构建，不在算法核心处。

## R08

边界与异常处理：

- `matrix_to_euler_zxz` 对输入形状做 `3x3` 校验；
- 对 `R[2,2]` 做 `clip([-1,1])`，避免浮点误差导致 `arccos` 域错误；
- 当 `sin(theta)` 很小（接近 0）时标记 `singular=True`，启用退化分支；
- 角度统一包装到 `[-pi, pi)`，避免多值表达导致日志不可读；
- 所有最终检查项都输出布尔值，便于自动验证。

## R09

MVP 取舍：

- 只实现一种明确约定（`ZXZ`），避免多约定混杂引入歧义；
- 仅做刚体坐标系角速度映射，不扩展到空间坐标系全套推导；
- 不引入符号推导框架，保持代码短小可直接运行；
- 使用 `scipy` 仅作数值对照，不作为主算法黑盒。

## R10

`demo.py` 主要函数：

- `rotation_z` / `rotation_x`：基础轴旋转矩阵；
- `euler_zxz_to_matrix`：手写 `ZXZ` 欧拉角到旋转矩阵；
- `matrix_to_euler_zxz`：旋转矩阵反解欧拉角（含奇异分支）；
- `euler_rates_to_body_omega`：欧拉角速度到刚体角速度；
- `omega_from_matrix_derivative`：通过矩阵导数提取角速度用于校验；
- `trajectory_state`：给定时间返回角度与角速度导数；
- `batch_kinetic_energy_torch`：批量动能计算（PyTorch）；
- `build_trajectory_table`：生成轨迹结果表；
- `main`：组织实验、打印检查项与摘要。

## R11

运行方式：

```bash
cd Algorithms/物理-经典力学-0102-欧拉角_(Euler_Angles)
uv run python demo.py
```

脚本无需任何交互输入。

## R12

输出字段说明：

- `orthogonality_error`：旋转矩阵正交误差；
- `determinant`：旋转矩阵行列式；
- `manual_vs_scipy_error`：手写矩阵与 SciPy 对照误差；
- `roundtrip_matrix_error`：反解再构造的矩阵误差；
- `omega_formula_vs_numeric_error`：解析角速度与数值角速度差；
- 轨迹表中的 `wx, wy, wz`：刚体系角速度分量；
- `omega_norm`：角速度模长；
- `kinetic_energy`：给定惯量下转动动能。

## R13

最小验收项：

- 脚本能直接运行结束（无输入、无异常）；
- 输出中 `all_core_checks_pass=True`；
- 轨迹表中 `all_energy_finite=True`；
- 奇异位姿测试返回 `singular_detected=True`。

## R14

关键参数与调节建议：

- `eps`（奇异阈值）：默认 `1e-10`，若噪声更大可放宽；
- `dt`（数值微分步长）：过大误差高，过小受浮点噪声影响，默认 `1e-6`；
- `inertia_diag`：主惯量设置会显著影响能量量级；
- 轨迹函数振幅应避免长期贴近奇异区，否则角度反解的多值性会更明显。

## R15

与其他姿态表示对比：

- 欧拉角：参数少、物理语义直观，但有奇异性；
- 旋转矩阵：无奇异、合成方便，但参数冗余（9 个数 + 约束）；
- 四元数：数值稳定、插值友好，但物理解释不如欧拉角直观。

本条目强调“经典力学可解释性”，因此以欧拉角为主，矩阵作核心计算载体。

## R16

典型应用场景：

- 刚体姿态与角速度建模（陀螺、飞行器、航天器姿态）；
- 拉格朗日方程中广义坐标选择与动能构造；
- 需要把实验/仿真姿态数据转为可解释角变量的分析任务。

## R17

可扩展方向：

- 增加 `ZYX`（yaw-pitch-roll）等其他约定并统一接口；
- 引入姿态积分器（欧拉法、RK4）实现完整转动仿真；
- 联立欧拉动力学方程 `I*omega_dot + omega x (I*omega)=tau`；
- 扩展到噪声观测下的姿态估计（EKF/UKF）。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `main` 固定一组欧拉角样例，调用 `euler_zxz_to_matrix` 通过 `Rz*Rx*Rz` 显式构造旋转矩阵。
2. 计算 `R^T R` 与 `det(R)`，验证该矩阵确为合法旋转。
3. 用 `scipy Rotation.from_euler('ZXZ')` 生成对照矩阵，比较与手写实现的误差，确保不是黑盒替代。
4. 对手写矩阵调用 `matrix_to_euler_zxz` 反解角度：先由 `R33` 得 `theta`，再在非奇异/奇异两分支下分别恢复 `phi, psi`。
5. 将反解角度再送回 `euler_zxz_to_matrix`，检查回代误差（round-trip consistency）。
6. 用 `euler_rates_to_body_omega` 根据解析公式计算角速度，同时用 `omega_from_matrix_derivative` 通过中心差分与 `R^T Rdot` 提取数值角速度。
7. 调用 `build_trajectory_table`：沿时间网格生成姿态与角速度，使用 `batch_kinetic_energy_torch` 计算 `T=0.5*omega^T I omega`，并整理成 `pandas` 表格。
8. 输出核心误差、轨迹统计、奇异位姿检测和布尔验收结论，形成可自动检查的最小闭环。
