# 四元数旋转 (Quaternion Rotation)

- UID: `PHYS-0103`
- 学科: `物理`
- 分类: `经典力学`
- 源序号: `103`
- 目标目录: `Algorithms/物理-经典力学-0103-四元数旋转_(Quaternion_Rotation)`

## R01

四元数旋转是三维刚体姿态表示与向量旋转的经典方法，广泛用于经典力学、机器人、航天姿态控制与图形学。

相较欧拉角，四元数的核心优势是：
- 无万向节锁奇异性；
- 旋转复合可通过代数乘法完成；
- 数值积分与插值更稳定。

本条目给出一个“可运行且可审计”的最小 MVP：手写四元数代数与旋转流程，再用 `scipy/sklearn/torch` 做对照验证，而不是把旋转逻辑完全交给库黑盒。

## R02

本目录 MVP 解决的问题：

1. 将轴角 `(axis, angle)` 转换为单位四元数 `q=[w,x,y,z]`（标量在前约定）。
2. 手写四元数乘法、共轭、归一化与向量旋转 `v' = q * [0,v] * q_conj`。
3. 手写四元数到旋转矩阵 `R(q)` 的显式公式。
4. 验证旋转组合律：先后两次旋转等价于一次组合四元数旋转。
5. 在统一角速度下生成刚体轨迹，计算速度与动能，形成表格输出。

## R03

数学定义（标量在前）：

- 四元数写作 `q = [w, x, y, z] = [w, u]`，其中 `u in R^3`。
- 单位四元数满足 `||q||=1`。
- 轴角到四元数：
  - `q = [cos(theta/2), sin(theta/2) * n_x, sin(theta/2) * n_y, sin(theta/2) * n_z]`，`n` 为单位轴。
- 向量旋转：
  - 把 `v` 嵌入纯四元数 `p=[0,v]`；
  - 计算 `p' = q * p * q^{-1}`，单位四元数时 `q^{-1}=q_conj`；
  - 取 `p'` 的虚部即旋转后向量。

## R04

四元数乘法（Hamilton Product）：

若 `q1=[w1,x1,y1,z1]`，`q2=[w2,x2,y2,z2]`，则

- `w = w1*w2 - x1*x2 - y1*y2 - z1*z2`
- `x = w1*x2 + x1*w2 + y1*z2 - z1*y2`
- `y = w1*y2 - x1*z2 + y1*w2 + z1*x2`
- `z = w1*z2 + x1*y2 - y1*x2 + z1*w2`

`demo.py` 中该公式完全手写实现，不调用库乘法接口。

## R05

四元数到旋转矩阵（主动旋转）显式公式：

设 `q=[w,x,y,z]`，则

`R11 = 1 - 2(y^2 + z^2)`  
`R12 = 2(xy - wz)`  
`R13 = 2(xz + wy)`  

`R21 = 2(xy + wz)`  
`R22 = 1 - 2(x^2 + z^2)`  
`R23 = 2(yz - wx)`  

`R31 = 2(xz - wy)`  
`R32 = 2(yz + wx)`  
`R33 = 1 - 2(x^2 + y^2)`

脚本会检查 `R^T R≈I` 与 `det(R)≈1`，并与 SciPy 结果做数值对照。

## R06

算法高层流程：

1. 读取固定测试轴角，生成单位四元数；
2. 旋转一组 3D 标记点（手写公式）；
3. 用 SciPy 对同一点集旋转，计算误差；
4. 由四元数手写生成旋转矩阵并做正交性与行列式检查；
5. 构造两段旋转，验证“顺序旋转 == 组合四元数”；
6. 用 Torch 再做一遍批量旋转，验证与 NumPy 一致；
7. 构造匀角速度轨迹并输出动能表；
8. 打印所有布尔验收项与总体结论。

## R07

核心数据结构：

- `q: np.ndarray(shape=(4,))`：四元数，约定 `[w,x,y,z]`。
- `points: np.ndarray(shape=(N,3))`：待旋转点集。
- `traj: pandas.DataFrame`：轨迹表，字段包含：
  - `t, angle`
  - `qw,qx,qy,qz`
  - `rx,ry,rz`
  - `speed, kinetic_energy`
- `checks: Dict[str, bool]`：核心一致性检查结果。

## R08

正确性校验设计：

- 手写点旋转 vs SciPy `Rotation.apply`；
- 手写 `R(q)` vs SciPy `as_matrix`；
- 组合律向量误差：`rotate(rotate(v,q_a),q_b)` vs `rotate(v,q_b*q_a)`；
- 组合律矩阵误差：`R(q_b*q_a)` vs `R(q_b)R(q_a)`；
- Torch 批量旋转 vs NumPy 批量旋转；
- 轨迹中的数值有限性与动能近似守恒性。

所有误差都打印为浮点量，并给出布尔阈值判断。

## R09

复杂度分析：

- 单个向量旋转（四元数夹乘）：`O(1)`；
- `N` 个点批量旋转：`O(N)`；
- 四元数转矩阵、组合律验证：`O(1)`；
- 轨迹长度为 `T` 时：
  - 时间复杂度 `O(T)`；
  - 存储轨迹表空间复杂度 `O(T)`。

MVP 成本主要在批量点旋转和轨迹表构建。

## R10

边界与异常处理：

- 向量维度严格检查（轴必须 3 维、四元数必须 4 维、点集必须 `N x 3`）；
- 输入包含 `nan/inf` 时抛 `ValueError`；
- 轴范数过小或四元数范数过小拒绝继续；
- `mass <= 0` 直接报错；
- 所有检查均在函数入口处完成，避免静默失败。

## R11

MVP 取舍说明：

- 只实现一种四元数约定（标量在前）；
- 不展开四元数时间积分器（如 RK4）和姿态估计滤波；
- 轨迹选择“匀角速度 + 单质点”用于演示力学意义（速度与转动动能）；
- SciPy 仅作校验，不承担核心算法逻辑；
- sklearn 只用于误差指标（RMSE）计算，保持最小依赖闭环。

## R12

`demo.py` 主要函数职责：

- `normalize_quaternion`：单位化四元数；
- `quaternion_multiply`：手写 Hamilton 乘法；
- `axis_angle_to_quaternion`：轴角到四元数；
- `quaternion_to_matrix`：手写 `R(q)`；
- `rotate_vector_by_quaternion`：单向量夹乘旋转；
- `rotate_points_numpy`：向量化批量旋转（NumPy）；
- `rotate_points_torch`：Torch 版本批量旋转；
- `compose_active_rotations`：先后旋转的四元数组合；
- `simulate_uniform_rotation`：生成轨迹并计算速度/动能；
- `main`：组织实验、打印结果与验收布尔值。

## R13

运行方式：

```bash
cd Algorithms/物理-经典力学-0103-四元数旋转_(Quaternion_Rotation)
uv run python demo.py
```

脚本不读取命令行参数，也不需要任何交互输入。

## R14

输出字段说明：

- `max_point_error`：手写点旋转与 SciPy 的最大点误差；
- `rmse`：扁平坐标 RMSE（sklearn 计算）；
- `matrix_error_manual_vs_scipy`：手写矩阵与 SciPy 矩阵差；
- `orthogonality_error`：`||R^T R - I||`；
- `determinant`：`det(R)`；
- `vector_composition_error`：组合律向量误差；
- `matrix_composition_error`：组合律矩阵误差；
- `torch_vs_numpy_max_error`：Torch 与 NumPy 批量结果差；
- `kinetic_energy_span`：轨迹中动能极差；
- `all_core_checks_pass`：总验收布尔值。

## R15

最小验收标准（建议）：

- `all_core_checks_pass=True`；
- 关键误差量级保持在机器精度附近（约 `1e-12 ~ 1e-15`）；
- 轨迹表全部为有限数值；
- `determinant` 足够接近 1（不应出现反射或缩放）。

## R16

与其他表示法对比（简述）：

- 欧拉角：参数少、直观，但存在奇异性；
- 旋转矩阵：无奇异但冗余参数多（9 个元素受约束）；
- 四元数：4 参数+单位约束，复合快、数值稳定，工程中常作姿态主表示。

本条目侧重“可审计实现 + 物理轨迹演示”，适合作为刚体姿态模块的基础实现。

## R17

可扩展方向：

1. 增加四元数运动学方程 `q_dot = 0.5 * q * [0,omega]` 的数值积分；
2. 加入外力矩驱动，与欧拉动力学方程联立；
3. 扩展到刚体多点惯量模型并计算角动量守恒；
4. 引入 SLERP 做姿态插值；
5. 加入噪声观测并接入姿态滤波（EKF/UKF）。

## R18

`demo.py` 源码级算法流程（9 步）：

1. `main` 固定轴角参数，调用 `axis_angle_to_quaternion` 构造单位四元数 `q`。
2. 调用 `rotate_points_numpy` 按向量化公式 `v' = v + 2*(s*(u×v) + u×(u×v))` 批量旋转点集。
3. 调用 `scipy_rotation_from_scalar_first(...).apply(...)` 得到参考结果，并用 `mean_squared_error` 计算 RMSE。
4. 调用 `quaternion_to_matrix` 以显式公式生成 `R(q)`，再计算 `orthogonality_error` 与 `determinant`。
5. 构造 `q_a, q_b`，调用 `compose_active_rotations` 得 `q_ab = q_b*q_a`，并在向量层面验证组合律。
6. 在矩阵层面比较 `R(q_ab)` 与 `R(q_b) @ R(q_a)`，确认代数与几何一致。
7. 调用 `rotate_points_torch` 对同一批点做 Torch 旋转，与 NumPy 结果做最大误差比较。
8. 调用 `simulate_uniform_rotation` 生成匀角速度轨迹：每个时刻用四元数旋转质点位置，再由 `v=omega×r` 计算速度与动能，保存为 `DataFrame`。
9. 汇总所有误差与布尔检查项，打印 `all_core_checks_pass` 作为最终验收信号。
