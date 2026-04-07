# 三维变换矩阵

- UID: `CS-0256`
- 学科: `计算机`
- 分类: `计算机图形学`
- 源序号: `413`
- 目标目录: `Algorithms/计算机-计算机图形学-0413-三维变换矩阵`

## R01

三维变换矩阵用于在统一线性代数框架下描述三维几何体的缩放、旋转与平移。  
在图形学管线里，点通常先从局部坐标系经模型变换进入世界坐标，再进入视图/投影空间。  
本条目聚焦“模型空间内的刚性+仿射变换”核心，即 4x4 齐次矩阵及其组合规则。

## R02

本条目要解决的问题：
- 输入：一组三维点 `points (N,3)`，以及变换参数 `scale/rotation/translation`。
- 输出：
  - 对应的 4x4 变换矩阵（缩放、绕 x/y/z 旋转、平移、组合矩阵）。
  - 变换后的点云。
  - 数值自检结果（组合一致性、逆变换恢复误差、非交换性差异）。

`demo.py` 为固定样例，不需要任何交互输入。

## R03

核心数学表示采用齐次坐标：
\[
\tilde{p} = [x, y, z, 1]^T
\]

常见 4x4 变换矩阵：
- 缩放：
\[
S = \begin{bmatrix}
 s_x&0&0&0\\
 0&s_y&0&0\\
 0&0&s_z&0\\
 0&0&0&1
\end{bmatrix}
\]
- 平移：
\[
T = \begin{bmatrix}
 1&0&0&t_x\\
 0&1&0&t_y\\
 0&0&1&t_z\\
 0&0&0&1
\end{bmatrix}
\]
- 旋转：分别由 `R_x(θx), R_y(θy), R_z(θz)` 给出。

本实现采用列向量约定：
\[
\tilde{p}' = M\tilde{p},\quad M = T\,R_z\,R_y\,R_x\,S
\]

## R04

算法高层流程：
1. 校验输入点集维度和数值合法性。
2. 构造 `S, Rx, Ry, Rz, T` 五个基础 4x4 矩阵。
3. 按 `M = T @ Rz @ Ry @ Rx @ S` 组合总变换。
4. 把点 `(N,3)` 扩展成齐次坐标 `(N,4)`。
5. 批量应用总变换得到输出点。
6. 分步应用五个基础矩阵，和组合矩阵结果做一致性比对。
7. 计算 `M^{-1}` 并验证“变换后再逆变换”可恢复原始点。
8. 对比另一种乘法顺序，展示三维变换一般不可交换。

## R05

核心数据结构：
- `points: np.ndarray (N,3)`：输入点云。
- `TransformComponents`：封装 `scale/rot_x/rot_y/rot_z/translation/composed`。
- `matrix: np.ndarray (4,4)`：任一齐次变换矩阵。
- `points_h: np.ndarray (N,4)`：齐次点。

实现使用 `numpy.float64`，便于稳定地做矩阵运算和误差评估。

## R06

正确性要点：
- 齐次坐标保证平移可写成矩阵乘法，能与缩放/旋转统一组合。
- 组合矩阵与分步矩阵若顺序一致，理论上应得到同一结果；代码以 Frobenius 范数误差校验。
- `M` 非奇异时存在逆矩阵；执行 `M` 后再执行 `M^{-1}` 应恢复输入点（数值误差允许微小偏差）。
- 交换变换顺序通常会改变结果，脚本显式构造重排矩阵并计算差异。

## R07

复杂度分析（`N` 个点）：
- 构造基础矩阵与组合矩阵：`O(1)`。
- 一次批量点变换（`(N,4) @ (4,4)^T`）：`O(N)`。
- 逆矩阵求解（4x4）：`O(1)`（常数规模）。
- 总体：`O(N)` 时间，`O(N)` 额外空间（齐次点缓存）。

## R08

边界与异常处理：
- 点集不是 `(N,3)` 或 `N=0`：抛 `ValueError`。
- 点集/参数包含 `nan` 或 `inf`：抛 `ValueError`。
- 齐次坐标反变换时 `w` 过小：抛 `ValueError`。
- 变换矩阵不是 `(4,4)`：抛 `ValueError`。
- 矩阵接近奇异、不可逆：抛 `ValueError`。

## R09

MVP 取舍说明：
- 采用手写 `S/R/T + 齐次坐标`，不依赖图形引擎黑箱 API。
- 依赖最小化，仅使用 `numpy`；不引入渲染窗口、交互 UI 或大型框架。
- 目标是验证“矩阵构造与组合逻辑正确”，而非做完整渲染管线。

## R10

`demo.py` 函数职责：
- `_validate_points` / `_validate_finite_tuple`：输入与参数校验。
- `to_homogeneous` / `from_homogeneous`：普通坐标与齐次坐标互转。
- `scale_matrix_3d` / `rotation_*_matrix` / `translation_matrix_3d`：生成基础矩阵。
- `compose_transform`：按固定顺序组装 `M` 并返回组件集合。
- `apply_transform`：批量点应用 4x4 变换。
- `inverse_transform`：逆矩阵计算与可逆性检查。
- `main`：构造样例、执行变换、输出指标并做断言。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机图形学-0413-三维变换矩阵
uv run python demo.py
```

脚本不读取参数、不请求输入，执行完成后会打印矩阵和误差指标。

## R12

输出字段解释：
- `Composed matrix M`：最终组合矩阵 `T*Rz*Ry*Rx*S`。
- `First 3 transformed points`：变换后点云前 3 个点，便于快速观察数值变化。
- `composition_error`：分步与组合方式差异（应接近 0）。
- `inverse_recovery_error`：逆变换恢复误差（应接近 0）。
- `non_commutativity_gap`：顺序重排前后矩阵差异（一般显著大于 0）。

## R13

建议最小测试集：
- 单位变换（缩放=1、旋转=0、平移=0），输出应与输入一致。
- 仅平移/仅旋转/仅缩放三类单操作测试。
- 组合测试：与分步结果逐点比较。
- 可逆性测试：`apply(M)` 后 `apply(M^{-1})` 还原原点云。
- 非法输入测试：空点集、错误维度、非有限值、奇异矩阵。

## R14

可调参数：
- `scale_xyz`：各轴缩放因子。
- `rotation_deg_xyz`：绕 `x/y/z` 的旋转角度（度）。
- `translation_xyz`：平移向量。
- 误差阈值：断言里使用 `1e-10` 到 `1e-8`。

调参建议：
- 若放大数据规模，可继续使用同样代码路径，复杂度线性增长。
- 若处理极端尺度数据，可根据数值范围适度放宽误差阈值。

## R15

与相关表示的对比：
- 与欧拉角直接逐点三角计算相比：矩阵形式更适合批量计算和链式组合。
- 与四元数相比：
  - 四元数更擅长表示纯旋转与插值；
  - 4x4 矩阵可统一表达旋转、缩放、平移，工程组合更直接。
- 与仅 3x3 旋转矩阵相比：4x4 齐次形式可把平移纳入同一乘法框架。

## R16

典型应用场景：
- 3D 模型的模型矩阵（Model Matrix）构造。
- 机器人/SLAM 中坐标系变换与点云对齐（仿射层面）。
- CAD/数字孪生中几何对象批量变换。
- 计算机视觉中点集刚体近似变换的基础表达。

## R17

可扩展方向：
- 接入视图矩阵与投影矩阵，形成完整 `MVP`（Model-View-Projection）链路。
- 从欧拉角扩展到四元数与轴角表示，减少万向锁风险。
- 增加 `SE(3)` 约束（旋转正交、行列式为 1）和李群/李代数更新。
- 支持批量对象变换、GPU 张量化计算或自动微分场景。

## R18

`demo.py` 源码级算法流程（9 步）：
1. `main` 构造 8 个立方体顶点 `cube (8,3)` 作为固定输入。  
2. `compose_transform` 校验参数后调用 `scale_matrix_3d / rotation_* / translation_matrix_3d` 生成五个基础 4x4 矩阵。  
3. 在 `compose_transform` 中按 `M = T @ Rz @ Ry @ Rx @ S` 组合总矩阵并打包到 `TransformComponents`。  
4. `apply_transform` 先把点经 `to_homogeneous` 扩展成 `(N,4)`，再执行 `points_h @ M.T` 完成批量变换。  
5. `from_homogeneous` 将结果除以 `w` 并还原为 `(N,3)`，得到 `transformed`。  
6. `main` 再按 `S -> Rx -> Ry -> Rz -> T` 逐步调用 `apply_transform`，得到 `stepwise`，并计算 `composition_error`。  
7. `inverse_transform` 对 `M` 做可逆性检查与 `np.linalg.inv`，`main` 用它把 `transformed` 还原为 `recovered`，计算 `recovery_error`。  
8. `main` 构造一个重排顺序矩阵 `translation @ scale @ rot_z @ rot_y @ rot_x`，与 `M` 比较得到 `non_commutativity_gap`。  
9. 若三项误差都满足阈值断言，打印矩阵与指标并输出 `All checks passed.`。  
