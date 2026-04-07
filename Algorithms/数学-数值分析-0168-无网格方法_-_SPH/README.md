# 无网格方法 - SPH

- UID: `MATH-0168`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `168`
- 目标目录: `Algorithms/数学-数值分析-0168-无网格方法_-_SPH`

## R01

本条目实现一个最小可运行的 SPH（Smoothed Particle Hydrodynamics，光滑粒子流体动力学）MVP，目标是把“粒子离散、核函数近似、密度-压强-动量更新”完整打通。

实现定位：
- 2D WCSPH（弱可压缩 SPH）教学版；
- 纯 `numpy` 向量化，避免外部黑盒求解器；
- 通过固定步长时间推进打印可解释诊断信息。

## R02

问题定义（本 MVP 版本）：
- 输入：
  - 初始粒子位置与速度；
  - 物理参数（`rho0, h, mass, stiffness, gravity` 等）；
  - 时间离散参数（`dt, steps`）。
- 输出：
  - 每个时间步更新后的粒子状态 `(x_i, v_i)`；
  - 每步密度 `rho_i` 与压强 `p_i`；
  - 运行日志中的统计量（平均密度、速度峰值、动能、质心）。

边界条件：
- 矩形容器，采用反弹+阻尼处理，不做开边界流出。

## R03

核心数学模型：

1) 密度求和（Summation Density）
- `rho_i = sum_j m_j * W(r_ij, h)`
- `W` 使用 2D cubic spline 核函数，支撑半径 `2h`。

2) 状态方程（线性化）
- `p_i = k * max(rho_i - rho0, 0)`
- 其中 `k` 为刚度系数，`rho0` 为静止密度。

3) 动量方程离散
- `dv_i/dt = - sum_j m_j * (p_i/rho_i^2 + p_j/rho_j^2 + Pi_ij) * grad W_ij + g`
- `Pi_ij` 为 Monaghan 人工黏性项，用于抑制粒子穿透和数值振荡。

4) 时间积分
- 显式更新：
  - `v_i^{n+1} = v_i^n + dt * a_i^n`
  - `x_i^{n+1} = x_i^n + dt * v_i^{n+1}`

## R04

算法流程（单步）：
1. 根据当前粒子位置计算两两距离矩阵。
2. 用核函数 `W` 做密度求和，得到 `rho`。
3. 由状态方程计算压强 `p`。
4. 计算核函数梯度 `grad W`。
5. 组装压强项与人工黏性项，得到每个粒子加速度 `a`。
6. 执行显式速度/位置更新。
7. 执行边界反弹与切向阻尼。
8. 输出或记录诊断信息。

## R05

核心数据结构：
- `SPHConfig(dataclass)`：集中管理物理与数值参数。
- `SPHState(dataclass)`：
  - `positions: ndarray (N,2)`
  - `velocities: ndarray (N,2)`
- 中间张量（向量化计算）：
  - `dx: (N,N,2)` 两两位移；
  - `r, r2: (N,N)` 两两距离与平方距离；
  - `grad: (N,N,2)` 核梯度。

## R06

正确性要点：
- 使用紧支撑核函数，只有 `r < 2h` 的邻域有相互作用，符合 SPH 局部性。
- 压强力采用对称形式 `p_i/rho_i^2 + p_j/rho_j^2`，减少非物理偏置。
- 人工黏性只在粒子相对接近（`v_ij·r_ij < 0`）时启用，避免无必要耗散。
- 每步先由位置计算密度，再由密度计算压强与受力，符合 WCSPH 常规顺序。

## R07

复杂度分析：
- 本实现使用全对全相互作用：
  - 时间复杂度：每步 `O(N^2)`；总计 `O(steps * N^2)`。
  - 空间复杂度：需要 `(N,N,2)` 级别中间数组，约 `O(N^2)`。
- 优点：实现短且透明。
- 局限：不适合大规模粒子（工程中通常用网格哈希/树结构做邻域搜索）。

## R08

边界与稳定性处理：
- 边界：
  - 触墙时将粒子位置投回边界；
  - 法向速度乘 `-restitution` 反弹；
  - 切向速度乘 `tangential_damping` 做耗散。
- 稳定性：
  - 使用较小 `dt`；
  - 用人工黏性缓解粒子穿透与高频噪声；
  - 压强仅对 `rho > rho0` 生效，防止过强负压导致拉伸不稳定。

## R09

MVP 取舍：
- 选用 `numpy` 实现，不依赖专用 SPH 库；
- 不引入空间哈希、XSPH、Riemann 求解等复杂增强模块；
- 保留可读性优先的数据流，便于逐行核对公式；
- 目标是“诚实可运行 + 可解释”，而不是工业级高性能。

## R10

`demo.py` 函数职责：
- `make_initial_block`：生成初始流体粒子块；
- `cubic_spline_kernel_2d`：核函数 `W`；
- `cubic_spline_dWdr_2d`：径向导数 `dW/dr`；
- `pairwise_geometry`：构造两两几何量；
- `compute_density_pressure`：密度与压强；
- `compute_acceleration`：压强力+人工黏性+重力；
- `apply_boundary_conditions`：边界反弹与阻尼；
- `run_step`：执行单步时间推进；
- `summarize`：日志统计；
- `main`：组织整段仿真并打印结果。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0168-无网格方法_-_SPH
python3 demo.py
```

脚本无交互输入，会直接输出从初始到终态的若干诊断行与最终粒子样本。

## R12

输出字段解读：
- `rho_mean`：全粒子平均密度，观察是否围绕 `rho0`；
- `rho_std`：密度离散程度，过大通常意味着数值噪声偏高；
- `vmax`：最大速度，监控稳定性与冲击强度；
- `E_k`：总动能（离散），可观察重力驱动与边界耗散；
- `center`：粒子群质心位置，体现整体流动趋势。

最后的 `Final particle sample` 用于快速检查粒子状态是否有限且落在边界内。

## R13

建议最小测试集：
- 正常运行：默认参数 `steps=220`，检查日志连续输出且无异常。
- 时间步稳定性：把 `dt` 提大到 `0.003`，观察是否出现明显振荡增大。
- 黏性敏感性：把 `alpha_viscosity` 改为 `0` 与 `0.2` 对比 `rho_std/vmax`。
- 边界行为：减小容器尺寸或增大初始块尺寸，确认粒子不会越界飞出。

## R14

关键可调参数：
- 分辨率相关：`particle_spacing`, `block_nx`, `block_ny`；
- 平滑与相互作用：`smoothing_length`；
- 可压缩性：`rest_density`, `stiffness`；
- 数值耗散：`alpha_viscosity`, `viscosity_eps`；
- 时间推进：`dt`, `steps`, `log_interval`；
- 边界响应：`boundary_restitution`, `tangential_damping`。

经验上先固定粒子数，再调整 `h` 与 `dt`，最后微调 `stiffness/viscosity`。

## R15

方法对比：
- 与网格法（FDM/FVM）相比：
  - SPH 无需生成网格，适合大变形/自由液面；
  - 但邻域搜索与核参数调优更敏感。
- 与 PIC/FLIP 相比：
  - SPH 完全拉格朗日、实现直观；
  - PIC/FLIP 常在低数值耗散和大规模场景更有优势。
- 与工业 SPH 实现相比：
  - 本 MVP 缺少邻居加速、边界粒子、张力模型等工程模块。

## R16

应用场景：
- 自由液面流（泼洒、液滴、破坝）原型演示；
- 多相/流固耦合研究的前置基础模型；
- 数值分析课程中“无网格离散”教学示例；
- 算法验证阶段的小规模可解释仿真。

## R17

后续扩展方向：
- 邻域搜索加速：cell-linked list / spatial hashing；
- 更稳健状态方程与时间积分（例如 predictor-corrector）；
- 增加边界粒子与固体交互模型；
- 引入表面张力、粘弹性或多相流模型；
- 输出轨迹到 CSV/VTK 便于可视化。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `main` 构造 `SPHConfig` 并通过 `make_initial_block` 生成初始粒子块与零速度。  
2. 每个时间步进入 `run_step`，先调用 `compute_density_pressure`。  
3. `compute_density_pressure` 内部使用 `pairwise_geometry` 得到两两距离，再用 `cubic_spline_kernel_2d` 做密度求和，得到 `rho` 与 `p`。  
4. `run_step` 调用 `compute_acceleration`，再次取两两几何量并由 `cubic_spline_dWdr_2d` 构造核梯度 `grad W`。  
5. 在 `compute_acceleration` 中按 `p_i/rho_i^2 + p_j/rho_j^2` 组装压强项，并基于 `v_ij·r_ij` 计算 Monaghan 人工黏性 `Pi_ij`。  
6. 将压强项与黏性项合并，做对所有邻域粒子的矢量求和，得到每个粒子加速度，再叠加重力。  
7. `run_step` 用显式公式更新速度与位置，然后调用 `apply_boundary_conditions` 做反弹和切向阻尼。  
8. `main` 依据 `log_interval` 打印 `rho_mean/rho_std/vmax/E_k/center`，最终输出粒子样本，完成一次无交互 SPH 仿真。  
