# 诺特定理 (Noether's Theorem)

- UID: `PHYS-0147`
- 学科: `物理`
- 分类: `理论物理`
- 源序号: `147`
- 目标目录: `Algorithms/物理-理论物理-0147-诺特定理_(Noether's_Theorem)`

## R01

诺特定理的核心结论是：
- 每一个连续对称性，对应一个守恒量；
- 在拉格朗日系统里，这个对应关系可写成可计算公式，而不是仅停留在概念层面。

本条目用一个最小数值 MVP 把该结论落地：
- 平移对称性 `x -> x + eps` 对应线动量守恒；
- 时间平移对称性对应能量守恒；
- 旋转对称性对应角动量守恒；
- 当旋转对称性被刻意破坏时（各向异性势），角动量不再守恒。

## R02

要解决的问题是：
如何在不依赖黑箱符号系统的前提下，用可运行代码验证“对称性 <-> 守恒量”的一一对应。

MVP 验证目标：
1. 对自由粒子验证平移不变性与 `p_x` 守恒；
2. 对二维各向同性简谐振子验证旋转不变性与 `L_z` 守恒；
3. 对同一体系验证时间平移不变性与总能量 `E` 守恒；
4. 构造二维各向异性振子作为反例，展示旋转不变性破坏后 `L_z` 漂移显著。

## R03

采用的理论框架（机械系统）：

- 拉格朗日量
`L(q, q_dot, t) = T(q_dot) - V(q, t)`

- 诺特电荷（`delta t = 0` 的常见情形）
`Q = sum_i p_i * Xi_i(q, t)`，其中 `p_i = dL/dq_dot_i`，`delta q_i = eps * Xi_i`

在本条目中的三个生成元：
- 平移：`Xi = (1)`，`Q = p_x`；
- 平面旋转：`Xi = (-y, x)`，`Q = x p_y - y p_x = L_z`；
- 时间平移：对应哈密顿量 `H = E`。

## R04

MVP 选用三组可解析/可检验模型：

1. 一维自由粒子
`L = 1/2 m v^2`
- 与 `x` 无关，平移不变；
- 预期 `p_x = m v` 守恒。

2. 二维各向同性简谐振子
`L = 1/2 m (vx^2 + vy^2) - 1/2 k (x^2 + y^2)`
- 势能只依赖 `r^2`，旋转不变；
- 预期 `E` 与 `L_z` 都守恒。

3. 二维各向异性简谐振子
`L = 1/2 m (vx^2 + vy^2) - 1/2 (kx x^2 + ky y^2), kx != ky`
- 仍然时间平移不变（`E` 守恒）；
- 旋转不变性破坏（`L_z` 不守恒）。

## R05

数值离散策略：
- 自由粒子：直接离散更新 `x_{n+1} = x_n + v_n dt`；
- 振子：使用 Velocity-Verlet（辛格式），更适合长时守恒量测试。

Velocity-Verlet 步进：
1. `q_{n+1} = q_n + v_n dt + 1/2 a_n dt^2`
2. `a_{n+1} = -grad V(q_{n+1}) / m`
3. `v_{n+1} = v_n + 1/2 (a_n + a_{n+1}) dt`

## R06

代码中的“对称性检测”不做黑箱推断，而是显式有限差分：

- 平移残差（自由粒子）
`res_translation = |L(x+eps, v) - L(x, v)| / eps`

- 旋转残差（二维系统）
`res_rotation = |L(R_eps q, R_eps v) - L(q, v)| / eps`

其中 `R_eps` 是小角度旋转矩阵。若残差接近 0，可视为局部对称；若显著非零，说明该对称性被破坏。

## R07

伪代码：

```text
配置参数 (dt, n_steps, m, k, kx, ky, eps)

模拟自由粒子轨道 -> 得到 x(t), v(t)
模拟各向同性振子轨道 -> 得到 q_iso(t), v_iso(t)
模拟各向异性振子轨道 -> 得到 q_aniso(t), v_aniso(t)

在轨道采样点上计算：
  平移残差 (free)
  旋转残差 (iso)
  旋转残差 (aniso)

计算诺特电荷时间序列：
  p_x_free
  E_iso, Lz_iso
  E_aniso, Lz_aniso

统计漂移 (max_abs_drift, relative_max_drift)
打印表格并执行断言：
  对称性残差与守恒量漂移满足预期
```

## R08

复杂度分析（`N = n_steps`）：
- 时间复杂度：`O(N)`
  - 3 条轨道积分均为线性；
  - 对称性残差和守恒量统计也为线性。
- 空间复杂度：`O(N)`
  - 需要存储轨道和守恒量时间序列。

默认参数下（`N=4000`）CPU 运行开销很小。

## R09

数值稳定与可复现策略：
- 使用确定性初值，不引入随机数；
- 振子积分选用辛格式（Velocity-Verlet），减小长时能量漂移；
- 对称性检测使用固定 `eps` 有限差分，避免符号系统依赖；
- 报告绝对漂移与相对漂移，防止“量纲误读”。

## R10

预期现象：
1. 自由粒子的平移残差接近 0，`p_x` 漂移接近 0；
2. 各向同性振子的旋转残差接近 0，`L_z` 漂移很小；
3. 各向同性/各向异性振子都不显含时间，`E` 都近似守恒；
4. 各向异性振子的旋转残差明显升高，`L_z` 漂移显著大于各向同性情形。

这正是诺特定理的“保留对称 -> 保留电荷；破坏对称 -> 失去对应守恒”的最小演示。

## R11

关键参数与调参建议：
- `dt`：越小通常漂移越小，但运行步数固定时总模拟时长会变短；
- `n_steps`：越大越能观察长期漂移趋势；
- `fd_eps`：有限差分尺度，太小会放大浮点误差，太大破坏线性近似；
- `kx_aniso, ky_aniso`：差异越大，旋转对称破坏越明显。

建议先保持默认值验证，再逐项改动观察守恒量统计变化。

## R12

`demo.py` 模块结构：
- `NoetherConfig`：集中参数配置；
- `lagrangian_free_1d` / `lagrangian_oscillator_2d`：显式拉格朗日量；
- `simulate_free_particle`：自由粒子离散轨道；
- `simulate_oscillator_verlet`：二维振子辛积分；
- `translation_symmetry_residual` / `rotation_symmetry_residual`：对称性残差；
- `momentum_1d` / `energy_2d` / `angular_momentum_z_2d`：诺特电荷序列；
- `drift_metrics`：漂移统计；
- `main`：实验编排、输出与断言。

## R13

运行方式（无交互）：

```bash
cd "Algorithms/物理-理论物理-0147-诺特定理_(Noether's_Theorem)"
uv run python demo.py
```

或在仓库根目录运行：

```bash
uv run python Algorithms/物理-理论物理-0147-诺特定理_(Noether's_Theorem)/demo.py
```

## R14

输出表格解读：
- `symmetry_probe`：
  - `residual_mean` / `residual_max` 越接近 0，表示该变换越接近连续对称；
- `conservation_report`：
  - `max_abs_drift` 越小，守恒越好；
  - `relative_max_drift` 用于跨量纲比较。

重点看四行：
- `p_x_free`（应很小）；
- `Lz_iso`（应很小）；
- `E_iso`、`E_aniso`（都应小）；
- `Lz_aniso`（应明显大于 `Lz_iso`）。

## R15

常见问题排查：
1. 若 `E_iso` 漂移偏大：减小 `dt` 或增大 `n_steps` 并缩短总模拟时长；
2. 若 `Lz_aniso` 漂移不明显：加大 `|kx-ky|`；
3. 若对称性残差异常噪声大：把 `fd_eps` 从 `1e-8` 调到 `1e-6` 或 `1e-5`；
4. 若断言偶发失败：优先收紧时间步长，再调整阈值。

## R16

适用边界：
- 本条目是“诺特定理机制演示”，不是一般场论证明器；
- 只覆盖有限自由度经典力学模型；
- 未实现含规范场、边界项或高阶导数的广义 Noether 情形；
- 结论用于教学和算法验证，不替代严格数学证明。

## R17

可扩展方向：
- 扩展到三维中心势并检测全矢量角动量守恒；
- 引入显含时势 `V(q,t)`，演示时间平移破坏后能量不守恒；
- 增加字段论离散格点版本，连接到连续场论 Noether 流；
- 输出 CSV 或绘图做长时漂移可视化。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main()` 创建 `NoetherConfig`，固定积分步长、总步数、势参数和有限差分尺度。  
2. 调用 `simulate_free_particle()` 生成自由粒子轨道，并据此计算 `p_x(t)`。  
3. 调用 `simulate_oscillator_verlet(..., kx=k, ky=k)` 生成各向同性轨道，计算 `E_iso(t)` 与 `Lz_iso(t)`。  
4. 调用 `simulate_oscillator_verlet(..., kx!=ky)` 生成各向异性轨道，计算 `E_aniso(t)` 与 `Lz_aniso(t)`。  
5. 在采样时刻上用 `translation_symmetry_residual()` 与 `rotation_symmetry_residual()` 逐点估计对称性残差，不把对称判断外包给第三方黑盒。  
6. 用 `drift_metrics()` 对每条守恒量序列提取 `max_abs_drift` 与 `relative_max_drift`，形成可比较指标。  
7. `main()` 组装 `symmetry_probe` 与 `conservation_report` 两张 `pandas` 表并打印，直观呈现“对称性强弱 vs 漂移大小”。  
8. 通过断言检查：平移/旋转残差、`p_x` 与 `Lz_iso` 小漂移、`Lz_aniso` 显著漂移、能量在自治系统中保持小漂移；全部通过后输出 `All checks passed.`。
