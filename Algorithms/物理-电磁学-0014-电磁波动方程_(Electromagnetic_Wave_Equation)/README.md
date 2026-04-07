# 电磁波动方程 (Electromagnetic Wave Equation)

- UID: `PHYS-0014`
- 学科: `物理`
- 分类: `电磁学`
- 源序号: `14`
- 目标目录: `Algorithms/物理-电磁学-0014-电磁波动方程_(Electromagnetic_Wave_Equation)`

## R01

电磁波动方程描述电场与磁场在介质中的传播。在线性、均匀、无源介质中，麦克斯韦方程可导出：
\[
\frac{\partial^2 E}{\partial x^2}=\mu\epsilon\frac{\partial^2 E}{\partial t^2},\qquad
\frac{\partial^2 H}{\partial x^2}=\mu\epsilon\frac{\partial^2 H}{\partial t^2}
\]
本条目给出最小可运行 MVP：使用 1D Yee-FDTD 直接时间推进 \(E\) 与 \(H\)，并与解析平面波解逐点比较误差。

## R02

从 1D 无源麦克斯韦旋度方程出发（取传播方向为 \(x\)）：
\[
\frac{\partial E}{\partial t}=-\frac{1}{\epsilon}\frac{\partial H}{\partial x},\qquad
\frac{\partial H}{\partial t}=-\frac{1}{\mu}\frac{\partial E}{\partial x}
\]
继续对时间/空间求导并消去另一变量，可得到二阶波动方程。波速与阻抗：
\[
c=\frac{1}{\sqrt{\mu\epsilon}},\qquad Z=\sqrt{\frac{\mu}{\epsilon}}
\]
解析行波可写为：
\[
E(x,t)=E_0\sin(kx-\omega t),\qquad H(x,t)=\frac{E_0}{Z}\sin(kx-\omega t),\quad \omega=ck
\]

## R03

MVP 问题设置：
- 空间区间：\(x\in[0,L)\)，采用周期边界。
- 网格：`nx` 个空间单元，\(\Delta x=L/nx\)。
- 时间：推进到 `t_end`。
- 初值：取与解析行波一致的 \(E(x,0)\)；\(H\) 按 leapfrog 时间错位初始化到 \(t=-\Delta t/2\)。
- 目标：得到终态 \(E,H\) 并与解析解对比 `L1/L2/Linf` 误差。

## R04

Yee-FDTD 离散（1D 周期边界）：
\[
H_{i+1/2}^{n+1/2}=H_{i+1/2}^{n-1/2}-\frac{\Delta t}{\mu\Delta x}\left(E_{i+1}^{n}-E_i^n\right)
\]
\[
E_i^{n+1}=E_i^n-\frac{\Delta t}{\epsilon\Delta x}\left(H_{i+1/2}^{n+1/2}-H_{i-1/2}^{n+1/2}\right)
\]
在实现上使用 `np.roll` 处理周期差分，不依赖 PDE 黑盒求解器。

## R05

稳定性由 Courant 数控制：
\[
S=\frac{c\Delta t}{\Delta x}\le 1
\]
`demo.py` 中先按 `courant_target` 估计步长，再把步数取整回写 `dt=t_end/n_steps`，得到 `courant_actual`。若 `courant_actual>1` 则直接抛错，防止不稳定结果。

## R06

数值性质：
- 时间与空间离散均为二阶中心格式，整体二阶精度。
- 真实色散关系 \(\omega=ck\) 在离散后会有相速度偏差，网格越细误差越小。
- Yee 交错网格天然匹配麦克斯韦耦合结构，相比把二阶波方程硬拆成普通差分，物理一致性更好。

## R07

复杂度分析：
- 每一步更新两个长度为 `nx` 的向量，时间复杂度 \(O(nx)\)。
- 总步数 `n_steps`，总时间复杂度 \(O(nx\cdot n\_steps)\)。
- 仅保留当前场变量，空间复杂度 \(O(nx)\)。

## R08

核心数据结构：
- `x_e: np.ndarray`：电场节点坐标（整数网格）。
- `x_h: np.ndarray`：磁场节点坐标（半格偏移）。
- `E, H: np.ndarray`：数值电场/磁场。
- `E_ref, H_ref: np.ndarray`：解析参考解。
- `errors_e/errors_h: dict`：`l1/l2/linf/rel_l2` 误差指标。
- `energy_initial/energy_final`：离散能量守恒检查。

## R09

伪代码：

```text
input epsilon, mu, L, nx, mode, amplitude, t_end, courant_target
c <- 1 / sqrt(mu * epsilon)
Z <- sqrt(mu / epsilon)
k <- 2*pi*mode / L
omega <- c*k

build x_e (integer grid), x_h (half grid)
dx <- L / nx
estimate dt by courant_target, then snap to integer n_steps so n_steps*dt=t_end
if c*dt/dx > 1: raise error

E <- exact plane wave at t=0 on x_e
H <- exact plane wave at t=-dt/2 on x_h
energy_initial <- discrete energy(E,H)

repeat n_steps:
    H <- H - (dt/(mu*dx)) * (roll(E,-1) - E)
    E <- E - (dt/(epsilon*dx)) * (H - roll(H,1))

compute E_ref at t=t_end and H_ref at t=t_end-dt/2
compute L1/L2/Linf/relative-L2 errors for E and H
report Courant number, errors, and energy drift
```

## R10

MVP 默认参数（`demo.py`）：
- `epsilon = 1.0`
- `mu = 1.0`
- `length = 1.0`
- `nx = 400`
- `mode = 3`
- `amplitude = 1.0`
- `t_end = 0.5`
- `courant_target = 0.95`

这组参数运行很快，同时误差与能量漂移都可观测。

## R11

脚本输出包含：
- 网格与步长：`nx`, `dx`, `n_steps`, `dt`
- 物理量：`c`, `Z`, `k`, `omega`
- 稳定性：`courant_actual`
- 误差：`E` 和 `H` 各自 `L1/L2/Linf/RelL2`
- 守恒检查：`energy_initial`, `energy_final`, `energy_drift`

一次运行即可完成，不需要交互输入。

## R12

`demo.py` 函数划分：
- `FDTDConfig`：参数集中管理。
- `derive_wave_constants`：计算 \(c,Z,k,\omega\)。
- `exact_plane_wave`：构造解析 \(E/H\) 场。
- `discrete_energy`：离散能量计算。
- `compute_errors`：误差范数计算。
- `run_fdtd_1d_periodic`：核心 Yee 时间推进与验证。
- `main`：组织参数并打印摘要。

## R13

运行方式：

```bash
uv run python demo.py
```

在当前算法目录下直接执行即可。

## R14

常见错误与规避：
- 忘记 \(H\) 的半网格和半时间步错位，导致相位误差异常大。
- 把 `courant_target` 当作最终 Courant 数而不回写 `dt`，终止时刻会偏离 `t_end`。
- 周期边界实现错位（`roll` 方向写反）会引入非物理反射。
- 只看单点值，不做整体范数和能量检查，难以及时发现符号错误。

## R15

最小验证策略：
1. 固定 `mode=3`，测试 `nx=100, 200, 400`，观察 `L2` 误差随网格加密下降。
2. 固定 `nx`，测试 `courant_target=0.5, 0.8, 0.95`，确认都满足稳定性且结果一致收敛。
3. 检查 `energy_drift` 量级应较小，不应出现爆炸式增长。

## R16

适用范围与局限：
- 适用：1D 无源均匀介质电磁传播教学、FDTD 基线验证、数值实验起点。
- 局限：当前不含材料色散、损耗、源项、PML 吸收边界和多维几何。
- 若要工程级仿真，需要扩展到 2D/3D、引入边界处理和更复杂介质模型。

## R17

可扩展方向：
- 增加电导率 \(\sigma\) 与损耗介质模型。
- 增加时域激励源（高斯脉冲、电流源）。
- 从周期边界扩展到 PEC/PMC/PML。
- 扩展到 2D TM/TE 或 3D 全波 FDTD。
- 加入 FFT 诊断，分析数值色散与频谱传播误差。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 创建 `FDTDConfig`，调用 `run_fdtd_1d_periodic`。
2. `run_fdtd_1d_periodic` 校验参数合法性（`epsilon/mu/nx/mode/t_end/courant_target`）。
3. 调用 `derive_wave_constants` 计算 `c, Z, k, omega`，并构建 `x_e` 与 `x_h` 两套交错网格。
4. 先按 `courant_target` 估算 `dt_guess`，再取整数 `n_steps` 并回写 `dt=t_end/n_steps`，保证最终时刻精确对齐 `t_end`。
5. 计算 `courant_actual=c*dt/dx`；若大于 1 则抛出异常阻止不稳定推进。
6. 用 `exact_plane_wave` 初始化 `E(x,0)` 与 `H(x,-dt/2)`，记录 `energy_initial`。
7. 在时间循环中按 Yee 顺序更新：先更新 `H`（用 `roll(E,-1)-E`），再更新 `E`（用 `H-roll(H,1)`）。
8. 循环结束后构造 `E_ref(t_end)` 与 `H_ref(t_end-dt/2)`，调用 `compute_errors` 和 `discrete_energy` 输出误差与能量漂移。
