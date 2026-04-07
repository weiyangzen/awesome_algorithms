# 麦克斯韦方程组 (Maxwell's Equations)

- UID: `PHYS-0011`
- 学科: `物理`
- 分类: `电磁学`
- 源序号: `11`
- 目标目录: `Algorithms/物理-电磁学-0011-麦克斯韦方程组_(Maxwell's_Equations)`

## R01

麦克斯韦方程组把电场、磁场、电荷与电流统一在同一组偏微分方程中，是经典电磁学与电磁波理论的核心。无源、均匀介质情况下可写为：

- `div(E) = 0`
- `div(B) = 0`
- `curl(E) = - dB/dt`
- `curl(H) = dD/dt`

其中 `D = epsilon * E`，`B = mu * H`。本条目提供一个最小可运行的 2D TMz Yee-FDTD MVP，用数值推进验证上述关系。

## R02

本实现聚焦 TMz 模式（`Ex, Ey, Hz` 非零）且无源（`rho=0, J=0`）：

- `dEx/dt =  (1/epsilon) * dHz/dy`
- `dEy/dt = -(1/epsilon) * dHz/dx`
- `dHz/dt = -(1/mu) * (dEy/dx - dEx/dy)`

这是旋度方程在二维下的直接展开；高斯定律通过离散 `div(E)` 检查，`div(B)` 在 TMz 且 `Bx=By=0` 时天然为 0。

## R03

为了做可量化验证，`demo.py` 使用解析平面波作为 manufactured solution：

- `Hz = H0 * sin(kx x + ky y - omega t)`
- `Ex = -(ky * H0 / (epsilon * omega)) * sin(...)`
- `Ey =  (kx * H0 / (epsilon * omega)) * sin(...)`
- `omega = c * sqrt(kx^2 + ky^2)`，`c = 1/sqrt(mu*epsilon)`

这样可以在每个场分量上计算相对 `L2` 误差，不依赖主观判断。

## R04

空间离散采用 2D Yee 交错网格与周期边界：

- `Ex(i,j)` 位于 `(x_i, y_{j+1/2})`
- `Ey(i,j)` 位于 `(x_{i+1/2}, y_j)`
- `Hz(i,j)` 位于 `(x_{i+1/2}, y_{j+1/2})`

交错布局能让旋度项在离散后仍保持几何一致性，降低非物理数值模式。

## R05

时间推进使用 leapfrog 显式格式：

1. `Hz` 在半步时间层更新；
2. 再用新的 `Hz` 更新 `Ex, Ey` 到整步时间层。

二维 CFL 条件：

`c * dt * sqrt(1/dx^2 + 1/dy^2) <= 1`

脚本会先按 `courant_factor` 估算 `dt`，再通过整数 `n_steps` 回写 `dt=t_end/n_steps`，并显式检查稳定性。

## R06

数值复杂度：

- 单步只包含若干 `np.roll` 差分和逐点加减乘，时间复杂度 `O(nx*ny)`；
- 总复杂度 `O(nx*ny*n_steps)`；
- 仅存当前三张场数组，空间复杂度 `O(nx*ny)`。

## R07

MVP 的正确性指标：

- `RelL2(Ex), RelL2(Ey), RelL2(Hz)`：与解析解比较；
- `max|div(E)|`：离散高斯定律残差；
- `energy_drift`：离散总电磁能漂移。

三个指标分别覆盖“解精度、约束保持、守恒性质”。

## R08

`demo.py` 数据结构最小化：

- `MaxwellConfig`：参数容器；
- 三个场数组：`ex, ey, hz`；
- 网格坐标数组：`ex_x/ex_y, ey_x/ey_y, hz_x/hz_y`；
- 结果字典：汇总步长、误差、散度、能量等标量。

不引入外部求解框架，核心算法均在源码中显式实现。

## R09

伪代码：

```text
input epsilon, mu, lx, ly, nx, ny, mode_x, mode_y, t_end
derive c, kx, ky, omega
build staggered grids for Ex/Ey/Hz
compute dt from 2D CFL and snap to integer n_steps

initialize Ex(t=0), Ey(t=0), Hz(t=-dt/2) from exact plane wave
record energy_initial

for step in 1..n_steps:
    Hz <- Hz - (dt/mu) * (dEy/dx - dEx/dy)
    Ex <- Ex + (dt/epsilon) * dHz/dy
    Ey <- Ey - (dt/epsilon) * dHz/dx

build exact references at t_end (Ex/Ey) and t_end-dt/2 (Hz)
compute relative L2 errors
compute div(E), energy drift
report metrics
```

## R10

默认参数（可直接运行）：

- `epsilon = mu = 1.0`
- `lx = ly = 1.0`
- `nx = 80, ny = 64`
- `mode_x = 2, mode_y = 1`
- `hz_amplitude = 1.0`
- `t_end = 0.4`
- `courant_factor = 0.7`

该配置能在很短时间内完成，同时给出稳定且可读的误差量级。

## R11

脚本输出包括：

- 网格与时间推进参数：`dx, dy, dt, n_steps, courant_2d`；
- 波参数：`c, impedance, kx, ky, omega`；
- 三个场分量的相对 `L2` 误差；
- 高斯约束：`max|div(E)|`, `max|div(B)|`；
- 能量守恒检查：`energy_initial`, `energy_final`, `energy_drift`。

## R12

关键函数说明：

- `derive_constants`：从物理参数得到传播常数；
- `build_staggered_grids`：构造 Yee 网格坐标；
- `exact_tmz_fields`：给出解析 TMz 场；
- `run_tmx_fdtd`：执行主时间循环与指标统计；
- `divergence_electric` / `electromagnetic_energy`：约束与守恒评估。

## R13

运行方式：

```bash
uv run python demo.py
```

脚本无需交互输入，单次运行即可完成全部数值验证。

## R14

常见错误与规避：

1. 忘记 `Hz` 的半步时间层初始化，导致相位误差偏大。
2. 把二维 CFL 误写成一维形式，可能造成发散。
3. `np.roll` 方向写反，会把 `curl` 符号搞错。
4. 只看某一分量误差，不检查 `div(E)` 与能量漂移，容易漏掉结构性错误。

## R15

最小验证建议：

1. 固定 `mode_x/mode_y`，将网格从 `40x32` 加密到 `80x64`，观察误差下降。
2. 固定网格，尝试 `courant_factor = 0.4, 0.7, 0.9`，确认都满足稳定条件。
3. 在不同 `t_end` 下比较 `energy_drift`，验证长期行为。

## R16

适用范围与局限：

- 适用：麦克斯韦方程数值离散教学、FDTD 基线验证、代码单元测试基准。
- 局限：仅限 2D TMz、无源均匀介质、周期边界；不含损耗、源项、色散与复杂几何。
- 若做工程仿真，需扩展到 3D、PML、材料模型与更严格误差控制。

## R17

可扩展方向：

- 引入 `J` 与 `rho`，验证有源 Maxwell 系统；
- 增加电导率/色散模型（Drude/Lorentz）；
- 增加 PEC/PMC/PML 边界条件；
- 扩展到 TEz 与 3D 全矢量 FDTD；
- 输出场快照并做频谱分析。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 构造 `MaxwellConfig` 并调用 `run_tmx_fdtd`。
2. `run_tmx_fdtd` 先做参数合法性检查，然后调用 `derive_constants` 得到 `c/kx/ky/omega`。
3. 调用 `build_staggered_grids` 构建 `Ex/Ey/Hz` 三套交错坐标。
4. 按二维 CFL 计算 `dt_limit`，再通过整数 `n_steps` 回写 `dt=t_end/n_steps` 并校验 `courant_2d<=1`。
5. 调用 `exact_tmz_fields` 初始化 `Ex(t=0), Ey(t=0), Hz(t=-dt/2)`，记录 `energy_initial`。
6. 进入时间循环：先更新 `Hz`（由 `dEy/dx - dEx/dy`），再更新 `Ex/Ey`（由 `dHz/dy` 与 `dHz/dx`）。
7. 循环结束后构造终态解析参考解，计算 `Ex/Ey/Hz` 的相对 `L2` 误差。
8. 计算 `div(E)`、能量漂移并汇总打印，形成一次非交互的可复现实验输出。

第三方库未被当作黑盒：`numpy` 仅提供数组与基础向量化算子，离散旋度更新、CFL 控制、误差与守恒评估均在源码中明确展开。
