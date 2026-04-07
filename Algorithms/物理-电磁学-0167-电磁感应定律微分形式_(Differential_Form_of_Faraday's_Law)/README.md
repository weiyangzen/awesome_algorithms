# 电磁感应定律微分形式 (Differential Form of Faraday's Law)

- UID: `PHYS-0166`
- 学科: `物理`
- 分类: `电磁学`
- 源序号: `167`
- 目标目录: `Algorithms/物理-电磁学-0167-电磁感应定律微分形式_(Differential_Form_of_Faraday's_Law)`

## R01

法拉第电磁感应定律的微分形式写作：
\[
\nabla\times\mathbf{E}=-\frac{\partial \mathbf{B}}{\partial t}
\]
它表达了“时变磁场会在局部产生旋涡电场”的点态关系。该条目的 MVP 不做大而全电磁仿真，而是做一件可验证的最小任务：在二维周期网格上构造解析场并用离散算子验证该微分关系。

## R02

该微分形式可由积分形式与 Stokes 定理得到：
\[
\oint_{\partial S}\mathbf{E}\cdot d\mathbf{l}=-\frac{d}{dt}\int_S \mathbf{B}\cdot d\mathbf{S}
\]
对任意足够小曲面 \(S\)，有
\[
\int_S (\nabla\times\mathbf{E})\cdot d\mathbf{S}=-\int_S\frac{\partial \mathbf{B}}{\partial t}\cdot d\mathbf{S}
\Rightarrow
\nabla\times\mathbf{E}=-\frac{\partial \mathbf{B}}{\partial t}.
\]
在本实现中，采用 2D 情形（仅 \(B_z\) 非零），方程退化为：
\[
\frac{\partial E_y}{\partial x}-\frac{\partial E_x}{\partial y}=-\frac{\partial B_z}{\partial t}.
\]

## R03

MVP 问题设置：
- 区域：\([0,L_x)\times[0,L_y)\)，周期边界。
- 网格：`nx × ny` 均匀网格。
- 场构造：给定解析 \(E_x,E_y,B_z\)（三角函数），使其在连续意义下严格满足 Faraday 微分式。
- 数值任务：
  1. 用中心差分计算 \((\nabla\times E)_z\)；
  2. 用显式时间推进 \(B_z^{n+1}=B_z^n-\Delta t(\nabla\times E)_z^n\)；
  3. 与解析 \(B_z\) 对比终态误差，并计算 \(\nabla\times E+\partial_t B\) 残差。

## R04

离散公式（周期边界 + 二阶中心差分）：
\[
\left(\frac{\partial E_y}{\partial x}\right)_{i,j}\approx
\frac{E_{y,i+1,j}-E_{y,i-1,j}}{2\Delta x},
\qquad
\left(\frac{\partial E_x}{\partial y}\right)_{i,j}\approx
\frac{E_{x,i,j+1}-E_{x,i,j-1}}{2\Delta y}
\]
\[
(\nabla\times E)_{z,i,j}\approx
\frac{E_{y,i+1,j}-E_{y,i-1,j}}{2\Delta x}-
\frac{E_{x,i,j+1}-E_{x,i,j-1}}{2\Delta y}
\]
时间推进（显式 Euler）：
\[
B_{z,i,j}^{n+1}=B_{z,i,j}^{n}-\Delta t\,(\nabla\times E)_{z,i,j}^{n}.
\]
实现中使用 `np.roll` 完成周期索引，不依赖 PDE 黑盒求解器。

## R05

稳定性与误差要点：
- 本例直接推进 \(\partial_t B=-\nabla\times E\)，时间离散是一阶 Euler，空间离散是二阶中心差分。
- 由于电场按解析式外给，不涉及全耦合 Maxwell 显式稳定上限；但 `dt` 过大会导致时间积分误差明显。
- 代码用 `cfl_like * min(dx,dy)` 生成 `dt_guess`，再把步数取整回写 `dt=t_end/n_steps`，保证终止时刻精确对齐。

## R06

数值性质：
- 空间误差阶约为 \(O(\Delta x^2+\Delta y^2)\)。
- 时间误差阶约为 \(O(\Delta t)\)。
- 随网格加密且 `dt` 同步减小，`B` 终态相对误差与方程残差应下降。
- 该方案适合做“关系式是否被离散实现正确”的基线验证，不替代工程级电磁仿真器。

## R07

复杂度分析（单次模拟）：
- 每个时间步：若干次同尺寸数组算子，复杂度 \(O(nx\cdot ny)\)。
- 总步数 `n_steps`，总时间复杂度 \(O(nx\cdot ny\cdot n\_steps)\)。
- 存储 `Ex/Ey/Bz` 及临时差分数组，空间复杂度 \(O(nx\cdot ny)\)。

## R08

核心数据结构：
- `FaradayConfig`：参数配置（网格、频率、终止时刻、步长比例）。
- `WaveNumbers`：`kx/ky` 与解析 `Bz` 系数缓存。
- `xx, yy`：二维网格坐标矩阵。
- `ex, ey, bz`：离散电场/磁场数组。
- `result` 字典：收集 `dt/n_steps`、`B` 误差、微分残差等摘要指标。

## R09

伪代码：

```text
input config
build periodic grid xx, yy and dx, dy
compute kx, ky and analytic coefficient for Bz
choose dt from cfl_like and align n_steps*dt = t_end

bz <- B_exact(t=0)
for n in [0, n_steps-1]:
    t <- n * dt
    ex, ey <- E_exact(t)
    curl_e <- central_difference_curl_z(ex, ey)
    bz <- bz - dt * curl_e

bz_ref <- B_exact(t=t_end)
report L2/Linf/RelL2(bz - bz_ref)

at probe time t_probe:
    compute residual = curl(E_exact) + dB_exact/dt
    report RMS/Linf/relative RMS residual
```

## R10

`demo.py` 默认参数：
- `length_x = length_y = 1.0`
- `nx = ny = 96`
- `mode_x = 2, mode_y = 3`
- `amplitude = 1.0`
- `omega = 7.0`
- `t_end = 0.8`
- `cfl_like = 0.06`

并附带 `24/48/96` 三组分辨率对比，观察误差收敛趋势。

## R11

脚本输出分三段：
- 主配置与离散信息：`grid/dx/dy/dt/n_steps/t_final`
- 终态磁场误差：`L2(Bz), Linf(Bz), RelL2(Bz)`
- 方程残差：`RMS(curlE+dB/dt), Linf residual, RelRMS residual`
- 分辨率对比表：展示 `B_rel_l2` 与 `residual_rel_rms` 随网格变化的趋势

## R12

代码模块分工：
- `build_grid`：生成周期网格和步长。
- `derive_wave_numbers`：计算 `kx/ky` 与解析系数。
- `electric_field`：解析 `Ex/Ey`。
- `magnetic_field_exact` / `magnetic_dt_exact`：解析 `Bz` 与时间导数。
- `curl_z_central`：离散旋度核心算子。
- `simulate_faraday`：主流程（时间推进 + 误差/残差评估）。
- `run_resolution_study`：多分辨率收敛快照。
- `main`：组织运行并打印结果。

## R13

运行方式：

```bash
uv run python demo.py
```

在当前算法目录执行即可，无需交互输入。

## R14

常见错误与规避：
- 把 `curl_z` 写成 `dEx/dy - dEy/dx`（符号反了）会让结论整体翻转。
- 周期差分 `roll` 轴写错（`axis=0/1` 混淆）会导致残差异常大。
- 只看单点误差不看范数，容易忽略全局偏差。
- `dt` 不与 `t_end` 对齐，会让终态解析对比时刻不一致。

## R15

最小验证策略：
1. 先跑默认参数，确认 `RelL2(Bz)` 与 `RelRMS residual` 显著小于 1。
2. 查看 `24 -> 48 -> 96` 的对比，确认误差总体下降。
3. 修改 `cfl_like`（如 `0.03` 和 `0.12`）观察时间离散误差随步长变化。
4. 将 `mode_x/mode_y` 改为其它正整数，检查实现对不同空间频率的稳健性。

## R16

适用范围与局限：
- 适用：Faraday 微分式教学演示、离散旋度算子正确性检查、代码基线回归测试。
- 局限：
  - 未求解完整 Maxwell 耦合系统；
  - 未包含介质非均匀性、源项、耗散、边界吸收层；
  - 时间推进为一阶显式方法，精度有限。

## R17

可扩展方向：
- 使用 leapfrog / RK 方法提高时间精度。
- 扩展到三维向量场并同时验证 Ampere-Maxwell 方程。
- 增加积分形式验证（闭合回路环量 vs 面通量变化率）。
- 加入噪声与离散误差统计，做鲁棒性分析。
- 引入介质参数 \(\mu,\epsilon,\sigma\) 的空间变化。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 创建 `FaradayConfig`，调用 `simulate_faraday` 执行单次验证。
2. `simulate_faraday` 先做参数合法性检查（网格规模、频率、终止时刻、步长比例）。
3. 调用 `build_grid` 生成 `xx/yy/dx/dy`，调用 `derive_wave_numbers` 计算 `kx/ky` 与 `coef_b`。
4. 调用 `compute_time_grid` 由 `cfl_like` 生成 `dt` 和 `n_steps`，并确保 `n_steps*dt=t_end`。
5. 用 `magnetic_field_exact(t=0)` 初始化 `bz`，进入时间循环。
6. 每一步先用 `electric_field(t_n)` 得到 `ex/ey`，再用 `curl_z_central` 计算离散旋度，最后执行 `bz <- bz - dt * curl_e_z`。
7. 循环结束后计算 `bz_ref = magnetic_field_exact(t_final)`，再由 `compute_error_metrics` 产出 `L2/Linf/RelL2`。
8. 在探测时刻 `t_probe` 计算 `curl(E)` 与 `dB/dt`（`magnetic_dt_exact`），得到 `residual = curlE + dB/dt` 的 `RMS/Linf/RelRMS` 并输出；随后 `run_resolution_study` 给出 `24/48/96` 三组网格对比。
