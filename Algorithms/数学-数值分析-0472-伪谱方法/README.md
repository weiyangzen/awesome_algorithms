# 伪谱方法

- UID: `MATH-0472`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `472`
- 目标目录: `Algorithms/数学-数值分析-0472-伪谱方法`

## R01

伪谱方法（Pseudo-Spectral Method）常用于周期光滑 PDE：导数在频域中计算，非线性乘积在物理空间中计算，再用 FFT 往返。  
本条目给出最小可运行 MVP：在 `x in [0, 2π)` 上，用 Fourier 伪谱法 + RK4 求解一维受迫粘性 Burgers 方程，并通过制造解验证精度。

## R02

本目录问题定义：
- 方程：`u_t + u u_x = nu * u_xx + s(x,t)`，周期边界。
- 制造解：`u_exact(x,t) = exp(-t) sin(x)`。
- 对应强迫项：
  - `s(x,t) = (nu-1) exp(-t) sin(x) + 0.5 exp(-2t) sin(2x)`。
- 输入：固定参数 `nu`、终止时间 `T`、网格规模序列 `N`。
- 输出：每个 `N` 的 `L2/L∞` 误差、收敛速率、步数与时间步长。

`demo.py` 无交互输入，直接运行完成实验并输出通过判定。

## R03

伪谱离散核心关系：
- Fourier 展开：`u(x,t) = sum_k u_hat(k,t) exp(ikx)`。
- 导数在频域：
  - `u_x = IFFT( i k * FFT(u) )`
  - `u_xx = IFFT( -k^2 * FFT(u) )`
- 非线性项采用伪谱思路：
  - 先在物理空间计算 `u * u_x`；
  - 再转到频域执行 2/3 去混叠（dealiasing）；
  - 逆变换回物理空间，形成稳定的非线性项近似。
- 时间离散采用显式 RK4。

## R04

算法高层流程：
1. 生成周期均匀网格 `x_j` 与波数 `k`。
2. 用解析初值 `u(x,0)=sin(x)` 初始化。
3. 在每个时间步中对当前 `u` 做 FFT，得到频域系数。
4. 在频域求 `u_x`、`u_xx`，回到物理空间。
5. 在物理空间组装非线性乘积 `u*u_x`。
6. 对该非线性项做 FFT，并施加 2/3 去混叠掩码。
7. 组装 RHS：`-nonlinear + nu*u_xx + s(x,t)`，用 RK4 推进一步。
8. 终止后与制造解比较，输出误差与收敛率。

## R05

核心数据结构：
- `x: np.ndarray`：周期网格（实数一维数组）。
- `k: np.ndarray`：角波数数组（与 FFT 排序一致）。
- `u: np.ndarray`：当前时刻数值解。
- `u_hat: np.ndarray(complex)`：`u` 的频谱系数。
- `dealias_mask: np.ndarray(bool)`：2/3 去混叠掩码。
- `records: list[dict[str, float]]`：误差、速率、步数等统计结果。

## R06

正确性关键点：
- 导数计算直接由 Fourier 微分公式给出，避免低阶差分截断误差。
- 非线性项在物理空间计算，符合“伪谱”定义；并对非线性频谱去混叠，抑制别名误差。
- 强迫项由制造解反推，保证存在可直接对照的真解。
- 通过网格加密下误差下降验证实现链路（空间离散 + 时间推进）逻辑正确。

## R07

复杂度（单次仿真，网格 `N`，时间步 `M`）：
- 单步主要代价为若干次 FFT/IFFT，时间复杂度 `O(N log N)`。
- 总时间复杂度 `O(M N log N)`。
- 空间复杂度 `O(N)`（存储状态、频谱、掩码与临时数组）。

在周期问题上，该复杂度通常优于显式构造稠密谱矩阵的实现。

## R08

边界与异常处理：
- `n < 8`、`length <= 0`、`t_final <= 0`、`cfl <= 0` 会抛出异常。
- `nu < 0` 或数组含 `nan/inf` 会抛出异常。
- 时间步长根据对流与扩散稳定性上界自动取 `min`，并重缩放以整除 `T`。
- 适用边界：周期、平滑解；若有间断/尖峰，需额外滤波或改用其他方法。

## R09

MVP 取舍说明：
- 只实现 1D 周期 Fourier 伪谱，不扩展到 Chebyshev 或非周期边界。
- 不调用 PDE 黑盒库，核心流程完全由 `numpy.fft`、数组运算和 RK4 显式写出。
- 引入最小必要稳定化：2/3 去混叠；不额外引入复杂高阶滤波器。
- 保持“可读 + 可跑 + 可验证”优先于大而全框架。

## R10

`demo.py` 函数职责：
- `periodic_grid`：生成周期网格。
- `angular_wavenumbers`：构造 FFT 对齐的角波数。
- `dealias_mask_23`：生成 2/3 去混叠掩码。
- `exact_solution`：制造解。
- `forcing_term`：制造解对应强迫项。
- `burgers_rhs_pseudospectral`：伪谱空间离散，返回 `du/dt`。
- `rk4_step`：执行单步 RK4。
- `integrate_burgers_pseudospectral`：完成从 `t=0` 到 `t=T` 的时间推进。
- `run_convergence_study`：多网格误差统计。
- `print_results_table`：格式化输出结果表。
- `main`：组织实验并给出 `pass` 结果。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0472-伪谱方法
python3 demo.py
```

脚本不需要命令行参数或交互输入。

## R12

输出字段说明：
- `N`：空间网格点数。
- `steps`：时间步数。
- `dt`：时间步长。
- `L2_error`：终态与制造解的离散 `L2` 误差。
- `Linf_error`：终态与制造解的离散 `L∞` 误差。
- `rate(L2) / rate(Linf)`：相邻网格误差收敛速率 `log2(e_old/e_new)`。
- `pass`：最细网格误差是否低于阈值。

## R13

最小实验集（内置）：
- 网格：`N = [32, 64, 128, 256]`。
- 参数：`nu = 0.01`，`T = 1.0`，周期区间长度 `2π`。
- 验证：对比 `u_num(x,T)` 与 `u_exact(x,T)`。
- 通过标准：最细网格 `L∞` 误差小于 `2e-5`。

## R14

可调参数：
- `grid_sizes`：收敛实验网格列表。
- `NU`：粘性系数。
- `T_FINAL`：终止时间。
- `cfl`：时间步选择系数。
- `pass_threshold`：自动验收阈值。

建议先用默认参数复现，再调整 `nu`、`T` 和网格密度观察误差趋势。

## R15

方法对比：
- 对比有限差分：伪谱在光滑周期问题上通常能以更少网格达到更高精度。
- 对比纯谱 Galerkin 黑盒实现：本实现更直观，便于追踪每个 FFT 步骤。
- 对比不去混叠版本：2/3 规则通常可降低非线性别名污染，稳定误差表现。
- 局限：非周期边界、强间断问题并非本 MVP 目标场景。

## R16

典型应用：
- 周期方向上的流体方程离散（如 Burgers、Navier-Stokes 简化模型）。
- 湍流教学/验证中的最小可复现实验。
- 需要高精度空间导数且解相对平滑的周期 PDE 原型验证。

## R17

可扩展方向：
- 从 1D 扩展到 2D/3D 张量积 FFT。
- 增加半隐式 IMEX 时间推进以放宽扩散稳定性限制。
- 增加更系统的谱滤波与能量谱诊断。
- 支持多种制造解/强迫项以做自动化回归测试。
- 对接 `scipy` 稀疏线性代数模块构建混合谱-隐式方案。

## R18

`demo.py` 源码级算法流程（9 步）：
1. `main` 设定网格序列与阈值，调用 `run_convergence_study`。  
2. `run_convergence_study` 对每个 `N` 调用 `integrate_burgers_pseudospectral`。  
3. `integrate_burgers_pseudospectral` 生成 `x`、`k` 与 `dealias_mask`，并用 `exact_solution(x,0)` 初始化。  
4. 每个时间步中，`rk4_step` 会多次调用 `burgers_rhs_pseudospectral` 计算阶段斜率。  
5. 在 `burgers_rhs_pseudospectral` 内先执行 `u_hat = FFT(u)`，再由 `i*k*u_hat` 与 `-k^2*u_hat` 逆变换得到 `u_x`、`u_xx`。  
6. 在物理空间计算非线性乘积 `u*u_x`，随后 `FFT` 到频域并乘以 2/3 掩码去混叠。  
7. 逆变换得到去混叠后的非线性项，和 `nu*u_xx`、`forcing_term(x,t)` 组装 RHS。  
8. RK4 将四个阶段斜率合成 `u^{n+1}`，迭代至 `t=T`。  
9. 回到 `run_convergence_study`：与 `exact_solution(x,T)` 比较，输出误差、速率与 `pass` 判定。  
