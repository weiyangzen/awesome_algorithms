# 弗里德曼方程 (Friedmann Equations)

- UID: `PHYS-0052`
- 学科: `物理`
- 分类: `宇宙学`
- 源序号: `52`
- 目标目录: `Algorithms/物理-宇宙学-0052-弗里德曼方程_(Friedmann_Equations)`

## R01

弗里德曼方程（Friedmann Equations）描述均匀各向同性宇宙（FRW 度规）下的整体膨胀动力学。核心未知量是尺度因子 `a(t)`，其变化率由宇宙总能量密度分量决定：

- 物质项 `Omega_m a^{-3}`
- 辐射项 `Omega_r a^{-4}`
- 曲率项 `Omega_k a^{-2}`
- 暗能量项 `Omega_Lambda`

它是现代观测宇宙学里把“红移、距离、宇宙年龄”连接起来的主方程。

## R02

本条目目标是做一个可运行且可审计的最小 MVP：

- 数值实现 `H(a)` 与 `H(z)`；
- 计算宇宙年龄 `t0`；
- 计算视线向共动距离、光度距离、角径距离；
- 计算减速参数 `q(z)` 和加速-减速转折红移 `z_acc`；
- 对三个典型宇宙学模型做自动断言验证。

重点是把方程链路完整展开，而不是调用黑盒宇宙学库。

## R03

MVP 问题设定为三组背景宇宙模型对比：

1. 近平直 `LambdaCDM`（`Omega_m=0.315`, `Omega_r=9e-5`, `Omega_k=0`, `Omega_Lambda=1-Omega_m-Omega_r`）；
2. Einstein-de Sitter（`Omega_m=1`, 其余为 0）；
3. 开宇宙（`Omega_m=0.3`, `Omega_k=0.7`, `Omega_Lambda=0`）。

统一使用 `H0 = 67.4 km/s/Mpc`，输出每个模型的年龄、`q0`、`H(z)`、距离量与 lookback time。

## R04

本实现使用的核心方程：

1. 第一弗里德曼方程（无量纲形式）：
`E(a)^2 = H(a)^2/H0^2 = Omega_r a^{-4} + Omega_m a^{-3} + Omega_k a^{-2} + Omega_Lambda`

2. 哈勃参数：
`H(z) = H0 * E(a),  a = 1/(1+z)`

3. 宇宙年龄：
`t0 = Integral_{0}^{1} da / [a H(a)]`

4. 第二弗里德曼方程对应的减速参数：
`q(z) = [Omega_m a^{-3} + 2 Omega_r a^{-4} - 2 Omega_Lambda] / [2 E(a)^2]`

5. 视线向共动距离：
`D_C(z) = (c/H0) * Integral_{0}^{z} dz' / E(z')`

6. 横向共动距离（含曲率）：
- `Omega_k = 0`: `D_M = D_C`
- `Omega_k > 0`: `D_M = (D_H/sqrt(Omega_k)) sinh(sqrt(Omega_k) D_C/D_H)`
- `Omega_k < 0`: `D_M = (D_H/sqrt(|Omega_k|)) sin(sqrt(|Omega_k|) D_C/D_H)`

7. 光度/角径距离：
`D_L = (1+z) D_M`, `D_A = D_M/(1+z)`。

## R05

设积分网格点数为 `N`（例如 `N=2e5`）：

- 年龄积分：`O(N)`
- 距离积分（每个 z）：`O(N)`
- `z_acc` 搜索（网格扫描）：`O(N)`

整体空间复杂度 `O(N)`（主要为网格数组）。

## R06

`demo.py` 的默认执行行为：

- 非交互运行 3 个宇宙学案例；
- 打印每个案例的关键物理量；
- 自动进行一致性断言，包括：
  - `LambdaCDM` 年龄在合理区间且 `q0<0`；
  - EdS 年龄接近解析解 `t0=2/(3H0)`；
  - 开宇宙模型在今天仍减速（`q0>0`）；
  - 距离与 `H(z)` 的单调性检查。

## R07

优点：

- 公式链路完整、透明，便于教学与审计；
- 仅依赖 `numpy`，环境负担小；
- 同时覆盖动力学量（`q(z)`, `t0`）和观测量（`D_L`, `D_A`）。

局限：

- 只处理均匀各向同性背景，不含结构形成扰动；
- 仅做背景积分，不拟合真实观测数据；
- 未实现中微子质量、动态暗能量 `w(z)` 等扩展模型。

## R08

前置知识：

- FRW 度规与尺度因子 `a(t)`；
- 红移与尺度因子的关系 `a=1/(1+z)`；
- 基本数值积分（梯形积分）与单位换算。

运行环境：

- Python `>=3.10`
- `numpy`

## R09

适用场景：

- 宇宙学课程中快速验证 `LambdaCDM` 基础量级；
- 做更复杂参数拟合前的公式链路自测；
- 需要一个不依赖黑盒库的背景宇宙学基线脚本。

不适用场景：

- CMB/LSS 精密拟合（需要 CAMB/CLASS 等）；
- 扰动增长、功率谱、再电离历史等高阶任务；
- 对极高红移或强精度误差控制有严格需求的生产级计算。

## R10

正确性直觉：

1. `E(a)^2` 是各能量密度分量在 FRW 中的标准缩放叠加；
2. `t0`、`D_C` 都是 `1/E` 的积分，离散积分应给出平滑且单调的结果；
3. `LambdaCDM` 的今天应表现为加速膨胀，即 `q0<0`；
4. EdS 是可解析对照组，`t0=2/(3H0)` 可直接检验数值误差；
5. 距离关系 `D_L=(1+z)^2 D_A`（由定义推导）保障了量之间的一致性。

## R11

数值稳定性策略：

- 年龄积分用 `a_min=1e-6` 截断 `a->0` 端点，避免奇点采样；
- 在 `validate_params` 中先检查 `E(a)^2>0`（`z in [0,10]`）再计算；
- 对 `z_acc` 采用高密度网格后线性插值，避免粗网格跳变；
- 所有关键输出都加断言，防止静默产生不物理结果。

## R12

关键参数：

- `h0_km_s_mpc`：控制时间尺度和距离尺度；
- `Omega_m`, `Omega_r`, `Omega_k`, `Omega_Lambda`：决定宇宙动力学；
- `n_grid`：积分网格密度，越大越精确但更慢；
- `a_min`：年龄积分下限，过大将产生系统偏差。

调参建议：

- 日常验证：`n_grid` 在 `5e4` 到 `2e5` 已足够；
- 如需更稳健 `z_acc`，提高 `find_acceleration_transition_z` 的 `n_grid`；
- 若做高红移任务，先确认模型在更大 `z_max` 上 `E(a)^2` 仍为正。

## R13

该算法不是优化近似问题，不存在“近似比”指标；可给出的保证是数值一致性与物理约束一致性：

- 固定输入参数和网格时结果可复现；
- 对于 EdS，年龄与解析解误差被脚本断言限制在 `0.02 Gyr`；
- 对于 `LambdaCDM`，断言保证 `q0<0`、`z_acc` 落在合理区间、距离和 `H(z)` 量级一致。

## R14

常见失效模式：

1. 参数组合导致某段红移上 `E(a)^2<=0`，模型不物理；
2. 网格太粗导致积分和转折点估计偏差大；
3. 忽略单位换算（特别是 `H0` 从 `km/s/Mpc` 到 `s^-1`）；
4. 在不含暗能量模型里错误期待出现加速转折点。

## R15

工程化扩展方向：

- 参数拟合：接入 SNe Ia / BAO 数据做 `chi^2` 或贝叶斯推断；
- 模型扩展：加入 `wCDM`、`w0wa`、早期暗能量；
- 性能优化：向量化多参数扫描、并行化积分；
- 可视化：输出 `H(z)`, `q(z)`, `D_L(z)` 曲线图用于报告。

## R16

相关主题：

- Robertson-Walker 度规与宇宙学原理；
- 临界密度与密度参数 `Omega_i`；
- 宇宙距离阶梯（共动/光度/角径距离）；
- 加速膨胀与暗能量观测证据。

## R17

`demo.py` 的 MVP 功能清单：

- 定义宇宙学参数数据结构 `CosmologyParams`；
- 实现 `E(a)`, `H(z)`, `q(z)`；
- 实现宇宙年龄、lookback time、`D_C/D_M/D_L/D_A`；
- 自动求解 `q(z)=0` 的加速转折红移；
- 运行三组案例并做自动断言验收。

运行方式：

```bash
cd Algorithms/物理-宇宙学-0052-弗里德曼方程_(Friedmann_Equations)
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程（9 步）：

1. `CosmologyParams` 定义输入参数，并通过 `with_derived_omega_k` 保证密度参数闭合。  
2. `validate_params` 先检查参数数值合法性，并在 `z in [0,10]` 上验证 `E(a)^2 > 0`。  
3. `e2_of_a` / `e_of_a` / `h_of_z` 计算第一弗里德曼方程给出的膨胀率。  
4. `deceleration_parameter_of_z` 根据第二弗里德曼方程计算 `q(z)`。  
5. `age_of_universe_gyr` 以 `Integral da/[aH(a)]` 数值积分得到宇宙年龄。  
6. `line_of_sight_comoving_distance_mpc` 积分 `Integral dz/E(z)` 得到 `D_C`，再由 `transverse_comoving_distance_mpc` 处理曲率项得到 `D_M`。  
7. `luminosity_distance_mpc` 与 `angular_diameter_distance_mpc` 用 `D_M` 构造观测距离量。  
8. `find_acceleration_transition_z` 对 `q(z)` 网格扫描并线性插值得到 `z_acc`。  
9. `run_case`/`main` 批量运行三种宇宙模型、打印指标并执行断言，形成一次性非交互验收。  

说明：实现仅使用 `numpy` 的基础数组与积分能力，核心方程都在源码中逐项展开，没有调用黑盒宇宙学求解器。
