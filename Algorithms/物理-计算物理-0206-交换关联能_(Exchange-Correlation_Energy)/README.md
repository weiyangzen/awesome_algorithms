# 交换关联能 (Exchange-Correlation Energy)

- UID: `PHYS-0205`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `206`
- 目标目录: `Algorithms/物理-计算物理-0206-交换关联能_(Exchange-Correlation_Energy)`

## R01

交换关联能 `E_xc[n]` 是密度泛函理论（DFT）里最关键、也最难精确表达的能量项。  
它把多体电子系统中超出经典 Hartree 静电项的量子效应统一打包，包括：

- 交换效应（Pauli 原理导致的同自旋“交换空穴”）
- 关联效应（电子间瞬时相关运动）

本条目给出一个最小可运行的 LDA-XC 自洽原型，用来演示 `E_x`、`E_c`、`E_xc` 如何进入 Kohn-Sham SCF 迭代。

## R02

MVP 求解问题如下：

1. 在 1D 网格上构造闭壳层电子体系（`N_e=4`）；
2. 建立有效势 `v_eff = v_ext + v_H + v_x + v_c`；
3. 使用三对角本征求解器迭代更新密度，直到 SCF 收敛；
4. 输出总能分解 `E_kin, E_ext, E_H, E_x, E_c, E_xc, E_total` 并做阈值验收。

该实现重点是“算法透明”，不是材料级定量预测。

## R03

本实现的 Kohn-Sham 总能量表达式为：

`E[n] = T_s[n] + ∫ v_ext(x)n(x)dx + E_H[n] + E_x[n] + E_c[n]`

其中：

- `T_s[n]`：非相互作用参考体系动能
- `E_H[n] = 0.5 * ∫∫ n(x)n(x') / sqrt((x-x')^2 + a^2) dx dx'`
- `E_x[n]`：LDA 交换能
- `E_c[n]`：LDA 相关能（PZ81 参量化）

代码中 `total_energy` 会显式拆分并返回以上能量贡献。

## R04

交换项使用 Dirac 交换（非自旋极化电子气）：

- `eps_x(n) = -C_x * n^(1/3)`
- `C_x = 3/4 * (3/pi)^(1/3)`
- `E_x = ∫ n * eps_x(n) dx`
- `v_x = d(n*eps_x)/dn = -(3/pi)^(1/3) * n^(1/3)`

在 `demo.py` 中对应：

- `lda_exchange_per_particle`
- `lda_exchange_energy_density`
- `lda_exchange_potential`

## R05

相关项采用 Perdew-Zunger 1981（PZ81）非自旋极化参数化：

- 先由密度计算 `r_s = (3/(4*pi*n))^(1/3)`
- `r_s < 1` 与 `r_s >= 1` 采用分段公式得到 `eps_c(r_s)`
- `E_c = ∫ n * eps_c dx`
- 相关势用链式法则：`v_c = eps_c - (r_s/3) * d eps_c / d r_s`

在代码中分别由：

- `pz81_correlation_per_particle_from_rs`
- `pz81_correlation_derivative_from_rs`
- `lda_correlation_energy_density`
- `lda_correlation_potential`

显式实现，未调用黑盒 DFT 包。

## R06

模型与边界条件：

- 空间离散：1D 均匀网格 `x in [-8, 8]`
- 外势：谐振子 `v_ext = 0.5 * omega^2 * x^2`
- Hartree 核：软库仑 `1/sqrt((x-x')^2 + a^2)`（避免奇点）
- 闭壳层非自旋极化：`N_e` 必须为偶数
- SCF 稳定器：线性密度混合（`mix=0.26`）

该模型是教学型可审计近似，不对应具体真实材料。

## R07

`demo.py` 输入输出约定：

- 输入：全部由 `Grid1D`、`XCConfig` 内置配置给定，无交互输入。
- 输出：
1. 最近 8 次 SCF 迭代记录（`E_total`、`dE`、`drho_L2`、本征值、`E_x/E_c/E_xc`）
2. 最终能量分解与电子数积分
3. 阈值检查列表与 `Validation: PASS/FAIL`

脚本可直接执行：`uv run python demo.py`。

## R08

复杂度分析（网格点数 `N`，SCF 迭代步数 `K`）：

- Hartree 势矩阵-向量乘：每步 `O(N^2)`
- 三对角本征求解（取占据态）：每步约 `O(N^2)` 量级
- 交换/相关势与积分：每步 `O(N)`

总时间复杂度可近似视为 `O(K * N^2)`，空间复杂度由 Hartree 核矩阵主导，为 `O(N^2)`。

## R09

数值稳定性设计：

- 对密度做 `clip` 下界，避免 `n->0` 时 `n^(1/3)` 与 `r_s` 病态。
- 每次密度更新后做粒子数归一化，控制 `∫n dx = N_e` 漂移。
- 使用线性密度混合降低 SCF 振荡。
- 收敛采用双判据：`drho` 与 `dE` 同时满足阈值。
- 轨道按连续范数归一化，减少离散误差累积。

## R10

MVP 技术栈：

- `numpy`：网格、势场、密度、能量积分、向量化计算
- `scipy.linalg.eigh_tridiagonal`：Kohn-Sham 三对角本征问题
- `pandas`：迭代数据表格化输出

除基础数值工具外，`E_xc` 计算链条是手写可追踪实现，不是第三方“一键算 DFT”黑盒。

## R11

运行方式：

```bash
cd Algorithms/物理-计算物理-0206-交换关联能_(Exchange-Correlation_Energy)
uv run python demo.py
```

若所有检查通过，会输出 `Validation: PASS`；否则返回非零退出码。

## R12

主要输出字段解释：

- `E_total`：总能
- `E_kin`：非相互作用动能 `T_s`
- `E_ext`：外势能 `∫ v_ext n dx`
- `E_H`：Hartree 能
- `E_x`：交换能（通常为负）
- `E_c`：相关能（通常为负）
- `E_xc`：`E_x + E_c`
- `drho_L2`：相邻迭代密度差 L2 范数
- `dE`：相邻迭代总能变化绝对值
- `N_integral`：离散积分电子数（应接近 `N_e`）

## R13

内置验收条件（全部通过才 PASS）：

1. `SCF converged`
2. `density residual < 1e-4`
3. `electron-number error < 1e-6`
4. `occupied-bandwidth > 1e-3`
5. `exchange energy is negative`
6. `correlation energy is negative`
7. `|E_xc| > |E_x|`

这组检查覆盖了收敛性、守恒性和 `xc` 项的基本物理符号性质。

## R14

当前实现局限：

- 1D 软库仑模型 + 3D 均匀电子气参数化属于教学近似。
- 使用 LDA 级别 `xc`，未含 GGA/meta-GGA/hybrid 修正。
- 仅非自旋极化闭壳层，不包含自旋分辨（LSDA）。
- SCF 加速仅线性混合，复杂体系可能收敛慢。

## R15

可扩展方向：

- 扩展到自旋分辨 LSDA（`n_up/n_down`）。
- 换用 GGA 或 meta-GGA 的 `E_xc` 形式并比较趋势。
- 引入 DIIS/Pulay/Anderson mixing 提升 SCF 收敛速度。
- 把 Hartree `O(N^2)` 矩阵法替换为 FFT 卷积或 Poisson 求解。
- 从 1D 网格扩展到 2D/3D 或基组表示。

## R16

应用语境：

- DFT 教学中展示 `E_xc` 如何进入 Kohn-Sham 自洽回路。
- 作为测试 `v_xc` 实现正确性的最小基线。
- 用于快速验证密度混合策略与收敛判据。
- 为后续 LDA/GGA/更高阶泛函实现提供可运行母版。

## R17

与相邻近似层级关系：

- 本条目：聚焦“`E_xc` 是什么、如何在 SCF 中被计算与使用”。
- LDA 条目（0207）：更偏向局域交换近似本身。
- GGA 条目（0208）：强调密度梯度修正 `∇n` 对交换能的增强机制。

因此本条目定位为“交换+相关合并项的基础实现层”，是后续泛函升级的前置台阶。

## R18

`demo.py` 源码级算法流（9 步）：

1. `Grid1D` 与 `XCConfig` 设定网格、电子数、混合参数和收敛阈值，`require_even_electrons` 强制闭壳层前提。  
2. `build_kinetic_tridiagonal` 构造动能三对角算符，`build_hartree_kernel` 预计算软库仑核矩阵。  
3. 初始化密度并用 `normalize_density` 归一化到指定 `N_e`。  
4. 每次 SCF 迭代中先计算 `v_H`、`v_x`、`v_c`：`v_x` 由 Dirac 交换给出，`v_c` 由 PZ81 的 `eps_c` 与 `d eps_c/dr_s` 组装。  
5. 组装 `v_eff = v_ext + v_H + v_x + v_c`，并用 `scipy.linalg.eigh_tridiagonal` 求最低占据轨道。  
6. `density_from_orbitals` 生成输出密度 `n_out`，做线性混合得到 `n_new` 并再次归一化。  
7. `total_energy` 显式计算 `E_kin, E_ext, E_H, E_x, E_c, E_xc, E_total`，同时记录 `drho_L2` 与 `dE`。  
8. 若 `drho` 与 `dE` 同时小于阈值则判定收敛，否则继续迭代直到 `max_iter`。  
9. `main` 打印末尾迭代记录、最终能量分解，执行 7 条阈值检查并给出 `Validation: PASS/FAIL`。

说明：虽然本实现调用了 `numpy/scipy/pandas`，但 `n -> r_s -> eps_c -> v_c -> SCF -> E_xc` 的核心算法链均可在源码逐行追踪。
