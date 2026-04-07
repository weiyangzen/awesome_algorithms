# LDA近似 (Local Density Approximation)

- UID: `PHYS-0206`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `207`
- 目标目录: `Algorithms/物理-计算物理-0207-LDA近似_(Local_Density_Approximation)`

## R01

LDA（Local Density Approximation）是密度泛函理论（DFT）中最基础的交换-相关近似：
在每个空间点 `r`，把真实非均匀电子气近似为“局域均匀电子气”，再使用均匀电子气的能量密度公式。

本条目实现的是一个可运行、可审计的 Kohn-Sham LDA 最小原型，重点是展示“自洽迭代（SCF）+ 局域交换势”的计算流程。

## R02

MVP 采用 1D 实空间网格上的玩具模型：
- 外势：谐振子势 `v_ext(x) = 0.5 * omega^2 * x^2`
- 电子数：闭壳层 `N_e = 4`（自旋简并，每个轨道占据数为 2）
- Hartree 项：软库仑核 `1/sqrt((x-x')^2 + a^2)`
- 交换项：使用 3D 均匀电子气的 LDA 交换公式（相关项在 MVP 中省略）

该模型并非材料级精确物理模型，而是用于清晰复现 LDA 的核心算法结构。

## R03

Kohn-Sham 总能量（本实现的 exchange-only 版本）写为：

`E[n] = T_s[n] + ∫ v_ext(x) n(x) dx + E_H[n] + E_x^{LDA}[n]`

其中：
- `T_s[n]`：非相互作用参考体系的动能
- `E_H[n] = 0.5 * ∫∫ n(x) n(x') / sqrt((x-x')^2 + a^2) dx dx'`
- `E_x^{LDA}[n]`：局域交换能

SCF 的目标是求解使能量泛函驻值的自洽密度 `n(x)`。

## R04

交换项采用标准 3D-LDA 交换表达式：

`E_x[n] = C_x ∫ n(x)^(4/3) dx,  C_x = -3/4 * (3/pi)^(1/3)`

对应交换势（泛函导数）：

`v_x(x) = dE_x/dn = -(3/pi)^(1/3) * n(x)^(1/3)`

在代码中分别由：
- `lda_exchange_energy_density`
- `lda_exchange_potential`

显式实现，不依赖黑盒 DFT 包。

## R05

本实现的边界与假设：
- 闭壳层、非自旋极化（`N_e` 必须为偶数）。
- 1D 网格离散与有限盒边界，属于教学型离散近似。
- 使用软库仑核避免 `x=x'` 奇点。
- 只做 exchange-only LDA，不含相关泛函（如 VWN/PZ81）。
- 通过线性密度混合稳定 SCF，而不是使用 Pulay/DIIS。

## R06

`demo.py` 输入输出约定：
- 输入：脚本内部固定参数（网格、电子数、混合系数、收敛阈值等）。
- 输出：
1. 末尾 8 次 SCF 迭代记录（能量、残差、占据本征值等）
2. 最终摘要表（总能、分解能、粒子数误差）
3. 阈值检查与 `Validation: PASS/FAIL`

脚本无交互输入，直接可运行。

## R07

算法主流程（高层）：
1. 构建网格、外势、动能三对角算符与 Hartree 核矩阵。
2. 初始化电子密度并归一化到 `N_e`。
3. 根据当前密度计算 `v_H` 与 `v_x`，组装 `v_eff`。
4. 求解 Kohn-Sham 三对角本征问题，得到占据轨道。
5. 从轨道重建新密度 `n_out(x)`。
6. 对密度做线性混合得到 `n_new` 并归一化。
7. 计算总能、密度残差 `drho`、能量变化 `dE`。
8. 若满足收敛阈值则停止，否则继续迭代。

## R08

设网格点数为 `N`，SCF 迭代步数为 `K`：
- Hartree 势（矩阵乘法）每步 `O(N^2)`。
- 三对角本征求解（取最低占据态）约 `O(N * n_occ)` 到 `O(N^2)`（取决于实现细节与选态方式）。
- 其余向量操作 `O(N)`。

总体时间复杂度可近似记为 `O(K * N^2)`，空间复杂度主要由 Hartree 核矩阵主导，为 `O(N^2)`。

## R09

数值稳定性处理：
- 密度在交换势计算前做 `clip`，避免 `n^(1/3)` 的零点病态。
- 每次迭代后都强制密度归一化，控制粒子数漂移。
- 轨道做连续归一化（含 `dx`），避免离散归一误差积累。
- 使用密度混合（`mix=0.28`）降低 SCF 振荡风险。
- 同时使用 `drho` 和 `dE` 双阈值判据防止伪收敛。

## R10

MVP 技术栈：
- `numpy`：网格、密度、势能与线性代数向量化
- `scipy.linalg.eigh_tridiagonal`：求解 Kohn-Sham 三对角本征问题
- `pandas`：迭代表格化输出

未调用任何“单函数完成 DFT”的黑盒包，关键步骤都在源码可追踪。

## R11

运行方式：

```bash
cd Algorithms/物理-计算物理-0207-LDA近似_(Local_Density_Approximation)
uv run python demo.py
```

通过后会输出 `Validation: PASS`；失败会 `SystemExit(1)` 并显示失败检查项。

## R12

关键输出字段说明：
- `E_total`：总能量 `T_s + E_ext + E_H + E_x`
- `E_kin`：动能 `T_s`
- `E_classical`：`E_ext + E_H`
- `E_x`：LDA 交换能（应为负）
- `drho_L2`：相邻迭代密度差的 L2 范数
- `dE`：相邻迭代总能变化绝对值
- `eps_min` / `eps_max_occ`：最低占据与最高占据 Kohn-Sham 本征值
- `N_integral` / `N_error`：粒子数积分及误差

## R13

demo 内置正确性检查：
1. `SCF converged` 为真
2. `drho_L2 < 1e-4`
3. `|N_integral - N_e| < 1e-6`
4. 占据能级带宽 `eps_max_occ - eps_min > 1e-3`
5. `E_x < 0`

全部满足则判定 `Validation: PASS`。

## R14

当前实现局限：
- 1D 软库仑 + 3D 交换公式是教学近似，不可直接用于定量材料预测。
- 未包含相关能泛函，物理精度有限。
- 未实现自旋极化（LSDA）或广义梯度近似（GGA）。
- SCF 加速仅为线性混合，复杂体系可能收敛慢。

## R15

可扩展方向：
- 增加相关泛函（如 PZ81）形成 LDA-xc。
- 引入自旋分辨密度，升级到 LSDA。
- 使用 DIIS/Pulay 或 Anderson mixing 提升收敛效率。
- 用 FFT 卷积或 Poisson 求解器替代 `O(N^2)` Hartree 计算。
- 将 1D 玩具模型扩展为 2D/3D 网格或基组框架。

## R16

典型应用语境：
- DFT 教学中演示 Kohn-Sham SCF 主循环。
- 测试新混合策略、新收敛判据的快速原型。
- 交换-相关势数值实现的单元级验证。
- 作为更复杂电子结构程序的“最小可运行基线”。

## R17

与常见近似方法对比（简述）：
- LDA：局域、结构简单、稳定，适合作为第一层基线。
- GGA：引入密度梯度，通常比 LDA 更准确但实现更复杂。
- meta-GGA/杂化泛函：精度更高但代价显著增加。

本条目选择 LDA 的原因是：公式闭合、实现短小、可完整走通从势构造到 SCF 收敛的算法链路。

## R18

`demo.py` 的源码级算法流（9 步）：
1. `Grid1D` 与 `LDAConfig` 定义离散空间和 SCF 参数，`require_even_electrons` 强制闭壳层前提。
2. `build_kinetic_tridiagonal` 构建 `T=-0.5*d^2/dx^2` 三对角离散算符；`build_hartree_kernel` 预计算软库仑核矩阵。
3. `run_scf_lda` 初始化密度 `n(x)` 并通过 `normalize_density` 归一到给定电子数。
4. 每轮迭代先由 `hartree_potential` 和 `lda_exchange_potential` 生成 `v_H` 与 `v_x`，再与 `v_ext` 组装 `v_eff`。
5. `solve_kohn_sham_orbitals` 调用 `scipy.linalg.eigh_tridiagonal` 求最低占据本征态；`normalize_orbitals` 用连续范数重标定轨道。
6. `density_from_orbitals` 计算 `n_out(x)=2*sum|psi_i|^2`，随后做线性混合得到 `n_new` 并再次归一化。
7. `total_energy` 通过 `kinetic_energy`、Hartree 项与 `lda_exchange_energy_density` 计算 `E_total` 及能量分解。
8. 记录 `drho_L2`、`dE`、本征值窗口与粒子数积分；满足双阈值 (`tol_density`, `tol_energy`) 则收敛退出。
9. `main` 汇总迭代表与终态指标，执行 5 条阈值检查并输出 `Validation: PASS/FAIL`。
