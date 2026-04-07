# Kohn-Sham方程 (Kohn-Sham Equations)

- UID: `PHYS-0204`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `205`
- 目标目录: `Algorithms/物理-计算物理-0205-Kohn-Sham方程_(Kohn-Sham_Equations)`

## R01

Kohn-Sham 方程是密度泛函理论（DFT）的核心计算框架：
它把“强相互作用多电子问题”映射为“在有效势中运动的非相互作用单电子方程组”，再通过自洽迭代得到电子密度。

本条目实现的是一个可运行、可审计的最小原型（MVP），重点展示 Kohn-Sham SCF 主循环，而不是追求材料级高精度。

## R02

MVP 采用 1D 实空间教学模型：

- 外势：`v_ext(x) = 0.5 * omega^2 * x^2`（谐振子阱）
- Hartree 项：软库仑核 `K(x,x') = 1/sqrt((x-x')^2 + a^2)`
- 交换相关项：为闭合方程，使用 exchange-only LDA 势
- 电子数：闭壳层 `N_e = 4`（每个占据轨道双占据）

这保证了算法链路完整，同时代码保持简洁。

## R03

本实现对应的能量泛函近似写为：

`E[n] = T_s[n] + ∫ v_ext(x)n(x)dx + E_H[n] + E_x[n]`

其中：

- `E_H[n] = 0.5 * ∫∫ n(x)n(x')/sqrt((x-x')^2+a^2) dxdx'`
- `E_x[n] = C_x * ∫ n(x)^(4/3) dx`
- `C_x = -3/4 * (3/pi)^(1/3)`

对应有效势：

`v_eff(x) = v_ext(x) + v_H(x) + v_x(x)`

并求解离散 Kohn-Sham 本征问题 `H_KS psi_i = eps_i psi_i`。

## R04

离散策略：

- 空间均匀网格 `x_j`，步长 `dx`
- 动能算符 `T = -0.5 d^2/dx^2` 用二阶中心差分，形成三对角矩阵
- Kohn-Sham 哈密顿量为三对角 + 对角势 `v_eff`
- 通过 `scipy.linalg.eigh_tridiagonal` 仅取最低 `n_occ = N_e/2` 个本征态
- 密度重建：`n(x)=2*sum_{occ}|psi_i(x)|^2`

该离散方式简单、透明、便于核验每个算子来源。

## R05

物理与数值假设：

- 非自旋极化闭壳层（`N_e` 必须为偶数）
- 有限计算盒 + 1D 网格（非真实三维周期边界条件）
- 软库仑核用于消除 `x=x'` 奇点
- exchange-only LDA（未加入相关泛函）
- SCF 稳定策略使用线性密度混合，不用 DIIS/Pulay

## R06

`demo.py` 的输入输出约定：

- 输入：脚本内固定参数（网格范围、电子数、混合系数、收敛阈值）
- 输出：
1. 最近 8 步 SCF 迭代记录
2. 最终能量分解与误差摘要表
3. 检查项明细与 `Validation: PASS/FAIL`

运行全程无交互输入，适合自动验证。

## R07

SCF 主流程（高层）：

1. 构建网格、动能三对角算符、外势与 Hartree 核矩阵。
2. 初始化密度并归一化到目标电子数。
3. 由当前密度计算 `v_H` 与 `v_x`，组装 `v_eff`。
4. 求解 Kohn-Sham 本征问题得到占据轨道。
5. 从轨道重建新密度 `n_out`。
6. 对密度线性混合得到 `n_new`，并再归一化。
7. 计算能量分解、密度残差 `drho`、能量变化 `dE`。
8. 满足双阈值则停止，否则继续迭代。

## R08

设网格点数为 `N`，迭代步数为 `K`：

- Hartree 势（核矩阵乘法）每步约 `O(N^2)`
- 三对角本征求解（最低占据态）约 `O(N * n_occ)` 到 `O(N^2)`
- 其余向量操作约 `O(N)`

整体时间复杂度可近似写为 `O(K * N^2)`，
空间复杂度由 Hartree 核矩阵主导，为 `O(N^2)`。

## R09

稳定性措施：

- 交换项前对密度做 `clip(n, 1e-14, +inf)`，避免 `n^(1/3)` 数值病态
- 每轮都强制密度归一化，控制粒子数漂移
- 轨道按连续范数 `∫|psi|^2dx=1` 重标定
- 采用密度混合 `mix=0.30` 缓解振荡
- 使用 `drho + dE` 双判据，降低伪收敛风险

## R10

最小工具栈：

- `numpy`：网格、势能、密度、向量化运算
- `scipy.linalg.eigh_tridiagonal`：三对角本征求解
- `pandas`：迭代表与摘要表输出

未调用高层 DFT 黑盒包，关键算法均在源码中显式实现。

## R11

运行方式：

```bash
cd Algorithms/物理-计算物理-0205-Kohn-Sham方程_(Kohn-Sham_Equations)
uv run python demo.py
```

成功时结尾会打印 `Validation: PASS`；
若检查失败会以非零退出码结束。

## R12

关键输出字段说明：

- `E_total`：总能 `E_kin + E_ext + E_H + E_x`
- `E_kin`：非相互作用参考体系动能
- `E_ext`：外势能 `∫ v_ext n`
- `E_H`：Hartree 能（应为正）
- `E_x`：LDA 交换能（应为负）
- `drho_L2`：相邻迭代密度差范数
- `dE`：相邻迭代总能变化
- `E_sum_eigs`：占据 Kohn-Sham 本征值之和
- `E_ks_identity`：按双计数修正后的能量恒等式估计
- `N_integral / N_error`：电子数积分与误差

## R13

内置验证条件：

1. `SCF converged`
2. `drho_L2 < 1e-4`
3. `|N_integral - N_e| < 1e-6`
4. `E_H > 0`
5. `E_x < 0`
6. `|E_total - E_ks_identity| < 2e-2`

全部通过才输出 `Validation: PASS`。

## R14

当前实现局限：

- 1D + 软库仑 + LDA 交换是教学近似，不能直接用于真实材料定量预测
- 未含相关能（仅 exchange-only）
- 未支持自旋极化、周期边界、k 点采样
- SCF 加速手段较基础，复杂体系可能收敛慢

## R15

可扩展方向：

- 加入相关泛函，形成完整 LDA-xc
- 升级到 LSDA/GGA
- 引入 DIIS、Anderson mixing 等收敛加速
- 用 FFT 卷积或 Poisson 求解加速 Hartree 项
- 扩展到 2D/3D 网格或基组表示

## R16

适用语境：

- 教学中解释 Kohn-Sham 方程如何被数值化
- 快速验证新的 SCF 终止准则或混合策略
- 作为更大电子结构程序的可运行最小骨架
- 做算法结构验证（算子、能量分解、收敛日志）

## R17

与相邻方法对比：

- Hartree 方法：缺少交换相关势，通常误差更大
- Kohn-Sham + LDA（本实现）：成本低、收敛稳、结构清晰
- Kohn-Sham + GGA/杂化泛函：精度更高但实现和计算成本显著增加

因此本条目优先体现“可解释 + 可运行 + 可验证”的 Kohn-Sham 基线。

## R18

`demo.py` 的源码级算法流（9 步）：

1. `Grid1D` 和 `KSConfig` 定义离散空间与 SCF 超参数，`require_closed_shell` 约束偶数电子数。
2. `build_kinetic_tridiagonal` 生成差分动能三对角算符，`build_soft_coulomb_kernel` 预计算 Hartree 核矩阵。
3. `run_scf_kohn_sham` 初始化高斯密度并用 `normalize_density` 归一化到目标电子数。
4. 每轮迭代先由 `hartree_potential` 与 `lda_exchange_energy_density_and_potential` 计算 `v_H`、`v_x`，再组装 `v_eff`。
5. `solve_kohn_sham` 调用 `eigh_tridiagonal` 求最低占据本征态；`normalize_columns` 用连续范数归一轨道。
6. `density_from_orbitals` 重建 `n_out`，随后进行线性混合得到 `n_new` 并再次归一化。
7. `total_energy` 计算 `E_total/E_kin/E_ext/E_H/E_x`；并额外计算 `E_sum_eigs` 与 `E_ks_identity` 进行双计数一致性检查。
8. 记录 `drho_L2`、`dE`、本征值窗口和电子数积分，满足双阈值则收敛退出。
9. `main` 输出尾部迭代表、最终摘要和 6 项阈值检查，最终给出 `Validation: PASS/FAIL`。
