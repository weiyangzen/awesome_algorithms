# 杂化泛函 (Hybrid Functionals)

- UID: `PHYS-0208`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `209`
- 目标目录: `Algorithms/物理-计算物理-0209-杂化泛函_(Hybrid_Functionals)`

## R01

杂化泛函（Hybrid Functionals）是在 Kohn-Sham DFT 中把一部分精确交换（Hartree-Fock exchange）与半局域交换泛函混合，从而兼顾可计算性与精度。

本条目实现一个可运行、可审计的最小原型：在 1D 实空间网格上做闭壳层 SCF，显式构造
`E_x^hyb = alpha * E_x^HF + (1-alpha) * E_x^LDA`，并输出收敛与能量分解指标。

## R02

MVP 范围（刻意简化）：

1. 体系是 1D 谐振子外势中的闭壳层电子（`N_e=4`）。
2. 电子-电子作用使用软库仑核 `1/sqrt((x-x')^2 + a^2)`。
3. 相关能不单独建模（只做 exchange-only hybrid）。
4. SCF 加速仅用线性密度混合，不使用 DIIS/Pulay。
5. 用密集对角化求占据轨道，强调流程透明而不是大规模性能。

## R03

本实现的能量泛函写为：

`E[n, {psi_i}] = T_s + E_ext + E_H + E_x^hyb`

其中：

- `T_s`: 非相互作用动能（由占据轨道求和）
- `E_ext = ∫ v_ext(x) n(x) dx`
- `E_H = 0.5 * ∫∫ n(x) n(x') w(x,x') dx dx'`
- `E_x^hyb = alpha * E_x^HF + (1-alpha) * E_x^LDA`
- `w(x,x') = 1/sqrt((x-x')^2 + a^2)`

## R04

交换项的两部分：

1. LDA 交换（局域）：
- `E_x^LDA = C_x ∫ n(x)^(4/3) dx`
- `v_x^LDA = dE_x^LDA/dn = -(3/pi)^(1/3) * n(x)^(1/3)`

2. 精确交换（非局域算符，离散后为矩阵）：
- `F_x[p,q] = -dx * sum_i psi_i[p] psi_i[q] * w[p,q]`
- `E_x^HF = sum_i <psi_i | F_x | psi_i>`

最终 Kohn-Sham 有效哈密顿量在代码中写成：
`H = T + diag(v_ext + v_H + (1-alpha) v_x^LDA) + alpha * F_x`。

## R05

模型假设与边界：

1. 仅支持闭壳层非自旋极化（`n_electrons` 必须为偶数）。
2. 1D 有限盒 + Dirichlet 边界，属于教学近似。
3. 软库仑核避免 `x=x'` 奇异性。
4. 未加入相关能（如 LYP/PBE correlation），因此不是生产级 B3LYP/PBE0 复现。
5. 数值目标是“流程正确 + 稳定收敛”，不是材料级定量精度。

## R06

`demo.py` 输入输出约定：

- 输入：脚本内置参数（网格、混合系数、`alpha_hf`、收敛阈值）。
- 输出：
1. SCF 末尾 8 轮迭代表。
2. 最终摘要表（总能、分量、残差、粒子数误差）。
3. 阈值检查结果和 `Validation: PASS/FAIL`。

脚本无交互输入，直接运行即可。

## R07

算法主流程（高层）：

1. 建立网格、动能矩阵、外势与软库仑核矩阵。
2. 初始化密度并归一化到 `N_e`。
3. 用初始密度解一次局域哈密顿量，得到初始占据轨道。
4. 每轮 SCF 用上一轮轨道构造非局域 `F_x`。
5. 组装杂化哈密顿量并对角化，得到新占据轨道。
6. 由轨道重建新密度，做线性混合并归一化。
7. 计算总能与 `drho/dE` 收敛指标。
8. 满足双阈值后停止，否则继续迭代。

## R08

设网格内点数为 `N`，占据轨道数为 `n_occ`，SCF 轮数为 `K`：

1. Hartree 势矩阵向量乘：每轮 `O(N^2)`。
2. 构造精确交换矩阵（含密度矩阵与逐点核乘）：每轮约 `O(N^2 + N*n_occ^2)`。
3. 密集对角化 `eigh`：每轮主成本 `O(N^3)`。

总体时间复杂度近似 `O(K * N^3)`，空间复杂度主要由核矩阵与哈密顿量 `O(N^2)` 决定。

## R09

数值稳定性措施：

1. 交换势中的密度先 `clip`，避免 `n^(1/3)` 数值病态。
2. 每轮混合后强制密度归一化，控制粒子数漂移。
3. 每个轨道按连续范数 `∫|psi|^2 dx=1` 重归一化。
4. 固定轨道相位（峰值点取正），减少迭代记录抖动。
5. 同时使用 `drho` 与 `dE` 判据，降低伪收敛风险。

## R10

MVP 技术栈：

- `numpy`：网格离散、核矩阵、密度与能量计算。
- `scipy.linalg.eigh`：对称密集哈密顿量对角化。
- `pandas`：迭代表和摘要表输出。

没有调用现成“单函数完成 DFT/Hybrid Functional”的黑盒软件包。

## R11

运行方式：

```bash
cd Algorithms/物理-计算物理-0209-杂化泛函_(Hybrid_Functionals)
uv run python demo.py
```

成功时会打印 `Validation: PASS`；失败则 `SystemExit(1)`。

## R12

关键输出字段说明：

- `E_total`: 总能 `E_kin + E_ext + E_H + E_x_hyb`
- `E_kin`: 动能 `T_s`
- `E_ext`: 外势能
- `E_H`: Hartree 能
- `E_x_hf`: 精确交换能（应为负）
- `E_x_lda`: LDA 交换能（应为负）
- `E_x_hyb`: 杂化交换能
- `drho_L2`: 相邻两轮密度差 L2 范数
- `dE`: 相邻两轮总能差
- `eps_min` / `eps_max_occ`: 最低占据与最高占据本征值
- `N_integral` / `N_error`: 粒子数积分及误差

## R13

`demo.py` 内置验收检查：

1. `SCF converged`。
2. `drho_L2 < 1e-4`。
3. `dE < 1e-5`。
4. `|N_integral - N_e| < 1e-6`。
5. 占据能级带宽 `eps_max_occ - eps_min > 1e-3`。
6. `E_x_hyb < 0`。

全部满足输出 `Validation: PASS`。

## R14

当前实现局限：

1. 1D 模型无法直接映射真实 3D 材料体系。
2. 只含交换混合，不含独立相关项，因此不是完整工业杂化泛函。
3. 未处理自旋极化、开壳层、周期边界。
4. 采用密集对角化，规模扩展能力有限。

## R15

可扩展方向：

1. 加入相关项并实现 PBE0/B3LYP 风格参数化。
2. 升级到自旋分辨（RKS/UKS）与开壳层处理。
3. 使用 DIIS/Pulay 加速 SCF 收敛。
4. 结合 FFT/Poisson 求解改进 Hartree 计算。
5. 将 1D 原型迁移到 2D/3D 网格或高斯基组框架。

## R16

典型应用语境：

1. 教学中解释“局域交换 + 精确交换混合”的核心机制。
2. 验证新型混合参数 `alpha` 对收敛和能量分解的影响。
3. 作为更复杂电子结构程序的前置原型与单元验证模块。

## R17

与相关近似的对比（简述）：

1. LDA/GGA：成本低、局域势简单，但交换处理较粗。
2. 纯 HF：交换精确但缺失 DFT 相关，且常导致体系误差偏大。
3. 杂化泛函：通过 `alpha` 在两者间折中，常在结构与能隙预测上更稳健，但代价更高。

本条目选择杂化原型，目的就是把“非局域交换算符进入 SCF”这一步透明化。

## R18

`demo.py` 源码级算法流（9 步）：

1. `Grid1D`/`HybridConfig` 定义离散空间和 SCF 参数，`require_even_electrons` 约束闭壳层前提。
2. `build_kinetic_matrix` 生成有限差分动能矩阵，`build_soft_coulomb_kernel` 预计算 `w[p,q]`。
3. `run_scf_hybrid` 初始化密度后，用局域哈密顿量先求一组初始占据轨道。
4. 每轮迭代由 `exact_exchange_matrix` 用当前占据轨道构造非局域 `F_x`，并与 `v_H`、`v_x^LDA` 组装杂化哈密顿量。
5. `solve_lowest_orbitals` 调用 `scipy.linalg.eigh(..., driver="evd")` 求最低占据本征态。
6. 在 SciPy/LAPACK 路径中，`evd` 对应实对称 `syevd`：先 Householder 把矩阵化为三对角，再用分治法求本征值/向量，最后反变换回原空间。
7. `density_from_orbitals` 生成 `n_out(x)`，随后线性混合并归一化得到 `n_new`。
8. `energy_components` 分别计算 `E_kin/E_ext/E_H/E_x_hf/E_x_lda/E_x_hyb` 与 `E_total`，同时记录 `drho`、`dE`、本征值窗口。
9. `main` 打印迭代尾表与摘要表，执行阈值检查并输出 `Validation: PASS/FAIL`。
