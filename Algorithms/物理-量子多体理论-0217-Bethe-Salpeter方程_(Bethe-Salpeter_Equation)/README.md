# Bethe-Salpeter方程 (Bethe-Salpeter Equation)

- UID: `PHYS-0216`
- 学科: `物理`
- 分类: `量子多体理论`
- 源序号: `217`
- 目标目录: `Algorithms/物理-量子多体理论-0217-Bethe-Salpeter方程_(Bethe-Salpeter_Equation)`

## R01

Bethe-Salpeter 方程（BSE）是多体理论中处理电子-空穴两体关联的核心工具，常用于计算激子（exciton）能级和光学吸收谱。

在固体光学问题里，独立粒子近似只给出准粒子跃迁能 `E_c - E_v`；BSE 通过显式电子-空穴相互作用核 `K`，把这些跃迁耦合成激子本征态，从而产生：
- 吸收峰红移；
- 带隙边下方的束缚激子峰；
- 振子强度在不同激子态之间重分配。

## R02

本条目目标是提供一个可运行、可审计的最小 BSE MVP（教学级，不是材料生产级计算）：

1. 构造一维 `k` 网格上的单价带-单导带电子-空穴基；
2. 构造无相互作用跃迁能 `DeltaE(k)`；
3. 显式组装直接吸引项 + 交换排斥项核矩阵；
4. 形成 `H_BSE` 并求解本征值/本征矢；
5. 计算振子强度与吸收谱；
6. 对比“独立粒子谱 vs BSE 谱”，输出束缚能与红移并做阈值校验。

## R03

MVP 使用的 BSE（Tamm-Dancoff 近似）离散形式：

`sum_{k'} H_{k,k'} A_S(k') = Omega_S A_S(k)`

其中

`H_{k,k'} = DeltaE(k) * delta_{k,k'} + K_{k,k'}`

`DeltaE(k) = E_c(k) - E_v(k)`

`K_{k,k'} = K_dir(k,k') + K_exc(k,k')`

本实现采用高斯形核：

`K_dir(k,k') = -W0 * exp(-(k-k')^2/(2*sigma_d^2)) / Nk`

`K_exc(k,k') = +Vx * exp(-(k-k')^2/(2*sigma_x^2)) / Nk`

激子束缚能定义为

`E_bind = E_gap_edge - Omega`

其中 `E_gap_edge = min_k DeltaE(k)`。

## R04

本 MVP 的模型设定：

- 单价带 + 单导带；
- 一维对称 `k` 网格（`Nk=61`）；
- 准粒子跃迁能取抛物线近似：
`DeltaE(k) = E_gap + (alpha_c + alpha_v) k^2`；
- 直接核为吸引（负号），交换核为排斥（正号）；
- 使用 TDA，仅求解 resonant block；
- 不包含自旋结构、多带耦合、`k` 点权重非均匀性、频率依赖核。

## R05

`demo.py` 默认参数（`BSEConfig`）：

- `qp_gap=2.20 eV`：准粒子直接带隙边；
- `alpha_c=10.0, alpha_v=8.0`：色散曲率；
- `direct_strength=1.20`：直接吸引强度；
- `exchange_strength=0.18`：交换排斥强度；
- `sigma_direct=0.07, sigma_exchange=0.16`：核宽度；
- `dipole_width=0.12`：偶极矩分布宽度；
- `broadening=0.03 eV`：Lorentz 展宽。

这组参数会产生可见束缚激子（`E_bind > 0`）和明显谱线红移。

## R06

输入输出约定：

- 输入：全部写死在 `BSEConfig`（无交互输入）；
- 输出：
1. 最低若干激子态表（能量、束缚能、振子强度）；
2. BSE 汇总表（QP 边、最低激子、亮激子、峰位红移、误差指标）；
3. 阈值检查列表与 `Validation: PASS/FAIL`。

若任一阈值失败，脚本以非零退出码结束。

## R07

算法流程（高层）：

1. 生成对称 `k` 网格；
2. 计算独立粒子跃迁 `DeltaE(k)`；
3. 构建偶极矩分布 `d(k)`；
4. 构建 `K_dir + K_exc` 核矩阵；
5. 组装 `H_BSE = diag(DeltaE) + K`；
6. 对 `H_BSE` 做对称本征分解得到 `Omega_S, A_S`；
7. 计算激子振子强度 `f_S = |sum_k d(k) A_S(k)|^2 / Nk`；
8. 构造 interacting 与 independent 两套吸收谱并比较峰位；
9. 汇总束缚能、红移和数值一致性检查。

## R08

复杂度分析（`N = Nk`）：

- 构建核矩阵：`O(N^2)`；
- 组装哈密顿量：`O(N^2)`；
- 对称本征分解 `eigh`：`O(N^3)`（主导项）；
- 计算光谱（`Nw` 个频点）：`O(Nw * N)`。

总复杂度主导于本征求解，空间复杂度约 `O(N^2)`（核与哈密顿量矩阵）。

## R09

数值稳定策略：

- 强制 `Nk` 为奇数，保证网格以 `k=0` 为中心对称；
- 对输入参数做正值/非负约束，避免非物理设定；
- 显式检查哈密顿量有限性后再本征分解；
- 通过 `max|H-H^T|` 验证对称性；
- 通过本征矢归一误差与振子强度和规则误差做后验一致性检查；
- 通过 Lorentz 展宽避免离散谱在可视化时的奇异尖峰。

## R10

MVP 技术栈：

- `numpy`：网格、矩阵和向量化计算；
- `scipy.linalg.eigh`：实对称 BSE 哈密顿量本征求解；
- `pandas`：表格化输出。

实现没有调用“黑盒 BSE 软件包”，核构造与物理量映射均在源码显式展开。

## R11

运行方式：

```bash
cd Algorithms/物理-量子多体理论-0217-Bethe-Salpeter方程_(Bethe-Salpeter_Equation)
uv run python demo.py
```

脚本不需要命令行参数，也不需要交互输入。

## R12

关键输出字段说明：

- `Omega_exciton`：激子本征能量；
- `binding_vs_qp_edge`：相对 QP 边的束缚能（正值表示束缚态）；
- `oscillator_strength`：对应激子态振子强度；
- `is_bright`：是否为亮态（阈值 `1e-3`）；
- `QP edge`：`min_k DeltaE(k)`；
- `independent/interacting absorption peak`：两种谱线峰位；
- `peak redshift`：独立粒子峰位减去 BSE 峰位；
- `oscillator sum-rule error`：`sum(f_S)` 与 `sum(d(k)^2/Nk)` 的差值。

## R13

`demo.py` 内置验收阈值：

1. `max|H-H^T| < 1e-12`；
2. 本征矢归一化误差 `< 1e-10`；
3. 振子强度和规则误差 `< 1e-10`；
4. 最低激子能量低于 QP 边；
5. 亮激子束缚能位于 `(0.05, 0.50) eV`；
6. interacting 吸收峰相对 independent 峰发生正红移；
7. 吸收谱数组全部有限。

全部满足时输出 `Validation: PASS`。

## R14

当前实现局限：

- 仅单带电子-空穴基，不含多带与自旋结构；
- 使用静态、参数化高斯核，不是从 `GW` 屏蔽相互作用直接构建；
- 未引入非 Hermitian 全 BSE（anti-resonant block）；
- 未处理真实晶体中的跃迁矩阵元、k 权重、对称性简化；
- 吸收谱只做线形函数卷积，未考虑声子和温度效应。

## R15

可扩展方向：

- 扩展到多价带/多导带块矩阵；
- 用 `GW` 输出替换模型化 `DeltaE` 与核参数；
- 加入频率依赖核或顶点修正；
- 从 TDA 扩展到完整 BSE（含 anti-resonant block）；
- 加入极化方向分辨、k 点权重与实验可比的介电函数输出。

## R16

典型应用语境：

- 半导体和绝缘体的激子能级预测；
- 光吸收起始边和激子峰解释；
- `DFT/GW + BSE` 工作流中的末端光学谱计算；
- 量子多体课程中“独立粒子谱 -> 激子谱”机制演示。

## R17

与相关方法的关系：

- 独立粒子近似（RPA without e-h attraction）：易算，但常缺失束缚激子峰；
- TDDFT：成本低，但结果依赖交换相关核近似；
- BSE：显式处理电子-空穴关联，通常对激子与光学峰位更可靠。

本条目选择最小 BSE-TDA，是为了在较低复杂度下保留“核耦合导致束缚激子与红移”的核心物理。

## R18

`demo.py` 的源码级算法流（9 步）：

1. `BSEConfig` 固定离散规模、QP 色散、核强度与光谱展宽等参数；`build_k_grid` 生成对称 `k` 网格并做合法性检查。  
2. `quasiparticle_transitions` 按 `DeltaE(k)=E_gap+(alpha_c+alpha_v)k^2` 计算独立粒子跃迁能；`dipole_profile` 生成 `d(k)`。  
3. `interaction_kernel` 逐元素构造 `delta_k = k_i-k_j`，再显式计算高斯直接项（负）和交换项（正），得到 `K_{ij}`。  
4. `build_bse_hamiltonian` 把 `diag(DeltaE)` 与 `K` 相加形成实对称 `H_BSE`。  
5. `solve_bse` 调用 `scipy.linalg.eigh`：底层 LAPACK 对实对称矩阵进行特征分解，返回升序本征值 `Omega_S` 与正交本征矢 `A_S`。  
6. `oscillator_strengths` 用 `f_S = |d^T A_S|^2/Nk` 把激子波函数映射到光学振子强度，并通过总和与 independent 权重比较做和规则检查。  
7. `lorentzian_spectrum` 把离散跃迁（BSE 的 `Omega_S` 或 independent 的 `DeltaE_k`）卷积为连续吸收谱，得到两条可比较曲线。  
8. `run_bse` 汇总 QP 边、最低/最亮激子束缚能、峰位红移、哈密顿量对称误差、本征矢归一误差等指标，并构造激子态表格。  
9. `main` 执行 7 条阈值检查，逐条打印 `PASS/FAIL`，最后给出 `Validation: PASS/FAIL`，失败时非零退出。

说明：本实现仅把 `eigh` 用作线性代数本征求解器；电子-空穴核构造、物理量定义、谱函数映射和验收逻辑均在源码中显式实现，不是“一键黑盒 BSE”。
