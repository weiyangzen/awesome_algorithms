# 多体微扰理论 (Many-Body Perturbation Theory)

- UID: `PHYS-0213`
- 学科: `物理`
- 分类: `量子多体理论`
- 源序号: `214`
- 目标目录: `Algorithms/物理-量子多体理论-0214-多体微扰理论_(Many-Body_Perturbation_Theory)`

## R01

多体微扰理论（MBPT）的核心目标是：在一个可解参考体系 `H0` 基础上，把相互作用 `V` 作为小参数逐阶展开，从而近似求解真实多体系统 `H = H0 + λV`。

本条目实现一个可运行、可审计的 MBPT(2) 最小原型：
- 用固定粒子数费米 Fock 基构造 `H0` 与 `V`；
- 显式计算 Rayleigh-Schrödinger 二阶展开系数 `E0, E1, E2`；
- 将二阶近似 `E_MBPT2(λ)` 与精确对角化 `E_exact(λ)` 做逐点对照。

## R02

MVP 任务定义：

1. 在 `n_orbitals=6, n_particles=3` 的固定粒子数子空间建立 Slater 行列式基；
2. 构造一体对角哈密顿量 `H0`；
3. 构造并反对称化双体矩阵元 `<pq||rs>`，形成微扰 `V`；
4. 由矩阵元直接计算 `E0, E1, E2`；
5. 在一组 `λ` 上比较 `E_MBPT2(λ)` 与 `E_exact(λ)`；
6. 输出误差表与阈值检查，给出 `Validation: PASS/FAIL`。

## R03

本实现使用的二阶 Rayleigh-Schrödinger 展开：

- `H(λ) = H0 + λV`
- `E_MBPT2(λ) = E0 + λE1 + λ^2E2`

其中（以 `|Φ0>` 为 `H0` 基态参考行列式）：

- `E0 = <Φ0|H0|Φ0>`
- `E1 = <Φ0|V|Φ0>`
- `E2 = Σ_{n!=0} |<Φn|V|Φ0>|^2 / (E0 - En^(0))`

并与精确结果对照：

- `E_exact(λ) = min eig(H0 + λV)`

## R04

模型与边界设定：

- 有限维离散轨道模型（不是连续空间、不是第一性原理材料计算）；
- 粒子数守恒，子空间维度为 `C(6,3)=20`；
- 轨道能量手工给定为单调上升序列；
- 双体作用由平滑衰减基张量构造，再做 `<pq||rs>` 反对称化；
- 主目标是“算法链路可审计与数值行为可验证”，不是追求真实材料参数拟合。

## R05

`demo.py` 默认参数（`MBPTConfig`）：

- `n_orbitals=6`, `n_particles=3`
- `orbital_energies=(-1.55, -1.00, -0.45, 0.20, 0.95, 1.70)`
- `interaction_strength=0.24`
- `interaction_range=1.30`
- `lambda_max=0.55`, `n_lambda=12`
- `quality_lambda=0.35`

这组参数对应弱到中等耦合区间，二阶 MBPT 在小 `λ` 区间内能稳定逼近精确基态能量。

## R06

输入输出约定：

- 输入：无命令行参数、无交互输入，配置固定在 `MBPTConfig`；
- 输出：
1. `E0/E1/E2` 系数表；
2. `lambda` 网格上的 `E_exact`, `E_MBPT2`, `abs_error` 对照表；
3. 张量反对称性、厄米性、分母稳定性与近似误差摘要；
4. 多条阈值检查与最终 `Validation: PASS/FAIL`。

## R07

算法流程（高层）：

1. 校验配置合法性；
2. 构造固定粒子数 Fock 基（bitstring 表示）；
3. 构造一体哈密顿量矩阵 `H0`；
4. 生成并反对称化双体张量 `<pq||rs>`；
5. 通过二次量子化算符作用规则组装 `V` 矩阵；
6. 在参考态上计算 `E0, E1`，并对全部激发态累加得到 `E2`；
7. 对每个 `λ` 做精确对角化得到 `E_exact(λ)`；
8. 计算 `E_MBPT2(λ)` 与误差；
9. 输出结果并执行阈值验收。

## R08

复杂度分析（`D` 为固定粒子数子空间维度，`M` 为轨道数）：

- 基构造：`O(D)`；
- 双体张量构造：`O(M^4)`；
- 组装 `V`：`O(D * M^4)`（本实现主耗时步骤）；
- 单个 `λ` 的精确对角化：`O(D^3)`；
- 全部 `n_lambda` 点：`O(n_lambda * D^3)`。

在默认参数下 `D=20`，整体运行代价很小，适合作为教学与验证型 MVP。

## R09

数值稳定策略：

- 检查 `n_particles < n_orbitals`、`quality_lambda <= lambda_max` 等前置约束；
- 在计算 `E2` 时检查分母 `|E0 - En^(0)|`，避免近简并导致发散；
- 对张量与矩阵分别检查反对称性/厄米性误差；
- 全过程检查输出是否有限值（`finite`）；
- 通过“MBPT vs exact”误差阈值约束近似有效区间。

## R10

MVP 技术栈：

- `numpy`：张量与矩阵计算、bit 运算辅助；
- `scipy.linalg.eigh`：精确对角化获取 `E_exact(λ)`；
- `pandas`：结果表格化与可读输出。

实现没有调用“黑盒多体软件包”；MBPT 系数构造、算符作用、误差评估都在源码中显式展开。

## R11

运行方式：

```bash
cd Algorithms/物理-量子多体理论-0214-多体微扰理论_(Many-Body_Perturbation_Theory)
uv run python demo.py
```

脚本不需要任何交互输入。

## R12

关键输出字段说明：

- `E0/E1/E2`：二阶展开系数；
- `E_exact`：`H0 + λV` 的精确基态能量；
- `E_MBPT2`：二阶微扰近似能量；
- `abs_error`：两者绝对误差；
- `min_abs_denominator`：`E2` 累加中的最小分母绝对值；
- `antisymmetry_error_*`：`<pq||rs>` 反对称约束误差；
- `interaction_matrix_hermitian_error`：多体 `V` 矩阵厄米误差。

## R13

`demo.py` 内置阈值检查：

1. 张量反对称误差（`pq`） `< 1e-12`；
2. 张量反对称误差（`rs`） `< 1e-12`；
3. 张量厄米误差 `< 1e-12`；
4. 相互作用矩阵厄米误差 `< 1e-12`；
5. 二阶分母最小值 `> 0.1`；
6. `E2 < 0`（相关能修正为负）；
7. `lambda <= quality_lambda` 区间的最大误差 `< 5e-3`；
8. `lambda=0` 误差 `< 1e-12`；
9. 所有关键输出有限。

全部通过才输出 `Validation: PASS`。

## R14

当前实现局限：

- 有限离散轨道 toy model，不含连续动量空间积分；
- 只实现到二阶能量修正，未实现高阶图重求和（如 GW/ladder）；
- 未实现频率依赖自能 `Σ(ω)` 与谱函数；
- 参考态固定为 `H0` 基态，未引入自洽参考（如 HF 自洽）；
- 参数是算法演示性质，不直接对应具体材料。

## R15

可扩展方向：

- 从 MBPT(2) 扩展到 MBPT(3)/Goldstone 图自动生成；
- 增加格林函数形式 `G0W0` 自能与准粒子能级修正；
- 支持自旋与动量守恒结构化矩阵元；
- 将有限轨道模型替换为来自 DFT/HF 的输入矩阵元；
- 加入误差外推与收敛分析（基组大小、`λ` 区间）。

## R16

典型应用语境：

- 量子多体课程中展示“参考体系 + 微扰修正”的完整流程；
- 小规模模型中对比“近似（MBPT）vs 精确对角化”的有效区间；
- 作为更高级方法（GW、BSE、coupled-cluster）的前置概念验证；
- 算法工程里用于检查符号、反对称性和费米算符实现正确性。

## R17

与相关方法关系：

- 精确对角化（ED）：小体系精确，但维度指数增长；
- MBPT：在弱耦合区间效率高，可系统加阶，但强耦合可能失效；
- GW/BSE：可视为在 MBPT 图展开基础上的特定重求和与响应框架。

本条目选择 MBPT(2) + ED 对照，是为了在最小复杂度下保留“可解释、可验证、可扩展”的核心价值。

## R18

`demo.py` 的源码级算法流（9 步）：

1. `MBPTConfig` 固定轨道数、粒子数、轨道能量、相互作用强度与 `λ` 扫描范围；`validate_config` 做边界检查。  
2. `build_fock_basis` 用 `itertools.combinations` 构造固定粒子数 Slater 行列式，并用 bitstring 编码。  
3. `build_h0_matrix` 根据占据轨道把一体能量求和，形成对角 `H0`。  
4. `build_antisymmetrized_interaction` 先构造平滑 `raw[p,q,r,s]`，再执行 `V_{pqrs}=V_{rspq}` 对称化与 `<pq||rs>` 反对称化。  
5. `annihilate/create/apply_two_body_operator` 显式实现费米反对易代数的符号规则；`build_interaction_matrix` 逐项组装 `V = (1/4) Σ<pq||rs> a†_p a†_q a_s a_r`。  
6. `compute_mbpt_coefficients` 选取 `H0` 参考态，计算 `E0`、`E1` 并按 `Σ |V_n0|^2/(E0-En)` 累加 `E2`。  
7. `exact_ground_energy` 调用 `scipy.linalg.eigh` 求 `H0+λV` 的最低本征值，得到基准 `E_exact(λ)`。  
8. `run_mbpt` 在全部 `λ` 上计算 `E_MBPT2` 与 `abs_error`，并汇总张量约束误差、分母下界、质量区间最大误差。  
9. `main` 打印系数表/误差表/摘要表，执行 9 条阈值检查，最后给出 `Validation: PASS/FAIL`，失败时非零退出。

说明：本实现只把 `eigh` 用作线性代数本征求解器；多体基构造、费米符号、微扰级数系数与误差评估都在源码中显式实现，不是黑盒 MBPT 调用。
