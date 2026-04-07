# 量子纠缠 (Quantum Entanglement)

- UID: `PHYS-0230`
- 学科: `物理`
- 分类: `量子信息`
- 源序号: `231`
- 目标目录: `Algorithms/物理-量子信息-0231-量子纠缠_(Quantum_Entanglement)`

## R01

问题定义：实现一个可运行、可追踪的两比特量子纠缠最小 MVP。

本条目关注三个核心问题：
- 如何从态的表示（密度矩阵）出发，定量判断是否纠缠；
- 如何通过 CHSH 不等式检测非定域关联；
- 如何用有限采样（shot-based）模拟实验统计噪声。

## R02

理论背景：量子纠缠是复合系统态无法写成各子系统独立态张量积的现象。

本 MVP 采用两个互补判据：
- 纠缠量化：
  - 子系统冯诺依曼熵 `S(ρ_A)`（针对纯态时可作为纠缠熵）；
  - Wootters concurrence `C(ρ)`（可用于两比特混合态）。
- 非定域检测：
  - CHSH 参数 `S_CHSH`；若 `S_CHSH > 2`，违反局域隐变量上界，体现 Bell 非定域性。

## R03

本 MVP 计算任务：
1. 构造代表性两比特态：Bell 态、可分离积态、Werner 混合态；
2. 对每个态计算子系统熵与 concurrence；
3. 计算给定测量轴下的理论 CHSH 值；
4. 基于投影测量概率做 Monte Carlo 抽样，估计实验 CHSH；
5. 通过参数扫描展示“纠缠存在”与“CHSH 违反”阈值不一致现象。

## R04

建模假设（MVP 级别）：
- 仅考虑二维局域希尔伯特空间（两比特，`4x4` 密度矩阵）；
- 器件理想化：不含探测效率失配、暗计数、漂移等实验噪声；
- 测量采用理想投影测量，有限统计误差仅来自抽样次数有限；
- CHSH 设置使用对 `|Φ+>` 近最优的一组固定测量轴。

这些假设适合算法演示与教学验证，不代表完整实验系统建模。

## R05

关键公式：

1. 冯诺依曼熵：
`S(ρ) = -Tr(ρ log2 ρ)`

2. 两比特 concurrence（Wootters）：
- `R = ρ (σ_y ⊗ σ_y) ρ* (σ_y ⊗ σ_y)`
- 设 `λ_i` 为 `R` 特征值平方根按降序排列，则
`C(ρ) = max(0, λ1 - λ2 - λ3 - λ4)`

3. CHSH 参数：
- `E(a,b) = Tr[ρ (σ_a ⊗ σ_b)]`
- `S = |E(a0,b0) + E(a0,b1) + E(a1,b0) - E(a1,b1)|`

4. Werner 态：
`ρ_W(p) = p|Φ+><Φ+| + (1-p)I/4`

## R06

`demo.py` 的执行流程：
1. 定义 Pauli 矩阵与基础态构造函数；
2. 构造 Bell / Product / Werner 状态密度矩阵；
3. 通过偏迹获得子系统态并计算熵；
4. 用 `R` 矩阵特征值实现 concurrence；
5. 用 `σ_a ⊗ σ_b` 期望值计算理论 CHSH；
6. 构造局域投影算符并得到四个联合测量概率；
7. 采样测量结果估计 `E(a,b)` 与实验 CHSH；
8. 输出摘要表 + Werner 参数扫描表，并执行断言检查。

## R07

复杂度分析（记抽样次数为 `K=shots_per_setting`，状态数量为 `M`）：
- 单状态理论量计算（熵、concurrence、CHSH）主要是固定维度矩阵运算，视作 `O(1)`；
- 单状态采样 CHSH 需要 4 组设置，每组 `K` 次抽样，复杂度 `O(K)`；
- 总体复杂度 `O(MK)`，空间复杂度为 `O(K)`（主要是采样中间数组）。

在默认参数 `K=6000, M=4` 下本地运行开销很小。

## R08

数值与统计稳定性策略：
- 对特征值做 `clip`，避免浮点误差导致负零或微小负值；
- 计算熵时忽略 `1e-12` 以下本征值，避免 `log(0)`；
- 固定随机种子 `seed=2026`，保证输出复现；
- 同时输出理论 CHSH 与采样 CHSH，区分“理论结构”与“有限样本波动”。

## R09

适用场景：
- 量子信息课程中两比特纠缠与 Bell 检验演示；
- 研究代码中的快速 sanity check（状态是否纠缠、是否可能违背 CHSH）；
- 教学中解释“纠缠不等于必然 CHSH 违反”。

不适用场景：
- 真实实验数据拟合与误差反演；
- 多体系统/高维系统纠缠分类；
- 完整噪声模型与器件级安全评估。

## R10

脚本内置正确性检查：
1. Bell 态应满足 `entropy_A ≈ 1`、`concurrence ≈ 1`、`CHSH > 2.7`；
2. 积态 `|00>` 应满足 `entropy_A ≈ 0`、`concurrence ≈ 0`、`CHSH <= 2`；
3. `Werner p=0.60` 应“有纠缠但不违反 CHSH”；
4. `Werner p=0.80` 应违反 CHSH。

这些断言保证了“量化指标”和“物理趋势”都正确。

## R11

默认参数（`EntanglementParams`）：
- `shots_per_setting = 6000`
- `seed = 2026`

默认状态集：
- `Bell_PhiPlus`
- `Product_00`
- `Werner_p0.60`
- `Werner_p0.80`

默认 CHSH 轴：
- `a0 = z`, `a1 = x`
- `b0 = (x+z)/sqrt(2)`, `b1 = (-x+z)/sqrt(2)`

## R12

一次实测输出（`uv run python demo.py`）：

主表：
- Bell_PhiPlus: `entropy=1.000000`, `concurrence=1.000000`, `chsh_theory=2.828427`, `chsh_sampled=2.837667`
- Product_00: `entropy=0.000000`, `concurrence=0.000000`, `chsh_theory=1.414214`, `chsh_sampled=1.398333`
- Werner_p0.60: `entropy=1.000000`, `concurrence=0.400000`, `chsh_theory=1.697056`, `chsh_sampled=1.696000`
- Werner_p0.80: `entropy=1.000000`, `concurrence=0.700000`, `chsh_theory=2.262742`, `chsh_sampled=2.258000`

Werner 扫描：
- `p=0.4` 开始出现非零 concurrence（纠缠）
- `p=0.8` 开始 `CHSH > 2`（非定域）

结果与理论预期一致：纠缠与 CHSH 违反阈值不相同。

## R13

正确性边界说明：
- 对混合态，子系统熵不能单独作为纠缠判据；本实现主要依赖 concurrence；
- CHSH 未违反不代表“无纠缠”，仅说明该判据下未见 Bell 非定域；
- 结论限定于两比特系统，不可直接外推到多体量子网络。

## R14

常见失败模式与修复：
- 失败：出现微小负概率或概率和不为 1。
  - 修复：对概率做非负裁剪并归一化。
- 失败：熵计算出现 `nan`。
  - 修复：忽略极小本征值，避免 `log(0)`。
- 失败：采样 CHSH 波动过大。
  - 修复：增大 `shots_per_setting` 或固定随机种子做回归。
- 失败：CHSH 未达到 Bell 态理论上限。
  - 修复：检查测量轴是否归一、符号是否按 CHSH 组合。

## R15

工程化建议：
- 将状态构造、测量、指标计算拆分为独立模块，便于接入实验数据；
- 引入批量参数扫描（多 seed、多 shots）并输出置信区间；
- 若需性能提升，可把重复 `kron` 运算做缓存或向量化；
- 加入单元测试覆盖边界状态（纯态、最大混态、临界 Werner 态）。

## R16

可扩展方向：
- 增加 negativity、PPT 判据等更多纠缠检测器；
- 扩展到三比特 GHZ/W 态与 multipartite witness；
- 加入去极化/退相干通道，研究噪声下的纠缠衰减；
- 把采样过程替换为真实实验计数导入接口。

## R17

本目录交付说明：
- `demo.py`：可直接运行的量子纠缠 MVP（无交互输入）；
- `README.md`：R01-R18 完整填写；
- `meta.json`：保持与任务元信息一致。

运行方式：

```bash
cd Algorithms/物理-量子信息-0231-量子纠缠_(Quantum_Entanglement)
uv run python demo.py
```

## R18

`demo.py` 源码级算法流拆解（8 步，非黑盒）：
1. 先定义 Pauli 矩阵与 `density_from_ket`，把态表示统一为密度矩阵计算框架。
2. 通过 `bell_phi_plus_density`、`product_00_density`、`werner_state_density` 显式生成测试态，不依赖外部量子库黑盒。
3. 在 `partial_trace_two_qubit` 中把 `4x4` 态重排为 `2x2x2x2` 张量并偏迹，得到子系统约化态。
4. `von_neumann_entropy` 对本征值做裁剪和筛选后计算 `-Σ λ log2 λ`，稳定获得子系统熵。
5. `concurrence` 构造 `R = ρ(σy⊗σy)ρ*(σy⊗σy)`，提取特征值平方根并按 Wootters 公式得到纠缠度。
6. `correlator/chsh_value` 基于 `E(a,b)=Tr[ρ(σa⊗σb)]` 计算理论 CHSH，直接对应 Bell 不等式定义。
7. `joint_outcome_probabilities` + `sampled_correlator` 用投影测量概率进行 Monte Carlo 抽样，得到有限样本 CHSH 估计。
8. `main` 汇总多状态结果、执行 Werner 扫描并用断言验证关键物理趋势（Bell 强违反、积态不违反、阈值分离）。
