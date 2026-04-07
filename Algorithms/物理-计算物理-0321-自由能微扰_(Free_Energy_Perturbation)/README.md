# 自由能微扰 (Free Energy Perturbation)

- UID: `PHYS-0318`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `321`
- 目标目录: `Algorithms/物理-计算物理-0321-自由能微扰_(Free_Energy_Perturbation)`

## R01

自由能微扰（Free Energy Perturbation, FEP）用于估计两个热力学状态 `A` 与 `B` 的自由能差 `ΔF = F_B - F_A`。

核心思想是 Zwanzig 公式：只要能从状态 `A` 的平衡分布采样，就可以通过指数重加权估计 `A->B` 的自由能差，而不必直接计算配分函数。

## R02

本条目使用 1D 谐振子双态模型作为最小可运行 MVP：
- `U_A(x) = 0.5 * k_A * (x - mu_A)^2 + c_A`
- `U_B(x) = 0.5 * k_B * (x - mu_B)^2 + c_B`
- 逆温 `beta = 1/(k_B T)`

两条势能面足够简单，可给出解析 `ΔF`，便于对 FEP 数值结果做硬性校验；同时保留了“采样 + 重加权 + 不确定度评估”的完整计算物理流程。

## R03

FEP 基本公式（前向）：

`ΔF_{A->B} = -1/beta * ln < exp[-beta * (U_B(x)-U_A(x))] >_A`

其中 `<·>_A` 表示对 `A` 态平衡分布取期望。

反向公式同理：

`ΔF_{B->A} = -1/beta * ln < exp[-beta * (U_A(x)-U_B(x))] >_B`

理论上 `ΔF_{A->B} = -ΔF_{B->A}`，两者的一致性可作为数值诊断。

## R04

对本谐振子模型，解析自由能差可直接写出：

`ΔF_exact = (c_B - c_A) + (1/(2*beta)) * ln(k_B/k_A)`

因此 demo 中可以同时给出：
1. 解析真值 `ΔF_exact`
2. 前向 FEP 估计 `ΔF_fwd`
3. 反向推回的 `ΔF_rev_inferred = -ΔF_{B->A}`

并比较误差与前后向一致性间隙。

## R05

本实现假设与适用边界：
- 系统为经典正则系综（NVT），`beta` 已知且常量。
- 采样链已达到平衡（通过 burn-in 近似处理）。
- 前向/反向相空间有一定重叠，否则 FEP 方差会迅速增大。
- 当前是 1D 势能演示；高维分子体系仅在算法结构上同构，不代表可直接替代生产级分子模拟流程。

## R06

`demo.py` 输入输出约定：
- 输入：脚本内置参数（`k, mu, c, beta`、MCMC 步数、提议步长、bootstrap 次数、随机种子）。
- 输出：
1. `ΔF` 真值与估计值对照表。
2. 前/反向误差、前反向差距、bootstrap 95% 置信区间。
3. 重加权有效样本数 ESS、直方图重叠系数、接受率。
4. 阈值检查与最终 `Validation: PASS/FAIL`。

脚本无需交互输入，`uv run python demo.py` 直接运行。

## R07

算法主流程（高层）：
1. 定义 `A/B` 两个势能面。
2. 用随机游走 Metropolis 在 `A` 与 `B` 各自产生平衡样本。
3. 计算 `ΔU_AB = U_B(x_A)-U_A(x_A)` 与 `ΔU_BA = U_A(x_B)-U_B(x_B)`。
4. 用稳定的 `logmeanexp` 计算前向与反向 FEP。
5. 对前向估计做 bootstrap，得到置信区间。
6. 计算 ESS、分布重叠系数、接受率等诊断量。
7. 基于阈值输出 PASS/FAIL。

## R08

设采样点数为 `N`，bootstrap 重采样次数为 `B`：
- Metropolis 采样：`O(N)`（每步常数代价）。
- `ΔU` 与 FEP 估计：`O(N)`。
- bootstrap：`O(B*N)`（本实现是直接重采样与重算）。
- 额外诊断（ESS、直方图）：`O(N)` 到 `O(N + bins)`。

空间复杂度主要由样本数组决定，为 `O(N)`。

## R09

数值稳定性处理：
- 指数平均采用 `logmeanexp`（先减最大值）避免 `exp` 上溢/下溢。
- ESS 使用对数权重公式计算，避免权重平方时数值爆炸。
- 固定随机种子，保证可复现。
- 同时监控接受率与重叠系数，避免“数值算出来但统计无效”的伪稳定结果。

## R10

MVP 技术栈：
- Python 3
- `numpy`：采样、向量化势能评估、稳定对数运算
- `pandas`：结果表格化输出
- 标准库 `dataclasses`

未使用黑箱自由能软件；FEP 关键步骤在源码中可逐行追踪。

## R11

运行方式：

```bash
cd Algorithms/物理-计算物理-0321-自由能微扰_(Free_Energy_Perturbation)
uv run python demo.py
```

若阈值检查全部通过，会输出 `Validation: PASS`；否则返回非零退出码。

## R12

输出字段说明（摘要表）：
- `DeltaF exact (B-A)`：谐振子解析真值。
- `DeltaF FEP forward A->B`：前向 Zwanzig 估计。
- `DeltaF inferred from reverse B->A`：反向估计取负后的 `B-A` 结果。
- `abs error ...`：相对于真值的绝对误差。
- `forward/reverse consistency gap`：前反向估计差值绝对值。
- `bootstrap 95% CI lower/upper`：前向估计置信区间。
- `ESS forward/reverse`：指数重加权有效样本数。
- `histogram overlap coefficient`：A/B 样本分布重叠度（0 到 1）。
- `acceptance rate ...`：Metropolis 接受率。

## R13

demo 中内置正确性检查：
1. 前向误差 `< 0.10`
2. 反向推回误差 `< 0.10`
3. 前反向一致性差 `< 0.12`
4. bootstrap 95% CI（允许 `±0.01` 容差）覆盖解析真值
5. `ESS forward/reverse > 2000`
6. 分布重叠系数 `> 0.35`
7. 两条链接受率在 `[0.2, 0.8]`

全部满足即判定 PASS。

## R14

当前实现局限：
- 势能仅 1D 谐振子，无法体现复杂分子体系的多模态地形。
- 采样器是基础 Metropolis，未处理自相关时间估计与高级调参。
- 只演示单步 FEP（A 到 B），未使用多 `λ` 窗口桥接困难跃迁。
- 误差评估以 bootstrap 为主，未引入更严格的时间相关误差分析。

## R15

可扩展方向：
- 引入 `λ`-路径分层（多窗口 FEP/TI 混合）降低重叠问题。
- 加入 BAR/MBAR 进行更稳健的双向或多态自由能估计。
- 用 Langevin/HMC 替换基础 Metropolis，提升高维采样效率。
- 把 1D 势能替换为小型粒子体系势（如 Lennard-Jones 对）进行更真实的计算物理演示。

## R16

典型应用场景：
- 分子模拟中的结合自由能、溶剂化自由能变化估计。
- 材料体系中不同构型/相态的相对稳定性比较。
- 计算化学中突变（alchemical transformation）路径上的能量差评估。
- 统计物理教学中“重加权估计”与“相空间重叠”概念演示。

## R17

方法对比（简述）：
- 直接热力学积分（TI）：对导数积分，路径设计灵活，但需要多点积分。
- FEP：单公式直接估计，代码最简，但对相空间重叠敏感。
- BAR/MBAR：统计效率更高、偏差更小，但实现复杂度高于单向 FEP。

本 MVP 选择 FEP，是因为它最能体现“指数重加权 + 采样质量诊断”的核心机制，且规模小、可审计。

## R18

`demo.py` 源码级算法流程（9 步）：
1. `HarmonicState` 定义 `U(x)`，明确两态参数 `k, mu, c` 与统一 `beta`。
2. `metropolis_sample_1d` 用随机游走 Metropolis 生成 `A/B` 两条平衡样本链，并返回接受率。
3. 在 `main` 中计算 `ΔU_AB` 与 `ΔU_BA`，分别作为前向和反向 FEP 输入。
4. `fep_delta_f_from_samples` 调用 `logmeanexp` 计算 `-1/beta * log <exp(-beta ΔU)>`，避免直接黑箱指数平均。
5. `harmonic_exact_delta_f` 给出解析真值，用于对照数值估计误差。
6. `bootstrap_fep` 对前向样本重采样，输出均值和 95% 区间，量化统计不确定度。
7. `effective_sample_size_from_logweights` 基于对数权重计算 ESS，评估重加权退化程度。
8. `histogram_overlap_coefficient` 通过直方图重叠估计两态相空间重叠，辅助解释 FEP 可用性。
9. `main` 汇总指标并执行阈值检查，统一输出 `Validation: PASS/FAIL`。
