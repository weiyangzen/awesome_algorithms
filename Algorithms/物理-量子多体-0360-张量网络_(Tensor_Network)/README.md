# 张量网络 (Tensor Network)

- UID: `PHYS-0342`
- 学科: `物理`
- 分类: `量子多体`
- 源序号: `360`
- 目标目录: `Algorithms/物理-量子多体-0360-张量网络_(Tensor_Network)`

## R01

张量网络（Tensor Network）是一类把指数维多体波函数拆成局域低阶张量并通过网络收缩计算物理量的方法。  
对一维量子链，最常见结构是 MPS（Matrix Product State）：

`|psi(s1,...,sN)> = A[1]^{s1} A[2]^{s2} ... A[N]^{sN}`

其中每个 `A[i]` 是三阶核（边界核退化为二维），中间指标维度称为 bond dimension（`chi`）。

## R02

本条目定位在“量子多体中的最小可运行张量网络闭环”：
1. 先用可验证的物理模型得到基态波函数；
2. 再用顺序 SVD 把全态压缩为 MPS；
3. 最后比较能量、局域观测量和态保真度，验证压缩质量。

这比单纯“调用黑盒张量库”更适合审计和教学。

## R03

`demo.py` 选用的模型是 1D 开边界横场 Ising 链（TFIM）：

`H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i`

默认参数：
- `N=10`（Hilbert 维度 `2^10=1024`）
- `J=1.0`
- `h=1.1`

先用稀疏本征求解器得到精确基态 `|psi0>`，再做张量网络压缩。

## R04

MVP 的算法目标：
- 输入：TFIM 基态向量 `|psi0>`；
- 算法：TT-SVD 风格的顺序 SVD，限制 `chi in {2,4,8,16,32}`；
- 输出：每个 `chi` 对应的
  - MPS 参数量与压缩比；
  - 态误差与保真度；
  - 能量误差；
  - 中心站点 `<Z>` 误差；
  - 中央键纠缠熵。

## R05

核心数学关系：
1. 逐站点展开当前残差张量为矩阵 `M`；
2. 做 `M = U S V^dagger`；
3. 依据 `chi` 和 `cutoff` 截断奇异谱；
4. `U` reshape 成当前 MPS 核，`S V^dagger` 传给下一站点；
5. 链尾 reshape 为最后一个核。

这就是一维张量网络里最基础、最透明的构造路径。

## R06

为什么这属于“量子多体张量网络”而非一般矩阵分解：
- 物理输入是多体哈密顿量基态；
- 目标表示是局域链式网络（MPS）；
- 观测量 `<Z_i>` 用转移矩阵（transfer matrix）收缩得到；
- bond 奇异谱直接对应双分区纠缠结构。

## R07

正确性闭环（脚本内自动检查）：
1. `chi` 增大时，态误差应非增、保真度应非减；
2. 最大 `chi=32` 时应近似恢复精确态（`fidelity -> 1`）；
3. 最大 `chi` 的能量误差和局域 `<Z>` 误差应接近机器精度；
4. `chi=8` 在默认参数下应给出可用近似（避免“压缩无效”）。

## R08

复杂度（设站点数 `N`，最大键维 `chi`）：
- 从全态做 MPS 分解时，前提仍要持有 `2^N` 向量；
- 顺序 SVD 的总体成本受中间展开矩阵尺寸主导，实操上随 `N` 指数增长；
- 但一旦得到 MPS，局域观测量收缩成本可降为多项式（约 `O(N*chi^3)`）。

本 MVP 重点在“表示与收缩机制”，不是超大系统可扩展实现。

## R09

空间复杂度对比：
- 稠密态：`O(2^N)`；
- MPS 参数：`sum_i r_{i-1} * 2 * r_i`（典型近似 `O(N*chi^2)`）；
- 当低纠缠条件成立且 `chi` 不大时，存储优势明显。

## R10

最小依赖栈：
- `numpy`：张量重排、收缩、误差指标；
- `scipy.sparse` + `scipy.sparse.linalg.eigsh`：构造并求解 TFIM 基态；
- `scipy.linalg.svd`：顺序 SVD 分解；
- `pandas`：结果表格输出。

说明：没有调用任何“现成 MPS 黑盒分解 API”，MPS 构造流程在源码中逐步展开。

## R11

运行方式（仓库根目录）：

```bash
uv run python "Algorithms/物理-量子多体-0360-张量网络_(Tensor_Network)/demo.py"
```

或进入目录后运行：

```bash
cd "Algorithms/物理-量子多体-0360-张量网络_(Tensor_Network)"
uv run python demo.py
```

脚本无交互输入。

## R12

输出字段说明：
- `chi_cap`：设定的最大 bond dimension；
- `max_observed_bond`：分解过程中实际出现的最大 bond；
- `mps_params`：MPS 总参数量；
- `compression_ratio(full/mps)`：稠密参数量与 MPS 参数量比值；
- `relative_state_error`：`||psi_mps - psi_exact||_2 / ||psi_exact||_2`（已做全局相位对齐）；
- `fidelity`：`|<psi_exact|psi_mps>|^2`；
- `energy_abs_error`：`|<psi_mps|H|psi_mps> - E0|`；
- `center_site_z_abs_error`：中心站点 `<Z>` 误差；
- `center_bond_entropy`：中心键由奇异值得到的纠缠熵。

## R13

数值稳定与工程处理：
- 基态向量、重构向量都做归一化；
- 对比前做全局相位对齐，避免“同一量子态不同相位”导致伪误差；
- SVD 截断保底保留至少 1 个奇异值，避免秩坍塌；
- 转移矩阵收缩使用复数路径，避免实数化引入偏差。

## R14

当前局限：
- 这是一条“先拿到全态再压缩”的教学路径，不能直接扩展到超大 `N`；
- 未实现 DMRG/TEBD/TDVP 等“在 MPS 空间直接优化或演化”的高级算法；
- 仅覆盖一维开边界、自旋 `1/2`、单点局域观测量示例。

## R15

可扩展方向：
1. 用 MPO 形式直接在 MPS 上算能量，避免回到全态；
2. 引入双站点 DMRG，直接优化基态而非先精确对角化；
3. 增加两点关联函数 `⟨Z_i Z_j⟩` 和纠缠谱分析；
4. 支持周期边界或更一般自旋模型（XXZ、Heisenberg 等）。

## R16

建议测试清单：
1. 调大 `N` 到 12，确认脚本仍可运行并观察压缩趋势变化；
2. 改 `h` 到强场区（如 2.0）与弱场区（如 0.5）对比纠缠熵；
3. 扫描更粗或更细的 `chi_grid`，检查误差-成本曲线是否单调合理；
4. 降低/提高 `svd_cutoff`，观察截断秩和误差变化。

## R17

与相关方法对比：
- 稠密 exact diagonalization：精确但受 `2^N` 限制；
- 本 MVP（exact ground state + MPS 压缩）：强调“张量网络表示与可解释压缩”；
- DMRG：直接在 MPS 流形求基态，更适合大系统；
- PEPS/MERA：面向更高维或多尺度结构，但实现复杂度更高。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `build_tfim_hamiltonian` 用显式 Kronecker 积逐项构造 `H = -J Σ Z_iZ_{i+1} - h Σ X_i`。  
2. `lowest_eigenpair` 调用 `eigsh` 求稀疏哈密顿量最低本征对，得到精确基态 `|psi0>` 与 `E0`。  
3. `state_to_tensor` 把 `|psi0>` reshape 成 `(2,2,...,2)` 多体张量，为顺序 SVD 做准备。  
4. `mps_from_state_by_svd` 从左到右执行“展开矩阵 -> `svd` -> 按 `chi/cutoff` 截断 -> 生成当前核心 -> 传递残差”的循环，最终得到 MPS cores。  
5. `mps_to_state` 把 MPS 重新收缩回向量，并在 `align_phase` 中消除全局相位后计算态误差与保真度。  
6. 用 `recon` 计算 `⟨psi_mps|H|psi_mps⟩`，与 `E0` 比较得到能量误差。  
7. `single_site_transfer` 与 `mps_local_z_expectation` 通过转移矩阵链式收缩计算中心站点 `<Z>`，并与 `exact_local_z_expectation` 对照。  
8. `run_checks` 验证误差/保真度随 `chi` 的单调性与大 `chi` 近精确性，最后打印 `All checks passed.`。  

说明：第三方库只提供线性代数原语（稀疏本征求解、SVD、数组计算）；张量网络分解、截断策略、收缩与物理量评估都在源码中显式实现，不是黑盒一键调用。
