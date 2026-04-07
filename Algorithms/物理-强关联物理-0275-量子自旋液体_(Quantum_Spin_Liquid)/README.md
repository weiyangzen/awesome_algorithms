# 量子自旋液体 (Quantum Spin Liquid)

- UID: `PHYS-0272`
- 学科: `物理`
- 分类: `强关联物理`
- 源序号: `275`
- 目标目录: `Algorithms/物理-强关联物理-0275-量子自旋液体_(Quantum_Spin_Liquid)`

## R01

量子自旋液体（QSL）是“无传统磁有序但保持强量子纠缠”的自旋相。典型特征包括：
- 在零温下没有 Néel/FM 长程序；
- 存在分数量子数激发（如任意子、spinon）；
- 可能具有拓扑序与基态拓扑简并。

本条目选用可精确求解、又能直接展示拓扑序的 `Z2` 自旋液体原型：`toric code`。

## R02

MVP 目标：在最小 `Lx=Ly=2` 的环面（torus）晶格上，构造 toric code 哈密顿量并做精确对角化，输出：
- 低能谱；
- 基态简并度（理论上 4 重）；
- 第一激发能隙；
- 基态子空间中的 Wilson loop 扇区标签（`Wx, Wy`）。

这对应“量子自旋液体”中最核心、最可计算的拓扑序证据。

## R03

toric code 哈密顿量（自旋放在边上）为：

`H = -Je * sum_s A_s - Jm * sum_p B_p`

其中：
- `A_s = prod_{i in star(s)} sigma_i^x`（顶点星算符）；
- `B_p = prod_{i in plaquette(p)} sigma_i^z`（面算符）。

该模型中所有 `A_s`、`B_p` 两两对易，且 `A_s^2 = B_p^2 = I`，因此可同时对角化。环面边界下基态在热力学意义表现为四重拓扑简并（有限尺寸可精确观测到 4 个最低能级重合）。

## R04

离散化与编码方式：
- 用 `L=2`，共有 `2*L*L=8` 条边，即 `8` 个量子比特；
- Hilbert 维度 `2^8 = 256`；
- 用 `scipy.sparse` 构造 `sigma_x/sigma_z` 的 Kronecker 积算符；
- 通过边索引映射自动生成全部 `A_s` 与 `B_p`。

这种实现避免硬编码大矩阵，便于扩展到更大 `L`。

## R05

`demo.py` 的算法主线：
1. 预构造每个 qubit 上的 `X_i, Z_i` 稀疏矩阵；
2. 由格点连接关系组装 `A_s, B_p`；
3. 组合成总哈密顿量 `H`；
4. 用 `numpy.linalg.eigh` 做精确对角化（`256x256`）；
5. 统计低能级简并与能隙；
6. 在基态子空间内对 Wilson loop 投影并分扇区。

强调：并非“黑箱调库结束”，星/面算符构造、边索引拓扑、子空间投影与扇区判定都在源码中显式实现。

## R06

输入（`ToricCodeConfig`）：
- `L`: 晶格线尺寸（默认 `2`）；
- `Je, Jm`: 电荷/磁通项耦合（默认都为 `1`）；
- `degeneracy_tol`: 基态判据阈值。

输出（终端）：
- 最低若干本征能；
- 基态简并度；
- 第一激发能隙；
- `A_s, B_p` 平均值（应接近 `+1`）；
- `Wx, Wy` 扇区标签与对应能量。

## R07

高层伪代码：

1. 生成边索引函数 `h_edge, v_edge` 与星/面邻接关系。
2. 为每个 qubit 构造全空间 `X_i, Z_i`。
3. 对每个顶点计算 `A_s = Π X_i`。
4. 对每个面计算 `B_p = Π Z_i`。
5. 构造 `H = -Je ΣA_s - Jm ΣB_p`。
6. 对角化 `H` 得到 `E, V`。
7. 由阈值统计基态重数 `g` 与能隙 `Δ = E[g]-E0`。
8. 构造非收缩 Wilson 回路 `Wx, Wy`。
9. 在基态子空间投影后提取 `(Wx, Wy)` 扇区标签。

## R08

复杂度（`Nq=2L^2`, `D=2^Nq`）：
- 构造单比特全空间算符：约 `O(Nq * D^2)`（稀疏 Kronecker 的实际常数较低）；
- 构造 `A_s/B_p` 项：`O(L^2 * D^2)` 量级；
- 精确对角化主导：`O(D^3)`；
- 内存（稠密对角化阶段）约 `O(D^2)`。

MVP 取 `L=2`，`D=256`，可在普通 CPU 上秒级运行。

## R09

数值稳定与鲁棒性：
- 对哈密顿量做 `(H + H^†)/2` 厄米化，抑制浮点误差；
- 用 `degeneracy_tol` 识别“数值近似重根”；
- 扇区判定在基态子空间内做投影，避免直接对简并态做不稳定标签；
- 对关键结果做有限性与物理范围断言。

## R10

正确性检查（脚本内已做）：
- `ground_degeneracy >= 4`（在 `L=2` torus 上应观测到 4 重）；
- `gap > 0`；
- `As_mean`、`Bp_mean` 接近 `+1`；
- Wilson 扇区标签 `Wx, Wy` 落在 `[-1, 1]`（理想为 `±1`）。

## R11

默认参数：
- `L=2`；
- `Je=1.0`，`Jm=1.0`；
- `degeneracy_tol=1e-8`。

调参建议：
- 想看不同耦合比例：改 `Je/Jm`；
- 若数值噪声导致简并判定敏感，可适度放宽到 `1e-7~1e-6`；
- 若要尝试更大系统，建议改用稀疏迭代本征解法（`eigsh`）并利用对称性分块。

## R12

运行方式：

```bash
cd Algorithms/物理-强关联物理-0275-量子自旋液体_(Quantum_Spin_Liquid)
uv run python demo.py
```

脚本无交互输入。

## R13

输出解读：
- `lowest energies` 的前四个值几乎相同，对应拓扑基态流形；
- `ground-state manifold gap` 是到第一激发态的能量差；
- `As_mean/Bp_mean ~ 1` 表示处于无激发真空扇区（无 e/m 任意子）；
- `Wilson sectors` 给出 `(Wx, Wy)`，可区分 4 个拓扑扇区。

## R14

局限性：
- toric code 是“可积理想模型”，不是具体材料的微观哈密顿量；
- `L=2` 仅做教学演示，有限尺寸效应显著；
- 未包含有限温度动力学、无序、长程耦合与实验噪声；
- 未展示真实候选 QSL（如 kagome Heisenberg）的数值困难。

## R15

可扩展方向：
- 加入磁场扰动（如 `-h ΣX` 或 `-h ΣZ`）研究去禁闭相变；
- 切换到 Kitaev honeycomb / kagome J1-J2 等模型；
- 引入张量网络或变分 Monte Carlo 处理更大尺度；
- 计算纠缠熵与拓扑纠缠熵指标。

## R16

适用场景：
- 强关联与拓扑序课程中的“QSL 最小可验证样例”；
- 算法开发前的基准（用于校验算符构造、投影和谱分析流程）；
- 量子信息方向对任意子编码与拓扑简并的入门演示。

不适用场景：
- 面向具体材料定量拟合的研究任务；
- 需要大系统极限精确标度的严肃数值研究。

## R17

参考方向（概念层）：
- A. Y. Kitaev, *Fault-tolerant quantum computation by anyons*, Annals of Physics 303, 2 (2003).
- X.-G. Wen, *Quantum Orders and Symmetric Spin Liquids*, Phys. Rev. B 65, 165113 (2002).
- L. Savary, L. Balents, *Quantum Spin Liquids: a review*, Rep. Prog. Phys. 80, 016502 (2017).

## R18

`demo.py` 源码级算法流（8 步，含第三方函数拆解）：

1. `build_pauli_tables` 先定义单比特 `sx/sz/id2`，再通过 `scipy.sparse.kron` 连乘，显式生成每个边自由度对应的全空间 `X_i/Z_i`。  
2. `star_edges` 与 `plaquette_edges` 根据环面周期边界计算每个星/面的 4 条边索引；这是模型拓扑约束的核心。  
3. `multiply_ops` 将 4 个 `X_i` 或 `Z_i` 连乘，得到 `A_s` 与 `B_p` 算符（而不是调用现成“toric code 黑箱”）。  
4. `build_toric_code_hamiltonian` 把全部项按 `H=-JeΣA_s-JmΣB_p` 相加并做厄米化，形成最终矩阵。  
5. `np.linalg.eigh` 对 `256x256` 哈密顿量求全谱，`solve_toric_code` 提取最低能级、基态重数和能隙。  
6. `build_wilson_loops` 构造两条非收缩回路 `Wx/Wy`（`Z` 算符乘积），用于区分拓扑扇区。  
7. `project_operator_to_subspace` 把 `Wx/Wy` 投影到基态子空间，再对投影矩阵对角化，得到每个基态在 `(Wx, Wy)` 上的标签。  
8. `main` 汇总成 `pandas.DataFrame` 打印，并用断言检查简并、能隙和算符期望值范围，完成一次非交互可复现验证。
