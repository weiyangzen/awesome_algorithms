# 特征值问题 (Eigenvalue Problem)

- UID: `PHYS-0340`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `356`
- 目标目录: `Algorithms/物理-计算物理-0356-特征值问题_(Eigenvalue_Problem)`

## R01

特征值问题是计算物理中最核心的数值任务之一：给定线性算子（或矩阵）`A`，求解

`A v = lambda v`。

在物理里，`lambda` 常对应可观测量（如能量、振动频率、增长率），`v` 对应模式函数（本征态）。很多 PDE 离散后都会变成大规模矩阵特征值问题。

## R02

本条目的 MVP 选用一维量子简谐振子（dimensionless form）作为可验证场景：

`H psi = E psi`,

`H = -1/2 * d^2/dx^2 + 1/2 * x^2`。

解析本征值为：

`E_n = n + 1/2`, `n = 0,1,2,...`

该模型兼具物理意义与解析对照，适合验证数值本征值算法是否正确。

## R03

离散化策略：

1. 在有限区间 `[-L, L]` 上取均匀网格；
2. 施加 Dirichlet 边界 `psi(-L)=psi(L)=0`；
3. 对二阶导数使用中心差分；
4. 仅保留内部网格点自由度。

得到对称三对角矩阵 `H`，形式为：

- 对角元：`d_i = 1/dx^2 + 1/2 * x_i^2`
- 次对角元：`e_i = -1/(2*dx^2)`

## R04

三对角结构非常关键：

- 存储从 `O(N^2)` 降到 `O(N)`；
- 矩阵向量乘也为 `O(N)`；
- 适合 Krylov 子空间方法（如 Lanczos）求前若干个低能本征值。

因此本 MVP 不构造稠密矩阵，而是以 `(diag, offdiag)` 形式保存算子。

## R05

MVP 采用 Lanczos 方法（带全重正交）求最低 `k` 个本征值：

1. 随机初始化单位向量 `q_1`；
2. 反复执行 `z = H q_j` 并正交化；
3. 构造小型三对角投影矩阵 `T_m`（由 `alpha, beta` 组成）；
4. 求解 `T_m` 的特征值（Ritz 值）近似 `H` 的本征值。

这样把“大问题”降到“小问题”，同时保留主要谱信息。

## R06

Ritz 对恢复：

- 设 `Q_m=[q_1,...,q_m]`，`T_m y = theta y`；
- 则 `theta` 近似 `H` 的本征值；
- 向量 `v = Q_m y` 近似本征向量。

`demo.py` 还会计算每个近似本征对残差：

`r = ||H v - theta v||_2`。

残差越小，说明特征对越可信。

## R07

复杂度（`N` 为矩阵维度，`m` 为 Lanczos 步数）：

- 每次三对角 matvec：`O(N)`；
- `m` 次迭代：`O(mN)`；
- 全重正交约 `O(m^2 N)`；
- 小矩阵 `T_m` 特征分解：`O(m^3)`。

对于“只求前几个本征值”的问题，这比直接稠密 `eigh` 更经济。

## R08

数值稳定性要点：

- 理想 Lanczos 向量应两两正交，但浮点误差会破坏正交；
- 本实现采用全重正交（full re-orthogonalization）抑制“幽灵特征值”；
- 边界区间 `L` 太小会引入截断误差，`L` 太大则在固定网格下增加离散误差；
- 需要在 `L` 与 `dx` 间平衡。

## R09

验证策略分三层：

1. **内部一致性**：检查 Ritz 残差是否足够小；
2. **交叉验证**：与 `scipy.linalg.eigh_tridiagonal` 的结果比较；
3. **物理对照**：与解析 `E_n=n+1/2` 比较低阶本征值误差。

三者同时通过，能有效避免“只看一个指标”的误判。

## R10

MVP 技术栈：

- `numpy`：网格、三对角运算、Lanczos 主流程；
- `scipy.linalg.eigh_tridiagonal`：仅作为参考解交叉校验；
- `pandas`：结果表格化展示。

核心算法不是黑盒调用：Lanczos 迭代、重正交、Ritz 恢复和残差评估均在源码显式实现。

## R11

运行方式：

```bash
cd Algorithms/物理-计算物理-0356-特征值问题_(Eigenvalue_Problem)
uv run python demo.py
```

脚本无交互输入，直接打印参数摘要和结果表，并在通过断言后输出 `Validation: PASS`。

## R12

输出字段说明：

- `n`：能级编号；
- `E_exact`：解析值 `n+1/2`；
- `E_lanczos`：Lanczos 近似值；
- `E_scipy_ref`：SciPy 三对角参考值；
- `|lanczos-exact|`：物理误差（离散+截断+迭代）；
- `|lanczos-scipy|`：同离散模型下与参考求解器差异；
- `residual_2norm`：本征残差。

## R13

`demo.py` 内置验收阈值：

1. `max_residual < 1e-5`；
2. `max(|lanczos-scipy|) < 1e-6`；
3. `max(|lanczos-exact|) < 5e-2`（低能级）；
4. 本征值严格递增。

这些阈值在当前网格与迭代步数下可稳定满足。

## R14

当前 MVP 局限：

- 只处理一维、实对称、定态问题；
- 边界固定为箱体 Dirichlet；
- 未实现移位反幂、隐式重启 Lanczos、块 Lanczos；
- 未处理广义本征值问题 `A v = lambda B v`。

## R15

可扩展方向：

- 改造成稀疏矩阵接口并支持 `scipy.sparse.linalg.eigsh`；
- 加入 shift-invert 提升高能态或内点特征值收敛；
- 支持非均匀网格、有限元离散；
- 扩展到二维/三维薛定谔方程与周期边界条件。

## R16

典型应用：

- 量子力学能谱计算（束缚态）；
- 结构振动模态分析；
- 稳定性分析（线性化算子谱）；
- 材料电子结构近似模型；
- 偏微分方程离散后的本征模求解。

## R17

常见错误与排查：

1. **边界处理错位**：会导致基态能量异常偏低或非物理解；
2. **矩阵符号写反**：可能出现负无穷能谱趋势；
3. **正交化不足**：出现重复/虚假特征值；
4. **只看解析误差不看残差**：可能掩盖迭代未收敛；
5. **网格过粗**：与解析值误差大，但与离散参考值可能仍接近。

## R18

`demo.py` 源码级算法流程（9 步）：

1. `EigenConfig` 定义网格、区间、目标本征值数量和 Lanczos 步数，并在 `validate()` 做边界检查。  
2. `build_harmonic_tridiagonal` 在 `[-L,L]` 上离散哈密顿量，生成三对角 `diag/offdiag`。  
3. `tridiag_matvec` 实现不显式建稠密矩阵的 `y = Hx`，时间与存储都为 `O(N)`。  
4. `lanczos_tridiagonalization` 从随机初始向量出发，迭代构建 Krylov 基 `Q` 与投影系数 `alpha/beta`。  
5. 在同一迭代中执行全重正交，抑制数值正交性丢失，减少幽灵特征值。  
6. `ritz_from_lanczos` 组装小矩阵 `T_m`，用 `numpy.linalg.eigh` 求其特征分解并恢复 Ritz 向量 `V = QY`。  
7. `residual_norms` 逐个计算 `||Hv-lambda v||_2`，形成可审计的收敛质量指标。  
8. `reference_with_scipy` 调用 `scipy.linalg.eigh_tridiagonal` 仅做交叉校验，不替代主算法链路。  
9. `main` 汇总成 `pandas` 表，执行四个断言阈值并输出 `Validation: PASS`。
