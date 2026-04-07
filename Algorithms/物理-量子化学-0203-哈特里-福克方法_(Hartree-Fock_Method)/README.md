# 哈特里-福克方法 (Hartree-Fock Method)

- UID: `PHYS-0202`
- 学科: `物理`
- 分类: `量子化学`
- 源序号: `203`
- 目标目录: `Algorithms/物理-量子化学-0203-哈特里-福克方法_(Hartree-Fock_Method)`

## R01

哈特里-福克（Hartree-Fock, HF）是多电子体系的平均场近似方法：

- 把 `N` 电子波函数近似为一个 Slater 行列式；
- 每个电子在“其余电子平均库仑场 + 交换效应”中运动；
- 通过自洽场（SCF）迭代求解轨道，使总能量达到驻值。

对闭壳层分子，常用受限哈特里-福克（RHF）Roothaan-Hall 形式：

`F C = S C epsilon`

其中 `F` 为 Fock 矩阵，`S` 为重叠矩阵，`C` 为分子轨道系数，`epsilon` 为轨道能级。

## R02

典型应用场景：

- 量子化学入门与电子结构教学（从头推导到可运行代码）；
- 作为后 HF 方法（MP2、CI、CC）与 DFT 的起点轨道；
- 在小基组下快速估计分子能量与轨道信息；
- 为更高精度程序提供初猜密度矩阵。

## R03

RHF 的核心矩阵公式（原子轨道基组）如下：

1. 密度矩阵（闭壳层）：
`P_{mu nu} = 2 * sum_{i in occ} C_{mu i} C_{nu i}`

2. Fock 矩阵：
`F_{mu nu} = H_{mu nu} + sum_{lambda sigma} P_{lambda sigma} [(mu nu|lambda sigma) - 1/2 (mu lambda|nu sigma)]`

3. 电子能：
`E_elec = 1/2 * sum_{mu nu} P_{mu nu} (H_{mu nu} + F_{mu nu})`

4. 总能：
`E_tot = E_elec + E_nuc`

这里 `(mu nu|lambda sigma)` 是双电子积分，`E_nuc` 为核-核排斥能。

## R04

直观理解：

- 给定一组轨道后，可构造电子密度 `P`；
- 给定 `P` 后，可构造新的平均场 `F`；
- 再解 `F C = S C epsilon` 得到新的轨道 `C`；
- 重复该闭环直到“输入密度 = 输出密度”（即自洽）。

HF 的非线性本质就在于：`F` 依赖 `P`，而 `P` 又由 `F` 的本征向量决定。

## R05

正确性与物理约束检查要点：

1. `C^T S C = I`（分子轨道在 AO 度量下正交归一）。
2. `Tr(P S) = N_e`（电子数守恒）。
3. 收敛点满足 Brillouin 驻值条件的矩阵形式：`F P S - S P F ≈ 0`。
4. SCF 迭代应使 `dE` 和 `dP` 同时收敛到阈值下。

`demo.py` 会输出上述诊断中的关键量并做断言。

## R06

设基函数数目为 `K`：

- 构造 Fock（朴素四指标求和）时间复杂度约 `O(K^4)`；
- 对角化 `K x K` 对称矩阵复杂度约 `O(K^3)`；
- 存储全双电子积分张量约 `O(K^4)` 空间。

本 MVP 使用 `K=2` 的 H2 最小基，计算量极小，重点在算法流程可审计。

## R07

标准 RHF-SCF 步骤：

1. 准备 `S, H, (mu nu|lambda sigma), E_nuc, N_e`；
2. 初始化密度矩阵 `P=0`；
3. 由 `P` 构造 `F`；
4. 通过对称正交化把广义本征问题转成标准本征问题并求解；
5. 取占据轨道重建 `P_new`；
6. 计算 `E_tot`、`dE`、`dP`；
7. 未收敛则 `P <- P_new` 循环；
8. 收敛后输出总能、轨道能、电子数与驻值残差。

## R08

`demo.py` 的 MVP 设计：

- 体系：`H2`，键长 `R=1.4 bohr`，闭壳层 2 电子；
- 基组：教学用最小 2 基函数（固定积分，不现场算高斯积分）；
- 工具栈：`numpy`（矩阵运算、特征分解、einsum）；
- 目标：用最少代码展示 HF 的完整自洽闭环，而不是调用一站式量化化学库黑盒。

## R09

`demo.py` 主要函数接口：

- `build_h2_sto3g_problem() -> dict`
- `set_eri_with_permutational_symmetry(eri, mu, nu, lam, sig, value) -> None`
- `symmetric_orthogonalization(overlap, tol=1e-12) -> np.ndarray`
- `build_fock(density, h_core, eri) -> np.ndarray`
- `solve_roothaan(fock, overlap, n_occ) -> (eps, coeff, density)`
- `electronic_energy(density, h_core, fock) -> float`
- `scf_rhf(...) -> dict`
- `commutator_residual(fock, density, overlap) -> float`

## R10

测试策略（脚本内自动完成）：

- 收敛性：`converged == True`；
- 能量范围：`E_total` 落在 H2 RHF 合理区间（本例断言 `-1.2 < E < -1.0 Ha`）；
- 电子数：`|Tr(P S)-N_e| < 1e-8`；
- 轨道正交性：`||C^T S C - I||_F < 1e-8`；
- 输出每次迭代 `E_total / dE / dP` 便于审计。

## R11

边界条件与异常处理：

- 奇电子体系会触发 `ValueError`（本 MVP 只实现 RHF 闭壳层）；
- 若重叠矩阵 `S` 接近奇异（最小本征值过小）触发 `ValueError`；
- 若超过最大迭代次数仍未满足阈值，`converged=False`，最终断言会失败。

## R12

与相关方法关系：

- HF 是“单行列式 + 平均场”基线；
- 后 HF（MP2/CC/CI）在 HF 轨道基础上补电子关联；
- Kohn-Sham DFT 的迭代骨架与 HF 类似，但交换相关项来源不同；
- UHF/ROHF 是对开壳层体系的扩展。

## R13

本条目示例参数：

- 分子：`H2`，`R=1.4 bohr`；
- `E_nuc = 1 / R = 0.714286 Ha`；
- 电子数：`N_e = 2`，占据轨道数 `n_occ = 1`；
- 收敛阈值：`e_tol=1e-12`，`p_tol=1e-10`，`max_iter=50`。

## R14

工程实现注意点：

- 广义本征问题不要直接用 `inv(S)F`，数值更稳的做法是对称正交化：`X=S^{-1/2}`；
- 双电子积分通过 8 重置换对称填充，避免手抄时漏项；
- `einsum` 下标应和公式一一对应，建议用可读性更高的显式字符串而不是压缩写法；
- 能量计算应使用 `E = 1/2 * sum P(H+F)`，避免双计数。

## R15

结果解释（本 MVP 典型输出）：

- 迭代约 2 步收敛（该对称小体系非常快）；
- 总能量约 `-1.1167 Ha`，与教材级 H2 最小基 RHF 值一致；
- `Tr(P S)` 接近 2，说明电子数保持正确；
- `||F P S - S P F||_F` 很小，说明达到 SCF 驻值条件。

## R16

可扩展方向：

- 加入阻尼、DIIS 提升困难体系收敛稳定性；
- 从固定积分升级为现场计算高斯基积分；
- 支持 UHF/ROHF 与开壳层；
- 增加基组规模与分子几何扫描，输出势能曲线。

## R17

本条目交付内容：

- `README.md`：完成 R01-R18，覆盖定义、公式、复杂度、实现与验证；
- `demo.py`：可直接运行的 RHF 最小实现（无需交互输入）；
- `meta.json`：保持任务元信息一致。

运行方式：

```bash
cd Algorithms/物理-量子化学-0203-哈特里-福克方法_(Hartree-Fock_Method)
uv run python demo.py
```

## R18

源码级算法流程拆解（对应 `demo.py`，8 步）：

1. **构建问题数据**
   `build_h2_sto3g_problem` 给出 `S/H/ERI/E_nuc/N_e`，并用 `set_eri_with_permutational_symmetry` 按 8 重对称补全双电子积分张量。

2. **初始化 SCF 状态**
   `scf_rhf` 设置 `P=0`、迭代计数器、收敛阈值和历史缓存，准备进入自洽循环。

3. **由密度构造 Fock**
   `build_fock` 使用两次 `einsum` 显式形成库仑项 `J` 与交换项 `K`，得到 `F = H + J - 1/2 K`。

4. **对称正交化降维**
   `symmetric_orthogonalization` 对 `S` 做 `numpy.linalg.eigh`，构造 `X = S^{-1/2}`，把广义本征问题变成标准对称本征问题。

5. **解 Roothaan 方程并回到 AO 基**
   `solve_roothaan` 对 `F' = X^T F X` 再做一次 `numpy.linalg.eigh` 得到 `eps, C'`，随后 `C = X C'`，取占据轨道重建 `P_new = 2 C_occ C_occ^T`。

6. **计算能量与收敛指标**
   用 `electronic_energy` 计算 `E_elec`，再加 `E_nuc` 得 `E_tot`；并计算 `dE` 与 `dP`，写入迭代日志。

7. **判收敛或继续迭代**
   若 `dE < e_tol` 且 `dP < p_tol`，置 `converged=True` 并退出；否则 `P <- P_new` 继续下一轮。

8. **后处理与物理一致性验证**
   `main` 计算 `Tr(P S)`、`||C^T S C-I||_F`、`||F P S - S P F||_F` 并断言，最后输出 `All checks passed.`。

说明：这里的 `numpy.linalg.eigh` 仅用于对称本征分解基元；SCF 主逻辑（Fock 组装、正交化变换、密度更新、能量与残差判据）均在源码中显式实现。
