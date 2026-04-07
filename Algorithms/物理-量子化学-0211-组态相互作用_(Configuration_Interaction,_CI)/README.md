# 组态相互作用 (Configuration Interaction, CI)

- UID: `PHYS-0210`
- 学科: `物理`
- 分类: `量子化学`
- 源序号: `211`
- 目标目录: `Algorithms/物理-量子化学-0211-组态相互作用_(Configuration_Interaction,_CI)`

## R01

组态相互作用（CI）是量子化学中求解多电子薛定谔方程的经典变分方法：
把波函数展开在一组 Slater 行列式基底上，将连续问题转为有限维本征值问题。

本目录给出一个最小可运行 MVP：
- 使用位串（bitstring）表示电子占据行列式；
- 手写二次量子化算符作用和 CI 哈密顿量矩阵元计算；
- 对比 `FCI`（全组态）和 `CIS`（参考态+单激发）基底能量。

## R02

本实现求解的问题定义：
- 输入：
  - 自旋轨道数 `n_orb`；
  - 电子数 `n_elec`；
  - 一电子积分矩阵 `h1[p,q]`；
  - 反对称化二电子积分 `g2[p,q,r,s] = <pq||rs>`。
- 哈密顿量（第二量子化）：
  - `H = sum_{pq} h1[p,q] a_p^† a_q + (1/4) sum_{pqrs} g2[p,q,r,s] a_p^† a_q^† a_s a_r`。
- 输出：
  - CI 本征值（重点是基态能量）；
  - 基态 CI 系数向量；
  - 数值校验（Hermiticity 残差、本征残差、变分不等式检查）。

脚本内置固定 toy 积分生成，不需要交互输入。

## R03

核心数学关系：

1. CI 波函数展开：
   - `|Psi> = sum_I c_I |Phi_I>`，其中 `|Phi_I>` 为行列式基函数。  
2. 变分原理给出矩阵本征方程：
   - `H c = E c`，`H_{IJ} = <Phi_I|H|Phi_J>`。  
3. 在给定基底下，最低本征值 `E0` 是该子空间内最优变分能量。  
4. 基底越完备，变分能量不升：
   - `E_FCI <= E_CIS <= E_ref`（同一哈密顿量下，子空间嵌套时成立）。

## R04

算法高层流程：
1. 构造 toy 一/二电子积分 `h1`、`g2`。  
2. 枚举 `n_elec` 个电子在 `n_orb` 个轨道上的全部行列式（FCI 基底）。  
3. 从参考占据出发构造 CIS 子空间（参考态 + 所有单激发）。  
4. 对任意行列式对 `(I, J)`，通过 `a^†/a` 的位运算作用计算 `H_{IJ}`。  
5. 分别构建 `H_FCI` 和 `H_CIS`。  
6. 调用 `numpy.linalg.eigh` 对称对角化，获得能谱与系数。  
7. 输出 `E_ref / E_CIS / E_FCI` 及数值残差并做一致性检查。  
8. 打印 FCI 基态中振幅最大的若干行列式，展示组态混合。

## R05

核心数据结构：
- `Determinant = int`：以整数位串表示占据。  
- `all_dets: list[int]`：FCI 基底行列式列表。  
- `cis_dets: list[int]`：CIS 子空间基底列表。  
- `h1: np.ndarray(shape=(n_orb,n_orb))`：一电子积分。  
- `g2: np.ndarray(shape=(n_orb,n_orb,n_orb,n_orb))`：反对称化二电子积分。  
- `h_fci / h_cis: np.ndarray`：CI 哈密顿量矩阵。  
- `e_fci, c_fci`：FCI 本征值与本征向量。

## R06

正确性要点：
- 费米符号：
  - `apply_annihilation/apply_creation` 用“目标轨道以下占据数奇偶”给出反交换符号。  
- 算符顺序：
  - 一电子项按 `a_p^† a_q`，二电子项按 `a_p^† a_q^† a_s a_r` 右乘作用。  
- 矩阵元一致性：
  - 若算符作用后的行列式不等于目标行列式，矩阵元贡献为 0。  
- 变分检查：
  - 输出中显式校验 `E_FCI <= E_CIS` 与 `E_FCI <= E_ref`。  
- 数值检查：
  - `||H-H^T||` 验证对称性；`||Hc-Ec||` 验证本征对精度。

## R07

复杂度分析：

设：
- `M = n_orb`；
- `N = C(M, n_elec)`（FCI 行列式数）。

则：
- 单个矩阵元计算复杂度约为 `O(M^2 + M^4)`，主导项 `O(M^4)`；
- CI 矩阵构建复杂度约为 `O(N^2 * M^4)`；
- 对称本征分解复杂度约为 `O(N^3)`；
- 空间复杂度主要为矩阵存储 `O(N^2)`。

本 demo 选择 `M=6, n_elec=3`，`N=20`，因此可在秒级运行。

## R08

边界与异常处理：
- `build_determinants` 检查 `0 < n_elec <= n_orb`；
- `diagonalize` 检查输入矩阵是有限实数方阵；
- 算符作用若不满足占据约束直接返回 `None`（自动给零贡献）；
- 若模型或矩阵出现非有限值会抛 `ValueError`。

## R09

MVP 取舍说明：
- 只做定长电子数、定轨道数的封闭体系 toy 演示；
- 不接入外部量化化学软件（如 PySCF），积分由脚本内生成；
- 不做 Davidson/稀疏迭代，直接用稠密 `eigh`；
- 用显式循环保留“可审计性”，强调算法机制而非工程极致性能。

## R10

`demo.py` 函数职责：
- `determinant_from_occupied`：把占据轨道列表编码为位串。  
- `occupied_orbitals/format_determinant`：解码并格式化行列式。  
- `apply_annihilation/apply_creation`：带费米符号的算符作用。  
- `one_body_transition/two_body_transition`：一体与二体算符链作用。  
- `ci_matrix_element`：计算 `<I|H|J>`。  
- `build_determinants`：构造 FCI 行列式基底。  
- `build_cis_subspace`：构造 CIS 子空间。  
- `build_ci_hamiltonian`：组装 CI 哈密顿量矩阵。  
- `make_toy_integrals`：生成对称/反对称结构的 toy 积分。  
- `diagonalize`：本征分解。  
- `top_coefficients`：提取主导组态振幅。  
- `main`：组织流程、打印结果与检查。

## R11

运行方式：

```bash
cd Algorithms/物理-量子化学-0211-组态相互作用_(Configuration_Interaction,_CI)
uv run python demo.py
```

或：

```bash
python3 demo.py
```

脚本无需命令行参数，不会请求输入。

## R12

输出字段说明：
- `FCI determinant count`：全组态基底维度。  
- `CIS determinant count`：截断子空间维度。  
- `Reference diagonal energy`：参考行列式对角能量。  
- `CIS ground-state energy`：CIS 子空间基态能量。  
- `FCI ground-state energy`：FCI 基态能量。  
- `Correlation gain (Ref-FCI)`：相关能回收量。  
- `CIS-into-FCI improvement`：CIS 到 FCI 的额外改进。  
- `Hermiticity residual`：`||H-H^T||`。  
- `Eigen residual`：`||Hc-Ec||`。  
- `Top FCI amplitudes`：基态波函数中最主要组态及其系数。

## R13

最小测试建议：
- 正常路径（已内置）：
  - 运行默认 `n_orb=6, n_elec=3`，检查脚本完成并输出三项布尔检查。  
- 参数边界：
  - 调 `build_determinants(n_orb=4, n_elec=0)` 应抛异常。  
- 数值一致性：
  - 人为打乱 `H` 对称性后传入 `diagonalize` 前可观测到较大 `||H-H^T||`。  
- 维度缩放：
  - 手动改 `n_orb`/`n_elec`，观察 `N=C(M,n_elec)` 对运行时间的组合爆炸。

## R14

可调参数（`main` 中）：
- `n_orb`：轨道数；
- `n_elec`：电子数；
- `seed`（在 `make_toy_integrals` 调用处）：积分随机种子；
- `top_k`：输出主导组态个数。

调参建议：
- 教学演示优先：`n_orb <= 8`；
- 若要快：降低 `n_orb` 或远离半充满（减小组合数）；
- 若要看到更明显相关能：可增大二电子积分随机尺度。

## R15

方法对比：
- `FCI`：
  - 优点：给定有限轨道基下“精确”解；
  - 缺点：组合爆炸，无法直接扩展到大体系。  
- `CIS`：
  - 优点：维度小，计算快；
  - 缺点：只含单激发，基态相关能回收有限。  
- `CC`（耦合簇，概念上）：
  - 常见于更大体系，精度/代价折中更好；
  - 但实现更复杂，本条目不展开。

## R16

典型应用场景：
- 量子化学课程中讲解“行列式基 + 变分本征值”流程；
- 新哈密顿量构造代码的基准验证（小体系先做 FCI 对照）；
- 研究中比较截断模型（CIS/CISD/…）的能量误差行为；
- 为后续稀疏迭代或并行 CI 实现提供正确性基线。

## R17

可扩展方向：
- 从 `CIS` 扩展到 `CISD`（加入双激发）；
- 用稀疏矩阵和 Davidson/Lanczos 替代稠密对角化；
- 接入真实分子积分（例如外部 SCF 程序输出）；
- 加入自旋/空间对称性分块，降低有效维度；
- 输出波函数可观测量（密度矩阵、自然轨道占据数等）。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 设定 `n_orb=6, n_elec=3`，调用 `make_toy_integrals` 生成 `h1/g2`。  
2. `build_determinants` 枚举全部 `C(6,3)=20` 个 FCI 行列式位串。  
3. 依据 `h1` 对角元最小占据构造参考态，并由 `build_cis_subspace` 生成 CIS 基底。  
4. `build_ci_hamiltonian` 双循环遍历基底对 `(I,J)`，逐项调用 `ci_matrix_element`。  
5. `ci_matrix_element` 内部通过 `one_body_transition/two_body_transition` 把 `a^†/a` 链作用到 `|J>`，仅当得到 `|I>` 时累加对应积分与符号因子。  
6. 分别得到 `H_FCI` 与 `H_CIS` 后，用 `diagonalize` (`numpy.linalg.eigh`) 求本征谱。  
7. 读取 `E_ref`、`E_CIS`、`E_FCI` 并计算 `||H-H^T||`、`||Hc-Ec||` 两类数值残差。  
8. 打印变分关系检查和 `top_coefficients` 主导组态系数，完成一次可审计 CI 求解。
