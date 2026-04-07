# 耦合簇理论 (Coupled Cluster Theory)

- UID: `PHYS-0212`
- 学科: `物理`
- 分类: `量子化学`
- 源序号: `213`
- 目标目录: `Algorithms/物理-量子化学-0213-耦合簇理论_(Coupled_Cluster_Theory)`

## R01

耦合簇理论（CC）用指数型波函数近似多电子基态：
`|Psi_CC> = exp(T)|Phi0>`，其中 `|Phi0>` 是参考行列式，`T` 为激发算符和（如 `T1+T2`）。

本目录给出一个可运行、可审计的最小 MVP：
- 在小型固定电子数空间中显式构建哈密顿量；
- 显式构造单/双激发算符矩阵；
- 用投影方程 `⟨Phi_mu|exp(-T)Hexp(T)|Phi0⟩=0` 求振幅；
- 输出 CC 能量并和同一模型的 FCI 基准能量对比。

## R02

问题定义（本实现）：
- 输入：
  - 自旋轨道数 `n_orb=4`；
  - 电子数 `n_elec=2`；
  - 一电子积分 `h1[p,q]`；
  - 反对称化二电子积分 `g2[p,q,r,s]=<pq||rs>`。
- 哈密顿量（二次量子化）：
  - `H = sum_{pq} h1[p,q] a_p^† a_q + 1/4 sum_{pqrs} g2[p,q,r,s] a_p^† a_q^† a_s a_r`。
- 输出：
  - 投影 CC 方程求得的振幅 `t_mu`；
  - `E_CC` 与 `E_FCI`；
  - 残差范数与数值检查项。

脚本不需要命令行参数，也不会请求交互输入。

## R03

核心数学关系：

1. CC 波函数：
   - `|Psi_CC> = exp(T)|Phi0>`，`T = T1 + T2`（本例截断到单、双激发）。
2. 相似变换哈密顿量：
   - `H_bar = exp(-T) H exp(T)`。
3. 振幅方程（投影方程）：
   - `R_mu(t) = <Phi_mu|H_bar|Phi0> = 0`。
4. CC 能量：
   - `E_CC = <Phi0|H_bar|Phi0>`。
5. 基准对照：
   - 同一有限模型下用 FCI 对角化得到 `E_FCI`，用于验证实现行为。

## R04

算法流程（高层）：
1. 生成 toy `h1/g2` 积分。  
2. 枚举固定电子数下全部行列式，建立有限维基底。  
3. 用二次量子化算符显式组装哈密顿量矩阵 `H`。  
4. 从参考行列式枚举 `S+D` 激发，构造每个激发算符矩阵 `tau_mu`。  
5. 构造 `T(t)=sum_mu t_mu tau_mu`，形成 `H_bar=exp(-T)Hexp(T)`。  
6. 以 `R_mu(t)=<Phi_mu|H_bar|Phi0>` 作为非线性方程，调用 `scipy.optimize.root` 求解。  
7. 计算 `E_CC=<Phi0|H_bar|Phi0>`。  
8. 对角化 `H` 得到 `E_FCI` 并比较误差与残差。

## R05

核心数据结构：
- `Determinant = int`：位串编码占据行列式。  
- `basis: list[int]`：固定电子数的全部行列式。  
- `h1: np.ndarray(shape=(n_orb,n_orb))`：一电子积分。  
- `g2: np.ndarray(shape=(n_orb,n_orb,n_orb,n_orb))`：反对称化二电子积分。  
- `h_mat: np.ndarray(shape=(dim,dim))`：行列式基底上的哈密顿量。  
- `Excitation`：单个激发的描述（名称、占据轨道、虚轨道、目标行列式）。  
- `tau_ops: list[np.ndarray]`：所有激发算符矩阵。  
- `t_opt: np.ndarray`：非线性求解得到的最终振幅。

## R06

正确性要点：
- 费米符号：
  - `apply_annihilation/apply_creation` 通过目标轨道以下占据数奇偶计算反交换符号。  
- 算符链顺序：
  - 按 `a^†/a` 的右乘顺序作用，确保矩阵元和激发定义一致。  
- 投影残差定义：
  - 直接读取 `H_bar[idx_exc, idx_ref]`，等价于 `⟨Phi_mu|H_bar|Phi0⟩`。  
- 基准校验：
  - 同模型下输出 `|E_CC - E_FCI|` 和残差范数，检验实现自洽。  
- Hermiticity 检查：
  - 打印 `||H-H^T||`，确认哈密顿量构造无明显对称性错误。

## R07

复杂度分析：

设：
- 轨道数 `M`；
- 电子数 `N`；
- 行列式维度 `D = C(M,N)`；
- 激发数 `K`（本例为 `S+D`）。

则：
- 组装 `H`：约 `O(D^2 * M^4)`（本实现显式遍历算符索引）。
- 单次残差评估：
  - `T` 组装 `O(K*D^2)`；
  - `expm` 与矩阵乘法主导 `O(D^3)`。
- 若非线性求解迭代 `I` 次，总体约 `O(I*D^3 + D^2*M^4)`。

本 demo 参数很小（`D=6`），运行在秒级。

## R08

边界与异常处理：
- `build_determinants` 检查 `0 < n_elec <= n_orb`；
- `apply_excitation` 检查激发阶数匹配（`len(occ)==len(virt)`）；
- 轨道占据不合法时算符作用返回 `None`（贡献自动为 0）；
- `assemble_cluster_operator` 检查振幅长度与算符数一致；
- 非线性求解先试 `hybr`，若残差未足够小则回退 `lm`。

## R09

MVP 取舍：
- 目标是“机制可见”的 CC 教学实现，而非生产级电子结构程序；
- 用矩阵指数直接计算 `exp(±T)`，避免手写大规模 BCH 展开公式；
- 模型规模刻意很小，保证可运行和可审计；
- 不接入外部量化化学库（如 PySCF）与高性能迭代加速（DIIS、张量分块）。

## R10

`demo.py` 函数职责：
- `determinant_from_occupied/occupied_orbitals/format_determinant`：行列式位串编码与展示。  
- `apply_annihilation/apply_creation`：带费米符号的基本算符作用。  
- `one_body_transition/two_body_transition`：一体/二体算符链作用。  
- `ci_matrix_element`：计算行列式对间哈密顿量矩阵元。  
- `build_determinants/build_hamiltonian`：构造固定电子数基底和 `H`。  
- `apply_excitation/enumerate_ccsd_excitations`：定义并枚举 `S+D` 激发。  
- `build_excitation_operator`：把激发提升为矩阵算符 `tau_mu`。  
- `assemble_cluster_operator`：组装 `T(t)`。  
- `cc_energy_and_residual`：计算 `E_CC` 与投影残差。  
- `initial_guess/solve_projected_cc`：初猜与非线性方程求解。  
- `main`：串联流程并打印结果。

## R11

运行方式：

```bash
cd Algorithms/物理-量子化学-0213-耦合簇理论_(Coupled_Cluster_Theory)
uv run python demo.py
```

或：

```bash
python3 demo.py
```

## R12

输出字段说明：
- `determinant_dim`：固定电子数子空间维度。  
- `Reference determinant`：参考行列式。  
- `Excitation count (S+D)`：投影方程维度（激发个数）。  
- `Nonlinear solver success/message`：求解器状态。  
- `Initial/Final residual norm`：残差收敛情况。  
- `CC projected energy`：CC 方程解对应能量。  
- `FCI benchmark energy`：同模型 FCI 基准。  
- `|E_CC - E_FCI|`：与基准偏差。  
- `Hermiticity ||H-H^T||`：哈密顿量对称性残差。  
- `Final amplitudes`：每个激发通道的最终振幅。

## R13

最小测试建议：
- 正常路径：
  - 直接运行脚本，检查 `Final residual norm` 很小且程序正常结束。  
- 参数边界：
  - 调 `build_determinants(4, 0)` 应抛异常。  
- 数值一致性：
  - 观察 `|E_CC - E_FCI|` 是否维持在小量级。  
- 稳健性：
  - 修改 `seed` 复跑，验证求解器回退逻辑可工作。

## R14

可调参数（`main` 中）：
- `n_orb`：轨道数；
- `n_elec`：电子数；
- `seed`：toy 积分随机种子。

调参建议：
- 若要更快演示：保持小维度（如 `n_orb<=6`）；
- 若要更多振幅分量：提高 `n_orb` 并保持 `n_elec` 较小；
- 若收敛变差：减小随机耦合尺度或换 `seed`。

## R15

与常见方法对比：
- FCI：
  - 优点：有限基下精确；
  - 缺点：维度组合爆炸。  
- 传统解析 CCSD 张量方程：
  - 优点：大体系效率高；
  - 缺点：公式复杂，实现门槛高。  
- 本 MVP（矩阵投影 CCSD）：
  - 优点：流程直观、每一步可检查；
  - 缺点：依赖矩阵指数，不适合大体系。

## R16

典型使用场景：
- 量子化学课程中讲解 CC 的指数波函数与投影方程；
- 在极小模型上验证二次量子化算符实现是否正确；
- 作为“从 CI 到 CC”过渡示例，帮助理解两者关系；
- 为后续张量化/高性能 CC 实现提供参考基线。

## R17

可扩展方向：
- 用标准 CCSD 张量方程替代矩阵指数形式；
- 加入 DIIS 加速与阻尼策略；
- 接入真实分子积分输入（例如 SCF 输出）；
- 扩展到更高激发层级（如 CCSDT 的 toy 版本）；
- 增加密度矩阵与期望值计算。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 固定 `n_orb=4, n_elec=2`，通过 `make_toy_integrals` 生成 `h1/g2`。  
2. `build_determinants` 枚举全部固定电子数行列式，`build_hamiltonian` 用 `ci_matrix_element` 组装 `H`。  
3. 通过 `np.linalg.eigh(H)` 得到 `E_FCI` 作为基准。  
4. 根据 `h1` 对角元选参考态 `|Phi0>`，`enumerate_ccsd_excitations` 枚举所有单/双激发并记录目标行列式。  
5. `build_excitation_operator` 把每个激发构造成矩阵 `tau_mu`，`assemble_cluster_operator` 形成 `T(t)=sum t_mu tau_mu`。  
6. `cc_energy_and_residual` 用 `expm(±T)` 构造 `H_bar=exp(-T)Hexp(T)`，读取 `E_CC` 和残差 `R_mu=H_bar[mu,ref]`。  
7. `solve_projected_cc` 以 `initial_guess` 为起点调用 `scipy.optimize.root`（必要时从 `hybr` 回退到 `lm`）求 `R_mu=0`。  
8. 打印收敛状态、`E_CC`/`E_FCI` 偏差、`||H-H^T||` 与最终振幅，完成最小 CC 求解示例。
