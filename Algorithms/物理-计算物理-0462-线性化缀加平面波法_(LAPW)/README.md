# 线性化缀加平面波法 (LAPW)

- UID: `PHYS-0441`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `462`
- 目标目录: `Algorithms/物理-计算物理-0462-线性化缀加平面波法_(LAPW)`

## R01

LAPW（Linearized Augmented Plane Wave）是全电子电子结构计算中的经典基组方法：
- 在晶胞间隙区（interstitial）使用平面波；
- 在 muffin-tin 球区使用原子型径向函数；
- 通过边界值与边界导数连续性把两部分缀加（augmented）起来。

本条目给出一个可运行的教学 MVP，用最小代码复现 LAPW 的核心计算链路，而不是调用黑盒 DFT 软件。

## R02

MVP 采用 1D 教学类比模型（保留 LAPW 关键结构，弱化几何复杂度）：
- 周期晶胞区间 `[-L/2, L/2]`，固定 `k=0`（Gamma 点）；
- `|x| <= R_mt` 视作 muffin-tin 区域，势能为局域吸引势；
- `|x| > R_mt` 为间隙区，势能设为 0；
- 间隙区基函数用 `cos(qx)`；球区基函数用 `A u(r,E_l) + B u_dot(r,E_l)`。

该设计不能替代真实 3D 材料计算，但可忠实体现 LAPW 的“线性化 + 缀加 + 广义本征值问题”主干。

## R03

与标准 LAPW 的对应关系：
- 标准 3D：
  - 间隙区 `exp(i(k+G)·r)`；
  - 球区 `sum_{lm}[A_{lm} u_l(r,E_l) + B_{lm} dot{u}_l(r,E_l)] Y_{lm}(r_hat)`。
- 本 MVP：
  - 用 1D 偶宇称基 `cos(qx)` 对应平面波分量；
  - 用 `u(r,E_l)` 与 `u_dot(r,E_l)=∂u/∂E|_{E_l}` 对应线性化径向基；
  - 用边界匹配求每个 `q` 的 `A_q, B_q`。

## R04

球区径向函数通过常微分方程获得：

`-1/2 * u''(r) + V_mt(r) u(r) = E u(r)`

取偶宇称初值 `u(0)=1, u'(0)=0`，在 `r ∈ [0, R_mt]` 上积分。线性化导数使用中心差分：

`u_dot(r) ≈ [u(r, E_l+ΔE) - u(r, E_l-ΔE)] / (2ΔE)`。

代码中分别对应 `solve_radial_profile` 与 `solve_linearized_radial_set`。

## R05

边界匹配条件（LAPW 关键步骤）在 `r=R_mt` 处写为：
- 函数连续：`A u + B u_dot = cos(qR_mt)`
- 导数连续：`A u' + B u_dot' = -q sin(qR_mt)`

由 2x2 线性方程直接解得 `A, B`。代码对应 `boundary_match_coefficients`。

## R06

得到分片基函数后，组装广义本征问题：

`H c = E S c`

其中：
- `S_ij = ∫ phi_i phi_j dx`
- `H_ij = ∫ [0.5 phi_i' phi_j' + V(x) phi_i phi_j] dx`

这一步由 `assemble_hs_matrices` 完成，再由 `scipy.linalg.eigh(H, S)` 求能带本征值。

## R07

程序输出是一组基组截断 `n_max` 的收敛表：
- `E0_lapw`, `E1_lapw`：最低两个本征能；
- `E0_ref`：独立参考解（周期有限差分）；
- `abs_err_ref`：基态能误差；
- `S_min_eig`：重叠矩阵最小特征值；
- `bc_val_max`, `bc_der_max`：边界连续性误差；
- `residual_max`：广义本征残差。

## R08

复杂度（设基函数数为 `M = n_max + 1`，积分网格点数为 `N_x`）：
- 基函数构造：`O(M * N_x)`
- 矩阵组装：`O(M^2 * N_x)`
- 广义本征求解：`O(M^3)`

在当前教学参数下（`M<=7`），主耗时来自数值积分与参考解求解，整体可在秒级完成。

## R09

数值稳定性措施：
- 对边界匹配矩阵检查条件数（过大则报错）；
- 对重叠矩阵检查最小特征值（防止近奇异基组）；
- 使用高精度 ODE 容差（`rtol=1e-9, atol=1e-11`）；
- 最终用边界误差与本征残差做双重验证，避免“能量看似合理但函数不连续”。

## R10

MVP 技术栈：
- `numpy`：网格、插值、矩阵和向量运算；
- `scipy.integrate.solve_ivp`：径向方程积分；
- `scipy.linalg.eigh`：广义本征值求解；
- `pandas`：结果表格化输出。

未使用任何 DFT/LAPW 黑盒库，关键算法步骤都在源码中可追踪。

## R11

运行方式：

```bash
cd Algorithms/物理-计算物理-0462-线性化缀加平面波法_(LAPW)
uv run python demo.py
```

脚本无交互输入，运行结束会打印阈值检查并给出 `Validation: PASS/FAIL`。

## R12

关键可调参数（`LAPWConfig`）：
- `cell_length`, `muffin_tin_radius`：晶胞尺度与球区半径；
- `linearization_energy`, `energy_step`：线性化点与差分步长；
- `potential_depth`, `potential_sigma`：球区势能形状；
- `integration_points`, `radial_points`：积分与径向离散精度；
- `n_max_list`：基组截断扫描列表。

这些参数既影响物理近似，也影响数值稳定性与收敛速度。

## R13

demo 内置验证条件：
1. `S_min_eig > overlap_tol`（重叠矩阵正定）
2. `bc_val_max < 1e-6`（边界函数连续）
3. `bc_der_max < 1e-6`（边界导数连续）
4. `residual_max < 1e-6`（广义本征残差）
5. `best_abs_err_ref < 0.15`（与独立参考解一致）

全部满足判定 `Validation: PASS`，否则退出码为 1。

## R14

当前实现局限：
- 1D 教学类比，不含真实 3D 球谐 `Y_lm` 与多原子结构因子；
- 只做固定势单次本征求解，不含 DFT 自洽循环（SCF）；
- 未实现多 `l` 通道与 local orbital；
- 参考解使用有限差分，不是材料数据库基准。

因此它是“算法结构演示器”，不是生产级材料计算器。

## R15

可扩展方向：
- 升级到 3D `k+G` 与 `l,m` 展开，支持多原子晶胞；
- 加入 SCF：电荷密度、Hartree/xc 势、混合迭代；
- 增加多线性化能量与 local orbital，改善高能态描述；
- 引入自旋极化与自旋轨道耦合；
- 接入真实晶体势与对称性约简。

## R16

典型应用语境：
- 讲解 LAPW 核心思想（分区基组 + 边界匹配 + 广义本征）；
- 作为完整电子结构代码开发前的最小验证原型；
- 评估线性化参数、匹配精度和基组条件数对能量的影响；
- 用于数值分析教学（BVP、插值、广义特征值问题）。

## R17

与近似方法的关系（简述）：
- 赝势平面波法：实现简洁、适合大体系，但核心电子通常被冻结；
- LAPW：全电子精度高、体系普适性强，但实现复杂、计算成本高；
- 本条目所选路线：保留 LAPW 的核心数学结构，用最小 1D 原型降低实现复杂度。

## R18

`demo.py` 源码级算法流程（9 步）：
1. `main` 构造 `LAPWConfig` 并调用 `run_lapw_study`。
2. `run_lapw_study` 先 `validate_config`，再调用 `solve_linearized_radial_set` 计算 `u`/`u_dot`。
3. `solve_linearized_radial_set` 内部三次调用 `solve_radial_profile(E_l, E_l±ΔE)`，由中心差分得到 `u_dot` 与 `u_dot'`。
4. 对每个 `n_max`，`build_piecewise_basis` 生成一组 `q=2πn/L` 基函数，并对每个 `q` 调 `boundary_match_coefficients` 解 `A_q,B_q`。
5. `build_piecewise_basis` 按区域拼接函数：间隙区用 `cos(qx)`，球区用 `A_q u + B_q u_dot`，同时构造导数并统计边界误差。
6. `assemble_hs_matrices` 在实空间网格上积分得到 `H,S`。
7. `solve_generalized_eigenproblem` 先检查 `S` 的最小特征值，再求解 `Hc=ESc`，`generalized_residual_norm` 计算残差。
8. `finite_difference_reference_energy` 给出独立周期有限差分基态参考能量，形成 `abs_err_ref`。
9. `main` 汇总 `n_max` 收敛表、最佳配置摘要，并执行 5 条阈值检查输出 `Validation: PASS/FAIL`。
