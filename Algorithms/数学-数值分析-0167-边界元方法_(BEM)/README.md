# 边界元方法 (BEM)

- UID: `MATH-0167`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `167`
- 目标目录: `Algorithms/数学-数值分析-0167-边界元方法_(BEM)`

## R01

边界元方法（Boundary Element Method, BEM）把偏微分方程从“区域离散”改为“边界离散”。
对于线性椭圆型问题（如 Laplace 方程），它利用基本解把域内问题写成边界积分方程，因此未知量只定义在边界上。

本目录的 MVP 目标是：
- 求解二维 Laplace 内问题 `Δu = 0`（单位圆内部）；
- 已知 Dirichlet 边界值 `u|Γ`，反求边界通量 `q = ∂u/∂n`；
- 再用边界积分表示公式回代域内若干点，验证数值精度。

## R02

本文选择可解析验证的模型：
- 计算域：单位圆内部 `Ω = {(x,y): x^2 + y^2 < 1}`；
- 方程：`Δu = 0`；
- 边界条件：`u(x,y) = x^2 - y^2`（边界 `Γ` 上）；
- 对应解析解（域内同样成立）：`u(x,y) = x^2 - y^2`。

该解的边界法向导数也可显式写出：
- `∇u = (2x, -2y)`；
- `q_exact = ∇u · n`。

因此我们可以同时检查：
- 边界通量恢复误差；
- 域内函数值误差。

## R03

二维 Laplace 基本解取：

`G(P,Q) = -(1/(2π)) ln |P-Q|`。

对内域 Dirichlet 问题，在边界配点 `P_i` 上使用直接边界积分方程：

`0.5 * u(P_i) + ∫_Γ u(Q) * ∂G(P_i,Q)/∂n_Q dΓ = ∫_Γ q(Q) * G(P_i,Q) dΓ`。

离散后得到线性系统：

`G_mat * q = (0.5 I + H_mat) * u`。

其中：
- `u` 为已知边界值向量；
- `q` 为待求边界通量向量。

## R04

离散策略（MVP）为常数单元 + 中点配点：
- 用 `N` 条线段逼近单位圆边界；
- 每条线段中点作为配点/源点评估位置；
- 每单元上 `u,q` 取常数近似；
- 积分采用“核函数在中点取值 × 单元长度”的低阶求积。

这是一套小而真实的 BEM 教学级实现，重点是流程可审计，而不是追求高阶几何与高阶积分。

## R05

奇异项处理：
- `G` 核在同一单元上有对数奇异；
- 本实现对 `G_ii = ∫_Γi G dΓ` 使用解析积分近似：
  `G_ii = L_i/(2π) * (1 - ln(L_i/2))`；
- 双层核 `H` 的对角主值项在本实现中置为 `0`，并在方程左侧显式加入 `0.5 I`。

这是常见的低阶 collocation BEM 处理方式。

## R06

数值流程分成两段：
1. 边界求解段：组装稠密矩阵 `G_mat, H_mat`，求解 `q`。
2. 域内后处理段：对每个内点 `X` 用表示公式
   `u(X) = ∫_Γ G(X,Q) q(Q)dΓ - ∫_Γ ∂G(X,Q)/∂n_Q u(Q)dΓ`
   计算近似值。

脚本会输出这两段的误差指标，确保不是“只解线性方程但不验证”。

## R07

复杂度（`N` 为边界单元数，`M` 为域内评估点数）：
- 组装边界算子：`O(N^2)`；
- 解稠密线性系统：`O(N^3)`；
- 内点后处理：`O(MN)`。

空间复杂度：
- 主要是 `N x N` 稠密矩阵，`O(N^2)`。

本 MVP 默认 `N=80`，在普通笔记本上可快速运行。

## R08

`demo.py` 关键数据结构：
- `BoundaryMesh`：
  - `vertices` 边界顶点；
  - `collocation` 单元中点；
  - `normals` 外法向；
  - `lengths` 单元长度。
- `BEMResult`：
  - 边界 `u/q` 数值与解析；
  - 边界通量相对 `L2` 误差；
  - 域内采样点数值与解析；
  - 域内 `Linf` 绝对误差。

## R09

伪代码：

```text
build_unit_circle_mesh(N)
u_bnd <- x^2 - y^2 on boundary collocation points
assemble G_mat, H_mat
rhs <- (0.5*I + H_mat) * u_bnd
solve q_bnd from G_mat * q_bnd = rhs

for each interior point X:
    evaluate u_bem(X) = Σ_j G(X,Q_j)*q_j*L_j - Σ_j dGdn(X,Q_j)*u_j*L_j

compute:
  boundary_flux_rel_l2 = ||q_bnd - q_exact||_2 / ||q_exact||_2
  interior_abs_linf = max |u_bem(X_k) - u_exact(X_k)|
print metrics and enforce error gates
```

## R10

默认参数（`run_demo`）：
- `num_elements = 80`（边界离散数）；
- 域内验证点 5 个（固定坐标，均满足 `r<1`）。

该规模下可在保持代码简洁的同时，得到稳定且可接受的误差水平。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0167-边界元方法_(BEM)
uv run python demo.py
```

脚本无交互输入，单次运行直接输出误差摘要和内点对比表。

## R12

输出字段说明：
- `elements`：边界单元数；
- `boundary flux rel-L2 err`：边界通量恢复相对误差；
- `interior abs-Linf err`：域内采样点最大绝对误差；
- 内点逐行对比：`(x, y, u_bem, u_exact, abs_err)`。

脚本还设置了最小质量闸门：
- 若边界通量误差或域内误差过大，则抛异常，避免“静默失败”。

## R13

函数职责划分：
- `fundamental_solution`：计算 `G(P,Q)`；
- `dG_dn_source`：计算源点法向导数核；
- `build_unit_circle_mesh`：构建边界几何与法向；
- `assemble_boundary_operators`：组装 `G_mat/H_mat`；
- `solve_dirichlet_bem`：边界方程求解 `q`；
- `evaluate_interior_u`：域内后处理；
- `run_demo`：组织一轮完整实验；
- `main`：打印结果并执行误差闸门。

## R14

工程上常见错误与规避：
- 核函数符号写反，导致整体解偏差很大；
- 忘记 `0.5 I` 跳项，只保留 `H`，会使边界方程错误；
- 把法向导数当成对场点求导而不是源点求导；
- 自项奇异积分直接按普通中点公式处理，数值会明显劣化；
- 把边界点拿去做内点后处理，触发奇异核。

本实现分别通过公式固定、自项解析式和内点半径校验规避上述问题。

## R15

最小验证方案：
- 边界验证：比较 `q_bnd` 与解析 `q_exact` 的相对 `L2` 误差；
- 域内验证：比较多个内点 `u_bem` 与解析 `u_exact` 的 `Linf` 误差；
- 运行一致性：固定参数下，多次运行结果应稳定（确定性算法，无随机性）。

这是一个“可重复、可定位问题”的最小验证闭环。

## R16

适用范围与局限：
- 适用：线性椭圆型 PDE、均匀介质、边界几何维度远小于域维度时；
- 优点：只离散边界，几何数据量通常更小；
- 局限：得到的是稠密矩阵，规模上去后 `O(N^3)` 成本明显；
- 本 MVP 为低阶常数单元，精度与收敛阶有限，主要用于教学与流程验证。

## R17

可扩展方向：
- 升级到线性/二次边界单元，提高几何和场变量近似精度；
- 使用高阶/自适应奇异积分技术提升近场精度；
- 引入快速算法（FMM、H-matrix）降低大规模稠密系统成本；
- 扩展到 Helmholtz、弹性力学等更复杂核函数；
- 增加边界条件混合（Dirichlet + Neumann）与多连通区域处理。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 调用 `run_demo(num_elements=80)`，固定一个可复现的最小实验规模。
2. `run_demo` 先执行 `build_unit_circle_mesh`：按角度均分生成圆周顶点，构造中点配点、单元长度和外法向。
3. `solve_dirichlet_bem` 在边界中点上计算已知 `u_bnd = x^2 - y^2`，并进入算子组装。
4. `assemble_boundary_operators` 双重循环构建 `G_mat/H_mat`：非对角项用核函数中点求积，对角 `G_ii` 用对数奇异积分解析式，`H_ii` 取主值 0。
5. 组装右端 `rhs = (0.5 I + H_mat) @ u_bnd`，再用 `np.linalg.solve` 解线性系统 `G_mat q = rhs`，得到边界通量 `q_bnd`。
6. `run_demo` 调用 `evaluate_interior_u`，对每个内点逐单元累计两类边界积分，按表示公式得到 `u_in_bem`。
7. 计算两类误差：边界通量相对 `L2`（对比 `q_exact`）与域内采样点 `Linf`（对比解析解 `x^2-y^2`）。
8. `main` 打印误差摘要和逐点对照，并执行阈值闸门；若误差超限则抛异常，保证 MVP 结果可审计。
