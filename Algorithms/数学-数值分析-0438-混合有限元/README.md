# 混合有限元

- UID: `MATH-0438`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `438`
- 目标目录: `Algorithms/数学-数值分析-0438-混合有限元`

## R01

混合有限元（Mixed FEM）的核心是把主变量与通量变量同时作为未知量求解。对椭圆问题而言，它比只求标量场 `u` 的标准位移型方法多了一个优势：可以直接得到满足守恒结构的通量 `q`。

本条目给出一个最小可运行 MVP：
- 问题：二维 Poisson 方程的一阶混合形式；
- 离散：结构网格上的 RT0/P0 风格离散（边法向通量 + 单元常数标量）；
- 实现：纯 `numpy` 可运行，若环境存在 `scipy` 则自动切到稀疏求解。

## R02

连续模型（单位方形区域 `Omega=[0,1]^2`）：

\[
\begin{aligned}
\mathbf q + \nabla u &= 0,\\
\nabla\cdot \mathbf q &= f,
\end{aligned}
\]

边界条件为 Dirichlet：
\[
u|_{\partial\Omega}=g.
\]

`demo.py` 采用解析解用于验证：
\[
u(x,y)=\sin(\pi x)\sin(\pi y),\quad
f(x,y)=-\Delta u=2\pi^2\sin(\pi x)\sin(\pi y).
\]

## R03

混合弱式的典型函数空间是：
- `q in H(div; Omega)`
- `u in L2(Omega)`

最低阶离散通常是：
- 通量空间取 RT0（每单元法向通量常数并在边上法向连续）；
- 标量空间取 P0（单元内常数）。

本 MVP 在规则网格上采用与 RT0/P0 兼容的“边通量 + 单元常数”代数构造，重点是把算法链路跑通并保留守恒结构。

## R04

离散未知量布局：
- 垂直边法向通量 `q_x(i,j)`，数量 `(nx+1)*ny`；
- 水平边法向通量 `q_y(i,j)`，数量 `nx*(ny+1)`；
- 单元标量 `u(i,j)`，数量 `nx*ny`。

总未知量：
\[
N = ((nx+1)ny + nx(ny+1)) + nx\,ny.
\]

## R05

离散方程由两部分组成：

1) 构成方程（边上）
- 内部垂直边：`q + (u_R-u_L)/hx = 0`
- 内部水平边：`q + (u_T-u_B)/hy = 0`
- 边界边：使用半网格差分，把 Dirichlet 值 `g` 注入右端项。

2) 守恒方程（单元内）
\[
\frac{q_{right}-q_{left}}{h_x}+\frac{q_{top}-q_{bottom}}{h_y}=f_{cell}.
\]

这使得每个控制体都满足离散守恒，符合混合方法的物理解释。

## R06

矩阵结构可写为块系统：
\[
\begin{bmatrix}
A & C \\
B & 0
\end{bmatrix}
\begin{bmatrix}
q \\
u
\end{bmatrix}
=
\begin{bmatrix}
b_q \\
f
\end{bmatrix}.
\]

其中：
- `A` 对应边通量自项（本 MVP 为单位对角）；
- `C` 对应离散梯度项；
- `B` 对应离散散度项；
- `b_q` 包含边界值贡献。

`demo.py` 中先以 triplet `(rows, cols, vals)` 装配，再根据可用库选择求解后端。

## R07

边界处理方式：
- 全边界使用 Dirichlet 条件；
- 通过“半网格距离”的离散梯度，把边界值写进通量方程右端；
- 本例解析解在边界上为 0，但代码路径保留了非零 `g` 的通用注入形式。

这样可避免额外引入拉格朗日乘子或罚项，MVP 结构更直接。

## R08

复杂度：
- 装配规模约 `O(N)`；
- 若使用 `scipy.sparse.linalg.spsolve`，通常远优于稠密 `O(N^3)`；
- 若环境无 `scipy`，MVP 自动回退 `numpy.linalg.solve` 稠密解法。

由于回退路径是教学友好的保底方案，默认网格设置较小（`8,12,16`）以保证可运行性与速度。

## R09

伪代码：

```text
input: nx, ny
compute hx, hy
build edge/cell indexing
assemble triplets for constitutive equations (with Dirichlet injection)
assemble triplets for cell divergence equations
if scipy available:
    solve sparse linear system
else:
    convert triplets -> dense matrix and solve by numpy.linalg.solve
split solution into q and u
compare with analytic solution; report L2/relative errors and convergence rates
```

## R10

关键参数（`main` 默认）：
- `grid_levels = [8, 12, 16]`
- 测试方程：`u = sin(pi x) sin(pi y)`
- 指标：
  - `u_L2`, `u_relL2`
  - `flux_L2`, `flux_relL2`
  - 线性系统残差 `res_inf`

可先把 `grid_levels` 改成更细网格（例如 `20,24`）观察误差趋势，再决定是否需要稀疏后端加速。

## R11

默认实验现象：
- `u` 与 `q` 误差随网格细化单调下降；
- 观测到近二阶收敛（该光滑测试解 + 当前离散设置下的数值表现）；
- `res_inf` 接近机器精度量级，说明线性系统求解与装配一致。

示例输出会打印每个网格级别和相邻级别估计阶数。

## R12

`demo.py` 函数分工：
- `assemble_triplets`：装配混合系统的稀疏三元组与右端；
- `solve_linear_system`：优先 `scipy` 稀疏求解，否则 `numpy` 稠密回退；
- `solve_mixed_poisson`：拼装、求解并回传结构化结果；
- `reshape_unknowns`：把解向量重排为 `qx/qy/u` 网格；
- `evaluate_errors`：计算标量与通量误差范数；
- `main`：跑网格收敛试验并打印摘要。

## R13

运行方式：

```bash
cd Algorithms/数学-数值分析-0438-混合有限元
python3 demo.py
```

脚本无交互输入，直接输出：
- 求解后端；
- 每档网格误差；
- 估计收敛阶；
- 最细网格的 sanity-check 指标。

## R14

常见实现错误：
- 边法向方向与散度符号不一致，导致守恒方程符号反转；
- 忘记对边界使用半网格距离系数（`2/h`），误差会明显偏大；
- 组装索引错位（尤其水平边偏移量），会直接破坏矩阵结构；
- 用解析通量对比时混淆 `q=-grad(u)` 与 `grad(u)` 的符号。

本实现逐条显式处理了以上风险点。

## R15

可扩展方向：
- 把规则网格离散推广到三角形/四边形网格上的标准 RT 元；
- 引入非均匀系数 `K(x)`，求解 `K^{-1}q + grad(u)=0`；
- 用混合混合杂交（Hybridized MFEM）降低全局自由度；
- 增加局部后处理，提升标量场重构精度。

## R16

验证策略建议：
- 网格收敛：逐步加密 `nx=ny`，记录 `u_L2/flux_L2`；
- 符号检查：把解析解替换为其他可微解验证鲁棒性；
- 守恒检查：对每个单元计算离散散度残差；
- 边界注入检查：将边界改为非零函数确认 `b_q` 路径正确。

## R17

适用范围与边界：
- 适用：需要可靠通量、局部守恒、教学/原型验证的椭圆问题；
- 不足：本 MVP 仍是规则网格简化版，未覆盖高阶基函数、非结构网格与工业级前处理；
- 性能边界：无 `scipy` 时采用稠密回退，建议使用小到中等网格。

## R18

`demo.py` 的源级算法流（8 步，非黑盒）：
1. `main` 设定网格序列，逐个调用 `solve_mixed_poisson`。  
2. `solve_mixed_poisson` 先进入 `assemble_triplets`，按“边方程 + 单元守恒方程”逐项写入 `(rows, cols, vals)` 与 `rhs`。  
3. 在边界边装配时，使用半网格系数 `2/h` 将 Dirichlet 条件折算进右端，不单独引入边界未知量。  
4. `solve_linear_system` 根据环境分支：若有 SciPy，则构造 CSR 稀疏矩阵并调用 `spsolve`；否则把三元组累加成稠密矩阵并调用 `numpy.linalg.solve`。  
5. 若走 SciPy 路径，`spsolve` 底层会执行稀疏矩阵重排/符号分析、LU 分解、前后代回代三阶段（SuperLU 流程）。  
6. 解向量按索引切分为边通量 `q` 与单元标量 `u`，再通过 `reshape_unknowns` 还原到二维网格形状。  
7. `evaluate_errors` 在单元中心与边中点采样解析解，分别计算 `u` 与 `q` 的绝对/相对 L2 误差，并给出加权通量范数。  
8. `main` 汇总各网格级别误差，计算相邻网格的经验收敛阶，最后打印最细网格的 sanity-check 指标。
