# Galerkin方法

- UID: `MATH-0434`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `434`
- 目标目录: `Algorithms/数学-数值分析-0434-Galerkin方法`

## R01

Galerkin 方法是一类“加权残差法”：

- 先把连续问题的解限制在一个有限维试探空间 `V_N` 中；
- 再令残差对所有测试函数都正交，即 `r ⟂ V_N`；
- 由此把微分方程转成线性代数方程组。

它是有限元、谱方法、DG 等大量数值 PDE 方法的共同核心。

## R02

本条目选择一维 Poisson 边值问题作为最小可运行实例：

- 强形式：`-u''(x)=1, x in (0,1)`
- 边界条件：`u(0)=u(1)=0`
- 精确解：`u(x)=x(1-x)/2`

输入：截断模态数 `N` 与积分阶数。  
输出：Galerkin 近似解误差（`L2`、`H1` 半范、最大误差）及线性系统诊断量。

## R03

弱形式：在 `V=H_0^1(0,1)` 中求 `u` 使

`a(u,v)=l(v), forall v in V`

其中

- `a(u,v)=int_0^1 u'(x)v'(x) dx`
- `l(v)=int_0^1 f(x)v(x) dx`, `f(x)=1`

Galerkin 离散：选 `V_N=span{phi_1,...,phi_N}`，求 `u_N in V_N` 满足

`a(u_N,v_N)=l(v_N), forall v_N in V_N`。

## R04

本 MVP 的离散空间采用正弦基：

`phi_k(x)=sin(k*pi*x), k=1..N`

特点：

- 自动满足 `phi_k(0)=phi_k(1)=0`（天然符合齐次 Dirichlet 边界）；
- 形成连续 Galerkin 谱离散；
- 系数向量 `c` 满足 `A c = b`，其中
  - `A_ij = int_0^1 phi_i'(x)phi_j'(x) dx`
  - `b_i = int_0^1 f(x)phi_i(x) dx`

## R05

复杂度（密集实现）可写为：

- 组装 `A,b`：`O(Q*N^2)`，`Q` 为积分点数；
- 线性求解（`numpy.linalg.solve`）：`O(N^3)`；
- 误差评估：`O(Qe*N)`，`Qe` 为误差积分点数。

对本目录的演示规模（`N<=32`）而言，计算开销非常小。

## R06

微型手算例（`N=1`）可直观看到 Galerkin 系数含义：

设 `u_1(x)=c_1 sin(pi x)`。

- `A_11 = int_0^1 (pi cos(pi x))^2 dx = pi^2/2`
- `b_1 = int_0^1 sin(pi x) dx = 2/pi`

故

`c_1 = b_1 / A_11 = 4/pi^3 ≈ 0.129006`

即一模态近似 `u_1(x)=4/pi^3 sin(pi x)`，在中心点 `x=0.5` 给出 `0.129`，接近精确值 `0.125`。

## R07

Galerkin 方法重要性：

- 统一了“连续算子问题 -> 离散线性系统”的转换路径；
- 可以结合不同基函数（有限元局部基、谱基、波形基）以适配不同问题；
- 在椭圆、抛物、双曲 PDE 的离散中都可复用这一思想；
- 提供了可分析的误差理论和稳定性框架。

## R08

理论关键点：

1. 在 Hilbert 空间中，弱形式对应一个双线性型与线性泛函。  
2. 取有限维子空间 `V_N` 后，Galerkin 条件等价于“残差对 `V_N` 正交”。  
3. 在椭圆问题场景下（本例 Poisson），离散系统矩阵通常对称正定。  
4. 误差满足 Céa 型估计：离散解接近 `V_N` 内最优逼近。  
5. 当基函数族逼近能力提升（`N` 增大）时，误差应下降。

## R09

适用条件与限制：

- 适合可以写成稳定弱形式的问题；
- 适合能构造合适试探/测试空间的场景；
- 本实现是 1D、齐次 Dirichlet、密集线代版本，不追求大规模性能；
- 若问题更复杂（高维、非线性、非齐次边界），需要扩展空间与装配策略。

## R10

本 MVP 的正确性验证框架：

- 数学层面：按定义组装 `A_ij` 与 `b_i`，求解 `A c=b`；
- 数值层面：检查 `||A c-b||_inf` 是否接近机器精度；
- 物理/边界层面：检查 `u_N(0),u_N(1)` 是否约为 0；
- 误差层面：与解析解比较 `L2/H1` 误差并观察随 `N` 下降趋势。

## R11

误差与收敛现象（本例）：

- `N=2,4,8,16,32` 时，`L2` 误差单调下降；
- 观测到约 `2.1~2.5` 的收敛斜率（按模态数翻倍计）；
- `H1` 半范误差和 `MaxAbs` 误差也同步下降。

说明实现既满足离散方程，也体现了 Galerkin 逼近能力。

## R12

性能视角：

- 当前代码使用 `numpy.einsum` 明确积分与矩阵构造，便于审计；
- 对小规模问题比引入大型框架更轻量、透明；
- 若扩展到大规模自由度，应改为稀疏装配 + 迭代求解（如 CG/Multigrid）。

## R13

本目录可验证保证：

- `demo.py` 无交互输入，直接运行；
- 输出每个 `N` 的误差、条件数、残差；
- 内置断言检查：
  - `L2` 误差随 `N` 单调下降；
  - 最高阶结果达到给定精度阈值；
  - 边界值与线性残差在容差内；
- 正常结束打印 `All checks passed.`。

## R14

鲁棒性与常见失效模式：

- `quadrature_order` 过低会导致积分误差偏大，影响精度；
- `N` 过大时密集系统条件数上升，数值误差会放大；
- 若替换为不满足边界的基函数，必须额外处理边界条件；
- 若 `f(x)` 不光滑，收敛速度会显著变化。

## R15

实现结构（对应 `demo.py`）：

- `gauss_legendre_unit`：生成 `[0,1]` 上高斯积分点与权重；
- `sine_basis_matrix` / `sine_basis_derivative_matrix`：构造试探/导数基；
- `assemble_system`：按弱形式组装 `A,b`；
- `run_case`：求解线性系统并计算诊断量；
- `compute_errors`：输出 `L2/H1/MaxAbs` 与边界值；
- `validate_results`：执行自动断言；
- `main`：批量跑多个 `N` 并打印收敛表。

## R16

相关方法链路：

- 有限元（FEM）：局部多项式基 + 网格装配，本质同属 Galerkin 家族；
- 谱方法：全局基（如正弦、Chebyshev）下的高精度 Galerkin；
- Petrov-Galerkin：测试空间与试探空间不同；
- DG：允许单元间不连续，通过数值通量耦合。

## R17

运行方式：

```bash
cd Algorithms/数学-数值分析-0434-Galerkin方法
python3 demo.py
```

依赖：

- `numpy`
- Python 标准库 `dataclasses`、`typing`

## R18

源码级算法流程拆解（`demo.py`，8 步）：

1. `main()` 创建 `GalerkinConfig`，设定模态序列 `N=(2,4,8,16,32)`。  
2. 对每个 `N`，`run_case()` 调用 `assemble_system()` 构造离散方程 `A c=b`。  
3. `assemble_system()` 先在 `gauss_legendre_unit()` 中生成 `[0,1]` 积分点与权重。  
4. `sine_basis_matrix()` 与 `sine_basis_derivative_matrix()` 计算 `phi_k` 与 `phi_k'` 的离散值。  
5. 用 `np.einsum` 按积分公式显式累计 `A_ij=int phi_i'phi_j'` 与 `b_i=int f phi_i`。  
6. `run_case()` 用 `numpy.linalg.solve` 解线性系统得到系数向量 `c`，并计算 `||A c-b||_inf` 与 `cond(A)`。  
7. `compute_errors()` 在高阶积分点上重建 `u_N` 与 `u_N'`，和解析解对比得到 `L2/H1/MaxAbs`，同时检查边界值。  
8. `validate_results()` 检查误差单调下降、终阶精度达标、边界与残差在容差内，最后打印 `All checks passed.`。

说明：实现没有调用任何 PDE 黑盒库，Galerkin 的组装与求解链路均在源码中显式展开。
