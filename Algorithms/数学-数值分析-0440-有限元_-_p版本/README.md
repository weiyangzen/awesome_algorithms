# 有限元 - p版本

- UID: `MATH-0440`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `440`
- 目标目录: `Algorithms/数学-数值分析-0440-有限元_-_p版本`

## R01

`p` 版本有限元（p-version FEM）是在网格拓扑固定（单元数量不变、单元尺寸 `h` 基本不变）的前提下，通过提高单元内试探函数的多项式阶数 `p` 来提升精度的离散方法。它与 `h` 版本（细分网格）互补，适合解足够光滑时快速降低误差。

## R02

本任务用一维 Poisson 边值问题演示 `p` 细化：

- 强形式：`-u''(x)=f(x), x in (0,1)`
- 边界：`u(0)=u(1)=0`
- 取 `f(x)=pi^2 sin(pi x)`，精确解 `u(x)=sin(pi x)`

该问题解析解已知，便于验证误差与收敛趋势。

## R03

`p` 版本核心思想：

- 固定分区（例如 `nelems=4`）
- 每个单元仍做局部近似，但局部空间从 `P1 -> P2 -> ... -> Pp`
- 通过更高阶形函数提升表示能力，尤其对光滑解常出现接近指数型误差衰减

在工程中常见 `hp-FEM`，其中 `p` 版本是 `hp` 体系的一条主线。

## R04

离散基函数与映射设置：

- 参考单元：`xi in [-1,1]`
- 局部节点：`p+1` 个等距节点
- 局部基：基于这些节点构造 Lagrange 形函数 `N_i(xi)`
- 几何映射：`x = (xl+xr)/2 + J*xi`, `J=(xr-xl)/2`

导数变换：`dN/dx = (dN/dxi)/J`。

## R05

弱形式：在 `V=H_0^1(0,1)` 中找 `u`，使得

`a(u,v)=l(v), forall v in V`

其中：

- `a(u,v)=int_0^1 u' v' dx`
- `l(v)=int_0^1 f v dx`

离散后得到线性系统：`K u_h = F`。

## R06

局部矩阵计算（单元 `e=[xl,xr]`）：

- `K_e(i,j)=int_e dN_i/dx * dN_j/dx dx`
- `F_e(i)=int_e f(x) * N_i(x) dx`

数值积分采用 Gauss-Legendre：

- 刚度积分点数取 `max(2p+3, 6)`
- 误差积分点数取更高阶 `max(2p+6, 10)`

以减少积分误差对收敛观察的干扰。

## R07

全局装配策略：

- 相邻单元共享端点自由度
- 单元内部高阶节点自由度在本单元独有
- 逐单元将 `K_e, F_e` 累加到全局 `K, F`

对固定 `nelems` 与统一阶数 `p`，自由度近似 `ndof = nelems * p + 1`。

## R08

边界条件处理：

- 采用强施加（Dirichlet elimination）
- 固定节点：左端 `0` 和右端 `ndof-1`
- 求解自由子系统 `K_ff u_f = F_f - K_fc u_c`

本示例是齐次边界，故 `u_c=0`。

## R09

复杂度（密集实现）概览：

- 装配阶段：每单元 `O((p+1)^2 * nq)`，总计约 `O(nelems * p^3)`
- 线性求解：当前示例用 `numpy.linalg.solve` 对密集矩阵求解，约 `O(ndof^3)`

该 MVP 重点是算法可读性，不是大规模稀疏求解性能。

## R10

数值特性与误差观察：

- 在光滑解条件下，固定 `h` 提升 `p` 通常显著降误差
- 高阶等距节点会导致条件数变差（Runge 风险与矩阵病态性上升）
- 工程实现常改用 Gauss-Lobatto 节点、层次基或正交基来提升稳定性

本示例保留“朴素实现”，用于清晰展示 p-version 机制。

## R11

MVP 代码结构：

- `lagrange_basis_and_derivative`: 计算形函数及导数
- `element_matrices`: 单元刚度/载荷积分
- `build_connectivity`: 构建全局自由度映射
- `assemble_and_solve`: 总装配 + 施加边界 + 求解
- `evaluate_l2_error`: 用高阶积分评估 `L2` 误差
- `run_p_refinement_demo`: 在固定网格上遍历多个 `p`

## R12

运行输出内容：

- 每个 `p` 的自由度 `ndof`
- 对应 `L2 error`
- 相对上一阶的改进倍率 `improve vs prev`

该输出直接验证“固定网格、提高阶数”的 p 版本收敛行为。

## R13

使用方式（无交互）：

```bash
python3 demo.py
```

程序会自动执行 `p=1..6` 的对比，不需要命令行参数。

## R14

预期现象：

- `p` 增大时，`ndof` 线性增加
- `L2 error` 明显下降
- 相邻 `p` 的改进倍率通常大于 1，说明提升阶数有效

若误差未下降，优先检查形函数导数、Jacobian 变换与积分阶数。

## R15

当前 MVP 的边界与限制：

- 仅覆盖 1D 标量椭圆方程
- 使用密集矩阵，不适合超大自由度
- 使用等距插值节点，高阶时条件数不理想
- 未实现自适应 `p` 选择与误差估计驱动

## R16

可扩展方向：

- 替换为稀疏组装 + 稀疏线性求解
- 节点改为 GLL（Gauss-Lobatto-Legendre）以改善稳定性
- 实现 `hp` 策略：局部同时调节 `h` 与 `p`
- 扩展到 2D/3D（三角形、四边形、六面体高阶单元）

## R17

最小验证清单：

- `python3 demo.py` 可直接运行
- `README.md` 与 `demo.py` 均已完成填充，无模板占位符残留
- 输出表中 `p` 增大时误差总体下降
- `meta.json` 的 UID/名称/路径与任务一致

## R18

源码级算法流程（对应 `demo.py`，无黑盒）：

1. 在 `run_p_refinement_demo` 固定 `nelems=4`，遍历 `p in {1..6}`。
2. 对每个 `p`，`build_connectivity` 生成单元到全局自由度映射（共享端点、独占内部点）。
3. `assemble_and_solve` 内逐单元调用 `element_matrices(xl, xr, p)`。
4. `element_matrices` 先在参考单元定义 `p+1` 节点，再调用 `lagrange_basis_and_derivative` 显式计算 `N_i` 与 `dN_i/dxi`。
5. 将导数通过 `J=(xr-xl)/2` 映射到物理坐标，按 Gauss-Legendre 积分公式构造 `K_e` 与 `F_e`。
6. 回到 `assemble_and_solve`，把 `K_e, F_e` 按映射累加进全局 `K, F`。
7. 强施加两端 Dirichlet 条件，提取自由子块 `K_ff`，用 `numpy.linalg.solve` 解线性系统得到离散解 `u_h`。
8. `evaluate_l2_error` 再次逐单元高阶积分：在积分点计算 `u_h` 与 `u_exact`，累计 `sqrt(sum((u_h-u)^2))` 得到 `L2` 误差。
9. 主程序打印 `p/ndof/L2 error/改进倍率` 表格，完成 p-version 的最小收敛演示。
