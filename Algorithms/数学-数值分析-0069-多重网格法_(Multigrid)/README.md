# 多重网格法 (Multigrid)

- UID: `MATH-0069`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `69`
- 目标目录: `Algorithms/数学-数值分析-0069-多重网格法_(Multigrid)`

## R01

多重网格法是一类用于快速求解离散椭圆型方程（如 Poisson 方程）的迭代框架。核心目标是在接近线性复杂度下求解大规模稀疏线性系统。与单纯 Jacobi/Gauss-Seidel 相比，多重网格显著降低了低频误差衰减缓慢的问题。

## R02

本目录中的 MVP 解决二维单位方形区域上的 Poisson 问题：

- 连续问题: `-Laplace(u) = f, (x,y) in (0,1)^2`
- 边界条件: `u = 0`（Dirichlet）
- 离散方式: 五点差分格式，仅存储内部网格点未知量

## R03

多重网格的关键思想是频率分解：

- 平滑器（如加权 Jacobi）快速消除高频误差
- 网格转移（限制/延拓）把低频误差搬到粗网格上，使其变成高频并被快速消除
- 通过 V-cycle 递归组合上述步骤，得到对全频段误差都高效的收敛机制

## R04

离散后每个内部点满足：

- `A u = f`
- `A u(i,j) = (4u(i,j)-u(i-1,j)-u(i+1,j)-u(i,j-1)-u(i,j+1))/h^2`

残差定义为 `r = f - A u`。多重网格在粗网格上近似求解误差方程 `A e = r`，再将误差修正回细网格：`u <- u + P e_c`。

## R05

本实现采用标准 V-cycle：

1. 细网格做若干次预平滑（加权 Jacobi）
2. 计算细网格残差 `r`
3. 用 Full Weighting 将 `r` 限制到粗网格
4. 在粗网格递归求解误差方程
5. 用双线性延拓把粗网格误差映射回细网格并修正
6. 做若干次后平滑
7. 最粗层使用小规模直接解

## R06

MVP 中的关键算子：

- `apply_operator`: 矩阵无关地应用五点离散算子
- `weighted_jacobi`: 平滑器，默认 `omega=2/3`
- `restrict_full_weighting`: 残差限制，九点模板加权平均
- `prolong_bilinear`: 双线性延拓
- `direct_solve_coarsest`: 最粗层用 `numpy.linalg.solve` 直接解
- `v_cycle`: 递归拼装完整多重网格流程

## R07

正确性直觉：

- 预/后平滑负责局部振荡误差
- 粗网格修正负责全局平滑误差
- 二者互补，避免单一迭代法在某些频段收敛慢
- 最粗层直接求解阻断误差在递归深层累积

## R08

复杂度与可扩展性：

- 单次 V-cycle 的工作量与未知量数 `N` 近似成正比，经验上可写作 `O(N)`
- 存储开销同样近似 `O(N)`
- 相比直接法（通常远高于线性复杂度），在大规模网格上更具优势

## R09

数值性质：

- 对 Poisson 类 SPD 系统，多重网格通常具备网格无关收敛特性（迭代次数随网格加密增长缓慢）
- 平滑器参数（如 `omega`）和网格转移算子的匹配会影响实际收敛因子
- 本示例选择最小实现优先，便于验证流程而非追求最优常数

## R10

`demo.py` 默认参数：

- `n=63`（内部网格点数，满足 `n=2^k-1` 以匹配当前限制/延拓实现）
- `cycles=10`
- `pre_sweeps=3, post_sweeps=3`
- `omega=2/3`

测试右端项来自已知解析解 `u*=sin(pi x)sin(pi y)`，可直接评估相对误差。

## R11

MVP 设计取舍：

- 只依赖 `numpy`
- 不引入外部黑盒求解器
- 重点展示多重网格各组件如何拼装
- 牺牲部分工程优化（如向量化限制/延拓、稀疏矩阵封装）以保持可读性

## R12

代码结构对应关系：

- 问题构造: `setup_poisson_problem`
- 细网格算子与残差: `apply_operator` / `compute_residual`
- 平滑: `weighted_jacobi`
- 网格转移: `restrict_full_weighting` / `prolong_bilinear`
- 最粗层精确解: `direct_solve_coarsest`
- 主循环: `main` 中反复调用 `v_cycle`

## R13

运行方式：

```bash
cd Algorithms/数学-数值分析-0069-多重网格法_(Multigrid)
python3 demo.py
```

无交互输入，直接输出每次 V-cycle 的残差与相对误差。

## R14

期望现象：

- `residual_ratio` 随 V-cycle 迭代持续下降
- `relative_error` 也同步下降
- 表明平滑 + 粗网格修正的组合在该问题上有效

## R15

常见实现坑：

- 网格尺寸不满足 `2^k-1` 时，简单二分限制关系会失效
- 残差符号写反（应为 `f-Au`）会导致不收敛
- 延拓插值权重错误会导致修正失真
- 仅做平滑不做粗网格修正时，低频误差下降很慢

## R16

可扩展方向：

- 改为 W-cycle 或 F-cycle
- 支持非零边界条件和非均匀网格
- 替换为红黑 Gauss-Seidel、Chebyshev smoother
- 用稀疏矩阵/并行实现提升大规模性能
- 扩展到代数多重网格（AMG）

## R17

与其他方法对比：

- 与 Jacobi/GS: 多重网格在低频误差处理上更强
- 与共轭梯度: MG 可单独用，也常作预条件器（PCG+MG）
- 与直接法: MG 对大规模 PDE 离散系统更省内存、扩展性更好

## R18

`demo.py` 的源码级算法流程（非黑盒）如下：

1. `setup_poisson_problem` 构造离散网格、真解 `u_true`、右端项 `f` 与初值 `u=0`。
2. `main` 先用 `compute_residual` 计算初始残差范数，作为后续相对收敛基准。
3. 每次 V-cycle 先在 `weighted_jacobi` 中执行预平滑：通过邻点平均 + 右端项更新抑制高频误差。
4. `compute_residual` 调用 `apply_operator` 得到 `r=f-Au`，明确当前误差方程右端。
5. `restrict_full_weighting` 按九点加权模板把细网格残差映射到粗网格。
6. 在粗网格上递归调用 `v_cycle`；若到最粗层（`n<=3`），`direct_solve_coarsest` 组装小型线性系统并直接求解误差。
7. `prolong_bilinear` 将粗网格误差按双线性权重插值回细网格，执行 `u <- u + e_f`。
8. 再次执行 `weighted_jacobi` 后平滑，清除修正后引入的高频分量。
9. `main` 输出每轮残差比例与相对误差，验证算法收敛行为。

上述 9 步完整覆盖了多重网格在本实现中的核心数据流和算子调用链。
