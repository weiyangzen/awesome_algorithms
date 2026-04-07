# 配点法

- UID: `MATH-0435`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `435`
- 目标目录: `Algorithms/数学-数值分析-0435-配点法`

## R01

配点法（Collocation Method）是一类“先选近似函数空间，再在若干离散点强制方程成立”的数值离散方法。它常用于常微分方程边值问题、积分方程、以及某些偏微分方程的半离散处理中。

本条目 MVP 选取一维二阶边值问题：

`y''(x) + y(x) = sin(pi x), x in [0,1], y(0)=0, y(1)=0`

该问题有解析解，可用于直接验证实现正确性。

## R02

本实现的输入输出定义：

- 输入：多项式次数 `m`（即基函数个数）、评估网格大小 `grid_size`
- 输出：
  - 近似系数 `c_1..c_m`
  - 配点残差 `r(x_i)`
  - 稠密网格上的数值解 `y_num`
  - 与解析解对比的最大绝对误差 `max_abs_error`

目标是用最小代码完整展示“从微分方程到线性代数系统”的配点法流程。

## R03

近似函数采用满足边界条件的基：

`phi_k(x)=x^k(1-x), k=1..m`

于是

`y_m(x)=sum_{k=1}^m c_k phi_k(x)`

天然满足 `y_m(0)=y_m(1)=0`，无需额外边界代数方程。配点条件设置为：在 `m` 个内部点 `x_i=i/(m+1)` 上强制

`y_m''(x_i)+y_m(x_i)=sin(pi x_i)`。

## R04

离散化推导：

1. 写出残差：`r(x)=y_m''(x)+y_m(x)-f(x)`。
2. 令 `r(x_i)=0, i=1..m`。
3. 因为 `y_m` 对系数线性，得到线性系统 `A c = b`。
4. 系数矩阵元素：
   `A_{i,k}=phi_k''(x_i)+phi_k(x_i)`。
5. 右端：`b_i=f(x_i)=sin(pi x_i)`。

本例中

`phi_k''(x)=k(k-1)x^(k-2)-k(k+1)x^(k-1)`。

## R05

核心数据结构（见 `demo.py`）：

- `SolveResult`：单次求解结果容器（次数、系数、配点、残差、网格、误差）
- `a: np.ndarray (m,m)`：线性系统矩阵
- `b: np.ndarray (m,)`：线性系统右端
- `coeffs: np.ndarray (m,)`：待求系数向量
- `grid/y_num/y_exact`：用于误差评估的稠密采样

整体保持轻量，仅依赖 `numpy`。

## R06

微型例子（`m=2`）：

- 试探解：`y_2=c1*x(1-x)+c2*x^2(1-x)`
- 配点：`x_1=1/3, x_2=2/3`
- 代入 `y_2''+y_2=f` 于两个点，得到 2x2 线性系统
- 解得 `(c1,c2)` 后即可恢复整段近似函数

这体现了配点法的核心特征：把连续算子约束压缩为有限个“点条件”。

## R07

正确性依据：

- 结构正确性：矩阵 `A` 与右端 `b` 完全来自配点方程定义。
- 边界正确性：基函数构造保证边界条件严格满足。
- 代数正确性：`np.linalg.solve` 精确求解离散线性系统（浮点意义下）。
- 结果正确性：对照解析解 `sin(pi x)/(1-pi^2)` 计算误差，验证数值行为。

## R08

复杂度分析：

- 组装矩阵：`O(m^2)`
- 线性求解（稠密直接法）：`O(m^3)`
- 网格评估（`G=grid_size+1`）：`O(mG)`

因此主导成本是 `O(m^3)`；空间复杂度约 `O(m^2 + G)`。

## R09

边界与异常处理：

- `m < 1`：抛出 `ValueError`
- `grid_size < 10`：抛出 `ValueError`
- 系统奇异或病态导致线性求解失败：由 `np.linalg.solve` 抛错
- 运行自检：
  - 配点残差无穷范数需足够小
  - 更高 `m` 相比低 `m` 应有更小误差（本示例参数下）

## R10

`demo.py` 模块职责：

- `basis_function / basis_second_derivative`：定义基函数及二阶导
- `build_linear_system`：组装 `A,b`
- `evaluate_trial_solution`：用系数重建近似解
- `evaluate_residual`：计算残差
- `solve_collocation_bvp`：执行单次配点求解并评估误差
- `run_convergence_demo`：批量次数实验，打印收敛表
- `print_solution_sample`：输出样例点值
- `main`：组织流程并执行断言

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0435-配点法
python3 demo.py
```

脚本无需任何交互输入，会自动打印误差表与样例解并进行断言检查。

## R12

输出字段解读：

- `m`：基函数个数（也可视作近似多项式自由度）
- `max_abs_error`：网格上 `|y_num - y_exact|` 最大值
- `collocation_residual_inf`：配点处残差无穷范数
- `prev/cur error`：相邻两次 `m` 的误差比，反映收敛趋势

在该平滑问题中，`m` 增大时误差通常显著下降。

## R13

建议最小测试集：

- 正常场景：`m = 2,3,4,5,6,7`
- 参数异常：`m=0`
- 网格异常：`grid_size=5`
- 健壮性观察：打印残差无穷范数，确认接近机器精度

可进一步增加 `m` 检查高阶下条件数影响（误差可能不再单调改善）。

## R14

关键可调参数：

- `degrees`：收敛实验的次数序列
- `grid_size`：误差评估网格精度
- `rows`：样例输出行数

经验上可先用小 `m` 做快速验证，再逐步提高 `m` 观察精度与稳定性折中。

## R15

与常见方法对比：

- 有限差分法：局部离散、矩阵稀疏，工程实现稳定；高精度通常需细网格。
- Galerkin / 有限元：弱形式框架更通用，适合复杂几何与变系数。
- 配点法：实现直观、精度高、原型速度快，但高阶时可能出现病态与稳定性问题。

本条目聚焦“最小透明实现”，适合作为后续谱法/伪谱法的前置理解。

## R16

典型应用：

- ODE 边值问题教学与快速验证
- 光滑问题上的高阶近似原型
- 与射线法、有限差分法做精度/成本基准对照

在工程中也常作为更复杂离散（如谱元、正交配点）的基础原型。

## R17

后续扩展方向：

- 配点节点改为 Chebyshev/Lobatto 节点以改善条件数
- 支持非齐次边界（通过 lifting 函数分解）
- 引入非线性方程 `y'' + g(y)=f`，用 Newton 迭代配点
- 用 `scipy.linalg` 做条件数分析与误差放大诊断

## R18

源码级算法流程拆解（对应 `demo.py`，8 步）：

1. `main` 设定次数序列 `degrees=[2,3,4,5,6,7]`，逐个调用 `solve_collocation_bvp(m)`。  
2. `solve_collocation_bvp` 调用 `build_linear_system(m)`，先生成内部配点 `x_i=i/(m+1)`。  
3. `build_linear_system` 逐点逐基计算 `A_{i,k}=phi_k''(x_i)+phi_k(x_i)`，并构造右端 `b_i=sin(pi x_i)`。  
4. 回到 `solve_collocation_bvp`，调用 `np.linalg.solve(A,b)` 得到系数向量 `c`。  
5. 调用 `evaluate_residual(x_col,c)` 计算配点残差，确认离散方程被满足。  
6. 在稠密网格 `grid` 上调用 `evaluate_trial_solution(grid,c)` 重建数值解 `y_num`。  
7. 同网格上计算解析解 `y_exact`，得到 `max_abs_error=max|y_num-y_exact|`。  
8. `run_convergence_demo` 汇总各 `m` 的误差与残差并打印表格，`main` 通过断言完成基本正确性验收。  

本实现未调用任何边值问题黑箱求解器（如 `scipy.integrate.solve_bvp`）；离散、组装与求解流程均在源码显式展开。
