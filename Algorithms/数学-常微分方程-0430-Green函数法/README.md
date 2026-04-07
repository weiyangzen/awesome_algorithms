# Green函数法

- UID: `MATH-0430`
- 学科: `数学`
- 分类: `常微分方程`
- 源序号: `430`
- 目标目录: `Algorithms/数学-常微分方程-0430-Green函数法`

## R01

Green 函数法把线性边值问题转化为积分表示。对本条目，MVP 处理区间 `[0,1]` 上的二阶常微分方程：

`-y''(x) + c y(x) = f(x),  y(0)=y(1)=0`，其中 `c>=0`。

核心思想：先构造满足边界条件的核函数 `G(x,xi)`，再用
`y(x)=∫_0^1 G(x,xi)f(xi)dxi` 直接得到解。

## R02

任务定义：

- 输入：
  - 系数 `c`（本 MVP 限定 `c>=0`）；
  - 右端函数 `f(x)`；
  - 均匀网格点数 `n_grid`。
- 输出：
  - Green 积分近似解 `y(x_i)`；
  - 与参考解的最大绝对误差 `max|green-ref|`；
  - 方程离散残差 `max|FD residual|`；
  - 采样点表格（`x, green, ref, abs_err`）。

## R03

本实现使用的 Green 核如下。

1. 当 `c>0`，令 `k=sqrt(c)`，有

`G(x,xi) = sinh(k min(x,xi)) sinh(k(1-max(x,xi))) / (k sinh(k))`。

2. 当 `c=0`（`-y''=f`）时，用极限核

`G(x,xi) = min(x,xi) * (1-max(x,xi))`。

于是统一求解公式：

`y(x)=∫_0^1 G(x,xi)f(xi)dxi`。

离散实现采用复化梯形：

`y_i ≈ Σ_j G(x_i,xi_j) w_j f(xi_j)`。

## R04

算法流程（高层）：

1. 构造均匀网格 `x_0...x_{N-1}`。
2. 在同一网格上计算 `f(x_j)`。
3. 根据 `c` 分支构造 Green 矩阵 `G_{ij}=G(x_i,x_j)`。
4. 生成梯形权重 `w_j`。
5. 通过矩阵向量乘法计算 `y = G @ (w*f)`。
6. 使用闭式解或高精度 `solve_bvp` 作为参考，计算最大误差。
7. 用中心差分计算 `-y'' + c y - f` 的离散残差。
8. 打印表格与汇总，并做阈值验收。

## R05

`demo.py` 的核心数据结构：

- `GreenCase`（`@dataclass`）：
  - `name`：样例名称；
  - `c`：方程系数；
  - `forcing`：右端函数；
  - `exact`：解析参考函数（可为空）；
  - `n_grid`：网格数量；
  - `error_tol`、`residual_tol`：验收阈值。
- `numpy.ndarray`：网格、Green 矩阵、解向量、残差向量。
- `pandas.DataFrame`：采样点误差表格输出。

## R06

正确性由三层保证：

1. 公式层：直接按 Green 表示公式离散，不用黑箱 ODE 解法替代主算法。
2. 对照层：
  - 样例 A/B 使用闭式解对照；
  - 样例 C 使用高精度 `solve_bvp` 仅作参考真值。
3. 方程层：额外检查离散残差 `-y'' + c y - f`，避免“误差小但方程不满足”的假阳性。

## R07

设网格点数为 `N`：

- 构造 Green 矩阵：`O(N^2)` 时间，`O(N^2)` 空间。
- 积分离散（矩阵向量乘）：`O(N^2)` 时间。
- 残差计算：`O(N)` 时间。

总体复杂度：`O(N^2)` 时间，`O(N^2)` 空间。对于 MVP 使用的 `N=401`，开销很小。

## R08

边界与异常处理：

- `c<0` 直接报错（本 MVP 只实现 `c>=0` 分支）。
- 非均匀网格报错。
- `forcing` 维度不匹配或含非有限值报错。
- `n_grid` 太小报错。
- `solve_bvp` 参考解失败时报错。
- 误差或残差超过阈值时报错，便于自动验证直接识别失败。

## R09

MVP 取舍：

- 保留：
  - Green 核闭式构造；
  - `c>0` 与 `c=0` 两个关键分支；
  - 误差与残差双重验收。
- 不做：
  - `c<0` 的三角函数核分支；
  - 非均匀网格或自适应积分；
  - 更高维 PDE Green 函数扩展。

目标是“实现短小但不黑箱，完整体现 Green 方法主干”。

## R10

`demo.py` 函数职责：

- `_trapezoid_weights`：生成复化梯形积分权重。
- `green_kernel_matrix`：按 `c` 分支构造 `G(x,xi)`。
- `solve_green_dirichlet`：执行 `y = G @ (w*f)`。
- `finite_difference_residual`：计算离散方程残差。
- `reference_solve_bvp`：为无闭式样例构造高精度参考。
- `run_case`：单样例运行、打印、阈值断言。
- `build_cases`：定义固定测试样例。
- `main`：串行执行全部样例并汇总。

## R11

运行方式（非交互）：

```bash
cd Algorithms/数学-常微分方程-0430-Green函数法
uv run python demo.py
```

脚本会直接输出每个样例和总结，不需要输入参数。

## R12

输出字段说明：

- `max|green-ref|`：Green 积分解与参考解在网格上的最大绝对误差。
- `max|FD residual|`：中心差分残差最大绝对值。
- 表格列：
  - `x`：采样点；
  - `green`：Green 方法近似；
  - `ref`：参考值；
  - `abs_err`：绝对误差。
- `Summary`：所有样例中的最差误差与最差残差。

## R13

内置样例：

1. `Case A: -y'' + 4y = sin(pi x)`
- 解析解：`sin(pi x)/(pi^2+4)`。

2. `Case B: -y'' = 1`
- 解析解：`0.5*x*(1-x)`；
- 同时验证 `c=0` 核分支。

3. `Case C: -y'' + 1.5y = exp(x)`
- 无闭式对照；
- 用高精度 `solve_bvp` 作为参考。

## R14

关键参数与经验：

- `n_grid`：越大积分误差越小，MVP 采用 `401` 点以保证误差在 `1e-6` 量级。
- `error_tol`：按样例难度设置在 `2e-6 ~ 3e-6`。
- `residual_tol`：设置为 `2e-4`，明显高于实测 `1e-6` 量级，留出平台浮动余量。

## R15

与其他常见方法对比：

- 对比有限差分直接解线性方程组：
  - 有限差分更通用；
  - Green 法更突出“解是核与右端卷积”的解析结构。
- 对比 shooting 法：
  - shooting 需要初值迭代；
  - Green 法直接满足边界条件并一次积分完成。
- 对比黑箱 `solve_bvp`：
  - `solve_bvp` 是通用求解器；
  - 本实现把 Green 核构造与积分离散过程显式写出。

## R16

典型应用：

- 线性边值问题教学与推导复现。
- 已知边界条件下的受迫振动/稳态响应计算。
- 作为验证器：用 Green 法结果校验其他数值算法实现。
- 构造核积分视角，为后续 Fredholm/Volterra 积分方程学习打基础。

## R17

可扩展方向：

- 支持 `c<0` 的核公式（双曲函数切换为三角函数）。
- 扩展到一般线性算子 `-(p(x)y')' + q(x)y = f(x)` 的数值 Green 构造。
- 引入高阶求积（Simpson / Gauss）降低积分离散误差。
- 用分块或低秩近似降低 `O(N^2)` 存储与计算成本。
- 增加参数扫描与误差收敛实验输出。

## R18

`demo.py` 源码级算法流（8 步）：

1. `main()` 调用 `build_cases()` 生成 3 个固定样例（含闭式与非闭式）。
2. `run_case()` 生成均匀网格并计算 `f(x)`，随后调用 `solve_green_dirichlet()`。
3. `solve_green_dirichlet()` 调用 `green_kernel_matrix()` 构造 `G(x_i,xi_j)`，`c=0` 与 `c>0` 分别走不同核公式。
4. `solve_green_dirichlet()` 再调用 `_trapezoid_weights()`，执行 `y = G @ (w*f)` 得到积分近似解。
5. `run_case()` 选择参考解来源：若 `exact` 存在则直接计算；否则调用 `reference_solve_bvp()` 获取高精度参考。
6. `run_case()` 计算 `max|green-ref|`，并调用 `finite_difference_residual()` 计算 `-y'' + c y - f` 的离散残差。
7. `run_case()` 打印采样 `DataFrame`，并用 `error_tol`/`residual_tol` 做硬性阈值判定，超限即抛异常。
8. 全部样例通过后，`main()` 输出全局最差误差与残差并结束。
