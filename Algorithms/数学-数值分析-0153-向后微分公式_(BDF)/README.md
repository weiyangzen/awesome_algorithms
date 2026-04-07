# 向后微分公式 (BDF)

- UID: `MATH-0153`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `153`
- 目标目录: `Algorithms/数学-数值分析-0153-向后微分公式_(BDF)`

## R01

本条目实现向后微分公式（Backward Differentiation Formula, BDF）的最小可运行版本，覆盖：

- `BDF1`（即 Backward Euler）；
- `BDF2`（二阶 BDF，含一步启动）；
- 每步通过 Newton 迭代求解隐式方程，而非调用黑盒 ODE 求解器。

MVP 目标是把“BDF 在刚性方程上为何稳定、在代码里如何求隐式步”讲清楚。

## R02

问题定义（固定步长标量 ODE）：

- 输入：`f(t,y)`、`t0`、`y0`、步长 `h`、步数 `N`、Newton 参数 `tol/max_iter`。
- 输出：离散解 `y_n ≈ y(t_n)`、每步 Newton 诊断信息、误差统计。

演示方程选取刚性线性问题：

`y' = λ (y - cos t) - sin t,  λ < 0`

当 `y(0)=1` 时，解析解为 `y(t)=cos t`，便于直接计算误差并观察稳定性。

## R03

核心离散公式：

- BDF1（Backward Euler）：

`y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})`

- BDF2：

`(3y_{n+1} - 4y_n + y_{n-1}) / (2h) = f(t_{n+1}, y_{n+1})`

二者都属于隐式格式。每步都需要解非线性方程 `g(z)=0`，本实现统一用 Newton：

`z_{k+1} = z_k - g(z_k)/g'(z_k)`。

## R04

算法流程概览：

1. 校验 `h/steps/tol/max_iter` 与初值有限性。
2. 构造刚性 RHS 与解析解函数。
3. 对 BDF1：每一步建立隐式残差 `g(z)=z-y_n-h f(t_{n+1},z)` 并 Newton 求根。
4. 对 BDF2：先用一步隐式梯形法生成 `y1`，再进入二阶公式迭代。
5. BDF2 的每一步用线性外推给 Newton 初值，提高收敛速度。
6. 记录每步 Newton 迭代次数和残差。
7. 用解析解计算误差并估计经验阶。
8. 与显式 Euler 对比大步长下的稳定性。

## R05

核心数据结构：

- `numpy.ndarray t, y`：保存离散时间网格与数值解。
- `NewtonDiagnostic`（`dataclass`）：记录
  - `method`（`BDF1` / `BDF2` / `BDF2-start(trapezoid)`）、
  - `step`、`t_next`、`initial_guess`、`value`、`iterations`、`residual`。
- `list[tuple(h, error)]`：用于阶数估计。

结构保持最小化，但足够支持可解释调试。

## R06

正确性与可验证性：

- 公式正确性：代码分别按 BDF1/BDF2 标准离散式构造残差函数。
- 隐式求解正确性：每步都对 `g(z)=0` 做 Newton 并检查收敛。
- 误差验证：同网格上与解析解 `cos(t)` 对比 `max_abs_error`。
- 收敛行为：`h` 逐次减半时，BDF1 经验阶应接近 1，BDF2 接近 2。

## R07

复杂度分析：

设总步数为 `N`，平均每步 Newton 迭代次数为 `K`。

- 时间复杂度：`O(NK)`；
- 空间复杂度：
  - 保留完整轨迹与诊断时为 `O(N)`；
  - 若只保留当前状态可降至 `O(1)`（本 MVP 为可观测性保留 `O(N)`）。

## R08

边界与异常处理：

- `h <= 0`、`steps <= 0`、`tol <= 0`、`max_iter <= 0`：`ValueError`。
- `t0/y0/lam` 非有限：`ValueError`。
- `lam >= 0`：当前刚性示例视为非法（要求 `lam < 0`）。
- Newton 导数过小或出现非有限值：`RuntimeError`。
- Newton 超过 `max_iter` 未收敛：`RuntimeError`。
- `(t_end - t0)/h` 非近整数：`ValueError`。

## R09

MVP 取舍：

- 仅实现标量 ODE，避免把重点从 BDF 核心流程转移到向量工程细节。
- 不依赖 `scipy.integrate.solve_ivp` 或 `BDF` 黑盒接口。
- Jacobian 用有限差分近似，既保持一般性，又避免推导专用导数。
- 启动策略采用“二阶隐式梯形启动 BDF2”，以避免低阶启动污染整体阶数。

## R10

`demo.py` 函数职责：

- `check_solver_controls` / `ensure_finite`：参数合法性检查。
- `stiff_tracking_rhs_factory`：构造刚性测试方程右端。
- `stiff_tracking_exact`：解析解。
- `numerical_jacobian_scalar`：标量 Jacobian 有限差分。
- `newton_solve_scalar`：隐式步的 Newton 求解器。
- `backward_euler_bdf1`：BDF1 主积分器。
- `bdf2`：BDF2 主积分器（含隐式梯形启动）。
- `explicit_euler`：显式基线方法。
- `run_convergence_experiment`：收敛阶实验。
- `run_stability_comparison`：刚性稳定性对照实验。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0153-向后微分公式_(BDF)
python3 demo.py
```

脚本无交互输入，会直接打印：

- BDF1/BDF2 的收敛表与经验阶；
- Newton 迭代诊断前缀；
- 大步长下 BDF 与显式 Euler 的稳定性对比。

## R12

输出字段解读：

- `BDF1 max_abs_err` / `BDF2 max_abs_err`：区间最大绝对误差。
- `mean/newton_max_iter`：Newton 平均迭代步数 / 最大迭代步数。
- `p`：步长减半得到的经验收敛阶。
- `final_value`：终点数值解。
- `exact_final`：终点解析解。
- `max_abs_state`：轨迹最大幅值（可用于观察是否数值爆炸）。

## R13

建议最小测试集：

1. 收敛测试：`h = 0.04, 0.02, 0.01, 0.005`，区间 `[0,1]`。
2. 稳定性测试：`h=0.2`，区间 `[0,4]`。
3. 参数异常：`h<=0`、`steps<=0`、`tol<=0`。
4. Newton 异常：将 `max_iter` 降到很小值（如 `1`）触发不收敛路径。
5. 网格异常：让 `(t_end-t0)/h` 非整数验证防御分支。

## R14

关键可调参数：

- `lam`：刚性强度（绝对值越大越刚性）。
- `h`：时间步长，决定稳定性与精度。
- `tol`：Newton 收敛阈值。
- `max_iter`：Newton 最大迭代轮数。
- BDF2 初值策略：当前用线性外推，可改为更高阶预测器。

调参建议：先固定 `tol/max_iter`，再扫描 `h` 与 `lam` 看误差和迭代代价。

## R15

方法对比：

- 显式 Euler：实现最简单，但刚性问题对步长约束严格，易失稳。
- BDF1：一阶、A-稳定，鲁棒性强。
- BDF2：二阶、在多数刚性场景仍具较好稳定性与更高精度。

本示例中，当 `λ=-20` 且 `h=0.2` 时，显式 Euler 超出稳定步长阈值（`h < 0.1`），而 BDF 仍保持可用。

## R16

典型应用场景：

- 刚性化学动力学/反应网络中的时间推进；
- 含快慢尺度的控制与热传导子问题；
- 需要较大步长稳定推进的隐式时间离散内核；
- 作为更完整 DAE/ODE 隐式框架的基础积木。

## R17

后续扩展方向：

- 扩展到向量系统 `y in R^d` 并显式支持 Jacobian 矩阵。
- 引入稀疏线性代数，提高大规模问题效率。
- 增加自适应步长与局部截断误差估计。
- 支持更高阶 BDF（BDF3~BDF6）及可变阶策略。
- 与 `scipy` 的 BDF 求解器做误差/性能基准对照。

## R18

源码级算法流（对应 `demo.py`，8 步）：

1. `main` 设置刚性参数 `lam=-20`、初值与实验配置，构造 `rhs` 与数值 Jacobian。  
2. `run_convergence_experiment` 对每个步长 `h` 调用 `backward_euler_bdf1` 与 `bdf2`。  
3. `backward_euler_bdf1` 在每一步建立隐式残差 `g(z)=z-y_n-h f(t_{n+1},z)`。  
4. `newton_solve_scalar` 对该残差做 Newton 迭代，得到 `y_{n+1}`，并回写迭代次数与残差。  
5. `bdf2` 先执行一步隐式梯形启动，再对 `n>=1` 使用 `3y_{n+1}-4y_n+y_{n-1}-2h f(t_{n+1},y_{n+1})=0`。  
6. 两个解序列与 `stiff_tracking_exact` 对齐，计算 `max_abs_error` 并由 `estimate_orders` 估计经验阶。  
7. `print_diag_prefix` 打印 BDF2 隐式步的 Newton 诊断前缀，检查求解器是否稳定收敛。  
8. `run_stability_comparison` 在大步长 `h=0.2` 下对比 BDF1/BDF2 与显式 Euler 的终值、最大误差与状态幅值。  
