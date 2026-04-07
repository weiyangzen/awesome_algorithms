# 乘子法

- UID: `MATH-0383`
- 学科: `数学`
- 分类: `约束优化`
- 源序号: `383`
- 目标目录: `Algorithms/数学-约束优化-0383-乘子法`

## R01

乘子法（Method of Multipliers）用于求解带等式约束的优化问题：
`min f(x), s.t. c(x)=0`。  
它将拉格朗日乘子更新与罚项思想结合，常通过增广拉格朗日函数迭代逼近 KKT 条件。

本目录的 MVP 聚焦“最小但可审计”的实现：
- 手写增广拉格朗日与梯度，不调用黑箱优化器；
- 内层子问题用梯度下降 + Armijo 回溯线搜索；
- 外层执行 `lambda <- lambda + rho*c(x)`，并根据约束收敛情况自适应增大 `rho`。

## R02

实现求解输入输出定义如下：
- 输入：
  - 目标函数 `f(x)` 及梯度 `grad f(x)`；
  - 约束函数 `c(x)` 及 Jacobian `J_c(x)`；
  - 初值 `x0` 与可选初始乘子 `lambda0`；
  - 迭代超参数（`rho0`, `tol_constraint`, `tol_aug_grad`, `max_outer_iter` 等）。
- 输出：
  - 近似最优解 `x*`；
  - 乘子估计 `lambda*`；
  - 目标值、约束残差范数、增广拉格朗日梯度范数；
  - 外层迭代轨迹与停止原因。

## R03

核心数学关系：

1. 原始问题：`min f(x), s.t. c(x)=0`。  
2. 增广拉格朗日函数：  
   `L_rho(x, lambda) = f(x) + lambda^T c(x) + (rho/2)||c(x)||_2^2`。  
3. 内层子问题：固定 `lambda_k, rho_k`，近似求解  
   `x_{k+1} ≈ argmin_x L_{rho_k}(x, lambda_k)`。  
4. 乘子更新：`lambda_{k+1} = lambda_k + rho_k * c(x_{k+1})`。  
5. KKT 近似检查：同时关注 `||c(x)||` 与 `||∇_x L_rho(x,lambda)||` 是否足够小。

## R04

算法高层流程：
1. 初始化 `x, lambda, rho`，并校验输入维度与数值有效性。
2. 外层迭代中，先解一个增广拉格朗日子问题（内层）。
3. 内层每步计算 `∇_x L_rho`，取负梯度方向并进行 Armijo 回溯线搜索。
4. 内层收敛后得到新的 `x`，计算约束残差和增广梯度范数。
5. 若残差与梯度都满足阈值，则整体收敛并退出。
6. 否则按公式更新 `lambda`。
7. 若约束收敛变慢（当前残差不明显下降），增大罚参数 `rho`。
8. 达到最大外层轮数或内层失败时，返回当前结果与状态信息。

## R05

核心数据结构：
- `SubproblemResult`：内层子问题返回结构。
  - `x`：内层结束点；
  - `aug_grad_norm`：内层结束时 `||∇_x L_rho||`；
  - `iterations`：内层迭代步数；
  - `converged` / `line_search_failed`：内层状态。
- `MultiplierResult`：外层总结果。
  - `x, lam, objective`；
  - `constraint_norm, aug_grad_norm`；
  - `iterations, converged, message`；
  - `history`（每轮外层日志）。
- `HistoryItem`：
  `(iter, objective, ||c||, ||grad L_rho||, rho, ||lambda||, inner_iters)`。

## R06

正确性要点：
- 增广梯度公式严格按  
  `∇_x L_rho = ∇f(x) + J_c(x)^T(lambda + rho*c(x))`  实现。
- 乘子更新使用标准公式 `lambda <- lambda + rho*c(x)`。
- 内层线搜索确保步长满足 Armijo 充分下降，避免盲目大步导致发散。
- 终止条件采用“双指标”：约束可行性 + 增广梯度驻点性，减少仅靠单一指标的误判。
- 示例提供已知解析解，可用相对误差验证实现有效性。

## R07

复杂度分析（设变量维度 `n`，约束维度 `m`，外层轮数 `T`，单轮内层步数 `S`）：
- 每次内层迭代主要成本：
  - 计算 `f, grad f, c, J_c`；
  - 向量矩阵组合 `J_c^T(lambda+rho*c)`，约 `O(mn)`。
- 单外层复杂度近似 `O(S * (C_eval + mn))`，`C_eval` 为函数/梯度/约束计算成本。
- 总体时间复杂度约 `O(T * S * (C_eval + mn))`。
- 空间复杂度：
  - 状态向量与 Jacobian 主体约 `O(n + mn)`；
  - 轨迹存储 `O(T)`。

## R08

边界与异常处理：
- `x0`、`lambda0` 必须是一维有限向量，且约束维度一致。
- `c(x)` 必须返回 1D 非空向量；`J_c(x)` 必须匹配 `(m, n)`。
- `rho0 > 0`、`rho_scale > 1`、`tol_* > 0`、迭代上限必须为正。
- 若出现非有限值（函数、约束、Jacobian、增广值），抛出 `ValueError`。
- 内层线搜索无法找到可接受步长时，返回失败状态并终止外层。

## R09

MVP 取舍：
- 仅依赖 `numpy`，控制依赖复杂度。
- 不使用 `scipy.optimize.minimize` 等黑箱求解器，保留源码可追踪性。
- 覆盖必要工程能力：
  - 输入校验；
  - 线搜索稳定化；
  - 历史日志；
  - 多样例自动断言。
- 未覆盖内容：
  - 不等式约束与互补条件；
  - 二阶子问题求解（如牛顿/拟牛顿内层）；
  - 大规模稀疏问题与并行加速。

## R10

`demo.py` 函数职责：
- `check_vector / ensure_constraint_vector / ensure_jacobian_shape`：输入与形状校验。
- `augmented_lagrangian`：计算 `L_rho(x,lambda)`。
- `augmented_gradient`：计算 `∇_x L_rho`。
- `solve_augmented_subproblem`：内层梯度下降 + Armijo 回溯。
- `multiplier_method`：外层乘子法主循环与罚参数调度。
- `print_history`：格式化打印迭代轨迹。
- `build_case_*`：构造固定测试问题。
- `run_case`：运行单例并输出误差统计。
- `main`：运行全部样例并做最终断言。

## R11

运行方式：

```bash
cd Algorithms/数学-约束优化-0383-乘子法
uv run python demo.py
```

脚本不读取交互输入，不依赖命令行参数。

## R12

输出字段说明：
- `Converged`：是否满足双收敛条件。
- `Stop reason`：停止原因（收敛、内层线搜索失败、或达到外层上限）。
- `Outer iterations`：外层轮数。
- `Final x`：最终原变量估计。
- `Final lambda`：最终乘子估计。
- `Final objective`：最终目标值。
- `Final ||c(x)||`：约束残差范数。
- `Final ||grad L_rho||`：增广拉格朗日梯度范数。
- 轨迹列：
  - `iter`：外层轮次；
  - `objective`：当前目标值；
  - `||c(x)||`：可行性误差；
  - `||grad L_rho||`：驻点误差；
  - `rho`：罚参数；
  - `||lambda||`：乘子范数；
  - `inner_iters`：该轮内层迭代步数。

## R13

内置最小测试集：
1. `Linear-Equality-Quadratic`
- 问题：`min (x1-1)^2+(x2-2)^2, s.t. x1+x2-2=0`
- 参考解：`x*=[0.5,1.5]`, `lambda*=[1.0]`
- 用途：线性等式约束下验证快速收敛与乘子更新正确性。

2. `Nonlinear-Constraint-Hyperbola`
- 问题：`min x1^2+x2^2, s.t. x1*x2-1=0`
- 从正初值出发参考解：`x*=[1,1]`, `lambda*=[-2]`
- 用途：验证非线性约束下的稳定性与可行性恢复能力。

## R14

关键参数与建议：
- `rho0`：初始罚参数。过小会导致约束收敛慢，过大可能使内层问题更难。
- `rho_scale`：罚参数放大倍率（本实现默认 `2.0`）。
- `rho_max`：罚参数上限，避免无限放大导致数值问题。
- `tol_constraint`：约束残差阈值。
- `tol_aug_grad`：增广梯度阈值。
- `max_outer_iter` / `max_inner_iter`：外层与内层迭代上限。

调参建议：
- 若约束残差长期不降，可增大 `rho0` 或 `rho_scale`；
- 若内层线搜索频繁失败，可放宽内层梯度阈值或减小增长速度。

## R15

方法对比：
- 对比纯罚函数法：
  - 乘子法通过 `lambda` 累积约束信息，通常不需把 `rho` 增大到极端值。
- 对比投影梯度法：
  - 投影法依赖可行域投影算子；乘子法仅需约束函数与 Jacobian。
- 对比 SQP：
  - SQP 通常收敛更快但实现更重；本 MVP 更轻量、可教学与可审计。

## R16

典型应用：
- 中低维等式约束优化问题；
- 有解析梯度和约束 Jacobian 的工程参数标定；
- 作为从“无约束优化”过渡到“约束优化”教学样例；
- 作为更高级方法（SQP/内点法）前的基线对照实现。

## R17

可扩展方向：
- 支持不等式约束（配合松弛变量或 ALM + 投影）；
- 内层换成牛顿/拟牛顿，提高收敛速度；
- 增加自适应线搜索和预条件策略；
- 接入自动微分框架自动构建 `grad` 与 `Jacobian`；
- 增加日志落盘、可视化曲线与批量基准测试。

## R18

`demo.py` 源码级算法流（8 步）：
1. `main` 构造两个固定问题，并调用 `run_case` 顺序执行。  
2. `run_case` 调用 `multiplier_method`，配置 `rho0/tol/max_iter` 等参数。  
3. `multiplier_method` 先初始化 `x, lambda, rho`，进入外层循环。  
4. 每轮外层调用 `solve_augmented_subproblem`，近似最小化 `L_rho(x,lambda)`。  
5. `solve_augmented_subproblem` 在每步内层计算 `∇_x L_rho`，沿负梯度方向做 Armijo 回溯并更新 `x`。  
6. 回到外层后，计算 `||c(x)||` 与 `||∇_x L_rho||`，记录到 `history`。  
7. 若满足阈值则收敛退出；否则执行 `lambda <- lambda + rho*c(x)`，并在约束下降不足时增大 `rho`。  
8. 所有样例运行完后，`main` 汇总最大残差与相对误差，若超阈值则抛异常，否则打印验证通过。  
