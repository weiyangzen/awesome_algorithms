# 牛顿法 (优化)

- UID: `MATH-0348`
- 学科: `数学`
- 分类: `优化`
- 源序号: `348`
- 目标目录: `Algorithms/数学-优化-0348-牛顿法_(优化)`

## R01

牛顿法（Newton Method for Optimization）是无约束光滑优化中的经典二阶方法。  
它在每一轮利用梯度和 Hessian（海森矩阵）构造局部二次模型，并据此求一个近似最优更新方向，常见于“比一阶法更快收敛”的中小规模问题。

本目录给出一个可运行、可审计的 MVP：
- 手写牛顿方向与阻尼策略（不是黑箱 `scipy.optimize.minimize`）；
- Armijo 回溯线搜索保证下降；
- 固定两组样例自动运行并做误差校验。

## R02

本实现求解问题：
- 输入：
  - 目标函数 `f(x)`；
  - 梯度函数 `grad(x)`；
  - Hessian 函数 `hess(x)`；
  - 初值 `x0`；
  - 收敛参数 `tol`、`max_iter` 等。
- 输出：
  - 近似最优点 `x*`；
  - 最终 `f(x*)` 与 `||grad(x*)||`；
  - 迭代轨迹 `[(iter, f, grad_norm, alpha, damping, step_norm), ...]`。

`demo.py` 内置固定样例，无需交互输入。

## R03

核心数学关系：

1. 目标的二阶近似模型（在 `x_k` 附近）：
   `m_k(p) = f(x_k) + g_k^T p + 0.5 * p^T H_k p`，
   其中 `g_k = grad f(x_k)`，`H_k = hess f(x_k)`。
2. 令近似模型一阶最优条件为 0：
   `H_k p_k = -g_k`。
3. 更新形式：
   `x_(k+1) = x_k + alpha_k * p_k`，`alpha_k` 由 Armijo 回溯线搜索获得。
4. 当 Hessian 退化或方向不下降时，本实现使用阻尼修正：
   `(H_k + lambda I) p_k = -g_k`，必要时退回 `p_k = -g_k`。

## R04

算法流程（高层）：
1. 校验初值和超参数合法性。
2. 计算初始 `f(x0)`、`g(x0)`。
3. 若 `||g||` 已达阈值，直接返回。
4. 计算 Hessian，并通过阻尼线性系统求牛顿方向。
5. 若方向不满足下降条件，回退到最速下降方向。
6. 运行 Armijo 回溯线搜索确定步长 `alpha`。
7. 更新 `x`，记录本轮轨迹（函数值、梯度范数、步长、阻尼、位移）。
8. 按梯度阈值或步长阈值判断停止，否则进入下一轮。

## R05

核心数据结构：
- `HistoryItem = (iter, f_x, grad_norm, alpha, damping, step_norm)`：单轮日志。
- `NewtonResult`：封装最终结果与计数信息：
  - `x, f, grad_norm, iterations, converged, message`；
  - `function_evals, gradient_evals, hessian_evals`；
  - `history`。
- `cases`：`main` 中的固定测试配置（Rosenbrock 与 SPD 二次型）。

## R06

正确性要点：
- 牛顿方向来自二阶模型驻点条件 `H p = -g`，不是经验更新。
- 阻尼与下降检查避免了 Hessian 非正定或病态时的错误方向。
- Armijo 回溯保证每步满足“充分下降”条件。
- 终止条件使用双判据：梯度阈值 + 步长阈值，避免单指标误判。
- 示例中将结果与参考解对比（Rosenbrock `[1,1]`、SPD 线性方程解），可验证实现有效性。

## R07

复杂度分析（变量维度 `n`，迭代轮数 `T`）：
- 每轮主要成本：
  - 梯度计算：记为 `C_grad`；
  - Hessian 计算：记为 `C_hess`；
  - 线性系统求解 `np.linalg.solve`：`O(n^3)`（稠密场景）；
  - 回溯线搜索若试探 `B` 次，额外约 `B * C_f`。
- 总时间复杂度近似：`O(T * (C_hess + n^3 + C_grad + B*C_f))`。
- 空间复杂度：
  - 状态向量与矩阵主存储约 `O(n^2)`；
  - 若保留全轨迹，额外 `O(T)`。

## R08

边界与异常处理：
- `x0` 必须是一维有限向量。
- `tol/max_iter/min_step/max_backtracks` 等参数非法时抛 `ValueError`。
- 若初始 `f/grad` 非有限，立即报错。
- 若 Hessian 形状错误或含非有限值，停止并返回失败消息。
- 若线搜索无法找到可接受步长，停止并返回 `line search failed`。
- 若梯度更新后出现非有限值，停止并返回对应消息。

## R09

MVP 取舍：
- 仅依赖 `numpy`，避免额外框架依赖。
- 不调用 `scipy.optimize.minimize`，核心流程完全源码可追踪。
- 覆盖“最小但真实”的稳健工程要素：
  - 阻尼方向、回溯线搜索、方向兜底、日志记录、结果自检。
- 不实现信赖域、稀疏线性代数、大规模分布式训练等重型扩展。

## R10

`demo.py` 主要函数职责：
- `check_vector`：检查输入向量合法性。
- `rosenbrock / rosenbrock_grad / rosenbrock_hess`：非凸样例三件套。
- `make_quadratic_problem`：生成 SPD 二次目标及其导数/Hessian。
- `modified_newton_direction`：阻尼牛顿方向求解与下降性检查。
- `armijo_backtracking`：回溯线搜索。
- `newton_optimize`：牛顿主循环。
- `print_history`：格式化打印迭代轨迹。
- `run_case`：运行单个样例并输出误差统计。
- `main`：组织样例、汇总结果并做最终断言。

## R11

运行方式：

```bash
cd Algorithms/数学-优化-0348-牛顿法_(优化)
python3 demo.py
```

脚本不读取命令行参数，也不会请求交互输入。

## R12

输出字段说明：
- `Converged`：是否满足收敛条件。
- `Stop reason`：停止原因（梯度阈值、步长阈值、线搜索失败等）。
- `Iterations`：成功更新的迭代轮数。
- `Final x / Final f(x) / Final ||grad||`：最终状态。
- `Function/Gradient/Hessian evals`：函数、梯度、Hessian 评估次数。
- 轨迹列：
  - `iter`：迭代编号；
  - `f(x_k)`：本轮更新后的目标值；
  - `||grad||`：本轮更新后的梯度范数；
  - `alpha`：线搜索步长；
  - `damping`：阻尼因子；
  - `||step||`：参数步长范数。
- `Summary`：跨样例收敛状态、最大梯度范数与最大相对误差。

## R13

最小测试集（已内置）：
1. `Rosenbrock-2D`
- 初值：`[-1.2, 1.0]`
- 参考解：`[1.0, 1.0]`
- 目标：验证非凸场景中阻尼牛顿 + 线搜索的稳定收敛。

2. `SPD-Quadratic-3D`
- 目标：`0.5*x^T*A*x - b^T*x`（`A` 对称正定）
- 参考解：`np.linalg.solve(A, b)`
- 目标：验证二阶信息下高精度与快速收敛。

可补充异常测试：
- 非有限初值；
- 非法 Hessian 返回形状；
- 极端参数（如 `tol<=0`）触发错误路径。

## R14

关键参数与建议：
- `tol`（默认 `1e-8`）：收敛阈值，越小越严格。
- `max_iter`（默认 `100` 或主函数 `120`）：最大迭代轮数。
- `base_damping`（默认 `1e-8`）：阻尼起点，Hessian 不稳时会逐步放大。
- `max_damping_trials`（默认 `8`）：阻尼尝试次数上限。
- `c1`（默认 `1e-4`）：Armijo 常数。
- `line_search_shrink`（默认 `0.5`）：回溯缩放率。
- `min_step`（默认 `1e-12`）：最小步长保护。

调参实践：
- 若震荡/不收敛，先增大 `base_damping` 或减小 `line_search_shrink`；
- 若收敛太慢，适度放宽阻尼或提高 `max_damping_trials`。

## R15

方法对比：
- 对比梯度下降：
  - 牛顿法使用二阶信息，局部收敛速度通常显著更快；
  - 但单轮成本更高（需 Hessian 与线性系统求解）。
- 对比 BFGS/L-BFGS：
  - BFGS 系列主要依赖一阶信息近似二阶曲率；
  - 牛顿法直接用 Hessian，局部精度更高但成本更重。
- 对比“纯牛顿步不线搜索”：
  - 本实现加 Armijo 回溯，远离最优点时更稳健。

## R16

典型应用场景：
- 中低维、可计算 Hessian 的无约束光滑优化。
- 参数估计、曲线拟合中的高精度求解阶段。
- 作为教学/验证基线：用于对比一阶法与拟牛顿法。
- 工程中可作为“后期精修优化器”（在较好初值附近快速收敛）。

## R17

可扩展方向：
- 引入信赖域牛顿法，提高非凸区域稳定性。
- 引入稀疏 Hessian 与稀疏线性求解，支持更高维问题。
- 增加强 Wolfe 线搜索替代简单 Armijo。
- 对接自动微分（如 PyTorch/JAX）自动构造梯度与 Hessian（或 Hessian 向量积）。
- 增加日志落盘和收敛曲线可视化。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 构造两组固定样例：`Rosenbrock-2D` 和 `SPD-Quadratic-3D`，并设置 `tol/max_iter`。  
2. 每个样例由 `run_case` 调用 `newton_optimize`，准备结果打印与参考解误差统计。  
3. `newton_optimize` 先检查 `x0` 与超参数，再计算初始 `f(x)`、`grad(x)`。  
4. 每轮迭代先取 Hessian；`modified_newton_direction` 尝试解 `(H + lambda I)p = -g`，逐步增加 `lambda` 保证下降方向。  
5. 若阻尼后方向仍不可靠，则退回最速下降 `p = -g`，确保 `g^T p < 0`。  
6. `armijo_backtracking` 从 `alpha=1` 开始回溯，找到满足 Armijo 条件的步长。  
7. 更新 `x -> x + alpha p`，重算梯度，记录 `(iter, f, ||grad||, alpha, damping, ||step||)` 到 `history`。  
8. 若满足梯度阈值或步长阈值则返回；`run_case/main` 输出收敛信息、误差与总体验证结论。  
