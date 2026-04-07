# 障碍函数法

- UID: `MATH-0387`
- 学科: `数学`
- 分类: `约束优化`
- 源序号: `387`
- 目标目录: `Algorithms/数学-约束优化-0387-障碍函数法`

## R01

障碍函数法（Barrier Method）把不等式约束 `g_i(x) <= 0` 吸收到目标中，通过对不可行边界设置“趋于无穷”的障碍项，让迭代始终留在可行域内部。

本目录给出一个最小可运行 MVP：
- 外层逐步增大 `t`（`t <- mu * t`），让障碍问题逐渐逼近原始约束问题；
- 内层在固定 `t` 下最小化 `phi_t(x)`；
- 内层使用“阻尼牛顿 + 回溯线搜索”，显式保证每步都严格可行。

## R02

`demo.py` 的演示问题：

- `min f(x1,x2) = (x1-1.5)^2 + (x2-0.5)^2`
- `s.t. g1(x)=x1+x2-1 <= 0`
- `g2(x)=-x1 <= 0`
- `g3(x)=-x2 <= 0`

可行域是三角形 `{x1>=0, x2>=0, x1+x2<=1}`，真解在边界点 `x*=[1,0]`。

## R03

对仅含不等式约束的问题，采用对数障碍：

- `phi_t(x) = t * f(x) - sum_i log(-g_i(x))`，定义域为 `g_i(x) < 0`。

当 `x` 接近边界 `g_i(x)=0` 时，`-log(-g_i(x))` 会急剧增大，从而阻止迭代越界。

对应梯度（代码中显式实现）：

- `grad phi_t(x) = t * grad f(x) - J_g(x)^T * (1 / g(x))`

其中 `1 / g(x)` 是按分量取倒数。

## R04

算法流程：

1. 选严格可行初值 `x0`（必须满足所有 `g_i(x0) < 0`）。
2. 设 `t=t0`、放大系数 `mu>1`。
3. 固定 `t`，求解子问题 `min phi_t(x)`。
4. 内层迭代构造牛顿方向 `d = -H^{-1} grad phi_t(x)`。
5. 用回溯线搜索同时检查：
   - 候选点仍严格可行；
   - 满足 Armijo 下降条件。
6. 子问题收敛后记录当前 `x_t`。
7. 用近似对偶间隙 `m/t`（`m` 为不等式个数）作为外层停止指标。
8. 若 `m/t < tol` 则结束，否则 `t <- mu*t` 继续。

## R05

核心数据结构：

- `InequalityConstrainedProblem`：统一封装问题定义。
  - `objective / objective_grad`
  - `ineq_constraints / ineq_jacobian`
- `BarrierIterRecord`：每轮外层记录。
  - `outer_iter, t, x`
  - `f_raw, max_g, barrier_value`
  - `surrogate_duality_gap`
  - `inner_iterations, inner_grad_norm`
- `history: list[BarrierIterRecord]`：完整轨迹。

## R06

正确性直觉：

- 障碍项在边界附近发散，保证内层迭代留在可行域内部；
- 随着 `t` 增大，`t*f(x)` 的权重上升，障碍问题解会靠近原问题最优边界；
- 对凸目标 + 仿射约束场景，解轨迹（central path）具有稳定逼近性；
- 示例里真解已知，可直接比较 `||x-x_true||` 与 `max g(x)`。

## R07

复杂度（记维度 `n`，约束数 `m`，外层轮数 `T`，第 `k` 轮内层步数 `I_k`）：

- 每个内层步需构造 `H = t∇²f + J^T diag(1/g^2) J` 并求解线性方程；
- 密集实现下单步主成本可近似为 `O(mn^2 + n^3)`；
- 外层总成本约 `O(sum_k I_k * (mn^2 + n^3))`；
- 空间复杂度约 `O(n^2 + mn)`（Hessian 与 Jacobian）。

本条目偏教学型小规模实现，重点是机制透明而非大规模性能。

## R08

边界与异常处理：

- `x0` 维度错误、含 `NaN/Inf`：抛 `ValueError`；
- `x0` 不是严格可行点（存在 `g_i(x0) >= 0`）：抛 `ValueError`；
- 约束或雅可比形状不匹配：抛 `ValueError`；
- 外层/内层参数非法（如 `mu<=1`、`tol<=0`）：抛 `ValueError`；
- 回溯线搜索无法找到可行下降步：抛 `RuntimeError`；
- 达到最大外层轮数仍未满足 `m/t` 阈值：抛 `RuntimeError`。

## R09

MVP 取舍：

- 采用 `numpy` 主实现，避免大框架复杂度；
- 内层没有直接调用黑盒约束优化器，而是显式实现牛顿方向与可行性保持线搜索；
- 只支持不等式约束（标准 log-barrier 形态）；
- 额外给出 SLSQP 参考解，仅用于结果对照。

## R10

`demo.py` 函数职责：

- `validate_problem`：检查维度、有限性、严格可行性与雅可比形状；
- `barrier_objective`：计算 `phi_t(x)`；
- `barrier_gradient`：计算 `grad phi_t(x)`；
- `barrier_hessian`：计算障碍子问题 Hessian；
- `solve_barrier_subproblem`：内层阻尼牛顿 + 回溯线搜索；
- `log_barrier_method`：外层 `t` 更新与停止判据控制；
- `build_demo_problem`：构造演示问题、初值与已知解；
- `solve_reference_slsqp`：约束原生求解参考（非主路径）；
- `print_history`：格式化打印外层收敛轨迹；
- `main`：组织运行并输出对比指标。

## R11

运行方式：

```bash
cd Algorithms/数学-约束优化-0387-障碍函数法
uv run python demo.py
```

无需命令行参数，也不需要交互输入。

## R12

输出字段说明：

- `outer`：外层编号；
- `t`：当前障碍参数权重；
- `f(x)`：原始目标值；
- `max g(x)`：最大不等式值（`<=0` 为可行）；
- `phi_t(x)`：障碍子问题目标值；
- `m/t`：近似对偶间隙；
- `inner`：该轮内层迭代步数；
- `||grad||`：内层结束时梯度范数。

末尾还会打印 `x*`、`f(x*)`、可行性指标，以及与真解和 SLSQP 参考的误差对比。

## R13

建议测试集：

- 基础正确性：默认参数应收敛到接近边界解 `[1,0]`；
- 初值鲁棒性：尝试不同严格可行初值（如 `[0.2,0.2]`）；
- 参数敏感性：修改 `mu`（如 `5`、`12`）观察外层轮数变化；
- 失败路径：将 `x0` 改成边界点（如 `[1,0]`）应触发严格可行性异常。

## R14

主要可调参数（`log_barrier_method`）：

- `t0`：初始障碍权重；
- `mu`：外层放大因子；
- `duality_gap_tol`：外层停止阈值（`m/t`）；
- `max_outer_iters`：最大外层轮数；
- `inner_tol`：内层梯度停止阈值；
- `max_inner_iters`：内层最大迭代步数；
- `line_search_alpha / line_search_beta`：回溯线搜索参数。

调参建议：

- 外层收敛慢：增大 `mu` 或提高 `max_outer_iters`；
- 内层步数多：适度放宽 `inner_tol`；
- 线搜索频繁缩步：减小 `alpha` 或将初值选得离边界更远。

## R15

与其他约束优化方法对比：

- 对比罚函数法：
  - 罚函数法可从不可行点启动；
  - 障碍函数法要求严格可行初值，但天然保持可行内部。
- 对比投影法：
  - 投影法每步可能需要额外投影算子；
  - 障碍法通过目标塑形避免显式投影。
- 对比原始-对偶内点法：
  - 本实现是简化版 barrier path；
  - 完整原始-对偶法通常收敛更快、数值更强。

## R16

典型应用：

- 凸工程设计中的线性不等式可行域约束；
- 需要全过程保持内部可行性的路径规划子问题；
- 作为内点法教学入门，理解 central path 与 `m/t` 停止准则；
- 快速搭建“可解释”约束优化基线原型。

## R17

可扩展方向：

- 换成原始-对偶内点牛顿系统，进一步提升收敛与稳健性；
- 扩展到“等式 + 不等式”场景（如消元或 KKT 子系统）；
- 引入自动微分后端（PyTorch）处理复杂目标；
- 增加批量实验脚本，输出参数-收敛曲线；
- 迁移到原始-对偶内点法以提升稳健性。

## R18

`demo.py` 源码级流程（8 步）：

1. `build_demo_problem` 构造凸目标与 3 个仿射不等式，给出严格可行初值 `x0`。  
2. `main` 调用 `log_barrier_method`，设置 `t0`、`mu`、`duality_gap_tol` 等超参数。  
3. `validate_problem` 在入口检查维度、有限性、雅可比形状，并强制 `g(x0)<0`。  
4. 每轮外层固定 `t`，进入 `solve_barrier_subproblem` 最小化 `phi_t(x)=t f(x)-sum log(-g_i)`。  
5. 内层每步由 `barrier_gradient` 与 `barrier_hessian` 计算牛顿方向 `d=-H^{-1}∇phi_t`。  
6. 回溯线搜索先检查候选点仍在严格内部，再检查 Armijo 下降，保证稳定前进。  
7. 内层结束后记录 `f(x)`、`max g(x)`、`phi_t(x)`、`m/t`、步数和梯度范数到 `history`。  
8. 外层依据 `m/t` 判断停止；若未达阈值则更新 `t <- mu*t`，最终输出解并与 SLSQP 参考对照。
