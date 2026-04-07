# 罚函数法

- UID: `MATH-0386`
- 学科: `数学`
- 分类: `约束优化`
- 源序号: `386`
- 目标目录: `Algorithms/数学-约束优化-0386-罚函数法`

## R01

罚函数法（Penalty Method）把“有约束优化”转成一系列“无约束优化”子问题。核心思想是：
- 原问题目标 `f(x)` 保留；
- 约束违背程度通过罚项加入目标；
- 逐步增大罚参数 `rho`，迫使迭代点靠近可行域。

本目录给出一个最小可运行 MVP：
- 外层：二次罚参数递增（`rho <- beta * rho`）；
- 内层：对每个 `Q_rho(x)` 用 `scipy.optimize.minimize(method="BFGS")` 求解；
- 示例同时包含等式约束与不等式约束，并打印每轮可行性收敛轨迹。

## R02

`demo.py` 求解的问题：

- `min f(x1,x2) = (x1-2)^2 + (x2-1)^2`
- `s.t. h(x) = x1 + x2 - 1 = 0`
- `g1(x) = -x1 <= 0`
- `g2(x) = -x2 <= 0`

可行域是线段 `x1+x2=1, x1>=0, x2>=0`，该问题的真解是 `x*=[1,0]`。

## R03

二次罚函数定义为：

- `Q_rho(x) = f(x) + (rho/2)*||h(x)||_2^2 + (rho/2)*||max(g(x),0)||_2^2`

其中：
- `h(x)=0` 是等式约束；
- `g(x)<=0` 是不等式约束；
- `max(g,0)` 只惩罚违背不等式的分量。

对应梯度（MVP 中显式实现）为：

- `grad Q_rho(x) = grad f(x) + rho * J_h(x)^T h(x) + rho * J_g(x)^T max(g(x),0)`

## R04

算法流程（外层-内层）：

1. 选择初值 `x0`（可不满足约束）。
2. 设初始罚参数 `rho0` 和增长因子 `beta>1`。
3. 固定当前 `rho`，构造无约束目标 `Q_rho(x)`。
4. 用 BFGS 求解该无约束子问题，得到新点 `x`。
5. 计算可行性指标：`||h(x)||_2` 与 `||max(g(x),0)||_2`。
6. 若 `max(两者) < tol`，则停止。
7. 否则增大罚参数 `rho <- beta * rho`，继续下一轮。

## R05

核心数据结构：

- `ConstrainedProblem`：统一封装问题定义。
- 字段包括：
  - `objective / objective_grad`
  - `eq_constraints / eq_jacobian`
  - `ineq_constraints / ineq_jacobian`
- `PenaltyIterRecord`：记录每轮外层信息：
  - `outer_iter, rho, x`
  - `f_raw, eq_norm, ineq_violation_norm`
  - `penalty_term, augmented_value`
  - `inner_iterations, inner_success`
- `history: list[PenaltyIterRecord]`：完整迭代轨迹。

## R06

正确性直觉：

- 当 `rho` 增大时，违反约束的代价快速上升，优化器会倾向于降低违约项。  
- 当内层子问题近似解得足够好，外层罚参数序列可把解推向可行域。  
- 本例中约束与目标都平滑，且真解已知，可通过 `||x-x_true||` 与约束残差直接验证。  
- 脚本额外调用 SLSQP 作为参考，避免只看单一实现的“自证正确”。

## R07

复杂度（记决策维度为 `n`，外层轮数为 `T`，第 `k` 轮 BFGS 迭代数为 `I_k`）：

- 外层总复杂度由内层求解决定，可写为 `sum_k O(BFGS(I_k, n))`。  
- 在密集实现下，BFGS 每步维护近似逆 Hessian 的代价常见为 `O(n^2)` 级。  
- 因此粗略可写为 `O(sum_k I_k * n^2)`；空间复杂度约 `O(n^2)`（BFGS 历史矩阵）。  
- 本条目是教学型小规模 MVP，重点是机制可解释，不追求大规模数值性能。

## R08

边界与异常处理：

- `x0` 不是一维、维度不匹配、含 `NaN/Inf`：抛 `ValueError`。  
- 问题函数/雅可比返回形状不一致：抛 `ValueError`。  
- 参数非法（如 `rho0<=0`、`penalty_growth<=1`、`tol<=0`）：抛 `ValueError`。  
- 罚目标出现非有限值：抛 `RuntimeError`。  
- 达到最大外层轮数仍未达到可行性阈值：抛 `RuntimeError`。

## R09

MVP 取舍：

- 采用 `numpy + scipy` 的最小依赖，不引入额外框架。  
- 内层使用 BFGS，但罚函数和梯度均在源码里显式写出，不是纯黑盒调用。  
- 示例维度固定为 2，仅用于演示“约束 -> 罚项 -> 外层收敛”机制。  
- 参考解（SLSQP）只做校验，不参与主算法路径。

## R10

`demo.py` 函数职责：

- `validate_problem`：检查问题定义、维度、有限性和雅可比形状。
- `penalty_components`：计算 `f`、残差范数、罚项和 `Q_rho`。
- `penalty_objective`：返回当前 `rho` 下的罚目标值。
- `penalty_gradient`：返回当前 `rho` 下的罚目标梯度。
- `quadratic_penalty_method`：外层罚参数循环 + 内层 BFGS 求解。
- `build_demo_problem`：构造固定演示问题与真解。
- `solve_reference_slsqp`：用 SLSQP 做约束原生求解作为参考。
- `print_history`：格式化输出外层迭代轨迹。
- `main`：组织运行并打印最终对比结果。

## R11

运行方式：

```bash
cd Algorithms/数学-约束优化-0386-罚函数法
uv run python demo.py
```

无需命令行参数，也不需要交互输入。

## R12

输出字段说明：

- `outer`：外层罚参数迭代编号。  
- `rho`：当前罚参数。  
- `f(x)`：原始目标值（不含罚项）。  
- `||h||2`：等式约束残差二范数。  
- `||g+||2`：不等式违约量 `max(g,0)` 的二范数。  
- `penalty`：当前罚项值。  
- `Q_rho`：增广目标（原目标 + 罚项）。  
- `inner`：该轮内层 BFGS 迭代步数。  
- `ok`：该轮内层求解器是否报告 `success`。  
- 末尾还会输出 `x*`、`f(x*)`、约束残差、与真解误差、以及 SLSQP 参考对照。

## R13

建议测试集：

- 基础正确性（已内置）：当前二维问题应收敛到接近 `[1,0]`。  
- 初值鲁棒性：把 `x0` 改为更远点（如 `[10,-7]`）检查是否仍能收敛。  
- 参数敏感性：尝试不同 `penalty_growth`（如 `5`、`20`）观察外层轮数变化。  
- 异常路径：将 `penalty_growth=1` 或 `feasibility_tol<=0`，应触发参数异常。

## R14

主要可调参数（`quadratic_penalty_method`）：

- `rho0`：初始罚参数。  
- `penalty_growth`：每轮放大倍率。  
- `feasibility_tol`：外层可行性停止阈值。  
- `max_outer_iters`：最大外层轮数。  
- `inner_gtol`：内层 BFGS 梯度停止阈值。  
- `inner_maxiter`：内层最大迭代步数。

调参建议：

- 约束收敛慢：增大 `penalty_growth` 或增加 `max_outer_iters`。  
- 子问题难解：放宽 `inner_gtol` 或增大 `inner_maxiter`。  
- 数值病态（罚参数过大引发）：减小 `penalty_growth`。

## R15

与其他约束优化思路对比：

- 对比投影法：
  - 投影法每步都需投影回可行域；
  - 罚函数法可从不可行点出发，工程实现常更直接。
- 对比内点法：
  - 内点法偏“保持可行域内部”；
  - 罚函数法通过惩罚逐步逼近可行域，常更易套到一般问题。
- 对比增广拉格朗日：
  - 纯罚函数法常需要较大 `rho`，可能带来病态；
  - 增广拉格朗日通常在数值稳定性和收敛速度上更强。

## R16

典型应用：

- 工程参数标定（物理边界 + 守恒等式约束）。  
- 机器学习超参数/结构搜索中的软约束处理。  
- 需要快速把“已有无约束求解器”迁移到“带约束版本”的场景。  
- 算法教学：展示从约束问题到无约束序列的转换过程。

## R17

可扩展方向：

- 把不光滑 `max(g,0)` 替换为平滑近似（如 softplus 型）以提升内层稳定性。  
- 内层从 BFGS 换成 L-BFGS-B / trust-constr，并比较收敛行为。  
- 增加自动梯度后端（如 PyTorch）支持更复杂目标。  
- 加入批量实验脚本，输出参数-收敛曲线。  
- 扩展成增广拉格朗日法，减少对超大罚参数的依赖。

## R18

`demo.py` 源码级流程（8 步）：

1. `build_demo_problem` 定义目标函数、约束函数及其雅可比，并给出不可行初值 `x0`。  
2. `main` 调用 `quadratic_penalty_method`，设置 `rho0`、`penalty_growth`、`feasibility_tol`。  
3. 每轮外层固定 `rho`，构造子问题 `min Q_rho(x)`。  
4. 内层使用 `scipy.optimize.minimize(method="BFGS")`，但目标值由 `penalty_objective` 逐项计算（`f + 罚项`）。  
5. 梯度由 `penalty_gradient` 显式计算：`grad f + rho*J_h^T h + rho*J_g^T max(g,0)`，不是数值差分黑盒。  
6. 子问题结束后，`penalty_components` 计算 `||h||2` 与 `||g+||2`，写入 `PenaltyIterRecord`。  
7. 若 `max(||h||2, ||g+||2) < feasibility_tol` 则终止；否则执行 `rho <- penalty_growth * rho` 进入下一轮。  
8. `main` 打印轨迹和最终解，并调用 `solve_reference_slsqp` 给出约束原生求解结果作交叉校验。
