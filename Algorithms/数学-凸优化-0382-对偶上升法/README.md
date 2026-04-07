# 对偶上升法

- UID: `MATH-0382`
- 学科: `数学`
- 分类: `凸优化`
- 源序号: `382`
- 目标目录: `Algorithms/数学-凸优化-0382-对偶上升法`

## R01

对偶上升法（Dual Ascent）用于求解带约束的凸优化问题：  
把约束并入拉格朗日函数后，先对原变量做极小化得到对偶函数，再在对偶变量上做梯度上升。

本条目给出一个最小可运行 MVP，聚焦最经典场景：
- 原问题是“等式约束 + 强凸二次目标”；
- 每轮先闭式求解 `x(lambda)`，再更新 `lambda`；
- 输出原目标、对偶目标、可行性残差和对偶步长。

## R02

本目录实现的问题定义：

- 原问题（primal）：
  - `min_x 0.5 * x^T Q x + c^T x`
  - `s.t. A x = b`
- 其中：
  - `Q in R^(n*n)` 为对称正定；
  - `A in R^(m*n)`，`c in R^n`，`b in R^m`。
- 输出：
  - 近似最优解 `x*`、`lambda*`；
  - 迭代轨迹 `[(k, primal_obj, dual_obj, ||A x - b||, ||delta_lambda||), ...]`；
  - 与 KKT 线性系统参考解的误差对照。

## R03

关键数学关系：

1. 拉格朗日函数：
   `L(x, lambda) = 0.5*x^TQx + c^Tx + lambda^T(Ax-b)`。
2. 固定 `lambda` 时对 `x` 极小化，一阶条件：
   `Qx + c + A^T lambda = 0`。  
   因而有闭式：
   `x(lambda) = -Q^{-1}(c + A^T lambda)`。
3. 对偶函数：
   `g(lambda) = inf_x L(x, lambda) = L(x(lambda), lambda)`。
4. 对偶梯度：
   `nabla g(lambda) = A x(lambda) - b`。
5. 对偶上升更新：
   `lambda_{k+1} = lambda_k + alpha * (A x(lambda_k) - b)`。

## R04

算法高层流程：

1. 检查 `Q` 对称正定、`A/c/b` 维度一致且输入有限。  
2. 预计算 `Q^{-1}`。  
3. 若未指定步长，估计 `L = lambda_max(A Q^{-1} A^T)`，并设 `alpha = step_scale / L`。  
4. 循环：
   - 计算 `x(lambda_k)`；
   - 计算可行性残差 `r_k = A x(lambda_k) - b`；
   - 计算原目标/对偶目标并记录历史；
   - 若 `||r_k||` 足够小则停止；
   - 更新 `lambda_{k+1} = lambda_k + alpha * r_k`。
5. 输出估计解与完整轨迹。

## R05

核心数据结构：

- `HistoryItem = (iter, primal_obj, dual_obj, residual_norm, dual_step_norm)`  
  含义：
  - `iter`: 迭代编号；
  - `primal_obj`: `f(x_k)`；
  - `dual_obj`: `g(lambda_k)`；
  - `residual_norm`: `||A x_k - b||`；
  - `dual_step_norm`: `||lambda_{k+1} - lambda_k||`。
- `history: list[HistoryItem]`：完整收敛轨迹。
- `cases: list[dict]`：`main` 中固定的两个可复现实验用例。

## R06

正确性与可验证性要点：

- `Q` 正定保证 `x(lambda)` 闭式解唯一。  
- `g(lambda)` 为光滑凹函数，其梯度正是约束残差。  
- 对偶上升在“凹函数 + 合理步长”下会把残差压到 0。  
- 参考解通过 KKT 系统直接求得：
  - `[[Q, A^T], [A, 0]] [x; lambda] = [-c; b]`；
  - 脚本将估计解与该参考解逐项对比。

## R07

复杂度分析（稠密矩阵）：

- 预处理：
  - `Q^{-1}` 计算约 `O(n^3)`；
  - `A Q^{-1} A^T` 及最大特征值估计约 `O(m n^2 + m^3)`。
- 每轮迭代：
  - 计算 `x(lambda)` 约 `O(n^2 + mn)`；
  - 计算残差 `A x - b` 约 `O(mn)`。
- 总时间复杂度：`O(n^3 + T*(n^2 + mn))`。  
- 空间复杂度：`O(n^2 + mn + T)`（含完整轨迹）。

## R08

边界与异常处理：

- `Q` 非方阵/非对称/非正定：抛 `ValueError`。  
- `A/c/b/lam0` 维度不匹配：抛 `ValueError`。  
- 输入中含 `nan/inf`：抛 `ValueError`。  
- `tol <= 0`、`max_iter <= 0`、`step_scale <= 0` 或 `step_size <= 0`：抛 `ValueError`。  
- 对偶迭代出现非有限值：抛 `RuntimeError`。  
- 超过 `max_iter` 未收敛：抛 `RuntimeError`。

## R09

MVP 取舍说明：

- 只实现“等式约束二次规划”的对偶上升，不扩展到不等式约束。  
- 核心更新完全手写，不调用 `scipy.optimize.minimize` 黑盒。  
- 用固定默认步长公式（由 `L` 估计）保证简单可复现。  
- 保留细粒度迭代日志，便于验证与教学。

## R10

`demo.py` 函数职责：

- `check_vector/check_matrix/check_spd_matrix`：输入与数值合法性检查。  
- `primal_objective`：计算原目标。  
- `primal_minimizer_given_lambda`：给定 `lambda` 的闭式 `x(lambda)`。  
- `lagrangian_value`：计算 `L(x, lambda)`（用于对偶目标值）。  
- `estimate_dual_gradient_lipschitz`：估计步长相关常数 `L`。  
- `solve_kkt_reference`：解 KKT 系统生成参考真值。  
- `dual_ascent_equality_qp`：对偶上升主循环。  
- `run_case/main`：组织样例、打印轨迹、输出汇总指标。

## R11

运行方式（无交互）：

```bash
cd Algorithms/数学-凸优化-0382-对偶上升法
uv run python demo.py
```

脚本内置样例数据，不读取外部参数，不请求用户输入。

## R12

输出字段说明：

- `primal_obj`：当前原目标值 `f(x_k)`。  
- `dual_obj`：当前对偶目标值 `g(lambda_k)`。  
- `||A x - b||`：约束残差范数（越接近 0 越可行）。  
- `||delta_lambda||`：对偶变量步长。  
- `x* estimate / reference`：估计原解与 KKT 参考解。  
- `lambda estimate / reference`：估计对偶解与 KKT 参考解。  
- `primal objective gap`：原目标误差。  
- `absolute duality gap surrogate`：`|f(x)-g(lambda)|`，用于诊断收敛。  
- `Summary`：跨样例最大误差与通过标记。

## R13

建议最小测试集（已内置）：

- `3 variables / 2 equality constraints`：基础正确性验证。  
- `4 variables / 2 equality constraints`：更高维稳健性验证。

建议补充异常测试：
- 非正定 `Q`（应报错）；  
- `A` 列数与 `Q` 维度不一致（应报错）；  
- 人工给 `step_size <= 0`（应报错）。

## R14

可调参数：

- `tol`：残差停止阈值（默认 `1e-10`）。  
- `max_iter`：最大迭代轮数（默认 `10000`）。  
- `step_scale`：自动步长比例，`alpha = step_scale / L`（默认 `1.0`）。  
- `step_size`：可手动覆盖自动步长。  
- `print_history(..., max_lines)`：控制打印行数。

调参建议：
- 收敛慢时可适当增大 `step_scale`（保持正值并注意稳定性）；  
- 数值震荡时减小 `step_scale`。

## R15

方法对比：

- 对比增广拉格朗日 / ADMM：
  - 对偶上升实现更轻量，但对步长更敏感；
  - 增广项通常更稳健，但每轮子问题更复杂。  
- 对比原始-对偶内点法：
  - 内点法在高精度时通常更快；
  - 对偶上升更适合教学、原型和分解思路演示。  
- 对比直接 KKT 一次求解：
  - KKT 直接法一次到位，但不体现迭代机制；
  - 对偶上升可观察约束逐步满足过程。

## R16

典型应用场景：

- 网络流/资源分配中的拉格朗日松弛子问题。  
- 大规模问题中的分解协调（主问题更新对偶变量）。  
- 约束优化课程中讲解“对偶函数 + 梯度上升”机制。  
- 构建更复杂算法（增广拉格朗日、ADMM）前的基线模块。

## R17

可扩展方向：

- 扩展到不等式约束并加入投影（投影对偶上升）。  
- 引入自适应步长或回溯线搜索，减少手工调参。  
- 使用共轭梯度近似 `Q^{-1}`，提升大规模稀疏场景效率。  
- 加入增广项形成 ALM，并与 ADMM 做统一 benchmark。  
- 把历史轨迹输出到 CSV 并绘制收敛曲线。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 定义两个固定二次规划样例及统一参数 `tol/max_iter/step_scale`。  
2. `run_case` 调用 `dual_ascent_equality_qp` 执行对偶上升，并打印迭代历史。  
3. `dual_ascent_equality_qp` 先做 `Q/A/c/b/lam0` 的维度和数值检查。  
4. 计算 `Q^{-1}`，并用 `estimate_dual_gradient_lipschitz` 得到 `L`，自动设置 `alpha = step_scale/L`。  
5. 每轮根据 `lambda_k` 用闭式 `x_k = -Q^{-1}(c + A^T lambda_k)` 求原变量最小化点。  
6. 计算残差 `r_k = A x_k - b`、原目标 `f(x_k)`、对偶目标 `g(lambda_k)=L(x_k,lambda_k)` 并记录。  
7. 若 `||r_k||` 达阈值则停止，否则按 `lambda_{k+1} = lambda_k + alpha * r_k` 更新对偶变量。  
8. `run_case` 额外调用 `solve_kkt_reference` 求参考解，汇总 `x/lambda` 相对误差、残差、原目标差与通过标志。
