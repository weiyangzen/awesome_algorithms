# 共轭梯度法 (优化)

- UID: `MATH-0344`
- 学科: `数学`
- 分类: `优化`
- 源序号: `344`
- 目标目录: `Algorithms/数学-优化-0344-共轭梯度法_(优化)`

## R01

共轭梯度法（Conjugate Gradient, CG）在优化视角下可表述为：
对对称正定（SPD）二次目标函数
`f(x) = 0.5 * x^T A x - b^T x`
进行无约束最小化。

它比“最速下降法”多利用了历史方向信息，使搜索方向在 `A`-内积下两两共轭，通常能显著减少迭代轮数。

## R02

本目录 MVP 解决的问题：
- 输入：`A (n*n, SPD)`、`b (n,)`、`x0 (n,)`、`tol`、`max_iter`。
- 目标：最小化 `f(x)=0.5*x^T*A*x-b^T*x`。
- 输出：
  - 近似最优解 `x*`；
  - 迭代历史 `[(iter, f(x_k), ||grad||, alpha, beta, ||step||), ...]`；
  - 与参考解 `np.linalg.solve(A, b)` 的误差对比。

`demo.py` 使用固定样例，运行时无需交互输入。

## R03

核心数学关系：
1. 梯度：`g_k = grad f(x_k) = A x_k - b`。
2. 残差与梯度关系：`r_k = b - A x_k = -g_k`。
3. 初始化：`p_0 = r_0`。
4. 步长：`alpha_k = (r_k^T r_k) / (p_k^T A p_k)`。
5. 更新：
   - `x_{k+1} = x_k + alpha_k p_k`
   - `r_{k+1} = r_k - alpha_k A p_k`
6. 共轭系数：`beta_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)`。
7. 方向更新：`p_{k+1} = r_{k+1} + beta_k p_k`。

在 SPD 场景中，上述线性 CG 兼具数值稳定性和可解释性。

## R04

算法高层流程：
1. 校验 `A` 为方阵、对称、正定，且输入向量维度匹配。
2. 计算初始残差 `r=b-Ax`，令 `p=r`。
3. 计算 `p^T A p`，得到步长 `alpha`。
4. 更新 `x` 与 `r`。
5. 基于新旧残差计算 `beta`，更新方向 `p`。
6. 记录迭代日志（目标值、梯度范数、步长等）。
7. 若满足收敛阈值（梯度或步长），结束。
8. 否则进入下一轮，直到 `max_iter` 或收敛。

## R05

核心数据结构：
- `HistoryItem = (iter, f_x, grad_norm, alpha, beta, step_norm)`：
  - `iter`：迭代编号；
  - `f_x`：当前目标值；
  - `grad_norm`：`||grad f(x)||`；
  - `alpha`：当前轮步长；
  - `beta`：当前轮共轭系数；
  - `step_norm`：`||x_{k+1}-x_k||`。
- `history: list[HistoryItem]`：完整轨迹。
- `cases: list[dict]`：`main` 中固定的 2D/5D/病态条件数样例。

## R06

正确性要点：
- `A` 为 SPD 时，`p_k^T A p_k > 0`（非零方向下），可安全计算 `alpha_k`。
- 方向 `p_k` 在 `A`-内积下共轭，减少不同方向上的重复“之字形”震荡。
- 残差范数 `||r_k|| = ||grad f(x_k)||`，可直接作为收敛判据。
- 脚本用 `np.linalg.solve(A,b)` 作为参考解核验实现正确性。

## R07

复杂度分析（稠密矩阵）：
- 单轮主成本：矩阵向量乘 `A @ p`，复杂度 `O(n^2)`。
- 总时间复杂度：`O(T * n^2)`，`T` 为迭代轮数。
- 空间复杂度：
  - 状态向量 `x/r/p` 为 `O(n)`；
  - 若保存完整历史，额外 `O(T)`。

在理想精确算术下，线性 CG 对 `n` 维 SPD 二次型至多 `n` 步收敛；有限精度下通常接近该量级。

## R08

边界与异常处理：
- `A` 非方阵/非对称/非正定：抛 `ValueError`。
- `b`、`x0` 非一维或与 `A` 维度不匹配：抛 `ValueError`。
- 输入含 `nan/inf`：抛 `ValueError`。
- `tol <= 0`、`max_iter <= 0`、`denominator_floor <= 0`：抛 `ValueError`。
- 迭代中若 `p^T A p` 过小或非有限：抛 `RuntimeError`。
- 迭代中若出现非有限迭代点/残差：抛 `RuntimeError`。
- 超过最大迭代仍未收敛：抛 `RuntimeError`。

## R09

MVP 取舍说明：
- 采用 `numpy` 手写线性 CG 主循环，不调用 `scipy.sparse.linalg.cg` 黑箱。
- 聚焦“优化中的 SPD 二次型”这个最经典、最可验证场景。
- 暂不引入预条件器、稀疏格式和并行线性代数，保持实现最小且可审计。
- 保留详细迭代日志，便于教学与排障。

## R10

`demo.py` 主要函数职责：
- `check_vector`：检查向量维度与有限性。
- `check_spd_matrix`：检查 SPD 条件（含 Cholesky）。
- `objective`：计算二次目标值。
- `gradient`：计算梯度 `A x - b`。
- `conjugate_gradient_spd`：线性 CG 主算法。
- `print_history`：格式化打印前若干轮迭代。
- `run_case`：单样例运行与参考解误差评估。
- `main`：组织样例、汇总指标并给出通过判定。

## R11

运行方式：

```bash
cd Algorithms/数学-优化-0344-共轭梯度法_(优化)
uv run python demo.py
```

脚本不读取命令行参数，不请求用户输入。

## R12

输出字段说明：
- `iter`：迭代编号。
- `f(x_k)`：该轮后目标函数值。
- `||grad||`：梯度范数（等于残差范数）。
- `alpha`：该轮搜索步长。
- `beta`：该轮方向共轭系数。
- `||step||`：参数更新步长范数。
- `x* estimate`：算法估计最优解。
- `x* reference`：线性方程参考解。
- `absolute/relative error`：估计解对参考解误差。
- `Summary`：所有样例的最大/平均相对误差、最大梯度范数与通过标记。

## R13

内置最小测试集：
- `2D SPD quadratic`：低维基础正确性。
- `5D banded SPD quadratic`：中维耦合项场景。
- `Ill-conditioned diagonal SPD`：病态条件数场景。

建议补充异常测试：
- 传入非对称 `A`（应报错）。
- 传入非正定 `A`（应报错）。
- 传入含 `nan` 的 `x0`（应报错）。

## R14

可调参数：
- `tol`：收敛阈值（默认 `1e-10`）。
- `max_iter`：最大迭代轮数（默认 `500`）。
- `denominator_floor`：`p^T A p` 安全下限（默认 `1e-18`）。
- `print_history(..., max_lines)`：轨迹打印行数上限。

调参建议：
- 调试时可放宽 `tol` 并降低 `max_iter`。
- 精度校验时收紧 `tol` 并观察 `max_rel_error` 与 `max_grad_norm`。

## R15

方法对比：
- 对比最速下降法：
  - CG 利用共轭方向，通常迭代数更少。
- 对比牛顿法：
  - 牛顿法局部收敛快，但每轮需解 Hessian 线性系统；
  - 线性 CG 每轮仅需矩阵向量乘，结构更轻量。
- 对比拟牛顿（BFGS/L-BFGS）：
  - 拟牛顿适合一般非线性目标；
  - 线性 CG 在 SPD 二次目标上更“问题结构特化”。

## R16

典型应用场景：
- 二次型能量最小化与参数估计。
- 大规模线性系统 `Ax=b` 的优化等价求解。
- 作为更复杂优化器中的子求解器（尤其 SPD 子问题）。
- 数值优化课程中用于展示“共轭方向”思想。

## R17

可扩展方向：
- 增加预条件共轭梯度（PCG）提升病态问题收敛速度。
- 扩展到稀疏矩阵（如 CSR）以降低内存和运算成本。
- 添加收敛曲线可视化与 CSV 日志导出。
- 在非线性目标上扩展为非线性共轭梯度（FR/PR/HZ 等变体）。
- 与最速下降、L-BFGS 做统一 benchmark。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 构造 3 个固定 SPD 二次样例，并设置 `tol/max_iter`。
2. 每个样例由 `run_case` 调用 `conjugate_gradient_spd` 执行求解。  
3. `conjugate_gradient_spd` 先做输入检查（SPD、维度、数值有限性、参数范围）。
4. 初始化 `r=b-Ax`、`p=r`、`rr=r^T r`，并用 `||r||` 判断是否已收敛。
5. 每轮计算 `ap=A@p` 与分母 `p^T A p`，据此得到 `alpha=rr/(p^T A p)`。
6. 更新 `x_next = x + alpha*p` 与 `r_next = r - alpha*ap`，再算 `beta=(r_next^T r_next)/rr`。
7. 将 `(iter, f(x), ||grad||, alpha, beta, ||step||)` 写入 `history`，并检查梯度/步长终止条件。
8. 若未终止，则按 `p_next = r_next + beta*p` 进入下一轮；结束后在 `run_case` 里与 `np.linalg.solve` 参考解做误差核验并汇总。
