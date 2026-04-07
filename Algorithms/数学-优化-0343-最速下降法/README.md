# 最速下降法

- UID: `MATH-0343`
- 学科: `数学`
- 分类: `优化`
- 源序号: `343`
- 目标目录: `Algorithms/数学-优化-0343-最速下降法`

## R01

最速下降法（Steepest Descent）是无约束优化中的基础一阶方法：  
在每轮迭代中沿“当前梯度的反方向”前进，使目标函数在局部下降最快。

本条目给出一个可直接运行、可审计迭代细节的 MVP：
- 优化目标选为 SPD（二次正定）二次型；
- 步长使用该场景下可解析的精确线搜索；
- 输出每轮 `f(x_k)`、`||grad||`、`alpha`、`||step||`。

## R02

问题定义（本目录实现）：
- 输入：
  - 对称正定矩阵 `A in R^(n*n)`；
  - 向量 `b in R^n`；
  - 初值 `x0 in R^n`；
  - 容差 `tol` 与最大迭代轮数 `max_iter`。
- 目标函数：
  - `f(x) = 0.5 * x^T A x - b^T x`。
- 输出：
  - 近似最优解 `x*`；
  - 迭代轨迹 `[(k, f(x_k), ||grad||, alpha_k, ||step||), ...]`。

`demo.py` 内置固定样例，无需交互输入。

## R03

数学基础：

1. 一般形式最速下降更新：
   `x_(k+1) = x_k - alpha_k * grad f(x_k)`。  
2. 对二次型 `f(x)=0.5*x^T A x - b^T x`，梯度为：
   `g_k = grad f(x_k) = A x_k - b`。  
3. 令搜索方向 `d_k = -g_k`，对 `phi(alpha)=f(x_k + alpha d_k)` 做一维最小化，可得精确步长：
   `alpha_k = (g_k^T g_k) / (g_k^T A g_k)`（`A` SPD 时分母严格正）。  
4. 因此每轮都保证沿下降方向前进，且在该方向上一维最优。

## R04

算法流程（高层）：
1. 校验 `A` 为对称正定、`b/x0` 维度匹配且数值有限。  
2. 初始化 `x <- x0`。  
3. 计算梯度 `g = A x - b` 与 `||g||`。  
4. 若 `||g||` 达到阈值则停止。  
5. 计算精确线搜索步长 `alpha = (g^T g)/(g^T A g)`。  
6. 更新 `x_next = x - alpha * g`。  
7. 记录本轮轨迹并检查步长停止条件。  
8. 未满足停止条件则继续下一轮，直至收敛或达到 `max_iter`。

## R05

核心数据结构：
- `HistoryItem = (iter, f_x, grad_norm, alpha, step_norm)`：
  - `iter`：迭代编号；
  - `f_x`：更新后目标函数值；
  - `grad_norm`：更新前梯度范数；
  - `alpha`：本轮精确步长；
  - `step_norm`：`||x_(k+1)-x_k||`。
- `history: list[HistoryItem]`：完整收敛轨迹。
- `cases: list[dict]`：`main` 中固定的 2D/3D/病态条件数样例。

## R06

正确性要点：
- 方向正确：`d_k=-g_k` 是局部最陡下降方向。  
- 步长正确：二次型上采用解析步长，等价于沿 `d_k` 的精确一维最小化。  
- 终止合理：同时使用梯度阈值和步长阈值，避免单一判据误判。  
- 可验证性：对 SPD 二次型，真解可由线性方程 `A x = b` 求得，脚本将迭代结果与 `np.linalg.solve` 对照。

## R07

复杂度分析（`A` 为稠密 `n*n`）：
- 每轮主成本：
  - 矩阵向量乘 `A @ x`：`O(n^2)`；
  - `g^T A g` 中 `A @ g`：`O(n^2)`。  
- 单轮时间复杂度：`O(n^2)`。  
- 总时间复杂度：`O(T * n^2)`（`T` 为迭代轮数）。  
- 空间复杂度：
  - 仅当前状态：`O(n)`；
  - 保留完整轨迹：`O(T)` 额外开销。

## R08

边界与异常处理：
- `A` 非方阵/非对称/非正定：抛 `ValueError`。  
- `b` 或 `x0` 不是一维向量，或维度与 `A` 不匹配：抛 `ValueError`。  
- 输入含 `nan/inf`：抛 `ValueError`。  
- `tol<=0`、`max_iter<=0`、`denominator_floor<=0`：抛 `ValueError`。  
- 迭代中若出现非有限数或 `g^TAg` 过小：抛 `RuntimeError`。  
- 超过 `max_iter` 未收敛：抛 `RuntimeError`。

## R09

MVP 取舍说明：
- 采用 `numpy` 手写核心迭代，不调用 `scipy.optimize.minimize` 黑盒。  
- 只覆盖“最速下降法最经典且可验证”的 SPD 二次型场景。  
- 不引入动量、Nesterov、随机梯度等扩展，保持最小可解释实现。  
- 保留迭代轨迹，便于教学和校验，牺牲少量内存换可观测性。

## R10

`demo.py` 主要函数职责：
- `check_vector`：向量形状与有限性检查。  
- `check_spd_matrix`：矩阵方阵/对称/正定检查。  
- `objective`：计算二次目标值。  
- `gradient`：计算梯度 `A x - b`。  
- `steepest_descent_spd`：最速下降主循环（精确线搜索）。  
- `print_history`：格式化打印迭代轨迹。  
- `run_case`：单样例执行 + 误差统计。  
- `main`：组织固定样例并输出汇总结论。

## R11

运行方式：

```bash
cd Algorithms/数学-优化-0343-最速下降法
python3 demo.py
```

脚本不读取命令行参数，不请求用户输入。

## R12

输出字段说明：
- `iter`：迭代编号。  
- `f(x_k)`：当前迭代点目标函数值（越小越好）。  
- `||grad||`：梯度范数，反映是否接近驻点。  
- `alpha`：解析线搜索步长。  
- `||step||`：相邻两轮参数向量位移范数。  
- `x* estimate`：算法求得近似最优解。  
- `x* reference`：线性方程解出的参考真值。  
- `absolute/relative error`：估计解与参考解的误差。  
- `Summary`：所有样例的最大/平均相对误差与通过标记。

## R13

建议最小测试集（脚本已内置）：
- `2D SPD quadratic`：低维可直观核对收敛行为。  
- `3D SPD quadratic`：验证多维场景实现正确性。  
- `Ill-conditioned diagonal SPD`：观察病态条件数下的迭代变慢现象。

建议补充异常测试：
- 构造非对称 `A`（应报错）；  
- 构造非正定 `A`（应报错）；  
- 传入含 `nan` 的 `x0`（应报错）。

## R14

可调参数：
- `tol`：收敛阈值（默认 `1e-10`）。  
- `max_iter`：最大迭代数（默认 `5000`）。  
- `denominator_floor`：保护分母下限（默认 `1e-18`）。  
- `print_history(..., max_lines)`：控制轨迹打印行数。

调参建议：
- 调试阶段可放宽 `tol`、减少 `max_iter`；  
- 精度验证阶段使用更严格 `tol` 并观察 `||grad||` 与误差。

## R15

方法对比：
- 对比固定步长梯度下降：
  - 最速下降每轮按局部几何自适应选步长，通常更稳。  
- 对比牛顿法：
  - 牛顿法收敛更快（局部二次），但需 Hessian 与线性系统求解；  
  - 最速下降每轮更便宜，实现简单。  
- 对比共轭梯度（SPD 二次型）：
  - 共轭梯度通常迭代更少；  
  - 最速下降概念更直观，适合教学与基线实现。

## R16

典型应用场景：
- 二次优化模型的基线求解器（参数估计、能量最小化）。  
- 大型优化算法中的子模块或 warm-up 过程。  
- 数值优化课程中展示“梯度方向 + 线搜索”核心思想。  
- 新优化器落地前的 correctness baseline。

## R17

可扩展方向：
- 将精确线搜索替换为 Armijo/Wolfe 回溯线搜索，支持一般非线性目标。  
- 增加预条件最速下降（Preconditioned SD）提升病态问题收敛。  
- 增加日志落盘（CSV）与收敛曲线可视化。  
- 扩展到稀疏矩阵场景，利用稀疏线性代数降低单轮成本。  
- 与共轭梯度、L-BFGS 做统一 benchmark。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 定义 3 组固定 SPD 二次型样例和全局参数 `tol/max_iter`。  
2. 每个样例由 `run_case` 调用 `steepest_descent_spd` 执行求解，并打印迭代轨迹。  
3. `steepest_descent_spd` 先调用 `check_spd_matrix` 与 `check_vector` 做输入合法性检查。  
4. 进入循环后计算梯度 `g = A x - b`，若 `||g||` 已小于阈值则直接返回。  
5. 计算精确步长分子分母：`g^T g` 与 `g^T A g`，并检查分母是否安全。  
6. 用 `alpha = (g^T g)/(g^T A g)` 更新参数 `x_next = x - alpha * g`。  
7. 计算 `f(x_next)` 和 `||x_next-x||`，将 `(iter, f, ||g||, alpha, ||step||)` 追加到 `history`。  
8. `run_case` 将结果与 `np.linalg.solve(A,b)` 参考解对比，`main` 汇总最大/平均相对误差与通过标志。  
