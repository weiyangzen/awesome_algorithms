# 梯度下降法

- UID: `MATH-0342`
- 学科: `数学`
- 分类: `优化`
- 源序号: `342`
- 目标目录: `Algorithms/数学-优化-0342-梯度下降法`

## R01

梯度下降法（Gradient Descent）是最常用的一阶优化方法之一：
在每轮迭代中，利用当前点的梯度信息，沿负梯度方向移动，从而逐步减小目标函数值。

本条目给出一个最小可运行 MVP：
- 目标函数选为带 `L2` 正则的线性回归（二次凸目标）；
- 核心迭代完全手写，不调用黑盒优化器；
- 输出每轮 `loss`、`||grad||`、`step_norm` 以便审计收敛过程。

## R02

问题定义（本目录实现）：
- 输入：
  - 特征矩阵 `X in R^(m*n)`；
  - 标签向量 `y in R^m`；
  - 初始参数 `w0 in R^n`；
  - 学习率 `lr`、正则系数 `l2`、容差 `tol`、最大迭代数 `max_iter`。
- 目标函数：
  - `J(w) = (1/(2m)) * ||Xw - y||^2 + (l2/2) * ||w||^2`。
- 输出：
  - 近似最优参数 `w*`；
  - 迭代历史 `[(iter, loss, grad_norm, step_norm), ...]`。

`demo.py` 内置固定数据构造与测试样例，不需要交互输入。

## R03

数学基础：

1. 梯度下降更新式：
   `w_(k+1) = w_k - lr * grad J(w_k)`。  
2. 对上述二次目标，梯度为：
   `grad J(w) = (1/m) * X^T (Xw - y) + l2 * w`。  
3. Hessian 为：
   `H = (1/m) * X^T X + l2 * I`，当 `l2 >= 0` 且 `X` 满列秩时为正定。  
4. 对二次凸函数，只要学习率满足 `0 < lr < 2/L`（`L` 为 Hessian 最大特征值），梯度下降可收敛到全局最优。

## R04

算法流程（高层）：
1. 检查 `X/y/w0` 形状与数值合法性。  
2. 若未显式给 `lr`，根据谱范数估计 `L` 并设置 `lr = 0.9 / L`。  
3. 初始化 `w <- w0`。  
4. 每轮计算 `loss` 与 `grad`。  
5. 若 `||grad||` 已低于阈值，则记录并终止。  
6. 执行参数更新 `w_next = w - lr * grad`。  
7. 记录 `(iter, loss, ||grad||, ||w_next-w||)`。  
8. 若步长已足够小则提前停止，否则继续。

## R05

核心数据结构：
- `HistoryItem = (iter, loss, grad_norm, step_norm)`：
  - `iter`：迭代编号；
  - `loss`：当前目标函数值；
  - `grad_norm`：梯度二范数；
  - `step_norm`：本轮参数更新向量的二范数。
- `history: list[HistoryItem]`：完整收敛轨迹。
- `cases: list[dict]`：`main` 中固定的测试样例配置。

## R06

正确性要点：
- 梯度表达式正确：由二次目标直接求导得到。  
- 方向正确：`-grad` 是局部最速下降方向。  
- 步长安全：默认用 `0.9/L`，满足常见稳定收敛区间。  
- 可验证：脚本把梯度下降结果与 ridge 闭式解对比，输出绝对/相对误差。  
- 可观测：每轮保留 `loss` 与 `||grad||`，便于判断是否真正收敛。

## R07

复杂度分析（`X` 为稠密 `m*n`）：
- 单轮主要计算：
  - 前向残差 `r = Xw - y`：`O(mn)`；
  - 梯度 `X^T r`：`O(mn)`。
- 单轮时间复杂度：`O(mn)`。  
- 总时间复杂度：`O(Tmn)`（`T` 为迭代轮数）。  
- 空间复杂度：
  - 状态向量与中间量：`O(m+n)`；
  - 若记录全部历史：额外 `O(T)`。

## R08

边界与异常处理：
- `X` 非二维、`y/w0` 非一维或维度不匹配：抛 `ValueError`。  
- 输入存在 `nan/inf`：抛 `ValueError`。  
- `l2 < 0`、`tol <= 0`、`max_iter <= 0`：抛 `ValueError`。  
- 自动学习率估计得到非有限或非正值：抛 `RuntimeError`。  
- 迭代中若出现非有限 `loss/grad/iterates`：抛 `RuntimeError`。  
- 达到 `max_iter` 仍未满足停止条件时返回当前结果（不交互中断）。

## R09

MVP 取舍说明：
- 只实现全批量（batch）梯度下降，不引入 mini-batch/SGD。  
- 目标限定为 ridge 线性回归，便于给出闭式真值对照。  
- 不调用 `scipy.optimize.minimize` 等黑盒优化 API。  
- 用 `numpy` 完成核心逻辑，保持最小依赖与可读性。

## R10

`demo.py` 主要函数职责：
- `check_vector`：一维向量与有限性检查。  
- `check_matrix`：二维矩阵与有限性检查。  
- `objective_ridge`：计算目标函数值。  
- `gradient_ridge`：计算梯度。  
- `estimate_lipschitz_constant`：估计 `L = lambda_max(X^T X / m + l2 I)`。  
- `gradient_descent_ridge`：梯度下降主循环。  
- `ridge_closed_form`：计算参考闭式解。  
- `print_history`：格式化输出迭代轨迹。  
- `run_case`：执行单个样例并做误差评估。  
- `main`：组织样例、运行并打印汇总。

## R11

运行方式：

```bash
cd Algorithms/数学-优化-0342-梯度下降法
python3 demo.py
```

脚本不会读取命令行参数，也不会请求用户输入。

## R12

输出字段说明：
- `iter`：迭代轮次。  
- `loss`：当前目标函数值（应总体下降）。  
- `||grad||`：梯度范数，反映离最优点的距离。  
- `||step||`：参数更新幅度。  
- `w* estimate`：梯度下降得到的参数。  
- `w* closed-form`：闭式解参数。  
- `absolute error`：`||w_est - w_ref||_2`。  
- `relative error`：相对 `||w_ref||_2` 的误差。  
- `Summary`：所有样例的最大/平均相对误差与是否通过严格阈值检查。

## R13

建议最小测试集（脚本已内置）：
- `Well-conditioned synthetic`：较好条件数，快速收敛。  
- `Correlated features`：特征强相关，验证在较难几何下仍能收敛。  
- `Higher-dimensional`：维度更高，验证实现的通用性。

建议补充异常测试：
- `l2 < 0`（应报错）；
- `X` 与 `y` 行数不一致（应报错）；
- `w0` 含 `nan`（应报错）。

## R14

可调参数：
- `lr`：学习率；传 `None` 时自动按谱上界估计。  
- `l2`：L2 正则系数。  
- `tol`：收敛阈值（影响 `grad` 与 `step` 停止判据）。  
- `max_iter`：最大迭代数。  
- `print_history(..., max_lines)`：控制输出行数。

调参建议：
- 收敛慢：增大 `max_iter` 或提高 `l2` 改善条件数。  
- 发散/震荡：减小 `lr`。  
- 要更高精度：减小 `tol` 并观察 `||grad||` 与相对误差。

## R15

方法对比：
- 对比闭式解：
  - 闭式解一步得到最优，但需解线性方程；
  - 梯度下降更适合大规模场景和流式扩展。  
- 对比牛顿法：
  - 牛顿法迭代次数少，但每步代价高；
  - 梯度下降每步计算简单、实现成本低。  
- 对比随机梯度下降（SGD）：
  - 本实现更稳定、噪声更小；
  - SGD 在超大数据下单步更便宜，但收敛轨迹抖动更大。

## R16

典型应用场景：
- 机器学习线性模型训练的基础优化器。  
- 复杂优化器上线前的 baseline 与正确性对照。  
- 数值优化教学中演示“梯度 + 学习率”核心机制。  
- 作为更高级方法（动量、Adam、L-BFGS）的起点实现。

## R17

可扩展方向：
- 增加 backtracking line search 自适应步长。  
- 增加动量/Nesterov，加速病态问题收敛。  
- 增加 mini-batch 与数据流式接口。  
- 输出 CSV 日志并绘制收敛曲线。  
- 在稀疏矩阵场景下改为稀疏线性代数实现。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 固定随机种子并构造三组可复现的 ridge 回归样例。  
2. `run_case` 调用 `gradient_descent_ridge` 运行优化，同时打印迭代轨迹。  
3. `gradient_descent_ridge` 首先调用 `check_matrix/check_vector` 完成输入合法性检查。  
4. 若 `lr is None`，调用 `estimate_lipschitz_constant` 计算 Hessian 最大特征值上界并设定稳定步长。  
5. 进入迭代后先计算 `loss = objective_ridge(...)` 和 `grad = gradient_ridge(...)`。  
6. 根据 `grad_norm` 判断是否收敛；未收敛则执行 `w_next = w - lr * grad`。  
7. 计算 `step_norm`，将 `(iter, loss, grad_norm, step_norm)` 追加到 `history`，并按步长阈值做提前停止。  
8. `run_case` 用 `ridge_closed_form` 生成参考真值，计算参数误差并在 `main` 中汇总最大/平均相对误差与通过标志。
