# Levenberg-Marquardt

- UID: `MATH-0350`
- 学科: `数学`
- 分类: `优化`
- 源序号: `350`
- 目标目录: `Algorithms/数学-优化-0350-Levenberg-Marquardt`

## R01

Levenberg-Marquardt（LM）算法是求解非线性最小二乘问题的经典方法，常被看作“高斯-牛顿法 + 阻尼控制”的折中：
- 当局部线性近似可靠时，行为接近高斯-牛顿，收敛快；
- 当模型失配或远离最优点时，增大阻尼后更像梯度下降，稳定性更好。

本目录实现一个可运行、可审计的 MVP：
- 手写 LM 主循环与阻尼参数更新（非黑盒优化器）；
- 支持解析 Jacobian 与有限差分 Jacobian；
- 固定两个案例自动运行并做严格误差检查。

## R02

本实现求解的问题形式：
- 目标：
  - `min_x 0.5 * ||r(x)||_2^2`
- 输入：
  - 残差函数 `r(x)`；
  - 初值 `x0`；
  - 可选 Jacobian 函数 `J(x)`；
  - 收敛参数 `tol_grad/tol_step/tol_cost`、阻尼参数 `lambda0/max_lambda`。
- 输出：
  - 近似解 `x*`；
  - 最终 `cost = 0.5*||r(x*)||^2`；
  - 梯度近似范数 `||J^T r||_inf`；
  - 迭代轨迹与评估计数。

`demo.py` 不需要交互输入，直接运行全部案例。

## R03

核心数学关系：

1. 最小二乘目标：
   - `F(x) = 0.5 * r(x)^T r(x)`。
2. 一阶梯度近似：
   - `g = J^T r`，其中 `J = dr/dx`。
3. 高斯-牛顿近似 Hessian：
   - `A = J^T J`。
4. LM 线性系统：
   - `(A + lambda I) * h = -g`。
5. 参数更新：
   - `x_new = x + h`。
6. 增益比（实际下降 / 预测下降）：
   - `rho = (F(x) - F(x_new)) / (0.5 * h^T (lambda h - g))`。
7. 步接受逻辑：
   - `rho > 0` 接受，否则拒绝并增大阻尼。

## R04

算法流程（高层）：
1. 读取 `x0` 并检查参数合法性。  
2. 计算初始残差 `r`、Jacobian `J`、`A=J^T J`、`g=J^T r`。  
3. 判断 `||g||_inf` 是否满足梯度收敛阈值。  
4. 解线性系统 `(A + lambda I)h = -g` 得到候选步。  
5. 计算试探点 `x+h` 的新代价与增益比 `rho`。  
6. 若 `rho>0` 则接受步并减小阻尼；否则拒绝步并增大阻尼。  
7. 记录轨迹（cost、梯度、步长、阻尼、rho、接受标记）。  
8. 满足梯度/步长/代价停止条件或达到迭代上限后返回。

## R05

核心数据结构：
- `HistoryItem = (iter, cost, grad_inf, step_norm, damping, rho, accepted)`：单轮日志。  
- `LMResult`：最终结果封装：
  - `x, cost, grad_inf_norm, iterations, accepted_steps`；
  - `converged, message`；
  - `function_evals, jacobian_evals`；
  - `history`。
- 案例生成函数：
  - `make_exponential_case`（解析 Jacobian）；
  - `make_circle_case`（有限差分 Jacobian）。

## R06

正确性要点：
- 更新方向来自 LM 正规方程，不是经验式步长。  
- 通过 `rho` 比较“真实下降”与“模型预测下降”，实现步质量判别。  
- 拒绝步时放大阻尼，可将过激步收敛到更保守更新。  
- 接受步时减小阻尼，逐步回到高斯-牛顿的快速局部收敛特性。  
- 输出对照参考参数并计算相对误差，确保实现在案例上可验证。

## R07

复杂度分析（参数维度 `n`，残差维度 `m`，迭代轮数 `T`）：
- 每轮主要成本：
  - 残差评估：`C_r`；
  - Jacobian 评估：`C_J`（解析 Jacobian 或有限差分）；
  - 构造 `J^T J`：约 `O(m n^2)`；
  - 求解 `n x n` 线性系统：`O(n^3)`（稠密）。
- 总时间复杂度近似：
  - `O(T * (C_r + C_J + m n^2 + n^3))`。
- 空间复杂度：
  - `J` 与 `A` 主存储约 `O(mn + n^2)`；
  - 保存全轨迹额外 `O(T)`。

## R08

边界与异常处理：
- `x0` 必须是非空一维有限向量。  
- `max_iter<=0`、阈值非正、`lambda0<=0` 等非法配置直接抛 `ValueError`。  
- Jacobian 形状与有限性检查失败会抛异常。  
- 有限差分中若残差维度发生变化会报错。  
- 线性系统求解失败时提前停止并给出消息。  
- 阻尼放大超过 `max_lambda` 时终止，避免无限拒绝步循环。

## R09

MVP 取舍：
- 仅依赖 `numpy`，保持轻量可运行。  
- 不调用 `scipy.optimize.least_squares` 黑盒，核心更新、阻尼策略、接受逻辑全部显式实现。  
- 覆盖两种 Jacobian 来源（解析 + 数值差分），兼顾透明性与通用性。  
- 不扩展到大规模稀疏系统、鲁棒损失（Huber/Cauchy）等进阶工程特性。

## R10

`demo.py` 主要函数职责：
- `ensure_vector`：向量输入合法性检查。  
- `finite_difference_jacobian`：前向差分 Jacobian。  
- `lm_solve`：LM 主循环（线性系统、rho 判别、阻尼更新、终止判据）。  
- `make_exponential_case`：构造指数曲线拟合样例（含解析 Jacobian）。  
- `make_circle_case`：构造圆参数拟合样例（使用有限差分 Jacobian）。  
- `print_history`：格式化打印迭代轨迹。  
- `run_case`：执行单案例并输出误差指标。  
- `main`：串行运行两案例并执行 Summary 级严格校验。

## R11

运行方式：

```bash
cd Algorithms/数学-优化-0350-Levenberg-Marquardt
python3 demo.py
```

脚本不读取命令行参数，不请求交互输入。

## R12

输出字段说明：
- `Converged`：是否达到收敛条件。  
- `Stop reason`：停止原因（梯度阈值、步长阈值、代价阈值、阻尼上限等）。  
- `Iterations`：记录的迭代轮数。  
- `Accepted steps`：被接受更新步数量。  
- `Final x / Final cost / Final ||J^T r||_inf`：最终估计与一阶残差。  
- `Residual evals / Jacobian evals`：残差和 Jacobian 评估次数。  
- 轨迹列：
  - `iter`：轮次；
  - `cost`：当前目标值；
  - `||J^T r||_inf`：梯度近似无穷范数；
  - `||step||`：候选步长度；
  - `lambda`：阻尼参数；
  - `rho`：增益比；
  - `acc`：步接受标记（1/0）。

## R13

最小测试集（已内置）：
1. `Exponential-Curve-Fit (analytic Jacobian)`
- 参数：`[a, b, c]`
- 数据：`y = a*exp(b*t) + c`（无噪声）
- 目标：验证解析 Jacobian 下的快速收敛与参数恢复精度。

2. `Circle-Fit (finite-difference Jacobian)`
- 参数：`[cx, cy, r]`
- 数据：圆周采样点
- 目标：验证仅给残差函数时，有限差分 Jacobian 的通用求解能力。

可补充测试：
- Jacobian 返回错误形状的异常路径；
- 非法超参数（负阈值、`lambda0<=0`）；
- 残差返回 `nan/inf` 的鲁棒性路径。

## R14

关键参数与调参建议：
- `lambda0`：初始阻尼，越大越保守。  
- `max_lambda`：阻尼上限，防止长期拒绝步失控。  
- `tol_grad`：`||J^T r||_inf` 收敛阈值。  
- `tol_step`：步长阈值，控制参数更新是否已足够小。  
- `tol_cost`：代价变化阈值，控制目标值收敛。  
- `max_iter`：最大迭代轮数。

调参经验：
- 频繁拒绝步可适度增大 `lambda0`；
- 收敛慢可减小 `lambda0` 或放宽 `tol_cost`；
- 参数尺度差异大时应先做变量缩放，再使用 LM。

## R15

方法对比：
- 对比高斯-牛顿：
  - 高斯-牛顿相当于 `lambda -> 0`，局部快但远离最优时易不稳；
  - LM 通过阻尼在稳定性与速度间自适应折中。
- 对比梯度下降：
  - 梯度下降每步成本低，但通常需要更多轮次；
  - LM 利用 Jacobian 曲率信息，常在中小规模最小二乘问题更高效。
- 对比黑盒 `least_squares`：
  - 黑盒更完整（约束、鲁棒损失、稀疏求解）；
  - 本实现更适合教学与源码审计。

## R16

典型应用场景：
- 曲线/曲面参数拟合（指数、幂律、Logistic 等）。  
- 计算机视觉中的几何参数估计（相机标定、位姿、圆/椭圆拟合）。  
- 系统辨识与参数反演。  
- 作为更复杂优化流程中的局部精修器。

## R17

可扩展方向：
- 引入鲁棒损失（Huber、Cauchy）处理离群点。  
- 增加参数缩放与预条件，改善病态问题收敛。  
- 扩展稀疏 Jacobian 与稀疏线性求解。  
- 增加信赖域半径视角的 LM 变体与更细致的接受准则。  
- 将轨迹导出为 CSV/JSON 并绘制收敛曲线。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 构造两个固定任务：指数拟合（解析 Jacobian）和圆拟合（有限差分 Jacobian）。  
2. `run_case` 调用 `lm_solve`，并在结束后输出收敛信息、梯度规模和参数误差。  
3. `lm_solve` 初始化 `x, r, J, A=J^T J, g=J^T r, lambda`，进入迭代循环。  
4. 每轮求解 `(A + lambda I)h = -g` 得到候选步 `h`，并检查步长收敛阈值。  
5. 计算试探点 `x+h` 的新残差与新代价，得到实际下降 `actual_reduction`。  
6. 用 `0.5*h^T(lambda h - g)` 计算预测下降，形成增益比 `rho`。  
7. 若 `rho>0` 则接受步、刷新 `x/r/J/A/g` 并减小阻尼；否则拒绝步并放大阻尼。  
8. 记录 `(iter, cost, ||J^T r||_inf, ||step||, lambda, rho, acc)` 到 `history`，直到满足停止条件或触发保护终止。
