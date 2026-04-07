# 信赖域方法

- UID: `MATH-0349`
- 学科: `数学`
- 分类: `优化`
- 源序号: `349`
- 目标目录: `Algorithms/数学-优化-0349-信赖域方法`

## R01

信赖域方法（Trust-Region Method）是经典二阶优化框架。  
与“先选方向再线搜索步长”不同，它在每轮直接求解一个受约束子问题：
- 在当前点附近建立二次模型 `m_k(p)`；
- 只允许步子 `p` 落在半径为 `Delta_k` 的球内（信赖域）；
- 通过实际下降与模型预测下降的比值 `rho_k` 决定“是否接受此步”和“如何调整半径”。

本目录给出一个可运行、可审计的 MVP：
- 使用 `numpy` 手写主流程；
- 子问题用 `dogleg`（柯西点 + 牛顿步折线）求近似解；
- 输出每轮 `f(x_k)`、`||g_k||`、`||p_k||`、`Delta_k`、`rho_k`、接受标记。

## R02

问题定义（本实现）：
- 输入：
  - 目标函数 `f(x)`；
  - 梯度函数 `g(x)=grad f(x)`；
  - Hessian 函数 `B(x)=nabla^2 f(x)`；
  - 初始点 `x0`；
  - 参数 `delta0, delta_max, eta, tol, max_iter`。
- 每轮子问题：
  - `min_p m_k(p) = f(x_k) + g_k^T p + 0.5 * p^T B_k p`
  - `s.t. ||p||_2 <= Delta_k`
- 输出：
  - 近似最优点 `x*`；
  - 迭代轨迹 `[(k, f, ||g||, ||p||, Delta, rho, accept), ...]`。

`demo.py` 内置固定案例（Rosenbrock + 病态二次型），无需交互输入。

## R03

核心数学关系：

1. 二次模型与半径约束：
   - `m_k(p)=f_k + g_k^T p + 0.5 p^T B_k p`
   - `||p|| <= Delta_k`
2. 预测下降与实际下降：
   - `pred_k = m_k(0)-m_k(p_k) = -(g_k^T p_k + 0.5 p_k^T B_k p_k)`
   - `ared_k = f(x_k)-f(x_k+p_k)`
3. 可信度比值：
   - `rho_k = ared_k / pred_k`
4. 接受准则：
   - 若 `rho_k > eta`，接受 `x_{k+1}=x_k+p_k`；
   - 否则拒绝，保留 `x_{k+1}=x_k`。
5. 半径更新（本实现）：
   - `rho_k < 0.25`：`Delta <- 0.25*Delta`；
   - `rho_k > 0.75` 且步长贴边：`Delta <- min(2*Delta, Delta_max)`；
   - 否则 `Delta` 保持不变。

## R04

算法流程（高层）：
1. 初始化 `x <- x0`，`Delta <- delta0`。  
2. 计算 `f(x), g(x), B(x)`。  
3. 在 `||p||<=Delta` 内，用 dogleg 近似求解信赖域子问题得到 `p`。  
4. 计算 `pred`、`ared` 与 `rho=ared/pred`。  
5. 根据 `rho` 决定是否接受 `x+p`。  
6. 根据 `rho` 与是否贴边调整 `Delta`。  
7. 记录轨迹并检查 `||g||<=tol` 等停止条件。  
8. 达到收敛则返回，否则进入下一轮。

## R05

核心数据结构：
- `HistoryItem = (iter, f_x, grad_norm, step_norm, delta, rho, accept)`：
  - `iter`：迭代编号；
  - `f_x`：当前点目标值；
  - `grad_norm`：当前梯度范数；
  - `step_norm`：本轮候选步长范数；
  - `delta`：更新后的信赖域半径；
  - `rho`：实际/预测下降比；
  - `accept`：是否接受（`0/1`）。
- `history: list[HistoryItem]`：完整迭代轨迹。
- `cases: list[dict]`：`main` 中固定测试样例配置。

## R06

正确性要点：
- 子问题近似解的可行性：dogleg 始终返回 `||p||<=Delta` 的步。  
- 下降判定可解释：通过 `rho` 直接比较模型与真实函数一致性。  
- 稳健性机制：当 `pred<=0` 时回退到边界最速下降方向，避免无意义比值。  
- 全流程可验证：案例中均给出参考最优点（Rosenbrock 为 `(1,1)`，二次型可线性方程求解），可直接检查误差与梯度范数。  
- 终止准则明确：使用 `||g||<=tol` 和最大迭代轮数双条件。

## R07

复杂度分析（`n` 维、稠密 Hessian）：
- 每轮主要成本：
  - 计算 Hessian：问题相关，记作 `C_hess`；
  - 向量/矩阵运算（`B@g`、线性方程 `B p = -g`）：`O(n^2)` 到 `O(n^3)`（稠密求解为 `O(n^3)`）。
- 单轮复杂度：`O(C_hess + n^3)`（由牛顿步线性求解主导）。  
- 总时间复杂度：`O(T * (C_hess + n^3))`。  
- 空间复杂度：
  - 当前状态 `x,g,p`：`O(n)`；
  - Hessian：`O(n^2)`；
  - 轨迹：`O(T)`。

## R08

边界与异常处理：
- `x0` 不是一维向量或含 `nan/inf`：抛 `ValueError`。  
- `delta0<=0`、`delta_max<=0`、`delta0>delta_max`：抛 `ValueError`。  
- `eta` 不在 `(0,1)`：抛 `ValueError`。  
- `tol<=0` 或 `max_iter<=0`：抛 `ValueError`。  
- `grad(x)` 维度不匹配或 `hess(x)` 形状非法：抛 `ValueError`。  
- 若 `pred` 非正（并且回退后仍非正）：抛 `RuntimeError`。  
- 达到 `max_iter` 仍未满足停止条件：抛 `RuntimeError`。

## R09

MVP 取舍：
- 采用 `numpy` 手写核心，不调用 `scipy.optimize.minimize` 黑盒。  
- 子问题只实现 dogleg（不实现完整截断 CG / 精确子问题求解）。  
- 样例覆盖“非凸 + 二次型”两类典型场景，强调可验证与可解释。  
- 不做命令行参数和配置系统，保持单文件、即跑即看。

## R10

`demo.py` 主要函数职责：
- `check_vector`：输入向量合法性检查。  
- `predicted_reduction`：计算模型预测下降。  
- `dogleg_step`：在给定半径内构造 dogleg 步。  
- `trust_region_dogleg`：信赖域主循环（接受/拒绝 + 半径更新）。  
- `rosenbrock_fun/grad/hess`：非凸测试函数及导数。  
- `make_quadratic_case`：构造二次型 `f/g/hess`。  
- `print_history`：打印迭代日志表。  
- `run_case`：单案例执行并与参考解比较。  
- `main`：组织样例、汇总指标与通过标记。

## R11

运行方式：

```bash
cd Algorithms/数学-优化-0349-信赖域方法
python3 demo.py
```

脚本无需任何交互输入。

## R12

输出字段说明：
- `iter`：迭代轮次。  
- `f(x_k)`：当前目标值。  
- `||g||`：当前梯度范数。  
- `||p||`：本轮候选步长度。  
- `delta`：本轮更新后的信赖域半径。  
- `rho`：实际下降/预测下降比值。  
- `acc`：步是否被接受（1 表示接受）。  
- `x* estimate/reference`：数值解与参考解。  
- `absolute/relative error`：解向量误差。  
- `Summary`：最大相对误差、最大终止梯度、平均迭代数与整体通过标记。

## R13

建议最小测试集（脚本已包含）：
- `Rosenbrock (non-convex)`：验证非凸地形与半径收缩/扩张机制。  
- `Ill-conditioned quadratic`：验证二次型上的快速收敛与精度。

建议补充异常测试：
- `hess(x)` 返回错误形状（应报错）；  
- `delta0 > delta_max`（应报错）；  
- 构造导致 `pred<=0` 的病态 Hessian（应触发保护逻辑或报错）。

## R14

可调参数：
- `delta0`：初始信赖域半径（默认 `1.0`）。  
- `delta_max`：半径上限（默认 `100.0`）。  
- `eta`：接受阈值（默认 `0.1`）。  
- `tol`：梯度收敛阈值（默认 `1e-8`）。  
- `max_iter`：最大迭代数（默认 `300`）。

调参建议：
- 收敛慢时可增大 `delta0`；  
- 震荡或拒绝过多时可减小 `delta0` 或提高 `eta`；  
- 精度评估时收紧 `tol` 并观察 `||g||` 与误差同步下降。

## R15

方法对比：
- 对比线搜索牛顿法：
  - 线搜索只调“步长”；信赖域同时约束“模型可信区域”，在 Hessian 不理想时更稳。  
- 对比拟牛顿（BFGS/L-BFGS）：
  - 拟牛顿每轮更省（不显式 Hessian）；  
  - 信赖域二阶模型信息更直接，局部收敛可更强。  
- 对比纯梯度下降：
  - 纯梯度依赖步长策略；  
  - 信赖域利用曲率信息，通常迭代更少。

## R16

典型应用场景：
- 非线性最小二乘与参数估计。  
- 机械/结构优化中的局部迭代求解。  
- 机器学习中中小规模可微目标的高精度优化。  
- 作为更复杂优化器（如 SQP、内点法）的子问题求解组件。

## R17

可扩展方向：
- 用截断共轭梯度（Truncated CG）替代 dogleg，支持更高维与稀疏 Hessian。  
- 增加 SR1/BFGS 近似 Hessian 以减少二阶导成本。  
- 加入日志落盘（CSV/JSON）与收敛曲线绘图。  
- 增加多起点批量评测与 benchmark。  
- 扩展到有约束优化（与罚函数、增广拉格朗日结合）。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 构造两个固定问题：Rosenbrock（非凸）与病态二次型（可线性求精确解）。  
2. `run_case` 调用 `trust_region_dogleg` 执行迭代，并打印每轮指标。  
3. `trust_region_dogleg` 读取 `x, Delta`，计算 `f(x), g(x), B(x)`，先检查 `||g||<=tol`。  
4. 调 `dogleg_step(g, B, Delta)` 生成可行步：先看柯西点，再尝试牛顿步，超边界则在 dogleg 折线上插值到边界。  
5. 计算 `pred = -(g^T p + 0.5 p^T B p)`；若 `pred<=0`，回退到边界最速下降步并重算。  
6. 计算试探点 `x_trial=x+p` 的 `ared=f(x)-f(x_trial)`，得到 `rho=ared/pred`。  
7. 若 `rho>eta` 则接受 `x_trial`，否则拒绝；随后按规则更新 `Delta`（缩小/扩大/保持）。  
8. 记录 `(iter, f, ||g||, ||p||, Delta, rho, accept)` 到 `history`，循环直至收敛；`run_case` 再与参考解计算误差并汇总到 `Summary`。
