# 镜像下降法

- UID: `MATH-0398`
- 学科: `数学`
- 分类: `凸优化`
- 源序号: `398`
- 目标目录: `Algorithms/数学-凸优化-0398-镜像下降法`

## R01

镜像下降法（Mirror Descent）是一类针对约束凸优化的一阶方法。它把“欧氏空间里的梯度步”替换成“在对偶空间做梯度更新，再通过镜像映射回原空间”。

核心动机：
- 传统梯度下降默认使用欧氏几何（`L2`）；
- 许多约束集合（例如概率单纯形）在欧氏几何下投影代价高或数值行为一般；
- 镜像下降可通过选取合适的镜像函数，让更新天然匹配问题几何。

本条目给出最小可运行 MVP：
- 在概率单纯形上优化凸二次目标；
- 使用负熵作为镜像函数（对应 KL/Bregman 几何）；
- 代码手写镜像下降主循环，不把算法交给第三方黑盒。

## R02

本目录 demo 的问题定义：

- 目标：
  - `min_{x in Delta_n} f(x) = 0.5 * x^T Q x + c^T x`
- 约束：
  - `Delta_n = {x in R^n | x_i >= 0, sum_i x_i = 1}`（概率单纯形）
- 输入：
  - `Q`（对称正定矩阵）、`c`、初始点 `x0`、迭代参数 `eta0/max_iter/tol_gap`
- 输出：
  - `x_last`（最后迭代点）、`x_best`（历史最优目标点）、迭代日志 `history`

`demo.py` 使用固定随机种子构造问题，无需交互输入。

## R03

镜像下降的关键数学关系：

1. 选镜像函数 `psi(x)`（严格凸可微），定义 Bregman 散度：
   - `D_psi(x, y) = psi(x) - psi(y) - <nabla psi(y), x-y>`
2. 通用镜像下降更新：
   - `x_{k+1} = argmin_{x in X} { <g_k, x> + (1/eta_k) D_psi(x, x_k) }`
3. 本实现选负熵：
   - `psi(x) = sum_i x_i log x_i`
   - 在单纯形上对应 KL 几何，更新有闭式形式。
4. 闭式更新（Exponentiated Gradient 形式）：
   - `x_{k+1,i} propto x_{k,i} * exp(-eta_k * g_{k,i})`
   - 再做归一化使 `sum_i x_{k+1,i} = 1`。
5. 最优性监控（单纯形上的线性最小化 gap）：
   - `gap_k = <x_k - s_k, g_k>`，`s_k = e_{argmin_i g_{k,i}}`
   - `gap_k -> 0` 可作为收敛指示。

## R04

算法流程（高层）：

1. 初始化 `x <- x0`，并校验其在单纯形上。  
2. 计算 `g_k = Qx_k + c` 与 `f(x_k)`。  
3. 计算单纯形一阶 gap：`gap_k = <x_k - s_k, g_k>`。  
4. 若 `gap_k <= tol_gap` 则停止。  
5. 设步长 `eta_k = eta0 / sqrt(k+1)`。  
6. 在对偶空间做更新：`log x - eta_k * g_k`。  
7. 通过指数与归一化映射回单纯形得到 `x_{k+1}`。  
8. 记录日志、更新最优点，继续下一轮。

## R05

核心数据结构：

- `MirrorDescentResult`（`dataclass`）：
  - `x_last`：最后迭代点；
  - `x_best`：历史最佳目标值对应点；
  - `history`：字典日志；
  - `iterations`：执行轮数；
  - `converged`：是否按 gap 阈值提前收敛。
- `history` 字段：
  - `objective`：每轮目标值；
  - `gap`：每轮一阶最优性 gap；
  - `step_size`：每轮步长 `eta_k`；
  - `step_norm`：`||x_{k+1} - x_k||_2`。

## R06

实现正确性要点：

- 可行性保持：
  - `exp` 后归一化，保证每轮 `x_i >= 0` 且 `sum_i x_i = 1`。
- 梯度正确：
  - 二次目标梯度为 `Qx + c`。
- 优化方向正确：
  - 负熵镜像下降等价于乘法权重更新，沿着降低线性化目标方向迭代。
- 收敛性检查可解释：
  - 使用单纯形 gap 作为停止指标。
- 可核验：
  - 用 `scipy.optimize.minimize(SLSQP)` 求同一问题参考解，比较目标差。

## R07

复杂度分析（维度 `n`，稠密 `Q`）：

- 单轮成本：
  - 梯度 `Qx + c`：`O(n^2)`；
  - 指数归一化更新：`O(n)`；
  - gap 计算：`O(n)`。
- 单轮总复杂度：`O(n^2)`（由矩阵向量乘主导）。
- 总时间复杂度：`O(T * n^2)`。
- 空间复杂度：
  - 存储 `Q`：`O(n^2)`；
  - 向量状态与中间变量：`O(n)`；
  - 历史日志：`O(T)`。

## R08

边界与异常处理：

- `Q` 非方阵或与 `c/x0` 维度不匹配：抛 `ValueError`。  
- `max_iter <= 0`、`eta0 <= 0`、`tol_gap <= 0`：抛 `ValueError`。  
- `x0` 不在单纯形上：抛 `ValueError`。  
- 迭代过程中若出现非有限数值（`nan/inf`）：抛 `RuntimeError`。  
- 为避免 `log(0)`，对 `x` 用小常数 `eps` 裁剪后再进入对偶更新。

## R09

MVP 取舍说明：

- 只实现“单纯形 + 负熵镜像映射”的最经典配置。  
- 主流程完全手写，不调用现成优化器黑盒执行镜像下降。  
- 仅用 `SLSQP` 作为参考最优值验证，不参与主算法迭代。  
- 不加入随机梯度、加速镜像下降或大规模稀疏工程优化，优先保持代码短小可审计。

## R10

`demo.py` 主要函数职责：

- `quadratic_objective`：计算目标值。  
- `quadratic_gradient`：计算梯度 `Qx+c`。  
- `is_on_simplex`：检查向量是否满足单纯形约束。  
- `simplex_linear_oracle`：求 `argmin_s <g,s>`（用于 gap）。  
- `mirror_descent_entropy_simplex`：镜像下降主循环。  
- `solve_reference_with_slsqp`：调用 SLSQP 求参考解。  
- `build_problem`：构造可复现实验问题。  
- `main`：运行算法、打印指标、执行质量断言。

## R11

运行方式：

```bash
cd Algorithms/数学-凸优化-0398-镜像下降法
uv run python demo.py
```

脚本无命令行参数、无交互输入。

## R12

输出字段说明：

- `dimension n`：变量维度。  
- `iterations`：实际迭代轮数。  
- `converged by gap`：是否触发 `gap <= tol_gap`。  
- `initial objective`：初始目标值。  
- `final objective (best MD)`：镜像下降历史最优目标值。  
- `reference objective (SLSQP)`：参考最优目标值。  
- `objective diff`：镜像下降与参考解目标差。  
- `final gap / min gap`：最后一轮与历史最小一阶 gap。  
- `sum(x)-1 error`、`min(x)`：可行性检查指标。  
- `objective decrease`：相对初值的下降量。

## R13

最小测试设计（脚本内置）：

- 固定随机种子生成一个中等维度（默认 `n=25`）凸二次问题。  
- 初值取均匀分布，验证镜像更新始终保持在单纯形上。  
- 终态与 `SLSQP` 参考解比较目标值误差。  
- 若可行性、下降性或精度阈值不满足，脚本直接抛错。

可扩充测试：
- 更高维（如 `n=200`）观察速度和稳定性；
- 更强病态 `Q` 观察步长策略鲁棒性；
- 改为随机梯度近似验证在线版本。

## R14

关键参数与调参建议：

- `eta0`：基础步长（默认 `1.2`）。  
- `max_iter`：最大迭代（默认 `6000`）。  
- `tol_gap`：gap 收敛阈值（默认 `1e-8`）。  
- `eps`：数值稳定常数（默认 `1e-15`）。

调参经验：
- 下降慢：适当增大 `eta0` 或增大 `max_iter`；
- 振荡明显：减小 `eta0`；
- 追求更高精度：收紧 `tol_gap` 并提高 `max_iter`。

## R15

与相关方法对比：

- 对比投影梯度下降：
  - 投影梯度每轮要做欧氏投影；
  - 镜像下降用 KL 几何更新，单纯形上可闭式归一化。  
- 对比 Frank-Wolfe：
  - Frank-Wolfe 解线性子问题并做凸组合；
  - 镜像下降直接用梯度信息做乘法权重更新。  
- 对比次梯度法：
  - 次梯度法几何通常是欧氏；
  - 镜像下降可用问题匹配几何改善常数和稳定性。

## R16

典型应用场景：

- 概率分布优化（单纯形约束）。  
- 在线学习与专家加权（乘法权重更新可视作镜像下降特例）。  
- 稀疏组合/混合权重优化。  
- 大规模凸优化中的一阶可行迭代基线。

## R17

可扩展方向：

- 换镜像函数：欧氏势、`p`-范数势、矩阵熵等。  
- 加入随机镜像下降（stochastic MD）支持大数据流。  
- 加入强凸场景的步长策略与理论界可视化。  
- 增加日志落盘（CSV）与收敛曲线绘图。  
- 扩展到鞍点问题与原始-对偶镜像下降。

## R18

`demo.py` 源码级算法流程拆解（8 步）：

1. `main` 调用 `build_problem` 构造固定随机种子的凸二次目标 `Q,c` 与单纯形初值 `x0`。  
2. 进入 `mirror_descent_entropy_simplex`，先做维度、超参数、初值可行性检查。  
3. 每轮计算当前梯度 `g_k = Qx_k + c` 与目标值 `f(x_k)`，写入 `history`。  
4. 通过 `simplex_linear_oracle(g_k)` 得到 `s_k=e_{argmin g_k}`，计算最优性 `gap_k=<x_k-s_k,g_k>`。  
5. 若 `gap_k <= tol_gap` 则停止；否则按 `eta_k = eta0/sqrt(k+1)` 设步长。  
6. 做镜像更新：`z = log(x_k) - eta_k g_k`，再经 `exp(z)` 与归一化得到 `x_{k+1}`。  
7. 记录 `step_norm`，并更新历史最优点 `x_best`（按目标值最小）。  
8. 迭代结束后，`main` 调用 `solve_reference_with_slsqp` 得到参考目标值，对比精度并执行可行性/下降性断言。

其中第三方 `SLSQP` 仅用于验证；镜像下降核心更新（步骤 3-7）全部由本地源码实现与可追踪。
