# Nesterov加速梯度

- UID: `MATH-0399`
- 学科: `数学`
- 分类: `凸优化`
- 源序号: `399`
- 目标目录: `Algorithms/数学-凸优化-0399-Nesterov加速梯度`

## R01

Nesterov 加速梯度（Nesterov Accelerated Gradient, NAG）是经典一阶凸优化加速方法。相较于普通梯度下降，它在“带前瞻的点”计算梯度，并结合动量项更新，通常能在病态（高条件数）问题上显著降低迭代次数。

本目录实现一个最小可运行 MVP：
- 使用可验证的强凸二次目标；
- 手写 `GD` 与 `NAG` 两套迭代循环；
- 在固定迭代预算下比较收敛 gap，并做自动阈值校验。

## R02

本实现优化的问题为对角强凸二次函数：

`f(x) = 0.5 * Σ_i c_i * (x_i - a_i)^2`，其中 `c_i > 0`。

- 输入：
  - 曲率向量 `c`（正数、1D）；
  - 目标点 `a`（1D）；
  - 初值 `x0`、迭代上限 `max_iter`。
- 输出：
  - `GD` 与 `NAG` 的最终解；
  - 每轮历史 `(iter, objective, grad_norm, step_norm)`；
  - 相对 objective gap、参数误差、对比统计。

该问题最优解是 `x* = a`，最优值 `f(x*) = 0`，方便直接验证正确性。

## R03

核心数学关系：

- 梯度：`∇f(x) = c ⊙ (x - a)`（`⊙` 为逐元素乘法）；
- Lipschitz 常数：`L = max(c_i)`；
- 强凸常数：`μ = min(c_i)`；
- 条件数：`κ = L / μ`。

当 `κ` 很大时，普通梯度下降会明显变慢，这正是 Nesterov 加速最有价值的场景。

## R04

两种方法的迭代公式：

1. 梯度下降（GD）
   - `x_{k+1} = x_k - (1/L) * ∇f(x_k)`

2. Nesterov 加速（强凸常动量形式）
   - `β = (sqrt(L) - sqrt(μ)) / (sqrt(L) + sqrt(μ))`
   - `y_k = x_k + β * (x_k - x_{k-1})`
   - `x_{k+1} = y_k - (1/L) * ∇f(y_k)`

MVP 采用固定步长 `1/L`，不依赖黑盒优化器。

## R05

为什么选这个测试问题：

- 目标函数是凸优化中最标准的一类强凸可微问题；
- 闭式最优解已知，便于严谨对照；
- 可通过构造 `c_i` 的跨度精确控制条件数，稳定复现实验现象；
- 代码短、逻辑透明，能突出 NAG 的“前瞻 + 动量”核心机制。

## R06

正确性依据（实现层面）：

- 梯度表达与二次目标严格一致；
- `GD` 与 `NAG` 更新均按标准公式逐行实现；
- 结果验证不依赖近似真值，而是直接对照 `x*=a`、`f*=0`；
- 设置自动检查：
  - `NAG` 最终相对 gap 必须小于 `5e-3`；
  - `NAG` gap 必须显著优于 `GD`（至少 20x，即 `nag_gap <= 0.05 * gd_gap`）。

## R07

复杂度分析（维度 `n`，迭代轮数 `T`）：

- 单轮计算：
  - 梯度 `O(n)`；
  - 目标值 `O(n)`；
  - 向量更新 `O(n)`。
- 总时间复杂度：`O(Tn)`。
- 空间复杂度：
  - 主向量 `O(n)`；
  - 历史记录 `O(T)`。

## R08

`demo.py` 核心数据结构：

- `HistoryItem = (iter, objective, grad_norm, step_norm)`；
- `CaseConfig`：每个测试样例的 `seed/n/mu/L/max_iter`；
- `MethodResult`：单个算法的 `x_final` 与 `history`；
- `metrics`：跨样例汇总结果（gap、误差、加速倍数）。

## R09

边界与异常处理：

- `curvature/target` 非 1D 或形状不一致：抛 `ValueError`；
- 数据含 `nan/inf`：抛 `ValueError`；
- `curvature` 存在非正数：抛 `ValueError`；
- `max_iter <= 0` 或 `tol <= 0`：抛 `ValueError`；
- 迭代中若目标值、梯度范数、步长范数非有限：抛 `RuntimeError`。

## R10

主要函数职责：

- `validate_problem`：输入合法性检查；
- `objective` / `gradient`：目标与梯度计算；
- `nesterov_beta_strongly_convex`：计算常动量系数 `β`；
- `run_gradient_descent`：普通梯度下降主循环；
- `run_nesterov_accelerated_gradient`：NAG 主循环；
- `build_problem`：生成可复现的高条件数测试问题；
- `print_history_preview`：打印前后若干轮日志；
- `run_case`：执行单样例并进行阈值校验；
- `main`：批量运行并输出汇总。

## R11

运行方式：

```bash
cd Algorithms/数学-凸优化-0399-Nesterov加速梯度
uv run python demo.py
```

脚本无交互输入，运行后直接打印三组样例与总览统计。

## R12

输出字段说明：

- `mu / L / condition number`：问题强凸与光滑常数；
- `Nesterov beta`：动量系数；
- `objective`：当前目标值；
- `||grad||`：梯度范数；
- `||step||`：本轮更新步长范数；
- `relative gap (GD/NAG)`：相对最优值 `f*=0` 的差距；
- `||x_gd - x*||_2` / `||x_nag - x*||_2`：参数误差；
- `speedup ratio`：`GD gap / NAG gap`，越大表示加速越明显。

## R13

内置测试样例（固定随机种子）：

- `Moderate condition number`：`n=32, mu=0.05, L=25.0, iter=120`；
- `High condition number`：`n=64, mu=0.01, L=40.0, iter=220`；
- `Very high condition number`：`n=96, mu=0.005, L=60.0, iter=320`。

样例条件数逐步增大，用于观察病态问题下 NAG 与 GD 的差异。

## R14

关键超参数与调参建议：

- `mu, L`：决定条件数 `κ`；`κ` 越大问题越难；
- `max_iter`：迭代预算；
- `tol`：提前停止阈值。

经验建议：
- 若想更明显展示加速，可增大 `κ`；
- 若要更小最终 gap，可增大 `max_iter`；
- 若希望完全固定轮数对比，可把 `tol` 设得很小（本实现默认已很小）。

## R15

与相关方法对比：

- 对比 GD：
  - GD 只用当前点梯度，收敛速度受条件数影响更大；
  - NAG 通过前瞻与动量缓解“锯齿式慢收敛”。
- 对比 Polyak 动量：
  - Polyak 常在当前点算梯度再叠加速度；
  - NAG 的关键在“先动量外推再算梯度”的前瞻机制。
- 对比二阶方法：
  - 牛顿法迭代更猛但单步成本更高；
  - NAG 保持一阶复杂度，适合大规模场景。

## R16

典型应用场景：

- 大规模机器学习中的平滑凸目标优化；
- 需要一阶方法但希望较快收敛的工程训练任务；
- 作为 Adam/Nadam 等现代优化器前置概念教学；
- 作为复杂优化器前的可解释 baseline。

## R17

可扩展方向：

- 增加一般凸版本（`t_k` 序列/FISTA 风格）并与强凸版本对比；
- 引入随机梯度得到 `SGD + Nesterov`；
- 加入回溯线搜索替代固定 `1/L`；
- 扩展到非对角二次、逻辑回归等更通用目标；
- 输出 CSV 日志与可视化收敛曲线。

## R18

`demo.py` 源码级流程拆解（8 步）：

1. `main` 调用 `build_cases`，生成三组固定 `seed/mu/L/n/max_iter` 的测试配置。  
2. `run_case` 对每个配置调用 `build_problem`，构造曲率向量 `c` 与目标点 `a`，并计算 `mu/L/kappa/beta`。  
3. `run_gradient_descent` 使用 `step=1/L` 执行 `x_{k+1}=x_k-step*grad(x_k)`，逐轮记录 objective、梯度范数、步长范数。  
4. `run_nesterov_accelerated_gradient` 使用常动量 `beta`，先算外推点 `y_k=x_k+beta(x_k-x_{k-1})`，再在 `y_k` 上做梯度步。  
5. 两个循环都调用 `objective` 与 `gradient`，并在每轮检查数值是否有限、是否满足提前停止条件。  
6. 迭代结束后，`run_case` 计算 `GD/NAG` 的最终 objective、相对 gap、参数误差（对照 `x*=a`）。  
7. `run_case` 执行硬性校验：`nag_rel_gap <= 5e-3` 且 `nag_rel_gap <= 0.05 * gd_rel_gap`，确保加速效果真实可见。  
8. `main` 汇总三组样例的最大 gap、最小加速倍数、最大参数误差，并在全部通过后输出 `All checks passed.`。  
