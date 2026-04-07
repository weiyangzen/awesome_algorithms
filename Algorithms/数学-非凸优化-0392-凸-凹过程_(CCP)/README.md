# 凸-凹过程 (CCP)

- UID: `MATH-0392`
- 学科: `数学`
- 分类: `非凸优化`
- 源序号: `392`
- 目标目录: `Algorithms/数学-非凸优化-0392-凸-凹过程_(CCP)`

## R01

凸-凹过程（Convex-Concave Procedure, CCP）用于求解可写成差分凸（DC）形式的非凸问题：

`min_x  f(x) = g(x) - h(x)`，其中 `g`、`h` 都是凸函数。

核心思想是：在当前点 `x_k` 对 `h(x)` 做一阶线性近似，把原非凸目标替换成凸上界子问题并迭代求解。

## R02

本目录 MVP 选择的具体任务是 `L1-L2` 非凸稀疏回归：

`min_x  (1/(2n))||Ax-b||_2^2 + lambda1||x||_1 - lambda2||x||_2`

其中：
- `A in R^(n*d)` 是特征矩阵，`b in R^n` 是观测向量。
- `lambda1 > lambda2 >= 0`。
- `lambda1||x||_1 - lambda2||x||_2` 是典型 DC 正则项。

输出包括：估计参数、外层 CCP 轨迹、收敛状态、支持集指标与目标函数单调性检查。

## R03

为什么用这个问题演示 CCP：
- 目标函数是标准的 DC 结构，数学表达清晰。
- 非凸性来自 `-lambda2||x||_2`，不是人造复杂约束。
- 线性化后子问题是凸问题，便于用手写 ISTA 实现，不依赖黑盒优化器。
- 可以展示“稀疏 + 非凸校正”的实际效果。

## R04

DC 分解如下：

- `g(x) = (1/(2n))||Ax-b||_2^2 + lambda1||x||_1`（凸）
- `h(x) = lambda2||x||_2`（凸）

在 `x_k` 处取 `h` 的次梯度 `u_k`：
- 若 `||x_k||_2 > 0`，`u_k = x_k / ||x_k||_2`
- 若 `x_k = 0`，取 `u_k = 0`

则 CCP 子问题为：

`x_{k+1} = argmin_x (1/(2n))||Ax-b||_2^2 + lambda1||x||_1 - lambda2 * u_k^T x`

这是凸问题。

## R05

算法流程（外层 CCP + 内层 ISTA）：

1. 校验输入维度、有限性与超参数约束。
2. 初始化 `x_0 = 0`。
3. 外层迭代：基于 `x_k` 计算 `u_k`。
4. 构造凸子问题，并用 ISTA 做近端梯度迭代。
5. ISTA 每步做“梯度下降 + 软阈值”更新。
6. 记录外层指标：目标值、`||x_{k+1}-x_k||_2`、非零个数、内层迭代次数。
7. 若外层步长小于阈值则收敛。
8. 输出模型与完整历史。

## R06

理论与正确性要点：
- 由于 `h` 凸，线性化给出对 `h` 的全局下界，因此对子问题构造的是原目标的上界近似。
- 每次外层最小化该凸近似，理论上会推动原目标下降到临界点附近。
- 本实现内层不是精确解，而是有限步 ISTA；因此提供“objective almost monotone”审计，而不是绝对数学保证。

## R07

复杂度（记样本数 `n`、特征数 `d`、外层 `T`、内层 `K`）：
- ISTA 单步主要成本是矩阵向量乘，`O(nd)`。
- 单次外层迭代约 `O(Knd)`。
- 总体约 `O(TKnd)`。
- 空间复杂度约 `O(nd + d)`（存储 `A`、向量与历史）。

## R08

边界条件与异常处理：
- `A` 必须二维、`b` 必须一维且样本维匹配。
- 输入必须全为有限值（禁止 `nan/inf`）。
- 要求 `lambda1 > 0`、`lambda2 >= 0`、迭代次数与容差为正。
- MVP 中强制 `lambda1 > lambda2`，避免正则项失去足够的下界控制。

## R09

MVP 设计取舍：
- 依赖仅 `numpy`，保持最小技术栈。
- 不调用第三方黑盒优化器，CCP 和 ISTA 都在源码中显式实现。
- 使用合成数据，保证脚本可复现、可离线运行。
- 重点是算法机制透明，不追求工业级最优性能。

## R10

`demo.py` 结构说明：
- `validate_inputs`：输入和参数合法性校验。
- `soft_threshold`：ISTA 近端算子。
- `objective`：计算非凸目标值。
- `convex_subproblem_ista`：解每个 CCP 凸子问题。
- `ccp_l1_minus_l2`：外层 CCP 主循环。
- `make_synthetic_regression`：构造可重复实验数据。
- `support_metrics`：稀疏支持集 precision/recall。
- `objective_is_almost_monotone`：轨迹审计。
- `run_case` / `main`：执行两组正则参数并打印报告。

## R11

运行方式：

```bash
cd Algorithms/数学-非凸优化-0392-凸-凹过程_(CCP)
uv run python demo.py
```

脚本无交互输入，直接执行内置实验。

## R12

输出字段解释：
- `objective`：当前非凸目标函数值。
- `outer_dx`：外层相邻解的 `L2` 变化量。
- `inner_iters`：当前外层对应的 ISTA 迭代次数。
- `nnz`：估计向量中非零系数数量。
- `converged`：是否在最大外层迭代内达到阈值。
- `train mse`：拟合误差（不含正则项）。
- `coef L2 error`：与真参数的 `L2` 距离。
- `support precision/recall`：支持集恢复质量。

## R13

最小验证集（内置）：
- 固定随机种子生成稀疏线性模型数据。
- 两组参数：
  - `lambda1=0.24, lambda2=0.10`
  - `lambda1=0.30, lambda2=0.12`
- 每组打印迭代表格、收敛状态、误差指标与支持集指标。

## R14

关键参数建议：
- `lambda1`：增大可提升稀疏性，但可能增大偏差。
- `lambda2`：增大可增强非凸性，常带来更激进的变量选择。
- `outer_max_iter`：CCP 外层预算，默认 `40~50` 对小规模示例足够。
- `inner_max_iter`：子问题求解精度预算，过小会影响外层下降质量。
- `outer_tol / inner_tol`：收敛阈值，越小越严格但耗时更高。

## R15

与相关方法对比：
- 对比纯 Lasso（`lambda2=0`）：L1-L2 可能在稀疏恢复上更接近真实支持集，但优化更难。
- 对比直接次梯度法：CCP 把问题拆成“稳定的凸子问题”，通常更可控。
- 对比黑盒 DC 求解器：本实现可完整追踪每个关键计算步骤，便于教学和审计。

## R16

典型应用场景：
- 非凸稀疏建模与特征选择。
- 希望在可解释性与预测性能间做折中的中小规模回归任务。
- 作为更复杂非凸优化流程（如 MM / DC 编程）的教学起点。

## R17

可扩展方向：
- 将 ISTA 升级为 FISTA 或坐标下降，提高内层效率。
- 增加回溯线搜索，提升不同数据尺度下的鲁棒性。
- 扩展到带约束的 DC 问题（盒约束、简单线性约束）。
- 引入验证集并做 `lambda1/lambda2` 网格搜索。
- 增加与纯 Lasso、Elastic Net 的统一实验对照。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `main` 调用 `make_synthetic_regression` 生成标准化特征、稀疏真值和带噪声观测。  
2. `run_case` 以一组 `(lambda1, lambda2)` 调用 `ccp_l1_minus_l2` 启动外层 CCP。  
3. `ccp_l1_minus_l2` 先经 `validate_inputs` 校验，再计算 ISTA 步长所需的 Lipschitz 常数。  
4. 在每个外层迭代中，根据当前 `x_k` 构造 `u_k`，得到线性化项 `lambda2 * u_k`。  
5. 调用 `convex_subproblem_ista` 解决凸子问题；该函数循环执行梯度步与 `soft_threshold` 近端步。  
6. 内层停止后返回 `x_{k+1}`，外层计算 `outer_dx`、`objective`、`nnz` 并写入 `history`。  
7. 若 `outer_dx < outer_tol` 则外层收敛并返回结果；否则继续下一次线性化。  
8. `run_case` 最后计算 `mse`、系数 `L2` 误差、支持集 `precision/recall` 以及目标函数近似单调性并打印。  
