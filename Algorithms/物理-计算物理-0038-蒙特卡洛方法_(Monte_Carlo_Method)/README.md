# 蒙特卡洛方法 (Monte Carlo Method)

- UID: `PHYS-0038`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `38`
- 目标目录: `Algorithms/物理-计算物理-0038-蒙特卡洛方法_(Monte_Carlo_Method)`

## R01

蒙特卡洛方法（Monte Carlo Method）是用随机采样近似数值积分与统计量的通用框架。  
本条目给出一个可运行的计算物理 MVP：在高维立方体上估计积分

`I_d = ∫_[0,1]^d exp(-||x||^2) dx`

并比较两种采样策略（标准 MC 与对偶变量法）在方差上的差异。

## R02

为什么选这个问题：

1. 该积分有解析解，可做严格正确性对照；
2. 维度 `d` 可调，能体现 MC 在高维积分中的优势（误差率与维度弱相关）；
3. 被积函数平滑且有物理直觉（类似截断高斯权重），适合作为计算物理入门模板。

## R03

数学定义与基线估计器：

- 随机变量 `X ~ Uniform([0,1]^d)`；
- 目标积分 `I_d = E[f(X)]`，其中 `f(x)=exp(-||x||^2)`；
- 标准蒙特卡洛估计：
  `I_hat = (1/N) * Σ f(X_i)`；
- 渐近标准误（SEM）：
  `SEM ≈ s_f / sqrt(N)`，`s_f` 为样本标准差。

解析参考值：

`I_d = ( ∫_0^1 exp(-x^2) dx )^d = (sqrt(pi)/2 * erf(1))^d`。

## R04

方差降低策略：对偶变量（Antithetic Variates）

- 若 `u ~ U([0,1]^d)`，构造配对样本 `(u, 1-u)`；
- 配对估计量：
  `g(u)=0.5*(f(u)+f(1-u))`；
- 最终估计：
  `I_hat_anti=(1/M) * Σ g(u_i)`。

对于单调型被积函数，`f(u)` 与 `f(1-u)` 往往负相关，可降低方差。

## R05

复杂度分析：

- 单次函数评估成本：`O(d)`；
- 总时间复杂度：`O(N*d)`；
- 空间复杂度：`O(N*d)`（批量采样实现）；
- 对偶法在同阶复杂度下通常带来更低 RMSE。

## R06

`demo.py` 结构说明：

1. `MCConfig`：配置维度、样本数和随机种子；
2. `integrand`：实现 `exp(-||x||^2)`；
3. `exact_integral`：给解析真值；
4. `standard_mc`：标准 i.i.d. 采样估计；
5. `antithetic_mc`：对偶变量估计；
6. `run_convergence_study`：按样本数网格输出收敛表；
7. `benchmark_variance_reduction`：重复试验比较 RMSE；
8. `main`：打印结果并做断言。

## R07

正确性闭环：

- 解析解可精确对照估计偏差；
- 报告 `abs_error`、`SEM`、`95% CI`；
- 通过 `n_trials` 重复试验统计 `bias/std_of_estimate/rmse`；
- 断言检查：
  1) 大样本下误差在合理范围；
  2) 对偶法 RMSE 小于标准法。

## R08

数值稳定与可复现策略：

- 固定随机种子（`numpy.random.default_rng(seed)`）；
- 使用向量化运算减少循环误差和实现噪声；
- 所有结果由脚本内参数驱动，无交互输入；
- 最终输出固定字段，便于后续自动校验。

## R09

关键参数：

- `dim`：积分维度，默认 `6`；
- `n_grid`：收敛实验样本数网格；
- `n_trials`：方差基准重复次数；
- `seed`：随机数种子。

经验上：
- 增大 `n_samples` 时误差约按 `O(N^{-1/2})` 衰减；
- `dim` 增大时，MC 误差率不直接爆炸，但方差常数会变化。

## R10

运行方式（无交互）：

```bash
cd Algorithms/物理-计算物理-0038-蒙特卡洛方法_(Monte_Carlo_Method)
uv run python demo.py
```

或从仓库根目录直接运行：

```bash
uv run python Algorithms/物理-计算物理-0038-蒙特卡洛方法_(Monte_Carlo_Method)/demo.py
```

## R11

输出内容包含两部分：

1. **Convergence Study**  
   列出不同 `n_samples` 下两种方法的 `estimate / abs_error / sem / ci95`。
2. **Variance Reduction Benchmark**  
   在固定 `n_samples` 下做多次重复，比较 `bias / std_of_estimate / rmse`。

## R12

本 MVP 使用的度量：

- `abs_error = |estimate - exact|`；
- `sem`（标准误）与 `95%` 置信区间；
- `bias = E[estimate] - exact`（经验估计）；
- `std_of_estimate`（重复试验中的估计波动）；
- `rmse = sqrt(E[(estimate-exact)^2])`。

## R13

适用场景：

- 高维积分与期望估计的教学与原型验证；
- 作为更复杂蒙特卡洛（MCMC、QMC、粒子方法）的最小起点；
- 用于测试方差降低技巧是否有效。

不适用场景：

- 需要极低方差但预算很小（需更高级采样设计）；
- 目标分布高度集中或稀有事件主导（应改用重要性采样/分层采样等）。

## R14

常见失效模式与排查：

1. 样本数太小导致误差和区间震荡大；
2. 只看单次运行，未做重复试验，误判算法优劣；
3. 对偶法样本数设置为奇数但未按配对处理；
4. 更换被积函数后仍沿用当前断言阈值导致误报。

## R15

可扩展方向：

1. 引入重要性采样（importance sampling）比较方差收益；
2. 增加分层采样/拉丁超立方采样；
3. 引入 Quasi-Monte Carlo（Sobol/Halton）；
4. 将目标改为配分函数比值或自由能差，连接真实物理任务；
5. 并行多链/多批次采样并汇总统计误差。

## R16

与相关算法关系：

- 与数值求积（梯形/辛普森）相比：MC 更适合中高维；
- 与马尔可夫链蒙特卡洛（MCMC）相比：本例采样独立、实现更简单；
- 与准蒙特卡洛（QMC）相比：本例是伪随机基线，更强调统计误差分析。

## R17

最小可交付能力清单（本条目已覆盖）：

1. 给出可验证的目标积分与解析真值；
2. 提供至少一种标准 MC 估计器；
3. 提供至少一种方差降低策略；
4. 输出结构化结果表（`pandas.DataFrame`）；
5. 提供可复现断言，确保脚本可自动化验证。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `main()` 固定 `dim=6`、`n_grid=[2000,5000,20000,100000]` 和基准随机种子。  
2. `exact_integral(dim)` 用 `sqrt(pi)/2 * erf(1)` 的闭式表达计算真值。  
3. `run_convergence_study()` 遍历每个 `n_samples`，分别调用 `standard_mc()` 与 `antithetic_mc()`。  
4. `standard_mc()` 生成 `N x d` 的均匀随机点，计算 `f(x)=exp(-||x||^2)`，返回均值与 `SEM`。  
5. `antithetic_mc()` 生成 `M=N//2` 个基础样本 `u`，构造 `(u,1-u)` 配对，计算配对均值并估计 `SEM`。  
6. 收敛结果聚合为 `pandas.DataFrame`，包含 `estimate/abs_error/sem/ci95_low/ci95_high`。  
7. `benchmark_variance_reduction()` 在固定样本数下重复 `n_trials` 次，统计两方法的 `bias/std_of_estimate/rmse`。  
8. `main()` 打印两张结果表并执行断言（大样本误差阈值、对偶法 RMSE 优势），最后输出 `All checks passed.`。  

说明：`numpy` 仅提供随机数与向量计算，估计器、误差统计、对偶配对与验证逻辑均在源码中显式实现，不是第三方黑盒调用。
