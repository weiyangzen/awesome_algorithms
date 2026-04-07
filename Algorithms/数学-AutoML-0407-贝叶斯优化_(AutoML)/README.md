# 贝叶斯优化 (AutoML)

- UID: `MATH-0407`
- 学科: `数学`
- 分类: `AutoML`
- 源序号: `407`
- 目标目录: `Algorithms/数学-AutoML-0407-贝叶斯优化_(AutoML)`

## R01

贝叶斯优化（Bayesian Optimization, BO）适合“评估代价高、梯度不可得、维度中低”的黑盒优化任务。
在 AutoML 场景里，单次评估通常是一次交叉验证训练，成本远高于普通数学函数调用，因此 BO 常用于高效搜索超参数。

本目录的 MVP 目标：
- 用高斯过程（Gaussian Process, GP）作为代理模型；
- 用期望改进（Expected Improvement, EI）作为采集函数；
- 在固定预算内搜索 `KernelRidge(RBF)` 的两个超参数。

## R02

MVP 要解决的具体问题：

给定一个非线性回归数据集，最小化 5 折交叉验证的均值 RMSE：

`f(theta) = mean_k sqrt(MSE_k(theta))`

其中 `theta = (log10(alpha), log10(gamma))`，搜索区间：
- `log10(alpha) in [-6, 1]`
- `log10(gamma) in [-6, 1]`

输出：
- 每次评估的当前 RMSE、历史最优 RMSE、对应参数；
- BO 最优超参数和最优 RMSE；
- 同预算随机搜索基线结果，用于对照。

## R03

选择该设定的原因：
- AutoML 的关键挑战是“每次评估昂贵”，与 BO 适用前提一致；
- `KernelRidge(RBF)` 含 `alpha/gamma` 两个连续超参数，足够展示 BO 的核心机制；
- 对比随机搜索可直观看到“基于不确定性的探索-利用”价值。

## R04

核心数学对象：

1. GP 后验
- 已观测点 `X=[x_1,...,x_n]`，目标值 `y=[f(x_1),...,f(x_n)]^T`
- 协方差矩阵 `K = k(X, X)`
- 对候选点 `x`，后验均值与方差：

`mu(x) = k(x,X) [K + sigma_n^2 I]^{-1} y`

`sigma^2(x) = k(x,x) - k(x,X)[K + sigma_n^2 I]^{-1}k(X,x)`

2. 最小化场景的 EI
- 记当前最优值 `f_best = min_i y_i`
- 令 `I(x)=max(f_best - f(x) - xi, 0)`
- 其期望为：

`EI(x)= (f_best-mu(x)-xi) * Phi(z) + sigma(x) * phi(z)`

`z = (f_best-mu(x)-xi)/sigma(x)`

其中 `Phi/phi` 分别是标准正态 CDF/PDF。

## R05

算法流程（高层）：

1. 在搜索空间内随机采样 `n_init` 个初始点并评估 CV RMSE。
2. 用当前观测 `(X_obs, y_obs)` 拟合 GP 代理模型。
3. 随机生成一批候选点（`n_candidates`）。
4. 计算每个候选点的 EI，选取 EI 最大点作为下一次评估点。
5. 评估黑盒目标（5 折 CV RMSE），加入观测集。
6. 重复步骤 2-5，直到预算耗尽（`n_init + n_iter`）。
7. 返回全局最优观测点，并与同预算随机搜索比较。

## R06

正确性与实现一致性要点：
- 代理模型使用 `GaussianProcessRegressor`，显式拟合在真实已评估点上；
- 采集函数不是黑盒封装，`expected_improvement` 中按公式逐项计算；
- 每轮只对被选中的下一个点进行一次真实评估，符合 BO 的序贯优化框架；
- `history` 保留每次评估的当前值与历史最优值，可审计优化轨迹。

## R07

复杂度分析（设维度 `d`，已评估点数 `n`，候选点数 `m`）：

- GP 拟合主成本约 `O(n^3)`（核矩阵分解）；
- EI 计算对 `m` 个候选点，预测成本通常为 `O(mn)` 到 `O(mn^2)`（实现相关）；
- 若迭代 `T` 轮，总体近似为：
  - `sum_t O(n_t^3 + m*n_t)`，其中 `n_t = n_init + t`。

在本 MVP 中 `n` 最大仅几十，计算瓶颈往往仍是 CV 训练开销。

## R08

边界与异常处理：
- `validate_bounds` 检查维度、有限值、上下界顺序；
- `n_init/n_iter/n_candidates/total_evals` 必须为正；
- 若某次模型训练失败或出现非有限值，`evaluate_cv_rmse` 返回大惩罚 `1e6`，保证流程不中断；
- 采样下一点时会跳过与历史点几乎重合的候选，避免重复昂贵评估。

## R09

MVP 取舍说明：
- 采用“候选池 + EI 最大化”而非连续优化器求采集函数最优，代码更短更稳定；
- 仅演示单目标、连续变量、无约束 BO；
- 未实现批量 BO、异步 BO、多保真 BO、离散/条件搜索空间；
- 重点放在可运行、可读、可审计，而不是工业级调度框架。

## R10

`demo.py` 主要函数职责：
- `make_automl_dataset`：生成确定性非线性回归数据；
- `evaluate_cv_rmse`：定义 AutoML 黑盒目标（5 折 CV RMSE）；
- `build_gp_model`：构建 GP 代理模型（Matern + WhiteKernel）；
- `expected_improvement`：实现 EI 采集函数；
- `choose_next_point`：在候选池中选 EI 最大点；
- `bayesian_optimization`：BO 主循环；
- `random_search_baseline`：同预算随机搜索对照；
- `print_history_snippet`：打印轨迹摘要；
- `main`：组织实验、输出结果与检查项。

## R11

运行方式：

```bash
cd Algorithms/数学-AutoML-0407-贝叶斯优化_(AutoML)
uv run python demo.py
```

脚本无交互输入，自动完成数据生成、BO 迭代与基线对比。

## R12

输出字段说明：
- `[BO] eval=...`：每次 BO 评估日志；
- `rmse`：当前点评估值（越小越好）；
- `best_rmse`：截至当前评估的历史最优值；
- `EI`：被选中点对应的期望改进值；
- `log_alpha/log_gamma`：搜索空间中的对数参数；
- `alpha/gamma`：换算后的真实超参数；
- `improvement_vs_random`：相对随机搜索的优势（正值表示 BO 更好）。

## R13

最小测试与验收项（脚本内已覆盖）：
- 固定随机种子，保证结果可复现；
- 运行 BO 与随机搜索，预算相同；
- 输出 `all_scores_finite`，确认无 NaN/Inf；
- 输出 `history_length_ok`，确认轨迹长度等于总预算；
- 观察 `best_rmse` 下降趋势与最终最优参数是否落在合理区间。

## R14

关键参数与调参建议：
- `n_init`：初始随机点数量，过小会导致 GP 冷启动信息不足；
- `n_iter`：BO 迭代轮数，决定总预算上限；
- `n_candidates`：每轮候选池规模，越大越接近 EI 全局最大但更耗时；
- `xi`（在 `expected_improvement` 中）：探索强度，越大越偏探索；
- `bounds`：搜索空间边界，建议优先按对数尺度设置。

实务上先保证边界合理，再逐步增加 `n_iter`，通常比盲目增大模型复杂度更有效。

## R15

与常见超参数搜索方法对比：
- 网格搜索：覆盖规则但维度升高后成本爆炸；
- 随机搜索：实现最简单，但无法利用历史评估信息；
- 贝叶斯优化：通过 GP+EI 在“潜在最优”与“不确定区域”间平衡，通常在小预算下更高效。

本 MVP 同预算输出随机搜索对照，便于横向比较。

## R16

典型应用场景：
- 训练成本高的模型超参数搜索（SVM、GPR、Boosting 等）；
- 特征工程流水线参数调优；
- 需要少量试验快速找到可用参数的 AutoML 原型阶段。

## R17

可扩展方向：
- 将候选池法替换为对 EI 的梯度优化或多起点局部优化；
- 引入并行批量 BO（q-EI）；
- 支持混合搜索空间（连续 + 离散 + 条件参数）；
- 扩展到多目标 BO（性能与训练时长联合优化）；
- 接入早停/多保真策略，进一步降低评估成本。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `main` 生成非线性回归数据与固定 `KFold`，定义对数搜索空间和预算。
2. 将 `evaluate_cv_rmse` 封装为黑盒目标：给定 `(log10(alpha), log10(gamma))` 返回 5 折均值 RMSE。
3. `bayesian_optimization` 先随机采样 `n_init` 个初始点并真实评估，建立第一批观测。
4. 每轮迭代调用 `build_gp_model` 用当前观测拟合 GP 后验。
5. `choose_next_point` 在随机候选池上调用 `expected_improvement`，按 `EI` 最大选择下一点。
6. 对该点执行一次真实 CV 评估，并把 `(x_next, y_next)` 追加到观测集与 `history`。
7. 重复“拟合 GP -> 计算 EI -> 选择下一点 -> 真实评估”直到预算耗尽，返回全局最优观测。
8. `main` 再运行 `random_search_baseline`（同预算）并打印两者最优 RMSE 与改进量。
