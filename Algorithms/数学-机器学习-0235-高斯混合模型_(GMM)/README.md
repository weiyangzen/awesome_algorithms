# 高斯混合模型 (GMM)

- UID: `MATH-0235`
- 学科: `数学`
- 分类: `机器学习`
- 源序号: `235`
- 目标目录: `Algorithms/数学-机器学习-0235-高斯混合模型_(GMM)`

## R01

高斯混合模型（Gaussian Mixture Model, GMM）用多个高斯分布的加权和来描述复杂数据分布：

`p(x) = sum_{k=1}^K pi_k * N(x | mu_k, Sigma_k)`

其中 `pi_k` 是混合权重（非负且和为 1），`mu_k` 是均值，`Sigma_k` 是协方差矩阵。

直观上，GMM 把“一个复杂簇”分解为多个“椭球形子簇”，每个样本属于各簇的概率是软分配（responsibility），不是硬划分。

## R02

本目录 MVP 要解决的问题：

给定无标签样本 `X in R^{N x D}` 和成分数 `K`，估计参数集：
- `pi = (pi_1, ..., pi_K)`
- `mu_1...mu_K`
- `Sigma_1...Sigma_K`

目标是最大化观测数据对数似然：

`L(theta) = sum_{i=1}^N log( sum_{k=1}^K pi_k * N(x_i | mu_k, Sigma_k) )`

输出包括：
- EM 收敛过程的对数似然轨迹；
- 最终参数（权重、均值、协方差）；
- AIC/BIC；
- 与合成真标签对齐后的聚类准确率（通过标签置换求最优映射）。

## R03

概率建模要点：
- `z_i` 是隐变量，表示第 `i` 个样本来自哪个高斯成分；
- 完整数据形式为 `p(x_i, z_i=k) = pi_k * N(x_i | mu_k, Sigma_k)`；
- 观测数据边缘化后得到混合分布；
- 难点在于 `log(sum(...))` 结构导致无法直接一次闭式求解全部参数。

因此采用 EM（Expectation-Maximization）做迭代极大似然估计。

## R04

EM 的核心更新公式（全协方差版本）：

E 步（计算责任度）：

`gamma_{ik} = p(z_i=k|x_i,theta) = pi_k * N(x_i|mu_k,Sigma_k) / sum_j pi_j * N(x_i|mu_j,Sigma_j)`

M 步（用软分配重估参数）：

- `N_k = sum_i gamma_{ik}`
- `pi_k = N_k / N`
- `mu_k = (1/N_k) * sum_i gamma_{ik} * x_i`
- `Sigma_k = (1/N_k) * sum_i gamma_{ik} * (x_i-mu_k)(x_i-mu_k)^T + reg*I`

其中 `reg*I` 是数值稳定项，防止协方差奇异。

## R05

`demo.py` 的高层流程：

1. 生成 2 维、3 成分的可复现合成数据（含真实参数和真标签）。
2. 输入校验（维度、有限值、超参数合法性）。
3. 用 kmeans++ 风格策略初始化均值，权重均匀，协方差初始化为全局协方差。
4. 循环执行 E 步与 M 步，记录每轮对数似然。
5. 以 `|L_t - L_{t-1}| < tol` 判定收敛，或达到 `max_iter` 停止。
6. 输出最终似然、AIC/BIC、估计参数和收敛轨迹尾部。
7. 用“标签置换后的最佳准确率”评估聚类质量。
8. 用断言做最小质量守卫（准确率阈值、权重正值）。

## R06

正确性与数值稳定性关键点：
- E 步使用 `logsumexp`，避免直接在概率空间下溢；
- 多元高斯密度采用 Cholesky 分解计算二次型与 `log|Sigma|`，避免显式求逆；
- M 步每个协方差加 `reg_covar * I`，缓解奇异矩阵问题；
- 责任度按样本归一化后每行和应接近 1；
- EM 在理论上保证似然非下降（数值误差下允许极小扰动），脚本打印“负增量计数”用于审计。

## R07

复杂度分析（样本数 `N`，维度 `D`，成分数 `K`，迭代数 `T`）：
- E 步：每成分一次高斯 logpdf，主要成本约 `O(K * (D^3 + N*D^2))`；
- M 步：均值与协方差更新约 `O(K * N * D^2)`；
- 总体：`O(T * K * (D^3 + N*D^2))`。

空间复杂度：
- 数据 `O(ND)`；
- 责任度矩阵 `O(NK)`；
- 参数 `O(KD^2)`。

## R08

边界与异常处理：
- `x` 必须是 2D 且全有限值；
- `n_components` 必须在 `[1, N]`；
- `reg_covar > 0`，`max_iter >= 1`，`tol > 0`；
- 若 `n_components` 太大，标签置换精度函数限制 `K <= 8`（避免阶乘爆炸）；
- 协方差通过正则化与 Cholesky 路径处理数值不稳定。

## R09

MVP 取舍说明：
- 训练主流程为纯 `numpy` 手写 EM，避免黑盒；
- 只实现全协方差版本，不展开对角协方差/球形协方差分支；
- 不实现在线 EM、变分贝叶斯、Dirichlet Process GMM；
- 不画图，专注可运行、可审计、可复现的最小实现。

## R10

`demo.py` 主要函数职责：
- `validate_inputs`：输入与超参数合法性检查。
- `initialize_parameters` / `init_means_kmeanspp`：参数初始化。
- `gaussian_log_pdf_full_cov`：稳定计算多元高斯对数密度。
- `logsumexp`：稳定聚合 `log(sum(exp(.)))`。
- `e_step`：计算责任度与总对数似然。
- `m_step`：由责任度回归权重/均值/协方差。
- `fit_gmm_em`：EM 主循环与收敛控制。
- `sample_from_true_gmm`：生成可复现实验数据。
- `best_label_permutation_accuracy`：最佳标签映射准确率。
- `count_free_parameters` + `aic_bic`：信息准则评估。
- `main`：组织端到端实验并输出结果。

## R11

运行方式：

```bash
cd Algorithms/数学-机器学习-0235-高斯混合模型_(GMM)
python3 demo.py
```

脚本不需要交互输入，运行后直接打印训练与评估信息。

## R12

输出字段解读：
- `converged`：是否在 `max_iter` 内满足容差收敛；
- `iterations used`：实际迭代轮数；
- `final log-likelihood`：最终总对数似然，越大越好；
- `AIC/BIC`：模型选择指标（同数据下越小通常越优）；
- `best-permutation clustering accuracy`：与真标签在最佳重命名下的一致率；
- `True vs Estimated Weights`：权重恢复质量；
- `Estimated Means`：学到的簇中心；
- `Log-Likelihood Trace (tail)`：末几轮收敛细节与增量。

## R13

内置最小实验配置：
- 数据：2 维高斯混合，`K=3`；
- 样本数：`N=900`；
- 真权重：`[0.50, 0.30, 0.20]`；
- EM 参数：`max_iter=200, tol=1e-4, reg_covar=1e-6`；
- 固定随机种子，保证结果可复现。

该设置通常会得到较高聚类准确率，并在几十轮内收敛。

## R14

关键超参数与调参建议：
- `n_components`：决定模型容量，过小欠拟合、过大易过拟合；
- `reg_covar`：协方差正则强度，过小可能数值不稳，过大可能过度平滑；
- `tol`：收敛灵敏度，越小越严格但迭代更多；
- `max_iter`：上限保护，避免异常情况下长时间迭代；
- `seed`：初始化相关，影响局部最优位置。

实践建议：先固定 `K`，用多随机种子跑多次，结合 `BIC` 与业务解释性选模型。

## R15

与相关方法比较：
- 对比 K-Means：
  - K-Means 是硬分配 + 球形簇偏好；
  - GMM 是软分配，可表达椭球簇和不同方差结构。
- 对比 KDE：
  - KDE 非参数灵活但高维代价高；
  - GMM 参数化更紧凑，便于解释与采样。
- 对比变分 Bayes GMM：
  - 变分方法能抑制过拟合并自动稀疏部分成分；
  - 本 MVP 采用经典 EM，路径更直接、教学更清晰。

## R16

典型应用场景：
- 无监督聚类与软分群；
- 密度估计与异常检测（低似然样本）；
- 语音建模、目标跟踪中的状态发射分布建模；
- 作为更复杂概率模型（如 HMM 发射分布）的基础组件。

## R17

可扩展方向：
- 增加协方差结构选项：`full/diag/tied/spherical`；
- 支持多次随机重启（multi-start），选最佳似然结果；
- 加入 BIC 网格搜索自动选 `K`；
- 输出 CSV 或可视化收敛曲线与责任度热图；
- 增加缺失值处理或半监督约束版本。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 调用 `sample_from_true_gmm` 采样数据，得到 `x`、`true_labels` 和真实参数。
2. `fit_gmm_em` 先走 `validate_inputs`，再调用 `initialize_parameters` 完成权重/均值/协方差初始化。
3. 每轮 EM 的 E 步在 `e_step` 中进行：
   `gaussian_log_pdf_full_cov` 计算各成分 logpdf，`logsumexp` 做稳定归一化，得到责任度 `gamma` 与总 log-likelihood。
4. 每轮 M 步在 `m_step` 中进行：
   用 `gamma` 计算 `N_k`，更新 `weights`、`means` 和全协方差矩阵，并加 `reg_covar * I`。
5. `fit_gmm_em` 用 `|L_t - L_{t-1}| < tol` 判定收敛；若未收敛则继续下一轮，直到 `max_iter`。
6. 训练结束后，`hard_labels` 通过 `argmax(gamma)` 生成硬标签预测。
7. `best_label_permutation_accuracy` 穷举标签置换，消除“簇编号任意性”，得到可解释的聚类准确率。
8. `main` 计算 `AIC/BIC`、打印参数与似然轨迹，并通过断言执行最小正确性守卫。
