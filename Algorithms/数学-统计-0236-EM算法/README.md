# EM算法

- UID: `MATH-0236`
- 学科: `数学`
- 分类: `统计`
- 源序号: `236`
- 目标目录: `Algorithms/数学-统计-0236-EM算法`

## R01

EM（Expectation-Maximization，期望最大化）算法用于“含隐变量”的极大似然估计问题。

当目标函数形如

`L(theta) = sum_i log(sum_z p(x_i, z | theta))`

时，`log(sum(.))` 往往难以直接优化。EM 的核心思想是交替执行：
- E 步：在当前参数下估计隐变量后验（责任度）；
- M 步：在责任度加权下更新参数。

本条目用“二项分布混合模型（混合硬币）”做一个最小、可运行、可审计的 EM 实现。

## R02

本目录 MVP 的问题定义：

- 观测数据：`x_i` 表示第 `i` 个样本在 `m` 次抛硬币中的正面次数（`x_i in {0,...,m}`）。
- 隐变量：`z_i in {1,...,K}` 表示该样本来自第 `k` 个硬币簇。
- 待估参数：
  - 混合权重 `w_k`，满足 `w_k >= 0` 且 `sum_k w_k = 1`；
  - 每个簇的正面概率 `p_k in (0,1)`。

目标：最大化观测对数似然并恢复参数，同时输出收敛轨迹与聚类一致性指标。

## R03

模型概率结构：

- 完整数据：
  `p(x_i, z_i=k | theta) = w_k * Binomial(x_i | m, p_k)`
- 观测数据边缘化：
  `p(x_i | theta) = sum_k w_k * Binomial(x_i | m, p_k)`
- 观测对数似然：
  `L(theta) = sum_i log(sum_k w_k * Binomial(x_i | m, p_k))`

直接最大化 `L(theta)` 不方便，因此采用 EM 迭代近似求解。

## R04

本实现的 E/M 更新公式：

E 步（责任度）：

`gamma_ik = p(z_i=k | x_i, theta)
= (w_k * Binomial(x_i | m, p_k)) / (sum_j w_j * Binomial(x_i | m, p_j))`

M 步（参数重估）：

- `N_k = sum_i gamma_ik`
- `w_k = N_k / N`
- `p_k = (sum_i gamma_ik * x_i) / (m * N_k)`

循环执行直到 `|L_t - L_{t-1}| < tol` 或达到 `max_iter`。

## R05

`demo.py` 的端到端流程：

1. 生成可复现的二项混合样本（含真实簇标签）。
2. 检查输入合法性（维度、取值范围、迭代超参数）。
3. 初始化 `weights` 与 `probs`。
4. E 步计算责任度矩阵和总对数似然。
5. M 步用责任度加权更新 `weights/probs`。
6. 记录每轮 log-likelihood 并执行收敛判定。
7. 输出参数估计结果、收敛尾部轨迹、负增量计数。
8. 用标签置换后的最佳准确率评估聚类质量，并做最小质量断言。

## R06

正确性与数值稳定性要点：

- 在对数空间计算混合概率，使用 `logsumexp` 防止下溢。
- `w_k` 与 `p_k` 通过 `clip` 与重新归一化约束在合法范围。
- 使用 `math.lgamma` 计算 `log C(m,x)`，避免阶乘溢出。
- 通过 `negative_ll_steps` 审计似然是否非下降（允许阈值 `1e-8`）。
- 训练结束通过断言检查参数有限性、权重和、精度与估计误差上限。

## R07

复杂度分析（`N` 样本数，`K` 成分数，`T` 迭代轮数）：

- E 步：对每个样本和成分计算一遍对数概率，时间 `O(NK)`。
- M 步：责任度加权求和，时间 `O(NK)`。
- 总训练：`O(TNK)`。
- 空间复杂度：
  - 责任度矩阵 `O(NK)`；
  - 参数向量 `O(K)`；
  - 数据 `O(N)`。

## R08

边界与异常处理：

- `head_counts` 必须是一维整数数组。
- 每个 `x_i` 必须满足 `0 <= x_i <= n_tosses`。
- `n_tosses >= 1`、`n_components >= 1`、`max_iter >= 1`、`tol > 0`。
- `best_permutation_accuracy` 限制 `K <= 8`，避免阶乘级枚举爆炸。
- 若输入不合法，立即抛出 `ValueError`，防止静默错误。

## R09

MVP 取舍说明：

- 只实现离线批量 EM，不做在线/增量版本。
- 只覆盖“固定抛掷次数 `m` 的二项混合”，不扩展到泊松/高斯等更广义混合。
- 不调用 sklearn 的黑盒 EM；E/M 主逻辑完全在源码中显式展开。
- 不绘图，优先保证脚本可运行、结果可复现、流程可审计。

## R10

`demo.py` 主要函数职责：

- `validate_inputs`：输入与超参数合法性校验。
- `logsumexp`：稳定计算 `log(sum(exp(.)))`。
- `log_binomial_coeff_vec`：计算二项系数对数项。
- `e_step`：输出责任度与总对数似然。
- `m_step`：基于责任度更新混合权重和正面概率。
- `fit_em_binomial_mixture`：EM 主循环、收敛判定与结果组织。
- `generate_synthetic_counts`：构造可复现实验数据。
- `best_permutation_accuracy`：消除簇标签置换歧义后的准确率。
- `print_trace_tail`：打印似然收敛尾部。
- `main`：组织训练、评估与断言守卫。

## R11

运行方式：

```bash
cd Algorithms/数学-统计-0236-EM算法
uv run python demo.py
```

也可用：

```bash
python3 demo.py
```

脚本无需交互输入。

## R12

输出字段说明：

- `converged`：是否在 `max_iter` 内达到容差收敛。
- `iterations_used`：实际使用迭代轮数。
- `final_log_likelihood`：最终对数似然。
- `negative_ll_steps`：似然下降步数（理想为 0）。
- `best_permutation_accuracy`：与真标签在最佳重命名下的一致率。
- `True/Estimated weights`：混合权重恢复质量。
- `True/Estimated probs`：各簇正面概率恢复质量。
- `Log-Likelihood Trace (tail)`：末几轮收敛增量。

## R13

内置最小实验配置：

- 成分数：`K=2`
- 样本数：`N=1200`
- 每样本抛掷次数：`m=12`
- 真权重：`[0.62, 0.38]`
- 真正面概率：`[0.84, 0.23]`
- EM 参数：`max_iter=200, tol=1e-7`
- 随机种子：`2026`

该配置在默认种子下通常可稳定收敛并得到较高聚类一致率。

## R14

关键超参数与调参建议：

- `n_components`：模型容量；过小欠拟合，过大可能引入冗余成分。
- `tol`：收敛阈值；更小更严格但迭代更长。
- `max_iter`：最大迭代保护上限。
- `seed`：影响初始化与局部最优落点。
- 数据侧 `n_tosses`：越大单样本信息量越高，簇可分性通常越强。

实践建议：多随机种子重启，选择最终似然更高且参数更可解释的一组。

## R15

相关方法比较：

- 对比 K-Means：
  - K-Means 是硬分配且非概率模型；
  - EM 混合模型是软分配，直接输出后验责任度。
- 对比 GMM-EM：
  - GMM 处理连续特征并估计均值/协方差；
  - 本例是离散计数（Binomial）版本，更贴近统计入门场景。
- 对比 MCMC：
  - MCMC 给后验采样分布，表达更完整但计算更重；
  - EM 提供点估计，收敛快、工程实现更轻量。

## R16

典型应用场景：

- 隐类模型参数估计（用户分群、题目作答风格分层）。
- 含缺失或隐状态的统计学习任务。
- 教学场景下演示“隐变量 + 极大似然”迭代优化框架。
- 作为更复杂模型（GMM/HMM/潜变量模型）的前置理解与原型验证。

## R17

可扩展方向：

- 从二项混合扩展到泊松混合、负二项混合。
- 增加多随机重启（multi-start）与最佳似然挑选。
- 加入 AIC/BIC 做成分数 `K` 选择。
- 将观测从“计数”扩展到“逐次 0/1 序列”，支持更细粒度建模。
- 引入先验（MAP-EM）提高小样本稳定性。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `main` 调用 `generate_synthetic_counts` 采样数据，得到 `head_counts` 与 `true_labels`。
2. `fit_em_binomial_mixture` 先执行 `validate_inputs`，并初始化 `weights/probs`。
3. `log_binomial_coeff_vec` 预计算 `log C(m, x_i)`，供每轮 E 步复用。
4. 每轮 E 步在 `e_step` 中计算每个样本对每个成分的 `weighted_log_prob`，再经 `logsumexp` 得到责任度 `gamma` 与总 log-likelihood。
5. 每轮 M 步在 `m_step` 中用 `gamma` 更新 `w_k` 与 `p_k`，并执行裁剪与归一化约束。
6. `fit_em_binomial_mixture` 依据 `|L_t-L_{t-1}| < tol` 判定收敛；未收敛则继续下一轮，直到 `max_iter`。
7. 训练结束后，`main` 对责任度做 `argmax` 得到硬标签预测，并在 `best_permutation_accuracy` 中穷举标签置换计算可解释准确率。
8. `main` 输出参数与收敛轨迹，统计似然负增量次数，并通过断言执行最小质量守卫。
