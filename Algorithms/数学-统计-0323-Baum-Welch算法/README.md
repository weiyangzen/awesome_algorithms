# Baum-Welch算法

- UID: `MATH-0323`
- 学科: `数学`
- 分类: `统计`
- 源序号: `323`
- 目标目录: `Algorithms/数学-统计-0323-Baum-Welch算法`

## R01

Baum-Welch 算法是离散隐马尔可夫模型（HMM）的经典参数学习方法，本质是把 EM（Expectation-Maximization）应用到序列隐变量模型。

在仅给出观测序列 `o_1:T`、看不到隐藏状态 `z_1:T` 的情况下，它迭代估计 HMM 三组参数：
- 初始分布 `pi`
- 状态转移矩阵 `A`
- 发射矩阵 `B`

本目录提供一个可直接运行的最小 MVP：从零实现 scaled 前向-后向 + Baum-Welch 训练，并验证对数似然单调不下降。

## R02

问题定义（本实现范围）：
- 输入：
  - 离散观测序列 `obs`（取值范围 `0..M-1`）；
  - 状态数 `N`、观测符号数 `M`；
  - 最大迭代次数 `max_iters`、收敛阈值 `tol`。
- 输出：
  - 学得参数 `pi_hat, A_hat, B_hat`；
  - 训练轨迹 `log_likelihood_trace`；
  - 迭代轮数 `iterations_run`。

`demo.py` 不依赖交互输入，自动生成一段合成观测序列进行训练演示。

## R03

数学形式（离散 HMM）：
- 隐状态 `z_t in {1..N}`，观测 `o_t in {0..M-1}`。
- 参数：
  - `pi_i = P(z_1=i)`
  - `A_ij = P(z_{t+1}=j | z_t=i)`
  - `B_i(k) = P(o_t=k | z_t=i)`

Baum-Welch 每轮分两步：
1. E 步：在当前参数下计算
- `gamma_t(i) = P(z_t=i | o_1:T)`
- `xi_t(i,j) = P(z_t=i, z_{t+1}=j | o_1:T)`
2. M 步：用期望计数更新参数
- `pi_i <- gamma_1(i)`
- `A_ij <- sum_{t=1}^{T-1} xi_t(i,j) / sum_{t=1}^{T-1} gamma_t(i)`
- `B_i(k) <- sum_{t:o_t=k} gamma_t(i) / sum_{t=1}^{T} gamma_t(i)`

## R04

E 步实现细节（本目录代码）：
- 使用 scaled 前向后向，避免概率连乘下溢：
  - 前向：每个时刻保存缩放因子 `c_t` 并归一化 `alpha_hat[t]`；
  - 后向：用对应 `c_{t+1}` 反缩放得到 `beta_hat[t]`；
  - `log P(o_1:T) = sum_t log(c_t)`。
- 然后计算：
  - `gamma[t] = normalize(alpha_hat[t] * beta_hat[t])`
  - `xi[t] = normalize(alpha_hat[t][:,None] * A * (B[:,o_{t+1}] * beta_hat[t+1])[None,:])`

这保证了 E 步在长序列下仍稳定可算。

## R05

复杂度分析（单条长度 `T` 序列，`N` 状态，`M` 观测符号）：
- 前向：`O(T*N^2)`
- 后向：`O(T*N^2)`
- `xi` 计算：`O((T-1)*N^2)`
- M 步更新：
  - 转移矩阵 `A`：`O(T*N^2)`（聚合 `xi`）
  - 发射矩阵 `B`：`O(T*N + N*M)`
- 单轮总时间：`O(T*N^2)`
- 总时间：`O(I*T*N^2)`（`I` 为 EM 迭代轮数）
- 空间：`O(T*N + (T-1)*N^2)`（保存 `gamma/xi`）

## R06

核心数据结构：
- `FBResult`：
  - `alpha_hat, beta_hat, gamma, xi, c, log_likelihood`
- `BaumWelchResult`：
  - `initial_pi/A/B`
  - `learned_pi/A/B`
  - `log_likelihood_trace`
  - `iterations_run`
- `obs: np.ndarray[int]`：观测序列

这些结构使“E 步统计量”和“训练结果”分离，便于调试与复用。

## R07

正确性与可验证性要点：
- 每轮更新后 `pi`、`A`、`B` 都保持概率约束（非负、和为 1）；
- 对同一序列，EM 迭代的观测对数似然应单调不下降（数值误差内）；
- `gamma` 与 `xi` 满足边缘一致性：
  - `sum_j xi_t(i,j) = gamma_t(i)`
  - `sum_i xi_t(i,j) = gamma_{t+1}(j)`

`demo.py` 对以上性质都做了断言检查。

## R08

边界与异常处理：
- 非法概率参数（负值、和不为 1）立即报错；
- `obs` 非一维或符号越界报错；
- 序列长度 `<2` 报错（无法构造 `xi`）；
- 缩放因子过小或归一化失败时报错，避免静默 NaN 传播；
- 若检测到 EM 轮间对数似然下降（超出容差）则抛 `RuntimeError`。

## R09

MVP 取舍说明：
- 仅实现“离散发射 + 单条观测序列”版本，聚焦 Baum-Welch 主体机制；
- 不调用 `hmmlearn` 等黑盒训练器，核心公式全部源码展开；
- 使用 `numpy` 作为最小工具栈，降低依赖复杂度；
- 采用合成数据演示，保证脚本开箱即跑且结果可复现。

## R10

`demo.py` 主要函数职责：
- `_assert_prob_vector/_assert_row_stochastic`：概率约束检查；
- `_normalize_vector/_normalize_rows`：稳健归一化工具；
- `_random_hmm_params`：随机初始化 HMM 参数；
- `sample_hmm_sequence`：从给定 HMM 采样状态与观测；
- `forward_backward_scaled`：计算 `gamma/xi/log-likelihood`；
- `baum_welch_train`：EM 主循环（E 步 + M 步）；
- `main`：组织样例、打印结果、执行断言。

## R11

运行方式：

```bash
cd Algorithms/数学-统计-0323-Baum-Welch算法
python3 demo.py
```

脚本无命令行参数、无交互输入，运行后会打印训练过程与校验结果。

## R12

输出内容解读：
- `initial/final log-likelihood`：训练前后观测对数似然；
- `total improvement`：EM 提升幅度；
- `log-likelihood trace`：每轮似然轨迹（用于检查单调性）；
- `True/ Learned pi, A, B`：真实参数与学习参数对比；
- `MAP smoothed states`：学习后模型给出的平滑后验最可能状态序列片段。

说明：状态标签具有置换不变性，学习出的状态顺序可能与真值不同，但似然与统计结构仍可正确。

## R13

建议最小测试集：
1. 正常样例：中等长度序列，验证可收敛且似然上升。  
2. 参数合法性测试：传入行和不为 1 的 `A/B`，应抛异常。  
3. 观测越界测试：`obs` 含负值或超出 `M-1`，应抛异常。  
4. 短序列测试：`len(obs)<2`，应抛异常。  
5. 数值一致性测试：检查 `gamma/xi` 归一化与边缘恒等式。

## R14

可调参数与建议：
- `max_iters`：最大 EM 轮数（默认 40/50）；
- `tol`：似然增益停止阈值（默认 `1e-6`）；
- `seed`：初始化随机种子，影响局部最优落点；
- `seq_len`：训练序列长度，越长通常估计越稳定。

实践建议：
- 多随机种子重启，选取最高最终似然；
- 若出现退化状态，可加先验平滑或多序列联合训练。

## R15

与相关算法关系：
- 前向-后向：只做推断（E 步），不更新参数；
- Viterbi training：用最优单路径近似计数，通常更快但精度/稳定性不如 EM；
- Baum-Welch：使用后验期望计数，理论上每轮不降低观测似然；
- 变分推断 / MCMC：适用于更复杂隐变量模型，但实现与计算代价更高。

## R16

典型应用场景：
- 语音识别中的音素状态转移学习；
- 生物序列（DNA/蛋白）区域状态建模；
- 用户行为序列的隐状态分段；
- 时间序列离散事件模式学习。

## R17

本目录实验设计：
- 先构造一个 3 状态、4 观测符号的“真实 HMM”；
- 从真实模型采样长度 300 的观测序列；
- 用随机初始化参数运行 Baum-Welch；
- 打印似然轨迹、参数对比，并执行一致性断言。

结果预期：
- 似然轨迹整体单调上升；
- 学习参数接近真实统计结构（可能存在状态置换）；
- 最终输出 `All checks passed.`。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 定义真实 `pi_true/A_true/B_true`，并采样得到观测序列 `obs`。  
2. `baum_welch_train` 随机初始化 `pi/A/B`，先计算初始 `log-likelihood`。  
3. 每轮先调用 `forward_backward_scaled` 执行 E 步，得到 `gamma` 与 `xi`。  
4. 由 `gamma[0]` 更新初始分布：`new_pi = normalize(gamma[0])`。  
5. 由 `xi` 与 `gamma[:-1]` 的期望计数比值更新转移矩阵 `new_A`。  
6. 按观测符号分组聚合 `gamma` 期望计数，更新发射矩阵 `new_B`。  
7. 用 `new_pi/new_A/new_B` 重新计算对数似然，检查“不下降”，并记录到 `log_likelihood_trace`。  
8. 达到收敛阈值或迭代上限后返回结果；`main` 再做概率约束、`gamma/xi` 边缘恒等式与输出展示。
