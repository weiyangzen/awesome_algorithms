# 因子分析

- UID: `MATH-0336`
- 学科: `数学`
- 分类: `统计`
- 源序号: `336`
- 目标目录: `Algorithms/数学-统计-0336-因子分析`

## R01

因子分析（Factor Analysis, FA）用于解释“多个观测变量为何同时相关”。其核心假设是：
- 观测向量 `x in R^d` 由少量潜在因子 `z in R^k`（`k << d`）线性生成；
- 每个观测维度还带有独立噪声（独特方差）。

经典高斯因子分析模型：
- `x = mu + Lambda z + eps`
- `z ~ N(0, I_k)`
- `eps ~ N(0, Psi)`，其中 `Psi` 为对角矩阵。

因此协方差结构可写成：`Cov(x) = Lambda Lambda^T + Psi`。这让我们把“共享相关性”和“维度私有噪声”分开建模。

## R02

本条目 MVP 解决的问题：
- 输入：矩阵数据 `X (n x d)`、因子数 `k`、EM 迭代参数（`max_iter/tol`）与数值稳定参数（`min_psi`）。
- 输出：
  - 均值 `mu`
  - 因子载荷矩阵 `Lambda (d x k)`
  - 独特方差向量 `psi (d,)`
  - 训练轨迹（平均对数似然随迭代变化）
  - 重构协方差 `Sigma_hat = Lambda Lambda^T + diag(psi)` 与误差指标。

`demo.py` 包含固定随机种子的数据生成与训练流程，无需交互输入。

## R03

核心思想是最大似然 + EM：
1. 将潜变量 `z` 视为“缺失数据”；
2. E 步在当前参数下计算 `E[z|x]` 与 `E[zz^T|x]`；
3. M 步利用这些期望更新 `Lambda` 与 `Psi`；
4. 重复直到似然增益小于阈值。

与 PCA 的关键区别：
- PCA 默认各维观测噪声同尺度，重在正交投影；
- FA 显式建模每个观测维度自己的噪声方差（`Psi` 对角），统计解释更细。

## R04

对中心化数据 `x_c = x - mu`，记 `Psi = diag(psi)`。

E 步关键量：
- `M = I_k + Lambda^T Psi^{-1} Lambda`
- `M^{-1}`
- `Beta = M^{-1} Lambda^T Psi^{-1}`
- `E[z_i | x_i] = Beta x_i`
- `E[z_i z_i^T | x_i] = M^{-1} + E[z_i|x_i] E[z_i|x_i]^T`

聚合统计：
- `S_xx = (1/n) X_c^T X_c`
- `S_xz = (1/n) sum_i x_i E[z_i|x_i]^T`
- `S_zz = (1/n) sum_i E[z_i z_i^T|x_i]`

M 步更新：
- `Lambda_new = S_xz S_zz^{-1}`
- `Psi_new = diag(S_xx - Lambda_new S_xz^T)`（再做下界截断，保证正值）

对数似然（单样本平均）按高斯分布计算：
- `Sigma = Lambda Lambda^T + Psi`
- `ll = -0.5 * (d*log(2pi) + log|Sigma| + x_c^T Sigma^{-1} x_c)`

## R05

算法流程：
1. 输入检查与数据中心化；
2. 用样本协方差初始化 `Lambda/psi`（基于特征分解）；
3. 迭代执行 E 步，得到 `S_xz/S_zz`；
4. M 步更新参数并裁剪 `psi >= min_psi`；
5. 计算平均对数似然并记录历史；
6. 若似然增益 `delta_ll` 足够小则停止；
7. 输出参数、收敛状态、指标与样例因子分数。

## R06

MVP 数据结构：
- `FAHistory`：保存 `iter`, `avg_loglike`, `delta`；
- `FAResult`：保存 `mean`, `loadings`, `psi`, `sigma`, `history`, `iterations`, `converged`, `message`；
- `SyntheticCase`：保存合成数据与真值参数（用于可重复评估）。

关键数组形状：
- `X: (n, d)`
- `Lambda: (d, k)`
- `psi: (d,)`
- `Ez: (n, k)`

## R07

正确性与数值稳定要点：
- `psi` 必须严格正，代码中对 `Psi_new` 做 `clip(min_psi)`；
- `M` 与 `S_zz` 的求逆加小尺度岭项，降低病态风险；
- 似然计算通过 Cholesky 分解实现 `logdet` 和二次型，避免直接求逆；
- 若出现 `nan/inf` 或 Cholesky 失败，立即抛错而非继续污染状态；
- EM 理论上应非降似然，实践中可因数值误差出现极小波动，代码记录 `delta` 便于审计。

## R08

复杂度（每轮 EM）：
- E 步：
  - 构造与求解 `M (k x k)`：`O(dk^2 + k^3)`
  - 计算 `Ez = X Beta^T`：`O(ndk)`
  - 聚合 `Ez^T Ez`：`O(nk^2)`
- M 步：
  - `S_xz @ inv(S_zz)`：`O(dk^2 + k^3)`
  - `S_xx` 若预计算可复用。

整体近似：`O(n d k + n k^2 + d k^2 + k^3)`。
空间复杂度：`O(nd + dk + nk)`，主项通常为数据矩阵 `X` 与 `Ez`。

## R09

边界与异常处理：
- `k` 必须满足 `1 <= k < d`；
- `X` 必须是二维有限矩阵且 `n >= 2`；
- 初始化特征值若过小，使用下界避免负平方根；
- 对角噪声若过小或为负，统一拉回 `min_psi`；
- 迭代到 `max_iter` 未收敛时返回 `converged=False` 与明确消息。

## R10

MVP 取舍说明：
- 采用手写 EM，而不是直接调用 `sklearn.decomposition.FactorAnalysis.fit`，以保证算法步骤可见可审计；
- 限定高斯线性 FA + 对角 `Psi`，不扩展到旋转（varimax）与贝叶斯 FA；
- 重点展示“训练可复现 + 指标可解释”，而非追求完整工业 API。

## R11

运行方式：

```bash
cd Algorithms/数学-统计-0336-因子分析
uv run python demo.py
```

也可使用：

```bash
python3 demo.py
```

脚本无命令行参数、无交互输入。

## R12

`demo.py` 输出字段说明：
- `Train shape / Test shape`：训练集与测试集规模；
- `Converged / Iterations / Message`：EM 终止状态；
- `Final avg loglike`：训练集最后一轮平均对数似然；
- `Train/Test avg loglike`：训练/测试平均对数似然；
- `Relative covariance error`：`||Sigma_hat - Sigma_true||_F / ||Sigma_true||_F`；
- `Mean principal cosine`：估计载荷子空间与真值子空间一致性（越接近 1 越好）；
- `First 5 uniqueness`：前 5 个独特方差估计；
- `First 5 latent scores`：部分样本在潜空间中的后验均值。

## R13

最小测试集（内置、确定性）：
1. 合成训练集
- `n_train=1200`, `d=8`, `k_true=2`
- 用固定 `Lambda_true`, `psi_true`, `mean_true` 生成样本。

2. 合成测试集
- `n_test=400`
- 与训练集同分布、不同随机流。

目标：
- 验证 EM 可稳定收敛；
- 验证协方差重构误差可控；
- 验证训练与测试对数似然一致性。

## R14

参数建议：
- `n_factors`：先基于业务先验/碎石图确定候选 `k`，再比较对数似然与泛化；
- `tol`：默认 `1e-6` 通常足够，若更关注精度可降到 `1e-7`；
- `max_iter`：默认 `200`，若似然仍持续增长可提高；
- `min_psi`：建议 `1e-6` 到 `1e-4`，太小易病态，太大则欠拟合噪声结构；
- 初始化：使用协方差特征分解比纯随机更稳定。

## R15

与相关方法对比：
- 对比 PCA：
  - PCA 强调方差最大投影；
  - FA 强调“共享因子 + 维度独特噪声”分解。
- 对比 PPCA：
  - PPCA 假设噪声各向同性 `sigma^2 I`；
  - FA 使用对角 `Psi`，表达力更强。
- 对比 ICA：
  - ICA 追求统计独立且非高斯成分；
  - FA 主要建模协方差结构与潜在线性因子。

## R16

典型应用场景：
- 心理测量与问卷维度提炼（人格、满意度等）；
- 金融资产共性风险因子抽取；
- 传感器多变量监控中的共因子解释；
- 生物统计中基因表达的低维潜机制探索。

## R17

可扩展方向：
- 加入旋转（如 varimax）提升解释性；
- 实现基于信息准则（AIC/BIC）的因子数选择；
- 引入缺失值 EM（在 E 步按观测掩码条件化）；
- 改写为 PyTorch 版本并支持小批量/自动微分联合训练；
- 增加与 `sklearn` FA 的指标对照基准。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main()` 固定随机种子生成训练/测试合成数据，给出真值 `Lambda_true/psi_true/mean_true`。  
2. 调用 `fit_factor_analysis_em()`，先执行输入检查和中心化，并基于样本协方差做参数初始化。  
3. 在每轮 E 步中，`e_step_stats()` 计算 `M, M^{-1}, Beta`，再批量求 `Ez = E[z|x]` 与 `S_xz/S_zz`。  
4. 在每轮 M 步中，用 `Lambda_new = S_xz S_zz^{-1}` 更新载荷，再用 `diag(S_xx - Lambda_new S_xz^T)` 更新独特方差。  
5. 对 `psi` 做下界截断并重构 `Sigma`，随后 `average_log_likelihood()` 通过 Cholesky 计算平均对数似然。  
6. 记录 `FAHistory(iter, avg_loglike, delta)`，当 `|delta| <= tol * max(1, |prev_ll|)` 时判定收敛。  
7. 训练结束后，`posterior_mean_factors()` 计算样本因子后验均值，用于展示潜变量得分。  
8. `evaluate_result()` 汇总协方差相对误差、子空间夹角余弦和训练/测试对数似然，并打印前几轮迭代轨迹。  
