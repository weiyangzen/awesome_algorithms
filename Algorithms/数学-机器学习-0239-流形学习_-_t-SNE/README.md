# 流形学习 - t-SNE

- UID: `MATH-0239`
- 学科: `数学`
- 分类: `机器学习`
- 源序号: `239`
- 目标目录: `Algorithms/数学-机器学习-0239-流形学习_-_t-SNE`

## R01

t-SNE（t-distributed Stochastic Neighbor Embedding）是一种非线性降维方法，核心目标是在低维空间中尽量保留高维数据的局部邻域关系。它常用于高维特征的可视化，尤其适合观察类簇、子群与异常点的局部分布结构。

与“保留全局欧氏距离”的方法不同，t-SNE 更强调“谁和谁是近邻”。因此它通常能把高维中的局部团簇在 2D/3D 中清晰分开。

## R02

问题建模：
- 输入：样本矩阵 `X in R^(n x D)`、目标维度 `d`（常取 2）、困惑度 `perplexity`。
- 输出：嵌入矩阵 `Y in R^(n x d)`。
- 目标：令高维邻接概率 `P_ij` 与低维邻接概率 `Q_ij` 尽可能一致。

本目录的 `demo.py` 还会输出：
- KL 损失检查点（训练前中后）；
- `trustworthiness`；
- kNN 邻域保持率（Jaccard）。

## R03

t-SNE 的直觉：
1. 在高维空间中，每个点 `i` 对其他点定义一个条件概率分布 `P(j|i)`，表示“j 是 i 邻居”的概率。
2. 在低维空间中，用 Student-t 分布（自由度 1）定义 `Q_ij`，让远处点之间有更重尾的排斥能力。
3. 通过最小化 `KL(P || Q)`，让低维邻近结构尽量复现高维邻近结构。

Student-t 重尾是 t-SNE 区别于 SNE 的关键，它能缓解“拥挤问题”（crowding problem）。

## R04

关键数学形式（简化版）：

1. 高维条件概率：
`P(j|i) = exp(-||x_i-x_j||^2 * beta_i) / sum_{k!=i} exp(-||x_i-x_k||^2 * beta_i)`。
其中 `beta_i = 1/(2 sigma_i^2)`，通过二分搜索使该分布熵对应指定 `perplexity`。

2. 对称联合概率：
`P_ij = (P(j|i) + P(i|j)) / (2n)`，并令 `P_ii = 0`。

3. 低维相似度（Student-t 核）：
`num_ij = 1 / (1 + ||y_i - y_j||^2)`，`Q_ij = num_ij / sum_{a!=b} num_ab`，且 `Q_ii=0`。

4. 目标函数：
`C = KL(P || Q) = sum_ij P_ij * log(P_ij / Q_ij)`。

## R05

`demo.py` 的执行流程：
1. 读取并标准化 digits 子集数据。
2. 计算高维两两平方距离矩阵。
3. 对每个样本二分搜索 `beta_i`，得到 `P(j|i)`。
4. 对称化得到 `P_ij`。
5. 用 PCA 生成小尺度初始嵌入。
6. 使用 t-SNE 梯度更新（支持 `torch` 和 `numpy` 两条实现路径），含 early exaggeration。
7. 输出 KL 检查点、trustworthiness、kNN 保持率和坐标预览。
8. 执行断言，形成最小自动验证闭环。

## R06

复杂度（`n` 为样本数）：
- 全对距离：`O(n^2 D)`；
- 概率构造（二分搜索）：约 `O(n^2 * T)`，`T` 为二分迭代轮数；
- 每次优化迭代：`O(n^2 d)`；
- 总训练：`O(n_iter * n^2 d)`。

空间复杂度主要由 `P/Q` 和距离矩阵主导，为 `O(n^2)`。

因此本 MVP 选择中等规模样本（默认 600）来保证可运行和可解释。

## R07

优点：
- 局部结构可视化能力强，常能清晰分簇；
- 对非线性流形更友好；
- 直观适合探索式分析。

局限：
- `O(n^2)` 时间/空间开销明显，不适合超大规模原始实现；
- 结果对超参数与随机种子较敏感；
- 低维中的全局距离与簇间绝对距离不宜过度解读。

## R08

前置知识：
- 概率分布与熵（perplexity 与熵的关系）；
- KL 散度；
- 梯度下降与动量；
- 降维评估指标（如 trustworthiness）；
- kNN 邻域概念。

## R09

适用场景：
- 高维特征可视化（图像、文本向量、表格嵌入）；
- 聚类前的数据形态探索；
- 异常点和子群的结构检查。

不适用或需谨慎：
- 需要严格保持全局几何距离的任务；
- 强依赖可解释坐标轴含义的场景；
- 样本规模巨大且无法使用近似加速版本时。

## R10

正确性直觉（实现层面）：
1. `conditional_probabilities` 通过熵匹配保证每个点的局部邻域尺度自适应。
2. `joint_probabilities` 把有向邻接概率转成对称概率，形成全局一致目标。
3. 优化步骤用 `KL(P||Q)` 的显式梯度更新 `Y`，并通过 early exaggeration 在前期增强簇内吸引。
4. 若训练正确推进，KL 检查点应整体下降，且局部保真指标（trustworthiness）优于线性基线。

## R11

数值稳定与鲁棒性策略：
- 对 `P`、`Q` 施加最小值 `1e-12`，避免 `log(0)`；
- 距离矩阵裁剪到非负，减轻浮点误差；
- 每轮对嵌入去中心化，避免整体漂移；
- 采用固定随机种子保证复现；
- 在无 `torch` 环境下提供 `numpy` 后备路径，避免运行中断。

## R12

关键超参数与经验：
- `perplexity`：控制有效邻域大小，常在 5~50。
- `learning_rate`：过大易震荡，过小收敛慢。
- `early_exaggeration` 与其迭代轮数：影响早期簇分离质量。
- `n_iter`：过少会欠收敛。

本 MVP 默认：
- `perplexity=30`，`n_iter=320`，`learning_rate=70`，`early_exaggeration=4`。

## R13

理论保证说明：
- 近似比保证：N/A（t-SNE 不是组合优化近似算法）。
- 随机化成功概率界：N/A（常用实现主要依赖非凸优化经验与初始化策略）。

实践中通常通过多随机种子、指标比较与可视化一致性来评估可靠性。

## R14

常见失效模式与应对：
- 失效：KL 不下降或震荡。
  - 应对：降低学习率、延长迭代、检查标准化。
- 失效：簇碎裂或过度挤压。
  - 应对：调整 `perplexity`（增大更平滑，减小更局部）。
- 失效：不同运行差异大。
  - 应对：固定种子，并做多次运行对比。
- 失效：规模过大导致内存/时间压力。
  - 应对：子采样、分层抽样，或采用 Barnes-Hut / FFT 近似实现。

## R15

工程实践建议：
- 先用 PCA 降到 30~50 维再做 t-SNE（大规模时常见）；
- 同时报告基线（如 PCA）与局部保真指标，不只看图形；
- 保存超参数与随机种子，保证可复现实验记录；
- 不要把簇间距离直接当作真实语义距离。

## R16

相关方法对比：
- SNE：t-SNE 前身，低维核重尾不足，拥挤问题更明显。
- Isomap：强调测地距离全局结构。
- UMAP：在速度和局部/部分全局结构间常有更均衡表现。
- PCA：线性、快速、可解释，但难处理复杂非线性流形。

## R17

本目录 MVP（`demo.py`）实现说明：
- 不调用 `sklearn.manifold.TSNE` 黑盒；
- 显式实现高维概率构造、对称化、KL 优化、early exaggeration；
- 依赖覆盖 `numpy / scipy / pandas / scikit-learn / torch`（缺失时有降级策略）。

运行方式：
```bash
cd Algorithms/数学-机器学习-0239-流形学习_-_t-SNE
python3 demo.py
```

脚本无交互输入，会打印指标与嵌入预览，并执行断言。

## R18

源码级流程拆解（对应 `demo.py`，9 步）：
1. `load_demo_data` 读取并标准化 digits 子集，得到 `X, y`。
2. `pairwise_squared_distances` 计算高维两两平方距离 `D^2`。
3. `conditional_probabilities` 对每个样本行做 `beta` 二分搜索，构造 `P(j|i)`（目标 perplexity）。
4. `joint_probabilities` 把 `P(j|i)` 对称化为 `P_ij`，并做归一化与数值下界裁剪。
5. `pca_init` 生成低维初始坐标 `Y0`，并缩放到小幅度以稳定初期更新。
6. `run_tsne_mvp` 选择优化后端：优先 `optimize_embedding_torch`，否则 `optimize_embedding_numpy`。
7. 在优化循环中，每轮显式计算 `Q_ij`、`PQ=(P-Q)*num`、梯度 `4*(row_sum*Y - PQ@Y)`，并用动量更新；前 `exaggeration_iters` 轮使用 `P*early_exaggeration`。
8. `trustworthiness_metric` 与 `knn_preservation` 分别评估局部邻域保真和近邻集合一致性，并和 `PCA_2D` 基线对比。
9. `main` 输出 KL 检查点、指标表和坐标预览，再执行断言（有限性、KL 下降、局部指标达标）完成最小可验证闭环。
