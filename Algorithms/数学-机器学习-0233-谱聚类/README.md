# 谱聚类

- UID: `MATH-0233`
- 学科: `数学`
- 分类: `机器学习`
- 源序号: `233`
- 目标目录: `Algorithms/数学-机器学习-0233-谱聚类`

## R01

谱聚类（Spectral Clustering）是把样本看成图上的节点，先基于样本相似度构造图，再利用图拉普拉斯矩阵的特征向量把原始非线性可分问题映射到低维“谱空间”，最后在该空间中做传统聚类（通常是 K-Means）的方法。

## R02

核心思想是“先图化，再线性化”：

1. 原空间中簇可能是弯曲或环形，K-Means 直接分会失败。
2. 通过相似度图把“局部近邻关系”编码进邻接矩阵。
3. 拉普拉斯矩阵的前几个特征向量能刻画图的连通结构。
4. 在特征向量空间中，原本复杂形状的簇变得更易被线性划分。

## R03

输入与输出：

- 输入: 样本矩阵 `X in R^(n x d)`，簇数 `k`，建图参数（如 `n_neighbors` 或 RBF 的 `gamma`）。
- 输出: 每个样本的簇标签 `labels in {0, ..., k-1}^n`。

## R04

常用记号：

- `W`: 邻接/相似度矩阵，`W_ij >= 0`。
- `D`: 度矩阵，`D_ii = sum_j W_ij`。
- `L = D - W`: 非归一化拉普拉斯。
- `L_sym = I - D^(-1/2) W D^(-1/2)`: 对称归一化拉普拉斯。
- `U`: 由最小的 `k` 个特征值对应特征向量组成的矩阵。

## R05

与图切分目标的关系：

- 谱聚类可看作 RatioCut 或 Ncut（Normalized Cut）问题的松弛解法。
- 直接求离散划分是组合优化难题，通常 NP-hard。
- 放宽离散约束后，问题转化为拉普拉斯特征分解，得到连续解，再通过 K-Means 离散化。

## R06

标准流程（对称归一化版本）：

1. 按近邻或核函数构造相似度图 `W`。
2. 计算 `D` 与 `L_sym`。
3. 求 `L_sym` 最小的 `k` 个特征向量，得 `U`。
4. 对 `U` 每一行做 L2 归一化。
5. 在行向量上运行 K-Means，得到最终簇标签。

## R07

正确性直觉：

- 理想情况下若图由 `k` 个互不连通子图组成，拉普拉斯会有 `k` 个零特征值。
- 对应特征向量在各连通分量上近似常数，因此样本在谱空间自然分块。
- 实际数据通常“近似分块”，所以前 `k` 个特征向量仍能提供稳定的簇结构线索。

## R08

复杂度（粗略）：

- 建图: 稠密图约 `O(n^2)`，kNN 图配合近邻搜索可降到更实用规模。
- 特征分解: 稠密 `O(n^3)`；稀疏图 + Lanczos/ARPACK 常用近似 `O(mk)` 级别（`m` 为边数，依赖迭代次数）。
- K-Means: `O(nkt)`，`t` 为迭代轮数。

## R09

工程注意点：

- 图连通性很关键，过小 `n_neighbors` 可能导致图碎裂。
- `k` 的选择会直接影响特征向量数量和聚类结果。
- 当特征值间隔很小（谱间隙不明显）时，聚类可能不稳定。
- 推荐固定 `random_state` 以保证可复现。

## R10

关键超参数建议：

- `n_clusters`: 通常先由业务先验或评估指标设定。
- `n_neighbors`: 常见从 `8~20` 起试；样本噪声大时可适度增大。
- `affinity`: `nearest_neighbors` 更强调局部结构；`rbf` 对 `gamma` 敏感。
- `assign_labels`: 通常 `kmeans`，也可尝试 `discretize`。

## R11

本目录 MVP 使用设置：

- 数据: `demo.py` 内置 NumPy 版本双月牙数据生成器（500 样本，噪声 0.06）。
- 建图: kNN 无权图并对称化（NumPy 实现）。
- 拉普拉斯: 对称归一化拉普拉斯 `L_sym = I - D^(-1/2) W D^(-1/2)`。
- 特征分解: `numpy.linalg.eigh` 取最小的 `k` 个特征向量。
- 离散化: 自实现 `kmeans_numpy`。
- 可选对照: 若环境安装了 scikit-learn，则额外运行 `SpectralClustering` 做对照指标。

## R12

`demo.py` 做了两件事：

1. 手写一个最小谱聚类流程（建图 -> 拉普拉斯 -> 特征向量 -> K-Means），不依赖 scipy/sklearn。
2. 在可用时与 `sklearn.cluster.SpectralClustering` 结果对比（ARI、轮廓系数、标签一致性）；不可用时会明确打印跳过原因。

## R13

运行方式：

```bash
python3 Algorithms/数学-机器学习-0233-谱聚类/demo.py
```

无交互输入，运行后直接打印评估指标与簇统计。

## R14

输出解读重点：

- `Connected components`: 图连通分量数，通常希望是 1。
- `Smallest eigenvalues`: 前 `k` 个特征值，反映谱结构。
- `ARI(MVP vs Truth)`: 手写流程和真实标签的一致度（越接近 1 越好）。
- `ARI(MVP vs sklearn)`: 手写流程与 sklearn 实现的一致度（仅当 sklearn 可用时输出）。

## R15

常见失败模式：

- 样本尺度差异过大，距离度量失真。
- 图过稀导致断连，或过密导致簇边界被抹平。
- 数据簇数与 `n_clusters` 明显不匹配。
- 噪声点过多时，谱嵌入会被扰动。

## R16

与 K-Means 对比：

- K-Means 假设簇近似凸、球形；对双月牙等非凸数据表现差。
- 谱聚类通过图结构捕捉流形形状，通常能更好处理非线性边界。
- 代价是额外的建图和特征分解开销。

## R17

可扩展方向：

- 用 RBF 权重图替代 0/1 邻接图。
- 对大规模数据采用 Nyström 近似或子采样谱方法。
- 使用自适应近邻数或局部缩放核提升鲁棒性。
- 在业务场景中可将谱嵌入后接入下游分类/检索模型。

## R18

`scikit-learn` 中 `SpectralClustering` 的源码级主流程（去黑盒化，按函数调用逻辑抽象）：

1. `SpectralClustering.fit` 先根据 `affinity` 构造邻接矩阵（如 `nearest_neighbors` 会走近邻图构造分支）。
2. 进入 `sklearn.cluster._spectral.spectral_clustering(...)` 主函数。
3. 该函数调用 `sklearn.manifold._spectral_embedding.spectral_embedding(...)` 计算图谱嵌入。
4. `spectral_embedding` 内部基于邻接矩阵构造归一化拉普拉斯，并通过 ARPACK/LOBPCG 等特征求解器取前 `k` 个特征向量。
5. 返回嵌入矩阵后，`spectral_clustering` 根据 `assign_labels` 选择离散策略。
6. 当 `assign_labels='kmeans'` 时，调用 `k_means` 在谱空间进行聚类并得到最终标签。
7. `fit` 保存 `labels_`，对外提供与其他 sklearn 聚类器一致的接口行为。

这正对应了本目录 `demo.py` 的手写 MVP: 建图 -> 拉普拉斯/特征向量 -> 谱空间聚类。
