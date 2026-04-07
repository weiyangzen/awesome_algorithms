# 层次聚类

- UID: `MATH-0231`
- 学科: `数学`
- 分类: `机器学习`
- 源序号: `231`
- 目标目录: `Algorithms/数学-机器学习-0231-层次聚类`

## R01

层次聚类（Hierarchical Clustering）是一类无监督学习方法，通过“逐步合并”或“逐步拆分”样本集合，形成一个树状聚类结构（dendrogram）。

本任务实现的是最常见的凝聚式（agglomerative）层次聚类：
- 初始时每个样本各自为一类；
- 每一步选择“距离最近”的两簇进行合并；
- 重复到只剩一个簇，或在树上切割得到目标簇数。

## R02

MVP 问题定义：

给定数据矩阵 `X in R^(n*d)`，希望在不使用标签的情况下，把 `n` 个样本分成 `k` 个簇（`2 <= k <= n`），并满足：
- 同簇样本尽量相似；
- 异簇样本尽量不相似。

层次聚类不直接优化单一全局目标函数，而是通过局部贪心地合并最近簇，构建完整合并历史，再从历史中截取 `k` 簇结果。

## R03

为什么本条目采用“手写 + 库对照”：
- 直接调用 `AgglomerativeClustering` 虽然只需几行代码，但不利于理解“簇间距离如何计算、合并历史如何形成”；
- 本 MVP 在 `demo.py` 中手写了核心合并循环与 linkage matrix 生成逻辑；
- 同时用 `scikit-learn` 和 `scipy` 做结果对照，保证实现不是黑盒猜测而是可验证流程。

## R04

数学定义（欧氏距离 + linkage）：

样本点距离：
`d(i,j) = ||x_i - x_j||_2`

给定两个簇 `A,B`，常见簇间距离定义：
- Single linkage: `D(A,B) = min_{i in A, j in B} d(i,j)`
- Complete linkage: `D(A,B) = max_{i in A, j in B} d(i,j)`
- Average linkage: `D(A,B) = mean_{i in A, j in B} d(i,j)`

本实现默认使用 `average linkage`，并支持 `single/complete/average` 三种模式。

## R05

算法高层流程（凝聚式）：

1. 每个样本初始化为独立簇。
2. 计算样本两两欧氏距离矩阵。
3. 在当前活动簇中找到簇间距离最小的一对。
4. 合并该簇对，记录一次 merge（左簇 id、右簇 id、距离、新簇样本数）。
5. 重复步骤 3-4，直到完成 `n-1` 次合并。
6. 得到 `linkage_matrix`（层次树的表格表示）。
7. 若需要 `k` 类，按合并顺序回放并在簇数降到 `k` 时停止。

## R06

正确性直觉：
- 每一步都在当前簇集合里选择最小簇间距离进行合并，因此对“当前一步”是贪心最优；
- 合并历史完整保留在 `linkage_matrix` 中，后续可在任意层切树得到不同粒度聚类；
- 该方法不保证全局最优（如最小化某个统一代价函数），但它的可解释树结构是其核心价值。

## R07

复杂度分析（朴素实现）：
- 预计算样本距离矩阵：`O(n^2 * d)` 时间，`O(n^2)` 空间；
- 共 `n-1` 轮合并，每轮扫描所有活动簇对并计算 linkage 距离，最坏约 `O(n^3)` 时间；
- 总体时间复杂度约 `O(n^3)`，空间复杂度约 `O(n^2)`。

本 MVP 选择朴素实现，优先可读性和机制透明，不做最近邻链等工程加速。

## R08

边界与异常处理：
- `X` 必须是二维且为有限值（无 `nan/inf`）；
- 样本数必须 `n >= 2`；
- `linkage` 仅允许 `single/complete/average`；
- `n_clusters` 必须在 `[2, n]`；
- 合并时若距离并列，按簇 id 字典序打破平局，保证可复现。

## R09

MVP 范围与取舍：
- 实现凝聚式层次聚类核心流程；
- 输出可用于树切割的 `linkage_matrix`；
- 支持三种常见 linkage；
- 提供与 `scikit-learn`、`scipy` 的非强依赖对照。

未实现：
- 可视化 dendrogram 图形；
- Ward linkage（需处理平方误差增量）；
- 大规模数据的高效数据结构与并行优化。

## R10

`demo.py` 关键函数说明：
- `HierarchicalClusteringMVP.fit`：执行 `n-1` 轮簇合并并构造 `linkage_matrix_`。
- `HierarchicalClusteringMVP._pairwise_euclidean`：计算样本距离矩阵。
- `HierarchicalClusteringMVP._cluster_distance`：计算簇间距离（single/complete/average）。
- `HierarchicalClusteringMVP.get_labels`：从合并历史切出 `k` 簇标签。
- `make_blobs_like_data`：生成三簇二维实验数据。
- `summarize_cluster_sizes`：统计每簇样本数。
- `run_sklearn_baseline`：可选 sklearn 对照。
- `run_scipy_baseline`：可选 scipy 对照。
- `main`：完整跑通并打印结果。

## R11

运行方式：

```bash
cd Algorithms/数学-机器学习-0231-层次聚类
python3 demo.py
```

脚本无需交互输入，直接输出聚类质量和对照信息。

## R12

输出解读：
- `dataset shape`：样本维度；
- `linkage / n_clusters`：算法配置；
- `MVP ARI vs ground truth`：与合成数据真标签的调整兰德指数（越接近 1 越好）；
- `MVP cluster sizes`：每簇样本数量；
- `First merge rows`：前若干次合并（簇 id、距离、簇大小）；
- `sklearn/scipy ARI`：库实现对照；
- `MVP vs sklearn/scipy ARI`：分簇一致性对照。

## R13

最小实验设计：
- 用固定随机种子生成 3 个高斯簇（总样本 90）；
- 使用 `average linkage` 聚成 `k=3`；
- 报告与真标签的 ARI；
- 若环境安装了 `sklearn/scipy`，同步输出两者对照与一致性分数。

该实验不追求极致指标，而是验证“手写实现逻辑正确 + 输出稳定可复现”。

## R14

关键参数与调优建议：
- `linkage`：
  - `single` 更容易形成链式簇；
  - `complete` 倾向得到更紧凑簇；
  - `average` 在二者间折中，通常更稳。
- `n_clusters`：决定切树层级，需结合业务粒度设定；
- 特征尺度：欧氏距离对尺度敏感，实战常先标准化。

## R15

与其他聚类算法对比：
- 对比 K-Means：层次聚类不要求预先固定中心迭代，也能输出多层级结构，但计算更重；
- 对比 DBSCAN：DBSCAN 擅长发现任意形状与噪声点，层次聚类更强调层次解释；
- 对比谱聚类：谱聚类在复杂流形上更强，但需要特征分解与图构建，工程代价更高。

## R16

典型应用场景：
- 客户分群与市场细分（多粒度观察）；
- 文档主题树构建、基因表达样本分组；
- 作为探索分析工具，先看层级结构再决定最终簇数。

## R17

可扩展方向：
- 加入 Ward linkage 与 cophenetic 评价；
- 使用堆/近邻链降低朴素 `O(n^3)` 开销；
- 增加标准化、降维和异常点预处理流程；
- 输出 dendrogram 图并支持阈值切树；
- 在大规模数据上采用近似层次聚类。

## R18

`demo.py` 源码级流程拆解（8 步）：

1. `main` 调用 `make_blobs_like_data` 生成固定随机种子的三簇二维数据与真标签。
2. 初始化 `HierarchicalClusteringMVP(linkage="average")`，进入 `fit(X)`。
3. `fit` 先校验输入并通过 `_pairwise_euclidean` 构造 `n x n` 距离矩阵。
4. 维护 `clusters`（簇 id -> 样本索引），循环扫描活动簇对，调用 `_cluster_distance` 计算 linkage 距离并选最小对。
5. 将最优簇对合并为新簇，记录到 `linkage_matrix_` 一行：`[left_id, right_id, distance, new_size]`。
6. 重复步骤 4-5 共 `n-1` 次，得到完整层次合并历史。
7. `get_labels(n_clusters=3)` 回放 `linkage_matrix_`，在簇数降到 3 时停止，并把簇成员映射为离散标签。
8. `main` 计算并打印 ARI、簇大小、前几条 merge 记录，再调用 `run_sklearn_baseline`/`run_scipy_baseline` 做对照一致性检查。
