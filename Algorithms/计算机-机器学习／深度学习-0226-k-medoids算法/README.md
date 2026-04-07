# k-medoids算法

- UID: `CS-0104`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `226`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0226-k-medoids算法`

## R01

本条目实现一个可运行、可审计的 `k-medoids` 最小 MVP（PAM 风格），目标是给出比“黑盒一行调用”更透明的聚类流程。

- 输入：样本矩阵 `X in R^(n*d)` 与簇数 `k`。
- 输出：`k` 个 medoid（必须来自原始样本）、每个样本的簇标签、目标函数值。
- 场景：异常值较多、对鲁棒性有要求的聚类任务。

## R02

`k-medoids` 与 `k-means` 的关键区别：

- `k-means` 的中心是“均值”，可能不对应任何真实样本；
- `k-medoids` 的中心是“样本点本身”，可解释性更强；
- `k-medoids` 优化的是样本到 medoid 的距离和，在含异常点数据上通常更稳健。

## R03

目标函数（以欧氏距离为例）：

给定 medoid 集合 `M`（`|M|=k`），最小化

`J(M) = sum_i min_{m in M} dist(x_i, x_m)`

其中 `m` 必须是某个样本索引。该约束是与 `k-means` 的根本差异。

## R04

本 MVP 实现 PAM 思路中的核心局部搜索：

1. 先初始化 `k` 个 medoid；
2. 固定 medoid 做一次全量分配（最近 medoid 归属）；
3. 枚举“medoid <-> 非 medoid”交换；
4. 若某个交换降低目标值，接受最优交换；
5. 重复直到无改进。

这是一个“可解释的离散优化”过程，便于教学与调试。

## R05

为何选“最小但诚实”实现：

- 不依赖 `scikit-learn-extra` 的现成 `KMedoids` 黑盒；
- 直接展示 build + swap 更新逻辑；
- 保留基础工程元素：断言门禁、固定随机种子、结果表格化输出。

## R06

`demo.py` 的数据实验设计：

- 使用 `make_blobs` 生成 3 团高斯簇（主数据）；
- 额外注入远距离均匀分布异常点（outliers）；
- 在同一数据上同时跑 `k-medoids` 与 `k-means`，比较：
  - 聚类轮廓系数（silhouette）；
  - 对非异常样本的 ARI（Adjusted Rand Index）；
  - 目标函数/惯性指标。

## R07

核心数据结构：

- `distance_matrix (n*n)`：样本两两距离矩阵；
- `medoid_indices (k,)`：medoid 的样本索引；
- `labels (n,)`：簇标签；
- `KMedoidsResult`：封装 medoid、标签、目标值、迭代数、是否收敛。

## R08

复杂度（PAM 直观估算）：

- 一次全量分配：`O(nk)`；
- 一轮 swap 枚举：约 `k*(n-k)` 个候选；
- 每个候选需一次分配评估：`O(nk)`；
- 单轮总成本近似 `O(k*(n-k)*n*k)`，在样本很大时较重。

因此本 MVP 更偏“正确性与透明度”，不追求超大规模性能。

## R09

数值与工程稳定性：

- 使用 `float64` 距离矩阵，避免精度过低；
- 目标改进判定加入 `1e-12` 容差，避免浮点抖动导致无意义交换；
- 固定随机种子确保可复现。

## R10

函数职责划分：

- `pairwise_distance_matrix`：构建全距阵；
- `assign_to_medoids`：给定 medoid 进行分配与目标计算；
- `initialize_medoids_greedy`：贪心初始化；
- `fit_k_medoids`：PAM 风格 swap 主循环；
- `make_dataset`：生成主簇 + 异常点；
- `main`：跑实验、断言、打印报表。

## R11

运行方式（无交互）：

```bash
cd Algorithms/计算机-机器学习／深度学习-0226-k-medoids算法
uv run python demo.py
```

脚本会直接输出 medoid 位置、指标表、簇大小统计。

## R12

输出指标解释：

- `objective_like`：`k-medoids` 为距离和，`k-means` 为平方距离和（量纲不同，仅作同模型内参考）；
- `silhouette`：簇内紧凑与簇间分离的折中；
- `ARI_on_non_outliers`：只在主簇样本上衡量与真标签的一致性；
- `mean_outlier_to_nearest_medoid`：异常点到最近 medoid 的平均距离，用于观察鲁棒行为。

## R13

代码内置正确性门禁：

1. medoid 数量必须等于 `k` 且互不重复；
2. 所有标签必须在 `[0, k-1]`；
3. 目标函数必须为有限正数；
4. 在默认配置下必须收敛（`converged=True`）。

这些检查确保脚本失败时能立即暴露问题。

## R14

常见失效模式与处理：

- 失效 1：`k > n` 或输入维度错误。  
  对策：入口参数显式校验并抛出 `ValueError`。

- 失效 2：距离矩阵出现 `NaN/Inf`。  
  对策：`pairwise_distance_matrix` 中执行 `isfinite` 检查。

- 失效 3：局部最优质量不理想。  
  对策：可增加多次随机重启或改进初始化策略（如 k-medoids++）。

## R15

与相关聚类算法对比：

- 相比 `k-means`：更抗异常值、中心可解释，但计算开销更高；
- 相比层次聚类：`k-medoids` 直接控制簇数，推理与部署更轻量；
- 相比 DBSCAN：`k-medoids` 需要预设 `k`，但更适合“已知簇数”的业务场景。

## R16

可扩展方向：

- 增加 `manhattan`、`cosine` 等距离度量；
- 引入 `CLARA/CLARANS` 以提升大规模数据效率；
- 增加多次重启与并行候选评估；
- 输出簇内代表样本、簇间距离矩阵等可解释诊断。

## R17

依赖与分工：

- `numpy`：矩阵与索引计算；
- `scipy`：`cdist` 距离计算；
- `pandas`：指标和簇规模表格化；
- `scikit-learn`：数据集生成、`k-means` 基线与评估指标。

整体实现保持“小依赖、可运行、可验证”。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `make_dataset` 先用 `make_blobs` 生成 3 个主簇，再注入远离主簇的异常点，形成测试鲁棒性的样本集。  
2. `fit_k_medoids` 调 `pairwise_distance_matrix`，通过 `scipy.spatial.distance.cdist` 构建全量距离矩阵 `D`。  
3. `initialize_medoids_greedy` 在 `D` 上选首个全局 1-medoid，然后逐步加入能最大降低目标函数的候选，得到初始 medoid 集合。  
4. `assign_to_medoids` 基于当前 medoid 计算每个样本到各 medoid 的距离，做最近分配并返回目标值 `J`。  
5. 主循环枚举每个 `old_medoid` 与每个 `non_medoid` 的交换，形成 `trial` medoid 集；对每个 `trial` 重新分配并计算 `trial_objective`。  
6. 若存在更优交换，则接受“本轮最优交换”；若整轮无改进，则判定收敛并返回 `KMedoidsResult`。  
7. `main` 同时训练 `KMeans` 基线，用 `silhouette_score` 与 `adjusted_rand_score` 对比 `k-medoids` 在异常点场景下的表现。  
8. 最后输出 medoid 坐标、指标汇总表、簇大小统计，并通过断言门禁保证结果合法且可复现。  

这 8 步覆盖了“数据构造 -> 距离建模 -> 离散优化 -> 基线对照 -> 指标验证”的最小闭环。
