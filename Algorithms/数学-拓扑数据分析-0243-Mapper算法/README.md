# Mapper算法

- UID: `MATH-0243`
- 学科: `数学`
- 分类: `拓扑数据分析`
- 源序号: `243`
- 目标目录: `Algorithms/数学-拓扑数据分析-0243-Mapper算法`

## R01

问题定义（Mapper 离散版）：
给定点云数据集 \(X=\{x_i\}_{i=1}^n\subset\mathbb{R}^d\)，以及一个实值 lens 函数
\[
f: X \to \mathbb{R},\quad x_i\mapsto f(x_i)
\]
在 \(f(X)\) 上构造重叠区间覆盖；对每个区间中的点做局部聚类；把“区间内聚类块”作为节点，若两个节点共享原始样本点，则连边，得到 Mapper 图 \(G_M=(V,E)\)。

## R02

数学背景：
- Mapper 来源于 Reeb 图思想，是“连续拓扑对象到离散图”的近似构造。
- Reeb 图按函数等值连通分量压缩空间；Mapper 通过“区间覆盖 + 局部聚类”近似这个过程。
- 覆盖需要重叠，重叠保证不同局部块之间可以通过共享样本建立拓扑连接。
- 实际中 Mapper 结构高度依赖三组参数：lens、cover、clusterer。

## R03

`demo.py` 的输入与输出：
- 输入（代码内置，无交互）：
  - `make_circles` 生成二维点云（500 点，含噪声）。
  - lens：用 PyTorch SVD 计算第一主方向投影值。
  - cover 参数：`n_intervals=8`，`overlap_ratio=0.35`。
  - 聚类参数：`DBSCAN(eps=0.11, min_samples=5)`。
- 输出：
  - Mapper 图节点数、边数、连通分量数。
  - 每个 cover 区间的边界、样本数、节点数。
  - 前若干节点的大小与 lens 范围。
  - 前若干条边列表。

## R04

核心思想：
1. 用一个低维可解释函数（这里是 1D PCA lens）给样本排序/分层。
2. 在 lens 轴上做重叠分桶，不直接在全局空间一次性聚类。
3. 每个桶内局部聚类，把“局部连通块”抽象为图节点。
4. 借由重叠区间导致的共享样本，把节点连接成全局拓扑骨架。

## R05

算法步骤（高层）：
1. 生成点云 `X`。
2. 计算 lens 向量 `f(X)`。
3. 在 `[min(f), max(f)]` 上生成等宽重叠区间。
4. 对每个区间取子样本 `X_I`。
5. 在 `X_I` 上跑 DBSCAN，忽略噪声标签 `-1`。
6. 每个聚类块生成一个 Mapper 节点，记录其原始样本索引集合。
7. 两节点样本集合若有交集，则添加无向边。
8. 统计连通分量并打印结果。

## R06

正确性要点（MVP 级）：
- 节点语义正确：每个节点都对应“某区间内一个聚类簇”。
- 边语义正确：仅当两节点共享原始点才连边，满足 Mapper 的 nerve 连接原则。
- 覆盖一致性：最后一个区间用闭区间，避免最大 lens 值丢失。
- 去重一致性：边使用有序二元组集合去重，保证图结构稳定。

## R07

复杂度分析：
设样本数 \(n\)，区间数 \(k\)，第 \(i\) 个区间样本数 \(m_i\)。
- 构造 cover：\(O(k)\)。
- 局部聚类：若采用 DBSCAN + 邻域查询，代价近似为 \(\sum_i C_{\text{db}}(m_i)\)。
- 连边阶段（当前实现）：两两节点集合求交，若节点数为 \(|V|\)，近似 \(O(|V|^2\cdot s)\)，\(s\) 为集合交操作平均成本。
- 空间复杂度主要由节点成员索引存储与中间标签构成。

## R08

边界与鲁棒性：
- `n_intervals <= 0` 或 `overlap_ratio` 不在 `[0,1)` 时抛出 `ValueError`。
- 某区间无样本时自动跳过，不创建节点。
- 某区间全部被 DBSCAN 视为噪声时，该区间节点数为 0。
- 无节点或无边时，连通分量函数有显式分支，避免稀疏图异常。

## R09

MVP 范围声明：
- 仅实现 1D lens 的基础 Mapper，不含多维 lens 组合与可视化 UI。
- 仅内置一个示例数据集，不做通用 CLI/文件输入。
- 不实现持久同调或稳定性理论分析，仅提供结构化可运行原型。

## R10

代码结构：
- `CoverInterval`：cover 区间数据结构。
- `MapperNode`：Mapper 节点（区间ID、簇标签、成员索引）。
- `MapperGraph`：节点/边/区间/统计的容器。
- `compute_lens_torch_pca1`：PyTorch SVD 计算 1D lens。
- `build_cover_intervals` / `points_in_interval`：cover 构造与样本筛选。
- `build_mapper_graph`：核心 Mapper 构图流程。
- `graph_connected_components`：用 SciPy 稀疏图计算连通分量。
- `main`：参数设定与结果打印。

## R11

运行方式：
```bash
uv run python demo.py
```

## R12

示例结果解读（默认参数）：
- 会得到一个非空图（通常 20+ 节点与若干连边）。
- 区间中部通常出现更多节点，因为重叠后样本更密且局部结构更复杂。
- 连边来自区间重叠处共享样本，能把局部簇串成全局骨架。
- 对“同心环 + 噪声”数据，Mapper 图常出现近似环状或多分支环状结构。

## R13

参数建议：
- `n_intervals`：建议 6~12；过小会过度压缩，过大会稀疏且碎片化。
- `overlap_ratio`：建议 0.2~0.5；过低连边不足，过高节点冗余。
- `dbscan_eps`：建议先从 0.08~0.15 网格试探。
- `dbscan_min_samples`：可在 4~10 调整，控制对噪声的敏感度。

## R14

可扩展方向：
- 支持多种 lens（到质心距离、密度估计、监督信号投影）。
- 支持不同 clusterer（HDBSCAN、层次聚类）。
- 引入边权（共享点数量）并导出到可视化工具。
- 将连边阶段优化为倒排索引而非节点两两求交。

## R15

建议测试清单：
- 参数有效性：非法 `n_intervals`/`overlap_ratio` 是否正确报错。
- 极端输入：单点、重复点、全噪声区间是否可运行。
- 稳定性：固定随机种子后输出规模是否稳定。
- 结构合理性：提高 `overlap_ratio` 后边数应倾向增加。

## R16

常见错误与规避：
- 错误 1：区间都用左闭右开导致最大值样本丢失。
- 错误 2：忽略噪声标签逻辑，导致把 `-1` 当作正常簇。
- 错误 3：连边按“簇中心距离”而非“成员交集”，偏离 Mapper 定义。
- 错误 4：只看节点数，不看每区间样本覆盖，难以调参。

## R17

与相关方法对比：
- 与 PCA：PCA 给连续坐标，Mapper 给离散拓扑骨架。
- 与 t-SNE/UMAP：后者侧重嵌入可视化，Mapper 显式输出图结构与局部簇连接关系。
- 与单次全局聚类：Mapper 保留了“沿 lens 分层”的局部到全局过渡信息。

## R18

`demo.py` 源码级流程拆解（8 步）：
1. `main` 调用 `make_circles` 生成带噪声环状点云 `X`。
2. `compute_lens_torch_pca1` 用 `torch.linalg.svd` 求第一主方向，并把样本投影成 1D lens。
3. `build_cover_intervals` 根据 `n_intervals` 和 `overlap_ratio` 计算等宽重叠区间；`points_in_interval` 取每个区间的样本索引。
4. `build_mapper_graph` 在每个区间内调用 `DBSCAN.fit_predict`；在 scikit-learn 源流程中，这一步会先做半径邻域查询，再按 `min_samples` 标记 core 点，并通过密度可达扩展形成簇标签。
5. 对每个非噪声簇创建 `MapperNode`，节点保存原始样本索引集合（不是区间内局部重编号）。
6. 对节点两两执行成员集合求交；只要交集非空就连边，形成 Mapper 图的 nerve 近似。
7. `graph_connected_components` 把边转为 SciPy 稀疏邻接矩阵，调用 `connected_components` 得到连通分量标签。
8. `main` 使用 pandas 组织区间/节点统计表并打印，输出可直接用于参数诊断与结构检查。
