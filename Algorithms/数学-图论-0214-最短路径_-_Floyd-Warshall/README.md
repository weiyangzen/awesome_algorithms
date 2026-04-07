# 最短路径 - Floyd-Warshall

- UID: `MATH-0214`
- 学科: `数学`
- 分类: `图论`
- 源序号: `214`
- 目标目录: `Algorithms/数学-图论-0214-最短路径_-_Floyd-Warshall`

## R01

本条目实现 `Floyd-Warshall` 的最小可运行版本（MVP），用于**有向带权图的全源最短路径**计算。目标是：
- 给出任意点对 `(i, j)` 的最短距离矩阵；
- 给出可重建路径的 `next_hop` 表；
- 显式检测负权回路（negative cycle）；
- 使用固定样例做到“运行即出结果、无需交互输入”。

## R02

问题定义（MVP 范围）：
- 输入：
  - 节点数 `n`；
  - 边集 `edges = (u, v, w)`，允许负边权；
  - 可选 `directed` 标记（默认有向图）。
- 输出：
  - `dist[i][j]`：从 `i` 到 `j` 的最短距离（不可达为 `+inf`）；
  - `next_hop[i][j]`：最短路重建的下一跳；
  - `has_negative_cycle`：是否存在负环。

## R03

数学基础与状态定义：

1. 设 `D^(k)[i][j]` 表示从 `i` 到 `j` 的最短路径长度，且中间点只允许来自集合 `{0,1,...,k}`。  
2. 初值：`D^(-1)[i][j] = w(i,j)`（邻接矩阵；无边为 `+inf`，对角线为 `0`）。  
3. 转移方程：
`D^(k)[i][j] = min(D^(k-1)[i][j], D^(k-1)[i][k] + D^(k-1)[k][j])`。  
4. 当 `k = n-1` 时，`D^(n-1)` 即全源最短路结果。  
5. 若最终某个 `D[i][i] < 0`，则图中存在可达负权回路。

## R04

算法流程（MVP）：
1. 用边集构建 `n x n` 权重矩阵 `weight`（无边填 `inf`，对角线 `0`）。
2. `dist = weight.copy()`；初始化 `next_hop`：若 `i->j` 有边则记 `j`，否则 `-1`。
3. 枚举中间点 `k=0..n-1`。
4. 对所有 `(i,j)`，比较“直达旧值”和“经 `k` 绕行”两者：
   `dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])`。
5. 若绕行更短，则更新 `dist[i][j]`，并把 `next_hop[i][j]` 改为 `next_hop[i][k]`。
6. 三层循环结束后检查 `diag(dist) < 0` 以检测负环。
7. 按需使用 `next_hop` 重建具体最短路径。

## R05

核心数据结构：
- `numpy.ndarray (float)`：
  - `weight`：原始权重矩阵；
  - `dist`：动态更新后的最短距离矩阵。
- `numpy.ndarray (int)`：
  - `next_hop`：路径重建下一跳表，`-1` 表示不可达。
- `FloydWarshallResult`（`dataclass`）：
  - `dist`、`next_hop`、`has_negative_cycle` 三个核心输出。

## R06

正确性要点：
- 动态规划不变量：第 `k` 轮后，`dist[i][j]` 等于“仅允许经过 `0..k` 作为中间点”的最短距离；
- 由转移方程穷尽“经过 `k` / 不经过 `k`”两种情形，因此不会漏解；
- 当算法结束（`k=n-1`）时，全部节点都已允许作为中间点，得到全局最短路；
- 负环检测使用经典判据 `dist[i][i] < 0`，与 Floyd-Warshall 理论一致。

## R07

复杂度：
- 时间复杂度：`O(n^3)`（核心三层更新）；
- 空间复杂度：`O(n^2)`（`dist` 与 `next_hop`）。

说明：该方法适合稠密图与“多次点对查询”；若图稀疏且只需单源结果，通常 Dijkstra/Bellman-Ford 更经济。

## R08

边界与异常处理：
- `n <= 0`：抛出 `ValueError`；
- 边端点越界：抛出 `ValueError`；
- 非有限边权（`nan/inf`）：抛出 `ValueError`；
- 输入矩阵非方阵：抛出 `ValueError`；
- 路径重建若不可达或受负环影响导致循环，返回 `None`（避免错误路径）。

## R09

MVP 取舍说明：
- 仅使用 `numpy` 作为最小工具栈，不引入图框架黑盒；
- 保留 `next_hop`，保证不仅有距离，还可解释“具体怎么走”；
- 演示两个案例：
  - 无负环（可正常输出最短路径）；
  - 有负环（展示检测结果与解释）；
- 不实现 Johnson 重标定、稀疏优化或并行加速，先保证正确与透明。

## R10

`demo.py` 职责划分：
- `build_weight_matrix`：从边集构建权重矩阵并做输入校验；
- `floyd_warshall`：执行 Floyd-Warshall 主过程，返回结果对象；
- `reconstruct_path`：依据 `next_hop` 还原一条最短路径；
- `format_matrix`：友好打印矩阵（含 `inf`）；
- `show_paths`：批量展示若干点对的路径与代价；
- `run_case_no_negative_cycle` / `run_case_with_negative_cycle`：组织演示样例；
- `main`：无交互入口。

## R11

运行方式：

```bash
cd Algorithms/数学-图论-0214-最短路径_-_Floyd-Warshall
python3 demo.py
```

脚本不需要输入参数，会直接打印两个案例的矩阵、路径和负环判定。

## R12

输出内容解读：
- `Weight matrix`：原始图权重（无边为 `inf`）；
- `All-pairs shortest distances`：Floyd-Warshall 计算后的最短路矩阵；
- `Has negative cycle`：是否存在负权回路；
- `path s->t: ... (cost=...)`：从 `s` 到 `t` 的重建路径及总代价。

关注点：
- 若 `has_negative_cycle=True`，部分点对的“最短路”在数学上未定义（可无限变小）；
- `reconstruct_path` 在此类受污染场景会返回 `None` 以避免误导。

## R13

建议最小测试集：
- 基础连通图：检验一般最短路是否正确；
- 含负边但无负环：检验算法能否正确处理负权；
- 含负环图：检验 `dist[i][i] < 0` 检测逻辑；
- 非法输入：`n<=0`、边越界、非有限权重、非方阵输入。

## R14

可调参数：
- 图规模 `n` 与边集 `edges`；
- `directed=True/False`（是否有向）；
- `show_paths` 里查询的点对列表 `pairs`；
- 输出精度（可在 `format_matrix` 中调整格式化位数）。

实践建议：先在小图验证路径重建，再放大规模做性能观察。

## R15

方法对比：
- 对比 Dijkstra：
  - Dijkstra 适合非负权单源；
  - Floyd-Warshall 一次求全源，且可处理负边。
- 对比 Bellman-Ford：
  - Bellman-Ford 处理单源负边并可测负环；
  - Floyd-Warshall 直接给出全点对结果。
- 对比 Johnson：
  - Johnson 在稀疏大图通常更快；
  - Floyd-Warshall 实现更简洁，教学与小中规模图更直观。

## R16

典型应用场景：
- 路网/通信网的任意两点最短延迟预计算；
- 依赖图中的“最小代价闭包”分析前的距离准备；
- 静态图上的批量路径查询；
- 图论课程中全源最短路与负环判定示例。

## R17

可扩展方向：
- 记录前驱矩阵（`predecessor`）以支持更多路径分析；
- 对负环影响范围做传播标记（把受影响点对标为未定义）；
- 用 `numba`/`Cython` 或分块矩阵优化加速大规模 `O(n^3)` 计算；
- 引入随机图生成器做基准测试与回归验证。

## R18

源码级算法流（对应 `demo.py`，9 步）：
1. `run_case_*` 定义节点数与边集，调用 `build_weight_matrix` 生成初始权重矩阵。  
2. `build_weight_matrix` 将无边位置置为 `+inf`、对角线置为 `0`，并校验边合法性。  
3. `floyd_warshall` 复制权重矩阵到 `dist`，并初始化 `next_hop`（有边则下一跳为终点，缺边为 `-1`）。  
4. 进入主循环，逐个选择中间点 `k`。  
5. 用广播计算 `through_k = dist[:,k] + dist[k,:]`，得到所有 `(i,j)` 经 `k` 的候选代价。  
6. 比较 `through_k < dist`，对更优位置批量更新 `dist`。  
7. 对相同位置同步更新 `next_hop[i][j] = next_hop[i][k]`，保持路径重建一致性。  
8. 主循环结束后通过 `diag(dist) < 0` 生成 `has_negative_cycle` 标志。  
9. `show_paths` 调 `reconstruct_path` 输出示例点对路径；若不可达或疑似受负环污染则返回 `None` 并提示。  
