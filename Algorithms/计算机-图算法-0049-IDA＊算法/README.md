# IDA*算法

- UID: `CS-0037`
- 学科: `计算机`
- 分类: `图算法`
- 源序号: `49`
- 目标目录: `Algorithms/计算机-图算法-0049-IDA＊算法`

## R01

IDA*（Iterative Deepening A*）是把 A* 的 `f(n)=g(n)+h(n)` 估价函数与迭代加深 DFS 结合的启发式搜索算法。它不维护全局优先队列，而是以 `f` 阈值为界反复做深度优先搜索：

- 当前迭代只扩展 `f(n) <= bound` 的状态。
- 若本轮找不到目标，则把下一轮 `bound` 提升为“本轮所有被剪枝节点中最小的 `f` 值”。

因此它通常能以接近 DFS 的空间开销，获得接近 A* 的最优解能力（在启发函数可采纳时）。

## R02

要解决的问题：在带权图或状态空间中，从起点到终点求最小代价路径，同时避免 A* 在大规模问题上的高内存占用。

本目录 MVP 用网格图路径搜索示例化：

- 节点：网格坐标 `(r, c)`。
- 边：上下左右可行走邻接，边权重为 `1.0`。
- 启发函数：到目标点的曼哈顿距离。

## R03

核心思想可概括为“`f` 阈值驱动的分层 DFS”：

- 初始化 `bound = h(start)`。
- 进行一次 DFS，遇到 `f > bound` 立即剪枝。
- 若找到目标，直接返回。
- 若未找到，收集到所有超界 `f` 的最小值 `next_bound`，作为下一轮阈值。
- 重复直到找到解或 `next_bound=+inf`（无解）。

这相当于按 `f` 值逐层展开搜索前沿。

## R04

简化伪代码：

```text
IDA*(start, goal):
    bound = h(start)
    while True:
        t = DFS(start, g=0, bound)
        if t is FOUND:
            return solution
        if t == +inf:
            return NOT_FOUND
        bound = t

DFS(node, g, bound):
    f = g + h(node)
    if f > bound:
        return f
    if node == goal:
        return FOUND

    min_over = +inf
    for each successor in ordered_successors(node):
        t = DFS(successor, g + cost(node, successor), bound)
        if t is FOUND:
            return FOUND
        min_over = min(min_over, t)
    return min_over
```

## R05

正确性与最优性要点：

- 若启发函数 `h` 可采纳（不高估到目标的真实剩余代价），IDA* 首次命中的解具有最优代价。
- 迭代阈值单调递增，且每次至少提升到一个真实出现过的 `f` 值，因此不会跳过可能的最优层。
- DFS 仅做剪枝不改动路径代价定义，目标判定 `node==goal` 与代价累加 `g` 保持一致。

## R06

复杂度（设分支因子为 `b`，最优解深度为 `d`）：

- 时间复杂度：最坏近似 `O(b^d)`，并可能因多轮迭代重复访问状态而放大常数。
- 空间复杂度：`O(d)`（递归栈 + 当前路径集合），显著小于 A* 常见的 `O(b^d)` 级开放列表占用。

实际表现高度依赖启发函数质量：`h` 越贴近真实代价，迭代次数越少。

## R07

适用场景：

- 状态空间大、A* 内存压力明显（如拼图、路径规划、组合搜索）。
- 能设计可采纳且计算开销可接受的启发函数。
- 可接受“更多重复搜索换更低内存”的工程取舍。

## R08

不适用或需谨慎场景：

- 启发函数过弱：迭代次数会显著增加。
- 图中大量低代价回路：若不做路径环检测，可能出现严重重复。
- 边权为负值：`g+h` 剪枝语义会失真，不适合 IDA*。

本实现要求边权非负，并通过 `in_path` 集合避免当前 DFS 路径内成环。

## R09

本 MVP 使用的数据结构：

- `Graph = dict[Node, list[(Node, cost)]]`：邻接表。
- `path: list[Node]`：当前 DFS 路径。
- `in_path: set[Node]`：`O(1)` 路径环检测。
- `SearchResult`：封装 `found/path/cost/expanded_nodes/iterations/bounds`，便于验证。

## R10

`demo.py` 的实验设计：

- 构造 `8x10` 网格，放置若干障碍格。
- 起点 `start=(0,0)`，终点 `goal=(7,9)`。
- 使用 IDA* 求最短路径并打印：
  - 是否找到路径
  - 迭代轮数
  - 阈值序列
  - 扩展节点数
  - 路径与网格可视化

## R11

运行方式（无交互输入）：

```bash
uv run python Algorithms/计算机-图算法-0049-IDA＊算法/demo.py
```

如果已进入该目录，也可：

```bash
uv run python demo.py
```

## R12

输出解读：

- `bounds=[...]`：每轮 IDA* 的 `f` 阈值；若启发较强，通常增长较慢且轮数较少。
- `expanded_nodes`：满足 `f<=bound` 并真正展开后继的节点计数。
- `path_cost`：路径总代价；本网格边权为 1，因此通常等于 `path_length-1`。
- 网格符号：
  - `S` 起点
  - `G` 终点
  - `#` 障碍
  - `*` 求得路径
  - `.` 可通行但未在最终路径中

## R13

建议的最小测试集：

- 可达场景：当前默认障碍，验证能返回一条从 `S` 到 `G` 的路径。
- 不可达场景：把终点周围全部设为障碍，验证 `found=False`。
- 退化场景：无障碍网格，检查路径代价是否等于曼哈顿距离。
- 边界场景：`start==goal` 时应立即返回零代价路径。

## R14

常见实现错误：

- 只返回布尔值，不返回“最小超界 `f`”，导致无法正确更新下一轮阈值。
- 递归中忘记回溯 `path` / `in_path`，导致路径污染。
- 对启发函数误用（高估）却仍假设最优性。
- 邻居遍历不排序，虽不影响正确性，但常导致扩展节点明显增多。

## R15

可扩展方向：

- 从网格拓展到任意稀疏图（道路网络、任务图）。
- 使用模式数据库等更强启发函数（例如拼图问题）。
- 增加转置表（transposition table）减少跨迭代重复展开。
- 改为显式栈实现，避免深递归场景的栈深限制。

## R16

与相关算法对比：

- 对比 A*：
  - A* 时间效率通常更好（少重复展开），但内存占用高。
  - IDA* 内存低，但会做多轮重复搜索。
- 对比 Dijkstra：
  - Dijkstra 无启发，扩展更“盲目”。
  - IDA* 依赖 `h`，在目标导向搜索中通常更高效。
- 对比 IDDFS：
  - IDDFS 以深度阈值加深。
  - IDA* 以 `f=g+h` 阈值加深，更适合带权与启发式问题。

## R17

参考资料：

- Richard E. Korf, "Depth-first Iterative-Deepening: An Optimal Admissible Tree Search", Artificial Intelligence, 1985.
- Stuart Russell, Peter Norvig, *Artificial Intelligence: A Modern Approach*（A* 与 IDA* 章节）。
- 本目录实现文件：`demo.py`。

## R18

`demo.py` 的源码级流程拆解（非黑箱，8 步）：

1. `build_grid_graph` 根据网格尺寸与障碍集合生成邻接表，边权固定为 `1.0`。
2. `manhattan` 用 `numpy` 计算节点到目标的 L1 距离，作为可采纳启发函数。
3. `ida_star` 设定初始阈值 `bound = h(start)`，并初始化统计量。
4. 在 `ida_star` 内部定义递归 `dfs`：先算 `f=g+h`，若 `f>bound` 直接返回该 `f` 作为超界候选。
5. 若 `dfs` 到达目标节点，返回当前路径副本和累计代价 `g`。
6. 展开邻接边前按 `g+cost+h(next)` 排序，优先探索更有希望的分支。
7. 若本轮未找到解，`dfs` 返回“本轮最小超界值”；外层将其作为下一轮 `bound`，继续迭代。
8. 找到解后输出路径、代价、迭代阈值序列和文本网格图，完成可验证 MVP 闭环。
