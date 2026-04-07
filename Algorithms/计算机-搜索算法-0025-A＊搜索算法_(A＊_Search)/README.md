# A*搜索算法 (A* Search)

- UID: `CS-0025`
- 学科: `计算机`
- 分类: `搜索算法`
- 源序号: `25`
- 目标目录: `Algorithms/计算机-搜索算法-0025-A＊搜索算法_(A＊_Search)`

## R01

A*（A-star）是带启发式信息的最短路径搜索算法。它通过最小化评估函数 `f(n)=g(n)+h(n)` 在“已走代价”与“预计剩余代价”之间做平衡，通常比纯 Dijkstra 扩展更少节点。

## R02

问题形式化（本目录 MVP 版本）：
- 输入：二维网格 `grid`（`0` 可走、`1` 障碍），起点 `start`，终点 `goal`
- 状态：网格坐标 `(r, c)`
- 邻接：4 邻域（上右下左），每步代价固定为 `1`
- 输出：
  - 若可达：返回一条从 `start` 到 `goal` 的路径及总代价
  - 若不可达：返回未找到标记

## R03

启发函数要求：
- A* 要保证最优解，通常需要 `h(n)` 可采纳（admissible）：不高估真实剩余代价
- 若 `h(n)` 还满足一致性（consistent/monotone），实现和证明更直接
- 本实现使用曼哈顿距离 `|dx|+|dy|`，与 4 邻域单位代价模型匹配，满足可采纳与一致性

## R04

核心思想：
- `g(n)`：起点到当前节点的最小已知代价
- `h(n)`：当前节点到终点的启发式估计
- `f(n)=g(n)+h(n)`：优先队列中的排序键
- 每轮从 open 集中取 `f` 最小节点扩展，并尝试用“更小的 `g`”松弛其邻居

## R05

简化伪代码：

```text
open = min-heap by f
push(start, f=h(start))
g[start] = 0
came_from = {}
closed = {}

while open not empty:
    current = pop_min_f(open)
    if current in closed: continue
    if current == goal: reconstruct path
    add current to closed

    for neighbor in neighbors(current):
        if neighbor in closed: continue
        tentative_g = g[current] + cost(current, neighbor)
        if tentative_g < g.get(neighbor, +inf):
            came_from[neighbor] = current
            g[neighbor] = tentative_g
            push(neighbor, tentative_g + h(neighbor))

return unreachable
```

## R06

正确性要点：
- `g` 字典始终记录“当前已知最好代价”
- 邻居更新仅在 `tentative_g` 更优时发生，不会丢失更短路径
- 在一致启发函数下，节点首次以最小 `f` 被稳定弹出并关闭时，其 `g` 已最优
- 因此当终点被弹出时，得到的是最短路径

## R07

复杂度（用 `V` 节点数、`E` 边数表示）：
- 时间复杂度：`O((V + E) log V)`（二叉堆优先队列）
- 空间复杂度：`O(V)`（open/closed/g/came_from）
- 在有效启发函数下，实际扩展节点通常显著少于无启发式搜索

## R08

边界与异常：
- 空网格、非矩形网格、包含非 `0/1` 值应拒绝
- 起点/终点越界或位于障碍格应拒绝
- `start == goal` 时应直接返回零代价路径
- 无法到达终点时应返回“未找到”而不是返回错误路径

## R09

示例（直观）：
- 可达场景：存在障碍但有绕行通路，A* 返回一条最短路径
- 不可达场景：障碍形成割裂，A* 返回 `found=False`
- 退化场景：`start==goal`，返回 `[start]` 且 `cost=0`

## R10

本目录 MVP 实现策略：
- 仅用 Python 标准库（`heapq`、`dataclasses`、`collections.deque`）实现，避免黑盒依赖
- `demo.py` 中显式实现：
  - 网格合法性检查
  - A* 主流程（open heap + g_score + came_from + closed）
  - 路径重建与可达/不可达结果建模
- 同时用 BFS 在同一网格上做最短路长度交叉验证，确保 A* 返回的是最短路径而非任意可行路径

## R11

运行方式：

```bash
uv run python demo.py
```

预期输出包含：
- 三个固定测试用例（可达、不可达、起终点相同）
- 每个用例的 `found/cost/expanded` 摘要
- 网格可视化（`S` 起点、`G` 终点、`#` 障碍、`*` 路径）
- 输入校验异常测试通过信息
- 最终 `All checks passed.`

## R12

典型应用：
- 游戏寻路（2D/3D 网格导航）
- 机器人路径规划的离散层
- 地图导航中的路网最短路（配合更复杂启发函数）
- 任务规划中的状态空间搜索

## R13

常见错误：
- 启发函数高估导致最优性丢失
- 忘记“只有更优 `g` 才更新”导致路径错误
- 不处理重复入堆（stale entry）导致扩展统计混乱或性能下降
- 路径重建方向写反或遗漏终点

## R14

常见变体：
- Weighted A*：`f=g+w*h`（`w>1`）以速度换最优性
- 双向 A*：从起点和终点同时搜索
- IDA*：迭代加深 + `f` 阈值，降低内存占用
- Jump Point Search（网格特化）：减少对称扩展

## R15

与 Dijkstra / BFS 对比：
- BFS：仅适用于单位代价图，无启发信息
- Dijkstra：适用于非负权，等价于 `h(n)=0` 的 A*
- A*：当 `h` 有信息量时，通常能在保持最优性的同时显著减少扩展节点

## R16

工程实践建议：
- 明确状态定义与代价模型（4 邻域/8 邻域、转向代价、地形权重）
- 为不可达场景设计稳定返回结构（不要抛模糊异常）
- 保留 `expanded_nodes` 等诊断指标，便于调参与性能分析
- 用基准算法（如 BFS/Dijkstra）做回归验证，防止启发函数改动引入隐性错误

## R17

最小测试清单：
- 合法可达网格：验证路径合法且代价等于最短路
- 合法不可达网格：验证 `found=False`
- `start==goal`：验证零长度/零代价
- 非矩形网格：验证输入校验生效
- 起点或终点落在障碍格：验证输入校验生效

## R18

源码级算法流程拆解（对应本目录 `demo.py` 的 `astar_search`）：
1. 入口先执行 `_validate_grid`，确保网格、起点、终点全部满足约束；不合法直接抛 `ValueError`。
2. 处理退化场景：若 `start == goal`，立即返回 `found=True, path=[start], cost=0`。
3. 初始化搜索状态：`g_score[start]=0`，`came_from={}`，并把起点按 `f=h(start)` 推入最小堆 `open_heap`。
4. 循环弹出堆顶节点 `current`；若该节点已在 `closed`，跳过（处理重复入堆条目）。
5. 将 `current` 标记为已扩展并累计 `expanded_nodes`；若 `current==goal`，调用 `_reconstruct_path` 逆向回溯并返回结果。
6. 枚举 `_neighbors4(current)` 的可行邻居，对每个邻居计算 `tentative_g = g[current] + 1`。
7. 仅当 `tentative_g` 严格优于旧值时，才更新 `came_from[neighbor]` 与 `g_score[neighbor]`，再计算 `f=tentative_g+h(neighbor)` 并入堆。
8. 当堆耗尽仍未命中终点时，返回 `found=False, path=[], cost=-1`，表示不可达。
9. `main()` 中再用 `_bfs_shortest_distance` 对可达用例做最短路交叉校验，确保 A* 返回路径代价与真实最短路一致。
