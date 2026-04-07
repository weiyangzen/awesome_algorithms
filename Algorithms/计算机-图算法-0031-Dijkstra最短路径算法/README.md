# Dijkstra最短路径算法

- UID: `CS-0028`
- 学科: `计算机`
- 分类: `图算法`
- 源序号: `31`
- 目标目录: `Algorithms/计算机-图算法-0031-Dijkstra最短路径算法`

## R01

Dijkstra 算法用于求解**非负权图的单源最短路径**（Single-Source Shortest Path, SSSP）。
本条目给出一个可直接运行的最小 MVP，实现目标是：
- 计算源点到所有节点的最短距离；
- 输出前驱数组并支持路径回溯；
- 对非法输入（负边、越界端点、非有限权重）做显式报错；
- 提供固定样例，`uv run python demo.py` 无交互即可复现结果。

## R02

问题定义（MVP 范围）：
- 输入：
1. 节点数 `n`（节点编号在 `[0, n-1]`）；
2. 有向加权边集 `edges=[(u,v,w), ...]`；
3. 源点 `source`。
- 约束：边权必须满足 `w >= 0`。
- 输出：
1. `dist[v]`：`source -> v` 的最短距离，不可达为 `inf`；
2. `parent[v]`：最短路径前驱，不存在时为 `-1`；
3. 统计信息：入堆次数、出堆次数、成功松弛次数。

## R03

算法核心不变量：
- 初始时 `dist[source]=0`，其余为 `inf`；
- 每轮从最小堆弹出当前距离最小的候选节点 `u`；
- 在边权非负前提下，`u` 一旦被“定型”（settled），`dist[u]` 就不会再被改写；
- 对每条边 `(u,v,w)` 做松弛：若 `dist[u] + w < dist[v]`，则更新 `dist[v]` 和 `parent[v]`。

## R04

`demo.py` 里 Dijkstra 的执行流程：
1. `validate_graph` 校验 `n/source` 范围、端点范围、权重非负且有限；
2. `build_adjacency` 将边集转为邻接表；
3. 初始化 `dist/parent/settled` 和最小堆；
4. 反复出堆当前最小候选节点；
5. 跳过过期堆项（stale entry）；
6. 对邻边执行松弛并把更新后的节点重新入堆；
7. 堆清空后得到最短距离与前驱；
8. 用 `reconstruct_path` 回溯指定目标节点路径。

## R05

关键数据结构：
- `adjacency: List[List[(v,w)]]`：邻接表；
- `dist: np.ndarray[float]`：最短距离数组；
- `parent: np.ndarray[int]`：前驱数组；
- `settled: np.ndarray[bool]`：节点是否已定型；
- `heap: List[(distance,node)]`：最小堆；
- `DijkstraResult`：统一封装结果与统计值。

## R06

正确性直觉：
- 所有边非负时，当前最小候选距离不可能被未来路径“反超”；
- 每次松弛都只会降低目标节点当前上界，且不会丢失更优解；
- 因此每个可达节点最多从“未定型”变为“定型”一次，最终 `dist` 收敛到最短距离；
- 不可达节点始终保持 `inf`，其 `parent` 维持 `-1`。

## R07

复杂度（设 `V=n`，`E=len(edges)`）：
- 时间复杂度：`O((V+E) log V)`（二叉堆实现，稀疏图常写作 `O(E log V)`）；
- 空间复杂度：`O(V+E)`（邻接表 + 状态数组 + 堆）。

## R08

边界与异常处理：
- `n <= 0`：抛 `ValueError`；
- `source` 越界：抛 `ValueError`；
- 边端点越界：抛 `ValueError`；
- 权重是 `nan/inf`：抛 `ValueError`；
- 权重为负：抛 `ValueError`（Dijkstra 不适用）；
- 路径回溯若目标不可达：返回 `None`。

## R09

MVP 设计取舍：
- 依赖仅使用 `numpy + heapq + dataclasses`，保持轻量；
- 不使用图算法黑箱库（例如 NetworkX 的一行调用），确保算法细节可追踪；
- 加入 `reference_relaxation` 作为交叉校验基线；
- 只实现单源最短路，不引入多源、动态更新或并行优化。

## R10

`demo.py` 函数职责：
- `validate_graph`：输入检查和边标准化；
- `build_adjacency`：构建邻接表；
- `dijkstra`：最短路主过程；
- `reference_relaxation`：重复松弛参考实现（用于校验）；
- `reconstruct_path`：按前驱回溯路径；
- `run_demo`：固定样例运行、断言和打印；
- `main`：无交互入口。

## R11

运行方式：

```bash
cd Algorithms/计算机-图算法-0031-Dijkstra最短路径算法
uv run python demo.py
```

脚本不读取任何交互输入，会直接输出统计、距离表、路径与校验结果。

## R12

输出说明（与 R04 流程对应）：
- `stats`：
1. `push_count` 对应“入堆次数”；
2. `pop_count` 对应“出堆次数（含过期项）”；
3. `relax_count` 对应“成功松弛次数”。
- `distance summary`：逐节点展示 `dist` 与可达性；
- `path summary`：对目标节点回溯路径，显示总花费；
- `All checks passed.`：表示 Dijkstra 结果同时通过“参考松弛法”和“预期值”双重断言。

## R13

最小验证清单（与 R11/R14 一致）：
- 直接运行固定样例，确认存在可达与不可达节点；
- 校验期望距离向量 ` [0, 7, 9, 20, 20, 11, inf] `；
- 校验 Dijkstra 与 `reference_relaxation` 一致；
- 校验不可达节点路径输出为 `unreachable`；
- 可手动加一条负边验证异常分支（应抛 `ValueError`）。

## R14

当前 demo 参数：
- `n=7`；
- `source=0`；
- 边集为固定 9 条非负有向边；
- 查询目标为 `targets=[3,4,5,6]`；
- 浮点比较容差为 `1e-10`（断言）与 `1e-12`（堆中过期判断）。

若要扩展测试，可只改 `run_demo` 内 `n/edges/source/targets`，无需改动核心函数。

## R15

与相关算法对比：
- Bellman-Ford：可处理负边，但复杂度通常更高（`O(VE)`）；
- Floyd-Warshall：求全源最短路，适合稠密图和全点对查询（`O(V^3)`）；
- A*：偏向单目标查询并依赖启发函数；当启发函数取 0 时退化为 Dijkstra。

## R16

适用场景（与 R08 前提一致）：
- 路网导航中的非负代价路径；
- 资源调度图中的最小累计成本路径；
- 网络转发中的最小代价路由建模；
- 作为 Johnson、A*（特殊情形）等方法的基础组件。

不适用场景：存在负边权或需要负环分析的任务，应切换到 Bellman-Ford 等算法。

## R17

可扩展方向（与 R18 源码流程衔接）：
- 在第 4-6 步加入“目标节点提前终止”优化；
- 在第 1 步支持无向图快捷输入（自动双向加边）；
- 在第 7 步增加批量源点模式；
- 在第 8 步增加路径条数统计或多条等长路径重建；
- 增加随机图性能基准，观察 `push/pop/relax` 的增长趋势。

## R18

`demo.py` 的源码级算法流（8 步，非黑箱）：
1. `main` 调用 `run_demo`，构造固定图 `n/source/edges`。  
2. `run_demo` 调用 `dijkstra`，进入核心求解。  
3. `dijkstra` 先执行 `validate_graph`，逐条检查节点范围、权重有限性与非负约束。  
4. `build_adjacency` 把边集整理为邻接表，并初始化 `dist/parent/settled` 与最小堆。  
5. 循环从堆弹出最小距离候选：过期条目直接丢弃，未定型节点标记为 settled。  
6. 遍历该节点的所有出边执行松弛：若 `candidate < dist[v]`，就更新 `dist[v]/parent[v]` 并重新入堆。  
7. 堆清空后返回 `DijkstraResult`，随后 `run_demo` 用 `reference_relaxation` 与预期向量做双重断言。  
8. 最后 `run_demo` 用 `reconstruct_path` 回溯并打印路径，输出 `All checks passed.` 作为成功标记。  
