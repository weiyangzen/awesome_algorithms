# 最短路径 - Dijkstra

- UID: `MATH-0212`
- 学科: `数学`
- 分类: `图论`
- 源序号: `212`
- 目标目录: `Algorithms/数学-图论-0212-最短路径_-_Dijkstra`

## R01

本条目实现 `Dijkstra` 的最小可运行版本（MVP），用于**非负权有向图的单源最短路径**。目标是：
- 输出源点到所有节点的最短距离；
- 提供前驱数组并可重建具体路径；
- 处理不可达节点（距离记为 `inf`）；
- 对输入做严格校验（尤其禁止负边）；
- 提供固定样例，`python3 demo.py` 可直接运行。

## R02

问题定义（MVP 范围）：
- 输入：
  - 节点数 `n`；
  - 边集 `edges = (u, v, w)`（有向，且 `w >= 0`）；
  - 源点 `source`。
- 输出：
  - `dist[v]`：`source -> v` 最短距离（不可达为 `+inf`）；
  - `pred[v]`：最短路径上的前驱节点（不存在时 `-1`）；
  - 运行统计：入堆次数、出堆次数、有效松弛次数。

## R03

数学基础与贪心不变量：

1. 初始化 `dist[source]=0`，其余 `dist=+inf`。  
2. 每次从优先队列中取出当前最小暂定距离的节点 `u`。  
3. 在所有边权非负时，`u` 一旦被“定型”（settled），其 `dist[u]` 即为全局最优值。  
4. 对每条边 `(u,v,w)` 执行松弛：若 `dist[u] + w < dist[v]`，更新 `dist[v]` 与 `pred[v]`。  
5. 重复直到队列为空，得到源点到所有可达节点的最短距离。

## R04

算法流程（MVP）：
1. 校验输入合法性：节点数、源点范围、边端点、权重有限且非负。
2. 构建邻接表 `adj[u] = [(v,w), ...]`。
3. 初始化 `dist`、`pred`、`settled`，并将 `(0, source)` 入最小堆。
4. 循环出堆：取当前距离最小节点。
5. 跳过“过期堆条目”（stale entry），避免重复无效计算。
6. 对未定型节点执行邻边松弛，成功则更新并入堆。
7. 循环结束后返回 `dist`、`pred` 与统计指标。
8. 按 `pred` 回溯可重建任意目标点路径。

## R05

核心数据结构：
- `edges: List[Tuple[int,int,float]]`：标准化后的边集合；
- `adj: List[List[Tuple[int,float]]]`：邻接表；
- `dist: numpy.ndarray(float)`：最短距离向量；
- `pred: numpy.ndarray(int)`：前驱向量；
- `settled: numpy.ndarray(bool)`：节点是否已定型；
- `heap: List[Tuple[float,int]]`：最小堆（优先队列）；
- `DijkstraResult`（`dataclass`）：统一封装输出与统计信息。

## R06

正确性要点：
- 非负边权是 Dijkstra 正确性的必要前提；
- 由最小堆弹出的最小 `dist` 节点，在非负边下不会再被更短路径改写；
- 每次松弛都保持“上界单调下降”，不会错过更优路径；
- 所有可达节点最终都会被定型，不可达节点保持 `inf`；
- `pred` 保存的是“最后一次成功松弛来源”，可重建一条最短路径。

## R07

复杂度：
- 设节点数 `V=n`，边数 `E=m`。
- 时间复杂度：
  - 使用二叉堆时为 `O((V+E) log V)`；
  - 稀疏图场景常写作 `O(E log V)`。
- 空间复杂度：
  - `dist/pred/settled` 为 `O(V)`；
  - 邻接表 `O(E)`；
  - 堆最坏 `O(E)`；
  - 总体 `O(V+E)`。

## R08

边界与异常处理：
- `n <= 0`：抛出 `ValueError`；
- `source` 越界：抛出 `ValueError`；
- 边端点越界：抛出 `ValueError`；
- 权重非有限值（`nan/inf`）：抛出 `ValueError`；
- 权重为负：抛出 `ValueError`（Dijkstra 不适用）；
- 路径回溯时若目标不可达，返回 `None`。

## R09

MVP 取舍说明：
- 仅依赖 `numpy` + Python 标准库（`heapq`、`dataclasses`），保持轻量；
- 不引入图算法黑盒库（如 NetworkX 的一行最短路接口）；
- 保留“结果校验”能力：用一个简单的边松弛参考实现做交叉验证；
- 不实现多源、动态更新、并行化、A* 启发式等扩展功能。

## R10

`demo.py` 职责划分：
- `validate_input`：输入合法性检查与边记录标准化；
- `build_adjacency`：由边集构建邻接表；
- `dijkstra`：核心最短路求解；
- `relaxation_reference`：参考解（重复松弛）用于结果对照；
- `reconstruct_path`：前驱回溯路径；
- `print_summary`：打印距离表和统计信息；
- `run_demo_case`：固定样例演示并执行断言；
- `main`：无交互入口。

## R11

运行方式：

```bash
cd Algorithms/数学-图论-0212-最短路径_-_Dijkstra
python3 demo.py
```

脚本无交互输入，会直接打印距离、路径和校验结果。

## R12

输出内容解读：
- `stats`：
  - `push_count`：入堆次数；
  - `popped_count`：出堆次数（含过期条目）；
  - `relax_count`：成功松弛次数。
- `distance summary from source`：每个节点最短距离与可达性；
- `paths from source`：演示节点的一条最短路径；
- `All checks passed.`：表示 Dijkstra 与参考松弛解一致。

## R13

建议最小测试集：
- 标准非负权图（验证基本正确性）；
- 含不可达节点（验证 `inf` 与 `None` 路径分支）；
- 含零权边（验证非负边界）；
- 非法输入：负权边、越界端点、`nan/inf` 权重、`source` 越界。

## R14

可调参数：
- 图规模 `n` 与边集 `edges`；
- 源点 `source`；
- 路径展示目标集合 `targets`；
- 浮点比较容差 `eps`（当前 `1e-12`）。

实践建议：先用小图手工验证，再逐步扩大图规模观察堆操作增长。

## R15

方法对比：
- 对比 Bellman-Ford：
  - Dijkstra 更快（常见 `O(E log V)`）；
  - 但 Dijkstra 不能处理负边。
- 对比 Floyd-Warshall：
  - Floyd-Warshall 是全源最短路，复杂度 `O(V^3)`；
  - Dijkstra 是单源最短路，适合稀疏图和单源查询。
- 对比 A*：
  - A* 针对单源单终点并利用启发函数；
  - Dijkstra 无需启发函数，结果更通用稳定。

## R16

典型应用场景：
- 路网/导航中的非负代价路径规划；
- 网络路由中的最短代价传播（思想层面）；
- 任务调度与依赖图中的最小累计成本分析；
- 作为 A*（`h=0` 特例）与 Johnson（重标定后）等算法的基础组件。

## R17

可扩展方向：
- 支持无向图快捷输入（自动双向加边）；
- 支持指定目标点的提前终止（single-target 优化）；
- 增加随机图基准测试与性能统计；
- 在大规模场景切换到更高阶堆结构或并行化方案；
- 增加“多组 source 批处理”接口。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `run_demo_case` 定义固定图 `n/source/edges`，调用 `dijkstra`。  
2. `validate_input` 检查 `n`、`source`、端点范围与权重合法性（非负且有限）。  
3. `build_adjacency` 将边集转换为邻接表，初始化 `dist`、`pred`、`settled` 与堆。  
4. `dijkstra` 从堆弹出当前最小 `(distance,node)`，若是过期条目则跳过。  
5. 对未定型节点遍历邻边，执行松弛：若 `cand < dist[v]`，更新 `dist[v]`、`pred[v]` 并入堆。  
6. 堆清空后返回 `DijkstraResult`，其中包含距离向量、前驱向量和三类计数统计。  
7. `run_demo_case` 调用 `relaxation_reference`（重复边松弛）并与 Dijkstra 距离逐点校验。  
8. 通过 `reconstruct_path` 回溯打印目标路径，最终输出 `All checks passed.`。  
