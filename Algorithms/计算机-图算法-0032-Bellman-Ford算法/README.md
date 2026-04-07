# Bellman-Ford算法

- UID: `CS-0029`
- 学科: `计算机`
- 分类: `图算法`
- 源序号: `32`
- 目标目录: `Algorithms/计算机-图算法-0032-Bellman-Ford算法`

## R01

Bellman-Ford 是一种用于“单源最短路径”（Single-Source Shortest Path, SSSP）的经典图算法。  
它允许边权为负，并且能够检测从源点可达的负权环，这一点是相对 Dijkstra 的核心优势。

## R02

问题定义：给定带权有向图 `G=(V,E)`、源点 `s`，求从 `s` 到每个顶点 `v` 的最短距离 `dist[v]`。  
若存在从 `s` 可达的负权环，则相关顶点的最短路不存在（可无限减小），需要显式标记。

## R03

`demo.py` 的输入输出约定：
- 输入：顶点数 `n`、边列表 `edges=[(u,v,w), ...]`、源点 `source`
- 输出：
1. `dist`：源点到各顶点距离（不可达为 `inf`，受负环影响为 `-inf`）
2. `parent`：用于路径重建的前驱数组
3. `has_negative_cycle`：是否存在从源点可达的负权环
4. `affected_by_neg_cycle`：受负环影响的顶点集合

## R04

核心思想是“反复松弛（relaxation）”：
- 一条从 `s` 到 `v` 的最短简单路径最多包含 `|V|-1` 条边。
- 每完成一轮对全部边的松弛，允许路径长度上限增加 1。
- 做 `|V|-1` 轮后，若无负环，最短路必然收敛。

## R05

初始化规则：
- `dist[source]=0`
- 其他顶点 `dist[v]=inf`
- `parent[v]=None`

松弛边 `(u,v,w)` 时，若 `dist[u] + w < dist[v]`，就更新：
- `dist[v] = dist[u] + w`
- `parent[v] = u`

## R06

主循环执行 `n-1` 轮全边松弛，并支持“提前停止”：
- 若某一轮没有任何更新，说明距离已稳定，可提前结束。
- 该优化在无负边或图较浅时能显著减少实际运行时间。

## R07

负权环检测：
- 在 `n-1` 轮后，再扫描所有边。
- 若仍存在可松弛边 `(u,v,w)` 且 `u` 可达，则说明存在从源点可达的负权环。

本 MVP 进一步把这类顶点向前传播：
- 从“仍可松弛”的终点顶点出发沿出边 BFS/DFS。
- 这些可达点都设为 `-inf`，表示最短路可被负环无限降低。

## R08

伪代码：

```text
dist = [inf] * n
parent = [None] * n
dist[source] = 0

repeat n-1 times:
    updated = false
    for (u, v, w) in edges:
        if dist[u] != inf and dist[u] + w < dist[v]:
            dist[v] = dist[u] + w
            parent[v] = u
            updated = true
    if not updated:
        break

queue = []
for (u, v, w) in edges:
    if dist[u] != inf and dist[u] + w < dist[v]:
        queue.push(v)  # 负环影响起点

从 queue 沿图遍历得到 affected
for x in affected:
    dist[x] = -inf
    parent[x] = None
```

## R09

正确性直觉：
- 经过第 `k` 轮后，`dist[v]` 不会大于任意“边数不超过 `k`”路径的最小代价。
- 因为无负环最短路可取简单路径（边数最多 `n-1`），所以 `n-1` 轮后得到真实最短路。
- 若第 `n` 次扫描仍能改进，说明存在可无限下降的环（负权环）。

## R10

复杂度分析：
- 时间复杂度：`O(VE)`（`n-1` 轮，每轮扫描 `E` 条边）
- 空间复杂度：`O(V + E)`（距离、前驱、邻接表与辅助队列）

相较 Dijkstra 的堆优化版本，Bellman-Ford 在大稀疏非负图上通常更慢，但支持负边与负环检测。

## R11

典型适用场景：
- 汇率套利检测（负环可解释为可套利环）
- 带惩罚项/补贴的路径规划（边权可能为负）
- 作为 Johnson 算法的子过程（用于重标定势能）
- 教学中演示“松弛”与“最短路收敛”机制

## R12

与常见最短路算法对比：
- Dijkstra：要求非负边权；在稀疏图通常更快。
- Bellman-Ford：允许负边，能检测负环；复杂度更高。
- Floyd-Warshall：求全点对最短路，`O(V^3)`，不适合超大图单源查询。

若只需单源并且可能存在负边，Bellman-Ford 是稳妥基线。

## R13

实现细节与坑点：
- 必须先判断 `dist[u] != inf` 再做 `dist[u] + w`，避免无意义运算。
- 顶点编号要在 `[0, n-1]`，否则应抛异常。
- 负环检测必须基于“源点可达”部分，不能把全图任意负环都算进来。
- 路径重建在检测到 `-inf` 或 `inf` 时应直接返回空路径。

## R14

路径重建策略（`demo.py`）：
- 从目标 `target` 反向读取 `parent[target]` 直到 `source`。
- 反转后得到正向路径。
- 若中途断链、超步数、或目标是 `inf/-inf`，返回空路径表示不可定义。

## R15

本目录 MVP 结构：
- `demo.py`：
1. `bellman_ford`：核心算法 + 负环影响传播
2. `reconstruct_path`：按前驱重建路径
3. `run_case`：运行固定样例并打印结果
- `README.md`：原理、复杂度、边界条件、源码流程拆解

## R16

运行方式：

```bash
cd Algorithms/计算机-图算法-0032-Bellman-Ford算法
uv run python demo.py
```

脚本无需交互输入，直接输出两组固定测试结果。

## R17

最小验证清单：
- 无负环样例：距离应与人工结果一致。
- 不可达顶点应保持 `inf`。
- 负环可达样例：`has_negative_cycle=True`。
- 受负环影响顶点应变为 `-inf`。
- `reconstruct_path` 对 `inf/-inf` 目标返回空路径。

## R18

`demo.py` 的源码级算法流程（8 步）：
1. 在 `main` 中构造两组图：一组无负环，一组含源点可达负环。  
2. `run_case` 调用 `bellman_ford(n, edges, source)` 获取距离、前驱和负环信息。  
3. `bellman_ford` 先初始化 `dist`、`parent`，并校验边端点是否越界。  
4. 执行最多 `n-1` 轮全边松弛：若某轮无更新则提前退出。  
5. 额外扫描一轮边，收集仍可松弛的终点顶点，作为“负环影响种子”。  
6. 通过邻接表从种子点做 BFS，把所有可被负环继续到达的顶点收集为 `affected`。  
7. 将 `affected` 顶点的 `dist` 设为 `-inf`、`parent` 置空，并返回 `has_negative_cycle` 标记。  
8. `run_case` 对查询点调用 `reconstruct_path`：若目标 `inf/-inf` 返回空，否则沿前驱回溯并打印路径与代价。  
