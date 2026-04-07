# Floyd-Warshall算法

- UID: `CS-0030`
- 学科: `计算机`
- 分类: `图算法`
- 源序号: `33`
- 目标目录: `Algorithms/计算机-图算法-0033-Floyd-Warshall算法`

## R01

Floyd-Warshall 是一种用于求解加权有向图中“任意两点最短路径”（All-Pairs Shortest Paths, APSP）的经典动态规划算法。它允许边权为负，但要求图中不存在可达的负权环，否则最短路无定义。

## R02

问题定义：给定 `n` 个顶点的带权有向图 `G=(V,E)`，边权函数 `w(u,v)` 可能为负，求所有点对 `(i,j)` 的最短路径长度 `dist[i][j]`，并可选输出路径本身。

## R03

输入输出约定（与 `demo.py` 一致）：
- 输入: `n x n` 邻接矩阵 `weights`
- 语义: `weights[i][j]` 表示边 `i -> j` 的权重；不可达记为 `float("inf")`
- 输出:
1. `dist`: 最短距离矩阵
2. `next_hop`: 路径重建用的下一跳矩阵
3. `has_negative_cycle`: 是否存在负权环

## R04

核心思想是逐步放宽“允许作为中间点的顶点集合”。
定义 `D^(k)[i][j]` 为只允许使用顶点集合 `{0..k}` 作为中间点时，从 `i` 到 `j` 的最短路长度。
当引入新中间点 `k` 时，路径要么不经过 `k`，要么经过 `k`，从而形成标准递推。

## R05

状态定义：
- `dist[i][j]` 在第 `k` 轮结束后，表示只用 `{0..k}` 作中间点时的最短距离
- 初始化时（相当于 `k=-1`）:
1. `dist[i][i] = 0`
2. 若有边 `i->j`，则 `dist[i][j] = w(i,j)`
3. 否则为 `+inf`

## R06

状态转移方程：

`dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])`

其中三层循环顺序通常为 `for k in V`、`for i in V`、`for j in V`。如果 `dist[i][k]` 或 `dist[k][j]` 为 `inf`，则该候选不可用。

## R07

正确性直觉：
- 归纳基: 未允许任何中间点时，`dist` 仅表示直接边或自环 0，显然正确。
- 归纳步: 加入中间点 `k` 后，任意 `i->j` 最短路只分两类: 不经过 `k`（保留旧值）或经过 `k`（分解为 `i->k` 与 `k->j`）。取两者最小即得到新最优。
- 因为每轮都完整覆盖所有点对，最终得到全体中间点允许下的全局最短路。

## R08

伪代码：

```text
initialize dist from weights
initialize next_hop for path reconstruction

for k in [0..n-1]:
    for i in [0..n-1]:
        if dist[i][k] == inf: continue
        for j in [0..n-1]:
            if dist[k][j] == inf: continue
            candidate = dist[i][k] + dist[k][j]
            if candidate < dist[i][j]:
                dist[i][j] = candidate
                next_hop[i][j] = next_hop[i][k]

has_negative_cycle = any(dist[v][v] < 0 for v in [0..n-1])
```

## R09

复杂度：
- 时间复杂度: `O(n^3)`
- 空间复杂度: `O(n^2)`（距离矩阵 + 路径矩阵）

这使其在稠密图、小中规模图上很常用；在超大稀疏图上通常不如多源 Dijkstra 组合。

## R10

适用场景：
- 需要一次性得到所有点对最短路
- 图可能有负边但不应有负环
- 需要判断负环存在性（通过 `dist[v][v] < 0`）
- 需要后续频繁查询任意两点最短距离

## R11

与常见算法对比：
- Dijkstra: 单源、不能处理负边（经典实现）；配合堆在稀疏图高效
- Bellman-Ford: 单源、可处理负边并检测负环，复杂度 `O(VE)`
- Floyd-Warshall: 全源全点对统一求解，写法最简洁，代价是 `O(n^3)`

## R12

实现细节与坑点：
- `inf` 运算要先判断可达性，避免无意义加法
- 邻接矩阵必须是方阵，否则应抛错
- 自环初始化通常设为 `0`（除非业务有特殊约束）
- 路径重建需维护 `next_hop`，仅有距离矩阵无法还原路径

## R13

负环检测：
- 运行主循环后，若存在顶点 `v` 使 `dist[v][v] < 0`，则存在负权环
- 此时部分点对的“最短路”实际上趋向 `-inf`（可绕环无限变小）
- `demo.py` 中返回 `has_negative_cycle` 标记，便于调用方按业务处理

## R14

路径重建方法：
- `next_hop[i][j]` 表示从 `i` 走向 `j` 的最短路第一跳
- 若为 `None`，表示不可达
- 重建时从 `u` 开始反复令 `u = next_hop[u][v]` 直至到达 `v`

## R15

本目录 MVP 设计：
- 语言: Python 3
- 依赖: 仅标准库（保证最小可运行）
- 文件:
1. `demo.py`: Floyd-Warshall 核心实现 + 路径重建 + 两组演示图
2. `README.md`: 原理、复杂度、实现说明与源码流程拆解

## R16

运行方式：

```bash
uv run python demo.py
```

运行后会打印：
- 无负环示例图的最短路矩阵
- 若干点对的最短路径与代价
- 含负环示例图的检测结果

## R17

最小验证清单：
- 可达路径距离正确
- 不可达点保持 `inf`
- 负边图（无负环）结果稳定
- 负环图能触发 `has_negative_cycle = True`
- `reconstruct_path` 对不可达点返回空路径

## R18

`demo.py` 的源码级算法流程（非黑盒）可分为 8 步：
1. 读取邻接矩阵，校验是否为 `n x n` 方阵。
2. 构造 `dist` 为权重拷贝，并把所有 `dist[i][i]` 归一为 `0`（若更小则保留）。
3. 初始化 `next_hop`：若 `i!=j` 且 `i->j` 可达，则第一跳设为 `j`，否则为 `None`。
4. 进入三重循环，以 `k` 为当前允许的新中间点。
5. 对每个 `(i,j)`，若 `i->k` 与 `k->j` 都可达，计算候选代价 `dist[i][k] + dist[k][j]`。
6. 若候选更短，则更新 `dist[i][j]`，并把 `next_hop[i][j]` 更新为 `next_hop[i][k]`（保持路径可追踪）。
7. 主循环结束后检查所有对角线元素，若任意 `dist[v][v] < 0`，置 `has_negative_cycle=True`。
8. 查询路径时，通过 `next_hop` 逐跳前进构造完整顶点序列；若某步为 `None` 则判定不可达。
