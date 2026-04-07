# 最短路径 - Johnson算法

- UID: `MATH-0216`
- 学科: `数学`
- 分类: `图论`
- 源序号: `216`
- 目标目录: `Algorithms/数学-图论-0216-最短路径_-_Johnson算法`

## R01

问题定义：在带权有向图中，允许存在负权边（但不允许负权环），要求求出所有点对 `(u, v)` 的最短路径距离。
Johnson 算法的目标是在稀疏图场景下，比 Floyd-Warshall 更高效地完成全源最短路计算。

## R02

输入：
- 顶点集合 `V`
- 边集合 `E`，每条边形如 `(u, v, w)`，`w` 可为负数

输出：
- 距离矩阵 `dist[u][v]`，不可达时为 `inf`
- 若图中存在负权环，应直接报错并终止

## R03

核心思想分两段：
1. 用 Bellman-Ford 计算“势能” `h(v)`，将所有边重赋权为非负边
2. 对每个源点跑一次 Dijkstra，最后把结果映射回原图权重

## R04

为什么能处理负权边：
- 原图直接跑 Dijkstra 会失效（负权破坏贪心）
- Johnson 先引入超级源点 `s*`，向所有点连权重 0 的边
- 在扩展图上跑 Bellman-Ford 得到 `h(v)=dist(s*,v)`
- 令新权重 `w'(u,v)=w(u,v)+h(u)-h(v)`，可证明 `w'(u,v) >= 0`

## R05

重赋权后的性质：
- 任意路径 `P: u -> v`，其新权重与原权重差一个端点常数：`w'(P)=w(P)+h(u)-h(v)`
- 因此两条 `u->v` 路径的大小关系不变，最短路径结构不变
- 只是在计算时把负边“平移”为非负边

## R06

Dijkstra 阶段：
- 对每个源点 `u` 在 `w'` 图上求 `d'(u, v)`
- 再恢复原图距离：`d(u,v)=d'(u,v)-h(u)+h(v)`
- 若 `d'(u,v)=inf`，则原图也不可达

## R07

复杂度：
- Bellman-Ford：`O(VE)`
- `V` 次 Dijkstra（二叉堆）：`O(V * (E log V))`
- 总体：`O(VE + V E log V)`，常写为 `O(VE log V)`

对比 Floyd-Warshall 的 `O(V^3)`，Johnson 更适合稀疏图（`E << V^2`）。

## R08

适用前提：
- 可以有负权边
- 不能有负权环（否则最短路无界）
- 图可不连通，算法仍可返回部分可达结果（其余为 `inf`）

## R09

与常见方案对比：
- Bellman-Ford 全源版：`V` 次 Bellman-Ford，通常更慢
- Floyd-Warshall：实现简洁，但稠密图更合适
- Johnson：兼顾负权支持与稀疏图效率，是工程上常见折中

## R10

工程实现要点：
- 使用邻接表存储重赋权后的图
- Dijkstra 用最小堆（`heapq`）
- 建议显式检测负权环并抛出异常，避免静默输出错误结果

## R11

`demo.py` 实现函数：
- `bellman_ford_potential`：求势能并做负环检测
- `dijkstra`：单源最短路
- `johnson_all_pairs_shortest_paths`：算法总控
- `pretty_print_matrix`：结果输出辅助

## R12

最小可运行 MVP 特征：
- 仅使用 Python 标准库（`heapq`, `math`, `typing`）
- 无交互输入，执行 `python3 demo.py` 即可看到输出
- 内置经典带负边但无负环的样例图，并包含断言校验

## R13

样例图（5 个点）包含边：
- `0->4` 权重 `-4`
- `3->2` 权重 `-5`

这是 Johnson 算法常用教学样例：有负边、无负环，适合验证重赋权 + Dijkstra 的组合流程。

## R14

正确性直觉：
- Bellman-Ford 提供合法势能 `h`
- 势能重赋权保证非负边，Dijkstra 贪心条件成立
- 端点常数修正只平移路径代价，不改变最短路径相对优劣
- 负权环检测保证问题本身有解

## R15

边界与异常：
- 空图：返回空结果
- 孤立点：到其他点距离为 `inf`
- 自环：按普通边处理
- 负权环：抛出 `ValueError` 并给出明确提示

## R16

测试建议：
1. 无负边普通图（应与多次 Dijkstra 一致）
2. 有负边无负环图（Johnson 主场景）
3. 含负权环图（应抛异常）
4. 非连通图（不可达为 `inf`）
5. 单点图与空边图（退化场景）

## R17

运行方式：
```bash
python3 Algorithms/数学-图论-0216-最短路径_-_Johnson算法/demo.py
```

预期行为：
- 打印顶点顺序与全源最短路矩阵
- 自动执行断言，若通过则打印成功提示

## R18

源码级算法流程（对应 `demo.py`）：
1. 读取 `nodes` 和 `edges`，构建基础图数据。
2. 在 `bellman_ford_potential` 中添加超级源点 `super_source`，并加入 `super_source -> v (0)` 的边。
3. 对扩展图做 `|V|` 轮松弛，得到每个点的势能 `h(v)`；若还能继续松弛，判定存在负权环并抛异常。
4. 在 `johnson_all_pairs_shortest_paths` 中按 `w'(u,v)=w+h(u)-h(v)` 构造重赋权邻接表。
5. 对每个源点 `src` 调用 `dijkstra`，在非负权图上得到 `d'(src, *)`。
6. 将 `d'` 映射回原权重：`d(src,dst)=d'(src,dst)-h(src)+h(dst)`；不可达点保持 `inf`。
7. 汇总为 `all_pairs[src][dst]` 字典并返回。
8. `main` 中打印矩阵，并用已知答案做断言，形成可验证的最小闭环。
