# 最小生成树 - Boruvka

- UID: `MATH-0219`
- 学科: `数学`
- 分类: `图论`
- 源序号: `219`
- 目标目录: `Algorithms/数学-图论-0219-最小生成树_-_Boruvka`

## R01

Boruvka 算法用于在无向带权图中构造最小生成树（Minimum Spanning Tree, MST）。

给定图 `G=(V,E)`，每条边有权重 `w(e)`，目标是在连通图中选出 `|V|-1` 条边，使得：
- 图保持连通且无环；
- 总权重最小。

若图不连通，Boruvka 会自然退化为最小生成森林（MSF）。

## R02

核心直觉是“并行贪心”：
- 把当前每个连通分量看成一个超级点；
- 每个分量独立挑出一条跨分量的最便宜边；
- 同时合并这些边连接到的分量。

因为每个分量都至少尝试合并一次，分量数量会快速下降（通常接近对半），因此轮数较少。

## R03

数学化描述：
- 输入：`n=|V|` 个顶点，边集 `E={(u,v,w)}`。
- 维护分量划分 `C_1, C_2, ..., C_k`（用并查集实现）。
- 对每个分量 `C_i`，选择满足 `u in C_i, v not in C_i` 的最小权边 `e_i`。
- 将所有 `e_i` 加入候选集合并执行 union（若形成环则跳过）。
- 直到 `k=1`（MST）或某轮无法继续合并（MSF）。

## R04

为何贪心是安全的：
- 对任意当前分量 `C`，跨越割 `(C, V\C)` 的最小边一定是某棵 MST 的安全边（割性质）。
- Boruvka 每轮为每个分量选择一条这样的安全边。
- 即便不同分量选到同一条边，union 时也会自动去重，不会破坏正确性。

## R05

MVP 使用的数据结构：
- `DisjointSetUnion`：维护动态连通性。
- `cheapest` 数组：`cheapest[root]` 记录该分量当前最便宜外连边。
- `BoruvkaResult`：封装输出（边集、总权重、轮数、是否生成树、剩余分量数）。

这套结构是最小可运行实现，不依赖外部图算法库黑盒。

## R06

伪代码：

```text
BoruvkaMST(n, edges):
    dsu <- make_set(n)
    components <- n
    mst <- []
    while components > 1:
        cheapest <- [None] * n

        for (u,v,w) in edges:
            ru <- find(u), rv <- find(v)
            if ru == rv: continue
            cheapest[ru] <- min_by_weight(cheapest[ru], (u,v,w))
            cheapest[rv] <- min_by_weight(cheapest[rv], (u,v,w))

        merged <- 0
        for e in cheapest:
            if e is None: continue
            if union(e.u, e.v):
                mst.append(e)
                components -= 1
                merged += 1

        if merged == 0: break

    return mst, components
```

## R07

正确性要点（简述）：
- 不变式 1：`mst_edges` 始终无环（由 `union` 失败时跳过保证）。
- 不变式 2：每条被接受的边都连接不同分量，且可由割性质证明安全。
- 终止时：
  - 若 `components=1`，得到 `|V|-1` 条边的连通无环图，即 MST；
  - 若 `components>1` 且无法继续合并，说明原图不连通，结果是 MSF。

## R08

复杂度：
- 每轮扫描全部边：`O(E)`。
- 轮数通常是 `O(log V)`（分量数快速减少）。
- 总复杂度：`O(E log V)`，并查集摊还近似常数（严格写法 `alpha(V)`）。
- 空间复杂度：`O(V + E)`（若输入边存储在内存中）。

## R09

`demo.py` 包含两个示例：
- 连通图（7 个点）：应输出完整 MST；
- 非连通图（5 个点）：应输出最小生成森林，并标记 `spanning=False`。

这样可以同时验证主路径与边界行为。

## R10

实现对应关系：
- `DisjointSetUnion.find/union`：路径压缩 + 按秩合并。
- `boruvka_mst(...)`：算法主循环。
- `cheapest` 维护每个分量的“本轮最便宜外连边”。
- `pretty_print_result(...)`：将结果格式化为可读输出。
- `main()`：构造样例并执行无交互验证。

## R11

边界与异常处理：
- `num_vertices < 0`：抛出 `ValueError`。
- `num_vertices = 0`：返回空结果，视为平凡生成树。
- 边端点越界：抛出 `ValueError`。
- 图不连通：不会死循环，某轮 `merged_in_round == 0` 时退出。
- 重边与等权边：允许存在，算法仍正确。

## R12

运行方式：

```bash
python3 Algorithms/数学-图论-0219-最小生成树_-_Boruvka/demo.py
```

无需输入参数，脚本会直接打印两组结果。

## R13

结果解读：
- 连通图场景中，`spanning = True` 且 `components = 1`。
- 非连通图场景中，`spanning = False` 且 `components > 1`。
- `total_weight` 为被接受边权重之和。
- `rounds` 反映 Boruvka 的迭代轮数。

## R14

适用与限制：
- 适用：稀疏图、并行环境、需要分轮合并的工程场景。
- 限制：
  - 需要无向图语义；
  - 对超大图若一次性载入全部边，内存会成为瓶颈；
  - 仅给出 MST/MSF，不处理动态边更新。

## R15

与经典 MST 算法对比：
- Kruskal：先全局排序边，复杂度常写为 `O(E log E)`；实现简单。
- Prim：适合配合堆与邻接表，常见 `O(E log V)`。
- Boruvka：天然多源并行，每轮对分量独立选最小外连边，适合并行/分布式思路。

实践上也常做混合策略：先 Boruvka 降维，再切换到 Kruskal/Prim。

## R16

可扩展方向：
- 并行化边扫描：按边块并行更新局部 cheapest，再归并。
- 外存/流式处理：分批读取边，降低峰值内存。
- 混合算法：当分量数降到阈值后切换算法，以减少常数开销。
- 结果校验：增加“边数 + 连通性 + 无环性”自动断言。

## R17

最小测试清单：
- 连通小图（人工可验）是否得到正确总权重。
- 非连通图是否返回森林并正确标记 `spanning=False`。
- 含重边/等权边图是否稳定结束。
- 空图、单点图是否返回合理结果。
- 非法边端点是否触发异常。

## R18

`demo.py` 的源码级流程可拆为 7 步：

1. 初始化并查集：`parent[i]=i`，每个点自成分量。  
2. 进入主循环：当 `components > 1` 时，创建 `cheapest` 容器。  
3. 全边扫描：对每条边 `(u,v,w)`，用 `find` 拿到分量根 `ru, rv`。  
4. 过滤内部边：若 `ru == rv`，说明边在同一分量内，跳过。  
5. 更新分量最优外连边：分别尝试更新 `cheapest[ru]` 与 `cheapest[rv]`。  
6. 批量合并：遍历 `cheapest`，对候选边执行 `union`；成功则加入结果并累加权重、减少分量数。  
7. 终止判定：若本轮没有任何成功合并则退出（得到森林）；若最终分量为 1，则结果是 MST。

这 7 步即 Boruvka 的完整实现路径，没有调用第三方图算法黑盒。
