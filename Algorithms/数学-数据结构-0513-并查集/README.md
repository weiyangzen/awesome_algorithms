# 并查集

- UID: `MATH-0513`
- 学科: `数学`
- 分类: `数据结构`
- 源序号: `513`
- 目标目录: `Algorithms/数学-数据结构-0513-并查集`

## R01

并查集（Disjoint Set Union, DSU / Union-Find）是维护“不相交集合划分”的经典数据结构。

核心目标是高效支持两类动态操作：
- `find(x)`：查询元素 `x` 的集合代表元（根）。
- `union(a, b)`：把 `a` 与 `b` 所在集合合并。

它特别适合“只增不删”的连通性问题：边或关系不断加入，需要频繁判断两点是否已连通。

## R02

典型应用场景：
- 无向图动态连通性（持续加边后做连通查询）。
- Kruskal 最小生成树（判环 + 合并连通块）。
- 社交关系归并（好友团体、社区聚合）。
- 图像连通域标记（像素等价类合并）。
- 离线等价关系维护（同类项、账户合并等）。

## R03

数学模型：

给定全集 `U={0,1,...,n-1}`，维护一个分划
`P={S1,S2,...,Sk}`，满足：
- `Si` 两两不交；
- `S1 ∪ S2 ∪ ... ∪ Sk = U`。

DSU 的语义是：
- `find(x)` 返回唯一集合 `Si` 的代表元 `rep(Si)`；
- `union(a,b)` 将 `a` 与 `b` 所在集合合并（若本就同集合则保持不变）。

## R04

树表示法（父指针森林）：
- 每个集合用一棵树表示，根节点是集合代表元。
- `parent[x]` 存储父节点；根满足 `parent[root] = root`。
- `find(x)` 沿父指针向上直到根。
- `union(a,b)` 本质是“连接两棵树的根”。

这种表示法空间 `O(n)`，并且便于应用路径压缩与启发式合并。

## R05

两种关键优化：
- 路径压缩（Path Compression）：`find` 后把路径节点直接挂到根，降低后续查询深度。
- 按大小合并（Union by Size）：`union` 时总让小树挂到大树，抑制树高增长。

本目录 `demo.py` 采用“路径压缩 + 按大小合并”组合，工程实践中与“按秩合并”同属主流实现。

## R06

正确性不变量：
1. 每个节点最终都可沿 `parent` 到达某个根。  
2. 根节点满足 `parent[root]=root`。  
3. `union` 仅在不同根之间连边，不会破坏不交分划。  
4. 对任意 `x,y`，`connected(x,y)` 当且仅当 `find(x)==find(y)`。  
5. 发生有效合并时，连通分量数 `components` 恰好减 1。

因此数据结构始终表示一个合法的集合划分。

## R07

复杂度结论（路径压缩 + 按大小/秩合并）：
- `find / union / connected` 的均摊时间复杂度均为 `O(alpha(n))`。
- 其中 `alpha(n)` 是反 Ackermann 函数，增长极慢，实际规模下可近似看作常数。
- 空间复杂度 `O(n)`（`parent`、`size_arr` 等数组）。

## R08

`demo.py` 中的 MVP 结构：
- `UnionFind`：正式实现，含
  - `find`
  - `union`
  - `connected`
  - `component_size`
  - `groups`
- `NaiveDisjointSet`：朴素基线，用于随机对拍验证正确性。
- `deterministic_demo()`：固定步骤演示。
- `randomized_cross_check()`：随机操作一致性校验。

## R09

接口说明（`UnionFind`）：
- `UnionFind(n: int)`：初始化 `n` 个单元素集合。
- `find(x: int) -> int`：返回 `x` 所在集合根。
- `union(a: int, b: int) -> bool`：若发生实际合并返回 `True`，否则 `False`。
- `connected(a: int, b: int) -> bool`：判断是否同集合。
- `component_size(x: int) -> int`：返回 `x` 所在集合大小。
- `groups() -> list[list[int]]`：返回当前分组（排序后便于阅读与断言）。

## R10

边界条件与异常处理：
- `n < 0`：构造器抛 `ValueError`。
- 下标越界：`find/union/connected/component_size` 触发 `IndexError`。
- `union(x, x)`：合法调用，返回 `False`（无结构变化）。
- `n = 0`：允许空结构，但不可进行带下标查询。

## R11

确定性示例（`deterministic_demo`）流程：
- 初始元素 `0..9` 各自独立。
- 依次执行  
  `(0,1),(1,2),(3,4),(5,6),(2,6),(7,8),(8,9),(4,5)`。
- 每步打印：
  - 是否发生真实合并
  - 当前分量数 `components`
- 最后打印若干 `connected` 查询、`component_size` 与最终 `groups`。

## R12

随机对拍策略（`randomized_cross_check`）：
- 同时维护 `UnionFind` 与 `NaiveDisjointSet`。
- 随机执行 500 轮操作（默认）：
  - 60% 执行 `union`
  - 30% 执行 `connected`
  - 10% 校验 `component_size`
- 每轮都断言两者结果一致，最终断言整体分组一致。

该策略可在小成本下覆盖大量状态转移路径。

## R13

实现细节与易错点：
- `union` 一定要先对输入做 `find`，只连接“根”。
- 路径压缩可能改变中间父指针，但不改变集合语义。
- 维护 `size_arr` 时只在新根上累加，旧根值不再作为集合大小来源。
- `groups()` 内部调用 `find` 会继续压缩路径，这是预期行为。

## R14

为何该 MVP 足够“最小而诚实”：
- 未引入外部依赖，仅标准库，便于复现与验证。
- 不是黑盒调用第三方 DSU，核心逻辑全部显式实现。
- 同时给出“演示 + 自动校验”，兼顾可读性与可靠性。
- 代码规模小，便于后续扩展到带权并查集/回滚并查集。

## R15

可扩展方向：
- 带权并查集：维护势能差/相对距离。
- 回滚并查集：支持撤销合并，常用于离线动态连通性。
- 持久化并查集：支持多版本查询。
- 与图算法组合：Kruskal、离线 LCA、连通块统计与约束传播。

## R16

运行方式（无交互输入）：

```bash
uv run python Algorithms/数学-数据结构-0513-并查集/demo.py
```

如果当前工作目录已在本目录下，也可直接：

```bash
uv run python demo.py
```

## R17

交付核对：
- `README.md`：`R01-R18` 已完整填写。
- `demo.py`：可直接运行，含并查集 MVP 与随机对拍验证。
- `meta.json`：UID/学科/分类/源序号/目录信息与任务保持一致。
- 目录自包含，不依赖交互输入即可完成验证。

## R18

`demo.py` 源码级算法流程（8 步）：

1. 初始化 `UnionFind(n)`：创建 `parent[i]=i`、`size_arr[i]=1`，并记录 `components=n`。  
2. 调用 `find(x)` 时，沿 `parent` 向上找根；路径上执行“路径折半压缩”（`parent[x]=parent[parent[x]]`）。  
3. 调用 `union(a,b)` 时，先分别求根 `ra=find(a)`、`rb=find(b)`。  
4. 若 `ra==rb`，说明两点已在同一集合，直接返回 `False`。  
5. 若根不同，比较 `size_arr[ra]` 与 `size_arr[rb]`，保证小树根挂到大树根（必要时交换 `ra, rb`）。  
6. 执行合并：`parent[rb]=ra`，并更新 `size_arr[ra]+=size_arr[rb]`、`components-=1`。  
7. `connected(a,b)` 通过比较 `find(a)` 与 `find(b)` 判断连通；`component_size(x)` 通过根索引读取集合规模。  
8. 主程序先跑确定性样例，再用 `NaiveDisjointSet` 做随机对拍，逐轮断言算法状态与结果一致。

该实现不依赖第三方并查集黑盒，步骤与源码函数完全一一对应。
