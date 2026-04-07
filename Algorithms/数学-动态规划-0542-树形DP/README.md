# 树形DP

- UID: `MATH-0542`
- 学科: `数学`
- 分类: `动态规划`
- 源序号: `542`
- 目标目录: `Algorithms/数学-动态规划-0542-树形DP`

## R01

树形 DP（Tree Dynamic Programming）是把动态规划状态定义在树节点上的方法。  
核心思想是把“整棵树问题”拆成“子树问题”，通过一次后序遍历把子树最优解向上汇总。

本条目用经典模型“树上最大权独立集”作为最小可运行示例。

## R02

问题定义（树上最大权独立集，Maximum Weight Independent Set on Tree）：
- 输入：一棵 `n` 个节点的无向树 `T=(V,E)`，每个节点 `u` 有非负权重 `w_u`。
- 约束：选择节点集合 `S ⊆ V`，且任意边 `(u,v) ∈ E` 不能同时满足 `u,v ∈ S`（相邻点不能同选）。
- 目标：最大化
\[
\max_{S} \sum_{u\in S} w_u.
\]

## R03

设树根为 `root`，定义两类状态：
- `take[u]`：强制选中节点 `u` 时，`u` 子树能取得的最大权重。
- `skip[u]`：强制不选节点 `u` 时，`u` 子树能取得的最大权重。

对每个子节点 `v`（`v` 是 `u` 的孩子）有转移：
\[
take[u] = w_u + \sum_{v} skip[v],
\]
\[
skip[u] = \sum_{v} \max(take[v], skip[v]).
\]

最终答案为 `max(take[root], skip[root])`。

## R04

遍历顺序要求“先子后父”，因此使用后序顺序：
1. 先把无向树定根（本实现默认根为 `0`）。
2. 再按后序顺序处理节点，使每个节点更新时子节点状态已知。

`demo.py` 使用迭代 DFS 生成父数组和后序序列，避免递归深度依赖。

## R05

初始化和边界：
- 叶子节点：`take[leaf]=w_leaf`，`skip[leaf]=0`（由通式自然得到）。
- 单节点树：答案是 `max(w_0, 0)`；在非负权设定下即 `w_0`。
- 输入必须是合法树：`|E|=n-1` 且连通，且无自环、端点编号合法。

本实现在 `validate_tree_instance` 中完成上述校验。

## R06

为了输出具体方案（不仅是最优值），`demo.py` 额外做一次自顶向下回溯：
- 若父节点已选，当前节点必须不选。
- 若父节点未选，当前节点在 `take[u]` 与 `skip[u]` 中选较大者（平局时选 `take[u]`）。

这样可恢复一组可行且与 DP 最优值一致的节点集合。

## R07

伪代码：

```text
input: tree (weights, edges), root
validate tree and build adjacency list
compute parent[] and postorder[]

for u in postorder:
    take[u] = weight[u]
    skip[u] = 0
    for child v of u:
        take[u] += skip[v]
        skip[u] += max(take[v], skip[v])

best = max(take[root], skip[root])
reconstruct one optimal set by top-down decisions
return best and selected nodes
```

## R08

正确性要点：
1. 最优子结构：`u` 子树最优值可由各个子树最优值组合得到。
2. 约束局部化：树无环，节点 `u` 的“选/不选”仅通过父子关系影响相邻节点选择。
3. 无后效性：`take[u]`、`skip[u]` 一旦确定，向上层传递时不需记录更细路径信息。

因此上述状态与转移可得到全局最优。

## R09

复杂度分析：
- 时间复杂度：`O(n)`，每条边仅在常数次操作中访问。
- 空间复杂度：`O(n)`，存储邻接表、父数组、`take/skip` 与回溯标记。

这也是树上 DP 相比一般图 DP 的关键优势之一。

## R10

常见错误：
- 把一般图当作树处理，未校验 `n-1` 条边与连通性。
- 在无向图上忘记“跳过父节点”，导致把父边重复算进子问题。
- 只计算最优值不回溯具体方案，难以排查错误。
- 无小规模真值校验，代码可能“有输出但不正确”。

本 MVP 内置 `brute_force_mwis` 做小规模对拍。

## R11

`demo.py` 模块划分：
- `TreeInstance`：树问题输入数据结构。
- `validate_tree_instance`：树合法性校验并构建邻接表。
- `build_rooted_parent_and_postorder`：迭代方式定根并生成后序序。
- `solve_tree_mwis`：核心树形 DP + 方案回溯。
- `brute_force_mwis`：穷举所有节点子集，作为真值基准。
- `is_independent_set`：可行性检查。
- `main`：构造样例、执行算法、打印结果并断言正确性。

## R12

运行方式（无交互）：

```bash
cd Algorithms/数学-动态规划-0542-树形DP
uv run python demo.py
```

脚本会直接打印节点表、最优值、选择方案与对拍结果。

## R13

内置样例：
- 节点数 `n=9`
- 权重：`[6, 4, 7, 3, 8, 5, 2, 9, 4]`
- 边：`(0,1),(0,2),(1,3),(1,4),(2,5),(2,6),(5,7),(5,8)`

该样例是分层树结构，能清晰体现“父子互斥、隔代可共选”的树形 DP 特征。  
`demo.py` 会输出 DP 结果与穷举真值是否一致。

## R14

可调参数：
- 改 `weights` 与 `edges` 可测试不同树结构。
- 如需压力测试，可增加节点数并关闭穷举对拍（穷举是指数级）。
- 当前穷举仅用于小规模可靠性验证，工程场景只保留 `O(n)` DP 主流程。

## R15

和常见线性 DP 的关系：
- 线性 DP 依赖固定序列（如数组前缀）。
- 树形 DP 依赖树拓扑关系（父子子树）。

和其他树上问题的关系：
- 换状态可得到“树上最小点覆盖”“树上匹配”“树上背包”等模型。
- 核心框架相同：定根 + 子树状态 + 后序合并。

## R16

应用场景：
- 组织层级预算选择：相邻层级岗位互斥时的效益最大化。
- 通信网络维护：相邻站点不可同时停机时选择收益最大维护集合。
- 依赖树上的冲突任务挑选：直接依赖节点不能共选。
- 分层风险隔离：相邻风险点不同时触发时的收益优化。

## R17

可扩展方向：
- 支持负权重：允许“宁可不选”，并在回溯策略中处理平局规则。
- 多状态树形 DP：例如“必须选 k 个节点”或“边带约束”。
- 重链剖分/树分治结合：处理在线修改与查询。
- 把本地穷举对拍升级为随机树自动化回归测试。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 构造固定树样例（节点权重 + 边），调用 `solve_tree_mwis`。
2. `solve_tree_mwis` 先调用 `validate_tree_instance` 检查输入是否真的是树并得到邻接表。
3. 调 `build_rooted_parent_and_postorder` 在根 `0` 上生成 `parent` 与后序序列。
4. 初始化 `take/skip` 两个长度为 `n` 的数组，并按后序遍历节点。
5. 对每个节点 `u`，依据 `take[u]=w_u+Σskip[child]`、`skip[u]=Σmax(take,skip)` 完成状态转移。
6. 取 `best=max(take[root],skip[root])`，再自顶向下按“父选则子不能选”的规则回溯出一组最优节点集。
7. `main` 调用 `brute_force_mwis` 穷举全部子集得到真值，并用 `is_independent_set` 校验方案可行性。
8. 打印节点级 DP 表、最优值与方案节点；若“回溯值不等于 DP 值”或“DP 值不等于穷举值”则抛错。
