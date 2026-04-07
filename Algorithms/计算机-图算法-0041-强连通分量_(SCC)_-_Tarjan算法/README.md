# 强连通分量 (SCC) - Tarjan算法

- UID: `CS-0031`
- 学科: `计算机`
- 分类: `图算法`
- 源序号: `41`
- 目标目录: `Algorithms/计算机-图算法-0041-强连通分量_(SCC)_-_Tarjan算法`

## R01

强连通分量（Strongly Connected Component, SCC）定义在有向图中：

- 若顶点集合内任意两点 `u, v` 均满足 `u -> v` 且 `v -> u`，则该集合强连通；
- 极大强连通子图称为一个 SCC。

Tarjan 算法用一次 DFS 就能在线性时间内求出全部 SCC，是 SCC 任务的经典实现之一。

## R02

本条目求解目标：

- 输入：有向图 `G=(V,E)`，顶点编号 `0..n-1`，边集合 `[(u,v), ...]`；
- 输出：
1. `sccs`：所有 SCC 的顶点集合；
2. `component_id_of_vertex`：每个顶点所属 SCC 编号；
3. `discovery_index`：DFS 首次访问时间戳；
4. `low_link`：Tarjan 低链值；
5. `condensation_edges` 与 `condensation_topo_order`：缩点 DAG 及其拓扑序。

`demo.py` 为固定样例，无需交互输入。

## R03

Tarjan 的核心是两个数组与一个栈：

- `discovery_index[u]`：顶点 `u` 被 DFS 首次发现的编号；
- `low_link[u]`：从 `u` 出发，经过若干树边和至多一条回边，能到达的最小发现编号；
- `stack` + `on_stack`：维护当前 DFS 路径上“尚未归属 SCC”的活跃顶点。

关键判定：当 `low_link[u] == discovery_index[u]` 时，`u` 是一个 SCC 的根，可从栈顶连续弹出直到 `u`，得到完整 SCC。

## R04

本目录 MVP 的 Tarjan 流程：

1. 构建邻接表并校验边端点范围；
2. 逐点启动 DFS（覆盖非连通图）；
3. 访问顶点时写入 `discovery_index/low_link`，并压栈；
4. 沿树边递归，回溯时用子节点 `low_link` 更新父节点；
5. 遇到“指向栈内节点”的回边时，用目标节点 `discovery_index` 更新 `low_link`；
6. 发现 SCC 根后弹栈形成分量；
7. 对分量做稳定排序，构建 `component_id_of_vertex`；
8. 依据跨分量边生成缩点 DAG，并做 Kahn 拓扑排序。

## R05

设 `V=|V|`、`E=|E|`：

- 时间复杂度：`O(V + E)`（每个点和每条边最多常数次处理）；
- 空间复杂度：`O(V + E)`（邻接表、索引数组、栈、结果结构）。

## R06

`demo.py` 的主样例（Case-1）结构：

- `0 -> 1 -> 2 -> 0` 形成 SCC-A；
- `3 -> 4 -> 5 -> 3` 形成 SCC-B；
- `6 <-> 7` 形成 SCC-C；
- 额外跨分量边 `2 -> 3`、`5 -> 6`。

预期输出：

- SCC 集合：`{0,1,2}`、`{3,4,5}`、`{6,7}`；
- 缩点边：`A -> B`、`B -> C`。

## R07

Tarjan 的优点：

- 单次 DFS，无需构建反图；
- 线性复杂度，静态图效率高；
- `low_link` 与栈状态提供较强可解释性。

局限：

- 递归实现在超深链图上可能触发 Python 递归深度限制；
- 在线动态图（频繁增删边）不适合直接重复全量 Tarjan；
- 理解门槛高于“二次 DFS”的 Kosaraju。

## R08

依赖与实现策略：

- Python 3.10+；
- `numpy` 仅用于组织固定测试边数据；
- SCC 提取、缩点构建、拓扑排序均为手写实现。

即：未调用 `networkx` 等黑盒 API，算法过程可直接追踪到源码细节。

## R09

适用场景：

- 代码依赖循环检测（模块/包引用环）；
- 工作流状态图中环结构识别；
- 图问题预处理：先缩点为 DAG，再做 DP 或拓扑优化。

不适用场景：

- 高频在线更新的动态图实时 SCC 维护；
- 目标是最短路/最大流等非 SCC 问题。

## R10

正确性直觉：

1. DFS 过程中，栈中顶点始终是“已访问但尚未归属 SCC”的候选；
2. `low_link[u]` 收集了 `u` 子树内可回到的最早祖先编号；
3. 若 `low_link[u] < discovery_index[u]`，说明 `u` 还能回到更早节点，不是分量根；
4. 若二者相等，`u` 以下栈内一段顶点与外部无法互相强连通扩展，恰好构成一个极大 SCC；
5. 每个顶点只会入栈/出栈一次，因此分量划分互斥且覆盖全体顶点。

## R11

实现防坑清单：

- 只在“目标点仍在栈中”时使用回边更新 `low_link`；
- 递归返回后必须执行 `low_link[parent] = min(low_link[parent], low_link[child])`；
- SCC 根判定必须用 `low_link[u] == discovery_index[u]`；
- 分量输出若用于测试，建议排序保证稳定；
- 缩点边应用集合去重，避免并行边重复。

## R12

工程化成本评估：

- 代码体量小（百行级），无外部图框架耦合；
- 调试路径明确：先看 `discovery_index/low_link`，再看 `sccs`；
- 测试建议覆盖：
1. 多个 SCC 串联；
2. 非连通图与孤立点；
3. 自环、重边、DAG（每点单独成分量）。

## R13

算法性质：

- 精确算法（非近似）；
- 对固定输入与固定遍历顺序输出可复现；
- 结果天然支持后续缩点 DAG 分析；
- 可作为编译器依赖分析、图数据库预处理基础组件。

## R14

常见失效模式与防护：

- 失效：忽略 `on_stack` 条件，把所有已访问点都当回边目标。  
  防护：仅在 `on_stack[v]` 为真时用 `discovery_index[v]` 更新 `low_link[u]`。

- 失效：漏掉“子树回溯更新”导致 `low_link` 偏大。  
  防护：树边递归返回后立即执行 `min` 合并。

- 失效：SCC 根判定条件写错导致分量切分异常。  
  防护：严格使用 `low_link[u] == discovery_index[u]`。

- 失效：缩点图边重复，影响后续统计。  
  防护：先放入 `set`，最终排序输出。

## R15

实践建议：

- 先验证 SCC 集合正确性，再验证缩点边与拓扑序；
- 打印 `discovery_index` 与 `low_link` 可快速定位错误递归分支；
- 深图场景可改显式栈版 DFS，规避递归深度限制；
- 若业务侧使用实体名，可在分量结果后追加 ID 映射层。

## R16

相关算法对比：

- Kosaraju：两次 DFS + 反图，思路直观；
- Tarjan：单次 DFS + low-link，常数因子通常更优；
- Gabow：双栈法，亦为线性 SCC 算法。

扩展方向：

- 支持显式栈 Tarjan（非递归）；
- 在缩点 DAG 上做关键路径、可达域压缩、DAG DP。

## R17

本目录 `demo.py` 主要函数：

- `build_directed_graph`：构建邻接表并做边合法性校验；
- `tarjan_scc`：单次 DFS 提取 SCC、构建分量编号与缩点信息；
- `topological_sort_dag`：对缩点 DAG 计算拓扑序；
- `assert_case`：校验 SCC/缩点边/顶点覆盖；
- `run_demo_cases`：运行 3 个固定样例并打印中间结果。

运行方式：

```bash
cd Algorithms/计算机-图算法-0041-强连通分量_(SCC)_-_Tarjan算法
uv run python demo.py
```

脚本无交互输入，成功时打印 `All demo cases passed.`。

## R18

`demo.py` 的源码级流程可拆为 9 步（非黑盒）：

1. `run_demo_cases` 使用 `numpy` 组织固定边数据并转成 `list[tuple[int,int]]`。  
2. `tarjan_scc` 调用 `build_directed_graph` 生成邻接表并校验端点范围。  
3. 初始化 `discovery_index/low_link/on_stack/stack/time` 等 Tarjan 状态。  
4. 对每个未访问顶点调用 `strong_connect`，写入发现时间并压栈。  
5. 遇到未访问邻点走树边递归，回溯后合并 `low_link`；遇到栈内邻点走回边更新 `low_link`。  
6. 当 `low_link[u] == discovery_index[u]` 时，从栈顶连续弹出直到 `u`，得到一个原始 SCC。  
7. 对 SCC 做稳定排序，构建 `component_id_of_vertex`，保证输出可复现。  
8. 扫描原图边并提取跨分量边，得到去重后的 `condensation_edges`，再做 DAG 拓扑排序。  
9. `assert_case` 对每个样例断言 SCC 集合、缩点边和顶点覆盖，全部通过后输出成功标记。

第三方库拆解：

- `numpy` 只负责样例数组承载；
- Tarjan 主过程、low-link 更新、SCC 弹栈、缩点构建、拓扑排序均为源码手写实现。
