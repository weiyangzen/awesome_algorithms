# 二分图匹配 - Hopcroft-Karp

- UID: `MATH-0492`
- 学科: `数学`
- 分类: `图论`
- 源序号: `492`
- 目标目录: `Algorithms/数学-图论-0492-二分图匹配_-_Hopcroft-Karp`

## R01

本条目实现 `Hopcroft-Karp` 的最小可运行版本（MVP），用于求解**二分图最大基数匹配**（maximum cardinality matching）。MVP 目标：
- 输入左部点集 `U`、右部点集 `V` 与边集 `E`；
- 输出最大匹配大小与匹配对；
- 明确分层 BFS + 限层 DFS 的核心流程；
- 包含最小正确性校验（与小规模暴力参考解比对）；
- `python3 demo.py` 无需交互即可直接运行。

## R02

问题定义（MVP 范围）：
- 二分图 `G=(U,V,E)`，其中 `|U|=n_left`，`|V|=n_right`，边只连接 `U -> V`。
- 匹配 `M` 满足任意两条边不共享端点。
- 目标是最大化 `|M|`（最大匹配边数），不要求权重，不涉及最小费用。
- 输出：
  - `pair_left[u]`：左节点 `u` 匹配到的右节点编号（未匹配为 `-1`）；
  - `pair_right[v]`：右节点 `v` 匹配到的左节点编号（未匹配为 `-1`）；
  - `matching_size` 与运行统计信息。

## R03

关键概念：增广路与分层图。
- 增广路：一条从未匹配左点出发、到未匹配右点结束、边在“未匹配/已匹配”间交替的路径。
- 若找到增广路并沿路翻转匹配状态，匹配大小增加 `1`。
- Hopcroft-Karp 在每轮先做 BFS，构造“最短增广路层次”，然后在该层次图上做 DFS 批量找点不相交增广路。
- 每轮可以同时增广多条最短路，因此总复杂度优于朴素逐条增广。

## R04

算法流程（MVP）：
1. 校验输入并去重重复边；
2. 构建左到右邻接表 `adj[u]`；
3. 初始化 `pair_left/pair_right=-1`（全未匹配）；
4. BFS：从所有未匹配左点出发，建立到可继续扩展左点的层次；
5. 若 BFS 未触达任何未匹配右点，结束（当前匹配即最大）；
6. DFS：仅沿 BFS 允许的层次边尝试增广；
7. 对每个未匹配左点执行 DFS，成功一次 `matching_size += 1`；
8. 重复 BFS+DFS 直到无增广路。

## R05

核心数据结构：
- `edges: List[Tuple[int,int]]`：标准化后的边集（已去重）；
- `adj: List[List[int]]`：邻接表，`adj[u]` 存右侧邻居；
- `pair_left: numpy.ndarray[int]`：左侧匹配数组；
- `pair_right: numpy.ndarray[int]`：右侧匹配数组；
- `dist: numpy.ndarray[int]`：BFS 分层距离（左侧节点）；
- `HopcroftKarpResult(dataclass)`：统一封装匹配输出与统计。

## R06

正确性要点：
- 若当前匹配非最大，则一定存在增广路（Berge 定理）。
- BFS 仅构造最短增广路层次；DFS 只在层次图内前进，因此找到的是最短增广路集合。
- 同一轮批量增广后，不会破坏匹配合法性，且匹配大小单调增加。
- 当 BFS 不再能到达任何未匹配右点时，不存在增广路，匹配达到最大。

## R07

复杂度：
- 设 `V = n_left + n_right`，`E = |edges|`。
- 时间复杂度：`O(E * sqrt(V))`（Hopcroft-Karp 的经典上界）。
- 空间复杂度：
  - 匹配与分层数组 `O(V)`；
  - 邻接表 `O(E)`；
  - 总体 `O(V + E)`。

## R08

边界与异常处理：
- `n_left` 或 `n_right` 为负：抛出 `ValueError`；
- 边端点越界：抛出 `ValueError`；
- 允许空图与孤立点，结果匹配大小可为 `0`；
- 重复边自动去重，不影响匹配结果；
- demo 中额外验证 `pair_left/pair_right` 一致性，防止伪匹配。

## R09

MVP 取舍：
- 仅依赖 `numpy` 与标准库，保持轻量；
- 不调用 NetworkX 或 SciPy 图匹配黑盒；
- 增加一个仅用于小图的暴力回溯参考解，交叉检查最大匹配大小；
- 不实现带权匹配（Hungarian / min-cost matching）与动态更新。

## R10

`demo.py` 模块职责：
- `validate_input`：输入检查 + 边去重；
- `build_adjacency`：构造左侧邻接表；
- `hopcroft_karp`：BFS + DFS 核心求解；
- `matching_pairs`：把数组形式转换为 `(u, v)` 列表；
- `is_valid_matching`：验证匹配合法性；
- `brute_force_maximum_size`：小图参考最优值；
- `run_demo_case`：固定样例运行与断言；
- `main`：无交互入口。

## R11

运行方式：

```bash
cd Algorithms/数学-图论-0492-二分图匹配_-_Hopcroft-Karp
python3 demo.py
```

脚本将打印最大匹配大小、匹配对、算法统计，并输出校验通过信息。

## R12

输出解读：
- `maximum matching size`：最大匹配边数；
- `matching pairs (left -> right)`：每个匹配对；
- `stats`：
  - `bfs_rounds`：BFS 分层轮数；
  - `dfs_calls`：DFS 调用次数；
  - `edge_scans`：扫描边次数（BFS+DFS）；
  - `augmenting_paths`：成功增广路径条数；
- `All checks passed.`：匹配合法性和参考解比对均通过。

## R13

建议最小测试集：
- 存在完美匹配的稀疏图；
- 含孤立左点或右点的非完美图；
- 重复边输入（验证去重逻辑）；
- 空边集与单边图；
- 非法输入（负规模、越界端点）。

## R14

可调参数：
- `n_left`、`n_right`：左右点集规模；
- `edges`：二分图结构；
- demo 是否启用参考暴力校验（大图可关闭以节省时间）。

实践建议：先在 `|U|,|V| <= 20` 范围内做回归，再扩大规模观测 `edge_scans` 与 `bfs_rounds`。

## R15

方法对比：
- 对比 Kuhn/DFS 逐条增广：
  - Kuhn 常见 `O(VE)`；
  - Hopcroft-Karp 通过“分层+批量增广”降到 `O(E*sqrt(V))`。
- 对比一般图匹配（Edmonds Blossom）：
  - Hopcroft-Karp 只适用于二分图，但实现更简单、速度更快；
  - Blossom 处理一般图奇环，复杂度与实现复杂度都更高。

## R16

典型应用：
- 任务-工人分配（只关心可分配性与最大覆盖）；
- 课程-教室、岗位-候选人等一对一匹配可行性；
- 稀疏推荐中的最大不冲突配对；
- 作为更复杂匹配/流模型的基础模块。

## R17

可扩展方向：
- 输出最小点覆盖（可由最大匹配导出，Kőnig 定理）；
- 增加按层可视化，展示增广过程；
- 支持从文本文件/CSV 读取大规模图；
- 针对超大图做迭代式内存优化与性能基准；
- 扩展到带权场景（改用 Hungarian / min-cost flow）。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `run_demo_case` 构造固定二分图，调用 `validate_input` 去重并检查端点范围。  
2. `hopcroft_karp` 初始化 `pair_left/pair_right` 为 `-1`，表示所有点未匹配。  
3. 内部 `bfs` 从全部未匹配左点入队，为左侧节点建立分层距离 `dist`。  
4. BFS 遇到未匹配右点时标记“存在可增广路径”，并继续完成本轮层次扩展。  
5. 若本轮 BFS 无法触达任何未匹配右点，外层循环终止，当前匹配即最大。  
6. 内部 `dfs` 仅沿满足 `dist[u2] == dist[u] + 1` 的层次边递归，成功时翻转匹配边。  
7. 每个未匹配左点尝试 DFS，成功一次就将 `matching_size` 与 `augmenting_paths` 增加 `1`。  
8. 返回结果后，`is_valid_matching` 检查匹配合法性，再用 `brute_force_maximum_size` 校验最优值。  
