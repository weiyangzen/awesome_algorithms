# 树形DP - 树的最大独立集

- UID: `CS-0060`
- 学科: `计算机`
- 分类: `动态规划`
- 源序号: `77`
- 目标目录: `Algorithms/计算机-动态规划-0077-树形DP_-_树的最大独立集`

## R01

问题定义（树的最大独立集，Maximum Independent Set on Tree）：
- 给定一棵树 `T=(V,E)`；
- 独立集 `S ⊆ V` 满足：任意边 `(u,v) ∈ E`，不允许 `u,v` 同时在 `S` 中；
- 目标是最大化 `S` 的总权重。

本 MVP 默认支持两种场景：
- 无权版本：每个节点权重默认为 `1`，目标是最大化节点数量；
- 加权版本：给定非负整数权重数组 `weights[i]`，目标是最大化权重和。

## R02

输入输出约定：
- 输入：
  - `n`：节点个数，节点编号为 `0..n-1`；
  - `edges`：无向边集合，长度应为 `n-1`；
  - `weights`：可选，若省略则自动设为全 `1`。
- 输出（主算法 `tree_max_independent_set`）：
  - `best_weight`：最大独立集权重；
  - `selected_nodes`：一组可执行的最优独立集节点；
  - `selected_weight`：`selected_nodes` 对应权重和；
  - `include/exclude`：DP 两状态数组（便于调试与解释）。

## R03

状态定义（以某个根 `root` 把树定向后）：
- `include[u]`：在 `u` 必须被选中的前提下，`u` 子树可取得的最大权重；
- `exclude[u]`：在 `u` 不被选中的前提下，`u` 子树可取得的最大权重。

边界（叶子节点）：
- `include[u] = w[u]`
- `exclude[u] = 0`

## R04

状态转移：
- 若选 `u`，其子节点都不能选：
  - `include[u] = w[u] + Σ exclude[child]`
- 若不选 `u`，每个子节点可自由选或不选：
  - `exclude[u] = Σ max(include[child], exclude[child])`

最终答案：
- `best_weight = max(include[root], exclude[root])`

## R05

最优子结构成立原因：
- 树没有环，切断父子边后，各子树相互独立；
- 在“父节点是否被选”条件固定时，子树最优决策互不干扰；
- 因此可以把整棵树最优值分解为多个子树最优值的可加组合。

这也是树形 DP 可成立的核心条件。

## R06

遍历与求值顺序：
- 先用 DFS 把无向树转成有根树（得到 `children`）；
- 再按后序顺序（代码里通过 `reversed(order)`）计算 `include/exclude`；
- 这样每个节点在计算时，其所有子节点状态都已就绪。

`demo.py` 使用非递归栈构建 `order`，避免大深度树上的递归栈风险。

## R07

复杂度分析：
- 主算法（树形 DP）：
  - 时间复杂度：`O(n)`（每条边、每个节点常数次处理）；
  - 空间复杂度：`O(n)`（邻接表、`include/exclude`、父子关系）。
- 记忆化递归基线：
  - 时间复杂度：`O(n)`；
  - 空间复杂度：`O(n)`。
- 暴力基线：
  - 时间复杂度：`O(2^n * (n + |E|))`，仅用于小规模校验。

## R08

`demo.py` 模块结构：
- 输入校验：`to_node_count`、`to_weight_array`、`to_edge_list`；
- 图构造：`build_adjacency`、`assert_connected_tree`、`prepare_tree`；
- 树定根：`build_rooted_tree`；
- 主算法：`tree_max_independent_set`；
- 独立基线：`tree_mis_top_down`、`brute_force_tree_mis`；
- 校验与运行：`is_independent_set`、`run_case`、`randomized_cross_check`、`main`。

## R09

核心接口说明：
- `tree_max_independent_set(n, edges, weights=None, root=0) -> TreeMISResult`
  - 主实现，返回最优值与一组最优解节点；
- `tree_mis_top_down(n, edges, weights=None, root=0) -> int`
  - 记忆化递归对照实现；
- `brute_force_tree_mis(n, edges, weights=None) -> tuple[int, list[int]]`
  - 小规模精确枚举，用于 correctness cross-check。

## R10

固定样例覆盖（`main()` 内置）：
1. 链式无权树：`n=5`，期望最优值 `3`；
2. 星型无权树：`n=6`，期望最优值 `5`；
3. 单节点树：期望 `1`；
4. 完全二叉结构加权树：验证一般加权场景，期望 `18`；
5. 空树：期望 `0`。

## R11

运行方式（无交互输入）：

```bash
uv run python Algorithms/计算机-动态规划-0077-树形DP_-_树的最大独立集/demo.py
```

若当前目录已在该算法目录下：

```bash
uv run python demo.py
```

## R12

输出字段解读：
- `dp_result`：主算法结果（最优权重 + 选中节点 + 节点权重和）；
- `top_down`：记忆化递归基线最优权重；
- `bruteforce`：暴力最优权重与节点（小规模时执行）；
- `checks`：
  - `independent`：返回节点是否真的是独立集；
  - `selected_sum_match`：返回节点权重和是否等于 `best_weight`；
  - `top_down_match` / `brute_match`：不同实现是否一致。

## R13

随机对拍策略：
- 默认 `300` 轮；
- 每轮随机生成：
  - `n ∈ [0, 14]`；
  - 一棵随机树（递增节点随机挂父）；
  - 权重 `1..20`；
- 对每轮断言：
  - 树形 DP 结果 == 记忆化递归结果；
  - 树形 DP 结果 == 暴力结果。

该策略覆盖大量非人工构造结构，能显著提升正确性信心。

## R14

边界与异常处理：
- `n < 0`、`n` 非整数：抛 `ValueError`；
- `edges` 形状非法、边数不是 `n-1`：抛 `ValueError`；
- 自环、重复边、端点越界：抛 `ValueError`；
- 图不连通（非树）：抛 `ValueError`；
- `weights` 维度、长度、数值非法（`NaN/Inf`、负数、非整数）：抛 `ValueError`。

## R15

MVP 取舍说明（小而诚实）：
- 依赖仅 `numpy + 标准库`，工具栈最小；
- 主算法不依赖任何第三方黑盒图算法库；
- 同时实现“主算法 + 递归基线 + 暴力基线”，确保可核验；
- 输出不仅给最优值，也给可执行节点集合，保证可解释性。

## R16

可扩展方向：
- 支持负权节点（当前限制为非负，便于教学与校验）；
- 增加“字典序最小最优解”等可控 tie-break 规则；
- 扩展到森林（多棵树）与动态增删边场景；
- 面向超大树加入迭代内存优化与并行子树求值。

## R17

交付核对：
- `README.md`：`R01-R18` 完整，且无占位符；
- `demo.py`：可直接运行、无交互输入、无占位符；
- `meta.json`：保留并与任务元数据一致（UID/学科/子类/源序号/目录）。

该目录已按单条算法任务要求自包含。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 组织固定样例并调用 `run_case`，最后执行随机对拍。  
2. `run_case` 先调用 `prepare_tree` 做输入校验（`n/edges/weights`）并构建邻接表。  
3. 主算法 `tree_max_independent_set` 调用 `build_rooted_tree`，把无向树转成有根树并得到遍历顺序。  
4. 按后序顺序计算 `include[u]` 与 `exclude[u]`：`include` 走“选当前节点”，`exclude` 走“不选当前节点”。  
5. 计算根节点最优值 `max(include[root], exclude[root])` 后，按同样的状态约束做一次非递归回溯，恢复 `selected_nodes`。  
6. `run_case` 再调用 `tree_mis_top_down`（`lru_cache`）作为独立 DP 基线。  
7. 对小规模样例，`run_case` 再调用 `brute_force_tree_mis` 穷举全部节点子集，筛掉非独立集并取最优值。  
8. 若主算法、记忆化基线、暴力基线任一不一致，立即断言失败；全部通过后打印结果与随机对拍通过信息。  
