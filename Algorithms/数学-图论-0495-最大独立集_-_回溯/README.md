# 最大独立集 - 回溯

- UID: `MATH-0495`
- 学科: `数学`
- 分类: `图论`
- 源序号: `495`
- 目标目录: `Algorithms/数学-图论-0495-最大独立集_-_回溯`

## R01

本条目实现**最大独立集（Maximum Independent Set, MIS）**的回溯（Backtracking）MVP，并加入基础分支限界（Branch and Bound）剪枝。

- 目标：在无向图 `G=(V,E)` 中找到一个顶点集合 `S`，使得任意 `u,v ∈ S` 都不相邻，且 `|S|` 最大。
- 输出：
  - 最大独立集大小 `alpha(G)`；
  - 一个可行最优顶点集合；
  - 搜索统计（访问节点数、被上界剪掉的分支数）。
- 实现定位：小到中等规模图的精确求解与教学演示，不追求工业级超大图性能。

## R02

问题定义（MVP 范围）：

- 输入：
  - 顶点数 `n`（顶点编号 `0..n-1`）；
  - 无向边集 `edges = [(u,v), ...]`。
- 约束：
  - 不允许自环；
  - 允许重复边（会被位掩码天然去重）；
  - 顶点编号必须合法。
- 输出：`MISResult(size, vertices, nodes_visited, pruned_by_bound)`。

## R03

数学建模：

1. 独立集定义：`S ⊆ V`，若 `∀(u,v)∈E, {u,v} ⊄ S`，则 `S` 为独立集。  
2. 目标函数：
   `maximize |S|`，约束为 `S` 独立。  
3. 等价视角：最大独立集与补图上的最大团等价，但本实现直接在原图上回溯。  
4. 复杂度背景：该问题是 NP-hard，通用精确算法最坏情况下指数时间。

## R04

算法流程（回溯 + 上界剪枝）：

1. 用位掩码邻接表 `adj[v]` 表示图。  
2. 递归状态：
   - `candidate_mask`：当前还可考虑的候选顶点集合；
   - `chosen_mask`：当前已选入独立集的顶点集合；
   - `chosen_size`：`chosen_mask` 的大小。  
3. 上界：`chosen_size + popcount(candidate_mask)`。若该上界不超过当前最优值，直接剪枝。  
4. 从候选中选一个分支顶点 `v`（策略：候选子图内度最大）。  
5. 分支 A（选 `v`）：进入 `candidate \ {v} \ N(v)`。  
6. 分支 B（不选 `v`）：进入 `candidate \ {v}`。  
7. 当 `candidate_mask=0` 时触发叶子更新最优解。

## R05

核心数据结构：

- `List[int] adj`：每个整数是一个位集，记录该顶点邻居。
- `int mask`：
  - 第 `i` 位为 `1` 表示顶点 `i` 在集合中；
  - 用 `popcount()`（位运算计数）快速取集合大小。
- `MISResult`（`dataclass`）：统一承载最优值与搜索统计。

这种表示法避免频繁创建 Python `set`，在回溯中常数较小。

## R06

正确性要点：

- 可行性保持：
  - 当选择顶点 `v` 时，递归前移除 `v` 的全部邻居，因此后续不会选到与 `v` 冲突的点。  
- 完备性：
  - 任意最优解对某个分支顶点 `v` 必然属于“选 `v`”或“不选 `v`”两类之一，双分支覆盖全部可能。  
- 最优性：
  - 对所有可行组合都进行枚举（除被安全上界剪枝的无效分支）；
  - 叶子节点处只在 `chosen_size` 更优时更新全局最优。

## R07

复杂度：

- 最坏时间复杂度：`O(2^n)`（NP-hard 问题的典型精确求解代价）。
- 空间复杂度：`O(n)`（主要是递归深度与常数状态，图存储为 `O(n + m)` 的位掩码邻接信息）。
- 实际表现依赖图结构与剪枝强度：稠密图通常更易被剪枝，稀疏图搜索树可能更大。

## R08

边界与异常处理：

- `n <= 0`：抛出 `ValueError`。
- 边端点越界：抛出 `ValueError`。
- 自环边 `(u,u)`：抛出 `ValueError`（本 MVP 处理简单无向图）。
- 空图（`|E|=0`）：答案为全部顶点。
- 完全图：答案大小为 `1`（`n>0`）。

## R09

MVP 取舍说明：

- 保留精确求解，不引入近似算法。
- 仅使用 Python 标准库，不依赖第三方图算法黑盒。
- 剪枝仅用简单可解释上界 `chosen + remaining`，未加入更复杂上界（如着色上界）。
- 演示同时包含：
  - 有解析答案的小图（用于断言）；
  - 中等随机图（展示搜索统计）。

## R10

`demo.py` 职责划分：

- `build_adjacency_masks`：构图与输入校验。
- `choose_branch_vertex`：选择分支点（候选子图内度最大）。
- `maximum_independent_set_backtracking`：主算法。
- `is_independent_mask`：结果可行性校验。
- `brute_force_mis`：小图基线（用于自检）。
- `cycle_graph_edges` / `complete_bipartite_edges` / `random_graph_edges`：样例图生成。
- `run_case`：单案例执行与打印。
- `main`：无交互入口，固定三组案例。

## R11

运行方式：

```bash
cd Algorithms/数学-图论-0495-最大独立集_-_回溯
python3 demo.py
```

脚本无需命令行参数，也不需要交互输入。

## R12

输出解读：

- `MIS size (backtracking)`：回溯算法求得的最优值。
- `MIS vertices`：一个最大独立集实例（不保证唯一）。
- `Search nodes visited`：递归访问状态数。
- `Pruned by bound`：被上界提前裁剪的分支数。
- 若启用 `verify_with_bruteforce=True`，会额外打印基线结果并断言两者大小一致。

## R13

建议最小测试集：

- `C5`（奇环）：`alpha=2`。
- `K3,4`（完全二部图）：`alpha=4`。
- 空图 `n=k`：`alpha=k`。
- 完全图 `n=k`：`alpha=1`。
- 非法输入：`n<=0`、越界顶点、自环。

本 MVP 在 `main()` 中已覆盖前两类并做断言校验。

## R14

可调参数：

- 图规模与结构：`n`、`edges`。
- 随机图参数：`edge_probability`、`seed`。
- 校验开关：`verify_with_bruteforce`（建议仅在小图开启）。
- 分支策略：当前为“候选子图最大度优先”，可替换成其他启发式。

## R15

方法对比：

- 对比暴力枚举：
  - 暴力需要遍历全部 `2^n` 子集；
  - 回溯 + 剪枝可显著减少实际搜索状态。  
- 对比近似/启发式算法：
  - 启发式通常更快但不保最优；
  - 本实现保最优，适合需要精确答案的小中规模图。  
- 对比“转补图求最大团”：
  - 理论等价；
  - 直接在原图上写 MIS 回溯更直观。

## R16

典型应用场景：

- 互斥任务集合最大化（冲突关系建图）。
- 无线网络中非干扰节点/链路选择。
- 编译器寄存器分配中的冲突子问题抽象。
- 图论教学中的 NP-hard 精确搜索示例。

## R17

可扩展方向：

- 更强上界：例如补图着色上界或分支重排策略。
- 记忆化/分解：对重复子问题做缓存（需注意状态压缩成本）。
- 并行搜索：把高层分支分派到多进程。
- 混合方法：先用启发式给出较好下界，再做精确分支限界。

## R18

源码级算法流（对应 `demo.py`，8 步）：

1. `build_adjacency_masks` 把边集编码为位掩码邻接表 `adj`，并完成输入合法性检查。  
2. `maximum_independent_set_backtracking` 初始化全局最优 `best_size/best_mask` 与搜索计数器。  
3. 递归进入 `dfs(candidate_mask, chosen_mask, chosen_size)`。  
4. 先计算上界 `chosen_size + popcount(candidate_mask)`；若不可能超过当前最优，则记一次剪枝并返回。  
5. 若无候选顶点（`candidate_mask=0`），把当前可行独立集与全局最优比较并更新。  
6. 否则 `choose_branch_vertex` 选择候选子图内度最大的顶点 `v` 作为分支点。  
7. 执行“选 `v`”分支：递归到 `candidate \ {v} \ N(v)`，保证可行性；再执行“不选 `v`”分支：递归到 `candidate \ {v}`。  
8. 递归结束后输出 `MISResult`；`run_case` 再用 `is_independent_mask`（及可选 `brute_force_mis`）做结果自检并打印统计。
