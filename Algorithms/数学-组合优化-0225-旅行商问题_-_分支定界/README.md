# 旅行商问题 - 分支定界

- UID: `MATH-0225`
- 学科: `数学`
- 分类: `组合优化`
- 源序号: `225`
- 目标目录: `Algorithms/数学-组合优化-0225-旅行商问题_-_分支定界`

## R01

问题定义（对称 TSP）：
- 给定 `n` 个城市及两两距离 `d(i, j)`，寻找一条哈密顿回路。
- 回路要求每个城市恰好访问一次，并返回起点。
- 目标是最小化总路长。

数学形式：
- 决策变量：访问顺序 `pi = (pi_0, pi_1, ..., pi_{n-1})`
- 目标函数：`min sum_{k=0..n-1} d(pi_k, pi_{k+1})`，其中 `pi_n = pi_0`
- 约束：`pi` 是城市集合的一个排列。

## R02

为什么用分支定界（Branch and Bound）：
- TSP 是 NP-hard，纯暴力 `O(n!)` 很快不可用。
- 分支定界在搜索排列树时，利用“下界估计 + 当前最优上界”提前砍掉不可能更优的分支。
- 在中小规模（尤其几何距离、结构较好）场景中，能明显减少枚举数量。

## R03

本实现的状态表示：
- `path`：当前已构建路径（起点固定为 0）。
- `visited`：布尔数组，标记哪些城市已访问。
- `level`：当前路径深度（已放入多少城市）。
- `curr_weight`：当前路径的真实累计代价。
- `curr_bound`：从当前状态出发的最小可能增量的下界估计。

## R04

下界函数（first_min / second_min）：
- 对每个城市 `i`，定义：
  - `first_min(i)`：从 `i` 出发的最小边权
  - `second_min(i)`：从 `i` 出发的次小边权
- 初始下界：
  - `LB0 = 0.5 * sum_i (first_min(i) + second_min(i))`
- 向下扩展一个决策边 `(u -> v)` 时，按层级修正下界：
  - 第一层：减去 `(first_min(u) + first_min(v)) / 2`
  - 其余层：减去 `(second_min(u) + first_min(v)) / 2`

该下界是可行的乐观估计（不高估最优剩余代价），因此可用于安全剪枝。

## R05

上界（当前最好解）初始化：
- `best_cost = +inf`。
- 每当搜索到完整回路时，用其真实代价更新 `best_cost` 和 `best_path`。
- 实战中可先用贪心解做初始上界以增强剪枝；本 MVP 保持最小实现，直接由首次可行回路建立上界。

## R06

剪枝规则：
- 对候选扩展后的状态，计算
  - `estimate = next_weight + next_bound`
- 若 `estimate >= best_cost`，该分支不可能优于当前最优解，立即剪枝。
- 否则继续递归扩展。

## R07

核心伪代码：

```text
init best_cost = +inf
compute first_min, second_min for every city
init lower_bound LB0
dfs(path=[start], visited={start}, curr_weight=0, curr_bound=LB0)

dfs(state):
  if all cities visited:
    try close tour to start
    update best_cost
    return

  for each unvisited city v:
    next_weight = curr_weight + dist(last, v)
    next_bound = update_bound(curr_bound, last, v, level)
    if next_weight + next_bound < best_cost:
      recurse
    else:
      prune
```

## R08

正确性要点（简述）：
- 穷举完整搜索树时覆盖所有哈密顿回路（起点固定只消除循环等价）。
- 使用的下界不高估任何可行完成路径的最小附加代价。
- 因此若 `curr_weight + curr_bound >= best_cost`，该分支不可能产生更优解，剪枝不影响最优性。
- 最终保留的 `best_cost` 即全局最优。

## R09

复杂度：
- 最坏时间复杂度仍为指数级（接近 `O(n!)`）。
- 空间复杂度主要来自递归栈和状态数组，约 `O(n)`（不含距离矩阵）。
- 分支定界的收益体现在平均情况：取决于下界质量和上界收敛速度。

## R10

数值与实现细节：
- `demo.py` 使用 `numpy` 构造欧氏距离矩阵，主对角线设为 `inf`（禁止自环）。
- 下界与总代价采用浮点计算，比较时直接用 `<`，演示中附加暴力校验确认正确性。
- 若输入图存在不可达边（`inf`），算法会自然跳过不可行扩展。

## R11

MVP 功能范围：
- 提供 `TSPBranchAndBound` 类，支持给定距离矩阵求解最优环。
- 提供 `build_euclidean_distance_matrix(points)` 从二维点集构图。
- 提供 `brute_force_tsp` 仅用于小规模 correctness 对照。
- 提供搜索统计：`expanded_nodes`、`pruned_nodes`、耗时。

## R12

运行方式：

```bash
cd Algorithms/数学-组合优化-0225-旅行商问题_-_分支定界
python3 demo.py
```

脚本无需交互输入，会自动运行两个案例。

## R13

输出解释：
- `B&B 最优路长`：分支定界求得的最小总路长。
- `B&B 路径`：最优访问顺序（包含回到起点）。
- `expanded/pruned`：搜索中被展开/被剪枝的候选节点数。
- 在 Case A 中还会打印 `Bruteforce 最优路长` 作为交叉验证。

## R14

边界情况：
- `n=1`：路径为 `[start, start]`，代价 0。
- 非方阵输入会报错。
- 若某城市有限出边不足 2 条，下界构造失败并报错（该图不满足标准 TSP 前提）。

## R15

可扩展方向：
- 初始上界增强：Nearest Neighbor / 2-opt 先给较优可行解。
- 更强下界：1-tree / Held-Karp bound（可进一步提升剪枝率）。
- 搜索顺序启发：优先扩展更短边或更有潜力节点。
- 并行化：对高层分支并行 DFS/BFS。

## R16

测试建议：
- 功能正确性：随机小规模 `n<=9` 与暴力解对照。
- 稳定性：非欧氏对称矩阵、存在 `inf` 边的图。
- 性能观察：记录不同 `n` 下 expanded/pruned 比例与运行时间。

## R17

与常见方案对比：
- 暴力枚举：最简单但几乎不可扩展。
- 动态规划（Held-Karp）：`O(n^2 2^n)`，适合中等规模且需精确解。
- 分支定界：仍是精确算法，但可借结构性剪枝，常在实际中优于纯暴力。
- 启发式（2-opt/遗传/蚁群）：可处理更大规模但不保证全局最优。

## R18

`demo.py` 源码级算法流程（8 步）：
1. 读取/构造距离矩阵：`build_euclidean_distance_matrix` 用 `numpy` 计算两两欧氏距离并置对角为 `inf`。
2. 初始化求解器：`TSPBranchAndBound.__init__` 校验矩阵形状，预计算每个城市的 `first_min` 与 `second_min`。
3. 计算初始下界：`solve()` 中按 `0.5 * sum(first+second)` 得到根节点下界 `initial_bound`。
4. 建立根状态：路径固定从 `start=0` 开始，`visited[start]=True`，当前代价为 0。
5. 深度优先分支：`_dfs()` 在当前城市上枚举所有未访问 `nxt`，形成候选扩展边 `(prev -> nxt)`。
6. 增量更新界：根据是否第一层，使用对应公式更新 `next_bound`，并计算 `estimate = next_weight + next_bound`。
7. 剪枝或递归：若 `estimate < best_cost` 则继续深入；否则计入 `pruned_nodes` 并跳过该分支。
8. 叶子回路收束：当 `level == n` 时尝试回到起点，若总成本更小则更新 `best_cost / best_path`，最终返回全局最优。
