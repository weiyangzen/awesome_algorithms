# 旅行商问题 - Held-Karp

- UID: `MATH-0224`
- 学科: `数学`
- 分类: `组合优化`
- 源序号: `224`
- 目标目录: `Algorithms/数学-组合优化-0224-旅行商问题_-_Held-Karp`

## R01

本条目实现旅行商问题（TSP）的 `Held-Karp` 动态规划最小可运行版本（MVP），目标是：
- 在给定距离矩阵下求出**精确最优回路**（非近似）；
- 返回最优代价与完整巡回路径（起点到起点）；
- 覆盖对称/非对称 TSP；
- 提供一个小规模暴力枚举校验，验证 DP 结果正确性。

## R02

问题定义（MVP 范围）：
- 输入：
  - `dist`：`n x n` 距离矩阵，`dist[i][j]` 表示城市 `i -> j` 成本；
  - `start`：固定起点（默认 `0`）。
- 输出：
  - `best_cost`：从 `start` 出发访问每个城市恰好一次并回到 `start` 的最小代价；
  - `tour`：如 `[0, 2, 4, 1, 3, 0]` 的闭环路径；
  - `states_explored`：DP 中实际被扩展的状态数，用于观测计算规模。

约束说明：
- 本实现允许非对称矩阵；
- 允许出现 `+inf`（表示不可达边）；
- 若不存在哈密顿回路，返回 `tour=None` 且 `best_cost=inf`。

## R03

Held-Karp 状态与转移：

1. 令 `S` 为已访问城市集合，要求 `start ∈ S`。  
2. 定义 `dp[S][j]`：从 `start` 出发，恰好访问 `S` 中所有点并以 `j` 结尾的最小代价。  
3. 初始条件：`dp[{start}][start] = 0`。  
4. 转移：
   `dp[S ∪ {k}][k] = min_j ( dp[S][j] + dist[j][k] )`，其中 `k ∉ S`。  
5. 终止：对全体城市集合 `V`，最优巡回值为
   `min_j ( dp[V][j] + dist[j][start] )`。

集合 `S` 在代码中用位掩码 `mask` 表示，因此可用数组实现 `O(n^2 2^n)` 的精确求解。

## R04

算法流程（MVP）：
1. 校验距离矩阵为方阵、规模 `n>=2`、主对角有限；对角统一设为 `0`。
2. 初始化 `dp[1<<start][start]=0`，其余为 `inf`。
3. 枚举所有包含 `start` 的 `mask`。
4. 枚举当前终点 `last`，若 `dp[mask][last]` 有限，则尝试扩展到未访问城市 `nxt`。
5. 用 `cand = dp[mask][last] + dist[last][nxt]` 松弛 `dp[new_mask][nxt]`。
6. 同步记录 `parent[new_mask][nxt] = last`，用于后续回溯路径。
7. 枚举终点 `end`，计算 `dp[full_mask][end] + dist[end][start]`，取最小值。
8. 从 `best_end` 通过 `parent` 回溯，恢复完整环路并补上终点 `start`。

## R05

核心数据结构：
- `numpy.ndarray[float] dist`：输入距离矩阵；
- `numpy.ndarray[float] dp`：形状 `(2^n, n)` 的最优值表；
- `numpy.ndarray[int] parent`：与 `dp` 同形状，记录前驱城市；
- `HeldKarpResult(dataclass)`：
  - `best_cost`、`tour`、`start`、`states_explored`。

## R06

正确性要点：
- 最优子结构：任意最优路径的前缀仍对应某个子问题最优解；
- 无后效性：`dp[S][j]` 只依赖更小集合的状态，与访问顺序细节无关；
- 转移穷举了“下一步去哪一个未访问城市”的全部可能，因此不会漏解；
- 终止时加上回到起点边，正好满足 TSP 闭环定义。

## R07

复杂度分析：
- 时间复杂度：`O(n^2 * 2^n)`；
- 空间复杂度：`O(n * 2^n)`。

适用范围：
- 城市数中小规模（例如 `n<=20` 左右，取决于机器内存/时延预算）；
- 需要精确最优解而非近似解的任务。

## R08

边界与异常处理：
- 非方阵输入：抛出 `ValueError`；
- `n < 2`：抛出 `ValueError`；
- 对角线存在非有限值：抛出 `ValueError`；
- 存在 `NaN`：抛出 `ValueError`；
- `start` 越界：抛出 `ValueError`；
- 若图结构导致无可行哈密顿回路：返回 `best_cost=inf, tour=None`。

## R09

MVP 取舍：
- 仅依赖 `numpy`，不调用 OR-Tools、Concorde 等黑盒求解器；
- 保留 `parent` 以支持路径重建，而不是只输出最优值；
- 增加 `brute_force_tsp` 仅用于小规模交叉验证，便于确认 DP 实现正确；
- 暂不实现剪枝、并行化或内存压缩版本，优先保证可读性与可验证性。

## R10

`demo.py` 模块职责：
- `validate_distance_matrix`：输入合法性检查与规范化；
- `held_karp_tsp`：Held-Karp 主算法；
- `tour_cost`：对路径计算总代价；
- `brute_force_tsp`：小规模精确枚举校验器；
- `pairwise_euclidean`：由 2D 坐标构造欧氏距离矩阵；
- `format_matrix`：矩阵友好打印；
- `run_case_euclidean` / `run_case_asymmetric`：两个固定案例演示；
- `main`：无交互入口。

## R11

运行方式：

```bash
cd Algorithms/数学-组合优化-0224-旅行商问题_-_Held-Karp
python3 demo.py
```

脚本无需传参，会自动运行两个案例并打印结果。

## R12

输出解读：
- `Held-Karp best cost`：DP 求得最优巡回代价；
- `Held-Karp tour`：对应最优回路；
- `DP states explored`：实际扩展的 DP 状态数量；
- `Brute-force best cost`：暴力基线（仅小规模）；
- `Cost check passed`：Held-Karp 与暴力最优值是否一致。

若 `tour=None` 且 `best_cost=inf`，表示输入图在当前约束下无可行巡回。

## R13

建议最小测试集：
- 对称欧氏 TSP：验证常见场景最优性；
- 非对称矩阵 TSP：验证算法不依赖对称性；
- 含不可达边（`inf`）场景：验证无解处理逻辑；
- 非法输入：非方阵、`NaN`、`start` 越界等。

## R14

可调参数：
- `start`：起点城市编号；
- 距离矩阵构造方式（坐标欧氏距离、业务代价矩阵等）；
- 示例规模 `n`（越大越能体现 `2^n` 增长）；
- 是否启用 `brute_force_tsp` 校验（建议只在小规模启用）。

## R15

方法对比：
- 对比暴力枚举：
  - 暴力为 `O(n!)`；Held-Karp 为 `O(n^2 2^n)`，在中等规模显著更优；
- 对比最近邻等启发式：
  - 启发式快但不保证全局最优；Held-Karp 给出精确最优；
- 对比分支定界：
  - 分支定界实际速度常更快但依赖剪枝质量；Held-Karp 上界更稳定、实现更直接。

## R16

典型应用场景：
- 小中规模配送路径优化（需最优而非近似）；
- 芯片钻孔、机械臂巡检等路径顺序优化；
- 作为更复杂 VRP/路径规划问题中的精确子模块；
- 教学场景下展示“位压缩 + 动态规划”经典范式。

## R17

可扩展方向：
- 内存优化为分层哈希表或滚动数组；
- 加入下界估计做剪枝（向 branch-and-bound 融合）；
- 支持时间窗、禁忌边、多起终点等约束；
- 用 `numba`/并行策略加速状态转移；
- 输出多条近最优路径用于鲁棒决策。

## R18

源码级算法流（对应 `demo.py`，9 步）：
1. `run_case_*` 先构造距离矩阵（欧氏或手工非对称矩阵）。  
2. `held_karp_tsp` 调 `validate_distance_matrix` 做形状、数值、起点合法性检查。  
3. 初始化 `dp` 为 `inf`、`parent` 为 `-1`，设置基态 `dp[start_mask][start]=0`。  
4. 外层遍历所有包含 `start` 的 `mask`，内层遍历可达终点 `last`。  
5. 对每个未访问城市 `nxt` 计算候选代价 `cand`，做最小化松弛更新。  
6. 一旦 `cand` 更优，同步写入 `parent[new_mask][nxt]=last`。  
7. 全部状态结束后，在 `full_mask` 层枚举终点 `end`，加上回边 `end->start` 得到全局最优值。  
8. 从 `best_end` 借助 `parent` 逆向回溯到 `start`，反转后再补 `start` 形成闭环 `tour`。  
9. 演示代码用 `brute_force_tsp` 进行小规模对拍，输出 `Cost check passed` 验证实现正确性。  
