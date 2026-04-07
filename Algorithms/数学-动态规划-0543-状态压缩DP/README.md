# 状态压缩DP

- UID: `MATH-0543`
- 学科: `数学`
- 分类: `动态规划`
- 源序号: `543`
- 目标目录: `Algorithms/数学-动态规划-0543-状态压缩DP`

## R01

状态压缩 DP（Bitmask DP）是把“一个集合是否已被选过”编码成二进制掩码 `mask`，再在 `mask` 上做动态规划。  
它适用于“元素总数不大（通常 `n <= 20~25`），但需要精确枚举子集状态”的问题。

## R02

本条目采用最经典的应用：旅行商问题（TSP）的 Held-Karp 算法。

问题定义：
- 给定 `n` 个城市和距离矩阵 `dist[i][j]`；
- 从起点 `start` 出发，恰好访问每个城市一次，再回到起点；
- 目标是最小化总路程。

这是 NP-hard 问题；状态压缩 DP 给出指数级但可精确求解的基线方法。

## R03

状态定义（源码一致）：
- `dp[mask][j]`：从 `start` 出发，恰好访问了 `mask` 中所有城市，并且当前停在 `j` 的最小代价。
- `mask` 是位集，`mask` 的第 `k` 位为 1 表示城市 `k` 已访问。

初始状态：
- `dp[1 << start][start] = 0`。

## R04

转移方程：

\[
dp[mask][j] = \min_{i \in mask, i \neq j} \big(dp[mask \setminus \{j\}][i] + dist[i][j]\big)
\]

结束条件：
- 令 `full = (1<<n)-1`；
- 最终答案是
\[
\min_{j \neq start} dp[full][j] + dist[j][start]
\]
即把最后一个城市接回起点。

## R05

为什么叫“状态压缩”：
- 常规“访问集合”可用布尔数组表示，状态维度很高；
- 用一个整数 `mask` 表示集合后，子集关系可通过位运算快速完成：
  - 判断是否访问：`mask & (1<<k)`
  - 删除元素：`mask ^ (1<<k)`（已知该位为 1）
  - 总状态数：`2^n`

这样把“集合 DP”降成可遍历的整数状态空间。

## R06

路径恢复：
- 仅有 `dp` 只能得到最优值，不能直接给出最优路径；
- `demo.py` 同步维护 `parent[mask][j]`，记录到达 `dp[mask][j]` 的最优前驱城市；
- 从最优终点开始反向回溯，得到路径后再拼上起点形成完整回路。

## R07

伪代码：

```text
dp[all_masks][n] = +inf
parent[all_masks][n] = -1
dp[1<<start][start] = 0

for mask in [0 .. 2^n-1]:
    if start not in mask: continue
    for end in nodes_in(mask):
        if end == start and mask != (1<<start): continue
        prev_mask = mask without end
        dp[mask][end] = min_{prev in prev_mask} dp[prev_mask][prev] + dist[prev][end]
        record parent

best = min_{end != start} dp[full][end] + dist[end][start]
reconstruct tour by parent from (full, best_end)
```

## R08

正确性要点（简述）：
1. 最优子结构：若最优路径在状态 `(mask, j)` 的最后一步来自 `i -> j`，则去掉 `j` 后必是状态 `(mask\{j}, i)` 的最优解。
2. 无后效性：`dp[mask][j]` 只依赖更小子集 `mask\{j}`，不依赖到达该子集的具体历史。
3. 边界完备：从 `dp[{start}][start]=0` 出发，所有合法 Hamilton 路径都能通过递推覆盖。
4. 最终再加回 `dist[end][start]`，得到完整回路最优值。

## R09

复杂度：
- 时间复杂度：`O(n^2 * 2^n)`；
- 空间复杂度：`O(n * 2^n)`（`dp` 与 `parent` 同级）。

因此本方法适合中小规模精确求解，不适合大规模 TSP。

## R10

常见错误：
- 忘记强制 `start` 必须在 `mask` 中，导致无效状态参与转移；
- 未处理 `end == start` 的中间状态，可能错误回到起点并破坏“每点一次”；
- 只算最优值不存前驱，后续无法验证路径；
- 路径回溯时 `mask` 更新顺序写错，导致 parent 链断裂。

`demo.py` 对这些点都做了显式分支与断言。

## R11

`demo.py` 模块划分：
- `TSPResult`：封装最优代价与路径。
- `validate_distance_matrix`：输入矩阵合法性检查。
- `held_karp_tsp`：状态压缩 DP 主算法 + 前驱记录 + 回溯。
- `brute_force_tsp`：小规模穷举真值解，用于交叉校验。
- `tour_to_dataframe`：把路径拆成逐边明细表。
- `run_case`：执行样例并比较 DP 与穷举。
- `build_demo_matrix`：构造固定坐标并生成距离矩阵。
- `main`：脚本入口，无交互运行。

## R12

运行方式：

```bash
cd Algorithms/数学-动态规划-0543-状态压缩DP
uv run python demo.py
```

脚本会打印：距离矩阵、DP 最优路径、穷举最优路径、逐步边代价以及一致性检查结果。

## R13

当前内置样例：
- `6` 个城市的二维坐标点；
- 通过欧氏距离构造对称距离矩阵；
- 起点固定为 `0`。

这个规模既能体现状态压缩 DP 过程，也能在演示中用穷举做真值对拍。

## R14

参数与可调项：
- 可在 `build_demo_matrix()` 替换坐标，构造不同图实例；
- 可调整 `start` 起点；
- 若把 `n` 提高太大，`O(n^2 2^n)` 会迅速增大，建议仅用于中小规模。

## R15

方法对比：
- 状态压缩 DP（本实现）：精确解，复杂度指数级但远优于全排列；
- 全排列穷举：`O((n-1)!)`，只适合非常小规模；
- 启发式（遗传、局部搜索等）：可扩展到大规模，但通常不保证全局最优。

在教学和基准验证场景中，Held-Karp 是标准的“可解释精确算法”。

## R16

适用场景（抽象到一般问题）：
- “元素是否已使用”是核心状态；
- 目标函数满足最优子结构；
- 规模中等，需要可验证的最优解。

除 TSP 外，状态压缩 DP 也常见于：最短超串、集合覆盖变体、任务分配等子集优化问题。

## R17

可扩展方向：
- 非对称 TSP（`dist[i][j] != dist[j][i]`）可直接复用当前框架；
- 增加剪枝（下界估计）以减少无效状态；
- 使用 meet-in-the-middle 或并行化降低常数；
- 对更大规模改用近似算法，再与小规模 DP 基线做质量评估。

## R18

`demo.py` 源码级算法流程（9 步）：
1. `main` 调用 `build_demo_matrix` 生成固定 6 城市距离矩阵，并传入 `run_case`。
2. `run_case` 先用 `validate_distance_matrix` 检查矩阵是否方阵、非负、对角为 0、起点合法。
3. `held_karp_tsp` 初始化 `dp[mask][j]=+inf` 与 `parent[mask][j]=-1`，并设置初值 `dp[1<<start][start]=0`。
4. 双层遍历 `mask` 与终点 `end`，仅处理包含起点的合法状态；对每个状态枚举前驱 `prev`。
5. 用转移式 `dp[prev_mask][prev] + dist[prev][end]` 更新 `dp[mask][end]`，并同步写入 `parent`。
6. 在 `full_mask` 上枚举最后城市 `end`，计算 `dp[full][end] + dist[end][start]` 得到全局最优回路代价。
7. 从最优终点沿 `parent` 反向回溯，恢复访问顺序，再补上起点得到完整 tour。
8. `run_case` 再调用 `brute_force_tsp` 穷举所有排列，得到同一实例的真值最优解。
9. 输出路径明细表与对拍结果；若 DP 与穷举代价不一致则抛错，保证实现可验证。
