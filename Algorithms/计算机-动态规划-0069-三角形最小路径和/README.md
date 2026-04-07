# 三角形最小路径和

- UID: `CS-0052`
- 学科: `计算机`
- 分类: `动态规划`
- 源序号: `69`
- 目标目录: `Algorithms/计算机-动态规划-0069-三角形最小路径和`

## R01

问题定义（Triangle Minimum Path Sum）：
- 输入一个三角形数表 `triangle`，第 `r` 行有 `r+1` 个数；
- 从顶点 `(0,0)` 出发，每一步只能移动到下一行的相邻位置：
  - `(r+1, c)` 或 `(r+1, c+1)`；
- 目标是让从顶到底路径上的数字和最小。

本条目除了给出最小路径和，还会重建一条对应路径（索引与数值）。

## R02

MVP 输入输出约定：
- 输入：`triangle: Sequence[Sequence[float]]`。
- 约束：
  - 三角形不能为空；
  - 第 `r` 行长度必须恰好等于 `r+1`；
  - 所有值必须是有限数（拒绝 `NaN/Inf`）。
- 输出：`TriangleMinPathResult`
  - `min_sum`：最小路径和；
  - `path_indices`：路径坐标列表，如 `[(0,0),(1,0),(2,1),...]`；
  - `path_values`：路径经过的值列表。

## R03

状态定义（自底向上 DP）：
- `dp[c]`：在“当前处理层的下一行”中，从列 `c` 出发到底部的最小路径和；
- `choices[r][c]`：当站在 `(r,c)` 时，下一步应走到第 `r+1` 行的哪一列（`c` 或 `c+1`）。

初始化：
- `dp = triangle[last_row]`，即最后一行到底部的最小和就是它本身。

## R04

状态转移方程：

对每个位置 `(r, c)`（从倒数第二行往上）：

`best_child = min(dp[c], dp[c+1])`

`new_dp[c] = triangle[r][c] + best_child`

并记录：
- 若 `dp[c] <= dp[c+1]`，`choices[r][c] = c`；
- 否则 `choices[r][c] = c+1`。

每层计算完成后令 `dp = new_dp`，最终 `dp[0]` 即全局最优值。

## R05

最优子结构说明：
- 任意最优路径在 `(r,c)` 的后续部分，必然是其两个子问题之一的最优解：
  - 从 `(r+1,c)` 到底的最优；
  - 从 `(r+1,c+1)` 到底的最优。
- 若后续不是子问题最优，则可替换为更优后缀并降低总和，与“原路径最优”矛盾。

因此可用局部最优后缀递推到全局最优。

## R06

路径重建方法：
- 已有 `choices` 后，从 `(0,0)` 开始逐行向下：
  - 当前在 `(r,c)`；
  - 下一列 `c = choices[r][c]`；
  - 直到最后一行。
- 重建时同步记录：
  - `path_indices`（坐标）；
  - `path_values`（经过值）。

该路径与 `min_sum` 一一对应，可直接做可解释性校验。

## R07

复杂度分析：
- 自底向上 DP：
  - 时间复杂度 `O(n^2)`（三角形共 `1+2+...+n` 个状态）；
  - 空间复杂度 `O(n^2)`（包含 `choices` 用于重建）+ `O(n)`（滚动 `dp`）。
- 记忆化递归基线：
  - 时间复杂度 `O(n^2)`；
  - 空间复杂度 `O(n^2)`（缓存）+ `O(n)`（递归深度）。
- 暴力 DFS 基线：
  - 时间复杂度指数级 `O(2^n)`，仅用于小规模随机校验。

## R08

`demo.py` 函数结构：
- `to_triangle`：输入校验与标准化；
- `triangle_min_path_bottom_up`：主算法（最小和 + 路径重建）；
- `triangle_min_path_top_down`：记忆化递归基线；
- `triangle_min_path_bruteforce`：暴力 DFS 基线；
- `is_valid_triangle_path`：检查路径索引合法性；
- `run_case`：运行单例并做断言；
- `randomized_cross_check`：随机对拍；
- `main`：组织固定样例与随机回归。

## R09

核心接口语义：
- `triangle_min_path_bottom_up(triangle) -> TriangleMinPathResult`
  - 主接口，返回最小路径和与路径。
- `triangle_min_path_top_down(triangle) -> float`
  - 只返回最小路径和，用于交叉验证。
- `triangle_min_path_bruteforce(triangle) -> float`
  - 小规模精确基线（无缓存），用于防止主实现逻辑偏差。

## R10

固定样例覆盖：
1. 经典正数三角形：期望最小和 `11`；
2. 含负数三角形：验证算法不依赖“非负性”；
3. 单行三角形：验证最小边界输入。

每个样例都会输出路径并执行一致性断言。

## R11

运行方式（无交互输入）：

```bash
uv run python Algorithms/计算机-动态规划-0069-三角形最小路径和/demo.py
```

若当前就在该目录：

```bash
uv run python demo.py
```

## R12

输出字段解读：
- `bottom-up`：主算法结果，含 `min_sum`、`path_indices`、`path_values`；
- `top-down`：记忆化递归结果；
- `bruteforce`：暴力 DFS 结果；
- 三者应当数值一致，否则程序抛出 `AssertionError`。

## R13

随机对拍策略：
- 默认 `300` 轮；
- 行数 `rows` 随机采样于 `[1, 9]`；
- 三角形元素采样于 `[-10, 20]`；
- 每轮比较：
  - 主算法 `min_sum` == 记忆化递归结果；
  - 主算法 `min_sum` == 暴力 DFS 结果；
  - 重建路径满足“逐行、相邻下降”的结构约束。

## R14

异常与边界处理：
- 空三角形：`ValueError`；
- 任一行长度不符合 `r+1`：`ValueError`；
- 存在 `NaN/Inf`：`ValueError`；
- 路径重建后若不满足相邻规则：断言失败。

这些检查使脚本在数据不合规时尽早失败。

## R15

为什么该实现是“最小但诚实”的 MVP：
- 仅依赖 `numpy + 标准库`，没有引入重型框架；
- 不调用第三方黑盒最优化器；
- 主算法、记忆化递归、暴力基线三方对照；
- 可读性优先，函数拆分清晰，便于复查与扩展。

## R16

可扩展方向：
- 支持返回“所有最优路径”而非一条；
- 支持自定义 tie-break（`dp[c] == dp[c+1]` 时的偏好策略）；
- 扩展为“最大路径和”“带禁行节点”“带转移代价”等变体；
- 增加性能基准，对比递归与迭代的常数开销。

## R17

交付核对：
- `README.md`：`R01-R18` 已完整填充；
- `demo.py`：可运行、无交互输入、无占位符残留；
- `meta.json`：UID/学科/分类/源序号/目录信息与任务一致；
- 目录自包含，可用于自动化验证。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 构造 3 个固定样例并调用 `run_case`，最后执行 `randomized_cross_check`。  
2. `run_case` 先调用 `triangle_min_path_bottom_up`，其中 `to_triangle` 会校验每一行长度是否为 `r+1`。  
3. `triangle_min_path_bottom_up` 用最后一行初始化 `dp`，并创建 `choices` 保存每个状态的下一跳列号。  
4. 算法从倒数第二行向上迭代：对每个 `(r,c)` 比较 `dp[c]` 与 `dp[c+1]`，选更小子路径并写入 `new_dp[c] = triangle[r][c] + child_best`。  
5. 完成所有行后，`dp[0]` 即全局最小路径和；随后从 `(0,0)` 按 `choices` 逐行下行，重建 `path_indices` 与 `path_values`。  
6. `run_case` 再调用 `triangle_min_path_top_down`（`lru_cache` 记忆化）独立计算最小路径和。  
7. `run_case` 对小样例调用 `triangle_min_path_bruteforce`（无记忆化 DFS）得到第三个基准值，并校验路径结构合法、路径值求和等于 `min_sum`。  
8. `randomized_cross_check` 用随机三角形重复上述一致性断言 300 次，确认主实现稳定可靠。  
