# 最优二叉搜索树

- UID: `CS-0041`
- 学科: `计算机`
- 分类: `动态规划`
- 源序号: `58`
- 目标目录: `Algorithms/计算机-动态规划-0058-最优二叉搜索树`

## R01

最优二叉搜索树（Optimal Binary Search Tree, OBST）问题：
- 已有有序关键字 `k1..kn`；
- 已知成功查找概率 `p1..pn`（命中关键字）与失败查找概率 `q0..qn`（落在相邻关键字间隙）；
- 目标是在保持中序有序的前提下，构造一棵 BST，使期望查找代价最小。

本目录给出一个可运行 MVP：
- 输出最小期望代价；
- 输出一棵对应的最优树结构；
- 用独立基线（记忆化、暴力）做一致性校验。

## R02

MVP 输入输出约定：
- 输入：
  - `p`：长度 `n` 的非负概率序列（成功查找）；
  - `q`：长度 `n+1` 的非负概率序列（失败查找）；
  - 约束：`sum(p) + sum(q) == 1`（容差 `1e-9`）。
- 输出：
  - `min_expected_cost`：最优期望查找代价；
  - `tree`：一棵最优 BST（通过 `kX(left,right)` 文本展示）；
  - `root_table` / `expected_cost_table` / `weight_table`：DP 过程表。

不在本 MVP 范围内：
- 未排序键值自动排序；
- 浮点概率估计过程；
- 外存/缓存感知型搜索代价模型。

## R03

状态定义（半开区间版）：
- 键索引用 `0..n-1` 表示 `k1..kn`；
- 区间 `[i, j)` 表示键 `ki..k(j-1)`；
- `e[i][j]`：区间 `[i, j)` 的最小期望代价；
- `w[i][j]`：该区间总概率质量；
- `root[i][j]`：使 `e[i][j]` 最小的根键索引。

边界：
- 空区间 `i == j` 对应一个虚拟叶（失败查找），`e[i][i] = q[i]`，`w[i][i] = q[i]`。

## R04

状态转移：

1. 区间权重：
`w[i][j] = w[i][j-1] + p[j-1] + q[j]`

2. 枚举根 `r in [i, j-1]`：
`e[i][j] = min( e[i][r] + e[r+1][j] + w[i][j] )`

解释：
- 左子树成本 `e[i][r]`；
- 右子树成本 `e[r+1][j]`；
- 本层把整段概率整体下压 1 层，增加 `w[i][j]`。

## R05

最优子结构成立：
- 若 `[i, j)` 的最优树根是 `r`，则左右子区间分别必须最优；
- 若某一侧可替换为更优子树，则总成本更低，矛盾。

无后效性成立：
- `e[i][j]` 仅依赖更短区间，不依赖决策历史。

因此可按区间长度从小到大做自底向上 DP。

## R06

最优树重建：
- 先用 `root` 表记录每个区间最优根；
- 从 `[0, n)` 递归建树：
  - `i==j` 返回空节点（失败叶）；
  - 否则取 `r=root[i][j]`，递归左右。

重建校验：
- 再用重建出的树按深度公式独立计算期望代价：
  - 命中键代价 `p[r] * depth`；
  - 失败叶代价 `q[t] * depth`；
- 结果应与 `e[0][n]` 一致。

## R07

复杂度：
- 时间复杂度：`O(n^3)`（区间长度、起点、根枚举三层）；
- 空间复杂度：`O(n^2)`（`e/w/root` 三个二维表）。

基线算法：
- 记忆化递归同样是 `O(n^3)` 时间、`O(n^2)` 状态；
- 无缓存暴力用于小规模校验，复杂度指数级。

## R08

`demo.py` 模块结构：
- `validate_probabilities`：输入概率合法性校验；
- `optimal_bst_dp`：主算法（自底向上 DP）；
- `build_tree` / `tree_to_expression`：从 `root` 重建并展示树；
- `evaluate_expected_cost_from_tree`：重建树的独立代价计算；
- `optimal_bst_top_down_cost`：记忆化基线；
- `optimal_bst_bruteforce_cost`：小规模暴力校验；
- `randomized_cross_check`：随机对拍；
- `main`：固定样例 + 随机校验（无交互）。

## R09

核心接口：
- `optimal_bst_dp(p, q) -> OptimalBSTResult`
  - 返回最优期望代价、三张 DP 表和重建树。
- `optimal_bst_top_down_cost(p, q) -> float`
  - 独立递归基线。
- `optimal_bst_bruteforce_cost(p, q) -> float`
  - 无缓存穷举基线。
- `left_biased_policy_cost(p, q) -> float`
  - 非最优策略（总选最左根）用于对比最优性。

## R10

固定样例覆盖：
1. CLRS 经典分布：`expected_cost = 2.75`（断言）；
2. 成功查找概率偏斜分布：观察最优树偏向高概率键；
3. 近均匀分布：观察结构更平衡的最优树；
4. 随机分布批量对拍：200 轮。

## R11

运行方式（无交互输入）：

```bash
uv run python Algorithms/计算机-动态规划-0058-最优二叉搜索树/demo.py
```

若当前目录已切到该算法目录：

```bash
uv run python demo.py
```

## R12

输出字段解读：
- `optimal`：最优期望代价与树表达式；
- `cross-check.top_down`：记忆化递归结果；
- `cross-check.brute_force`：小规模时暴力结果；
- `cross-check.rebuilt`：由重建树重新积分得到的代价；
- `cross-check.left_biased`：固定劣策略代价（一般不低于最优）。

程序对关键等式做断言，不一致会抛 `AssertionError`。

## R13

随机对拍策略：
- 默认 `trials=200`；
- 每轮随机 `n in [1, 7]`；
- 生成 `2n+1` 个随机正数后归一化为概率；
  - 前 `n` 个作为 `p`；
  - 后 `n+1` 个作为 `q`；
- 断言：
  - `DP == top-down`；
  - `n<=5` 时 `DP == brute-force`；
  - `DP == rebuilt(tree)`。

## R14

边界与异常处理：
- `p/q` 非一维、空、含 `NaN/Inf`：抛 `ValueError`；
- 含负概率：抛 `ValueError`；
- `len(q) != len(p)+1`：抛 `ValueError`；
- 总概率不为 1（容差外）：抛 `ValueError`；
- `root` 表损坏导致重建越界：抛 `RuntimeError`。

## R15

“最小但诚实”的实现选择：
- 工具栈仅使用 `numpy + 标准库`；
- 不依赖任何现成 OBST 黑盒 API；
- 同时提供主算法、记忆化基线、暴力基线和树重建校验；
- 输出不仅给出数值，还给出可解释的树结构，便于验证与教学。

## R16

可扩展方向：
- Knuth 优化（在满足四边形不等式/单调性条件时）把 `O(n^3)` 降到 `O(n^2)`；
- 将代价函数扩展为“比较成本不等权”“磁盘页访问成本”等工程模型；
- 与真实检索日志联动，用频率估计 `p/q` 自动构建离线索引树；
- 输出 Graphviz/JSON 结构以便可视化与服务端加载。

## R17

交付核对：
- `README.md`：`R01-R18` 全部已填写，无占位符；
- `demo.py`：可直接运行、无交互输入、无占位符；
- `meta.json`：UID/学科/分类/源序号/目录字段与任务一致；
- 目录自包含，可直接用于该算法项验证。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 组织 3 个固定分布样例并调用 `run_case`，最后运行 `randomized_cross_check`。  
2. `run_case` 调用 `optimal_bst_dp`：初始化 `e/w/root` 表，空区间基值设为 `q[i]`。  
3. 主循环按区间长度 `1..n` 递增，计算每个 `[i,j)` 的 `w[i][j]`。  
4. 对每个 `[i,j)` 枚举根 `r`，用 `e[i][r] + e[r+1][j] + w[i][j]` 更新最优值并记录 `root[i][j]`。  
5. 填表完成后，`build_tree(root, 0, n)` 递归重建最优 BST，`tree_to_expression` 输出结构字符串。  
6. `run_case` 同时调用 `optimal_bst_top_down_cost`（记忆化）与小规模 `optimal_bst_bruteforce_cost` 做独立基线。  
7. `evaluate_expected_cost_from_tree` 以“键命中 + 失败叶”深度加权重新积分，验证重建树与 DP 数值一致。  
8. 随机对拍阶段重复上述断言，确保在多组概率分布下 `DP/记忆化/暴力/重建` 结果一致。
