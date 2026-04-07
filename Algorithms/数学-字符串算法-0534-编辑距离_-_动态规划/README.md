# 编辑距离 - 动态规划

- UID: `MATH-0534`
- 学科: `数学`
- 分类: `字符串算法`
- 源序号: `534`
- 目标目录: `Algorithms/数学-字符串算法-0534-编辑距离_-_动态规划`

## R01

编辑距离（Levenshtein distance）用于度量两个字符串之间的差异：
把 `source` 变成 `target` 所需的最少编辑次数。
允许的基本操作是：插入、删除、替换（每次代价均为 1）。
本条目给出一个可运行、可对拍、可回放编辑路径的动态规划 MVP。

## R02

问题定义：
1. 输入两个字符串 `source` 与 `target`。
2. 操作集合：
   - `insert`：插入一个字符。
   - `delete`：删除一个字符。
   - `replace`：将一个字符替换为另一个字符。
3. 输出最小总代价（即最小编辑步数）。

当两个字符相同且选择“匹配前进”时，代价为 0。

## R03

动态规划状态与转移：
1. `dp[i][j]`：`source` 前 `i` 个字符转成 `target` 前 `j` 个字符的最小编辑距离。
2. 边界：
   - `dp[i][0] = i`（只能连续删除）
   - `dp[0][j] = j`（只能连续插入）
3. 转移：
   - 删除：`dp[i-1][j] + 1`
   - 插入：`dp[i][j-1] + 1`
   - 匹配或替换：`dp[i-1][j-1] + cost`
     其中 `cost = 0`（字符相同）或 `1`（字符不同）
4. 取三者最小值：`dp[i][j] = min(...)`。

## R04

`demo.py` 的主算法流程：
1. 在 `build_distance_table` 中建立 `(n+1) x (m+1)` 的 DP 表。
2. 初始化第一行和第一列，编码空串边界。
3. 双层循环填表，按 R03 转移方程取最小代价。
4. 得到最终距离 `dp[n][m]`。
5. 在 `backtrack_operations` 中从 `dp[n][m]` 回溯一条最优编辑路径。
6. 在 `apply_operations` 中把该路径应用到 `source`，验证可还原出 `target`。

## R05

初始化设计要点：
1. `dp[0][0] = 0`：空串到空串无需操作。
2. 第一列递增：`dp[i][0]=i`，表示删掉 `source` 的前 `i` 个字符。
3. 第一行递增：`dp[0][j]=j`，表示向空串插入 `target` 的前 `j` 个字符。

该初始化保证所有后续状态都能由左、上、左上三个方向合法转移。

## R06

最优编辑路径回溯策略：
1. 从右下角 `(n,m)` 反向走到 `(0,0)`。
2. 优先尝试左上角（匹配/替换），若满足代价方程则选择该步。
3. 否则尝试向上（删除），再尝试向左（插入）。
4. 反向收集操作后再 `reverse`，得到正向执行序列。

本实现输出 `EditOperation(action, src_char, dst_char)`，并提供可读的 `operation_trace`。

## R07

正确性要点（简述）：
1. 最优子结构：`dp[i][j]` 的最优解一定来自三个更小子问题之一。
2. 无后效性：状态只依赖 `(i,j)`，不依赖构造该状态的历史路径。
3. 数学归纳：
   - 基础情形由 R05 的边界保证正确。
   - 假设规模更小状态都正确，则 `dp[i][j]` 取三种合法操作后的最小值，故正确。
4. 回溯严格遵守转移等式，因此回放路径总代价等于 `dp[n][m]`。

## R08

复杂度分析（`n=len(source)`, `m=len(target)`）：
1. 时间复杂度：`O(n*m)`（每个单元常数时间计算一次）。
2. 空间复杂度：`O(n*m)`（完整 DP 表，便于后续回溯路径）。
3. 回溯复杂度：`O(n+m)`。

若只要距离值、不要路径，可降为滚动数组 `O(min(n,m))` 空间。

## R09

与近似问题/算法对比：
1. LCS（最长公共子序列）关注“最长保留结构”，编辑距离关注“最小编辑代价”。
2. Hamming 距离仅适用于等长字符串且只允许替换。
3. Damerau-Levenshtein 会额外允许“相邻字符交换”，本题不包含该操作。

在拼写纠错、模糊检索中，Levenshtein 是最常见基础度量之一。

## R10

边界与样例覆盖（`demo.py` 已内置）：
1. `("", "")`：距离应为 `0`。
2. `("abc", "")` 与 `("", "abc")`：纯删除/纯插入场景。
3. `("kitten", "sitting")`：经典案例，距离为 `3`。
4. Unicode 场景如 `("星期三", "星期四")`、`("算法", "算术")`。
5. 重复字符场景 `("aaaa", "aaab")`。

## R11

`demo.py` 的核心接口：
1. `build_distance_table(source, target)`：构建 DP 表并返回统计信息。
2. `backtrack_operations(source, target, dp)`：回溯最优编辑路径。
3. `apply_operations(source, operations)`：执行路径并生成转换结果。
4. `brute_force_distance(source, target)`：递归+记忆化真值函数。
5. `validate_case(source, target)`：单样例对拍与一致性校验。
6. `run_demo_samples()`：运行固定样例并打印结果。

## R12

运行方式（无需交互输入）：

```bash
uv run python demo.py
```

运行时会：
1. 逐个执行内置样例。
2. 打印距离、回放结果、编辑路径和 DP 统计。
3. 若任一样例失败则抛异常；全部通过会输出最终成功提示。

## R13

输出字段说明：
1. `source` / `target`：输入字符串。
2. `distance`：DP 求得的编辑距离。
3. `expected`：`brute_force_distance` 的对拍值。
4. `transformed`：按回溯路径把 `source` 转换后的结果。
5. `edit_steps`：非 `match` 操作数，应等于 `distance`。
6. `rows/cols/cells_computed`：DP 表规模与计算单元统计。
7. `operation_trace`：压缩后的编辑操作轨迹。

## R14

工程化建议：
1. 大规模文本场景可先做长度阈值剪枝，再进入 DP。
2. 仅做相似度阈值判断时可用 banded DP（带宽限制）降低计算量。
3. 批量匹配可缓存常见词的 DP 前缀结果，减少重复计算。
4. 若只需距离值，可切换滚动数组版本降低内存占用。

## R15

常见错误：
1. 把 `dp[i][j]` 错写成从 `source[i]`、`target[j]` 开始，导致下标混乱。
2. 漏掉第一行/第一列初始化，导致边界样例错误。
3. 回溯时未验证转移等式，可能得到非最优路径。
4. 把“匹配”也算作一次编辑，导致操作数与距离不一致。

## R16

本 MVP 验证策略：
1. 每个样例都进行三重一致性检查：
   - DP 距离 `distance`
   - 递归真值 `expected`
   - 回放结果 `transformed`
2. 断言 `distance == expected`。
3. 断言 `transformed == target`。
4. 断言 `edit_steps == distance`，保证回溯路径代价正确。

## R17

可扩展方向：
1. 加权编辑距离：为插入/删除/替换赋予不同权重。
2. Damerau 扩展：支持相邻字符交换操作。
3. 阈值早停：当最小可能代价已超过阈值时提前终止。
4. 面向检索系统：与倒排索引、BK-tree 组合做候选召回。

## R18

`demo.py` 源码级算法流（8 步）：
1. `validate_case` 调用 `build_distance_table`，创建 `dp` 并初始化第一行/列。
2. `build_distance_table` 双层循环填表，对每个 `(i,j)` 计算删除、插入、替换/匹配三种候选代价并取最小。
3. 从 `dp[len(source)][len(target)]` 读取最小编辑距离 `distance`。
4. `backtrack_operations` 从右下角逆推路径：优先走左上（匹配/替换），否则走上（删除）或左（插入）。
5. 回溯完成后反转操作序列，得到正向 `operations`。
6. `apply_operations` 逐步回放 `operations`，把 `source` 转换成 `transformed`。
7. 同时调用 `brute_force_distance`（递归+记忆化）得到独立真值 `expected`。
8. `validate_case` 对 `distance/expected/transformed/edit_steps` 做断言，`run_demo_samples` 执行多组样例并输出汇总成功信息。
