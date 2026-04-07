# 编辑距离 (Levenshtein Distance)

- UID: `CS-0039`
- 学科: `计算机`
- 分类: `动态规划`
- 源序号: `56`
- 目标目录: `Algorithms/计算机-动态规划-0056-编辑距离_(Levenshtein_Distance)`

## R01

编辑距离（Levenshtein Distance）衡量两个字符串之间的最小编辑操作数。
允许的基础操作有 3 种，且每种代价为 `1`：
- 插入一个字符
- 删除一个字符
- 替换一个字符

本目录实现该问题的动态规划 MVP，并给出二维 DP、空间优化 DP 与小规模递归对拍。

## R02

MVP 问题定义：
- 输入：两个字符串 `word_a`、`word_b`
- 输出：
  - `distance`：最小编辑距离（非负整数）
  - `normalized_similarity = 1 - distance / max(len(a), len(b))`（当两串都为空时定义为 `1.0`）

示例：
- `kitten -> sitting` 的编辑距离为 `3`
- `"" -> abc` 的编辑距离为 `3`

## R03

状态定义（二维 DP）：
- `dp[i][j]`：`word_a` 前 `i` 个字符变换到 `word_b` 前 `j` 个字符的最小编辑距离。

转移方程：
- 删除：`dp[i-1][j] + 1`
- 插入：`dp[i][j-1] + 1`
- 替换/匹配：`dp[i-1][j-1] + cost`
  - 当 `word_a[i-1] == word_b[j-1]` 时 `cost=0`
  - 否则 `cost=1`

最终答案为 `dp[m][n]`，其中 `m=len(word_a)`，`n=len(word_b)`。

## R04

边界初始化：
- `dp[i][0] = i`：把长度为 `i` 的前缀变为空串，只能连续删除 `i` 次。
- `dp[0][j] = j`：把空串变为长度为 `j` 的前缀，只能连续插入 `j` 次。

该初始化保证了第一行和第一列可作为后续状态转移的基础。

## R05

最优子结构与无后效性：
- 任意最优编辑序列在最后一步前，必定对应一个更短前缀问题的最优解；
- 最后一步只可能是插入、删除、替换/匹配三类之一；
- 因此可通过“子问题最优值 + 最后一步代价”构成整体最优值。

这就是动态规划可用的核心原因。

## R06

复杂度分析：
- 二维 DP：
  - 时间复杂度 `O(mn)`
  - 空间复杂度 `O(mn)`
- 一维滚动数组 DP（`levenshtein_distance_optimized`）：
  - 时间复杂度仍为 `O(mn)`
  - 空间复杂度降为 `O(min(m, n))`

本 MVP 同时实现两者，并在运行时做结果一致性断言。

## R07

边界与异常处理：
- 输入类型不是字符串时抛出 `TypeError`；
- 空字符串合法，支持 `("", "")`、`("", "abc")` 等情形；
- 对非常长字符串，不建议使用递归校验版（仅用于小规模对拍）。

## R08

`demo.py` 模块结构：
- `validate_word`：输入类型校验。
- `levenshtein_distance_dp`：二维 DP 主实现，返回距离和 DP 表。
- `levenshtein_distance_optimized`：一维滚动数组版本。
- `levenshtein_distance_memo`：记忆化递归基线（小规模校验）。
- `normalized_similarity`：归一化相似度。
- `format_dp_table`：将 DP 表打印为可读文本。
- `run_case`：执行单例并做多实现一致性检查。
- `randomized_cross_check`：随机字符串对拍。
- `main`：组织固定样例与汇总输出。

## R09

MVP 取舍说明：
- 仅依赖 `numpy + Python 标准库`，最小化环境负担；
- 没有调用第三方“黑盒编辑距离函数”；
- 通过“三实现交叉验证”保证结果可信；
- 代码重点放在算法透明度，而非工程级 SIMD/并行优化。

## R10

运行方式（无交互输入）：

```bash
uv run python Algorithms/计算机-动态规划-0056-编辑距离_(Levenshtein_Distance)/demo.py
```

如果已在该目录下：

```bash
uv run python demo.py
```

## R11

输出字段说明：
- `distance`：编辑距离整数值。
- `similarity`：归一化相似度，范围 `[0, 1]`。
- `len_a / len_b`：两字符串长度。
- `DP table`：仅在短字符串下打印，便于人工审计。
- `summary`：固定样例汇总统计。
- `global checks pass`：全局断言结果。

## R12

固定测试样例：
1. `kitten -> sitting`，期望 `3`
2. `flaw -> lawn`，期望 `2`
3. `intention -> execution`，期望 `5`
4. `"" -> abc`，期望 `3`
5. `algorithm -> algorithm`，期望 `0`

这些样例覆盖了替换、插入、删除、空串、完全相同字符串等关键场景。

## R13

随机对拍策略：
- 使用种子 `2026`，保证复现；
- 字母表限定为 `abcd`；
- 随机生成长度 `0..8` 的字符串；
- 默认执行 `200` 轮，逐轮断言：
  - 二维 DP 距离 == 一维优化 DP 距离
  - 二维 DP 距离 == 记忆化递归距离

若任一轮不一致，直接抛出 `AssertionError`。

## R14

正确性审计点：
- 二维 DP 的边界行列必须是线性递增；
- 任意状态仅依赖左、上、左上三个已知状态；
- 一维优化版与二维版结果完全一致；
- 小规模上再用记忆化递归做第三方程式校验。

## R15

局限与工程注意：
- 本实现默认三类操作代价均为 `1`，未支持加权编辑距离；
- 未支持 Damerau-Levenshtein（相邻交换）操作；
- 未针对超长文本做带宽剪枝（如 Ukkonen）或并行优化；
- 更偏教学与验证，不是极致性能实现。

## R16

典型应用：
- 拼写纠错与搜索建议；
- OCR/ASR 输出后处理中的字符串匹配；
- 去重与近似匹配（如用户名、地址、商品标题）；
- 生物信息学中的序列近似比对（简化模型）。

## R17

交付核对：
- `README.md`：`R01-R18` 已完整填写；
- `demo.py`：可直接运行，且无交互输入；
- `meta.json`：`UID/学科/分类/源序号/目录路径` 与任务要求一致；
- 目录内实现自包含，可直接用于该算法项验证。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 定义 5 组固定样例，逐个调用 `run_case` 执行。  
2. `run_case` 首先调用 `levenshtein_distance_dp`：创建 `(m+1) x (n+1)` 的 `dp` 表并初始化第一行、第一列。  
3. `levenshtein_distance_dp` 双层循环填表：对每个 `dp[i][j]` 计算删除、插入、替换/匹配三种候选代价并取最小值。  
4. `run_case` 再调用 `levenshtein_distance_optimized`，使用滚动数组重复同一转移逻辑，得到空间优化结果。  
5. 若字符串较短（长度都不超过 10），`run_case` 还调用 `levenshtein_distance_memo`（`lru_cache`）做递归基线计算。  
6. `run_case` 断言三种实现结果一致，并与给定 `expected` 对比；随后计算 `normalized_similarity`。  
7. 对短字符串案例，`run_case` 调用 `format_dp_table` 打印 DP 表，便于人工核查状态转移是否合理。  
8. 全部固定样例完成后，`main` 调用 `randomized_cross_check` 做 200 轮随机对拍，最后汇总并输出全局检查结果。
