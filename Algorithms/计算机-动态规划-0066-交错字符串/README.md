# 交错字符串

- UID: `CS-0049`
- 学科: `计算机`
- 分类: `动态规划`
- 源序号: `66`
- 目标目录: `Algorithms/计算机-动态规划-0066-交错字符串`

## R01

交错字符串（Interleaving String）问题：
- 给定字符串 `s1`、`s2`、`s3`；
- 判断 `s3` 是否可以由 `s1` 与 `s2` 交错组成；
- “交错”要求保持 `s1`、`s2` 各自字符的相对顺序，但两者可交替取字符。

例如：
- `s1="abc"`、`s2="def"`、`s3="adbcef"` 是合法交错；
- `s3="abdecf"` 也合法；
- `s3="abdfec"` 不合法（`s2` 中 `e`、`f` 的相对顺序被破坏）。

## R02

MVP 输入输出定义：
- 输入：`s1: str`、`s2: str`、`s3: str`。
- 输出：
  - `is_interleaving`：是否为合法交错；
  - `source_path`：当合法时，长度为 `len(s3)` 的来源路径（字符 `'1'` 表示取自 `s1`，`'2'` 表示取自 `s2`）；
  - `reconstructed`：根据 `source_path` 重建得到的字符串，应与 `s3` 完全一致。

必要条件：
- `len(s1) + len(s2) == len(s3)`，否则必定不可能。

## R03

状态定义（二维 DP）：
- 令 `m = len(s1)`，`n = len(s2)`；
- `dp[i][j]` 表示：`s1` 前 `i` 个字符和 `s2` 前 `j` 个字符，是否能交错组成 `s3` 前 `i+j` 个字符。

边界：
- `dp[0][0] = True`；
- 第一列 `dp[i][0]` 仅由 `s1` 前缀与 `s3` 前缀比较得到；
- 第一行 `dp[0][j]` 仅由 `s2` 前缀与 `s3` 前缀比较得到。

## R04

状态转移：

对任意 `i, j`（不同时为 0），若以下任一条件成立，则 `dp[i][j] = True`：

1. 来自 `s1`：
   - `i > 0`
   - `dp[i-1][j] == True`
   - `s1[i-1] == s3[i+j-1]`
2. 来自 `s2`：
   - `j > 0`
   - `dp[i][j-1] == True`
   - `s2[j-1] == s3[i+j-1]`

终值为 `dp[m][n]`。

## R05

最优子结构/可分解性说明：
- 位置 `(i,j)` 的可行性只依赖更小子问题 `(i-1,j)` 或 `(i,j-1)`；
- 若 `s3` 的第 `i+j` 个字符最终取自 `s1`，前缀必须由 `(i-1,j)` 合法生成；
- 若取自 `s2`，前缀必须由 `(i,j-1)` 合法生成。

因此该问题天然满足动态规划条件，可通过自底向上填表求解。

## R06

路径重建策略：
- 额外维护 `parent[i][j]`：
  - `1` 表示该格最优/可行来源选了 `s1`；
  - `2` 表示来源选了 `s2`；
  - `0` 表示起点 `(0,0)`；
- 当两种来源都可行时，固定优先选择 `s1`（保证输出稳定可复现）。

从 `(m,n)` 逆向沿 `parent` 回溯到 `(0,0)`，反转后得到 `source_path`。

## R07

复杂度：
- 时间复杂度：`O(m*n)`，每个状态只做常数次比较；
- 空间复杂度：`O(m*n)`，用于保存 `dp` 与 `parent`；
- 额外重建路径开销：`O(m+n)`。

## R08

`demo.py` 结构：
- `InterleaveResult`：封装求解结果；
- `validate_text`：输入类型校验；
- `build_interleaving_from_path`：按路径重建字符串；
- `interleave_dp_with_path`：主算法（二维 DP + parent 回溯）；
- `interleave_memoized`：记忆化 DFS 基线校验；
- `interleave_bruteforce`：小规模朴素 DFS 校验；
- `run_case`：执行单个样例并断言一致性；
- `randomized_regression`：随机回归测试；
- `main`：组织固定样例并运行全部验证。

## R09

核心接口：
- `interleave_dp_with_path(s1, s2, s3) -> InterleaveResult`
  - 返回 `is_interleaving/source_path/reconstructed`。
- `interleave_memoized(s1, s2, s3) -> bool`
  - 记忆化 DFS 判定，仅输出可行性。
- `interleave_bruteforce(s1, s2, s3, max_total=18) -> bool`
  - 朴素 DFS 精确判定（仅小规模使用）。
- `build_interleaving_from_path(s1, s2, path) -> str`
  - 对可行路径做可解释验证。

## R10

固定样例覆盖：
1. 经典正例：`"aabcc" + "dbbca" -> "aadbbcbcac"`；
2. 经典反例：`"aabcc" + "dbbca" -> "aadbbbaccc"`；
3. 全空串边界：`"" + "" -> ""`；
4. 长度不匹配：`"" + "" -> "a"`；
5. 仅 `s1` 非空、仅 `s2` 非空场景；
6. 含重复字符的歧义路径场景（验证稳定 tie-break）。

## R11

运行方式（无交互输入）：

```bash
uv run python Algorithms/计算机-动态规划-0066-交错字符串/demo.py
```

若已在目录下：

```bash
uv run python demo.py
```

## R12

输出解读：
- `dp_result`：主算法输出；
- `memoized_check`：记忆化 DFS 结果；
- `bruteforce_check`：小规模朴素 DFS 结果；
- `checks`：
  - `path_valid`：路径能否被正确解析；
  - `reconstructed_match`：重建串是否等于 `s3`；
  - `memoized_match/bruteforce_match`：与基线判定是否一致。

若任一关键一致性失败，脚本会抛出 `AssertionError`。

## R13

随机回归策略：
- 默认 `240` 轮；
- 随机生成 `s1/s2`（长度 `0..6`，字母表 `"abc"`）；
- 约一半样本构造“保证可行”的 `s3`（按随机合并路径生成）；
- 另一半直接随机生成同长度字符串（可能可行也可能不可行）；
- 对每轮样本断言：
  - DP 判定 == 记忆化 DFS 判定 == 朴素 DFS 判定；
  - 若可行，则 `source_path` 可被重建且结果等于 `s3`。

## R14

边界与异常处理：
- 非字符串输入：抛 `TypeError`；
- 长度不匹配：直接返回不可行（不抛异常）；
- 路径重建阶段若出现非法父指针：抛 `RuntimeError`（理论上不应发生）；
- `build_interleaving_from_path` 对非法路径字符或越界使用抛 `ValueError`。

## R15

MVP 的“最小但诚实”体现：
- 仅依赖 `numpy + 标准库`，运行成本低；
- 主算法不是调用第三方黑盒，而是显式给出状态、转移、回溯；
- 使用两种独立基线（记忆化与朴素 DFS）做交叉验证；
- 输出包含可解释路径，不仅给布尔值。

## R16

可扩展方向：
- 空间优化到一维 DP（仅判断可行性时可用）；
- 输出“所有”可行交错路径（需要去重与剪枝）；
- 支持更大字符集与性能基准；
- 推广到 `k` 个字符串交错（高维状态，复杂度显著增长）。

## R17

交付核对：
- `README.md`：`R01-R18` 已完整填写；
- `demo.py`：可直接运行、无交互输入、无占位符；
- `meta.json`：UID/学科/分类/源序号/目录信息与任务一致；
- 本目录自包含，可用于该算法项自动化验证。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 先执行固定样例，再运行随机回归。  
2. `run_case` 调用 `interleave_dp_with_path`，先做长度检查并初始化 `dp` 与 `parent`。  
3. `interleave_dp_with_path` 在二维网格上按 `i=0..m, j=0..n` 填表：比较 `s1/s2` 当前字符与 `s3[i+j-1]`，判定从左或从上是否可达。  
4. 当某格可达时，记录 `parent[i][j]`（若双可达则按固定规则优先 `s1`），保证后续路径重建稳定。  
5. 若终点 `dp[m][n]` 为真，则从 `(m,n)` 逆向按 `parent` 回溯得到 `source_path`，再用 `build_interleaving_from_path` 重建字符串。  
6. `run_case` 再调用 `interleave_memoized`（`lru_cache` DFS）做独立判定交叉检查。  
7. 对总长度较小的样例，`run_case` 调用 `interleave_bruteforce`（朴素 DFS）做精确第三方验证。  
8. `randomized_regression` 大量重复上述一致性断言，确保实现在随机数据上稳定正确。  
