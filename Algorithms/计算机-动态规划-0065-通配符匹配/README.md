# 通配符匹配

- UID: `CS-0048`
- 学科: `计算机`
- 分类: `动态规划`
- 源序号: `65`
- 目标目录: `Algorithms/计算机-动态规划-0065-通配符匹配`

## R01

通配符匹配（Wildcard Matching）问题定义：
- 给定文本串 `s` 和模式串 `p`；
- 模式中普通字符需与文本同字符匹配；
- `?` 匹配任意单个字符；
- `*` 匹配任意长度字符串（可为空串）；
- 目标是判断 `p` 是否能完整匹配 `s`（不是子串匹配，而是全匹配）。

## R02

MVP 输入输出约定：
- 输入：
  - `text: str`，被匹配文本；
  - `pattern: str`，含普通字符 / `?` / `*` 的模式串。
- 输出：
  - `matched: bool`，是否全匹配。

约束：
- `text` 与 `pattern` 必须是字符串，否则抛 `ValueError`；
- `demo.py` 无交互输入，直接执行固定样例和随机对拍。

## R03

符号语义与预处理：
- 普通字符：仅能匹配同字符；
- `?`：匹配任意一个字符；
- `*`：匹配任意长度字符序列（`""`、`"a"`、`"abc"` 均可）。

为了减少冗余状态，先做模式归一化：
- 将连续多个 `*` 压缩为一个 `*`（例如 `ab**c***d -> ab*c*d`）；
- 不改变语义，但可减少 DP 列数并提升性能。

## R04

二维 DP 状态定义：
- 设 `m=len(text)`，`n=len(pattern)`；
- `dp[i][j]` 表示 `text` 前 `i` 个字符（`text[:i]`）与 `pattern` 前 `j` 个字符（`pattern[:j]`）是否匹配。

边界：
- `dp[0][0] = True`；
- `dp[i][0] = False (i>0)`，非空文本不能被空模式匹配；
- `dp[0][j]` 仅在模式前缀全是 `*` 时为 `True`。

## R05

状态转移方程：

当 `pattern[j-1] == '*'` 时：
- `dp[i][j] = dp[i][j-1] or dp[i-1][j]`
- 含义：`*` 匹配空串（左侧）或多匹配一个字符（上侧）。

当 `pattern[j-1] == '?'` 或 `pattern[j-1] == text[i-1]` 时：
- `dp[i][j] = dp[i-1][j-1]`

否则：
- `dp[i][j] = False`

## R06

一维滚动数组优化：
- 观察到 `dp[i][j]` 只依赖当前行左侧 `dp[i][j-1]`、上一行同列 `dp[i-1][j]`、上一行左上 `dp[i-1][j-1]`；
- 可将二维压缩成一维 `row[j]`，并用变量 `prev_diag` 保存“旧左上角”；
- 空间复杂度从 `O(mn)` 降为 `O(n)`。

本目录把一维 DP 作为主接口实现，同时保留二维 DP 作为教学和对照基线。

## R07

复杂度分析：
- 二维 DP：
  - 时间复杂度 `O(mn)`；
  - 空间复杂度 `O(mn)`。
- 一维滚动 DP：
  - 时间复杂度 `O(mn)`；
  - 空间复杂度 `O(n)`。
- 贪心回溯基线（双指针 + 最近 `*` 回退）：
  - 均摊常见表现接近线性；
  - 最坏情况下可视为 `O(mn)` 级别行为。

## R08

`demo.py` 结构：
- `validate_text_and_pattern`：输入类型校验；
- `normalize_pattern`：压缩连续 `*`；
- `wildcard_match_dp_2d`：二维 DP 基线；
- `wildcard_match_dp_1d`：一维滚动 DP（主解）；
- `wildcard_match_greedy`：双指针回溯基线；
- `run_case`：单样例运行与一致性断言；
- `randomized_cross_check`：随机文本与模式对拍；
- `main`：固定样例 + 随机回归入口。

## R09

核心接口：
- `wildcard_match_dp_1d(text, pattern) -> bool`
  - 推荐主接口，空间开销更小。
- `wildcard_match_dp_2d(text, pattern) -> bool`
  - 全状态表版本，便于理解与调试。
- `wildcard_match_greedy(text, pattern) -> bool`
  - 独立思路基线，用于交叉验证。
- `solve_wildcard_match(text, pattern) -> WildcardMatchResult`
  - 统一 API，返回归一化模式与匹配结论。

## R10

固定样例覆盖：
1. `text="", pattern=""` -> `True`；
2. `text="", pattern="*"` -> `True`；
3. `text="", pattern="?"` -> `False`；
4. `text="aa", pattern="a"` -> `False`；
5. `text="aa", pattern="*"` -> `True`；
6. `text="cb", pattern="?a"` -> `False`；
7. `text="adceb", pattern="*a*b"` -> `True`；
8. `text="acdcb", pattern="a*c?b"` -> `False`；
9. `text="abcde", pattern="a*?e"` -> `True`；
10. 含冗余星号样例 `"ab**cd?i*de"`，验证归一化逻辑不改变语义。

## R11

运行方式（无交互输入）：

```bash
uv run python Algorithms/计算机-动态规划-0065-通配符匹配/demo.py
```

若当前已在该目录下：

```bash
uv run python demo.py
```

## R12

输出解读：
- 每个固定 case 会打印：
  - `text`、`pattern`、`normalized`；
  - `dp_2d`、`dp_1d`、`greedy`、`api` 四路结果；
  - `expected`（若提供）。
- 若四路结果不一致，程序抛 `AssertionError`；
- 固定样例结束后打印随机回归通过摘要（轮数、长度上限、随机种子）。

## R13

随机回归策略：
- 默认 `300` 轮；
- 文本长度随机采样 `0..12`；
- 模式长度随机采样 `0..12`；
- 文本字母表默认 `"abc"`；
- 模式 token 从 `{a,b,c,?,*}` 随机生成。

每轮断言：
- `wildcard_match_dp_2d == wildcard_match_dp_1d == wildcard_match_greedy`。

这样可以覆盖：
- 空串边界；
- 多 `*` 场景；
- `?` 与普通字符混合场景；
- 容易写错的初始化与状态更新细节。

## R14

边界与异常处理：
- `text` 或 `pattern` 非字符串：抛 `ValueError`；
- 空文本/空模式是合法输入，按 DP 基础状态处理；
- 超长字符串会提高 `O(mn)` DP 开销，但不会出现整数溢出问题（仅布尔状态）。

## R15

“最小但诚实”的 MVP 取舍：
- 依赖仅 `numpy + 标准库`，轻量可复现；
- 核心匹配逻辑完全手写，不调用黑盒匹配库；
- 同时提供 DP 与贪心两套独立逻辑互证；
- 代码规模小，便于快速审阅与测试。

## R16

可扩展方向：
- 支持转义语义（例如 `\*` 表示字面星号）；
- 扩展为字符类（如 `[a-z]`）或简化正则子集；
- 针对超长串引入分块 / SIMD / 位并行优化；
- 提供批量匹配接口（一个模式匹配多文本）。

## R17

交付核对：
- `README.md` 已填写 `R01-R18`，无占位符；
- `demo.py` 可直接运行，且无交互输入；
- `meta.json` 保持与任务元数据一致（UID/学科/分类/源序号/目录）；
- 目录内容自包含，可直接用于验证该算法条目。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 先构造 10 个固定样例，调用 `run_batch -> run_case` 逐个执行。  
2. `run_case` 分别调用 `wildcard_match_dp_2d`、`wildcard_match_dp_1d`、`wildcard_match_greedy` 与统一 API，收集四路结果。  
3. 每个实现都会先走 `validate_text_and_pattern` 校验输入类型，再走 `normalize_pattern` 把连续 `*` 压缩，减少后续状态规模。  
4. `wildcard_match_dp_2d` 创建 `(m+1) x (n+1)` 布尔表，初始化 `dp[0][0]` 与首行可由 `*` 延展的状态。  
5. 二维 DP 双层循环填表：遇到 `*` 用 `dp[i][j-1] or dp[i-1][j]`，遇到 `?`/同字符用 `dp[i-1][j-1]`。  
6. `wildcard_match_dp_1d` 用一维数组 `row` 滚动更新，并通过 `prev_diag` 保存更新前左上角值，实现与二维同逻辑但 `O(n)` 空间。  
7. `wildcard_match_greedy` 使用双指针记录最近 `*` 位置并在失配时回退，形成独立于 DP 的验证基线。  
8. `run_case` 断言四路结果一致并校验期望值；全部固定样例通过后，`randomized_cross_check` 再做 300 轮随机对拍确保实现稳定。  
