# 最短超串问题

- UID: `CS-0046`
- 学科: `计算机`
- 分类: `动态规划`
- 源序号: `63`
- 目标目录: `Algorithms/计算机-动态规划-0063-最短超串问题`

## R01

最短超串问题（Shortest Superstring）要求：
- 给定一组字符串 `words`；
- 构造一个尽可能短的字符串 `S`；
- 使得每个 `words[i]` 都是 `S` 的子串。

这是一个经典 NP-hard 组合优化问题。该条目实现的是工程常用的精确解路线：当 `n`（字符串个数）不大时，使用状态压缩动态规划求全局最优。

## R02

MVP 输入输出定义：
- 输入：`words: list[str]`，非空字符串列表。
- 输出：
  - `superstring`：一个最短超串；
  - `length`：其长度；
  - 同时保留 `normalized_words`（去重/去包含后的等价求解集合）。

本实现约束：
- 列表不能为空；
- 元素必须是非空字符串；
- 无交互输入，`demo.py` 直接运行固定样例与随机回归。

## R03

核心建模（状态压缩 DP）：
- 先计算 `overlap[i][j]`：`words[i]` 后缀与 `words[j]` 前缀的最大重合长度。
- 状态：`dp[mask][j]` 表示“覆盖集合 `mask` 且最后一个单词是 `j`”时的最优超串。
- 转移：
  - 从 `dp[mask][end]` 转移到 `dp[mask | (1<<nxt)][nxt]`；
  - 新串为 `cur + words[nxt][overlap[end][nxt]:]`。
- 比较规则：先比长度，长度相同按字典序最小，保证输出稳定。

## R04

为何该转移是正确的：
- 任意“覆盖 `mask` 且以 `j` 结尾”的候选，最后一步都可视作“从某个 `i` 连接到 `j`”；
- 连接代价只与 `i -> j` 的重合长度有关，满足最优子结构；
- 穷举所有前驱 `i` 可以覆盖所有可能最优方案；
- 对每个状态保留最短（再按字典序打破平局）即可得到全局最优。

## R05

复杂度分析（`n` 为归一化后单词数，`L` 为平均长度）：
- 预处理重合矩阵：`O(n^2 * L^2)`（朴素后缀-前缀匹配）。
- DP 主过程：`O(2^n * n^2 * L)`（字符串拼接比较会带来与长度相关的常数）。
- 额外空间：`O(2^n * n * L)`（保存每个状态对应的最优字符串）。

结论：适合中小规模 `n` 的精确求解。

## R06

`demo.py` 模块划分：
- `validate_words`：输入校验。
- `normalize_words`：去重并移除被其他字符串完全包含的项。
- `build_overlap_matrix`：计算 `overlap`。
- `shortest_superstring_dp`：主算法（返回具体最优超串）。
- `shortest_superstring_length_dp`：独立的长度 DP 校验器。
- `bruteforce_shortest_superstring`：小规模排列暴力真值。
- `run_case`：单例执行、打印、断言。
- `randomized_regression`：随机回归。

## R07

最小工具栈：
- `numpy`：用于重合矩阵和长度 DP 数组。
- Python 标准库：`dataclasses`、`itertools`、`random`。

没有依赖外部黑盒“最短超串求解器”，状态定义、转移、对拍都在源码中可追踪。

## R08

输入与边界处理：
- 非 `list[str]` 输入会抛 `TypeError`。
- 空列表会抛 `ValueError`。
- 空字符串元素不支持（抛 `ValueError`），避免语义歧义。
- 归一化后若只剩一个字符串，答案就是它本身。
- 对重复单词和被包含单词，归一化后自动消除冗余状态。

## R09

归一化策略说明：
- `set` 去重：重复单词不影响最短长度。
- 删除被包含字符串：若 `a in b` 且 `a != b`，保留 `b` 即可，`a` 约束已自动满足。
- 归一化不改变最优目标，但会减少状态数，提升求解速度。

同时在结果中保留 `input_words` 与 `normalized_words`，便于审计“原始约束是否都被覆盖”。

## R10

正确性保障采用三层校验：
- 主解：`shortest_superstring_dp`（返回实际字符串）。
- 独立长度校验：`shortest_superstring_length_dp`（只算最优长度）。
- 小规模精确真值：`bruteforce_shortest_superstring`（全排列）。

在 `run_case` 和随机回归中，断言：
- 主解覆盖全部输入字符串；
- 主解长度等于长度 DP；
- 在小规模下主解字符串与暴力真值完全一致。

## R11

运行方式：

```bash
cd Algorithms/计算机-动态规划-0063-最短超串问题
uv run python demo.py
```

脚本无需交互输入，会打印固定样例结果和随机回归摘要。

## R12

控制台输出字段说明：
- `input_words`：原始输入。
- `normalized_words`：归一化后的求解集合。
- `dp_result`：主 DP 得到的最短超串及长度。
- `length_dp_check`：独立长度 DP 的最优长度。
- `bruteforce_check`：小规模时的暴力最优超串。
- `checks`：
  - `covers_input`：是否覆盖原始输入所有词；
  - `covers_normalized`：是否覆盖归一化词集；
  - `length_match`：主解长度是否等于长度 DP；
  - `bruteforce_match`：是否与暴力真值一致。

## R13

建议测试集：
- 经典案例：`["alex","loves","leetcode"]`。
- 高重合案例：`["catg","ctaagt","gcta","ttca","atgcatc"]`。
- 包含关系：`["abc","bc","c"]`。
- 简单链式拼接：`["ab","bc","cd"]`。
- 平局场景：`["ab","ba"]`（验证字典序 tie-break）。
- 重复输入：`["aba","bab","aba"]`。
- 随机小规模回归：持续和暴力法对拍。

## R14

工程注意点：
- 若不做“长度优先 + 字典序次优”的比较规则，输出可能不稳定。
- 若不移除被包含字符串，状态数会增加，性能明显下降。
- `overlap` 计算方向不能写反（必须后缀对前缀）。
- 暴力法仅用于小规模验证，不可作为主求解。

## R15

与其他方法对比：
- 相比全排列暴力：从 `O(n!)` 降到 `O(2^n n^2)` 量级（忽略字符串常数）。
- 相比纯贪心合并：DP 能保证全局最优，贪心只能给近似结果。
- 相比黑盒库调用：本实现完整暴露了建模、转移和校验路径，更适合教学和审计。

## R16

应用场景：
- 基因片段拼接的简化抽象模型。
- 多日志模板片段合并与压缩展示。
- 文本片段去冗余拼接。
- 状态压缩 DP 教学案例（与 TSP/Held-Karp 思想相通）。

## R17

可扩展方向：
- 用哈希或 KMP/Z 函数优化 `overlap` 计算。
- 仅求长度时保留父指针而非整串，降低内存占用。
- 增加近似算法（贪心、局部搜索）用于更大 `n`。
- 引入基准脚本，绘制 `n` 增长下的时间/内存曲线。

## R18

`demo.py` 源码级流程（8 步）：
1. `main` 先准备固定样例，逐个调用 `run_case`，最后运行随机回归。  
2. `run_case` 调 `solve_shortest_superstring`，内部先执行 `validate_words` 与 `normalize_words`。  
3. `build_overlap_matrix` 计算任意 `i -> j` 的最大后缀-前缀重合长度，作为后续拼接代价基础。  
4. `shortest_superstring_dp` 初始化 `dp[1<<j][j] = words[j]`，表示只选一个词时的最优超串。  
5. 在子集枚举中尝试把 `nxt` 追加到 `end` 后面，构造 `cand = cur + words[nxt][overlap[end][nxt]:]`，并用“长度优先、字典序次优”更新状态。  
6. 遍历满集合 `full_mask` 的所有结尾状态，取全局最优字符串作为主解。  
7. `run_case` 再调用 `shortest_superstring_length_dp` 做独立长度校验；若规模足够小，再调用 `bruteforce_shortest_superstring` 取排列真值。  
8. 所有断言通过后输出结果；固定样例结束后 `randomized_regression` 连续对拍，最终打印 `All shortest-superstring checks passed.`。  
