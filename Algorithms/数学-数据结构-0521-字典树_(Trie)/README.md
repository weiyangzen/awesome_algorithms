# 字典树 (Trie)

- UID: `MATH-0521`
- 学科: `数学`
- 分类: `数据结构`
- 源序号: `521`
- 目标目录: `Algorithms/数学-数据结构-0521-字典树_(Trie)`

## R01

字典树（Trie，前缀树）是一种按“字符路径”组织字符串集合的树结构。  
它将公共前缀合并存储，使前缀查询不再依赖全量字符串比较。

本条目提供一个最小可运行 MVP：
- 支持插入、精确查找、前缀判断；
- 支持计数（单词出现次数、前缀开头单词数）；
- 支持删除一个单词实例与按前缀枚举；
- 脚本内置样例，无需交互输入，可直接运行验证。

## R02

问题定义（本目录实现）：
- 输入：字符串序列（可重复、支持 Unicode）、查询词、前缀。
- 核心操作：
  - `insert(word)`：插入一个单词；
  - `search(word)`：是否存在完整单词；
  - `starts_with(prefix)`：是否存在该前缀；
  - `count_words_equal_to(word)`：该词出现次数；
  - `count_words_starting_with(prefix)`：此前缀下单词总数；
  - `erase(word)`：删除一个该词实例；
  - `list_words_with_prefix(prefix)`：按前缀枚举（带上限）。
- 输出：布尔结果、计数结果、前缀枚举结果与断言通过信息。

## R03

数学/结构直觉：

1. 每条“根到节点”的路径可视作一个前缀。  
2. 字符串 `s = c1 c2 ... ck` 会沿 `c1->c2->...->ck` 走到终点。  
3. 在终点记录 `end_count` 表示该完整词出现了多少次。  
4. 在路径节点记录 `pass_count` 表示有多少词经过该节点。  
5. 因此前缀计数可在 `O(|prefix|)` 时间完成，无需扫描全部词。

## R04

算法流程（高层）：
1. 建立根节点 `root`。  
2. 插入时逐字符下探，不存在子节点则创建。  
3. 路径上节点 `pass_count += 1`，末节点 `end_count += 1`。  
4. 精确查询时按字符下探并检查末节点 `end_count > 0`。  
5. 前缀查询时只需判断前缀路径是否存在。  
6. 删除时先验证词存在，再反向回收无用分支。  
7. 前缀枚举时从前缀节点 DFS 收集词。

## R05

核心数据结构：
- `TrieNode`：
  - `children: dict[str, TrieNode]`，字符到子节点映射；
  - `pass_count: int`，经过该节点的单词数；
  - `end_count: int`，在该节点结束的单词数。
- `Trie`：
  - `root: TrieNode`；
  - 提供插入、查询、计数、删除、枚举操作。

`demo.py` 使用 `dataclass` 表达节点结构，代码短且可读。

## R06

正确性要点：
- 插入正确性：每次插入沿词路径递增 `pass_count`，末节点递增 `end_count`，保证频次可还原。  
- 查询正确性：`search` 必须落在末节点且 `end_count > 0`，避免把前缀误判为完整词。  
- 前缀计数正确性：前缀节点 `pass_count` 等于该前缀覆盖的词总数（含重复）。  
- 删除正确性：仅当词存在时执行减计数，并只剪枝“无子节点且计数为 0”的节点，不影响其他词。

## R07

复杂度分析（设字符串长度为 `L`，字母表大小为 `Sigma`）：
- `insert(word)`：时间 `O(L)`，额外空间最坏 `O(L)`（新建分支时）。
- `search(word)`：时间 `O(L)`，空间 `O(1)`。
- `starts_with(prefix)`：时间 `O(|prefix|)`，空间 `O(1)`。
- `count_words_equal_to(word)`：时间 `O(L)`。
- `count_words_starting_with(prefix)`：时间 `O(|prefix|)`。
- `erase(word)`：时间 `O(L)`，空间 `O(L)`（记录路径用于回收）。
- `list_words_with_prefix(prefix)`：
  - 先定位前缀 `O(|prefix|)`；
  - DFS 与输出规模相关，记为 `O(|prefix| + K)`（`K` 为遍历输出成本）。

## R08

边界与异常处理：
- 非字符串输入：抛 `TypeError`。
- 查询不存在词：`search=False`，计数为 `0`。
- 删除不存在词：`erase=False`，结构不变。 
- 重复插入：通过 `end_count` 支持多重计数。 
- 空前缀 `""`：视为匹配根，可统计全部词总数。 
- 多语言字符（例如中文）：按 Python `str` 逐字符处理，不需要额外编码逻辑。

## R09

MVP 取舍说明：
- 采用纯 Python 标准库实现，不依赖第三方库黑盒。  
- 重点覆盖 Trie 的“核心可验证功能”，不扩展到压缩 Trie（Radix Tree）或 AC 自动机。  
- 使用计数字段提升可观测性，便于验证重复词与删除行为。  
- 在 `demo.py` 中添加断言，保证输出不仅可看，还可自动自检。

## R10

`demo.py` 函数职责：
- `Trie.insert`：插入词并更新路径计数。  
- `Trie._find_node`：返回某字符串路径终点节点（内部复用）。  
- `Trie.search`：完整词判断。  
- `Trie.starts_with`：前缀存在判断。  
- `Trie.count_words_equal_to`：完整词频次统计。  
- `Trie.count_words_starting_with`：前缀词数统计。  
- `Trie.erase`：删除一个词实例并可选剪枝。  
- `Trie.list_words_with_prefix`：按前缀列出词（限制数量）。  
- `run_demo/main`：执行固定样例、打印结果、做断言。

## R11

运行方式：

```bash
cd Algorithms/数学-数据结构-0521-字典树_(Trie)
uv run python demo.py
```

脚本不读取交互输入；运行结束应看到 `All assertions passed.`。

## R12

输出字段说明：
- `search('x')`：词 `x` 是否作为完整词存在。  
- `starts_with('p')`：是否存在以 `p` 开头的词。  
- `count_words_equal_to('x')`：词 `x` 的出现次数（支持重复插入）。  
- `count_words_starting_with('p')`：以 `p` 为前缀的总词数。  
- `Words with prefix ...`：枚举出的匹配词（含重复词条）。  
- `erase` 三次结果：展示“存在两次、第三次删除失败”的预期行为。

## R13

最小测试集（已内置）：
- 英文前缀族：`cat/car/cart`。  
- 共享前缀另一组：`do/dog/dove`。  
- 中文词条：`中文/中秋/中关村`。  
- 重复词：`car` 插入两次，验证多重计数与删除逻辑。

建议补充测试：
- 空字符串作为单词插入与查询；
- 超长字符串（性能/稳定性）；
- 随机字符串批量插入与计数一致性检查。

## R14

可调参数：
- `list_words_with_prefix(prefix, limit=20)` 的 `limit` 可控输出规模。  
- 演示词集合 `words` 可替换成业务词典。  
- 若需要区分大小写规则，可在 `insert/search` 前统一做 `lower()`（本 MVP 不强制）。

## R15

与替代方案对比：
- 对比哈希表（`set/dict`）：
  - 精确查找都接近 `O(L)`（含哈希成本）；
  - 哈希表不擅长“前缀查询”，Trie 更自然。  
- 对比排序数组 + 二分：
  - 二分可做前缀范围检索，但实现复杂度更高；
  - Trie 在动态插入/删除场景更直接。  
- 对比压缩 Trie：
  - 压缩 Trie 节点更省空间；
  - 本实现更易读、适合教学和最小验证。

## R16

典型应用场景：
- 输入法联想、搜索建议词前缀匹配；
- 路由/配置键前缀组织；
- 敏感词前缀过滤前置索引；
- 词典自动补全与拼写辅助。

## R17

可扩展方向：
- 压缩边（Radix/Patricia）减少节点数量；
- 增加 `top-k` 频次联想（节点维护热度统计）；
- 支持序列化/反序列化（词典持久化）；
- 多线程读写安全（读写锁或无锁快照）；
- 与 AC 自动机结合做多模式串匹配。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main()` 调用 `run_demo()`，创建一个空 `Trie`。  
2. `run_demo()` 将固定词表逐个调用 `insert`，建立前缀树与计数。  
3. `insert` 从 `root` 出发，逐字符走 `children`，缺失节点即创建，并维护 `pass_count/end_count`。  
4. 查询阶段分别调用 `search`、`starts_with`、计数函数；这些函数都依赖 `_find_node` 完成路径定位。  
5. `_find_node` 按字符迭代下探，若某字符不存在则立即返回 `None`，存在则持续推进到终点。  
6. 删除阶段 `erase('car')` 先检查词是否存在，再递减终点 `end_count` 与路径 `pass_count`。  
7. `erase` 最后逆序回溯路径，对“计数为 0 且无子节点”的分支执行剪枝，保持结构紧凑。  
8. `run_demo()` 打印关键结果并用 `assert` 校验预期，最终输出 `All assertions passed.` 作为成功标志。
