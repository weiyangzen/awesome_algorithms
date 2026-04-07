# 后缀树 - Ukkonen

- UID: `MATH-0526`
- 学科: `数学`
- 分类: `字符串算法`
- 源序号: `526`
- 目标目录: `Algorithms/数学-字符串算法-0526-后缀树_-_Ukkonen`

## R01

本条目给出 `Ukkonen` 后缀树构建算法的最小可运行 MVP，实现目标：
- 对输入字符串在线构建压缩后缀树（线性时间思路）；
- 支持子串存在性查询 `contains(pattern)`；
- 从树结构提取后缀数组并与朴素后缀数组对照校验；
- 使用固定样例直接运行，无需交互输入。

## R02

问题定义（MVP 范围）：
- 输入：任意 Python 字符串 `text`（可为空）。
- 处理：在 `text` 尾部追加一个不在原串中的终止符，构建后缀树。
- 输出能力：
  - `contains(pattern)`：判断 `pattern` 是否为 `text` 的子串；
  - `suffix_array()`：返回不含终止符后缀的字典序起点下标；
  - `leaf_count()`：叶子数（应为 `len(text)+1`，包含终止符后缀）。

## R03

Ukkonen 核心思想：
1. 逐字符扩展（phase），每一 phase 处理所有待补齐后缀（extension）。
2. 维护活动点（`active_node`, `active_edge`, `active_length`），避免重复从根扫描。
3. 维护 `remaining_suffix_count` 记录本 phase 尚未显式处理的后缀数。
4. 使用共享可变叶端点 `leaf_end`，让当期所有叶边自动延伸（Trick 3）。
5. 通过后缀链接 `suffix_link` 快速跳转到下一个待处理上下文（Trick 2）。

## R04

MVP 中的构建流程：
1. 选择唯一终止符并拼接为 `text + terminal`。
2. 初始化根节点、活动点、叶端点与计数器。
3. 对每个位置 `pos` 调用 `_extend(pos)`。
4. `_extend` 中若活动边缺失，则直接新增叶子。
5. 若活动边存在，先尝试 `walk_down`（Skip/Count）。
6. 若边上字符匹配当前字符，执行 Rule 3：`active_length += 1` 并结束该 phase 的 while。
7. 若字符不匹配，拆分内部节点并新增叶子，更新后缀链接。
8. 每次显式扩展后减少 `remaining_suffix_count`，并按根/非根规则更新活动点。
9. 全部 phase 结束后 DFS 设置每个叶子的 `suffix_index`。

## R05

核心数据结构（`demo.py`）：
- `End(value)`：叶边共享终点。
- `Node(start, end, suffix_link, children, suffix_index)`：
  - `start/end` 描述从父节点到该节点入边在文本中的区间；
  - `children: Dict[str, int]` 记录出边（按首字符映射到子节点下标）；
  - `suffix_link` 用于 Ukkonen 跳转；
  - `suffix_index` 仅叶子有效。
- `SuffixTree`：封装构建、查询、后缀数组提取与校验辅助。

## R06

正确性要点（简述）：
- 终止符确保每个后缀在树中对应唯一叶子，避免一个后缀成为另一后缀前缀时的歧义。
- 活动点始终指向“下一次扩展从哪里继续”，保证在线构建的线性摊还复杂度。
- `walk_down` 按边长整体跳边，不逐字符重走，保持不变量一致。
- 发生分裂时：
  - 新内部节点承接原边前缀；
  - 原子边起点右移；
  - 新叶子挂在分裂点；
  - 后缀链接把同 phase 的内部节点串联。
- DFS 的 `suffix_index = size - label_height` 给出后缀起点。

## R07

复杂度：
- 构建后缀树：`O(n)` 时间（摊还），`O(n)` 空间。
- `contains(pattern)`：`O(m)`，`m` 为模式串长度。
- `suffix_array()`：`O(n)` 遍历叶子，若对子边字符排序则额外有局部排序成本。

## R08

边界与异常：
- 空串：树中仅有终止符后缀，`leaf_count() == 1`。
- 空模式串：按定义 `contains("") == True`。
- 模式串含任意字符：若路径不存在则返回 `False`。
- 输入中若含常见终止符字符：实现会自动选择其他未出现字符作为终止符。

## R09

MVP 取舍：
- 重点是“可追踪、可验证”的 Ukkonen 实现，不依赖第三方后缀树黑盒包。
- 不实现广义后缀树、多文本拼接、在线删除与动态编辑。
- 通过“后缀数组对照 + 子串朴素对照 + 叶子数断言”做最小可信验证。

## R10

`demo.py` 职责划分：
- `SuffixTree.__init__`：初始化并触发整串构建。
- `_extend`：单 phase 核心扩展逻辑。
- `_walk_down`：按边长跳转活动点。
- `_set_suffix_index_dfs`：构建后设置叶子后缀下标。
- `contains`：子串存在性查询。
- `suffix_array`：从树叶提取后缀数组。
- `run_case`：单样例构建 + 三重断言验证。
- `main`：固定样例批量运行并打印结果。

## R11

运行方式：

```bash
cd Algorithms/数学-字符串算法-0526-后缀树_-_Ukkonen
uv run python demo.py
```

脚本不读取 stdin，直接运行内置样例。

## R12

输出解释：
- `Text`：当前测试字符串。
- `Nodes`：树节点数（根 + 内部节点 + 叶子）。
- `Leaves`：叶子数，理论应为 `len(text)+1`。
- `Suffix Array`：由树提取的后缀数组（不含仅终止符后缀）。
- `Pattern Checks`：若 `pattern:True`，表示 `contains(pattern)` 与 `pattern in text` 一致且为真。

若任一断言失败，程序会抛异常并终止。

## R13

内置最小测试集：
- `banana`：经典重复模式。
- `mississippi`：高重复与多分裂。
- `xabxac`：教材常见示例。
- `aaaaa`：同字符极端重复。
- `""`：空串边界。

每个样例都校验：
- 树后缀数组 == 朴素后缀数组；
- 叶子数正确；
- 所有后缀都可被 `contains` 命中；
- 多个模式串查询与朴素 `in` 一致。

## R14

可调参数与可替换点：
- 终止符候选集合：`_pick_terminal` 可按业务字符集扩展。
- 输出规模：可在 `main()` 增减样例与模式串。
- 排序策略：`suffix_array()` 中目前按出边首字符排序，可替换为其他遍历顺序。
- 输出内容：可增加边区间打印，用于教学可视化调试。

## R15

方法对比：
- 后缀树（本实现）：
  - 优点：子串查询 `O(m)`；一次构建后支持多次查询。
  - 缺点：实现复杂，常数和内存占用较大。
- 后缀数组 + LCP：
  - 优点：结构紧凑，工程实现常更简洁。
  - 缺点：单次子串查询通常 `O(m log n)`（不加额外索引时）。
- 朴素扫描：
  - 优点：实现最简单。
  - 缺点：多查询场景总成本高。

## R16

典型应用：
- 多模式串匹配前的索引构建。
- 重复子串、最长重复片段分析（可在树上扩展统计）。
- 文本挖掘中的子串频次/位置查询底座。
- 生物序列（DNA/RNA）片段检索。

## R17

可扩展方向：
- 增加 `find_occurrences(pattern)` 返回全部匹配位置（对子树叶子收集）。
- 增加 LCP / longest repeated substring 的树上统计函数。
- 扩展为广义后缀树（多字符串，分隔符策略）。
- 增加随机对拍与性能基准（对比朴素搜索与后缀数组）。

## R18

源码级算法流（`demo.py`，9 步）：
1. `main` 准备固定字符串样例与每例模式串集合。  
2. `run_case` 为每个样例实例化 `SuffixTree(text)`，触发 Ukkonen 在线构建。  
3. `SuffixTree.__init__` 先选唯一终止符并初始化根、活动点、共享 `leaf_end`。  
4. 构建循环逐个位置调用 `_extend(pos)`，每次先增加 `remaining_suffix_count`。  
5. `_extend` 在 while 中依据活动点执行三类动作：新增叶子、沿边跳转、边分裂。  
6. 边分裂时创建内部节点 `split`，重挂旧边与新叶，并串接 `suffix_link`。  
7. 每次显式扩展后更新活动点（根规则或走后缀链接），直到当前 phase 收敛。  
8. 构建完成后 `_set_suffix_index_dfs` 为叶子写入后缀起点，`suffix_array()` 由叶子导出字典序下标。  
9. `run_case` 将树结果与朴素后缀数组、`pattern in text`、叶子数量逐项断言，最后打印报告。  
