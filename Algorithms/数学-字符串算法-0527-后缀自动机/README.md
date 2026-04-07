# 后缀自动机

- UID: `MATH-0527`
- 学科: `数学`
- 分类: `字符串算法`
- 源序号: `527`
- 目标目录: `Algorithms/数学-字符串算法-0527-后缀自动机`

## R01

后缀自动机（Suffix Automaton, SAM）是对“一个字符串全部子串”进行压缩表示的有穷自动机。
它在 `O(n)` 时间内构建，常用于：
- 子串是否出现查询；
- 不同子串个数统计；
- 子串出现次数统计（配合 endpos 计数传播）；
- 两串最长公共子串（将一串建 SAM，再在线扫描另一串）。

本目录给出一个最小可运行 MVP：
- 核心构建过程完全手写（含 clone 状态）；
- 不依赖黑盒字符串库；
- 自带暴力校验，保证结果可审计。

## R02

本实现的问题定义：
- 输入：
  - 主串 `text`；
  - 可选查询串列表 `patterns`；
  - 可选另一字符串 `other`（用于 LCS 演示）。
- 输出：
  - SAM 状态数与边数；
  - `distinct_substrings`：`text` 的不同子串数量；
  - 对每个 `pattern` 的：是否存在、出现次数；
  - 与 `other` 的最长公共子串（长度+示例内容）。

`demo.py` 固定内置样例，不读取交互输入。

## R03

关键理论要点：

1. 对长度为 `n` 的字符串，SAM 状态数至多 `2n-1`。  
2. 每个状态 `v` 表示一组 endpos 等价类，并维护：
   - `len[v]`：该类可表示串的最大长度；
   - `link[v]`：后缀链接，指向“去掉一个前缀后”的等价类。  
3. 不同子串数量公式：
   `sum_{v != root} (len[v] - len[link[v]])`。  
4. 若每次插入新字符令新状态 `occ=1`，再按 `len` 降序把 `occ[v]` 累加到 `occ[link[v]]`，
   则任意模式串终止状态的 `occ` 即其在原串中的出现次数（可重叠计数）。

## R04

算法流程（高层）：
1. 初始化根状态：`len=0, link=-1`。  
2. 依次读入字符 `c`，创建新状态 `cur`。  
3. 沿后缀链接回溯，补齐缺失转移 `p --c--> cur`。  
4. 若回溯到 `-1`，令 `link[cur]=root`。  
5. 否则找到已有转移目标 `q`：
   - 若 `len[p]+1 == len[q]`，直接 `link[cur]=q`；
   - 否则创建 `clone`，复制 `q` 的转移并重定向若干边。  
6. 全串插入完后执行 `finalize_occurrences`，传播出现次数。  
7. 基于自动机提供查询：存在性、计数、不同子串数、LCS。

## R05

核心数据结构：
- `State`：
  - `length: int`，最大串长；
  - `link: int`，后缀链接；
  - `next: dict[str, int]`，转移表；
  - `occ: int`，endpos 出现计数。
- `SuffixAutomaton`：
  - `states: list[State]`，状态池；
  - `last: int`，当前整串所在状态。

## R06

正确性要点：
- 构建阶段：
  - 对每个字符扩展都保持 DFA 可达性与最小性约束；
  - clone 仅在“需要分裂等价类”时创建，保证 `len/link` 关系成立。
- 计数阶段：
  - `occ` 先在每个新字符终止状态累加 1；
  - 再按 `length` 降序向 `link` 汇总，保证父状态覆盖全部子状态 endpos。
- 查询阶段：
  - 子串存在性由自动机路径是否可走完决定；
  - 出现次数取终止状态 `occ`；
  - 不同子串公式直接来自 SAM 状态区间贡献。

`demo.py` 使用暴力枚举结果做交叉验证。

## R07

复杂度分析（`n = len(text)`）：
- 构建：`O(n)` 时间，`O(n)` 空间。  
- `finalize_occurrences`：
  - 排序/桶序可做到 `O(n)`，本 MVP 用 Python 排序为 `O(n log n)`；
  - 在小到中等规模演示中可接受。  
- 查询：
  - `contains(pattern)`：`O(|pattern|)`；
  - `count_occurrences(pattern)`：`O(|pattern|)`；
  - 与另一串 LCS：`O(|other|)`。

## R08

边界与异常处理：
- 空串：
  - 状态仅根节点；
  - 不同子串数为 0；
  - 任意非空模式不存在。  
- 模式串为空：
  - `contains("")` 视为 `True`；
  - `count_occurrences("")` 返回 0（避免定义歧义）。
- 输入含非字符串元素：通过类型注解约束，MVP 不做复杂运行时反射。
- 若调用 `count_occurrences` 前忘记 `finalize_occurrences`，类内部会自动补做一次。

## R09

MVP 取舍：
- 只实现最核心 SAM 操作，不做多字符集压缩与持久化优化。  
- `next` 使用 `dict`，优先可读性而非极致常数。  
- 不引入后缀数组/后缀树联合实现，避免框架膨胀。  
- 暴力校验只在小样例上运行，用于正确性背书。

## R10

`demo.py` 函数职责：
- `State`：定义 SAM 状态结构。  
- `SuffixAutomaton.extend`：插入一个字符并维护 clone 逻辑。  
- `SuffixAutomaton.build`：从整串批量构建。  
- `SuffixAutomaton.finalize_occurrences`：传播 `occ`。  
- `SuffixAutomaton.contains`：子串存在性查询。  
- `SuffixAutomaton.count_occurrences`：子串出现次数查询。  
- `SuffixAutomaton.count_distinct_substrings`：不同子串计数。  
- `SuffixAutomaton.longest_common_substring`：与另一串求 LCS。  
- `brute_*`：暴力对照实现。  
- `run_case/main`：组织样例、输出结果并断言校验。

## R11

运行方式：

```bash
cd Algorithms/数学-字符串算法-0527-后缀自动机
uv run python demo.py
```

脚本不会请求用户输入。

## R12

输出字段说明：
- `text`：主串。  
- `states / edges`：自动机规模。  
- `distinct_substrings`：SAM 计数及对应暴力计数。  
- `contains(pattern)`：模式串是否为主串子串。  
- `occurrences(pattern)`：模式串出现次数（可重叠）。  
- `LCS with ...`：和另一串的最长公共子串长度与样例子串。  
- 末尾 `All checks passed.` 表示所有断言验证通过。

## R13

内置最小测试集：
- `text="ababa"`：重复结构明显，适合验证 clone 与重叠计数。  
- `text="banana"`：经典字符串，子串统计与出现次数都较有代表性。

每个样例都验证：
- 不同子串数（SAM vs 暴力）；
- 多个模式串的存在性与出现次数（SAM vs 暴力）；
- LCS 长度（SAM vs DP 暴力）。

## R14

可调参数（当前通过 `main` 中样例配置给出）：
- `text`：构建 SAM 的主串。  
- `patterns`：待查询模式列表。  
- `other`：用于 LCS 的对比串。  
- 若要做性能实验，可把 `text` 换成更长随机串。

## R15

与相近方法对比：
- 后缀数组：
  - 适合离线排序与区间查询；
  - 单次在线子串插入不如 SAM 自然。  
- 后缀树：
  - 也可线性构建，功能强；
  - 实现复杂度通常高于 SAM。  
- KMP：
  - 强于单模式匹配；
  - 不直接表达“所有子串”的整体结构。

## R16

典型应用：
- 搜索系统中的字典串统计特征；
- 生物序列分析中的重复片段统计；
- 文本挖掘中的子串频率分析；
- 作为更复杂字符串算法（如多串广义处理）的基础模块。

## R17

可扩展方向：
- 使用计数排序替代 `sorted`，将 `finalize_occurrences` 降为严格线性。  
- 扩展为广义 SAM（多串）并记录来源集合。  
- 支持按状态输出 endpos 区间统计。  
- 增加随机压力测试与 benchmark 输出。  
- 将 `dict` 转移改为紧凑数组（已知字符集时）以优化常数。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 准备固定测试样例（`text/patterns/other`），逐个调用 `run_case`。  
2. `run_case` 创建 `SuffixAutomaton`，执行 `build(text)`，依次调用 `extend` 完成在线构建。  
3. `extend` 中先创建新状态 `cur`，沿 `link` 回溯补齐缺失转移。  
4. 若遇到冲突状态 `q` 且 `len[p]+1 != len[q]`，创建 `clone` 复制 `q.next` 与 `q.link`，再回溯重定向所有指向 `q` 的相关边。  
5. 构建完成后执行 `finalize_occurrences`，按状态长度降序把 `occ[v]` 累加到 `occ[link[v]]`。  
6. `count_distinct_substrings` 按 `sum(len[v]-len[link[v]])` 计算不同子串个数。  
7. 对每个模式串，`contains` 走转移判存在，`count_occurrences` 找终止状态后读取 `occ`。  
8. `longest_common_substring` 在线扫描 `other`，用“当前状态+匹配长度”与后缀链接回退机制维护最优答案，并与暴力结果比对断言。
