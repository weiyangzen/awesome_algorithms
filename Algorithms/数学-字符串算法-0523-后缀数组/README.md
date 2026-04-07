# 后缀数组

- UID: `MATH-0523`
- 学科: `数学`
- 分类: `字符串算法`
- 源序号: `523`
- 目标目录: `Algorithms/数学-字符串算法-0523-后缀数组`

## R01

后缀数组（Suffix Array, SA）是把字符串所有后缀按字典序排序后，记录其起始下标的数组。

本条目给出一个最小可运行 MVP：
- 用倍增法手写构建后缀数组；
- 用 Kasai 算法手写构建 LCP（最长公共前缀）数组；
- 用 SA 上的二分搜索实现子串匹配；
- 用朴素方法对照校验结果正确性。

## R02

问题定义（本目录实现）：
- 输入：
  - 文本串 `text`（非空字符串）；
  - 查询串 `pattern`（非空字符串）。
- 输出：
  - `sa`：长度为 `n` 的数组，`sa[i]` 为第 `i` 小后缀的起点；
  - `lcp`：长度为 `n` 的数组，`lcp[i]` 为 `suffix(sa[i])` 与 `suffix(sa[i-1])` 的最长公共前缀长度，且 `lcp[0]=0`；
  - `occurrences`：`pattern` 在 `text` 中出现的所有起始下标。

`demo.py` 内置固定样例并直接运行，不依赖交互输入。

## R03

数学与算法基础：

1. 后缀定义：`suffix(i) = text[i:]`，`i in [0, n-1]`。
2. 后缀数组定义：`text[sa[0]:] < text[sa[1]:] < ... < text[sa[n-1]:]`（字典序）。
3. 倍增法思想：按长度 `2^k` 的前缀排名迭代，利用二元组 `(rank[i], rank[i+2^k])` 排序得到新排名。
4. LCP 可用于减少重复比较，Kasai 算法利用相邻后缀关系在线性时间计算全部 LCP。
5. 子串匹配可转为“在有序后缀集合中找边界区间”的二分搜索。

## R04

算法流程（高层）：
1. 校验输入字符串合法性（类型、非空）。
2. 用倍增法构建后缀数组 `sa`。
3. 基于 `sa` 构建逆映射 `rank`。
4. 用 Kasai 算法计算 `lcp`。
5. 对每个查询串，利用二分搜索定位第一个 `>= pattern` 的后缀位置。
6. 再次二分定位第一个 `> pattern` 的后缀位置。
7. 取两者区间内候选并做 `startswith` 过滤。
8. 输出 SA/LCP/匹配位置并与朴素解对照断言。

## R05

核心数据结构：
- `sa: list[int]`：后缀数组。
- `rank: list[int]`：`rank[pos]` 表示后缀 `text[pos:]` 在 SA 中的名次（Kasai 使用）。
- `lcp: list[int]`：相邻后缀 LCP 数组。
- `Case`（`dataclass`）：固定测试样例，字段为 `text` 与 `patterns`。
- `patterns: tuple[str, ...]`：每个样例要查询的多个子串。

## R06

正确性要点：
- 倍增法每轮基于长度 `2^k` 排名构造长度 `2^(k+1)` 排名，排名关系单调细化，直到所有后缀排名唯一。
- `sa` 是 `0..n-1` 的排列，且对应后缀按字典序递增。
- Kasai 算法在相邻名次间复用已知前缀长度，得到与定义一致的 LCP。
- 查询阶段通过二分得到字典序边界，再用 `startswith` 做精确筛选，保证匹配结果正确。
- `demo.py` 用朴素 SA、朴素 LCP、朴素匹配三重对照，若有不一致会直接抛出断言。

## R07

复杂度分析（`n = len(text)`, `m = len(pattern)`）：
- 构建 SA（当前排序实现）：
  - 每轮排序 `O(n log n)`；
  - 轮数 `O(log n)`；
  - 总计 `O(n log^2 n)`。
- 构建 LCP（Kasai）：`O(n)`。
- 单次查询：二分 `O(log n)` 轮，每轮比较切片最多 `O(m)`，总计约 `O(m log n)`。
- 空间复杂度：`O(n)`（`sa/rank/lcp` 等数组）。

## R08

边界与异常处理：
- `text` 不是字符串：抛 `TypeError`。
- `text` 为空：抛 `ValueError`。
- `pattern` 非字符串或为空：抛异常，避免语义歧义。
- `sa` 长度不等于 `len(text)`：抛 `ValueError`。
- `sa` 不是 `0..n-1` 的排列：在 LCP 构建时抛 `ValueError`。

## R09

MVP 取舍说明：
- 采用纯 Python 标准库实现，减少依赖，便于直接运行和阅读。
- 选择倍增法而非更复杂的 SA-IS/DC3，实现更短、教学更直接。
- 查询实现选择“二分 + 精确过滤”，保证正确且结构清晰。
- 不引入大型框架，不做工程化封装，聚焦算法核心链路。

## R10

`demo.py` 主要函数职责：
- `validate_text` / `validate_pattern`：输入合法性检查。
- `build_suffix_array`：倍增法构建 SA。
- `build_lcp_array`：Kasai 算法构建 LCP。
- `find_occurrences`：基于 SA 的二分子串查找。
- `naive_suffix_array` / `naive_lcp_array` / `naive_occurrences`：朴素对照实现。
- `run_case`：执行单个样例并做一致性断言。
- `main`：组织样例并打印结果。

## R11

运行方式：

```bash
cd Algorithms/数学-字符串算法-0523-后缀数组
python3 demo.py
```

在仓库根目录也可运行：

```bash
uv run python Algorithms/数学-字符串算法-0523-后缀数组/demo.py
```

脚本不会读取命令行参数，也不会请求用户输入。

## R12

输出字段说明：
- `text`：当前测试字符串。
- `length`：字符串长度。
- `suffix array`：后缀数组。
- `lcp array`：相邻后缀 LCP。
- `suffix table (rank, idx, suffix)`：逐行展示排序后的后缀。
- `pattern query results`：每个查询串对应的出现位置列表。
- `All cases passed.`：全部样例和断言通过。

## R13

最小测试集（已内置）：
- `banana`：经典示例，含重叠匹配（如 `ana`）。
- `mississippi`：重复子串密集，适合检验 SA/LCP 与边界查询。
- `abracadabra`：含多次重复前缀，便于验证多模式查询。

建议补充测试：
- 单字符重复串（如 `aaaaaa`）；
- 全异字符（如 `abcdefg`）；
- 更长随机串做性能观察。

## R14

可调参数与实现可变点：
- 在 `main` 的 `cases` 中可增减文本或查询串。
- 如需返回 SA 区间顺序，可在 `find_occurrences` 中去掉 `result.sort()`。
- 如需支持空模式（匹配全部位置），可单独定义返回约定并在 `validate_pattern` 放宽限制。
- 如需更高性能，可把倍增排序替换为计数排序/基数排序版本。

## R15

方法对比：
- 对比朴素后缀排序：
  - 朴素方法直观但通常更慢（大量长切片比较）；
  - 倍增法通过排名压缩比较成本，扩展性更好。
- 对比前缀函数（KMP）：
  - KMP 适合单模式一次匹配；
  - SA 适合同一文本上多次查询与区间检索。
- 对比后缀自动机（SAM）：
  - SAM 在线构建、统计能力强；
  - SA 更容易做字典序相关操作与 LCP 邻接分析。

## R16

典型应用场景：
- 搜索系统中的子串检索基础结构。
- 生物序列（DNA/RNA）模式匹配预处理。
- 文本压缩和重复片段分析。
- 作为 LCP、RMQ、后缀树等结构的工程基础模块。

## R17

可扩展方向：
- 引入 SA-IS / DC3，把构建复杂度降到线性或近线性。
- 在 `lcp` 上叠加 RMQ，实现任意两后缀 LCP 的 `O(1)` 查询。
- 支持批量模式查询接口，复用二分边界过程。
- 增加基准测试与随机回归测试脚本。
- 扩展到 Unicode 规范化预处理，处理多语言文本。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 固定三组文本与查询串，逐组调用 `run_case`。
2. `run_case` 先调用 `build_suffix_array(text)`，用倍增法生成 `sa`。
3. `build_suffix_array` 在每轮按二元键 `(rank[i], rank[i+k])` 排序后缀下标，并重编号得到新 `rank`，直到排名唯一。
4. `run_case` 调用 `build_lcp_array(text, sa)`；该函数先构造逆映射 `rank[pos]`，再按 Kasai 线性扫描得到 `lcp`。
5. `run_case` 再调用 `naive_suffix_array` 与 `naive_lcp_array`，对 SA/LCP 进行断言校验。
6. 对每个查询串，`find_occurrences` 在 SA 上执行两次二分，分别得到 `>= pattern` 与 `> pattern` 的边界。
7. `find_occurrences` 对边界区间做 `startswith` 过滤并返回匹配位置，再与 `naive_occurrences` 断言一致。
8. 所有样例通过后，`main` 打印 `All cases passed.`，表示 MVP 端到端链路成功。
