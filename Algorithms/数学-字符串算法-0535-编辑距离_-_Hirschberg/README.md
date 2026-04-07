# 编辑距离 - Hirschberg

- UID: `MATH-0535`
- 学科: `数学`
- 分类: `字符串算法`
- 源序号: `535`
- 目标目录: `Algorithms/数学-字符串算法-0535-编辑距离_-_Hirschberg`

## R01

本条目实现“编辑距离（Levenshtein 距离）的 Hirschberg 线性空间重建”最小可运行版本（MVP）：
- 计算两个字符串的单位代价编辑距离；
- 返回一组可验证的对齐结果（含对齐字符串与操作序列）；
- 使用 Hirschberg 分治把空间从 `O(mn)` 降到 `O(n)`（`n=len(target)`）；
- 用全矩阵 Wagner-Fischer 距离做交叉校验，确保结果可信。

## R02

问题定义（MVP 范围）：
- 输入：两个字符串 `source`、`target`。
- 编辑代价：
  - 插入 `I = 1`
  - 删除 `D = 1`
  - 替换 `S = 1`
  - 匹配 `M = 0`
- 输出：
  - `distance`：最小编辑距离；
  - `aligned_source` 与 `aligned_target`：等长对齐串（缺口用 `-`）；
  - `operations`：逐位操作码序列，取值为 `M/S/D/I`。

说明：最优对齐一般不唯一，MVP 返回其中一条最优路径即可。

## R03

Hirschberg 核心思想：
- 直接回溯最优路径通常需要完整 `m x n` DP 表，空间 `O(mn)`；
- Hirschberg 通过“前向最后一行 + 反向最后一行”定位最优切分点，仅保留一行状态；
- 然后把问题拆成左右两个子问题递归求解，并拼接对齐；
- 这样保持总时间 `O(mn)` 不变，但峰值额外空间降为 `O(n)`（不计输出对齐本身）。

## R04

算法流程（MVP）：
1. 校验输入类型必须为 `str`。
2. 若 `m==0` 或 `n==0` 或 `m==1` 或 `n==1`，直接走小规模全 DP 回溯（基线分支）。
3. 否则将 `source` 在中点 `mid=m//2` 拆成 `left_source` 与 `right_source`。
4. 计算 `score_left = DP_last_row(left_source, target)`。
5. 计算 `score_right_rev = DP_last_row(reverse(right_source), reverse(target))`。
6. 枚举 `j in [0..n]`，最小化 `score_left[j] + score_right_rev[n-j]`，得到切分 `split`。
7. 递归求左子问题 `align(left_source, target[:split])`。
8. 递归求右子问题 `align(right_source, target[split:])`。
9. 拼接左右结果得到全局 `distance/alignment/operations`，并与 Wagner-Fischer 距离做一致性断言。

## R05

核心数据结构：
- `AlignmentResult`（`dataclass`）：
  - `distance: int`
  - `aligned_source: str`
  - `aligned_target: str`
  - `operations: list[str]`
- 一维滚动数组：`prev/curr`（用于“最后一行”DP）；
- 二维矩阵：仅在 `_small_alignment_dp` 与 `wagner_fischer_distance` 内部用于基线与校验；
- 对齐操作码：`M/S/D/I`，用于可解释输出与合法性检查。

## R06

正确性要点：
- 最优子结构：编辑距离满足标准 DP 递推（删/插/替三选一最小）；
- `DP_last_row(a,b)` 给出所有 `b` 前缀与 `a` 的最优代价，是 Hirschberg 切分依据；
- 对任意切分点 `j`，`cost(j)=left(j)+right(j)`；最小 `cost(j)` 对应某条全局最优路径穿过该列；
- 对左右子问题分别递归求最优并拼接，得到全局最优对齐；
- 基线分支（小规模全 DP 回溯）与分治分支使用同一代价模型，因此组合后仍保持一致最优性。

## R07

复杂度：
- Hirschberg 主流程：
  - 时间复杂度：`O(mn)`
  - 额外空间复杂度：`O(n)`（`n=len(target)`，不计输出对齐串）
- 基线校验（Wagner-Fischer 距离）：
  - 时间复杂度：`O(mn)`
  - 空间复杂度：`O(mn)`

## R08

边界与异常处理：
- 空串：
  - `source=""` 时仅插入；
  - `target=""` 时仅删除；
- 单字符场景：直接走 `_small_alignment_dp`，避免不必要递归；
- 相同字符串：距离应为 `0` 且操作全为 `M`；
- 任意 Unicode 字符：按 Python `str` 字符逐位处理；
- 非字符串输入：抛出 `TypeError`；
- 所有输出通过 `_assert_alignment_valid` 做结构与语义校验。

## R09

MVP 取舍：
- 仅使用 Python 标准库，保证可运行性与可移植性；
- 不调用第三方黑盒编辑距离 API，核心逻辑完全可读可追踪；
- 保留全矩阵距离函数仅用于校验，不参与主流程对齐重建；
- 不扩展带权编辑距离、仿射 gap 罚分、多序列对齐等高级变体。

## R10

`demo.py` 职责划分：
- `_last_row_edit_distance`：计算 Hirschberg 前向/反向切分所需的一维最后行；
- `_small_alignment_dp`：小规模/边界场景的全 DP 回溯；
- `hirschberg_edit_alignment`：分治主过程，输出最优对齐；
- `wagner_fischer_distance`：基线距离计算；
- `_assert_alignment_valid`：验证对齐可还原性与操作合法性；
- `run_case`：单样例执行、打印和断言；
- `main`：固定样例集入口。

## R11

运行方式：

```bash
cd Algorithms/数学-字符串算法-0535-编辑距离_-_Hirschberg
uv run python demo.py
```

脚本无需交互输入，会自动运行内置样例并打印结果。

## R12

输出解读：
- `distance`：Hirschberg 结果，并在括号中展示基线 `baseline`；
- `ops count`：`M/S/D/I` 各操作数量；
- `align S` / `align T`：等长对齐字符串（`-` 表示空位）；
- `ops`：逐位操作码序列。

一致性条件：
- `distance == baseline`；
- `aligned_source/aligned_target` 去掉 `-` 后可分别还原原字符串；
- `operations` 与对齐列逐位语义一致（`M` 必须同字符，`D/I` 必须含 `-`）。

## R13

最小测试集（`main` 已覆盖）：
- `kitten -> sitting`（经典样例）；
- `intention -> execution`（教材样例）；
- `"" -> abc`（纯插入）；
- `sunday -> ""`（纯删除）；
- `algorithm -> altruistic`（混合编辑）；
- `Hirschberg -> Hirshberg`（近似字符串）。

## R14

可调参数与实现约束：
- 当前固定单位代价模型：`I=D=S=1, M=0`；
- 若要改成加权替换/插删，需同时修改：
  - `_last_row_edit_distance`
  - `_small_alignment_dp`
  - `wagner_fischer_distance`
  - `_assert_alignment_valid` 的距离核验逻辑
- 当前实现按字符级处理，不做 token 化或大小写归一化；
- 保持 `run_case` 断言开启，避免“看似可跑但结果错误”。

## R15

方法对比：
- Wagner-Fischer（全矩阵）
  - 优点：实现直接、回溯简单；
  - 缺点：空间 `O(mn)`，长串时内存压力明显。
- Hirschberg（本实现）
  - 优点：保持 `O(mn)` 时间下，把额外空间降到 `O(n)`；
  - 缺点：实现复杂度高于纯全矩阵。
- 第三方库黑盒调用
  - 优点：上手快；
  - 缺点：难解释切分与回溯细节，不满足本条目的教学透明目标。

## R16

适用场景与限制：
- 适用：
  - 长字符串相似度计算且内存预算有限；
  - 需要“距离 + 一条可解释对齐路径”的场景；
  - 教学/面试中演示线性空间 DP 分治思想。
- 限制：
  - 当前仅单位代价、字符级别；
  - 未实现仿射 gap（生物序列常见）；
  - 递归深度随字符串长度增长，极端长串可能需要改写为显式栈版本。

## R17

可扩展方向：
- 支持加权编辑距离（如键盘邻近替换代价）；
- 支持仿射 gap 罚分（Needleman-Wunsch/Gotoh 体系）；
- 增加大小写/标点归一化与 token 级编辑距离；
- 增加随机回归测试与性能基准（短串到长串分档）；
- 增加“只算距离”快速模式，关闭对齐输出以进一步节省开销。

## R18

源码级算法流（对应 `demo.py`，9 步）：
1. `main` 构造固定测试对并逐个调用 `run_case`。
2. `run_case` 调用 `hirschberg_edit_alignment(source, target)` 进入主算法。
3. `hirschberg_edit_alignment` 先做类型校验；若命中边界（空串/单字符）则转 `_small_alignment_dp`。
4. 对一般情形，在 `mid` 处分割 `source`，并调用 `_last_row_edit_distance(left_source, target)` 取前向最后行。
5. 再调用 `_last_row_edit_distance(right_source[::-1], target[::-1])` 取反向最后行。
6. 枚举 `j=0..n`，最小化 `score_left[j] + score_right_rev[n-j]` 得到最优切分 `split`。
7. 对左右子串递归调用 `hirschberg_edit_alignment`，分别得到左对齐与右对齐。
8. 拼接左右 `distance/aligned_source/aligned_target/operations` 得到全局解。
9. `run_case` 继续调用 `wagner_fischer_distance` 与 `_assert_alignment_valid` 做双重校验，然后打印最终结果。

说明：本实现未依赖第三方黑盒编辑距离函数，主算法流程可逐函数追踪到源代码级细节。
