# Beatty序列

- UID: `MATH-0038`
- 学科: `数学`
- 分类: `数论`
- 源序号: `38`
- 目标目录: `Algorithms/数学-数论-0038-Beatty序列`

## R01

Beatty 序列定义为：对任意实数 `alpha > 1`，序列
`B_alpha(n) = floor(n * alpha)`（`n = 1, 2, 3, ...`）。
本条目聚焦经典情形：`alpha` 为无理数时的结构性质，尤其是与互补序列的分拆定理（Rayleigh/Beatty theorem）。

## R02

核心思想有两层：
- 构造层：通过 `floor(n * alpha)` 直接生成单调递增整数序列。
- 结构层：若 `beta = alpha / (alpha - 1)` 且 `alpha` 无理，则
  `B_alpha` 与 `B_beta` 恰好把正整数集合分成不相交的两部分（每个正整数恰出现一次）。

## R03

关键恒等式：
- `1/alpha + 1/beta = 1`，其中 `beta = alpha/(alpha-1)`。
- `alpha > 1` 时有 `beta > 1`。

Beatty 定理（简述）：
- 当 `alpha` 无理且 `beta = alpha/(alpha-1)` 时，
  `A = {floor(n*alpha)}` 与 `B = {floor(n*beta)}` 满足：
  `A ∩ B = ∅` 且 `A ∪ B = Z_{>0}`。

这意味着它们是正整数的“完美互补划分”。

## R04

本 MVP 的输入/输出定义：
- 输入（代码内固定样例，无交互）：若干 `alpha` 与验证上界 `limit`。
- 输出：
  - 序列前若干项（用于观察形态）；
  - 在区间 `[1, limit]` 上的互补性验证结果：
    `disjoint`（是否不相交）与 `covering`（是否完全覆盖）。

## R05

前置条件与约束：
- `alpha > 1`；
- `limit >= 1`；
- 为降低浮点误差，MVP 使用 `decimal.Decimal` 高精度计算并取下整；
- 本实现验证的是有限前缀区间 `[1, limit]`，不是形式化证明无限情形。

## R06

伪代码：

```text
function beatty_prefix(alpha, n):
    seq = []
    for k in 1..n:
        seq.append(floor(k * alpha))
    return seq

function beatty_upto(alpha, limit):
    n_est = floor(limit / alpha) + margin
    seq = beatty_prefix(alpha, n_est)
    while last(seq) < limit:
        n_est = n_est * 2
        seq = beatty_prefix(alpha, n_est)
    return filter(x <= limit, seq)

function verify_complement(alpha, limit):
    beta = alpha / (alpha - 1)
    A = beatty_upto(alpha, limit)
    B = beatty_upto(beta, limit)
    return (A ∩ B == ∅) and (A ∪ B == {1..limit})
```

## R07

正确性要点（有限区间验证）：
- `beatty_prefix` 直接按定义生成 `floor(n*alpha)`，因此每项计算正确。
- `beatty_upto` 通过扩大 `n_est` 直到末项不小于 `limit`，保证不会漏掉 `<= limit` 的候选项。
- 验证阶段用集合操作检查两个条件：
  - `A ∩ B = ∅`（无重叠）；
  - `A ∪ B = {1..limit}`（无缺失）。
- 若二者都成立，则在该有限区间内互补性成立。

## R08

复杂度分析（验证到上界 `limit`）：
- 记 `n1 ≈ limit/alpha`，`n2 ≈ limit/beta`，且 `n1 + n2 ≈ limit`。
- 时间复杂度：`O(limit)`（序列生成与集合检查均线性量级）。
- 空间复杂度：`O(limit)`（保存前缀与集合）。

## R09

与朴素“逐个整数判归属”的对比：
- 朴素法：对每个 `m in [1, limit]` 反向求解是否存在 `n` 使 `floor(n*alpha)=m`，实现复杂且效率低。
- Beatty 前缀法：直接生成两个单调序列再做集合校验，结构清晰、实现简单、可读性高。
- 对教学和算法演示而言，前缀生成法更透明。

## R10

典型应用：
- 数论与离散数学教学中展示“地板函数 + 无理数斜率”的结构分拆；
- Wythoff Nim 等组合博弈中的位置构造（黄金比例相关 Beatty 序列）；
- 需要构造“互补整数分配规则”的建模场景。

## R11

边界与异常处理：
- `alpha <= 1`：不符合 Beatty 标准设定，代码抛出异常；
- `limit < 1`：验证区间非法，代码抛出异常；
- 精度问题：若使用普通 `float`，在较大 `n` 可能出现边界舍入误差；MVP 用 `Decimal` 提高稳定性；
- 即使高精度，有限精度近似也不等价于数学证明，因此结果应解释为“数值验证”。

## R12

`demo.py` 包含以下函数：
- `beatty_prefix(alpha, n)`：生成前 `n` 项；
- `complementary_beta(alpha)`：计算互补参数 `beta = alpha/(alpha-1)`；
- `beatty_upto(alpha, limit)`：生成不超过 `limit` 的全部项；
- `verify_complementarity(alpha, limit)`：检查区间覆盖与互斥；
- `run_demo_cases()`：运行多个 `alpha` 样例并输出 `PASS/FAIL`。

## R13

运行方式：

```bash
python3 demo.py
```

脚本无交互输入，直接打印样例序列与验证结果。

## R14

输出解读：
- 每个样例先打印 `alpha` 与对应 `beta` 的近似值；
- 接着显示两条 Beatty 序列前若干项；
- 然后给出在 `[1, limit]` 上的：
  - `disjoint=True/False`
  - `covering=True/False`
  - 缺失值与重叠值预览（若存在）；
- 两项都为 `True` 则该样例 `PASS`。

## R15

本 MVP 覆盖测试点：
- `alpha = sqrt(2)`：经典无理数互补对；
- `alpha = phi = (1+sqrt(5))/2`：与 Wythoff 结构相关；
- 多个 `limit`（如 `500`、`800`）验证不同规模前缀；
- 自动与断言式汇总结合，出现失败会抛出异常。

## R16

常见错误与规避：
- 把 `beta` 写成 `1/(alpha-1)`（错误），应为 `alpha/(alpha-1)`；
- 只生成固定项数就直接验证覆盖，可能导致尾部漏项；
- 使用二进制浮点直接 `floor(n*alpha)`，在临界附近可能误差翻转；
- 忽略 `alpha > 1` 与 `limit >= 1` 的输入约束。

## R17

工程实践建议：
- 若需要严格数学结论，请使用符号推导或已证明定理，而非仅依赖有限样例；
- 若需要更大规模数值实验，可提高 `Decimal` 精度并分块处理以控制内存；
- 生产环境若性能优先且可容忍微小误差，可改用 `float` + 误差缓冲策略；
- 若用于教学，保留“前缀生成 + 集合验证”这条透明路径最利于理解。

## R18

`demo.py` 的源码级算法流程可拆为 8 步：
1. 读取一个样例 `alpha` 与验证上界 `limit`，先检查 `alpha > 1`、`limit >= 1`。
2. 通过公式 `beta = alpha / (alpha - 1)` 构造互补参数。
3. 估算 `n_est = floor(limit / alpha) + margin`，生成 `A = floor(k*alpha)` 前缀。
4. 若 `A` 末项仍小于 `limit`，倍增 `n_est` 重算，直到覆盖上界。
5. 同样方法生成 `B = floor(k*beta)` 的上界前缀。
6. 过滤得到 `A<=limit`、`B<=limit`，再转为集合 `SA`、`SB`。
7. 计算 `overlap = SA ∩ SB` 与 `missing = {1..limit} - (SA ∪ SB)`。
8. 当 `overlap` 与 `missing` 都为空时，判定该样例在 `[1, limit]` 上通过互补性验证并输出 `PASS`。
