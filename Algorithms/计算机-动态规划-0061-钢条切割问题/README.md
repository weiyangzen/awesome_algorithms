# 钢条切割问题

- UID: `CS-0044`
- 学科: `计算机`
- 分类: `动态规划`
- 源序号: `61`
- 目标目录: `Algorithms/计算机-动态规划-0061-钢条切割问题`

## R01

钢条切割问题（Rod Cutting）是经典动态规划问题：
- 给定一根长度为 `n` 的钢条；
- 给定价格表 `prices[i-1]`，表示长度 `i` 的整段售价；
- 可选择“切”或“不切”，目标是最大化总收益。

本目录实现一个最小可运行 MVP：输出最优收益，并重建一种对应的切割方案。

## R02

MVP 输入输出定义：
- 输入：
  - `prices`：长度从 `1` 到 `m` 的价格表（一维数值序列）；
  - `n`：目标钢条长度，约束 `0 <= n <= m`；
  - `cut_cost`：每进行一次实际切割的成本（默认 `0`）。
- 输出：
  - `max_revenue`：最大收益；
  - `cuts`：一种达到最优收益的切割方案（如 `[2, 6]`）；
  - 且满足 `sum(cuts) == n`。

## R03

状态定义（自底向上 DP）：
- `best[L]`：长度为 `L` 的钢条可获得的最大收益；
- `first_cut[L]`：在最优解中，长度 `L` 首次切下的段长。

初值：
- `best[0] = 0`（空钢条收益为 0）。

## R04

状态转移方程：

对每个 `L = 1..n`，枚举第一刀 `cut = 1..L`：

`candidate = prices[cut-1] + best[L-cut] - (cut_cost if L-cut > 0 else 0)`

取最大值：

`best[L] = max(candidate)`

并记录达到最优值时的 `cut` 到 `first_cut[L]`。

说明：只有当 `L-cut > 0`（确实把一段再切成两段）时才扣除一次切割成本。

## R05

最优子结构成立原因：
- 若长度 `L` 的最优方案第一段为 `cut`，剩余部分长度为 `L-cut`；
- 剩余部分必须也是长度 `L-cut` 的最优方案，否则可替换为更优子方案从而提升总收益，与“最优”矛盾。

因此可以把原问题拆为“第一刀决策 + 子问题最优值”。

## R06

重建与不变量：
- 通过 `first_cut` 可从 `n` 反推切割方案：
  - `remain = n`
  - 反复取 `cut = first_cut[remain]`，再令 `remain -= cut`，直到 `0`。
- 重建过程中始终保持：
  - 每个 `cut` 满足 `1 <= cut <= remain`；
  - `cuts` 中段长和最终等于 `n`。

## R07

复杂度分析：
- 自底向上 DP：
  - 时间复杂度 `O(n^2)`（双层枚举 `L` 与 `cut`）；
  - 空间复杂度 `O(n)`（`best` 和 `first_cut`）。
- 记忆化递归校验版：
  - 时间复杂度同为 `O(n^2)`；
  - 空间复杂度 `O(n)`（缓存 + 递归深度）。

## R08

`demo.py` 结构：
- `to_price_array`：输入校验并转为 `numpy` 一维数组；
- `rod_cut_bottom_up`：主算法，返回最优收益和切割方案；
- `rod_cut_top_down_revenue`：记忆化递归基线，只返回收益；
- `revenue_from_cuts`：按切割方案回算收益，做一致性检查；
- `run_case`：执行单个测试用例并断言；
- `randomized_cross_check`：随机对拍；
- `main`：组织固定样例并运行全部验证。

## R09

核心接口说明：
- `rod_cut_bottom_up(prices, n, cut_cost=0.0) -> RodCutResult`
  - 返回 `length / max_revenue / cuts`。
- `rod_cut_top_down_revenue(prices, n, cut_cost=0.0) -> float`
  - 返回同问题最优收益，用于校验主算法。
- `revenue_from_cuts(prices, cuts, cut_cost) -> float`
  - 验证切割方案收益是否与 `max_revenue` 一致。

## R10

固定样例（`main()`）包含三组：
1. 经典价格表、`n=8`、`cut_cost=0`，期望收益 `22`；
2. 同价格表、`n=8`、`cut_cost=2`，期望收益 `20`；
3. 非单调价格表、带切割成本，验证算法对一般表格也有效。

## R11

运行方式（无交互输入）：

```bash
uv run python Algorithms/计算机-动态规划-0061-钢条切割问题/demo.py
```

若已在该目录下：

```bash
uv run python demo.py
```

## R12

输出解读：
- `bottom-up`：主算法给出的最优收益与切割方案；
- `cross-check`：
  - `top-down`：记忆化递归收益；
  - `reconstructed`：由 `cuts` 回算得到的收益。

三者应一致；若不一致，程序会抛出 `AssertionError`。

## R13

随机对拍策略：
- 默认执行 `200` 轮；
- 每轮随机生成：
  - 价格表长度 `m ∈ [1, 10]`；
  - 价格值 `0..30`；
  - 目标长度 `n ∈ [1, m]`；
  - 切割成本 `0..3`。
- 对每轮样本断言：
  - 自底向上收益 == 记忆化递归收益；
  - 自底向上收益 == 按切割方案回算收益；
  - `sum(cuts) == n`。

## R14

边界与异常处理：
- `prices` 非一维、空数组、含 `NaN/Inf`：抛 `ValueError`；
- `n < 0` 或 `n > len(prices)`：抛 `ValueError`；
- 重建阶段若出现非法切割长度（理论上不应发生）：抛 `RuntimeError`。

## R15

为什么这个 MVP 是“最小但诚实”的：
- 只依赖 `numpy + 标准库`，环境负担低；
- 没有调用第三方“黑盒最优化接口”；
- 同时提供主算法与独立基线算法做交叉验证；
- 代码规模小，便于直接阅读、调试和扩展。

## R16

可扩展方向：
- 允许 `n > len(prices)` 时通过插值或业务规则补全价格表；
- 支持“必须切 k 段”或“每段长度有上下限”等约束；
- 加入多目标优化（收益、加工损耗、库存成本）；
- 与生产计划系统结合，批量求解多根钢条的切割策略。

## R17

交付核对：
- `README.md`：`R01-R18` 已完整填写；
- `demo.py`：可直接运行，且无交互输入；
- `meta.json`：UID/学科/分类/源序号/目录信息与任务一致；
- 目录自包含，可独立用于该算法项验证。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 先定义固定价格表与三个确定性样例，再调用 `run_case`。  
2. `run_case` 调用 `rod_cut_bottom_up`：创建 `best[0..n]` 与 `first_cut[0..n]`，设 `best[0]=0`。  
3. 在 `rod_cut_bottom_up` 中，外层枚举钢条长度 `L=1..n`，内层枚举第一刀 `cut=1..L`。  
4. 对每个候选计算 `prices[cut-1] + best[L-cut]`，若 `L-cut>0` 再扣一次 `cut_cost`，取最大值写入 `best[L]`，并记录 `first_cut[L]`。  
5. DP 填表完成后，从 `remain=n` 开始按 `first_cut` 反向重建，得到 `cuts`，直到 `remain=0`。  
6. `run_case` 再调用 `rod_cut_top_down_revenue`（`lru_cache` 记忆化递归）计算同一问题的最优收益。  
7. `run_case` 调用 `revenue_from_cuts` 按 `cuts` 回算收益，并断言“主算法收益 = 递归收益 = 回算收益”，同时断言 `sum(cuts)=n`。  
8. 最后 `randomized_cross_check` 对 200 组随机样本重复上述一致性检查，确认实现稳定可靠。
