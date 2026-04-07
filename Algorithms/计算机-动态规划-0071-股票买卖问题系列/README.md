# 股票买卖问题系列

- UID: `CS-0054`
- 学科: `计算机`
- 分类: `动态规划`
- 源序号: `71`
- 目标目录: `Algorithms/计算机-动态规划-0071-股票买卖问题系列`

## R01

“股票买卖问题系列”是动态规划中的经典状态机建模题组，核心是：
在按天给定价格序列下，结合交易次数、冷冻期、手续费等约束，求最大可实现利润。  
这类问题的关键不是搜索所有交易路径，而是把“是否持股、已完成交易数、是否处于冷冻状态”压缩为少量可递推状态。

## R02

本条目在一个 `demo.py` 里覆盖 5 个常见变体：
- `at most 1 transaction`（最多一次交易）。
- `unlimited transactions`（不限交易次数，同一时间只能持有一股）。
- `unlimited + cooldown=1 day`（卖出后冷冻 1 天）。
- `unlimited + fee`（每次卖出扣手续费）。
- `at most k transactions`（最多 `k` 次完整买卖）。

输入为价格数组 `prices`；输出为各变体最大利润。脚本使用固定样例，运行时无需任何交互输入。

## R03

统一 DP 视角：
- `hold`：当天结束后持股时的最大利润。
- `cash/sell`：当天结束后不持股时的最大利润。

典型转移：
- 无限制交易：
  - `new_cash = max(cash, hold + price)`
  - `new_hold = max(hold, cash - price)`
- 含手续费：
  - `new_cash = max(cash, hold + price - fee)`
- 含冷冻期（`hold/sold/rest` 三状态）：
  - `hold = max(prev_hold, prev_rest - price)`
  - `sold = prev_hold + price`
  - `rest = max(prev_rest, prev_sold)`
- 最多 `k` 次交易（`buy[t], sell[t]`）：
  - `buy[t] = max(prev_buy[t], prev_sell[t-1] - price)`
  - `sell[t] = max(prev_sell[t], prev_buy[t] + price)`

## R04

算法高层流程：
1. 校验 `prices` 合法（1D、有限数值、长度允许为 0）。
2. 对每个变体调用对应 DP 函数。
3. 在每个函数中按“日”线性扫描价格，并用常数或 `O(k)` 状态递推。
4. 打印各变体利润结果。
5. 做基本一致性校验：当 `k >= n//2` 时，`k` 次交易结果应等于无限次交易结果。
6. 演示异常输入触发路径，验证健壮性。

## R05

核心数据结构：
- `prices_arr: np.ndarray(shape=(n,))`：价格序列。
- 标量状态：`cash, hold, sold, rest`。
- `buy, sell: np.ndarray(shape=(k+1,))`：`k` 次交易版本状态数组。
- `dict[str, float]`：收集并输出各变体利润结果。

MVP 仅依赖 `numpy`，不引入更重框架。

## R06

正确性要点：
- 最优子结构：第 `i` 天最优决策只依赖第 `i-1` 天状态，不依赖更早细节路径。
- 无后效性：用“是否持股/交易计数/冷冻状态”即可刻画未来决策所需信息。
- 状态转移完备：每个状态都覆盖“保持原状”与“执行操作”两种选择，并取 `max`。
- `k` 次交易版本通过 `t` 维度约束交易上限，确保不越界地完成买卖配对。

## R07

复杂度：
- 单次交易：时间 `O(n)`，空间 `O(1)`。
- 无限次/手续费/冷冻期：时间 `O(n)`，空间 `O(1)`。
- 最多 `k` 次交易：时间 `O(nk)`，空间 `O(k)`。

其中 `n` 为价格天数。

## R08

边界与异常处理：
- 空数组或单元素数组：利润为 `0`。
- 价格数组非 1 维：抛出 `ValueError`。
- 存在 `NaN/Inf`：抛出 `ValueError`。
- `k < 0` 或 `fee < 0`：抛出 `ValueError`。
- `k == 0`：直接返回 `0`。

## R09

MVP 取舍：
- 采用手写 DP 状态机，不调用第三方“现成股票题求解器”黑盒。
- 使用固定数据样例，强调算法逻辑可复现和可验证。
- 使用 `numpy` 仅做数据容器与数值校验，不掩盖核心状态转移细节。

## R10

`demo.py` 函数职责：
- `validate_prices`：统一校验并标准化价格输入。
- `max_profit_one_transaction`：最多一次交易。
- `max_profit_unlimited_transactions`：无限次交易。
- `max_profit_with_fee`：含手续费。
- `max_profit_with_cooldown`：含 1 天冷冻期。
- `max_profit_k_transactions`：最多 `k` 次交易。
- `main`：运行示例、打印结果、执行一致性和异常测试。

## R11

运行方式：

```bash
cd Algorithms/计算机-动态规划-0071-股票买卖问题系列
uv run python demo.py
```

脚本不读取标准输入，不依赖命令行参数。

## R12

输出字段说明：
- `one_transaction`：最多一次交易利润。
- `unlimited_transactions`：无限次交易利润。
- `with_fee`：含手续费利润。
- `with_cooldown`：含冷冻期利润。
- `k_transactions`：最多 `k` 次交易利润。
- `consistency_check_passed`：`k` 大时与无限次版本的一致性是否通过。

## R13

建议最小测试集：
- 单调递增价格：应倾向“买入最早、卖出最晚”。
- 单调递减价格：应输出 `0`。
- 高频震荡价格：检验多次交易策略优势。
- 冷冻期敏感样例（如 `[1,2,3,0,2]`）。
- 手续费敏感样例（手续费高时抑制频繁交易）。
- `k=0`、`k=1`、`k>=n//2` 三组对比。

## R14

可调参数：
- `fee`：每次卖出手续费。
- `k`：最多交易次数上限。
- `prices`：可替换为任意长度的历史价格序列。

工程建议：批量评测时可把多个样例打包循环，统计不同约束下的利润变化曲线。

## R15

与其它解法对比：
- 与暴力 DFS/回溯相比：DP 把指数复杂度降到线性或线性乘 `k`。
- 与贪心相比：无约束场景可贪心求和正差分，但冷冻期/手续费/交易上限场景更适合状态机 DP。
- 与通用 MDP 求解器相比：这里状态维度小、转移显式，手写 DP 更直接且可解释。

## R16

典型应用：
- 交易策略原型设计中的约束收益评估。
- 金融教学中的状态机建模示例。
- 面试/竞赛中的动态规划综合题。
- 带交易成本和频率约束的简化决策模拟。

## R17

可扩展方向：
- 支持“最多持仓多股”或“做空”状态扩展。
- 加入最短持有天数、禁买窗口等业务约束。
- 从单资产扩展到多资产联合状态（维度会上升）。
- 在 DP 之上叠加风险惩罚项（如回撤惩罚）做多目标优化。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 构造固定价格序列、`fee`、`k`，并调用 `run_all_variants` 汇总结果。  
2. `run_all_variants` 先调用 `validate_prices`，确保输入是一维有限浮点数组。  
3. `max_profit_one_transaction` 维护 `min_price` 与 `best`，逐日更新单次交易最优利润。  
4. `max_profit_unlimited_transactions` 使用 `cash/hold` 两状态，按日做“继续持有/买入/卖出”转移。  
5. `max_profit_with_fee` 复用两状态框架，只在卖出转移时扣除 `fee`。  
6. `max_profit_with_cooldown` 使用 `hold/sold/rest` 三状态，显式编码“卖出后次日不可买入”。  
7. `max_profit_k_transactions` 建立 `buy[t], sell[t]`，对每个价格和每个交易层 `t` 执行两条转移，得到 `O(nk)` 解。  
8. `main` 打印各利润，并验证 `k >= n//2` 时 `k` 次交易结果与无限次交易结果一致，同时演示非法输入异常。  
