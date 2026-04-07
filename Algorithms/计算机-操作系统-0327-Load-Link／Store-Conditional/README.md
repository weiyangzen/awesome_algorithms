# Load-Link/Store-Conditional

- UID: `CS-0180`
- 学科: `计算机`
- 分类: `操作系统`
- 源序号: `327`
- 目标目录: `Algorithms/计算机-操作系统-0327-Load-Link／Store-Conditional`

## R01

Load-Link/Store-Conditional（LL/SC）是并发同步中的经典原子原语对：

- `LL(addr)`：读取地址 `addr` 当前值，并在本线程建立 reservation（保留监视）；
- `SC(addr, new)`：仅当 reservation 仍有效时才写入 `new` 并返回成功；
- 若 reservation 失效（中间有冲突写入、上下文切换或体系结构规定事件），`SC` 失败且不写入。

它将“读-改-写”的正确性依赖从值比较扩展到“是否发生过干扰写入”的检测。

## R02

在操作系统和并发运行时里，LL/SC 常用于：

- 实现无锁计数器、无锁队列节点更新、短临界区状态位切换；
- 与 CAS（Compare-and-Swap）并列作为硬件级同步基元；
- 在一些架构中降低 CAS 的 ABA 风险（通过 reservation 失效机制）。

典型使用方式是“LL/SC 重试循环”：失败后重新 LL，再尝试 SC。

## R03

可把单地址 LL/SC 形式化为状态转移：

- 共享状态：`x`（值）与 `ver`（写版本）；
- `LL_t`：线程 `t` 读取 `(x, ver)` 并记录本地 reservation=`ver`；
- `SC_t(new)` 成功条件：
  - `reservation_t` 仍存在；
  - `reservation_t == ver`（期间无成功写入导致版本变化）；
  - 满足调用者自己的值约束（本 demo 中要求写回基于 LL 时读到的值）。

不变量：

- 每次成功 SC 都导致 `ver` 递增；
- 成功更新次数与最终计数增量一致；
- 任意中间写入都会使旧 reservation 过期。

## R04

本目录 MVP 用“多线程竞争同一整数计数器”展示 LL/SC：

1. `AtomicLLSCInteger.load_link(tid)` 返回 `(value, token)`；
2. `store_conditional(tid, token, expected, expected+1)` 进行条件写入；
3. 失败时按原因分类统计并指数退避后重试；
4. 最终对计数值、账本守恒、失败原因分解做断言；
5. 额外给出一个 ABA-like 场景：值被改后又改回，旧 token 的 SC 仍失败。

说明：Python 无直接 LL/SC 指令接口，demo 用短锁临界区模拟单地址原子语义，目标是教学与可验证性。

## R05

记线程数 `T`、每线程成功操作数 `K`、总成功数 `N=T*K`、总失败重试数 `F`：

- 总尝试次数 `A = N + F`；
- 时间复杂度：`O(A)`；
- 空间复杂度：
  - 核心共享状态 `O(1)`；
  - 线程级统计数组 `O(T)`；
  - 每次成功的重试记录 `O(N)`（用于统计分位数）。

## R06

简化时序（两线程冲突）如下：

1. `T1` 执行 LL，读到 `x=9`、token=17；
2. `T2` 执行 LL，读到 `x=9`、token=17；
3. `T1` 先 SC 成功：`x=10`，版本变为 18，并使旧 reservation 失效；
4. `T2` 带旧 token=17 执行 SC 失败（lost reservation）；
5. `T2` 重新 LL 得到新 token=18，再 SC 才可成功。

该过程说明 LL/SC 能直接检测“LL 到 SC 之间是否被他人写入”。

## R07

优点：

- 语义直接表达“中间无干扰写入才提交”；
- 在不少架构上可自然应对“值被改回”情形；
- 失败可观测，便于退避与冲突分析。

局限：

- reservation 可能被多种事件打断，失败率受实现细节影响；
- 高冲突下重试开销显著；
- 真实系统仍需配套内存序、对象回收与公平性策略。

## R08

前置知识：

- 线程并发、竞态条件、线性化点；
- 原子原语（CAS / LL/SC）与自旋重试；
- 指数退避和活锁风险。

运行环境：

- Python `>=3.10`
- `numpy`（用于统计汇总：均值、p95、最大值）

## R09

适用场景：

- 单地址或少量地址的原子更新；
- 需要观测并分析冲突失败率的同步路径；
- 教学/实验中对比 CAS 与 LL/SC 的语义差异。

不适用场景：

- 跨多个独立地址的原子事务提交；
- 极高冲突且不具备退避/分片优化；
- 需要完整生产级内存回收安全保证而未配套方案。

## R10

`demo.py` 的正确性检查点：

1. 所有线程都在超时前结束；
2. `final_value == num_threads * ops_per_thread`；
3. `total_successes == expected_value`；
4. `total_attempts == total_successes + total_failures`；
5. `total_failures == lost + value_mismatch + invalid_reservation`；
6. 每个成功操作对应一条重试记录；
7. ABA-like 场景中，旧 token 的 SC 必须失败。

## R11

并发语义与风险说明：

- ABA 相关：
  - CAS 仅比较值时，`A->B->A` 可能误判无变化；
  - LL/SC 通过 reservation 失效检测中间写入，可降低该风险。

- 活锁风险：
  - 多线程反复冲突会导致长期重试；
  - 本 demo 使用指数退避，减少同相位重碰撞。

- 内存模型：
  - 真实硬件还涉及 acquire/release/seq_cst 语义；
  - 本实现主要验证控制流与失败分类，不展开底层内存序细节。

## R12

`demo.py` 主要参数：

- `num_threads`：并发线程数（默认 8）；
- `ops_per_thread`：每线程成功增量次数（默认 260）；
- `seed`：随机种子，便于复现；
- `ll_sc_pause_prob / ll_sc_pause_max_s`：在 LL 与 SC 间引入竞争窗口。

调参建议：

- 想提高冲突可见性：增大线程数或 pause 概率；
- 想快速 smoke test：减小 `ops_per_thread`；
- 想降低波动：固定 seed 并减少系统背景负载。

## R13

理论保证（本题性质）：

- 近似比：N/A（非近似优化任务）；
- 随机化正确性保证：N/A（随机仅用于扰动时序）；
- 可验证性质：
  - 成功次数与最终值一致；
  - 失败仅延迟进度，不破坏计数单调正确性；
  - 旧 reservation 在中间写后不可再提交。

## R14

常见失效模式与防护：

1. 仅做普通读写而不使用原子原语，导致丢失更新；
2. SC 失败后不重试，造成操作遗漏；
3. 不统计失败原因，无法判断是冲突还是实现错误；
4. 只看最终值，不检查 attempt/success/failure 账本守恒。

本实现通过重试循环、失败原因分类、指数退避和完整断言覆盖这些风险。

## R15

工程落地建议：

- 在压测里同时看吞吐和失败率分解（lost reservation 占比）；
- 高冲突路径优先考虑分片计数、局部聚合、批量提交；
- 需要严格实时性时，评估 LL/SC 失败抖动对 tail latency 的影响；
- 生产环境务必结合架构内存序语义与对象回收安全机制。

## R16

相关算法与主题：

- CAS、Test-and-Set、Fetch-and-Add；
- 无锁数据结构：Treiber Stack、Michael-Scott Queue；
- RCU、seqlock、ticket lock（语义与适用边界不同）；
- 活锁/饥饿/公平性在无锁同步中的权衡。

## R17

本目录交付内容：

- `README.md`：LL/SC 定义、复杂度、风险、参数与工程建议；
- `demo.py`：并发竞争仿真 + ABA-like 场景验证；
- `meta.json`：任务元数据（UID/分类/路径）一致。

运行方式：

```bash
cd Algorithms/计算机-操作系统-0327-Load-Link／Store-Conditional
uv run python demo.py
```

脚本无交互输入，运行结束后会输出统计并打印断言通过信息。

## R18

`demo.py` 源码级算法流程（8 步）：

1. 初始化 `AtomicLLSCInteger`，维护 `value`、`version` 与 `reservations[tid]`；`load_link` 记录线程 reservation token。  
2. 工作线程先执行 `LL` 取 `(expected, token)`，然后在 LL 与 SC 之间按概率短暂停顿，主动制造竞争窗口。  
3. 线程执行 `store_conditional(tid, token, expected, expected+1)`；函数内按顺序检查 reservation 是否存在、token 是否匹配当前版本、值是否仍等于 expected。  
4. 若检查全部通过，SC 成功写入新值，`version += 1`，并清空 reservation（表示这次写使先前链接全部失效）。  
5. 若任一检查失败，SC 返回明确失败原因（`lost_reservation`/`value_mismatch`/`invalid_reservation`），线程进入重试并执行指数退避。  
6. 所有线程完成后，程序汇总 attempts/successes/failures 与失败原因分解，并用 `numpy` 计算重试均值、p95、最大值。  
7. 执行一致性断言：最终值正确、账本守恒、失败原因可加总、每次成功都有重试记录，保证语义与统计同时正确。  
8. 额外运行 ABA-like 场景：线程 A 先 LL，线程 B 做 `5->6->5` 两次成功写；A 用旧 token 的 SC 必须失败，展示 LL/SC 对中间写入的可检测性。  

本实现没有把关键机制交给黑盒库：reservation 建立、版本失效、SC 判定、失败分类、退避重试与断言都在源码中显式展开。
