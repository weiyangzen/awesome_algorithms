# 拜占庭容错

- UID: `CS-0317`
- 学科: `计算机`
- 分类: `并行与分布式`
- 源序号: `478`
- 目标目录: `Algorithms/计算机-并行与分布式-0478-拜占庭容错`

## R01

拜占庭容错（Byzantine Fault Tolerance, BFT）研究的是：在部分节点可能任意作恶（发送冲突消息、伪造状态、选择性失联）的情况下，分布式系统仍然能对外给出一致结果。

本条目采用最经典的工程路线之一 `PBFT`（Practical BFT）作为 MVP 载体，聚焦“状态机复制 + 多阶段投票”。

## R02

动机来自“故障不再是仅崩溃（crash）”：

- 在开放网络或跨组织协作环境，节点可能表现为任意错误，而不仅是宕机；
- 仅靠主备复制或 Raft/Paxos（主要防 crash fault）不足以处理恶意行为；
- 金融记账、联盟链、关键控制平面等场景要求在攻击下仍保持一致性与可用性。

BFT 的目标是把“最多 `f` 个拜占庭节点”的破坏限制在可证明边界内。

## R03

PBFT 常见模型假设：

- 副本总数 `n = 3f + 1`；
- 最多 `f` 个副本是拜占庭行为；
- 网络可异步，但最终会传递消息（eventual delivery）；
- 使用认证信道（例如 MAC/签名）避免冒名顶替；
- 客户端请求会被编号（`view, seq`）并附带摘要 `digest`。

本目录 `demo.py` 固定使用 `n=4, f=1`。

## R04

PBFT（简化）主流程：

1. `Pre-Prepare`：主节点为请求分配序号并广播摘要；
2. `Prepare`：副本验证后广播“我看到同一个摘要”；
3. `Commit`：副本在达到准备条件后广播提交票；
4. `Execute/Reply`：副本在达到提交阈值后执行请求并可回复客户端。

阈值（本示例采用）：

- prepared：持有合法 pre-prepare，且收到不少于 `2f` 个匹配 prepare；
- committed-local：在 prepared 基础上，收到不少于 `2f+1` 个匹配 commit。

## R05

复杂度（单请求）：

- 消息量近似 `O(n^2)`（prepare 与 commit 都是全互发）；
- 每个副本的投票状态存储近似 `O(n)`；
- 当批量请求并行时，吞吐受网络广播和签名验证成本限制。

PBFT 的核心代价是“用更多消息换恶意容错”。

## R06

`demo.py` 的 MVP 设计：

- 4 个副本：`R0,R1,R2,R3`，其中 `R3` 固定为拜占庭；
- 主节点 `R0` 发起一个请求的 pre-prepare；
- 诚实副本对正确摘要广播 prepare/commit；
- 拜占庭副本发送冲突投票（对不同接收者发送不同摘要）；
- 诚实副本仅统计与已接受 pre-prepare 匹配的消息；
- 程序最终断言诚实副本仍然达到提交并执行。

## R07

优点：

- 相比仅防崩溃协议，能抵御更强故障模型；
- 在 `n=3f+1` 前提下有明确安全性阈值；
- 对抗单点恶意领导者时，仍可通过多副本投票保一致。

局限：

- 消息复杂度高，节点规模扩大后开销明显；
- 工程实现复杂（视图切换、日志裁剪、加密认证）；
- 在高延迟网络中性能劣于 crash-only 共识。

## R08

前置知识与运行环境：

- 分布式状态机复制、法定人数（quorum）概念；
- 拜占庭故障与 crash fault 区别；
- Python 3.10+；
- `numpy`（用于投票矩阵与统计展示）。

运行：

```bash
cd Algorithms/计算机-并行与分布式-0478-拜占庭容错
uv run python demo.py
```

## R09

适用场景：

- 联盟链/许可链共识层；
- 高信任要求的多机构账本；
- 需要抵御恶意节点的控制平面复制。

不适用场景：

- 节点超大规模且极度带宽敏感；
- 故障模型仅为宕机、无作恶风险；
- 对延迟要求极端苛刻而不能承担多轮广播。

## R10

正确性直觉（简版）：

- 任何两个大小至少 `2f+1` 的副本集合必有交集至少 `f+1`；
- 诚实副本最多 `2f`，因此两个已提交值不可能都由互不相交的诚实见证支持；
- 诚实副本只会为与 pre-prepare 匹配的摘要计票；
- 拜占庭副本即便发送冲突消息，也无法同时让两个不同摘要都跨过诚实阈值。

因此在阈值满足时，系统可维持一致提交。

## R11

`demo.py` 的健壮性检查：

- 初始化时断言 `n >= 3f+1`；
- 断言主节点存在于副本集合；
- 诚实副本仅接受来自主节点的 pre-prepare；
- prepare/commit 仅统计“摘要匹配 + 发送者合法”的消息；
- 结束时断言：
  - 至少 `2f+1` 个诚实副本提交成功；
  - 没有诚实副本对错误摘要达成 prepared/committed。

## R12

实现细节：

- `ReplicaState` 保存单副本状态（已接受摘要、prepare/commit 发送者集合）；
- `PBFTSimulator` 负责消息投递与阈值判断；
- 使用 `set` 去重发送者，避免重复消息虚增票数；
- 使用 `numpy` 维护 `prepare/commit` 接收矩阵（行=接收者，列=发送者）；
- 明确区分 `good_digest` 与 `bad_digest`，便于观察拜占庭干扰。

## R13

理论边界与工程注意：

- 本 MVP 不含 view-change（主节点切换），只演示单视图安全性；
- 未实现密码学签名验证流程，仅用“发送者身份可信”抽象代替；
- 未覆盖重放攻击、消息延迟重排缓存、日志检查点裁剪；
- 真实 PBFT 系统还需持久化日志与恢复协议，才能保证崩溃恢复后的安全性。

## R14

常见错误：

1. 把“收到消息总数”当票数，而不是“不同发送者数量”；
2. 没把 pre-prepare 绑定到 `(view, seq, digest)` 三元组；
3. prepared/committed 阈值写错（尤其 `2f` 与 `2f+1` 混淆）；
4. 允许诚实副本为多个摘要重复投票；
5. 只验证活性，不验证安全性断言。

## R15

可扩展方向：

- 增加 view-change，模拟主节点作恶后的领导切换；
- 支持批处理请求与流水线，提高吞吐；
- 引入签名与消息认证码开销建模；
- 注入网络乱序/丢包/重复消息，做鲁棒性压力测试；
- 对接真实状态机（键值存储）而非仅模拟摘要投票。

## R16

相关机制：

- `Paxos / Raft`：主要容忍 crash fault，不直接覆盖拜占庭作恶；
- `Tendermint / HotStuff`：现代 BFT 共识家族，改进消息流程与工程可用性；
- `BLS 聚合签名`：可降低部分投票消息体积；
- `Quorum Certificate`：将投票证明结构化为可验证证据。

## R17

`demo.py` 模块清单：

- `digest_request`：请求摘要计算；
- `ReplicaState`：副本局部状态；
- `PBFTSimulator`：
  - pre-prepare 投递与验证；
  - prepare/commit 收集；
  - prepared/committed 判定；
  - 最终安全性断言；
- `print_matrix`：打印 `numpy` 投票矩阵；
- `main`：构造固定场景并输出结果。

脚本无交互输入，适合直接批量验证。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `main` 初始化 `PBFTSimulator(n=4,f=1)`，指定 `R0` 主节点与 `R3` 拜占庭节点，并生成 `good/bad` 两个摘要。  
2. `broadcast_preprepare` 由主节点向所有副本广播 `(view,seq,digest)`；诚实副本在 `deliver_preprepare` 中仅接受来自主节点的合法摘要。  
3. 诚实副本通过 `broadcast_prepare_honest` 全互发 `PREPARE(good)`；`deliver_prepare` 用“接收者本地已接受摘要”做过滤。  
4. 拜占庭副本通过 `broadcast_prepare_byzantine` 向不同接收者发送冲突摘要（`good/bad` 混发），制造分叉噪声。  
5. `update_prepared` 逐副本检查 `len(prepare_senders[digest]) >= 2f`；满足则标记 prepared。  
6. prepared 的诚实副本执行 `broadcast_commit_honest`，拜占庭副本再通过 `broadcast_commit_byzantine` 混发冲突 commit。  
7. `update_committed_and_execute` 检查 `len(commit_senders[digest]) >= 2f+1`，满足后将副本标记为 committed 并“执行请求”。  
8. `assert_safety` 验证至少 `2f+1` 个诚实副本提交，且无诚实副本对错误摘要提交；最后打印 prepare/commit 矩阵与提交结果。

本实现未调用外部共识库黑盒函数，阈值判断、消息过滤、状态推进都在源码中逐步展开，可直接追踪。
