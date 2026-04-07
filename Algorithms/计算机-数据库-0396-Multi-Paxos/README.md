# Multi-Paxos

- UID: `CS-0241`
- 学科: `计算机`
- 分类: `数据库`
- 源序号: `396`
- 目标目录: `Algorithms/计算机-数据库-0396-Multi-Paxos`

## R01

Multi-Paxos 是 Paxos 在“稳定领导者”场景下的工程化版本，用于在不可靠网络和节点故障下，为复制状态机（Replicated State Machine）提供一致的日志顺序。

核心思想：
- 第一次由 leader 完成 Phase 1（Prepare/Promise）建立领导权。
- 后续多个日志槽位（slot）可复用该领导权，直接走 Phase 2（Accept/Accepted）。
- 任一时刻如果领导者失效，新领导者可通过更高 ballot 接管，并继承已接受值，保证安全性。

## R02

问题建模（简化）：
- 节点集合：`N = {acceptor_1 ... acceptor_n}`，`n = 2f + 1`，可容忍 `f` 个故障。
- 提案编号：`ballot = (epoch, leader_id)`，按字典序全序比较。
- 日志槽位：`slot = 1, 2, ...`，每个 slot 选择一个 value。
- 多数派：`quorum_size = floor(n/2) + 1`。

目标：
- Safety（安全性）：同一 slot 不会有两个不同的已选定值。
- Liveness（活性，条件性）：在网络稳定且有唯一活跃 leader 时，新值最终可被选定。

## R03

MVP 假设：
- 崩溃故障（crash-stop），不考虑拜占庭行为。
- 网络可延迟/丢包在抽象层被忽略，示例按同步步骤执行。
- Acceptor 的 `promised` 与 `accepted` 视为持久化。
- Learner 通过观测 acceptor 状态推导 chosen 值（示例中由集群辅助统计）。
- 不实现客户端重试、日志压缩、快照安装等工业细节。

## R04

关键数据结构：
- `ProposalID(epoch, proposer_id)`：可比较的 ballot 标识。
- `AcceptorState`：
  - `promised_id`: 当前承诺不接受更小 ballot。
  - `accepted[slot] = (ballot, value)`: 每个 slot 的已接受提案。
- `Acceptor`：
  - `on_prepare(ballot)` 返回是否 promise 以及已接受快照。
  - `on_accept(ballot, slot, value)` 返回是否接受该提案。
- `Cluster`：
  - 维护 acceptor 列表与多数派阈值。
  - 计算每个 slot 的 chosen 值（某 value 被多数 acceptor 接受）。
- `MultiPaxosLeader`：
  - 执行 Phase 1 获取领导权并恢复历史 slot。
  - 执行 Phase 2 为新 slot 提交值。

## R05

算法伪代码（简化）：

```text
function PHASE1_PREPARE(ballot):
    promises = []
    for a in acceptors:
        ok, accepted_map = a.on_prepare(ballot)
        if ok: promises.append(accepted_map)
    if len(promises) < quorum: fail
    recovered = {}
    for each slot in union(promises.accepted slots):
        recovered[slot] = value with highest ballot among promises
    leader_active = true
    return recovered

function PROPOSE(value):
    if not leader_active:
        recovered = PHASE1_PREPARE(my_ballot)
        for slot, v in recovered (ordered):
            PHASE2_ACCEPT(slot, v)   # 可重放已恢复槽位
    slot = next free slot
    ok = PHASE2_ACCEPT(slot, value)
    if ok: return slot
    else: fail

function PHASE2_ACCEPT(slot, value):
    accepted_count = 0
    for a in acceptors:
        if a.on_accept(my_ballot, slot, value):
            accepted_count += 1
    return accepted_count >= quorum
```

## R06

正确性直觉（Safety）：
- 任意两个多数派必然相交。
- 一旦某 slot 的某 value 在多数派上被接受，任何后续成功的 Phase 1 必会从相交节点看到该 slot 的已接受记录。
- 新 leader 必须选择该 slot 中“最高 ballot 的已接受值”继续推进，因而不会与已选值冲突。

因此，单 slot 的“唯一 chosen value”性质可扩展到多 slot，形成一致日志前缀。

## R07

活性条件：
- 系统存在一段足够长的稳定期（partial synchrony）。
- 只有一个 leader 持续尝试且其 ballot 最终最大。
- 多数派节点存活并可通信。

在这些条件下，Phase 1 最终成功；随后每个新 slot 只需 Phase 2，因此吞吐相对基础 Paxos 更高。

## R08

复杂度（每个 slot）：
- 首次建立领导权（一次性开销）：
  - 消息：`O(n)` Prepare + `O(n)` Promise。
- 稳定 leader 下新 slot：
  - 消息：`O(n)` Accept + `O(n)` Accepted。
- 时间（同步轮次）：
  - 首次可视作 2 轮（Phase1+Phase2）。
  - 后续 slot 常见 1 轮（仅 Phase2）。
- 空间：
  - 每个 acceptor 保存 `O(number_of_slots)` 的 accepted 记录（未做截断时）。

## R09

典型边界情况：
- 过期 leader 使用更小 ballot 发起提案：会被已 promise 的 acceptor 拒绝。
- 新 leader 接管时发现历史 accepted 值：必须优先继承这些值，而不是直接覆盖。
- 并发 leader 竞争：最终较大 ballot 获胜，较小 ballot 进入重试/退避。
- 个别 acceptor 宕机：只要多数派仍在，系统仍可推进。

## R10

与相关协议对比（简述）：
- Basic Paxos：每个实例都可能做 Phase 1，工程成本高。
- Multi-Paxos：将“实例”扩展成日志槽位，并在稳定 leader 下复用 Phase 1。
- Raft：把领导者与日志复制流程设计得更易实现和理解，但安全本质同样依赖多数派交集。

## R11

本目录 MVP 覆盖内容：
- 3 个 acceptor 的最小集群。
- Leader-1 完成 Phase 1 后连续提交多个 slot。
- 模拟过期 leader 失败。
- Leader-2 用更高 ballot 接管，恢复历史并继续提交新值。
- 运行后打印最终 chosen 日志并做断言验证安全性。

## R12

运行方式：

```bash
uv run python Algorithms/计算机-数据库-0396-Multi-Paxos/demo.py
```

预期行为：
- 输出每次提交结果、过期 leader 失败信息、接管后继续提交结果。
- 最后打印所有 chosen slots，并显示安全性检查通过。

## R13

示例输出（节选，实际顺序可能略有不同）：

```text
=== Multi-Paxos MVP Demo ===
Leader L1 active with ballot=(1, 'L1')
L1 proposed slot 1 -> SET x=1
L1 proposed slot 2 -> SET y=2
Stale leader rejected as expected: Phase1 failed: no quorum promises
Leader L2 active with ballot=(2, 'L2')
L2 proposed slot 3 -> SET z=3
Chosen log:
  slot 1: SET x=1
  slot 2: SET y=2
  slot 3: SET z=3
Safety check passed.
```

## R14

局限性：
- 未实现真实网络、超时、重试与心跳。
- 未区分“已接受 accepted”和“已提交 committed”的完整传播机制。
- 未实现日志截断、快照、成员变更（reconfiguration）。
- 未提供持久化介质与崩溃恢复代码，仅在内存中模拟。

## R15

可扩展方向：
- 加入超时与重试策略，模拟 leader 选举与抢占。
- 在 `Cluster` 层增加消息延迟/丢包模型，观察活性边界。
- 增加持久化（例如 SQLite/文件）验证崩溃恢复。
- 增加批量提案、流水线提交与性能指标采样。

## R16

测试建议：
- 单元测试：
  - `on_prepare` 对高低 ballot 的承诺行为。
  - `on_accept` 对 promise 约束的拒绝路径。
  - `compute_chosen_values` 的多数判定逻辑。
- 场景测试：
  - 稳定 leader 连续提交。
  - 旧 leader 被拒绝。
  - 新 leader 接管后不破坏既有 chosen 值。
- 属性检查：
  - 任意时刻同一 slot 最多一个 chosen value（可用随机事件序列做 quickcheck）。

## R17

术语表：
- `leader/proposer`：发起提案并尝试驱动共识的节点角色。
- `acceptor`：投票核心，保存 promise 与 accepted 状态。
- `learner`：学习最终结果的角色（示例中由集群统计代替）。
- `ballot`：提案编号，决定新旧领导权。
- `slot`：复制日志的索引位置。
- `quorum`：多数派，任何两个 quorum 必相交。
- `chosen`：某 slot 的某值被多数 acceptor 接受，视为已定值。

## R18

`demo.py` 的源码级算法流程（8 步）：
1. 创建 `Cluster(acceptor_count=3)`，初始化 3 个 acceptor 的空状态（无 promise、无 accepted）。
2. 创建 `MultiPaxosLeader("L1", epoch=1)`，调用 `phase1_prepare()`：
   - 向所有 acceptor 发送 prepare。
   - 收集 quorum promises 与历史 accepted 快照。
   - 根据“每个 slot 最高 ballot”规则构造恢复映射。
3. L1 调用 `propose("SET x=1")` 与 `propose("SET y=2")`：
   - 为新 slot 发送 accept 请求到全部 acceptor。
   - 获得多数 accepted 后判定该 slot chosen。
4. 创建过期 leader `("OLD", epoch=0)` 并尝试 `propose(...)`：
   - 因 ballot 过小拿不到 quorum promise，抛出失败异常。
5. 创建新 leader `("L2", epoch=2)` 并执行 `phase1_prepare()`：
   - 使用更高 ballot 成功拿到 quorum。
   - 从 promises 中恢复旧 slot 已接受值，保证接管不丢历史一致性。
6. L2 调用 `propose("SET z=3")`，在下一个空 slot 完成 Phase 2 并 chosen。
7. `Cluster.compute_chosen_values()` 遍历各 acceptor 的 accepted 记录，按 slot 统计 value 计数，筛出达到 quorum 的 chosen 映射。
8. 主程序对 slot1/2/3 做断言，验证：
   - 旧值未被覆盖（安全性）。
   - 新值可继续提交（活性条件下可推进）。
