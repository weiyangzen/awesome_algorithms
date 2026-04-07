# 分布式快照

- UID: `CS-0316`
- 学科: `计算机`
- 分类: `并行与分布式`
- 源序号: `477`
- 目标目录: `Algorithms/计算机-并行与分布式-0477-分布式快照`

## R01

本条目实现 Chandy-Lamport 分布式快照（Distributed Snapshot）的最小可运行 MVP，目标是：
- 在异步、FIFO 信道系统中记录一个一致全局状态；
- 明确展示“在途消息”会进入信道状态而不是进程本地状态；
- 提供可重复的固定场景，`uv run python demo.py` 无交互即可复现。

## R02

问题定义（MVP 范围）：
- 系统由多个进程和有向通信信道组成；
- 进程状态：这里用账户余额表示；
- 应用消息：这里用转账消息 `APP(amount)` 表示；
- 控制消息：快照标记 `MARKER(snapshot_id)`；
- 输出：
1. 每个进程记录时刻的本地状态；
2. 每条信道的在途消息集合；
3. 一致性校验（局部状态总和 + 信道在途总和 = 初始总量）。

## R03

模型假设：
- 信道可靠，不丢消息；
- 每条有向信道满足 FIFO；
- 进程不会崩溃，不考虑恢复；
- 单次快照（同一时段内不并发多个 snapshot id）；
- 离散时间仿真仅用于演示，算法语义对应异步消息系统。

## R04

Chandy-Lamport 核心规则：
- 发起者先记录本地状态，再向所有出边发送 `MARKER`；
- 进程首次收到某个快照的 `MARKER` 时：
1. 立即记录本地状态；
2. 将该 `MARKER` 所在入边信道状态记为空；
3. 向全部出边转发 `MARKER`；
4. 开始记录其余入边后续到达的应用消息，直到这些入边收到 `MARKER`。
- 进程后续收到某条入边的 `MARKER` 时，停止记录该入边；
- 所有进程都收到全部入边的 `MARKER` 后，快照完成。

## R05

`demo.py` 的关键数据结构：
- `ProcessState`：进程本地余额；
- `Message`：`APP` 与 `MARKER` 统一消息结构；
- `Envelope`：信道在途包裹（带 `deliver_step` 与 FIFO 序号 `seq`）；
- `SnapshotRuntime`：
1. `local_state`：每个进程记录到的余额；
2. `channel_state`：每条信道记录到的在途 `APP` 消息；
3. `marker_received`：每个进程已收到标记的入边集合；
4. `recording_channels`：仍在录制中的入边集合。

## R06

本 MVP 固定场景（3 进程，6 条有向信道）：
- 初始余额：`P0=100, P1=100, P2=100`；
- `t=0`：`P2 -> P1` 发送 `T1(amount=30)`；
- `t=1`：`P0` 发起快照 `S1`；
- 通过延迟设置使 `P1` 先记录本地状态，再收到 `T1`，并在来自 `P2` 的 `MARKER` 到达前将 `T1` 记入 `C2->1` 信道状态；
- 这正是“在途消息被快照捕获”的典型行为。

## R07

正确性直觉：
- FIFO 保证同一信道上“先发送的应用消息”不会被“后发送的 `MARKER`”超车；
- 因此，若接收方已开始记录该入边，且尚未收到该入边 `MARKER`，到达的应用消息必属于该切面上的在途消息；
- 由此得到的全局切面不会遗漏这类消息，也不会重复计入，形成一致快照。

## R08

复杂度（单次快照）：
- 控制消息开销：每条有向信道 1 个 `MARKER`，即 `O(E)`；
- 本地状态记录开销：每个进程 1 次，`O(V)`；
- 在途消息记录量：与快照窗口期间穿过“尚未关闭入边”的应用消息数成正比，记为 `M`；
- 总体空间开销：`O(V + E + M)`。

## R09

MVP 中的输入与边界检查：
- 不允许空进程集合；
- 不允许自环信道；
- 信道端点必须是已存在进程；
- 每个进程必须至少 1 条入边和 1 条出边（便于演示完整标记闭合）；
- 转账金额必须为正且余额充足；
- 消息延迟必须非负；
- 同名 `transfer_id` 会报错，避免重复记账。

## R10

`demo.py` 函数职责：
- `DistributedSnapshotSimulator._enqueue`：FIFO 入信道调度；
- `send_app`：发送应用消息并更新发送方余额；
- `initiate_snapshot`：创建快照运行时并由发起者启动；
- `_record_local_state`：首次记录本地状态并发出标记；
- `_receive_app`：接收应用消息并在需要时记入信道状态；
- `_receive_marker`：执行“首次/后续 marker”分支逻辑；
- `_update_snapshot_completion`：检测全局快照完成；
- `run_scenario`：运行固定脚本；
- `validate_snapshot`：执行一致性断言；
- `print_summary`：输出可读报告。

## R11

运行方式：

```bash
cd Algorithms/计算机-并行与分布式-0477-分布式快照
uv run python demo.py
```

脚本不读取任何交互输入，直接打印事件日志、快照结果和断言结论。

## R12

输出阅读说明：
- `event log`：按时间列出 `APP/MARKER` 的发送与接收；
- `final process balances`：仿真结束时余额（非快照值）；
- `snapshot local states`：各进程记录瞬间的余额；
- `snapshot channel states`：被识别为在途的应用消息；
- `conservation check`：验证 `local_total + channel_total == initial_total`；
- `All checks passed.`：表示本次快照一致性断言通过。

## R13

最小验证清单：
- 运行脚本应无异常退出；
- 快照必须完成（所有进程入边 marker 收齐）；
- 必须记录到 `T1` 且仅记录到 `T1` 这条在途消息；
- 对每条被记录消息检查：
1. `send_step < 发送方记录时刻`；
2. `接收方记录时刻 <= recv_step < 对应 marker 到达时刻`；
- 守恒关系必须成立：`local + channel = initial`。

## R14

当前 demo 关键参数：
- 进程数：`3`（`P0,P1,P2`）；
- 初始总量：`300`；
- 快照发起者：`P0`；
- 快照 ID：`S1`；
- 标记默认延迟：`1`；
- 仅覆盖一条标记延迟：`(P0->P2)=4`（用于制造跨进程记录窗口）；
- 应用消息：`T1: P2->P1, amount=30, delay=4`。

## R15

与相关机制对比：
- 与向量时钟：向量时钟用于判定因果关系，不直接给出“信道在途消息状态”；
- 与全局停机检查点：停机检查点会打断业务，Chandy-Lamport 可在不停机下采样一致状态；
- 与简单“读各节点本地状态”：后者可能形成不一致切面，遗漏/重复统计跨节点消息。

## R16

典型应用场景：
- 分布式系统一致性检查与调试；
- 分布式数据库/流处理中的检查点与恢复元数据采集；
- 稳定性质检测（如死锁检测、终止检测）前置状态抽样；
- 教学场景下演示“全局状态 != 本地状态简单拼接”。

## R17

可扩展方向：
- 支持并发多快照 `snapshot_id`；
- 支持随机拓扑和随机消息流压力测试；
- 引入故障模型（丢包、进程崩溃）并切换到容错检查点协议；
- 将离散时间仿真替换为真实网络协程/线程实现；
- 增加快照序列化输出（JSON）用于回放与可视化。

## R18

`demo.py` 源码级算法流（9 步，非黑箱）：
1. `run_demo` 构造 3 进程、全双向信道、固定消息脚本。  
2. `run_scenario` 在 `t=0` 发送应用消息 `T1`，发送方立即扣减余额并把消息入 FIFO 队列。  
3. `t=1` 调用 `initiate_snapshot`，创建 `SnapshotRuntime` 并让发起者 `P0` 先记录本地状态。  
4. `P0` 对所有出边发送 `MARKER`；后续每步 `deliver_due_messages` 按 `(deliver_step, seq)` 分发消息。  
5. 某进程首次收到 `MARKER` 时，`_record_local_state` 记录本地状态、关闭该 marker 入边、打开其余入边录制，并继续转发 marker。  
6. `APP` 到达时执行 `_receive_app`：先更新接收方本地余额；若该入边仍在 `recording_channels`，则把该消息写入 `channel_state[src,dst]`。  
7. 某进程后续收到同一快照的其他入边 `MARKER` 时，在 `_receive_marker` 中关闭对应录制通道。  
8. `_update_snapshot_completion` 检查所有进程是否都已收到全部入边 marker，满足则标记快照完成。  
9. `validate_snapshot` 对快照做三类断言：在途消息时序条件、记录消息集合（本例仅 `T1`）、守恒关系；通过后由 `print_summary` 输出报告。  
