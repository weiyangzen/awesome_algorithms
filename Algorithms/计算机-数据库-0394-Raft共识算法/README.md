# Raft共识算法

- UID: `CS-0240`
- 学科: `计算机`
- 分类: `数据库`
- 源序号: `394`
- 目标目录: `Algorithms/计算机-数据库-0394-Raft共识算法`

## R01

Raft 共识算法用于在分布式副本集里实现“多数派一致提交”：
- 在任意时刻最多只有一个 leader 负责接收客户端写入；
- 写入先进入 leader 日志，再复制到 followers；
- 只有当日志条目被多数节点确认后，才会被提交并应用到状态机。

本目录的 MVP 采用“单进程 + 离散时间 tick”方式实现 Raft 核心路径：
- 选主（`RequestVote`）；
- 日志复制与心跳（`AppendEntries`）；
- 提交与状态机应用；
- leader 崩溃与恢复后的追赶同步。

## R02

问题定义（本实现）：
- 输入：
  - 固定 5 节点集群；
  - 一组客户端命令（示例：`set x 1`、`add y -3`）。
- 输出：
  - 经过 Raft 提交后的一致日志前缀；
  - 每个存活节点一致的状态机快照。

约束与目标：
- 不依赖外部数据库和网络框架；
- 代码中显式展开 Raft RPC 逻辑，不调用黑盒共识库；
- `uv run python demo.py` 可一次跑完，无交互输入。

## R03

Raft 关键术语（对应 `demo.py`）：
- `term`：任期号，单调递增；
- `role`：`follower | candidate | leader`；
- `log`：日志条目数组，元素为 `LogEntry(term, command)`；
- `commit_index`：当前节点已提交日志的最后索引；
- `last_applied`：已应用到状态机的最后索引；
- `next_index`：leader 视角下每个 follower 下一次应发送的日志索引；
- `match_index`：leader 视角下每个 follower 已复制成功的最大索引。

核心不变式（MVP 中通过断言与流程保证）：
- 任期更大的节点信息会压制旧任期 leader；
- 已提交日志在多数派上保持前缀一致；
- 活跃节点状态机结果保持一致。

## R04

高层流程：
1. 所有节点初始为 follower，等待随机选举超时。  
2. 超时节点转 candidate，递增任期并发起投票请求。  
3. 票数达到多数后成为 leader，初始化复制游标。  
4. leader 接收客户端命令，追加本地日志。  
5. leader 向 followers 发送 `AppendEntries` 复制日志。  
6. leader 统计 `match_index`，多数派达成后推进 `commit_index`。  
7. 各节点按提交索引顺序应用命令到状态机。  
8. 若 leader 崩溃，其他节点重新选主；恢复节点通过心跳追赶。

## R05

核心数据结构：
- `LogEntry(term, command)`：不可变日志条目。  
- `Node`：单节点状态容器，包含任期、投票、日志、提交位点、状态机等。  
- `RaftCluster`：集群编排器，包含：
  - 节点集合 `nodes`；
  - 当前 leader 标识 `leader_id`；
  - 复制进度 `next_index`、`match_index`；
  - 离散时间 `time_tick` 与事件日志 `events`。

## R06

正确性要点（非形式化）：
- 选主安全：候选者必须拿到多数票才可成为 leader。  
- 日志一致性：`AppendEntries` 先校验 `prev_log_index/term`，再追加新条目。  
- 冲突修复：leader 复制失败时回退 `next_index`，直到找到共同前缀。  
- 提交规则：仅当“当前任期”的日志在多数节点复制成功，leader 才提交。  
- 状态机安全：只按 `commit_index` 顺序应用，避免未提交条目污染状态机。

## R07

复杂度分析（`N` 节点，单条命令日志长度为 `L`）：
- 选主一次投票广播：`O(N)` RPC（本实现为函数调用）。  
- 单条命令复制：理想情况下 `O(N)`；发生回退时最坏 `O(N * L)`。  
- 状态机应用：按新增已提交条目线性推进，`O(k)`（`k` 为本轮新增提交数）。  
- 空间复杂度：每节点日志 `O(L)`，集群总计 `O(N * L)`。

## R08

边界与异常处理：
- 节点数不是奇数或小于 3：初始化时报错。  
- 心跳间隔非正：初始化时报错。  
- 无 leader 时提交客户端命令：抛 `RuntimeError`。  
- 非法命令格式或未知操作符：状态机应用时报错。  
- 指定 tick 内无法选出 leader：抛 `RuntimeError`。

此外，节点有 `active` 标志，模拟崩溃与恢复，不走交互输入。

## R09

MVP 取舍说明：
- 保留：Raft 最核心的选主、复制、提交、故障恢复机制。  
- 简化：
  - 单进程模拟网络，不含真实 RPC/序列化；
  - 不实现日志快照与成员变更；
  - 不实现持久化存储（仅内存）；
  - 统一命令语法为 `set/add key value`。

该取舍让代码更短，但仍能直观看到共识关键路径。

## R10

`demo.py` 主要函数职责：
- `_reset_election_timer`：设置随机选举超时。  
- `request_vote`：处理投票请求与日志新旧比较。  
- `start_election`：候选者发起选主流程。  
- `append_entries`：处理心跳/日志复制请求。  
- `_replicate_to_follower`：leader 侧复制与回退重试。  
- `_advance_commit_index`：多数派提交推进。  
- `_apply_entries`：把已提交日志应用到状态机。  
- `tick`：驱动离散时间和选举/心跳。  
- `client_submit`：客户端写入入口。  
- `assert_safety`：验证活跃节点日志前缀与状态机一致。

## R11

运行方式：

```bash
cd Algorithms/计算机-数据库-0394-Raft共识算法
uv run python demo.py
```

脚本会自动执行：初始选主 -> 写入 -> leader 崩溃 -> 新 leader 写入 -> 节点恢复 -> 一致性校验。

## R12

输出解读：
- `Cluster Summary`：
  - `leader_id`：结束时 leader 编号；
  - `node=... role=... term=...`：节点角色与任期；
  - `log_len/commit_index/last_applied`：复制与应用进度；
  - `state_machine`：节点键值状态。
- `Recent Events`：
  - 选主事件（谁在何时成为 leader）；
  - 客户端命令复制/提交过程；
  - 崩溃与恢复事件。
- 末行 `All assertions passed...`：表示内置一致性断言全部通过。

## R13

内置测试场景（`main` 已覆盖）：
1. 5 节点冷启动后自动选主。  
2. 第一任 leader 连续提交三条命令。  
3. 第一任 leader 崩溃，触发新一轮选主。  
4. 第二任 leader 继续处理写入。  
5. 崩溃节点恢复并通过心跳追赶。  
6. 最终校验活跃节点一致日志前缀与一致状态机。

建议补充测试：
- 两个 follower 同时宕机（多数派不足）时提交应失败；
- 构造日志冲突后验证回退复制；
- 添加随机压力命令并统计收敛时间。

## R14

可调参数：
- `seed`：随机超时种子，控制选主可复现性。  
- `heartbeat_interval`：心跳周期，影响 failover 速度。  
- `run_until_leader(max_ticks)`：选主等待上限。  
- `print_recent_events(keep_last)`：输出事件条数。

调参建议：
- 需要更快恢复：适当减小超时窗口或心跳间隔；
- 需要更稳定日志：保留较大的随机超时离散度，减少平票概率。

## R15

与相关协议对比：
- 对比 2PC：Raft 解决“复制一致性 + leader 故障恢复”，2PC 主要解决分布式事务原子提交。  
- 对比 Paxos：Raft 在工程表达上更强调分角色与日志复制流程，可读性更高。  
- 对比单主异步复制：Raft 明确多数派提交语义，故障切换时更容易保持一致性边界。

## R16

典型应用场景：
- 分布式 KV 存储元数据管理；
- 数据库主从自动选主与配置管理；
- 服务发现、任务调度等控制平面的一致状态复制。

这些场景共同点：
- 写入量中等；
- 强调一致性与故障恢复；
- 可以接受多数派可用性模型。

## R17

可扩展方向：
- 增加持久化（term、vote、log）与重启恢复；  
- 增加快照与日志截断（snapshot/install snapshot）；  
- 增加成员变更（joint consensus）；  
- 注入网络分区、延迟、丢包做更真实仿真；  
- 输出指标（选主耗时、复制延迟、提交吞吐）用于实验分析。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 初始化 `RaftCluster(5 nodes)`，每个节点随机选举超时，初始均为 follower。  
2. `run_until_leader -> tick` 驱动时间，超时节点调用 `start_election` 发起 `RequestVote`，拿到多数票后成为 leader。  
3. leader 初始化 `next_index/match_index`，并通过 `send_heartbeats` 向 followers 建立当前任期心跳。  
4. 客户端调用 `client_submit(command)`，leader 先把命令追加到本地 `log`。  
5. leader 对每个 follower 调用 `_replicate_to_follower`，内部执行 `append_entries`；若一致性校验失败则递减 `next_index` 重试，直到找到共同前缀并完成复制。  
6. leader 执行 `_advance_commit_index`：当某条“当前任期”日志在 `match_index` 统计中达到多数，推进 `commit_index` 并 `_apply_entries` 到状态机。  
7. 若当前 leader 崩溃，`crash_node` 使其失活；剩余节点继续 `tick`，再次通过投票选出新 leader，并继续处理写入。  
8. 恢复节点通过 `recover_node` 回到 follower，新 leader 心跳/复制使其追上提交进度；`assert_safety` 最终验证活跃节点的提交日志前缀和状态机快照一致。

本实现没有把第三方共识库当黑盒，选主、日志一致性检查、提交推进、故障恢复路径都在源码中可逐步追踪。
