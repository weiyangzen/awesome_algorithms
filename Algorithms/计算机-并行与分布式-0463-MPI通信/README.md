# MPI通信

- UID: `CS-0302`
- 学科: `计算机`
- 分类: `并行与分布式`
- 源序号: `463`
- 目标目录: `Algorithms/计算机-并行与分布式-0463-MPI通信`

## R01

MPI 通信（Message Passing Interface Communication）是并行程序最常见的通信模型之一：
- 每个进程有独立地址空间；
- 通过显式消息交换完成数据协作；
- 通信原语通常分为点对点（`send/recv`）和集合通信（`broadcast/scatter/gather/allreduce`）。

本条目给出一个可运行 Python MVP，在单进程环境下模拟 MPI 语义，重点展示通信流程与可验证结果。

## R02

问题定义（MVP 范围）：
- 输入：
1. `world_size=4`；
2. 根进程 `root=0`；
3. 待分发数据向量 `1..16`；
4. 广播配置 `{"scale": 1.5}`；
5. 环形通信令牌 `{0:100,1:200,2:300,3:400}`。
- 输出：
1. 每个 rank 的分块数据和局部求和结果；
2. `gather` 结果（root 收集的局部和）；
3. `allreduce` 结果（所有 rank 一致的全局和）；
4. 环形点对点通信结果；
5. 完整通信操作轨迹表与断言结论。

## R03

本实现覆盖的 MPI 核心通信语义：
- 点对点通信：`send` / `recv`；
- 集合通信：
1. `bcast`：根进程把同一值广播给全部 rank；
2. `scatter`：根进程把数组切块后分发给全部 rank；
3. `gather`：root 从每个 rank 收集一个标量；
4. `allreduce(sum)`：先归约到 root，再广播回所有 rank。

此外加入一次 `ring exchange`，展示多进程邻接通信模式。

## R04

`demo.py` 高层执行流程：
1. 创建 `MiniMPIWorld(world_size=4)`；
2. 广播缩放配置 `scale=1.5`；
3. `scatter` 把长度 16 向量分为 4 块；
4. 各 rank 本地计算 `chunk * scale` 与局部和；
5. `gather` 收集局部和到 root；
6. `allreduce(sum)` 让所有 rank 得到同一个全局和；
7. 进行一轮环形 `send/recv` 令牌交换；
8. 做确定性断言与通信计数校验；
9. 打印通信轨迹和最终结论。

## R05

关键数据结构：
- `CommRecord`：单条通信轨迹，字段包括步骤号、原语、源/目的 rank、tag、payload 摘要、说明；
- `mailboxes: dict[(src,dst,tag), queue]`：模拟消息缓冲队列；
- `records: list[CommRecord]`：通信事件日志；
- `dict[int, np.ndarray]`：`scatter` 后每个 rank 的本地块；
- `dict[int, float]`：每个 rank 的局部和。

## R06

正确性直觉：
- `send/recv` 通过 `(src,dst,tag)` 精确匹配，保证消息不串线；
- `bcast/scatter/gather` 分别对应“同值复制、按 rank 切分、按 rank 汇聚”；
- `allreduce(sum)` 语义是“全局归约 + 全员复制”，每个 rank 应得到相同总和；
- 环形通信中，rank `i` 接收 `(i-1)` 的令牌，验证点对点拓扑通信行为。

## R07

复杂度（设进程数 `P`，总数据长度 `N`）：
- `scatter`：
1. 数据切分 `O(N)`；
2. 消息数 `P-1`；
3. 总传输量 `O(N)`。
- `gather`：消息数 `P-1`，聚合开销 `O(P)`；
- `allreduce(sum)`（本实现采用 `gather + bcast`）：消息数 `2(P-1)`，聚合开销 `O(P)`；
- 环形交换：消息数 `P`。

在本 MVP 的固定参数 `P=4` 下，总通信轨迹条数为 38（19 次 send + 19 次 recv）。

## R08

边界与异常处理：
- `world_size < 2` 或非法 `root`：抛 `ValueError`；
- `recv` 时邮箱无匹配消息：抛 `ValueError`；
- `gather_scalars` / `run_ring_exchange` 若缺失 rank 键：抛 `ValueError`；
- 结束后若邮箱仍有残留消息：断言失败；
- 若 `send`/`recv` 数量不相等：断言失败。

## R09

MVP 设计取舍：
- 选择“语义仿真”而不是依赖 `mpirun`，确保 `uv run python demo.py` 可直接运行；
- 不把第三方 MPI 库当黑箱，通信队列和集合操作都在源码里显式实现；
- 只保留最核心原语，避免把重点分散到进程启动器、集群部署和网络栈细节；
- 使用固定输入与断言，优先可复现性。

## R10

`demo.py` 函数/类职责：
- `CommRecord`：记录通信轨迹；
- `summarize_payload`：把 payload 压缩为可读摘要；
- `MiniMPIWorld.send/recv`：点对点通信基础；
- `MiniMPIWorld.bcast`：广播；
- `MiniMPIWorld.scatter_array`：数组分发；
- `MiniMPIWorld.gather_scalars`：标量收集；
- `MiniMPIWorld.allreduce_sum`：全局求和归约；
- `run_ring_exchange`：环形拓扑点对点交换；
- `print_comm_table`：输出完整通信操作表；
- `main`：构造实验、执行断言、打印结果。

## R11

运行方式：

```bash
cd Algorithms/计算机-并行与分布式-0463-MPI通信
uv run python demo.py
```

脚本无交互输入，运行后直接给出通信轨迹与断言结果。

## R12

输出说明：
- `Local transformed chunks`：每个 rank 的本地块与局部和；
- `Gathered local sums on root`：`gather` 后 root 的聚合结果；
- `Allreduce(global sum)`：每个 rank 的全局和副本；
- `Ring received tokens`：环形通信后各 rank 收到的令牌；
- `Communication Trace`：逐步列出 `send/recv`、`tag` 和 payload 摘要；
- `All checks passed for CS-0302`：全部验证通过。

## R13

最小验证清单：
- `scatter` 后 4 个块精确为 `[1..4]、[5..8]、[9..12]、[13..16]`；
- `gather` 结果必须是 `[15, 39, 63, 87]`；
- `allreduce` 结果必须是 `[204, 204, 204, 204]`；
- 环形接收结果必须是 `{0:400,1:100,2:200,3:300}`；
- 所有消息必须被消费完（邮箱为空）；
- `send_count == recv_count`。

## R14

固定实验参数：
- `world_size = 4`，`root = 0`；
- 输入向量：`np.arange(1, 17)`；
- 广播配置：`scale = 1.5`；
- 局部和预期：
1. rank0: `15`；
2. rank1: `39`；
3. rank2: `63`；
4. rank3: `87`；
- 全局和预期：`204`；
- 环形令牌输入：`100/200/300/400`。

## R15

与相关并行通信方案差异：
- 与共享内存线程模型（如 OpenMP）相比：MPI 语义强调显式消息传递，不共享进程地址空间；
- 与 RPC/微服务调用相比：MPI 原语更偏高性能并行计算通信，不强调服务治理；
- 与真实 `mpi4py + mpirun` 运行相比：本条目是“单进程语义仿真”，用于教学与机制验证，不做真实网络并行加速测试。

## R16

适用场景：
- 学习 MPI 点对点与集合通信语义；
- 在无集群环境下验证通信流程与数据正确性；
- 作为后续接入真实 `mpi4py` 实现的原型。

不适用场景：
- 评估真实多机带宽、延迟、扩展性；
- 需要真实故障模型（进程崩溃、网络抖动、重试）；
- 需要与 HPC 作业调度系统联调。

## R17

可扩展方向：
- 把 `MiniMPIWorld` 替换为 `mpi4py.MPI.COMM_WORLD`，在 `mpirun -n P` 下真实执行；
- 增加 `allgather`、`reduce_scatter`、非阻塞 `isend/irecv`；
- 为集合通信加入树形或环形算法实现并比较消息复杂度；
- 增加随机延迟与乱序仿真，验证鲁棒性与死锁检测；
- 引入多轮迭代（如并行梯度求和）进行端到端实验。

## R18

`demo.py` 源码级算法流（9 步，非黑箱）：
1. `main` 初始化 `MiniMPIWorld(4)`，创建邮箱与通信日志容器。  
2. 调用 `bcast`：root 将 `{"scale": 1.5}` 逐个 `send` 给其他 rank，随后各 rank `recv`。  
3. 调用 `scatter_array`：root 将向量按 rank 切块并逐个发送，其他 rank 接收自己的块。  
4. 每个 rank 在本地做 `chunk * scale` 并计算局部和，形成 `local_sums`。  
5. 调用 `gather_scalars`：非 root 把局部和发给 root，root 汇总成有序向量。  
6. 调用 `allreduce_sum`：先复用 `gather_scalars(tag="REDUCE")` 求总和，再 `bcast` 给全部 rank。  
7. 调用 `run_ring_exchange`：每个 rank 向右邻发送令牌，再从左邻接收，形成一轮环形通信。  
8. 对块内容、`gather`、`allreduce`、环形结果、邮箱空状态、`send/recv` 数量做确定性断言。  
9. 打印通信轨迹表和最终结论，输出 `All checks passed for CS-0302 (MPI通信).`。

本实现没有把第三方库通信调用当不可见黑箱，而是把消息匹配、集合通信和验证过程全部落在可读源码中。
