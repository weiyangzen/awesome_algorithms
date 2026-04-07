# 工作窃取调度

- UID: `CS-0311`
- 学科: `计算机`
- 分类: `并行与分布式`
- 源序号: `472`
- 目标目录: `Algorithms/计算机-并行与分布式-0472-工作窃取调度`

## R01

工作窃取调度（Work-Stealing Scheduling）是一类面向并行任务执行的动态负载均衡策略。  
核心思想是：
- 每个 worker 维护自己的本地任务双端队列（deque）；
- 本地有活时优先执行自己的任务；
- 本地无活时，主动去“偷”其他 worker 的任务；
- 通过“忙者分担、闲者自取”减少全局空转和负载倾斜。

本条目给出一个可运行 Python MVP，展示从“无窃取”到“有窃取”的调度差异。

## R02

为什么它重要：
- 并行程序里任务粒度和耗时常不均匀，静态分配容易导致“有人忙到结束、有人很早空闲”；
- 工作窃取在运行时动态平衡负载，常用于任务并行运行时（如分治递归、任务图执行）；
- 相比中心化全局队列，按 worker 本地队列 + 仅空闲时窃取，通常具备更好的可扩展性。

## R03

本 MVP 解决的问题：
1. 构造一批耗时不均的任务；
2. 把大部分任务倾斜分配给某个 worker（模拟真实热点）；
3. 对比两种策略：
- `no-steal`：worker 只能处理自己队列；
- `work-stealing`：空闲 worker 可从其他 worker 顶端窃取任务；
4. 输出吞吐、总耗时、窃取次数、负载均衡指标并做一致性校验。

## R04

数据结构约定：
- 本地队列：每个 worker 一个线程安全 `deque`；
- `push_bottom` / `pop_bottom`：owner 在底端压入/弹出（LIFO 本地性）；
- `steal_top`：thief 从顶端偷取（与 owner 端分离，减少热点冲突）；
- 任务模型：`Task(task_id, cost_s, home_worker)`，其中 `home_worker` 用于统计任务是否被跨 worker 执行。

## R05

调度流程（简化版）
1. 线程启动后先尝试 `pop_bottom` 本地任务；
2. 若本地为空且允许窃取，则随机选择 victim，执行 `steal_top`；
3. 拿到任务后执行（MVP 用 `time.sleep(cost_s)` 模拟任务耗时）；
4. 记录 worker 统计（执行数、忙时、窃取成功/尝试次数）；
5. 全局剩余任务计数降为 0 时全部线程退出。

## R06

复杂度分析（设任务数 `T`，worker 数 `P`）：
- 总工作量：`O(T)`（每个任务只执行一次）；
- 本地弹出与窃取单次操作平均 `O(1)`；
- 额外开销主要来自：
- 失败窃取尝试；
- 锁竞争；
- 线程调度与上下文切换。

因此工作窃取的收益通常依赖任务不均衡程度和任务粒度。

## R07

正确性不变量：
- 每个任务最多被执行一次；
- 总执行任务数必须等于输入任务数；
- worker 退出条件基于全局剩余任务计数为 0；
- 任一策略下总“工作量秒数”应与任务定义总和一致（允许极小浮点误差）。

`demo.py` 在每轮实验后都包含这些断言。

## R08

与常见调度策略对比：
- 静态分配：实现简单，但负载不均时 makespan 大；
- 全局共享队列：实现直观，但中心队列容易成为并发热点；
- 工作窃取：本地优先 + 空闲窃取，通常在不规则并行中更稳健。

本 MVP 不追求复杂运行时特性（优先级、NUMA、亲和性），而聚焦核心机制可见性。

## R09

实验设置（`main` 中固定）：
- workers: `8`
- tasks: `240`
- 任务耗时分布：`lognormal`（重尾，模拟任务长短不一）
- 初始倾斜：`85%` 任务落到 worker-0
- 随机种子：固定（可复现）

输出两组结果：
- `Baseline(no-steal)`
- `WorkStealing`

并给出 speedup 与均衡指标（CV）。

## R10

`demo.py` 主要组件：
- `Task`：任务定义；
- `WorkStealingDeque`：线程安全本地双端队列；
- `SchedulerStats`：运行统计；
- `generate_task_costs`：生成重尾任务耗时；
- `build_initial_assignment`：按倾斜比例分配任务；
- `run_scheduler`：核心调度循环（可开关窃取）；
- `summarize_result`：用 `pandas` 输出统计表；
- `main`：构造实验、运行对照、打印结论。

## R11

运行方式：

```bash
cd Algorithms/计算机-并行与分布式-0472-工作窃取调度
uv run python demo.py
```

脚本无交互输入，直接完成对照实验并输出结论。

## R12

输出字段解释：
- `makespan_s`：整体完成时间（越小越好）；
- `throughput_task_per_s`：任务吞吐（越大越好）；
- `steal_success` / `steal_attempt`：窃取命中与尝试；
- `stolen_executed`：执行了非本地任务的数量；
- `busy_time_s`：worker 实际忙时；
- `task_count_cv`：各 worker 执行任务数的变异系数（越小越均衡）。

## R13

工程注意点：
- Python 线程受 GIL 影响，不代表 CPU 密集型场景下的极限并行性能；
- 这里用 `sleep` 模拟任务耗时，目的是把焦点放在调度行为而非数值计算；
- 因此该 MVP 用于“机制验证”和“可解释对照”，不是生产性能基准框架。

## R14

常见实现错误：
1. 只做本地队列，不提供窃取通道；
2. 退出条件错误导致线程提前结束或永不结束；
3. 未校验任务执行唯一性，出现重复处理；
4. 把窃取实现为与 owner 同端操作，导致冲突加重。

本实现分别通过：
- 顶端窃取接口；
- 全局剩余计数；
- 执行总数断言；
- owner/thief 分端操作规约 来规避上述问题。

## R15

可扩展方向：
- 引入任务优先级与截止期；
- 窃取策略从随机 victim 升级为负载感知 victim；
- 增加分层队列（本地 + NUMA 域 + 全局）；
- 结合多进程/原生语言后端评估 CPU 密集型真实加速；
- 接入真实分治任务（并行 quicksort / fork-join DAG）。

## R16

相关算法与系统：
- Cilk / Fork-Join 运行时中的工作窃取；
- Chase-Lev deque；
- 分布式场景中的任务拉取（pull-based scheduling）；
- 动态负载均衡（DLB）与抢占式调度对比；
- 图并行或任务图执行器中的任务迁移机制。

## R17

一次典型运行应体现：
- `WorkStealing` 的 `makespan_s` 明显小于 `Baseline(no-steal)`；
- `WorkStealing` 的 `task_count_cv` 更低（负载更均衡）；
- `WorkStealing` 出现正数窃取命中；
- 两种策略都通过完整性断言（执行总任务数一致）。

这表示 MVP 已实现“正确 + 可解释 + 可复现”的最小闭环。

## R18

`demo.py` 源码级算法流（8 步）：
1. `main` 固定随机种子，调用 `generate_task_costs` 生成重尾任务耗时，并打印总工作量。  
2. `build_initial_assignment` 按 `skew_ratio=0.85` 创建初始任务分配，其中大部分任务属于 worker-0。  
3. `run_scheduler(..., allow_steal=False)` 先执行静态基线：每个线程只从自己的 `WorkStealingDeque.pop_bottom` 取活。  
4. `run_scheduler(..., allow_steal=True)` 再执行窃取版本：线程本地无活时，随机选择 victim 调 `steal_top`。  
5. worker 拿到任务后执行 `_execute_task`（`sleep` 模拟耗时），并更新 `SchedulerStats` 中的任务数、忙时、窃取统计。  
6. 每完成一个任务都原子递减全局 `remaining_tasks`，线程循环在其变为 0 时退出，保证终止性。  
7. `run_scheduler` 返回结果后做完整性断言：执行任务总数、处理工作量与任务总量必须一致。  
8. `summarize_result` 用 `pandas` 生成每 worker 统计表，`main` 输出两策略 makespan、吞吐、CV 和 speedup，形成对照结论。 
