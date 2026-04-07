# 磁盘调度 - SSTF

- UID: `CS-0189`
- 学科: `计算机`
- 分类: `操作系统`
- 源序号: `336`
- 目标目录: `Algorithms/计算机-操作系统-0336-磁盘调度_-_SSTF`

## R01

问题定义：
给定磁盘柱面请求序列 `requests`、初始磁头位置 `start_head` 与柱面上界 `max_cylinder`，实现 SSTF（Shortest Seek Time First，最短寻道时间优先）调度，输出服务顺序与总寻道距离。

本题聚焦“算法本身可解释实现”，而不是调用系统调度器黑盒。

## R02

输入与输出：

- 输入：
  - `requests: list[int]`，每个请求满足 `0 <= req <= max_cylinder`；
  - `start_head: int`，初始磁头柱面；
  - `max_cylinder: int`，最大柱面号。
- 输出：
  - `service_order`：SSTF 实际服务顺序（包含重复请求）；
  - `moves`：逐步寻道轨迹（起点、终点、距离、剩余请求数）；
  - `total_seek`：总寻道距离；
  - `average_seek_per_request`：平均每请求寻道距离。

`demo.py` 采用固定样例，运行无需交互输入。

## R03

SSTF 核心思想：

- 每一步都选择“当前磁头位置最近”的请求；
- 处理完成后，以新位置继续选择最近请求；
- 直到请求队列为空。

这是一个局部贪心策略，通常能降低平均寻道距离，但不保证全局最优，也可能产生饥饿（远端请求长期等待）。

## R04

关键数据结构：

- `HeadMove`：记录一次磁头移动事件：
  - `step`：第几步；
  - `from_pos / to_pos / distance`；
  - `remaining_after`：该步后剩余请求数。
- `SSTFResult`：封装 SSTF 输出（服务顺序、轨迹、统计量）。
- `FCFSResult`：封装 FCFS 基线输出（用于对照）。
- `pandas.DataFrame`：格式化轨迹与统计摘要。

## R05

`sstf_schedule`（SSTF 主流程）可以拆为：

1. 校验输入范围合法性；
2. 拷贝请求到 `pending`，防止原地修改调用方数据；
3. 在 `pending` 非空时循环：
   - 计算每个候选请求到当前磁头的距离；
   - 选择最小距离请求；
   - 若距离并列，按“柱面号更小者优先”打破并列；
   - 记录移动并从 `pending` 移除该请求的一次出现；
4. 汇总为 `SSTFResult`。

## R06

正确性要点：

- 完整性：`service_order` 与输入请求多重集一致（含重复值）；
- 贪心选择：每一步服务对象都满足“当前时刻最小距离”；
- 距离一致性：`total_seek == sum(move.distance)`；
- 终止性：每轮恰好移除一个请求，有限步后必然结束。

`demo.py` 中使用断言覆盖上述性质。

## R07

复杂度分析：

设请求数为 `n`。

- 每一轮都要扫描剩余请求找最近项：`O(k)`；
- 总时间复杂度：`O(n^2)`；
- 空间复杂度：`O(n)`（存储 `pending`、`service_order`、`moves`）。

说明：可用平衡树/双端结构优化“最近值查询”，但实现复杂度更高，不属于本 MVP 目标。

## R08

边界与异常处理：

- `max_cylinder < 0`：非法；
- `start_head` 越界：非法；
- 任一请求越界：非法；
- 空请求列表：合法，结果总寻道为 0；
- 存在重复柱面：合法，按出现次数分别服务；
- 存在与 `start_head` 相等请求：该步距离为 0。

## R09

MVP 范围：

- 实现单磁头、离线请求集合下的 SSTF 调度；
- 实现 FCFS 基线，方便量化对比；
- 输出可审计逐步轨迹与统计表；
- 使用 `numpy` 生成随机负载，用 `pandas` 做结果汇总。

未覆盖：

- 在线请求到达（动态插队）；
- 设备固件层队列（如 NCQ）；
- 老化机制（anti-starvation）与公平性保障。

## R10

固定样例（经典教材数据）：

- `requests = [98, 183, 37, 122, 14, 124, 65, 67]`
- `start_head = 53`
- `max_cylinder = 199`

在本实现的并列策略下：

- `SSTF total_seek = 236`
- `FCFS total_seek = 640`

## R11

样例行为解释：

- 从 53 开始，最近请求为 65、67，再到 37、14；
- 随后跳向高柱面 98、122、124、183；
- 相比 FCFS 的“按到达顺序大幅来回跳转”，SSTF 通过“就近服务”显著减少总位移。

## R12

指标定义：

- 单步寻道：`|to - from|`
- 总寻道：`sum(step_distance)`
- 平均每请求寻道：`total_seek / len(requests)`
- 随机实验改进量：`improvement = fcfs_seek - sstf_seek`
- 随机实验“更优比例”：`sstf_seek < fcfs_seek` 的试验占比

## R13

与相关算法关系：

- FCFS：公平但可能抖动大；
- SSTF：平均寻道通常更小，但可能饿死远端请求；
- SCAN：引入方向性，通常在吞吐与公平上更稳；
- LOOK/C-SCAN：进一步调整折返行为与等待分布。

SSTF 更像“纯局部贪心”，而 SCAN 家族更偏“轨迹规划”。

## R14

工程化注意事项：

- 若用于生产系统，需叠加等待时间上限/老化策略防饥饿；
- 并列距离策略必须固定（本实现：选择更小柱面）以保证可复现；
- 评估应同时看吞吐、尾延迟与公平性，而非只看平均寻道。

## R15

常见实现错误：

- 把 SSTF 误写成“全局排序后顺序访问”；
- 忽略重复请求，只服务一次；
- 并列最近请求处理不稳定，导致结果不可复现；
- 只输出总距离，不保留逐步轨迹，难以审计；
- 统计总距离时遗漏 0 距离步骤或重复累加。

## R16

最小测试清单：

- 基础混合样例（教材样例）；
- 含重复柱面请求；
- 含与 `start_head` 相等请求；
- 空请求列表；
- 非法输入（越界请求、越界起点、负边界）。

## R17

目录交付内容：

- `README.md`：R01-R18 完整说明；
- `demo.py`：可运行 MVP（SSTF + FCFS + 随机对照）；
- `meta.json`：任务元数据（与本任务保持一致）。

运行方式：

```bash
cd Algorithms/计算机-操作系统-0336-磁盘调度_-_SSTF
uv run python demo.py
```

预期输出包含：

- SSTF/FCFS 轨迹表；
- 总寻道摘要；
- 随机实验平均寻道与更优比例；
- `All assertions passed.`

## R18

源码级算法拆解（对应 `demo.py`，非黑盒）：

1. `main()` 构造固定样例，调用 `sstf_schedule` 与 `fcfs_schedule`，并断言教材样例的总寻道值。
2. `sstf_schedule` 先调用 `_validate_inputs`，逐项检查边界、起点和请求范围是否合法。
3. 初始化 `pending`（待服务列表）与 `current`（当前磁头），进入循环直到 `pending` 为空。
4. 每轮对 `pending` 中每个请求计算 `abs(req - current)`，得到“当前时刻候选距离集合”。
5. 取最小距离作为贪心目标；若多个请求并列，按柱面号更小优先（确定性 tie-break）。
6. 调用 `_record_move` 记录一步寻道事件（step/from/to/distance/remaining_after），并更新 `current`。
7. 从 `pending` 删除本轮被服务请求的一次出现，追加到 `service_order`，继续下一轮。
8. `SSTFResult.total_seek` 由 `moves` 求和得到，`main()` 再用 `run_random_comparison`（`numpy` 采样、`pandas` 汇总）做轻量统计验证。

第三方库使用边界：

- `numpy` 仅用于构造随机测试负载；
- `pandas` 仅用于打印轨迹/统计表；
- 调度决策（最近请求选择、并列打破、轨迹更新）全部在源码中显式实现。
