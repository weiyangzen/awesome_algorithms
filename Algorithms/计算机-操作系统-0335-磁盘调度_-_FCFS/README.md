# 磁盘调度 - FCFS

- UID: `CS-0188`
- 学科: `计算机`
- 分类: `操作系统`
- 源序号: `335`
- 目标目录: `Algorithms/计算机-操作系统-0335-磁盘调度_-_FCFS`

## R01

问题定义：
给定磁盘柱面请求序列 `requests`、初始磁头位置 `start_head` 与柱面上界 `max_cylinder`，实现 FCFS（First-Come, First-Served，先来先服务）磁盘调度，输出服务顺序、移动轨迹和总寻道距离。

本题目标是“源码可解释的调度实现”，不是调用操作系统调度黑盒。

## R02

输入与输出：

- 输入：
  - `requests: list[int]`，每个请求满足 `0 <= req <= max_cylinder`；
  - `start_head: int`，初始磁头位置；
  - `max_cylinder: int`，最大柱面号。
- 输出：
  - `service_order`：实际服务顺序（FCFS 下应与输入顺序一致）；
  - `moves`：逐步移动轨迹（步号、起点、终点、距离、剩余请求数）；
  - `total_seek`：总寻道距离；
  - `average_seek_per_request`：平均每请求寻道距离。

`demo.py` 使用固定样例，运行无需交互输入。

## R03

FCFS 核心思想：

- 请求按到达顺序进入队列；
- 磁头严格按队列顺序依次服务；
- 每一步只移动到“下一个到达请求”，不进行重排优化。

FCFS 的优点是简单、公平、可预测；缺点是可能出现长距离来回跳转，导致平均寻道偏高。

## R04

关键数据结构：

- `HeadMove`：记录单步移动事件：
  - `step`：步骤编号；
  - `from_pos / to_pos / distance`；
  - `remaining_after`：该步完成后剩余请求数。
- `FCFSResult`：封装 FCFS 输出（服务顺序、轨迹、统计量）。
- `SSTFResult`：作为可解释对照基线（非本题主算法）。
- `pandas.DataFrame`：用于打印轨迹表和摘要表。

## R05

`fcfs_schedule`（FCFS 主流程）可拆为：

1. 校验输入参数合法性；
2. 初始化 `current = start_head`；
3. 按 `requests` 原始顺序遍历；
4. 对每个请求计算 `abs(req - current)` 作为该步距离；
5. 记录 `HeadMove`，更新 `current = req`；
6. 追加 `service_order`；
7. 汇总并返回 `FCFSResult`。

## R06

正确性要点：

- 顺序性：`service_order == requests`（完全一致）；
- 完整性：每个请求都被且仅被服务一次（含重复值）；
- 距离一致性：`total_seek == sum(move.distance)`；
- 终止性：遍历长度为 `n` 的请求序列后必然结束。

`demo.py` 中包含断言覆盖上述关键性质。

## R07

复杂度分析：

设请求数为 `n`。

- 时间复杂度：`O(n)`（单次线性遍历）；
- 空间复杂度：`O(n)`（保存 `service_order` 和 `moves`）。

这是磁盘调度里实现复杂度最低的基线方法之一。

## R08

边界与异常处理：

- `max_cylinder < 0`：非法；
- `start_head` 越界：非法；
- 任一请求越界：非法；
- 空请求列表：合法，总寻道为 0；
- 重复请求：合法，按出现次数逐个服务；
- 请求值等于当前磁头：该步距离为 0。

## R09

MVP 范围：

- 实现离线请求集合下的 FCFS 调度；
- 输出可审计逐步轨迹；
- 提供 SSTF 对照函数用于结果参照；
- 使用 `numpy` 生成随机负载，`pandas` 输出统计。

未覆盖：

- 在线请求动态到达；
- 电梯类策略（SCAN/LOOK/C-SCAN）切换；
- 设备固件层队列优化（如 NCQ）。

## R10

固定样例（教材常见数据）：

- `requests = [98, 183, 37, 122, 14, 124, 65, 67]`
- `start_head = 53`
- `max_cylinder = 199`

本实现中断言结果：

- `FCFS total_seek = 640`
- `SSTF total_seek = 236`（仅作对照）

## R11

样例行为解释：

- FCFS 按输入顺序依次走：`53 -> 98 -> 183 -> 37 -> 122 -> 14 -> 124 -> 65 -> 67`；
- 该轨迹存在多次大跨度折返（如 `183 -> 37`、`122 -> 14`）；
- 因此总寻道明显高于按“就近优先”的 SSTF。

这说明 FCFS 公平但不追求寻道最小化。

## R12

指标定义：

- 单步寻道：`|to - from|`
- 总寻道：`sum(step_distance)`
- 平均每请求寻道：`total_seek / len(requests)`
- 随机实验差值：`fcfs_minus_sstf = fcfs_seek - sstf_seek`
- 随机实验劣势比例：`fcfs_seek > sstf_seek` 的试验占比

## R13

与相关算法关系：

- FCFS：最简单、公平性直观，但可能抖动大；
- SSTF：局部贪心，平均寻道通常更小；
- SCAN/LOOK：通过方向约束减少抖动并改善等待分布。

实践中，FCFS 常作为“基线算法”用于评估更复杂调度策略的收益。

## R14

工程化注意事项：

- 在生产系统中，FCFS 作为默认策略通常仅适用于低负载或对公平顺序敏感场景；
- 若目标是降低机械寻道，应结合 SSTF/SCAN 类策略；
- 日志中应保留逐步移动明细，便于回放与性能归因。

## R15

常见实现错误：

- 把 FCFS 错写成排序后服务（破坏到达顺序）；
- 漏掉重复请求，导致请求数不一致；
- 未做边界校验，出现非法柱面访问；
- 只输出总距离，不记录过程轨迹；
- 忽略 0 距离步骤，导致统计偏差。

## R16

最小测试清单：

- 教材固定样例（可断言总寻道）；
- 空请求列表；
- 含重复请求与零距离请求；
- 越界请求与越界起始位置；
- 随机请求批量统计（仅做弱性质校验）。

## R17

目录交付内容：

- `README.md`：R01-R18 完整说明；
- `demo.py`：可运行 MVP（FCFS 主算法 + SSTF 对照 + 随机实验）；
- `meta.json`：任务元数据（与本任务一致）。

运行方式：

```bash
cd Algorithms/计算机-操作系统-0335-磁盘调度_-_FCFS
uv run python demo.py
```

预期输出包含：

- FCFS 与 SSTF 的轨迹表；
- 总寻道摘要；
- 200 次随机实验平均寻道与劣势比例；
- `All assertions passed.`

## R18

源码级算法拆解（对应 `demo.py`，非黑盒）：

1. `main()` 构造固定样例，请求 `fcfs_schedule` 得到 FCFS 服务顺序和逐步移动轨迹。
2. `fcfs_schedule` 先调用 `_validate_inputs`，逐项检查边界、起点和请求柱面是否合法。
3. 进入线性遍历：按原始 `requests` 顺序逐个取请求，不做重排。
4. 每一步调用 `_record_move` 计算 `abs(target - current)`，记录 `HeadMove`，并更新当前磁头位置。
5. 同步维护 `service_order`，因此可直接验证 `service_order == requests`。
6. `FCFSResult.total_seek` 通过对所有 `move.distance` 求和得到总寻道，`average_seek_per_request` 计算平均开销。
7. `main()` 用确定性断言校验固定样例结果（`FCFS=640`），并用 `sstf_schedule` 生成可解释对照。
8. `run_random_comparison` 用 `numpy` 生成随机请求，用 `pandas` 汇总 `fcfs_seek/sstf_seek` 和劣势比例，验证实现在批量输入下稳定运行。

第三方库边界：

- `numpy` 仅用于随机样本生成；
- `pandas` 仅用于表格化展示与统计；
- FCFS 与 SSTF 调度逻辑全部由源码显式实现，无调度黑盒调用。
