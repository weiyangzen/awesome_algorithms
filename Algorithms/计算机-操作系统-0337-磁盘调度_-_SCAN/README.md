# 磁盘调度 - SCAN

- UID: `CS-0190`
- 学科: `计算机`
- 分类: `操作系统`
- 源序号: `337`
- 目标目录: `Algorithms/计算机-操作系统-0337-磁盘调度_-_SCAN`

## R01

问题定义：
在磁盘柱面编号区间 `[0, max_cylinder]` 内，给定初始磁头位置 `start_head`、一批 I/O 请求柱面 `requests`，以及初始移动方向（`left` 或 `right`），使用 SCAN（电梯）调度策略决定服务顺序，并计算总寻道距离。

本题关注的是“调度策略本身”的可解释实现，而不是调用现成系统 API 黑盒执行。

## R02

输入与输出：

- 输入：
  - `requests: list[int]`，每个请求柱面满足 `0 <= req <= max_cylinder`；
  - `start_head: int`，初始磁头位置；
  - `max_cylinder: int`，磁盘最大柱面号；
  - `direction: str`，初始方向，取值 `"left"` 或 `"right"`。
- 输出：
  - `service_order`：请求被服务的实际顺序；
  - `moves`：逐步移动记录（起点、终点、距离、阶段、目标类型）；
  - `total_seek`：总寻道距离；
  - `average_seek_per_request`：平均每请求寻道距离。

`demo.py` 内置固定样例，可直接运行，无需交互输入。

## R03

SCAN 核心思想（电梯模型）：

- 磁头像电梯一样沿一个方向连续移动并服务沿途请求；
- 当该方向请求处理完后，若另一侧仍有请求，则继续移动到磁盘边界（0 或 `max_cylinder`）再反向；
- 反向后服务另一侧请求。

本实现采用“严格 SCAN（非 LOOK）”：
只有在确实需要反向处理另一侧请求时，才触达边界后再折返。

## R04

关键数据结构：

- `HeadMove`：单步移动事件，字段包括：
  - `from_pos / to_pos / distance`
  - `phase`（`right_sweep` / `left_sweep` / `fcfs`）
  - `target_type`（`request` 或 `boundary`）
- `ScanResult`：封装 SCAN 结果（服务顺序、移动轨迹、总寻道等）
- `FCFSResult`：封装 FCFS 对照结果
- `pandas.DataFrame`：用于打印轨迹表和结果摘要表

## R05

`scan_schedule` 高层流程：

1. 校验参数合法性（方向、柱面范围、起始位置）；
2. 按 `start_head` 将请求分成三组：`==start`、`<start`、`>start`；
3. 按初始方向构建“第一趟扫描”和“第二趟扫描”的顺序；
4. 第一趟逐个服务请求并记录移动；
5. 若第二趟存在请求，则先移动到对应边界（0 或 `max_cylinder`）；
6. 反向服务第二趟请求并记录移动；
7. 返回 `ScanResult`。

## R06

正确性要点：

- 完整性：`service_order` 必须覆盖全部请求（含重复值）。
- 顺序性：第一趟与第二趟均按扫描方向单调服务（除 `==start` 的零距离服务）。
- 距离一致性：`total_seek` 等于所有 `HeadMove.distance` 之和。
- 边界行为：仅当存在反向需求时才追加 `boundary` 移动。

`demo.py` 中通过断言验证了关键性质。

## R07

复杂度分析：

设请求数量为 `n`。

- 预处理排序：`O(n log n)`（对左右两侧请求排序）
- 扫描服务：`O(n)`
- 总时间复杂度：`O(n log n)`
- 空间复杂度：`O(n)`（保存分组、服务顺序和移动轨迹）

## R08

边界与异常处理：

- `max_cylinder < 0`：非法；
- `start_head` 不在 `[0, max_cylinder]`：非法；
- `direction` 非 `left/right`：非法；
- 任一请求越界：非法；
- 空请求列表：允许，输出总寻道为 0；
- 存在与 `start_head` 相等的请求：以 0 距离服务并保留次数。

## R09

MVP 范围：

- 实现单磁头、单批请求的离线调度；
- 实现 SCAN 主算法 + FCFS 对照基线；
- 输出可审计的逐步移动轨迹；
- 用 `numpy` 做轻量随机负载实验，用 `pandas` 展示结果。

未覆盖：
- 实时到达队列（在线调度）；
- 多磁臂/NCQ 等设备级优化；
- C-SCAN / N-step SCAN 等完整家族实现。

## R10

`demo.py` 固定样例：

- `requests = [176, 79, 34, 60, 92, 11, 41, 114]`
- `start_head = 50`
- `max_cylinder = 199`
- `direction = "right"`

该样例下断言结果：

- `SCAN total_seek = 337`
- `FCFS total_seek = 510`

## R11

样例现象解释：

- SCAN 首先向右处理 `60,79,92,114,176`；
- 因左侧仍有请求，继续到边界 `199` 后反向；
- 再处理左侧 `41,34,11`；
- 相比 FCFS 的“请求顺序跳跃”，SCAN 通过方向一致的批处理降低了寻道震荡。

## R12

指标定义：

- 单步寻道：`|to - from|`
- 总寻道：`sum(step_distance)`
- 平均每请求寻道：`total_seek / len(requests)`
- 随机实验改进量：`improvement = fcfs_seek - scan_seek`
- 随机实验“更优比例”：`scan_seek < fcfs_seek` 的试验占比

## R13

与相关算法关系：

- FCFS：实现简单，但可能频繁大跨度来回跳转；
- SCAN：有方向约束，通常减少机械臂抖动；
- LOOK：与 SCAN 类似，但不必触达物理边界；
- C-SCAN：单向服务，回程不服务，等待分布更均匀。

本实现特意把 `boundary` 作为显式事件记录，便于和 LOOK 做行为差异对比。

## R14

工程参数建议：

- `direction` 可以按历史趋势、当前队列分布或上次停留方向确定；
- `max_cylinder` 要与设备真实地址空间一致；
- 若工作负载长期偏向一侧，严格 SCAN 的边界折返可能引入额外开销，可评估 LOOK/C-SCAN。

## R15

常见实现错误：

- 将 SCAN 写成 LOOK（遗漏边界触达步骤）；
- 把 `start_head` 位置请求漏掉或重复处理；
- 忘记处理重复柱面请求，导致计数不一致；
- 只统计请求间距离，遗漏边界移动距离；
- 总寻道与移动明细不一致（缺少可追溯事件日志）。

## R16

最小测试清单：

- 正常混合分布（左右两侧都有请求）；
- 请求全部在同一侧；
- 包含多个等于 `start_head` 的请求；
- 空请求列表；
- 非法输入（越界请求、非法方向、非法边界）。

## R17

目录交付内容：

- `README.md`：R01-R18 说明文档；
- `demo.py`：可运行 MVP（SCAN + FCFS + 随机实验）；
- `meta.json`：任务元数据。

运行方式：

```bash
cd Algorithms/计算机-操作系统-0337-磁盘调度_-_SCAN
uv run python demo.py
```

预期输出包含：

- SCAN/FCFS 的逐步轨迹表；
- 总寻道对比摘要；
- 200 次随机实验的平均寻道统计；
- `All assertions passed.`

## R18

源码级算法拆解（对应 `demo.py`，非黑盒）：

1. `main()` 构造固定输入并调用 `scan_schedule` 与 `fcfs_schedule`。
2. `scan_schedule` 先用 `_validate_inputs` 做范围与方向检查，确保柱面访问合法。
3. 将请求分解为 `equal / less_desc / greater_asc` 三组，并按初始方向拼装“第一趟/第二趟”顺序。
4. 第一趟逐个请求调用 `_record_move` 记录事件：起点、终点、距离、扫描阶段、目标类型。
5. 若第二趟存在请求，则追加一次 `target_type="boundary"` 的边界移动，体现严格 SCAN 的折返条件。
6. 第二趟按反向顺序继续 `_record_move`，最终得到完整 `service_order` 与 `moves`。
7. `ScanResult.total_seek` 通过对 `moves.distance` 求和得到总寻道；`average_seek_per_request` 由属性计算。
8. `main()` 用固定断言校验样例数值，再用 `run_random_comparison`（`numpy` 生成随机请求，`pandas` 统计展示）做轻量对照分析。

第三方库边界：

- `numpy` 仅用于随机样本生成；
- `pandas` 仅用于表格化输出和均值统计；
- SCAN/FCFS 的调度决策与移动路径完全由源码手写实现，没有使用任何调度黑盒函数。
