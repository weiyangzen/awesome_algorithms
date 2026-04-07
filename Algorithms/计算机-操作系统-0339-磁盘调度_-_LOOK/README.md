# 磁盘调度 - LOOK

- UID: `CS-0192`
- 学科: `计算机`
- 分类: `操作系统`
- 源序号: `339`
- 目标目录: `Algorithms/计算机-操作系统-0339-磁盘调度_-_LOOK`

## R01

问题定义：
在磁盘柱面编号区间 `[0, max_cylinder]` 内，给定请求序列 `requests`、初始磁头位置 `start_head` 与初始方向 `direction`（`left/right`），实现 LOOK 磁盘调度并输出服务顺序、逐步移动轨迹与寻道统计。

本题目标是“可解释源码实现”，不是调用操作系统或第三方调度黑盒。

## R02

输入与输出：

- 输入：
  - `requests: list[int]`，每个请求满足 `0 <= req <= max_cylinder`；
  - `start_head: int`，初始磁头柱面；
  - `max_cylinder: int`，磁盘最大柱面号；
  - `direction: str`，初始方向，取值 `"left"` 或 `"right"`。
- 输出：
  - `service_order`：LOOK 实际服务顺序（含重复请求）；
  - `moves`：逐步移动事件（步号、起点、终点、距离、扫描阶段、剩余请求数）；
  - `total_seek`：总寻道距离；
  - `average_seek_per_request`：平均每请求寻道距离。

`demo.py` 使用固定样例，运行无需交互输入。

## R03

LOOK 核心思想：

- 与 SCAN 一样沿单方向连续服务请求；
- 与 SCAN 不同，LOOK 到达“该方向最后一个待服务请求”就反向；
- 不强制触达物理边界 `0` 或 `max_cylinder`。

因此 LOOK 往往比严格 SCAN 少一次“到边界再折返”的额外位移。

## R04

关键数据结构：

- `HeadMove`：记录一次磁头移动事件：
  - `step`：步骤编号；
  - `from_pos / to_pos / distance`；
  - `phase`：`right_sweep`、`left_sweep` 或 `fcfs`；
  - `remaining_after`：该步后剩余请求数。
- `LookResult`：封装 LOOK 结果（服务顺序、轨迹、总寻道等）。
- `FCFSResult`：封装 FCFS 对照结果（用于基线比较）。
- `pandas.DataFrame`：用于打印轨迹表和统计摘要。

## R05

`look_schedule` 主流程可拆为：

1. 校验输入合法性（边界、起点、方向、请求范围）；
2. 将请求拆分为 `==start`、`<start`、`>start` 三组；
3. 按初始方向构建第一趟与第二趟扫描顺序；
4. 第一趟逐个请求记录 `HeadMove`；
5. 不追加任何边界移动（这是 LOOK 与 SCAN 的关键差异）；
6. 第二趟按反向顺序继续服务；
7. 汇总为 `LookResult` 返回。

## R06

正确性要点：

- 完整性：`service_order` 与输入请求多重集一致（含重复值）；
- 顺序性：每一趟扫描内服务顺序与扫描方向一致；
- 距离一致性：`total_seek == sum(move.distance)`；
- 终止性：每服务一个请求就减少一个待处理请求，有限步结束。

`demo.py` 中使用断言覆盖样例总寻道和请求覆盖性质。

## R07

复杂度分析：

设请求数为 `n`。

- 预处理排序：`O(n log n)`（左右两侧排序）；
- 双趟扫描服务：`O(n)`；
- 总时间复杂度：`O(n log n)`；
- 空间复杂度：`O(n)`（分组、轨迹、服务顺序）。

## R08

边界与异常处理：

- `max_cylinder < 0`：非法；
- `start_head` 越界：非法；
- `direction` 非 `left/right`：非法；
- 任一请求越界：非法；
- 空请求列表：合法，总寻道为 0；
- 请求等于 `start_head`：合法，产生 0 距离服务步骤；
- 重复柱面请求：合法，按出现次数分别服务。

## R09

MVP 范围：

- 实现单磁头、离线请求集合下的 LOOK 调度；
- 提供 FCFS 对照基线，便于量化效果；
- 输出可审计移动轨迹和摘要指标；
- 使用 `numpy` 做随机负载生成，`pandas` 做表格输出。

未覆盖：

- 在线到达请求（动态插入）；
- 设备固件队列（如 NCQ）和多磁臂场景；
- 饥饿抑制、QoS、混合策略切换。

## R10

固定样例（`demo.py` 内置）：

- `requests = [176, 79, 34, 60, 92, 11, 41, 114]`
- `start_head = 50`
- `max_cylinder = 199`
- `direction = "right"`

断言结果：

- `LOOK total_seek = 291`
- `FCFS total_seek = 510`

## R11

样例行为解释：

- 向右扫描阶段服务 `60,79,92,114,176`；
- 到达 176 后右侧已无请求，立即反向；
- 再服务左侧 `41,34,11`；
- 没有 `176 -> 199` 的边界延伸，因此比严格 SCAN 的同样例更短。

## R12

指标定义：

- 单步寻道：`|to - from|`
- 总寻道：`sum(step_distance)`
- 平均每请求寻道：`total_seek / len(requests)`
- 随机实验改进量：`improvement = fcfs_seek - look_seek`
- 随机实验更优比例：`look_seek < fcfs_seek` 的试验占比

## R13

与相关算法关系：

- FCFS：按到达顺序，公平但常有大跳跃；
- SSTF：就近优先，平均寻道低但可能饥饿；
- SCAN：方向性强，但可能包含边界额外移动；
- LOOK：保留方向性，同时去掉不必要边界移动；
- C-SCAN/C-LOOK：更强调等待分布均匀性。

LOOK 常被看作 SCAN 的“边界优化版”。

## R14

工程化注意事项：

- `direction` 需稳定可复现（可由历史方向或队列分布决定）；
- 应记录完整轨迹事件，便于性能归因和回放；
- 评估时不能只看平均寻道，建议同时关注尾延迟和公平性；
- 在机械盘场景，LOOK 通常比 FCFS 更平滑，但仍需结合负载特征调参。

## R15

常见实现错误：

- 把 LOOK 错写成 SCAN（误加边界移动）；
- 忽略 `start_head` 位置请求（漏掉 0 距离步骤）；
- 重复请求被去重，导致服务次数错误；
- 方向排序写反，导致服务顺序不符合扫描语义；
- 只输出总寻道，不输出逐步轨迹，难以审计。

## R16

最小测试清单：

- 经典混合分布样例（左右两侧均有请求）；
- 请求全部在一侧；
- 含多个等于 `start_head` 的请求；
- 空请求列表；
- 非法输入（越界请求、越界起点、非法方向）。

## R17

目录交付内容：

- `README.md`：R01-R18 完整说明；
- `demo.py`：可运行 MVP（LOOK 主算法 + FCFS 对照 + 随机统计）；
- `meta.json`：任务元数据（与本任务保持一致）。

运行方式：

```bash
cd Algorithms/计算机-操作系统-0339-磁盘调度_-_LOOK
uv run python demo.py
```

预期输出包含：

- LOOK 与 FCFS 的移动轨迹表；
- 总寻道与平均寻道摘要；
- 200 次随机对照统计；
- `All assertions passed.`

## R18

源码级算法拆解（对应 `demo.py`，非黑盒）：

1. `main()` 构造固定样例，调用 `look_schedule` 和 `fcfs_schedule`，并用断言校验样例总寻道值（`LOOK=291`、`FCFS=510`）。
2. `look_schedule` 先调用 `_validate_inputs`，验证方向与柱面范围，避免非法访问。
3. 将请求分为 `equal`、`less_desc`、`greater_asc`，并按 `direction` 组装第一趟与第二趟扫描序列。
4. 第一趟逐请求调用 `_record_move`，记录 `step/from/to/distance/phase/remaining_after` 并更新当前磁头位置。
5. LOOK 不执行“到边界再折返”动作，第一趟结束后直接切换到第二趟请求序列。
6. 第二趟继续 `_record_move`，直到全部请求服务完，得到完整 `service_order` 和 `moves`。
7. `LookResult.total_seek` 通过 `sum(m.distance)` 计算，`average_seek_per_request` 由属性统一给出。
8. `run_random_comparison` 使用 `numpy` 生成随机请求并轮换方向，用 `pandas` 汇总 `look_seek/fcfs_seek/improvement`，只做统计展示，不参与调度决策。

第三方库边界：

- `numpy`：仅用于随机测试样本生成；
- `pandas`：仅用于表格化输出和统计均值；
- LOOK 决策流程（分组、排序、方向扫描、反向时机）全部在源码中手写实现。
