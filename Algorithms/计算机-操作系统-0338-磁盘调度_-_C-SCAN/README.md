# 磁盘调度 - C-SCAN

- UID: `CS-0191`
- 学科: `计算机`
- 分类: `操作系统`
- 源序号: `338`
- 目标目录: `Algorithms/计算机-操作系统-0338-磁盘调度_-_C-SCAN`

## R01

问题定义：
在磁盘柱面区间 `[0, max_cylinder]` 中，给定请求序列 `requests`、初始磁头位置 `start_head` 与初始方向 `direction`（`left/right`），实现 C-SCAN（Circular SCAN，循环电梯）调度，并输出服务顺序、移动轨迹和寻道统计。

本题要求“可解释源码实现”，而不是调用操作系统调度黑盒。

## R02

输入与输出：

- 输入：
  - `requests: list[int]`，每个请求满足 `0 <= req <= max_cylinder`；
  - `start_head: int`，初始磁头柱面；
  - `max_cylinder: int`，磁盘最大柱面号；
  - `direction: str`，初始方向，取值 `"left"` 或 `"right"`。
- 输出：
  - `service_order`：请求被服务的顺序（含重复请求）；
  - `moves`：逐步移动事件（起点、终点、距离、阶段、目标类型）；
  - `total_seek`：总寻道距离；
  - `average_seek_per_request`：平均每请求寻道距离。

`demo.py` 内置固定样例，运行无需交互输入。

## R03

C-SCAN 核心思想：

- 磁头只在一个方向上服务请求（例如一直向右）；
- 到达边界后，不在回程服务请求，而是“循环跳转”到另一端；
- 然后继续按同一方向服务剩余请求。

相比 SCAN，C-SCAN 的等待时间分布更均匀，代价是可能有额外跨端跳转成本。

## R04

关键数据结构：

- `HeadMove`：单步移动记录：
  - `from_pos / to_pos / distance`；
  - `phase`：`right_sweep`、`left_sweep` 或 `fcfs`；
  - `target_type`：`request`、`boundary` 或 `jump`。
- `CScanResult`：封装 C-SCAN 输出（服务顺序、轨迹、统计量）。
- `FCFSResult`：对照基线结果。
- `pandas.DataFrame`：用于轨迹与统计表输出。

## R05

`cscan_schedule` 主流程：

1. 校验输入合法性（边界、方向、请求范围）；
2. 按 `start_head` 分组：`equal`、`less`、`greater`；
3. 依据方向构建“第一趟服务”和“循环后第二趟服务”；
4. 第一趟按方向服务并记录移动；
5. 若第二趟存在请求：
   - 先到当前方向边界（`boundary` 事件）；
   - 再跨端跳转到另一侧边界（`jump` 事件）；
6. 第二趟继续按同一方向服务请求；
7. 汇总返回 `CScanResult`。

## R06

正确性要点：

- 完整性：`service_order` 与输入请求多重集一致（含重复值）；
- 方向一致性：两趟服务都保持同向扫描语义；
- 距离一致性：`total_seek == sum(move.distance)`；
- 循环语义：`jump` 事件仅在存在另一侧待服务请求时出现。

`demo.py` 通过断言覆盖样例总寻道值、服务覆盖和随机统计基本性质。

## R07

复杂度分析：

设请求数为 `n`。

- 排序开销：`O(n log n)`（左右两侧排序）；
- 扫描服务：`O(n)`；
- 总时间复杂度：`O(n log n)`；
- 空间复杂度：`O(n)`（服务序列与移动轨迹）。

## R08

边界与异常处理：

- `max_cylinder < 0`：非法；
- `start_head` 越界：非法；
- `direction` 非 `left/right`：非法；
- 任一请求越界：非法；
- 空请求列表：合法，总寻道为 0；
- 请求等于 `start_head`：产生 0 距离服务步骤；
- 重复请求：按出现次数分别服务。

## R09

MVP 范围：

- 实现离线请求集合下的 C-SCAN 调度；
- 实现 FCFS 对照基线，便于量化效果；
- 输出逐步事件轨迹（含 `boundary`、`jump`）；
- 用 `numpy` 生成随机请求，用 `pandas` 做统计展示。

未覆盖：

- 在线动态到达请求；
- 多队列优先级与 QoS；
- 多磁臂、固件级优化（如 NCQ）。

## R10

固定样例（`demo.py` 内置）：

- `requests = [176, 79, 34, 60, 92, 11, 41, 114]`
- `start_head = 50`
- `max_cylinder = 199`
- `direction = "right"`

样例断言结果：

- `C-SCAN total_seek = 389`
- `FCFS total_seek = 510`

## R11

样例行为解释：

- 先向右服务：`60 -> 79 -> 92 -> 114 -> 176`；
- 因左侧仍有请求，继续到右边界 `199`；
- 执行循环跳转 `199 -> 0`（不服务请求）；
- 再向右服务左侧请求：`11 -> 34 -> 41`。

这样每个请求都在统一方向下被访问，等待分布更均衡。

## R12

指标定义：

- 单步寻道：`|to - from|`
- 总寻道：`sum(step_distance)`
- 平均每请求寻道：`total_seek / len(requests)`
- 随机实验改进量：`improvement = fcfs_seek - cscan_seek`
- 随机实验更优比例：`cscan_seek < fcfs_seek` 的试验占比

## R13

与相关算法关系：

- FCFS：请求顺序公平，但寻道常有大跳跃；
- SCAN：往返两向都服务，寻道更平滑；
- LOOK：像 SCAN 但不触达边界；
- C-SCAN：只单向服务，跨端跳转后继续单向，等待分布更均匀；
- C-LOOK：C-SCAN 的“非边界版”，跳转到最小/最大待服务请求而非物理边界。

## R14

工程化注意事项：

- 机械盘场景下，C-SCAN 常用于降低请求位置偏置带来的等待不均；
- 需要显式记录 `jump` 事件，便于区分“服务移动”与“循环移动”；
- 评估时应同时关注平均寻道与尾延迟，而不仅是总距离。

## R15

常见实现错误：

- 把 C-SCAN 误写成 SCAN（回程也服务）；
- 忘记在循环时记录跨端 `jump` 距离；
- 误把 `jump` 当作服务步骤写入 `service_order`；
- 对重复请求去重，导致服务次数错误；
- 未做边界校验，出现非法柱面访问。

## R16

最小测试清单：

- 左右两侧都有请求的标准样例；
- 全部请求在同一侧（不应触发 jump）；
- 含多个等于 `start_head` 的请求；
- 空请求列表；
- 非法输入（越界请求、越界起点、非法方向）。

## R17

目录交付内容：

- `README.md`：R01-R18 完整说明；
- `demo.py`：可运行 MVP（C-SCAN + FCFS + 随机实验）；
- `meta.json`：任务元数据，与本任务保持一致。

运行方式：

```bash
cd Algorithms/计算机-操作系统-0338-磁盘调度_-_C-SCAN
uv run python demo.py
```

预期输出包含：

- C-SCAN 与 FCFS 的轨迹表；
- 总寻道与平均寻道摘要；
- 200 次随机对照统计；
- `All assertions passed.`

## R18

源码级算法拆解（对应 `demo.py`，非黑盒）：

1. `main()` 构造固定样例，分别调用 `cscan_schedule` 与 `fcfs_schedule`，并断言样例结果 `C-SCAN=389`、`FCFS=510`。
2. `cscan_schedule` 先执行 `_validate_inputs`，逐项检查方向、起点与请求柱面范围。
3. 请求被分解为 `equal/less/greater`，再根据方向拼出第一趟与第二趟服务序列（例如向右时第二趟为左侧升序）。
4. 第一趟服务逐个调用 `_record_move(..., target_type="request")`，累计服务顺序与移动距离。
5. 若第二趟存在请求，则先记录一次到当前边界的 `boundary` 移动，再记录一次跨端 `jump` 移动。
6. 第二趟继续以同一方向服务剩余请求，依然仅把真实请求写入 `service_order`。
7. `CScanResult.total_seek` 通过所有 `HeadMove.distance` 求和得到，`average_seek_per_request` 在属性中统一计算。
8. `run_random_comparison` 用 `numpy` 生成随机请求与起点，用 `pandas` 汇总 `cscan_seek/fcfs_seek/improvement` 及更优比例，验证实现可稳定批量运行。

第三方库边界：

- `numpy` 仅用于随机测试样本生成；
- `pandas` 仅用于展示轨迹表和统计表；
- C-SCAN 调度逻辑（分组、排序、边界移动、跨端跳转、服务顺序）全部由源码显式实现，无黑盒调度调用。
