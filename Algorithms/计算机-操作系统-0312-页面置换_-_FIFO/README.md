# 页面置换 - FIFO

- UID: `CS-0167`
- 学科: `计算机`
- 分类: `操作系统`
- 源序号: `312`
- 目标目录: `Algorithms/计算机-操作系统-0312-页面置换_-_FIFO`

## R01

FIFO（First-In-First-Out，先进先出）页面置换算法用于虚拟内存管理。

当发生缺页且物理页框已满时，FIFO 总是淘汰“最早进入内存”的页面，而不考虑该页最近是否被访问过。

## R02

问题定义（本条目 MVP）：

- 输入：
  - 页面访问序列 `reference = [p0, p1, ...]`（非负整数）；
  - 页框数 `frame_count > 0`。
- 输出：
  - `hits`（命中次数）；
  - `faults`（缺页次数）；
  - 命中率 / 缺页率；
  - 每一步访问日志（命中/缺页、被替换页、当前页框顺序）。

## R03

FIFO 决策规则：

- 命中：页框状态不变；
- 缺页且有空框：直接装入新页；
- 缺页且无空框：淘汰队头（最早进入页框的页），新页入队尾。

不变量：
- 页框顺序始终表示 `oldest -> newest`；
- 队头永远是下一次替换候选页。

## R04

`demo.py` 的核心数据结构：

- `deque`：维护 FIFO 顺序，队头最老、队尾最新；
- `set`：`O(1)` 判断页面是否命中；
- `events`：逐步日志，记录 `step/page/action/replaced/frames_oldest_to_newest`；
- `ReplacementResult`：统一封装统计结果。

## R05

主流程（FIFO 部分）：

1. 校验输入合法性；
2. 初始化空 `queue` 和空 `resident` 集合；
3. 顺序扫描 `reference`；
4. 若命中：`hits += 1`；
5. 若缺页：`faults += 1`，并根据“是否满框”执行填充或替换；
6. 每步记录快照日志；
7. 扫描结束后返回统计结果与事件轨迹。

## R06

正确性直觉：

- 每次新页进入时都被放到队尾；
- 只有缺页且满框时才弹出队头；
- 因此被淘汰页一定是“驻留时间最长”的页。

该行为与 FIFO 的定义完全一致，不依赖额外启发式。

## R07

复杂度分析（访问长度 `n`，页框数 `m`）：

- 时间复杂度：
  - 单步命中检测：集合查找均摊 `O(1)`；
  - 单步替换：`popleft/append` 为 `O(1)`；
  - 总体均摊 `O(n)`。
- 空间复杂度：
  - `queue + resident` 为 `O(m)`；
  - 若保留全量日志 `events`，额外 `O(n)`。

## R08

固定样例（`demo.py` 默认输入）：

- `reference = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2]`
- `frame_count = 3`

运行结果：

- FIFO：`faults = 10`, `hits = 3`
- LRU 基线：`faults = 9`, `hits = 4`

这体现了 FIFO 可能落后于利用局部性的策略。

## R09

边界与异常处理：

- `frame_count <= 0`：抛出 `ValueError`；
- `reference` 中出现负页号或非整数：抛出 `ValueError`；
- 空访问序列：命中和缺页都为 0；
- `frame_count = 1`：除连续重复访问外，几乎每次都是缺页。

## R10

MVP 范围：

- 重点实现 FIFO 页面置换本身；
- 附带一个手写 `LRU-baseline` 仅用于对照统计；
- 不模拟页表/TLB/脏页回写等内核细节；
- 不调用任何页面置换黑盒库。

## R11

运行方式（无交互输入）：

```bash
uv run python Algorithms/计算机-操作系统-0312-页面置换_-_FIFO/demo.py
```

脚本会自动打印：
- 算法汇总表；
- FIFO 前 10 步事件轨迹；
- Belady 异常示例与随机扫描统计；
- 断言通过提示。

## R12

输出字段说明：

- `action`：`hit` / `fault_fill` / `fault_replace`；
- `replaced`：当发生替换时被淘汰页，否则为 `None`；
- `frames_oldest_to_newest`：页框 FIFO 顺序快照；
- `fault_ratio`：`faults / len(reference)`。

`frames_oldest_to_newest` 是审计 FIFO 决策正确性的关键字段。

## R13

与其他策略对比：

- 相比 LRU：
  - FIFO 不关心“最近性”，实现更简单；
  - 但在时间局部性明显时通常更容易缺页。
- 相比 OPT：
  - FIFO 无需未来信息，能在线执行；
  - OPT 仅用于理论下界，不可直接落地。

## R14

实现中的常见错误：

- 误把命中页挪到队尾，代码会退化成接近 LRU；
- 替换后忘记同步更新 `resident` 集合，导致命中判断污染；
- 记录日志时没保留“老到新”的顺序，难以人工复核；
- 统计不闭合（`hits + faults != total`）。

`demo.py` 使用断言持续检查关键一致性。

## R15

FIFO 的经典性质：Belady 异常。

`demo.py` 内置经典序列：

- `reference = [1,2,3,4,1,2,5,1,2,3,4,5]`

结果：

- 3 个页框时 `faults = 9`
- 4 个页框时 `faults = 10`

即“页框更多却缺页更多”，这是 FIFO 的代表性缺陷。

## R16

最小测试清单：

- 教材标准序列回归（验证固定缺页数）；
- 全重复序列（验证高命中）；
- 严格递增序列（验证持续替换路径）；
- `frame_count=1` 极端场景；
- 非法输入（负页号、非整数页号、非法页框数）。

## R17

目录交付物：

- `README.md`：R01-R18 说明；
- `demo.py`：FIFO 最小可运行实现（含断言和对照输出）；
- `meta.json`：任务元数据（UID/学科/分类/源序号/路径）。

本条目已满足“可运行、可审计、可复现”的最小交付要求。

## R18

`demo.py` 的 FIFO 源码级流程（非黑盒）可拆为 8 步：

1. `main()` 构造固定 `reference` 与 `frame_count`，调用 `fifo_page_replacement`。  
2. `fifo_page_replacement` 初始化 `queue`（FIFO 队列）和 `resident`（驻留页集合）。  
3. 每步先检查 `page in resident`，命中则仅计数，不改变 FIFO 队列顺序。  
4. 若缺页且队列未满，直接 `append(page)` 完成装入（`fault_fill`）。  
5. 若缺页且队列已满，执行 `popleft()` 淘汰最老页，再 `append(page)`（`fault_replace`）。  
6. 每一步把 `action/replaced/frames_oldest_to_newest` 写入 `events`，保证可追踪。  
7. 全序列结束后返回 `hits/faults/evictions/fault_ratio` 等统计结果。  
8. `main()` 再执行 Belady 异常检查与随机扫描，并打印表格化结论。

第三方库边界：
- `numpy` 仅用于随机访问序列生成；
- `pandas` 仅用于汇总表与事件展示；
- 页面置换决策（命中判定、队头淘汰、队尾入队）全部为源码手写实现。
