# 页面置换 - LFU

- UID: `CS-0169`
- 学科: `计算机`
- 分类: `操作系统`
- 源序号: `314`
- 目标目录: `Algorithms/计算机-操作系统-0314-页面置换_-_LFU`

## R01

LFU（Least Frequently Used，最不经常使用）页面置换算法用于在缺页时选择淘汰页。

核心思想：
- 记录每个驻留页的访问频次；
- 发生缺页且页框已满时，优先淘汰访问频次最小的页；
- 若频次相同，需要额外平局规则（本实现采用 LRU 平局）。

本题目标是给出可运行、可审计的 LFU 页面置换 MVP，而不是调用黑盒缓存库。

## R02

输入与输出定义：

- 输入：
  - 页面访问序列 `reference = [p0, p1, ...]`（非负整数）；
  - 页框数 `frame_count > 0`。
- 输出：
  - `hits`（命中次数）；
  - `faults`（缺页次数）；
  - 命中率 / 缺页率；
  - 每步事件日志（动作、被替换页、替换页频次、页框状态、频次快照）。

`demo.py` 使用固定序列直接运行，不需要交互输入。

## R03

LFU 在本实现中的规则：

- 命中：
  - 对应页频次 `freq[page] += 1`；
  - 更新最近访问时刻 `last_used[page] = step`。
- 缺页且有空框：
  - 直接填入，初始化 `freq=1`。
- 缺页且无空框：
  - 在驻留页中选择 `freq` 最小者淘汰；
  - 若 `freq` 并列，淘汰 `last_used` 最早（最久未使用）者；
  - 若仍并列，按槽位索引最小者。

因此该版本是“LFU + LRU 平局”的确定性实现。

## R04

`demo.py` 维护的关键状态：

- `frames: list[int | None]`：页框内容；
- `freq: dict[int, int]`：驻留页访问频次；
- `last_used: dict[int, int]`：最近访问时间戳；
- 统计量：`hits / faults / evictions`；
- `events`：逐步日志，记录行为与状态快照。

这些状态都在源码中显式更新，便于逐步验证算法决策。

## R05

MVP 主流程（LFU 部分）：

1. 初始化 `frames/freq/last_used`；
2. 顺序扫描访问序列；
3. 命中则增频并更新时间戳；
4. 缺页时先尝试空框填充；
5. 若无空框，按 `(freq, last_used, slot)` 选择受害页；
6. 完成替换并初始化新页频次为 1；
7. 记录事件日志；
8. 输出总览表与随机负载均值统计。

## R06

固定样例：

- `reference = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2]`
- `frame_count = 3`

本实现结果：

- LFU：`faults = 8`, `hits = 5`
- LRU：`faults = 9`, `hits = 4`
- FIFO：`faults = 10`, `hits = 3`

并打印 LFU 前 10 步事件轨迹用于人工检查。

## R07

复杂度分析（访问长度 `n`，页框数 `m`）：

- 时间复杂度：
  - 命中检测线性扫描 `O(m)`；
  - 缺页选择受害页扫描 `O(m)`；
  - 总体最坏 `O(nm)`。
- 空间复杂度：
  - 核心状态 `frames/freq/last_used` 为 `O(m)`；
  - 若保留全量 `events` 为 `O(n)`。

这是教学版实现，优先可读性与可验证性。

## R08

输入边界与异常处理：

- `frame_count <= 0`：抛出 `ValueError`；
- 序列中出现负页号：抛出 `ValueError`；
- 空访问序列：命中/缺页都为 0；
- 页框数为 1：退化为近似“连续重复才命中”的行为。

这些约束由 `_validate_inputs` 和主循环逻辑共同保证。

## R09

LFU 与常见策略差异：

- 对比 FIFO：
  - FIFO 不关心访问热度，只看进入先后；
  - LFU 会更偏向保留长期热点页。
- 对比 LRU：
  - LRU关注“最近性”；LFU关注“累计频次”；
  - 短期突发访问可能让 LRU 更快响应，LFU 可能更保守。

在稳定热点场景，LFU 通常有优势；在工作集快速漂移时不一定最优。

## R10

实现层正确性不变量：

1. 任一步骤都只会发生一次 `hit/fault_fill/fault_replace` 动作；
2. 始终满足 `hits + faults == len(reference)`；
3. `freq` 与 `last_used` 只包含驻留页；
4. 替换时被淘汰页一定来自当前 `frames`；
5. 替换后新页频次固定初始化为 1。

`main()` 中断言会持续检查关键一致性。

## R11

为了可解释对比，`demo.py` 同时实现了 FIFO 和 LRU：

- 三个算法共用相同输入；
- 输出统一结构的汇总表（`algorithm/hits/faults/ratio`）；
- 避免把 LFU 放在孤立场景，便于检验“趋势是否合理”。

这是一种面向算法教学与验证的最小实验设计。

## R12

参数建议：

- `frame_count` 是最关键参数，增大通常降低缺页率；
- 随机对比默认：
  - `trials=120`
  - `ref_len=60`
  - `page_kinds=10`
  - `seed=20260407`
- 固定随机种子可复现实验。

若要贴近真实负载，建议替换为带局部性的访问生成器。

## R13

本目录 MVP 覆盖：

- 手写 LFU 核心逻辑（含 LRU 平局规则）；
- 手写 FIFO/LRU 作为可解释基线；
- 使用 `numpy` 生成随机访问样本；
- 使用 `pandas` 格式化输出结果与事件表。

未覆盖：

- 内核级页表/TLB/脏页回写；
- 多进程全局页置换与 NUMA 等系统细节。

## R14

常见错误与防护：

- 错误：只计频次，不处理同频平局，导致结果不确定。  
  防护：固定 `(freq, last_used, slot)` 三级键。

- 错误：被替换页的 `freq/last_used` 未删除，污染后续决策。  
  防护：替换时显式 `del` 两个字典项。

- 错误：新换入页频次沿用旧值。  
  防护：统一初始化 `freq[new_page] = 1`。

- 错误：统计不闭合。  
  防护：断言 `hits + faults == len(reference)`。

## R15

实践建议：

- 先看前 10~20 步事件，确认频次演化是否符合直觉；
- 再做随机负载均值比较，观察整体趋势；
- 若希望减轻 LFU“历史包袱”，可扩展 Aging（频次衰减）；
- 可新增 OPT 作为理论下界辅助评估。

## R16

相关算法与扩展方向：

- 页面置换：FIFO / LRU / Clock / LFU / Random；
- LFU 变体：LFU with Dynamic Aging、TinyLFU（缓存领域）；
- 系统策略：局部置换 vs 全局置换；
- 性能议题：抖动（thrashing）与工作集动态变化。

## R17

目录文件说明：

- `README.md`：R01-R18 说明文档；
- `demo.py`：可运行 LFU MVP（含对照与断言）；
- `meta.json`：任务元数据（UID/学科/分类/源序号/路径）。

运行方式：

```bash
cd Algorithms/计算机-操作系统-0314-页面置换_-_LFU
uv run python demo.py
```

预期输出包含：
- 三算法汇总表；
- LFU 前 10 步事件；
- 随机负载平均缺页统计；
- `All assertions passed.`。

## R18

`demo.py` 中 LFU 的源码级流程可分 8 步：

1. `main()` 构造固定 `reference` 和 `frame_count`，调用 `lfu_page_replacement`。  
2. `lfu_page_replacement` 初始化 `frames`、`freq`、`last_used`、计数器。  
3. 每一步先判断 `page in frames`；若命中则执行 `freq[page] += 1` 并更新时间戳。  
4. 若缺页且存在 `None` 空框，直接装入新页并设置 `freq=1`。  
5. 若缺页且无空框，执行 `min(enumerate(frames), key=(freq, last_used, slot))` 选受害页。  
6. 删除受害页在 `freq/last_used` 的记录，将新页写入同槽位并初始化频次。  
7. 将动作、被替换页、受害频次、页框快照、频次快照写入 `events`。  
8. 返回结果后由 `main()` 进行断言、表格汇总和随机工作负载统计。

第三方库边界：
- `numpy` 仅用于随机访问序列生成；
- `pandas` 仅用于结果表格化展示；
- LFU 决策逻辑（命中、增频、平局裁决、替换）全部为手写源码实现，不依赖黑盒算法包。
