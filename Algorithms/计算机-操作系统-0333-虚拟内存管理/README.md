# 虚拟内存管理

- UID: `CS-0186`
- 学科: `计算机`
- 分类: `操作系统`
- 源序号: `333`
- 目标目录: `Algorithms/计算机-操作系统-0333-虚拟内存管理`

## R01

`虚拟内存管理` 的目标是把“进程看到的连续虚拟地址空间”映射到“有限且离散的物理内存帧”，并在内存不足时通过置换机制维持系统运行。  
本条目实现一个可运行 MVP：单级页表 + TLB + 需求分页 + CLOCK 置换，直观展示地址翻译、缺页异常与脏页回写的完整闭环。

## R02

核心机制由四层组成：
- 虚拟地址拆分：`vaddr -> (vpn, offset)`。
- 快路径：先查 TLB（命中直接得到 frame）。
- 次快路径：TLB 未命中时查页表（在内存则补回 TLB）。
- 慢路径：页不在内存时触发缺页，必要时执行 CLOCK 置换并回写脏页。

## R03

MVP 输入输出约定：
- 输入：`demo.py` 内部固定参数（虚拟页数、物理帧数、页大小、TLB 容量、操作次数、随机种子）。
- 处理：执行确定性 sanity check + 一段带局部性的读写访问序列。
- 输出：访问统计（TLB 命中率、缺页率、置换次数、回写次数）、驻留页状态与一致性校验结果。

## R04

典型应用场景：
- 操作系统课程中理解页式内存管理的关键路径。
- 评估工作集变化对缺页率和 TLB 命中率的影响。
- 作为后续实验基础（例如对比 FIFO/LRU/CLOCK、多级页表或 NUMA 策略）。

## R05

本实现的数据结构：
- `PageTableEntry`：记录 `present/frame_id/dirty/referenced/last_touch`。
- `frame_to_vpn`：物理帧到虚拟页的反向映射，便于淘汰受害页。
- `tlb(OrderedDict)`：维护固定容量 LRU TLB。
- `memory(np.ndarray)`：二维字节数组，模拟物理内存帧内容。
- `backing_store`：页后备存储，模拟磁盘页镜像。

## R06

地址翻译与访问（fast path）流程：
1. 解析虚拟地址得到 `vpn` 与 `offset`。
2. 查询 TLB；命中则直接定位帧。
3. 若 TLB 未命中但页表显示驻留，则补写 TLB 并访问。
4. 对读操作返回字节值；对写操作更新内存并置 `dirty=True`。
5. 每次访问都置 `referenced=True`，供 CLOCK 使用。

## R07

缺页与置换（slow path）流程：
1. 页表 `present=False` 时记一次 `page_fault`。
2. 若存在空闲帧，直接装入新页。
3. 否则执行 CLOCK：沿 `clock_hand` 扫描，遇到 `referenced=True` 先清零并跳过，直到找到 `referenced=False` 的受害页。
4. 受害页为脏页时写回 `backing_store`，并记 `write_back`。
5. 更新受害页页表项为不驻留，失效其 TLB 项。
6. 将缺页从后备存储装入选定帧，更新页表与 TLB。

## R08

时间复杂度（均摊）：
- 地址翻译（TLB 命中）：`O(1)`。
- 地址翻译（TLB miss + 页表命中）：`O(1)`。
- 缺页处理：平均 `O(1)`，最坏受 CLOCK 扫描影响为 `O(F)`（`F` 为物理帧数）。
- 一致性校验 `validate()`：`O(V + F + T)`（虚拟页、帧、TLB 项）。

## R09

空间复杂度：
- 物理内存数据区：`O(F * page_size)`。
- 页表：`O(V)`。
- TLB：`O(tlb_capacity)`。
- 后备存储（模拟）：`O(V * page_size)`。

## R10

MVP 技术栈：
- Python 3
- `numpy`（内存页字节数组与统计计算）
- 标准库：`dataclasses`、`collections.OrderedDict`、`typing`、`time`

实现保持最小依赖，不依赖大型框架。

## R11

运行方式（仓库根目录）：

```bash
uv run python Algorithms/计算机-操作系统-0333-虚拟内存管理/demo.py
```

脚本不需要参数，也不会请求交互输入。

## R12

输出字段说明：
- `Reads/Writes`：读写操作计数。
- `TLB`：命中/未命中次数与命中率。
- `Paging`：缺页次数、缺页率、淘汰次数、脏页回写次数。
- `Resident`：当前驻留页数、脏页数、空闲帧数。
- `Fault-rate checkpoints`：每 200 次访问采样一次累计缺页率分布。
- `Validation`：内部不变式是否全部通过。

## R13

边界与鲁棒性：
- 初始化时检查 `num_virtual_pages/num_frames/page_size/tlb_capacity` 全为正。
- 访问时检查虚拟地址越界，非法地址直接抛 `ValueError`。
- 置换时确保受害页与反向映射一致，避免悬挂映射。
- `validate()` 检查页表-帧表-TLB 三方一致性，以及统计量守恒关系。

## R14

当前实现局限：
- 单进程、单级页表，不含进程切换与地址空间隔离。
- 未模拟页保护位（R/W/X）、缺页类型区分与异常返回路径。
- CLOCK 仅按 `referenced` 位近似 LRU，不含真实硬件访问位更新延迟。

## R15

可扩展方向：
- 增加多进程与上下文切换，模拟 TLB shootdown。
- 增加页面共享、写时复制（COW）与匿名页/文件页区分。
- 对比 FIFO、LRU、CLOCK-Pro 在不同工作集下的缺页率曲线。
- 引入二级或多级页表，量化页表内存开销与翻译路径成本。

## R16

最小测试建议：
- 功能测试：构造少量页和帧，验证“访问-缺页-装入-再访问命中”闭环。
- 置换测试：帧满后访问新页，确认触发淘汰且映射更新正确。
- 脏页测试：写后淘汰，确认回写计数增加且后备存储内容更新。
- 一致性测试：长序列访问后 `validate()` 仍返回空错误列表。

## R17

方案对比：
- `FIFO`：实现简单，但容易出现 Belady 异常。
- `LRU`：理论效果好，但精确实现成本高。
- `CLOCK`：通过引用位近似 LRU，工程实现更轻量。
- 本 MVP 选择 `CLOCK + LRU-TLB`，在可读性和真实性之间保持平衡。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main()` 先执行 `run_sanity_checks()`，用小规模用例验证缺页、淘汰与写回行为正确。
2. 构造 `VirtualMemoryManager`，在 `__post_init__` 中初始化页表、帧表、TLB、物理内存和后备存储。
3. `build_trace()` 生成带局部性的读写访问序列，形成可复现实验输入。
4. `run_trace()` 逐条执行访问；每次调用 `access()` 先做地址解码，再尝试 `TLB` 查找。
5. TLB 未命中时走页表：若页已驻留则补写 TLB；若不驻留则 `_handle_page_fault()` 进入缺页处理。
6. `_handle_page_fault()` 在“空闲帧装入”与“CLOCK 选受害页淘汰”之间决策，并在脏页时执行回写与 TLB 失效。
7. 访问完成后更新 `referenced/dirty/last_touch` 与统计计数，周期性记录累计缺页率采样点。
8. 收尾阶段调用 `validate()` 做一致性审计，输出关键指标与 `Validation: PASS/FAIL`。
