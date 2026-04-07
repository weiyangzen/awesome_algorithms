# 并行BFS

- UID: `CS-0309`
- 学科: `计算机`
- 分类: `并行与分布式`
- 源序号: `470`
- 目标目录: `Algorithms/计算机-并行与分布式-0470-并行BFS`

## R01

并行 BFS（Parallel Breadth-First Search）是在广度优先搜索中引入并发执行的图遍历方法。  
BFS 的关键结构是“按层推进”（frontier expansion）：先处理距离起点为 `d` 的所有节点，再处理距离为 `d+1` 的节点。  
并行化的核心机会也来自这一层结构：同一层中的多个节点可以同时扩展邻居。

本条目给出一个最小可运行 Python MVP，包含：
- 顺序 BFS 基线；
- 层同步线程并行 BFS；
- 正确性校验（可达集合 + 最短距离一致）；
- 性能对比输出。

## R02

为什么并行 BFS 重要：
- BFS 是最短路径（无权图）和可达性分析的基础组件；
- 图规模变大后，单线程逐点扩展容易成为瓶颈；
- 在社交网络、路网分析、依赖图扫描中，“降低遍历延迟”常比“单核吞吐”更关键。

并行 BFS 不是单纯开线程，而是要保证：
- 层次语义不被破坏；
- `visited` 去重在并发场景下仍正确；
- 距离数组 `dist` 仍满足最短层数定义。

## R03

本实现采用“层同步（level-synchronous）并行 BFS”策略：
1. 维护当前层 frontier；
2. 将 frontier 切分为若干块，交给多个 worker 并行扩展；
3. 每个 worker 把新发现节点写入本地 `local_next`；
4. 主线程合并各 worker 的 `local_next` 形成下一层 frontier；
5. 深度 `depth` 加 1，进入下一轮。

这样能清晰保持 BFS 的层边界，便于验证距离正确性。

## R04

与顺序 BFS 的关键差异：
- 顺序 BFS：单队列、单执行流；
- 并行 BFS：同层多执行流并发扩展；
- 顺序版访问顺序更稳定，并行版同层内顺序可能变化；
- 并行版的正确性重点是“集合与距离一致”，不是逐节点顺序一致。

因此本条目以 `dist` 数组一致性作为主校验指标，而非访问序列完全相同。

## R05

复杂度分析（邻接表）：
- 顺序 BFS：时间 `O(V + E)`，空间 `O(V)`；
- 并行 BFS：总工作量仍约 `O(V + E)`，理论时间近似 `O((V + E) / p + 同步开销)`（`p` 为并行度）。

实践中的额外成本：
- 锁竞争（`visited` 原子检查+标记）；
- frontier 分块与结果合并；
- Python GIL 对 CPU 密集线程并行的限制。

## R06

`demo.py` 的实验配置：
- 图模型：可复现连通稀疏无向图；
- 规模：`n=10000` 节点，`m=50000` 边；
- 生成方法：先链式连通，再随机补边；
- 并行度：`workers=min(8, os.cpu_count())`。

输出包括：
- 配置参数；
- 顺序/并行耗时；
- 速度比；
- 距离一致性验证结果。

## R07

并发一致性保障机制：
- 全局共享 `visited` 和 `dist`；
- 使用互斥锁保护 “检查 visited + 首次标记 + 写 dist”；
- 只有抢到访问权的线程才可把节点放入下一层；
- 每个节点最多进入一次 frontier，避免重复扩展。

这等价于并行图遍历里常见的原子 test-and-set visited 模式。

## R08

运行环境与依赖：
- Python `>=3.10`；
- `numpy`（用于可复现随机图生成）；
- 标准库：`collections`、`threading`、`concurrent.futures`、`time`、`os`。

运行方式：

```bash
cd Algorithms/计算机-并行与分布式-0470-并行BFS
uv run python demo.py
```

## R09

适用场景：
- 大图可达性扫描；
- 无权图最短层数计算；
- 并行图算法教学与最小工程验证。

不适用场景：
- 强依赖完全确定遍历序列的业务逻辑；
- 追求极致 CPU 并行加速的生产任务（需更底层实现）；
- 跨机器分布式容错和通信优化场景（需专门分布式框架）。

## R10

正确性要点：
1. BFS 本质要求按层推进；
2. 并行只发生在“同层扩展”内部，不跨层混淆；
3. 通过受控抢占 `visited`，每节点首次发现时确定唯一距离；
4. 因而并行版 `dist` 应与顺序版完全一致。

`demo.py` 使用 `_validate_results` 对以下条件做断言：
- 访问无重复；
- 可达集合一致；
- 距离数组一致。

## R11

边界与鲁棒性处理：
- 空图返回空结果；
- 非法起点抛出明确异常；
- `workers<=1` 自动回退顺序 BFS；
- 起点孤立时可正常返回仅起点可达结果；
- 图生成器过滤自环并自动去重边。

此外脚本先跑小图自检，再进行大图基准，先保正确性再看性能。

## R12

`demo.py` 关键函数职责：
- `build_undirected_graph`：边集转邻接表并排序；
- `generate_connected_sparse_graph`：生成可复现连通稀疏图；
- `sequential_bfs`：顺序基线（返回 `order, dist`）；
- `_chunk_frontier`：frontier 分块；
- `parallel_bfs`：层同步并行 BFS；
- `_validate_results`：一致性校验；
- `_benchmark`：计时封装；
- `run_self_test`/`main`：自检与基准入口。

## R13

性能解释建议：
- frontier 越宽，单层并发潜力越高；
- 图更像“长链”时，并发空间有限；
- 锁竞争越激烈，并行收益越小；
- 在 CPython 下线程模型主要用于展示并行算法结构，不保证线性加速。

因此该 MVP 重点是“流程和正确性可追踪”，不是绝对性能极值。

## R14

常见错误与规避：
1. 仅做 `if not visited` 但无锁，导致重复入队；
2. 跨层混合扩展，破坏 BFS 距离语义；
3. 只比较访问顺序，不比较 `dist`，掩盖错误；
4. 随机图不固定种子，结果难复现。

本实现对应措施：锁保护抢占、层同步循环、距离断言、固定随机种子。

## R15

可扩展方向：
- 换用多进程或 C/CUDA 后端降低 GIL 影响；
- 使用分段锁/原子位图降低锁竞争；
- 引入动态工作窃取改善负载均衡；
- 拓展到分布式 BFS（消息传递 + frontier 同步 + 终止检测）；
- 结合真实图数据集输出更系统的性能报告。

## R16

相关算法与技术：
- 顺序 BFS、双向 BFS、多源 BFS；
- 并行 DFS（分支并发）与并行 BFS（层并发）的对比；
- 无权最短路径问题中的层次遍历；
- 图计算中的 frontier 模型与活跃顶点模型；
- 并发程序中的原子标记与去重协议。

## R17

运行 `demo.py` 可看到类似输出：
- `[self-test] passed`；
- `[config] n=..., m=..., workers=..., seed=...`；
- `[time] sequential_bfs: ...`；
- `[time] parallel_bfs(level-synchronous, threaded): ...`；
- `[time] speedup(sequential/parallel): ...`；
- `[check] distance consistency passed, max_depth=...`。

这构成完整的最小验证闭环：自检 -> 基准 -> 校验。

## R18

`demo.py` 源码级并行 BFS 流程（8 步）：
1. `main` 先执行 `run_self_test`，在小图上确认顺序/并行 BFS 的可达集合和距离一致。  
2. `generate_connected_sparse_graph` 先造链保证连通，再用 `numpy` 随机补边，得到可复现图。  
3. `sequential_bfs` 作为基线，使用队列逐层遍历并写入 `dist`。  
4. `parallel_bfs` 初始化 `frontier=[start]`，并设置 `visited[start]=True, dist[start]=0`。  
5. 每一层用 `_chunk_frontier` 把 frontier 切块，提交到线程池并行执行 `_worker`。  
6. `_worker` 遍历块内节点邻居，在锁内做 `visited` 抢占；首次发现时立刻写 `dist[v]=depth+1` 并加入局部下一层。  
7. 主线程汇总所有 worker 的局部结果成为 `next_frontier`，更新 `depth` 后继续下一层，直到 frontier 为空。  
8. `_validate_results` 校验顺序与并行结果，`_benchmark` 输出耗时与速度比，完成可复现实验闭环。
