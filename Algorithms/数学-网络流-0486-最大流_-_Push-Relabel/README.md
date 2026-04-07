# 最大流 - Push-Relabel

- UID: `MATH-0486`
- 学科: `数学`
- 分类: `网络流`
- 源序号: `486`
- 目标目录: `Algorithms/数学-网络流-0486-最大流_-_Push-Relabel`

## R01

Push-Relabel（推送-重贴标签）是最大流问题的经典算法。它不依赖“找增广路”，而是维护一个允许超额流的预流（preflow），再通过局部 `push` 和 `relabel` 操作逐步把多余流量推向汇点。

本条目提供一个可直接运行的 Python MVP：
- 显式实现残量网络与反向边；
- 显式实现 `push / relabel / discharge`；
- 内置固定测试图，运行后输出最大流值与每条边的流量。

## R02

问题定义（本目录实现）：
- 输入：
  - 顶点数 `n`（节点编号 `0..n-1`）；
  - 有向边列表 `[(u, v, capacity), ...]`，容量非负；
  - 源点 `source` 与汇点 `sink`。
- 输出：
  - 最大流值 `max_flow`；
  - 每条原始边的 `(u, v, capacity, flow)`；
  - 最终高度数组 `height` 与超额数组 `excess`（便于审计）。

`demo.py` 不读取外部输入，直接运行内置样例。

## R03

核心数学与约束：

1. 残量容量：
   - 对任意边 `(u,v)`，残量网络记录可继续发送的容量 `c_f(u,v)`。
2. 预流（preflow）：
   - 除源点外，允许节点入流大于出流，即 `excess[v] >= 0`。
3. 高度函数（label）`h(v)`：
   - 只允许沿满足 `h(u) = h(v) + 1` 的“下坡边”推流。
4. `push` 操作：
   - `delta = min(excess[u], c_f(u,v))`，将 `delta` 从 `u` 推到 `v`。
5. `relabel` 操作：
   - 当 `u` 仍有超额且没有可推的下坡边时，设置
     `h(u) = min_{(u,w), c_f(u,w)>0} h(w) + 1`。

算法终止时，所有非源非汇节点超额为 0，得到可行最大流。

## R04

算法流程（高层）：
1. 构建残量图（每条原边配一条反向边）。
2. 初始化高度：`h(source)=n`，其余为 0。
3. 预流初始化：饱和源点所有出边，形成初始超额。
4. 将有超额的中间点加入活动队列（active queue）。
5. 反复取出活动点 `u` 执行 `discharge(u)`：
   - 能 `push` 就推；
   - 不能推则 `relabel`。
6. 若 `u` 仍有超额，重新入队（采用 relabel-to-front 风格优化）。
7. 队列为空即结束。
8. 返回 `excess[sink]` 作为最大流，并恢复原边流量。

## R05

核心数据结构：
- `ResidualEdge(to, rev, cap)`：
  - `to`：终点；
  - `rev`：反向边在 `graph[to]` 中的下标；
  - `cap`：当前残量容量。
- `graph: List[List[ResidualEdge]]`：残量网络邻接表。
- `height: List[int]`：节点高度标签。
- `excess: List[float]`：节点当前超额流。
- `seen: List[int]`：`discharge` 时当前扫描到的邻边位置。
- `active: deque[int]`：活动节点队列。
- `original_refs`：记录每条输入边对应的前向残量边位置，用于回读最终流量。

## R06

正确性要点（实现对应）：
- 容量约束：`push` 只在残量 `cap > 0` 时进行，且推送量不超过残量。
- 反向可撤销性：每次推流都会同步增加反向边残量，保证后续可回退。
- 标签合法性：仅沿 `h(u)=h(v)+1` 推流，`relabel` 只增不减，避免无限振荡。
- 流守恒（终态）：循环终止时，除源汇外无超额节点，恢复为可行流。
- 最优性：无可继续推进的活动节点时，预流对应一个最大流（Push-Relabel 定理）。

## R07

复杂度分析：
- 理论上，基础 Push-Relabel 的最坏时间复杂度为 `O(V^2 E)`。
- 空间复杂度为 `O(V + E)`（残量图与状态数组）。
- 本实现使用邻接表 + 活动队列 + current-arc（`seen`）以减少重复扫描，属于教学友好的最小可解释版本。

## R08

边界与异常处理：
- `n < 2`、`source == sink`、端点越界：抛 `ValueError`。
- 负容量或非有限容量：抛 `ValueError`。
- 若活动点没有任何残量出边（实现上不应发生）：抛 `RuntimeError`。
- 脚本样例均为有向图；若要处理无向边，应显式转成两条有向边。

## R09

MVP 取舍说明：
- 仅用 Python 标准库实现，不依赖黑盒图算法库。
- 不实现 gap heuristic / global relabel 等高级优化，优先保证可读性与可审计性。
- 输出原边流量与高度标签，方便教学和手工核对。
- 选用两组固定样例：
  - CLRS 经典网络（期望最大流 23）；
  - 小型瓶颈网络（期望最大流 13）。

## R10

`demo.py` 主要函数职责：
- `_validate_input`：检查图规模、端点与容量合法性。
- `_add_edge`：向残量网络加入前向边和反向边。
- `push_relabel_max_flow`：主算法入口，执行 preflow + discharge。
- `push`（内部函数）：执行一次合法推流。
- `relabel`（内部函数）：提升活动点高度。
- `discharge`（内部函数）：持续处理单个活动点直到超额清零或需重排。
- `run_case`：运行单个样例并打印结果。
- `main`：组织内置样例并执行。

## R11

运行方式：

```bash
cd Algorithms/数学-网络流-0486-最大流_-_Push-Relabel
uv run python demo.py
```

脚本无交互输入，直接输出结果。

## R12

输出字段说明：
- `max_flow`：汇点最终超额，等于最大流值。
- `expected` / `check`：样例期望值与通过状态。
- `edge flows (u -> v | flow/capacity)`：每条原始边最终流量与容量。
- `vertex heights`：算法结束后各节点高度标签。

这些信息可以快速检查：
- 源点总流出是否接近 `max_flow`；
- 边流量是否不超过容量；
- 结果是否与已知答案一致。

## R13

建议最小测试集（已内置）：
- `CLRS directed network`：验证经典答案 `23`。
- `Small bottleneck network`：验证多路径 + 瓶颈下的推流行为，答案 `13`。

建议补充异常测试：
- 含负容量边（应报错）；
- 非法节点编号（应报错）；
- `source == sink`（应报错）。

## R14

可调参数与工程注意：
- `EPS = 1e-12`：浮点比较阈值。
- 队列策略：当前实现是活动队列 + relabel-to-front 风格；可替换为最高标签优先策略。
- 数据规模增大时可加：
  - 周期性 `global relabel`（按汇点反向 BFS 重新设高）；
  - `gap heuristic`（发现高度空洞时批量抬高）。

## R15

与其他最大流方法对比：
- 与 Ford-Fulkerson：
  - FF 依赖增广路选择，可能退化；
  - Push-Relabel 更偏局部操作，实践中常更稳。
- 与 Edmonds-Karp：
  - EK 使用 BFS 找最短增广路，思路直观但在稠密图上可能慢。
- 与 Dinic：
  - Dinic 依赖分层图与阻塞流；
  - Push-Relabel 对某些图族（尤其稠密图）表现优秀，且易做工程优化。

## R16

典型应用场景：
- 网络带宽/管道容量分配。
- 二分图最大匹配（转最大流模型）。
- 图像分割中的 s-t cut（最小割与最大流等价）。
- 资源调度与供应链网络可行流分析。

## R17

可扩展方向：
- 增加 `global relabel` 与 `gap heuristic` 提升大图性能。
- 输出最小割顶点集（由最终残量图可达性导出）。
- 支持从文件读取边集并做批量 benchmark。
- 提供 `networkx` 对照验证模式（仅用于测试，不作为求解主路径）。

## R18

`demo.py` 源码级算法流程（9 步）：
1. `main` 构造两组固定测试边集，并调用 `run_case`。
2. `run_case` 调用 `push_relabel_max_flow`，接收 `max_flow/edge_flows/heights`。
3. `push_relabel_max_flow` 先执行 `_validate_input`，再通过 `_add_edge` 构建残量图和反向边。
4. 初始化 `height[source]=n`、`excess[*]=0`，随后饱和源点出边形成初始 preflow，并把新活动点入队。
5. 进入主循环，持续弹出活动点 `u`，执行 `discharge(u)`。
6. `discharge` 内部按 `seen[u]` 顺序扫描邻边：若满足 `h(u)=h(v)+1` 且有残量则调用 `push`。
7. `push` 更新前向残量、反向残量和两端 `excess`；若目标点刚变成活动点则入队。
8. 当 `u` 无可推边时调用 `relabel(u)`，将其高度改为“可达邻居最小高度 + 1”，继续尝试推流，直到 `u` 超额清零或需重排。
9. 队列清空后算法结束，用 `excess[sink]` 作为最大流，并由 `原始容量 - 前向残量` 回读每条原边流量并打印。
