# 成功最短增广路

- UID: `MATH-0489`
- 学科: `数学`
- 分类: `网络流`
- 源序号: `489`
- 目标目录: `Algorithms/数学-网络流-0489-成功最短增广路`

## R01

本条目将“成功最短增广路”解释为 **Successive Shortest Augmenting Path（SSAP）**：
在残量网络中反复寻找一条当前最短费用增广路，并沿该路增广，直到无法继续增广（得到最小费用最大流），或达到指定需求流量（得到最小费用可行流）。

它是最小费用流问题里最常见、最易审计的基础算法之一。

## R02

本 MVP 目标：
- 解决有向图上的最小费用流问题；
- 同时支持两种模式：
  - `max_flow_limit=None`：计算最小费用最大流；
  - `max_flow_limit=K`：计算满足 `K` 单位需求的最小费用流；
- 输出每次增广路径、瓶颈、累计流量与累计费用；
- 对最终结果做容量约束、流守恒、总费用一致性校验。

范围限定：
- 容量为非负整数；
- 单位费用为整数（可为负，但不允许可达负环）；
- 固定内置样例，无交互输入。

## R03

问题形式化：
- 输入：有向图 `G=(V,E)`，容量 `cap(e) >= 0`，单位费用 `cost(e)`，源点 `s`，汇点 `t`。
- 目标一（最小费用最大流）：在所有 `s->t` 最大流中最小化总费用。
- 目标二（定额最小费用流）：在流量为 `F` 的可行流中最小化总费用。

总费用定义：
`TotalCost = sum_e flow(e) * cost(e)`。

## R04

SSAP 核心思路：
1. 在残量网络上找一条从 `s` 到 `t` 的最短费用路；
2. 沿该路增广一个瓶颈流量；
3. 更新正反向残量边；
4. 重复以上过程。

为兼顾负费用边与效率，MVP 使用：
- 初始一次 Bellman-Ford 生成势能（potential）；
- 后续每轮用“势能修正后的 reduced cost”跑 Dijkstra。

## R05

本实现数据结构：
- `ResidualEdge(to, rev, cap, cost)`：残量边与反向边索引；
- `graph: List[List[ResidualEdge]]`：邻接表残量图；
- `potential[v]`：Johnson 势能，保证 reduced cost 非负；
- `parent_v / parent_e`：记录最短路父节点与父边；
- `AugmentationRecord`：记录每轮增广路径与累计状态；
- `MinCostFlowResult`：统一输出流量、费用、边流与轨迹。

## R06

单轮增广流程：
1. 在当前残量图上求 `s->t` 最短路（按 reduced cost）；
2. 若不可达则终止；
3. 沿父指针回溯求瓶颈 `delta`；
4. 再回溯一遍：前向边 `cap -= delta`，反向边 `cap += delta`；
5. 累计 `total_flow += delta`；
6. 累计 `total_cost += delta * path_unit_cost`。

## R07

势能与 reduced cost：
- 定义 `c'(u,v)=c(u,v)+pi[u]-pi[v]`；
- 若 `pi` 合法，则残量图中所有可用边满足 `c'(u,v) >= 0`；
- 因而每轮可用 Dijkstra 求最短路；
- 完成一轮后对可达点更新 `pi[v] += dist[v]`，维持下一轮合法性。

这是一种把“可能有负权边”的最短路问题转为“非负权”问题的标准做法。

## R08

复杂度（`V=|V|, E=|E|`）：
- 初始 Bellman-Ford：`O(VE)`；
- 每轮最短路（Dijkstra + 堆）：`O(E log V)`；
- 设增广轮数为 `A`，总复杂度约 `O(VE + A * E log V)`；
- 空间复杂度 `O(V + E)`。

在整数容量场景下，`A` 一般不超过可行增广次数上界。

## R09

正确性要点：
- 残量更新保持容量约束与可撤销性（反向边）；
- 每轮都选当前残量网络的最短费用增广路；
- 势能变换不改变真实最短路相对优先级；
- 无可增广路时，当前流已达可达最大流；
- 若在给定流量约束下停止，则得到该流量下的最小费用解。

脚本还通过独立一致性校验（容量、守恒、费用重算）做实现层自检。

## R10

边界与异常处理：
- `n <= 1`、`source == sink`、端点越界：抛 `ValueError`；
- 负容量：抛 `ValueError`；
- `max_flow_limit < 0`：抛 `ValueError`；
- 费用要求为整数，非整数费用在本 MVP 中直接报错；
- 若势能状态被破坏导致 reduced cost 为负，抛 `RuntimeError`。

## R11

`demo.py` 模块划分：
- `_validate_input`：输入合法性检查；
- `_add_edge` / `_build_residual_graph`：构建残量网络；
- `_init_potential_with_bellman_ford`：初始化势能；
- `_shortest_path_with_potential`：每轮 Dijkstra 最短路；
- `successive_shortest_augmenting_path`：SSAP 主流程；
- `_assert_result_valid`：结果一致性校验；
- `run_case`：运行并打印单个样例；
- `main`：组织固定样例。

## R12

内置样例：
- `Case 1`：定额 3 单位流量的最小费用流，期望 `(flow=3, cost=7)`；
- `Case 2`：含一条负费用边（无负环）的最小费用最大流，期望 `(flow=3, cost=11)`。

两个样例都打印每轮增广路径，方便人工核对。

## R13

运行方式：

```bash
cd Algorithms/数学-网络流-0489-成功最短增广路
uv run python demo.py
```

无交互输入。末尾出现 `All SSAP checks passed.` 即表示样例通过。

## R14

输出字段解读：
- `flow`：最终送达汇点的总流量；
- `cost`：该流量下的总费用；
- `augmentations`：每轮路径、瓶颈、单位费用、累计流量与累计费用；
- `edge flows`：每条原始边的 `flow/capacity @ cost`。

若 `expected` 检查为 `PASS` 且最终无异常，说明实现与样例预期一致。

## R15

常见实现错误：
- 忘记维护反向边容量，导致无法回退流量；
- 直接在含负权边图上用 Dijkstra（未做势能修正）；
- 只更新流量不更新费用，或费用按 reduced cost 错算；
- 路径回溯时父边索引错位；
- 把“最大流模式”和“定额流模式”的停止条件混淆。

## R16

与其他网络流算法对比：
- 与 Edmonds-Karp / Dinic：它们主要优化“最大流值”，不直接处理费用；
- 与 Cost-Scaling：后者更偏工程性能，代码复杂度高；
- SSAP 优势在于：逻辑清晰、可解释性强、适合教学与中小规模问题。

## R17

可扩展方向：
- 引入容量/费用缩放优化以提升大图性能；
- 增加负环检测并给出可诊断报错；
- 支持下界流（lower bound）与点需求平衡；
- 增加随机图回归测试和性能基准脚本；
- 输出 Graphviz 可视化，展示每轮增广轨迹。

## R18

`demo.py` 源码级流程可拆为 8 步：
1. `main` 构造两组固定图，并分别调用 `run_case`。
2. `run_case` 调用 `successive_shortest_augmenting_path` 求解，再调用 `_assert_result_valid` 做结果审计。
3. `successive_shortest_augmenting_path` 先做 `_validate_input`，再用 `_build_residual_graph` 建立带反向边的残量图。
4. 调用 `_init_potential_with_bellman_ford` 初始化势能，使后续 reduced cost 最短路可由 Dijkstra 处理。
5. 进入主循环：每轮调用 `_shortest_path_with_potential` 计算 `s->t` 最短增广路；若不可达则退出。
6. 对可达点执行势能更新 `pi[v] += dist[v]`，然后沿父指针回溯计算瓶颈流量 `delta`。
7. 再次沿路径回溯，更新前向/反向残量，累计 `total_flow` 与 `total_cost`，并把本轮细节写入 `AugmentationRecord`。
8. 循环结束后从原边引用回读每条边实际流量，组装 `MinCostFlowResult`；`run_case` 打印并断言期望值。
