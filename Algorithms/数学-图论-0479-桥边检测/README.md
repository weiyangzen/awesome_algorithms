# 桥边检测

- UID: `MATH-0479`
- 学科: `数学`
- 分类: `图论`
- 源序号: `479`
- 目标目录: `Algorithms/数学-图论-0479-桥边检测`

## R01

本条目实现“无向图桥边（Bridge / Cut Edge）检测”的最小可运行 MVP。

目标：
- 给定无向图，找出所有桥边；
- 采用 Tarjan 的 DFS 时间戳 + low-link 方法，时间复杂度 `O(V+E)`；
- 支持非连通图；
- 显式处理重边（parallel edges）场景；
- `demo.py` 内置固定样例，运行即输出结果，无需交互输入。

## R02

问题定义（本实现范围）：
- 输入：无向图 `G=(V,E)`，顶点编号 `0..n-1`，边集 `[(u,v), ...]`。
- 输出：桥边集合 `B ⊆ E`。
- 桥边定义：若删除边 `e` 后，图的连通分量数量增加，则 `e` 是桥边。

在工程表示上，每条输入边分配唯一 `edge_id`，输出使用 `edge_id` 标识，避免重边时歧义。

## R03

核心数学量：
- `tin[v]`：DFS 首次访问顶点 `v` 的时间戳。
- `low[v]`：从 `v` 出发，经过“若干树边 + 至多一条返祖边”可到达的最小 `tin`。

桥边判据（树边 `v -> to`）：
- 若 `low[to] > tin[v]`，则边 `(v,to)` 为桥边。

直观解释：
- `to` 子树无法通过返祖边回到 `v` 或 `v` 的祖先，
- 因而删除 `(v,to)` 会把该子树与上方部分断开。

## R04

算法流程（Tarjan 桥边检测）：
1. 初始化 `visited/tin/low` 与全局 `timer`。
2. 对每个未访问顶点启动 DFS（覆盖非连通图）。
3. 进入 `v` 时设置 `tin[v]=low[v]=timer`。
4. 枚举 `v` 的邻接边 `(to, edge_id)`。
5. 若该边是 DFS 入边（`edge_id == parent_edge_id`），跳过。
6. 若 `to` 已访问，则用 `tin[to]` 更新 `low[v]`（返祖边）。
7. 若 `to` 未访问，递归 DFS 到 `to`，回溯后用 `low[to]` 更新 `low[v]`。
8. 若满足 `low[to] > tin[v]`，记录该 `edge_id` 为桥边。

## R05

核心数据结构：
- `Graph`（`dataclass`）
  - `n`: 顶点数。
  - `edges`: 原始边列表，索引即 `edge_id`。
  - `adjacency[u] = [(v, edge_id), ...]`：带边编号的邻接表。
- `BridgeResult`（`dataclass`）
  - `bridge_edge_ids`: 桥边 id 列表。
  - `tin`, `low`: DFS 产物，便于调试与教学展示。

为何要存 `edge_id`：
- 在无向图中，父边判断必须“按边身份”而非仅按父节点；
- 这保证重边场景不会误判。

## R06

正确性要点：
- 不变量 1：`tin[v]` 是 DFS 访问序，严格单调递增。
- 不变量 2：`low[v]` 始终是当前已知可回溯到的最小祖先时间戳。
- 对树边 `v->to`：
  - 若 `low[to] <= tin[v]`，说明子树可回到 `v` 或更高祖先，删该边不致断开；
  - 若 `low[to] > tin[v]`，说明子树与外界只靠此边相连，删之必断开。
- 对非连通图，对每个连通分量重复该推理仍成立。

## R07

复杂度分析：
- 时间复杂度：`O(V + E)`
  - 每条无向边在邻接表中出现两次，常数次处理。
- 空间复杂度：`O(V + E)`
  - 邻接表 `O(V+E)`；
  - `visited/tin/low` 为 `O(V)`。

对比朴素删边法（每条边都重跑一次连通性）：通常是 `O(E*(V+E))`。

## R08

边界与异常处理（MVP）：
- `num_vertices <= 0`：抛 `ValueError`。
- 边端点越界：抛 `ValueError`。
- 自环（`u==v`）：抛 `ValueError`（桥边语义下通常无意义，MVP 直接拒绝）。
- 非连通图：正常支持。
- 重边：正常支持，且不会把平行边误判为桥边。

## R09

MVP 取舍：
- 采用 Tarjan 原生实现，不依赖 NetworkX 等黑盒 API。
- 额外提供 `find_bridges_bruteforce` 作为可读验证器，便于教学和回归测试。
- 不实现动态桥边维护（在线增删边）与并行优化，优先保证正确性与可解释性。

## R10

`demo.py` 函数职责：
- `build_undirected_graph`：构图并做输入校验。
- `find_bridges_tarjan`：主算法，返回桥边及 `tin/low`。
- `count_connected_components`：连通分量计数（支持临时禁用某条边）。
- `find_bridges_bruteforce`：删边验证器。
- `canonical_edge_repr` / `format_bridge_set`：稳定、可读地打印桥边。
- `run_case`：单样例执行并对比 Tarjan 与朴素法结果。
- `main`：组织多个固定图样例。

## R11

运行方式：

```bash
cd Algorithms/数学-图论-0479-桥边检测
python3 demo.py
```

脚本会依次打印 3 个案例的桥边检测结果、一致性检查以及 `tin/low` 数组。

## R12

输出字段说明：
- `Tarjan bridges`：主算法给出的桥边集合。
- `Bruteforce check`：删边法得到的桥边集合（参考真值）。
- `match`：两种方法是否一致。
- `tin`：DFS 首次访问时间戳数组。
- `low`：low-link 数组。

若 `match=True`，可认为实现在该样例上通过一致性校验。

## R13

内置测试集覆盖点：
1. `cycle + chain + cycle`：应存在 2 条桥边。
2. `parallel edges`：并行边不应被判为桥，验证重边处理。
3. `two disconnected cycles`：非连通图且每个分量有环，应无桥边。

建议额外补充：
- 单条边图、树图、星型图；
- 大规模稀疏图性能测试；
- 随机图 + 与其他实现交叉验证。

## R14

可调参数：
- `num_vertices` 与 `edges`：决定图规模与结构。
- `run_case` 样例数量与内容。

实践建议：
- 教学演示优先小图（便于手工核对 `tin/low`）；
- 回归测试可加入随机图并自动比较 Tarjan 与朴素法输出。

## R15

方法对比：
- Tarjan 桥边检测：`O(V+E)`，适合一次性离线求解。
- 朴素删边判定：实现直观但通常更慢，适合做基线验证。
- 动态连通结构（如 Link-Cut Tree / ET-Tree）：可支持在线更新，但实现复杂，非本条目 MVP 目标。

## R16

典型应用场景：
- 网络鲁棒性分析（关键链路识别）；
- 道路/管网的单点故障风险评估；
- 社交/通信图中脆弱连接定位；
- 图算法教学中的 DFS low-link 经典案例。

## R17

可扩展方向：
- 在同一次 DFS 中扩展实现割点（articulation points）检测；
- 支持边权与业务属性，桥边输出时附带风险评分；
- 改造为迭代版 DFS，避免超深递归；
- 增加随机图基准与性能统计（耗时、内存）。

## R18

`demo.py` 源码级算法流程（9 步）：
1. `main` 构造三个固定无向图案例，分别覆盖一般桥边、重边与非连通场景。  
2. `build_undirected_graph` 为每条边分配唯一 `edge_id`，并生成 `adjacency[u] = (v, edge_id)`。  
3. `run_case` 先调用 `find_bridges_tarjan` 执行 `O(V+E)` 主算法。  
4. 在 `find_bridges_tarjan` 中初始化 `visited/tin/low` 与 `timer`，并对每个未访问顶点启动 DFS。  
5. DFS 进入节点 `v` 时写入 `tin[v]=low[v]=timer`，随后遍历邻接边。  
6. 遇到返祖边（目标点已访问且非父边）时，用 `tin[to]` 收紧 `low[v]`。  
7. 遇到树边时递归处理子节点 `to`，回溯后用 `low[to]` 更新 `low[v]`，并依据 `low[to] > tin[v]` 判定桥边。  
8. `run_case` 再调用 `find_bridges_bruteforce`：逐条删边并比较连通分量，得到参考答案。  
9. 最后打印 Tarjan 与朴素法结果是否一致，以及 `tin/low` 数组，完成可解释验证闭环。  
