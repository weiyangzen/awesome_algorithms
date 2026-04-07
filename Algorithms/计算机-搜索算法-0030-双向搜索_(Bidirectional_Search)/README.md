# 双向搜索 (Bidirectional Search)

- UID: `CS-0027`
- 学科: `计算机`
- 分类: `搜索算法`
- 源序号: `30`
- 目标目录: `Algorithms/计算机-搜索算法-0030-双向搜索_(Bidirectional_Search)`

## R01

双向搜索（Bidirectional Search）用于在图中求两点之间的最短路径（通常指无权图的最少边数路径）。
它同时从起点 `s` 与终点 `t` 各做一轮 BFS，直到两侧搜索前沿相遇。

## R02

问题形式化：
- 输入：图 `G=(V,E)`、起点 `s`、终点 `t`
- 输出：若存在路径，返回一条从 `s` 到 `t` 的最短路径；否则返回空结果（如 `None`）
- 代价模型：每条边代价相同（无权图）

## R03

适用前提：
- 图是有限图，邻接关系可迭代
- 对无向图可直接双向扩展
- 对有向图，后向搜索需要在反向图 `G^R` 上展开
- 若边有不同权重，应使用 Dijkstra/A* 等算法，不应直接套本实现

## R04

核心思想：
- 单向 BFS 的搜索半径约为 `d`（`d` 是最短路径长度）
- 双向 BFS 理想情况下各走 `d/2`，节点扩展规模可从 `O(b^d)` 降到 `O(b^(d/2))` 量级（`b` 为平均分支因子）
- 每轮扩展时优先扩展“当前更小的队列”，降低总扩展节点数

## R05

简化伪代码：

```text
if s == t: return [s]
front_s = queue([s]); front_t = queue([t])
parent_s = {s: None}; parent_t = {t: None}
visited_s = {s}; visited_t = {t}

while front_s and front_t:
    expand the smaller frontier by one BFS layer
    if any new node u is in the opposite visited set:
        meet = u
        reconstruct s -> meet using parent_s
        reconstruct meet -> t using parent_t
        return merged path

return None
```

## R06

正确性要点（直观）：
- 两侧都按 BFS 分层扩展，保证首次到达某节点时所用边数最少
- 一旦出现交汇节点 `m`，组合路径 `s -> m -> t` 的长度等于两侧最短层深之和
- 在无权图上，这样得到的总边数是最短的

## R07

复杂度：
- 时间复杂度：最坏仍可写作 `O(|V| + |E|)`（每边最多被常数次访问）
- 典型情况下：节点扩展量显著小于单向 BFS，近似受 `O(b^(d/2))` 控制
- 空间复杂度：`O(|V|)`（两侧 visited、parent、queue）

## R08

边界与异常情况：
- `s == t`：直接返回 `[s]`
- `s` 或 `t` 不在图中：抛出 `KeyError`
- 图不连通或不可达：返回 `None`
- 自环、重边不影响正确性（visited 会去重）
- 有向图若不构造反向邻接表，后向搜索会失真

## R09

与单向 BFS 对比：
- 单向 BFS：实现更简单，但在路径很深时扩展量可能爆炸
- 双向 BFS：实现稍复杂（维护两套状态并拼接路径），但在“大图+稀疏目标”场景通常更快
- 当起点和终点距离很近时，两者差距可能不明显

## R10

本目录 MVP 实现策略：
- `demo.py` 中 `bidirectional_search` 返回 `SearchResult(found, path, expanded_nodes, meeting_node)`
- 支持 `directed=False/True`
- 当 `directed=True` 时，自动构建反向邻接表供后向搜索使用
- 提供多组固定测试：无向命中、无向不可达、有向命中、有向不可达、`start==goal`

## R11

运行方式：

```bash
uv run python demo.py
```

脚本无交互输入，直接打印每个用例结果与断言检查。

## R12

输出解读：
- `found=True`：找到路径，`path` 为从起点到终点的节点序列
- `expanded_nodes`：两侧累计弹出/扩展的节点数量，用于观察搜索成本
- `meeting_node`：双向搜索在该节点交汇

## R13

最小验证清单：
- 起点等于终点
- 无向图可达与不可达
- 有向图可达与不可达
- 结果路径首尾正确（首元素是 `start`，末元素是 `goal`）
- 返回路径中相邻节点确实存在对应边

## R14

常见错误：
- 后向搜索错误地复用正向邻接，导致有向图结果错误
- 路径拼接时把交汇点重复两次
- 只比较当前节点，不比较“新发现邻居”与对侧 visited 的交集
- 未限制按层扩展，破坏 BFS 分层语义

## R15

工程实践建议：
- 图规模很大时，优先选择更小前沿扩展（本实现已采用）
- 若图是静态多次查询，可缓存反向邻接表
- 若要极致性能，可把节点映射成整数并用数组/位图加速 visited

## R16

扩展方向：
- 多终点双向搜索（终点集合）
- 在无权图基础上增加启发式，形成双向 A* 变体
- 对超大图接入外存或分布式前沿管理

## R17

工具栈与职责：
- `Python 标准库`：`collections.deque` 管理 BFS 队列，`dataclasses` 封装结果
- 无第三方依赖，便于在最小环境直接运行与验证算法行为

## R18

源码级算法流程拆解（以本目录 `demo.py` 为准，8 步）：
1. `bidirectional_search` 先校验 `start/goal` 是否在图中；若 `start == goal` 直接返回长度为 1 的路径。
2. 通过 `_normalize_graph` 统一邻接表格式，并在 `directed=True` 时调用 `_build_reverse_graph` 构造反向图。
3. 初始化两侧队列、访问集合与父指针：前向 `front_*` 与后向 `back_*`。
4. 进入主循环，每轮比较两侧队列长度，调用 `_expand_one_layer` 扩展较小前沿的一层。
5. `_expand_one_layer` 对该层节点逐个出队，遍历邻居；若邻居未访问则记录父指针并入队。
6. 在扩展过程中，一旦发现当前节点或新邻居已存在于对侧 visited，立即返回交汇点 `meet`。
7. 主循环拿到 `meet` 后调用 `_construct_path`：分别沿两侧父指针回溯，并把 `start -> meet` 与 `meet -> goal` 拼成完整路径。
8. 若任一队列耗尽仍未相遇，返回 `found=False` 与空路径，表示两点不可达。
