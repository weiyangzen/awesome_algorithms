# 动态树 - Link-Cut Tree

- UID: `MATH-0511`
- 学科: `数学`
- 分类: `数据结构`
- 源序号: `511`
- 目标目录: `Algorithms/数学-数据结构-0511-动态树_-_Link-Cut_Tree`

## R01

Link-Cut Tree（LCT）是一种维护“动态森林（可连边/断边）”的经典数据结构。它适合以下混合操作场景：

- `link(u, v)`：把两棵不同树通过一条边连接；
- `cut(u, v)`：删除一条现有树边；
- `connected(u, v)`：判断两点是否连通；
- `path_sum(u, v)`：查询路径上点权和；
- `set_value(x, val)`：单点改权。

本目录给出一个可运行、可审计、无黑盒依赖的 Python MVP（核心算法手写，不调用第三方动态树库）。

## R02

本实现的问题定义：

- 输入：初始节点点权数组（1-based），以及动态森林操作序列（连边、断边、连通性查询、路径和查询、点权修改）。
- 约束：图始终保持森林（`link` 不允许制造环），`cut` 只删除真实存在的边，`path_sum` 仅在两点连通时有定义。
- 输出：确定性样例的关键查询结果、随机回归测试通过信息、最终成功标志。

`demo.py` 无需交互输入，直接运行即可。

## R03

核心结构思想（Sleator-Tarjan 动态树框架）：

- 用“实树（represented tree）”表达原森林结构；
- 用“辅助 Splay（aux tree）”维护偏好路径；
- `access(x)` 把根到 `x` 的路径改造成一条可操作的偏好路径；
- `make_root(x)` 通过翻转懒标记把 `x` 变为实树根；
- 在 Splay 节点上维护路径聚合量（本实现为点权和 `sum`）。

通过上述机制，路径查询与结构修改可在均摊 `O(log n)` 内完成。

## R04

高层算法流程：

1. 初始化每个点为独立树，维护 `ch/fa/rev/val/sum`。  
2. `connected(u,v)`：比较两点实树根是否相同。  
3. `link(u,v)`：先 `make_root(u)`，若 `u` 与 `v` 已连通则拒绝；否则挂接。  
4. `cut(u,v)`：`make_root(u)` 后 `access(v)`，检查 `u-v` 是否直接边，再删除。  
5. `path_sum(u,v)`：`make_root(u)` + `access(v)`，答案即 `sum[v]`。  
6. `set_value(x,new)`：`access(x)` 后更新点权并回收聚合。

## R05

`demo.py` 的核心状态数组（1-based）：

- `ch[x][0], ch[x][1]`：辅助 Splay 左右子；
- `fa[x]`：父指针（用于 Splay/辅助树关系）；
- `rev[x]`：路径翻转懒标记；
- `val[x]`：节点点权；
- `sum[x]`：Splay 子树聚合和。

核心内部函数：

- `_is_root`、`_push`、`_pull`、`_rotate`、`_splay`；
- `_access`、`_make_root`、`_find_root`。

## R06

正确性要点：

- Splay 局部正确性：`_rotate` 保持中序结构并在旋转后 `pull`，`_splay` 在旋转前按祖先链 `push`，因此聚合值与懒标记传播一致。
- 动态树语义正确性：`make_root` 负责路径翻转，`link` 先判环确保森林约束，`cut` 只在“直接边”判定成立时断边，避免误删路径中非边关系。
- 路径聚合正确性：`path_sum(u,v)` 在 `make_root(u); access(v)` 后，`v` 的辅助树对应整条 `u-v` 路径，故 `sum[v]` 即答案。

## R07

复杂度（`n` 为节点数，`m` 为操作数）：

- 单次 `link/cut/connected/path_sum/set_value`：均摊 `O(log n)`；
- 总体 `m` 次操作：均摊 `O(m log n)`；
- 空间复杂度：`O(n)`。

注：LCT 的复杂度结论是均摊意义，不是每一步严格最坏界。

## R08

边界与异常处理：

- 非法节点编号（不在 `1..n`）会抛 `IndexError`；
- `link(u,u)`、`cut(u,u)` 直接返回 `False`；
- `link` 若会成环返回 `False`；
- `cut` 若目标边不存在返回 `False`；
- `path_sum` 若两点不连通抛 `ValueError`。

随机回归会把 LCT 与暴力森林模型逐步对拍，及时捕获边界错误。

## R09

MVP 取舍：

- 仅实现“点权路径和”这一种聚合，不扩展到 `min/max/xor`；
- 仅实现森林版本（不支持一般图）；
- 不引入外部动态树黑盒库，所有核心逻辑在 `demo.py` 源码中展开；
- 使用 `numpy` 仅做可复现随机数据生成与测试驱动，不替代算法主体。

## R10

`demo.py` 函数职责：

- `LinkCutTree.__init__`：初始化数组状态；
- `_is_root/_push/_pull`：维护 Splay 基础不变量；
- `_rotate/_splay`：Splay 旋转与伸展；
- `_access/_make_root/_find_root`：动态树核心路径操作；
- `connected/link/cut`：动态连通结构操作；
- `set_value/path_sum`：点修改与路径聚合查询；
- `brute_connected/brute_path_sum`：暴力基线；
- `run_deterministic_case`：固定样例；
- `run_randomized_regression`：随机对拍；
- `main`：串联执行。

## R11

运行方式：

```bash
cd Algorithms/数学-数据结构-0511-动态树_-_Link-Cut_Tree
uv run python demo.py
```

脚本不会读取用户输入，也不依赖命令行参数。

## R12

输出说明：

- `[Case] deterministic`：固定演示开始；
- `path_sum(...)=...`：路径和查询结果；
- `after set/link/cut ...`：结构更新后结果；
- `deterministic case passed`：固定样例通过；
- `[Case] randomized regression`：随机回归开始；
- `randomized regression passed (...)`：随机对拍通过；
- `Link-Cut Tree MVP finished successfully.`：脚本完整成功。

## R13

内置测试策略：

- 确定性样例：手工构造小树并交替执行 `link/cut/set_value/path_sum/connected`，对关键路径和做断言。
- 随机回归：随机混合四类操作（连边、断边、查询、改权），与暴力森林 `adj + BFS` 对拍连通性和路径和，最后做全点对一致性检查。

该策略能覆盖大部分实现错误（懒标记、旋转、断边判定、路径暴露等）。

## R14

可调参数：

- `run_randomized_regression(seed=..., rounds=...)`：随机种子与轮数；
- `n`：回归节点规模；
- 随机值范围：初始点权与更新点权区间。

调参建议：

- 快速冒烟：减小 `rounds`；
- 压力测试：增大 `n` 与 `rounds`；
- 复现实验：固定 `seed`。

## R15

与常见替代方案对比：

- 对比“重链剖分 + 线段树”：更适合静态树或边变动较少场景，对动态连边断边不自然。
- 对比“Euler Tour Tree（ETT）”：也可处理动态森林连通，但路径聚合设计通常更复杂。
- LCT 优势：原生支持 `link/cut` 与路径操作，在动态树问题中表达力更强。

## R16

典型应用：

- 在线维护森林网络拓扑（连接/断开）；
- 动态 MST/动态图算法中的子结构维护；
- 竞赛与面试中的高级动态树题；
- 需要频繁路径查询且树结构不断变化的工程问题。

## R17

可扩展方向：

- 把 `sum` 扩展为多种聚合（如 `max/min/xor`）；
- 支持边权版本（拆点或边映射到节点）；
- 增加路径区间懒更新（如路径加法）；
- 增加 `subtree` 相关查询（需改造虚子树信息维护）；
- 针对性能做迭代优化与批量操作封装。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 先跑 `run_deterministic_case`，再跑 `run_randomized_regression`。  
2. 构造 `LinkCutTree` 时初始化 `ch/fa/rev/val/sum`，每个点先自成一棵树。  
3. 每次结构操作前通过 `_access` + `_splay` 暴露并维护偏好路径，`_push/_pull` 维持懒标记与聚合正确。  
4. `link(u,v)` 先 `make_root(u)`，再用 `find_root(v)` 判环，合法时令 `fa[u]=v`。  
5. `cut(u,v)` 执行 `make_root(u); access(v)`，仅当判定为直接边时删除 `v` 的左孩子引用。  
6. `path_sum(u,v)` 先检查连通，再 `make_root(u); access(v)`，读取 `sum[v]` 作为路径和。  
7. `set_value(x,new)` 在 `access(x)` 后更新 `val[x]`，并 `pull` 回写聚合值。  
8. 随机回归中同一操作序列同步作用于暴力森林模型，逐步断言 LCT 与暴力结果一致，最后全点对复核。
