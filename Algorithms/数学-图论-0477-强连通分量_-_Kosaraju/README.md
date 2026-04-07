# 强连通分量 - Kosaraju

- UID: `MATH-0477`
- 学科: `数学`
- 分类: `图论`
- 源序号: `477`
- 目标目录: `Algorithms/数学-图论-0477-强连通分量_-_Kosaraju`

## R01

本条目实现 `Kosaraju` 算法的最小可运行版本（MVP），用于在**有向图**中识别全部强连通分量（SCC, Strongly Connected Components）。
目标是：
- 给出每个 SCC 的节点集合；
- 给出每个节点所属的分量编号；
- 给出 SCC 压缩后的 DAG 边集（便于后续分析拓扑结构）；
- 保证 `python3 demo.py` 可直接运行并输出结果。

## R02

问题定义（MVP 范围）：
- 输入：
  - 节点数量 `n`；
  - 有向边集 `edges = (u, v)`。
- 输出：
  - `components`：SCC 列表，每个元素是一个分量内的节点列表；
  - `component_of[x]`：节点 `x` 对应的 SCC 编号；
  - `condensation_edges`：分量图（压缩 DAG）中的有向边 `(cid_u, cid_v)`。

约束：
- 节点编号范围为 `[0, n-1]`；
- `n > 0`；
- 图允许不连通、允许重边输入（构图时去重）。

## R03

数学基础与关键性质：

1. 在有向图 `G=(V,E)` 中，若 `u` 可达 `v` 且 `v` 可达 `u`，则 `u,v` 属于同一强连通分量。  
2. 全体 SCC 可把顶点集划分为互不相交的等价类。  
3. 将每个 SCC 缩成一个点可得到**分量图**（condensation graph），该图一定是 DAG。  
4. Kosaraju 基于如下事实：
   - 第一趟 DFS 在 `G` 上按“完成时间”给出顺序；
   - 第二趟在反图 `G^T` 上按完成时间逆序启动 DFS，每次恰好提取一个 SCC。

## R04

算法流程（Kosaraju MVP）：

1. 根据边集建立有向图邻接表 `G`。  
2. 构建反图 `G^T`。  
3. 在 `G` 上做第一趟 DFS，按节点“完成时刻”写入 `order`。  
4. 将 `order` 逆序遍历。  
5. 对每个未访问节点，在 `G^T` 上做一次 DFS，收集得到一个完整 SCC。  
6. 为该 SCC 分配编号，并写入 `component_of`。  
7. 扫描原图边 `(u,v)`，若 `component_of[u] != component_of[v]`，则在压缩图加入边。

## R05

核心数据结构：
- `List[List[int]]`：邻接表（原图与反图）；
- `List[int]`：
  - `order`（第一趟 DFS 完成序）；
  - `component_of`（节点到分量编号映射）；
- `List[List[int]]`：`components`（每个 SCC 的节点集）；
- `set[Tuple[int,int]]`：临时去重的压缩图边集；
- `SCCResult`（`dataclass`）：统一封装算法输出。

## R06

正确性要点：
- 第一趟 DFS 的完成序能够把“在分量 DAG 中靠后的分量”优先放到 `order` 尾部；
- 第二趟在 `G^T` 中按逆序启动 DFS，第一次可达区域恰好对应一个 SCC，且不会越界进入尚未处理的其他 SCC；
- 每个节点在第二趟仅被访问一次，因此每个节点被分配且只分配到一个分量；
- 对跨分量边进行压缩后得到的图无环，符合 SCC 分解理论。

## R07

复杂度分析：
- 时间复杂度：`O(|V| + |E|)`
  - 两次 DFS 各 `O(|V| + |E|)`；
  - 压缩边扫描 `O(|E|)`。
- 空间复杂度：`O(|V| + |E|)`
  - 原图、反图、访问标记、顺序数组和结果存储。

## R08

边界与异常处理：
- `n <= 0`：抛出 `ValueError`；
- 边端点越界：抛出 `ValueError`；
- 空边集但 `n>0`：合法，结果是 `n` 个单点 SCC；
- 重边：构图阶段去重，不影响 SCC 正确性；
- 非连通有向图：可正常处理，输出多个互不连通分量集合。

## R09

MVP 取舍说明：
- 不依赖 `networkx` 等黑盒 SCC API，直接实现两趟 DFS，便于教学与审计；
- 使用迭代版 DFS（显式栈）而非递归，降低深图下递归深度风险；
- 输出额外提供分量 DAG 边集，方便衔接拓扑排序、DP on DAG 等后续算法；
- 不扩展到动态维护 SCC 或并行图计算，优先保证可读性与可运行性。

## R10

`demo.py` 职责划分：
- `build_graph`：输入校验 + 邻接表构建；
- `reverse_graph`：构造反图；
- `finish_order`：第一趟 DFS 计算完成序；
- `collect_component`：第二趟 DFS 收集单个 SCC；
- `kosaraju_scc`：整体编排并生成压缩图边；
- `print_result`：展示每个样例的 SCC 与压缩 DAG；
- `main`：固定样例入口，无需交互输入。

## R11

运行方式：

```bash
cd Algorithms/数学-图论-0477-强连通分量_-_Kosaraju
python3 demo.py
```

脚本会自动运行两个示例并打印分量拆解结果。

## R12

输出解读：
- `SCC count`：强连通分量总数；
- `SCC#k: [...]`：编号 `k` 的分量内节点列表；
- `component_of(node)`：下标是节点，值是对应分量编号；
- `condensation DAG edges`：压缩后分量图边集，形式为 `(源分量, 目标分量)`。

若某案例输出 `SCC count = 1`，表示整张图强连通。

## R13

建议最小测试集：
- 单节点无边图；
- 线性链图（每个节点应形成单独 SCC）；
- 单大环图（应形成 1 个 SCC）；
- 多个环通过单向桥连接（应得到多个 SCC + 有向压缩边）；
- 非法输入（`n<=0`、越界边）。

## R14

可调参数：
- `n` 与 `edges`（图规模与结构）；
- `print_result` 中案例数量与图形态；
- 节点输出排序策略（当前组件内升序，便于稳定对比）；
- 若需要可加 `random seed` 构造随机图做压力测试。

## R15

方法对比：
- 对比 Tarjan：
  - Tarjan 单次 DFS 即可完成；
  - Kosaraju 逻辑直观，便于解释“原图完成序 + 反图提取”。
- 对比 Gabow：
  - Gabow 也为单趟 DFS 栈法，常数项有时更优；
  - Kosaraju 概念更容易和图转置思想关联。
- 在教学和工程快速实现中，Kosaraju 是稳定且可读性高的选择。

## R16

典型应用场景：
- 依赖关系分析中的循环依赖识别；
- 编译器/构建系统的模块强连通压缩；
- 社交网络中的互达群体识别；
- 状态机、控制流图中的环结构归并。

## R17

可扩展方向：
- 在压缩 DAG 上做拓扑排序与最长路/计数 DP；
- 输出每个 SCC 的入度、出度与规模统计；
- 增加随机图回归测试与基准测试；
- 支持从文件读取大规模图并做批处理输出。

## R18

源码级算法流追踪（对应 `demo.py`，9 步）：

1. `main` 定义固定样例 `(n, edges)`，调用 `print_result` 统一执行。  
2. `print_result` 调 `kosaraju_scc`，进入核心流程。  
3. `kosaraju_scc` 先调用 `build_graph`：检查 `n` 与边端点，构建去重且有序的邻接表 `G`。  
4. 调 `reverse_graph` 构造反图 `G^T`。  
5. 在 `G` 上调用 `finish_order`（第一趟 DFS），把每个节点在“回溯完成”时压入 `order`。  
6. 逆序遍历 `order`；遇到未访问节点就在 `G^T` 上调用 `collect_component`，一次收集一个 SCC。  
7. 每得到一个 SCC，就分配 `cid`，并回填 `component_of[node] = cid`。  
8. 再遍历原图全部边 `(u,v)`，若两端分属不同 SCC，就向 `condensation_edges` 加入 `(cid_u, cid_v)`。  
9. 返回 `SCCResult`，`print_result` 将分量列表、节点归属和压缩 DAG 边集打印出来。
