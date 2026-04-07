# 哈密顿路径/回路

- UID: `MATH-0482`
- 学科: `数学`
- 分类: `图论`
- 源序号: `482`
- 目标目录: `Algorithms/数学-图论-0482-哈密顿路径／回路`

## R01

哈密顿路径（Hamiltonian Path）要求在图中访问每个顶点恰好一次；
哈密顿回路（Hamiltonian Cycle）进一步要求首尾相连形成环。

本条目给出一个最小但完整可运行 MVP：
- 用位压缩动态规划（bitmask DP）做**精确判定**；
- 若存在，返回一条具体路径/回路作为见证；
- 同时支持无向图与有向图；
- `demo.py` 内置多个固定测试图，运行后直接输出结果。

## R02

问题定义（本目录实现）：
- 输入：
  - 图 `G=(V,E)`，顶点编号为 `0..n-1`；
  - 图由 `build_graph_from_edges(num_vertices, edges, directed)` 构造；
  - `directed=False` 表示无向图，`True` 表示有向图。
- 输出：
  - `find_hamiltonian_path(G)`：
    - 若存在哈密顿路径，返回节点序列 `list[int]`（长度 `n`）；
    - 否则返回 `None`。
  - `find_hamiltonian_cycle(G)`：
    - 若存在哈密顿回路，返回节点序列 `list[int]`（长度 `n+1`，且首尾相同）；
    - 否则返回 `None`。

脚本不读取命令行参数，不需要交互输入。

## R03

数学与状态建模：

1. 用一个二进制掩码 `mask` 表示“已访问顶点集合”。
   - 第 `v` 位为 `1` 表示顶点 `v` 已访问。
2. 定义状态 `dp[mask][v]`：
   - 表示存在一条路径，恰好访问 `mask` 中所有顶点并且以 `v` 结尾；
   - 本实现将 `dp` 存为 `parent` 表，记录前驱顶点以便回溯。
3. 转移：
   - 若 `dp[mask][v]` 可达，且存在边 `v -> u` 且 `u` 不在 `mask` 中，
     则 `dp[mask | (1<<u)][u]` 可达。
4. 哈密顿路径判定：
   - `full_mask = (1<<n)-1`；
   - 若存在某个 `v` 使 `dp[full_mask][v]` 可达，则存在哈密顿路径。
5. 哈密顿回路判定：
   - 固定起点 `s`，做同样 DP；
   - 若 `dp[full_mask][v]` 可达且有边 `v -> s`，则存在哈密顿回路。

## R04

算法流程（高层）：

1. 读入图并校验参数合法性。  
2. 预计算每个顶点的邻接位集 `neighbor_bits[v]`。  
3. 对哈密顿路径：把每个单点状态 `dp[1<<v][v]` 置为可达。  
4. 枚举 `mask` 与 `end`，按邻接关系扩展到未访问顶点。  
5. 到达 `full_mask` 后检查是否有终点可达。  
6. 若可达，依据 `parent` 表回溯并反转得到完整路径。  
7. 对哈密顿回路：固定起点 `s` 重复步骤 3-6。  
8. 若终点 `v` 与 `s` 有边相连，则在路径末尾追加 `s` 形成回路。

## R05

核心数据结构：

- `Graph`：
  - `adjacency: np.ndarray`，`n x n` 邻接矩阵（`0/1`）；
  - `directed: bool`，有向/无向标记。
- `neighbor_bits: list[int]`：
  - `neighbor_bits[v]` 的二进制位表示 `v` 可达的下一顶点集合。
- `parent: np.ndarray(shape=(2^n, n), dtype=int16)`：
  - `-2`（`UNREACHABLE`）表示状态不可达；
  - `-1`（`START_SENTINEL`）表示路径起点；
  - `0..n-1` 表示前驱顶点编号。

## R06

正确性要点：

- 完备性：
  - DP 枚举了所有“已访问集合 + 终点”的合法状态，
  - 因此不会漏掉任何哈密顿路径候选。
- 合法性：
  - 转移只允许沿真实边前进，且只向未访问顶点扩展，
  - 保证“每顶点最多访问一次”。
- 终止判据正确：
  - 到达 `full_mask` 等价于“全部顶点都访问过”。
- 回路判据正确：
  - 在 `full_mask` 基础上额外检查 `end -> start` 边，
  - 恰好对应哈密顿回路定义。
- 见证可回溯：
  - `parent` 存储前驱，能恢复一条实际路径而非仅布尔结论。

## R07

复杂度分析（`n=|V|`）：

- 状态数：`2^n * n`。  
- 每个状态最多尝试 `n` 个扩展。  
- 时间复杂度：`O(n^2 * 2^n)`。  
- 空间复杂度：`O(n * 2^n)`（主要是 `parent` 表）。

这是 NP 完全问题的典型精确算法规模，适合小图验证与教学。

## R08

边界与异常处理：

- `num_vertices <= 0`：抛 `ValueError`。  
- 边端点越界：抛 `ValueError`。  
- 自环边：抛 `ValueError`（MVP 不处理）。  
- 图顶点数过大（`n > 20`）：抛 `ValueError`，防止状态爆炸。  
- 无解情况：返回 `None`，而不是抛错。

## R09

MVP 取舍说明：

- 采用精确 DP，不做随机化近似。  
- 采用邻接矩阵 + 位运算，代码短且便于审计。  
- 不引入 NetworkX 等图算法黑盒 API。  
- 仅实现“存在性 + 一条见证路径”，不枚举所有解。

## R10

`demo.py` 主要函数职责：

- `build_graph_from_edges`：从边列表构建图并做输入校验。
- `_neighbor_bitmasks`：将邻接矩阵压缩为位集，提升 DP 转移效率。
- `_reconstruct_path`：根据 `parent` 表回溯并恢复顶点序列。
- `find_hamiltonian_path`：哈密顿路径精确求解（DP）。
- `_find_hamiltonian_cycle_from_start`：固定起点的回路 DP。
- `find_hamiltonian_cycle`：遍历起点并返回任意一个回路见证。
- `is_valid_hamiltonian_path`：校验路径是否合法。
- `is_valid_hamiltonian_cycle`：校验回路是否合法。
- `run_case`：执行单个样例、计时并打印结果。
- `main`：组织内置测试图并批量运行。

## R11

运行方式：

```bash
cd Algorithms/数学-图论-0482-哈密顿路径／回路
python3 demo.py
```

脚本会打印每个样例的路径/回路结果与校验布尔值。

## R12

输出字段说明：

- `=== Case: ... ===`：测试样例名。  
- `directed=..., vertices=...`：图类型与顶点数。  
- `Hamiltonian path: ... / NOT FOUND`：路径结果。  
- `path valid: True/False`：路径校验结果。  
- `Hamiltonian cycle: ... / NOT FOUND`：回路结果。  
- `cycle valid: True/False`：回路校验结果。  
- `path search time (ms)`：路径搜索耗时。  
- `cycle search time (ms)`：回路搜索耗时。

## R13

内置最小测试集：

- `Undirected: cycle exists`：存在哈密顿回路，因此路径也存在。  
- `Undirected: path only`：链式图，存在路径但无回路。  
- `Undirected: none`：图不连通，路径/回路都不存在。  
- `Directed: cycle exists`：有向图场景下验证回路逻辑。

建议补充测试：
- 单顶点图（约定是否视为平凡回路）；
- 稠密随机图与稀疏随机图；
- `n=20` 边界性能测试。

## R14

可调参数与约束：

- `directed`：切换有向/无向模式。  
- `MAX_EXACT_VERTICES`：精确 DP 最大顶点数上限（默认 `20`）。  
- 输入边集合 `edges`：直接决定图结构与求解难度。

调参建议：
- 若只做教学演示，`n<=12` 体验更流畅；
- 若要压力测试，可逐步增加 `n` 并观测 `2^n` 增长带来的耗时变化。

## R15

方法对比：

- 对比回溯 DFS：
  - 回溯实现简单，但最坏情况剪枝不足时会严重爆炸；
  - 位压缩 DP 在“判定 + 见证”任务上更稳定、可复现。
- 对比 ILP/SAT 建模：
  - ILP/SAT 可借助求解器处理更复杂约束；
  - 但本条目强调无黑盒、源码可审计。
- 对比启发式算法（遗传、蚁群等）：
  - 启发式在大图上更快，但不保证精确性；
  - 本实现提供严格正确的存在性结论。

## R16

典型应用场景：

- 图论课程中 NP 完全问题的算法演示。  
- 小规模路径覆盖问题的精确可行性检查。  
- 复杂组合优化前的 baseline 校验器。  
- 用于验证启发式算法输出的正确性（作为真值对照）。

## R17

可扩展方向：

- 返回“所有”哈密顿路径/回路（当前只返回一个）。  
- 加入剪枝策略（割点、度约束、连通性预检）。  
- 支持带权版本并衔接 TSP（最短哈密顿回路）。  
- 改为 meet-in-the-middle 或并行化以提升规模上限。  
- 增加文件输入与批处理基准输出（CSV/JSON）。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 构造四个固定图实例（无向有回路、无向仅路径、无向无解、有向有回路）。
2. `run_case` 对每个图分别调用 `find_hamiltonian_path` 与 `find_hamiltonian_cycle`，并计时。
3. `find_hamiltonian_path` 创建 `parent[2^n][n]` 状态表，初始化单点状态 `parent[1<<v][v] = START_SENTINEL`。
4. `_neighbor_bitmasks` 预计算每个顶点的邻接位集，DP 转移时直接做位运算过滤候选顶点。
5. 在 DP 主循环中，针对每个可达状态 `(mask, end)`，把 `end` 能到达且未访问的顶点 `nxt` 扩展为 `next_mask`，并记录前驱 `parent[next_mask][nxt] = end`。
6. 当某个终点在 `full_mask` 上可达时，`_reconstruct_path` 依据 `parent` 逆向回溯并反转，得到一条哈密顿路径见证。
7. `find_hamiltonian_cycle` 通过 `_find_hamiltonian_cycle_from_start` 固定起点重复 DP，并在 `full_mask` 上检查终点是否能连回起点。
8. `is_valid_hamiltonian_path/is_valid_hamiltonian_cycle` 对返回序列做结构化校验，最终打印“结果 + 校验布尔值 + 毫秒级耗时”。
