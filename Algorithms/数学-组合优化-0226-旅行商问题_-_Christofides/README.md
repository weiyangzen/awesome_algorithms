# 旅行商问题 - Christofides

- UID: `MATH-0226`
- 学科: `数学`
- 分类: `组合优化`
- 源序号: `226`
- 目标目录: `Algorithms/数学-组合优化-0226-旅行商问题_-_Christofides`

## R01

Christofides 算法用于求解**度量旅行商问题**（Metric TSP）的近似解：
- 输入为满足三角不等式的完全图（或可等价看作度量空间上的点集）；
- 输出一条哈密顿回路；
- 理论上保证回路长度不超过最优解 `1.5` 倍。

本目录给出一个可运行、可验证的 MVP：在欧氏距离实例上实现 Christofides 全流程，并用 Held-Karp 精确解对比近似比。

## R02

问题定义（MVP 范围）：
- 输入：`n` 个二维点（`n >= 3`），边权为欧氏距离；
- 输出：
  - Christofides 回路与长度；
  - MST 权重、奇度点集合、最小完美匹配权重；
  - 精确最优解（Held-Karp）与近似比 `Chr/Opt`。

约束：脚本无交互输入，使用固定种子自动生成多个实例并打印结果。

## R03

理论背景（Metric TSP）：
- TSP 是 NP-hard；
- 对一般 TSP 很难给出常数近似保证；
- 对满足三角不等式的度量 TSP，Christofides 可保证 `3/2` 近似。

保证依赖的关键事实：
1. 最小生成树 `MST` 权重不超过最优回路 `OPT`；
2. MST 奇度顶点数为偶数；
3. 在奇度点诱导完全图上做最小权完美匹配，匹配权重 `<= OPT / 2`；
4. 合并后得到欧拉图并 shortcut，不会因三角不等式增大路径长度。

## R04

算法主流程：
1. 计算完整距离矩阵；
2. 在完整图上求 MST；
3. 提取 MST 的奇度顶点集合 `O`；
4. 在 `O` 上求最小权完美匹配；
5. 将 MST 边与匹配边合并为欧拉多重图；
6. 用 Hierholzer 求欧拉回路；
7. 对欧拉回路做 shortcut（跳过重复顶点）得到哈密顿回路；
8. 计算最终回路长度。

## R05

本实现的数据结构：
- `numpy.ndarray dist (n x n)`：对称距离矩阵；
- `Edge = Tuple[int, int]`：无向边；
- `List[Edge]`：MST 边、匹配边；
- 多重图邻接表 `List[List[int]]`：支持重边；
- `ChristofidesResult`、`InstanceReport`：结构化输出。

## R06

正确性要点（实现视角）：
- `prim_mst` 保证构造 `n-1` 条连接全体点的最小生成树；
- `odd_degree_vertices` 基于握手定理保证奇度点数为偶数；
- `min_weight_perfect_matching_dp` 在奇度点集上做精确匹配（非贪心）；
- 合并后每个点度数为偶，`eulerian_tour` 可遍历每条边恰一次；
- `shortcut_to_hamiltonian` 只保留首次出现点并回到起点，得到合法 TSP 回路。

## R07

复杂度分析：
- 距离矩阵构建：`O(n^2)`；
- Prim MST（当前矩阵版）：`O(n^2)`；
- 奇度点数设为 `k`（必为偶数），完美匹配 DP：
  - 时间 `O(k^2 * 2^k)`；
  - 空间 `O(2^k)`；
- 欧拉回路与 shortcut：`O(n + |E|)`（这里 `|E|` 为多重图边数）。

因此本 MVP 适合中小规模教学与验证，不是超大规模工程实现。

## R08

边界与异常：
- 点集形状不是 `(n,2)`：抛 `ValueError`；
- `n < 3`：Christofides 部分直接拒绝；
- odd 顶点数异常为奇数：抛 `ValueError`；
- shortcut 后若未覆盖所有顶点：抛 `ValueError`；
- 近似比在度量实例上若超过 `1.5 + eps`：抛 `AssertionError`。

## R09

MVP 设计取舍：
- 只处理欧氏度量 TSP（天然满足三角不等式）；
- 完美匹配使用源码级 bitmask DP，避免把核心步骤交给黑盒；
- 为了可验证性，额外实现 Held-Karp 精确解做对照；
- 未实现大规模 Blossom matching 或并行加速。

## R10

`demo.py` 模块职责：
- `euclidean_distance_matrix`：从点构建度量矩阵；
- `prim_mst`：最小生成树；
- `odd_degree_vertices`：奇度点识别；
- `min_weight_perfect_matching_dp`：奇度点最小完美匹配（精确 DP）；
- `eulerian_tour` + `shortcut_to_hamiltonian`：欧拉化到哈密顿化；
- `held_karp_exact_tsp`：小规模精确最优对照；
- `run_one_instance` / `main`：批量运行与断言验证。

## R11

运行方式：

```bash
cd Algorithms/数学-组合优化-0226-旅行商问题_-_Christofides
python3 demo.py
```

脚本会自动运行 3 个固定随机种子的欧氏实例，不需要任何输入。

## R12

输出字段解释：
- `Odd-degree vertices in MST`：MST 中奇度点集合；
- `MST weight`：MST 总权；
- `Matching weight`：奇度点最小完美匹配权重；
- `Christofides length`：近似回路长度；
- `Optimal length (Held-Karp)`：精确最优长度；
- `Approx ratio (Chr/Opt)`：近似比（应 `<= 1.5`）；
- `Baseline ratio (NN/Opt)`：最近邻基线比值。

## R13

最小测试关注点：
1. 可行性：回路长度应为 `n+1`，首尾节点相同，且中间节点不重复；
2. 质量：`Chr/Opt <= 1.5 + 1e-9`；
3. 对比：Christofides 一般优于或接近最近邻启发式；
4. 稳定性：固定种子下结果可复现。

## R14

关键可调参数：
- `configs`：`(n, seed)` 列表，控制实例规模与随机性；
- 点坐标范围：`uniform(0, 100)`；
- 近似比检查容差：`1e-9`。

建议：若想更快运行，可降低 `n`；若要更强对照，可增加实例数量但注意 Held-Karp 指数复杂度。

## R15

与其他方法比较：
- 最近邻/2-opt：实现简单但无统一常数近似保证；
- Christofides：在度量 TSP 下有 `1.5` 理论保证；
- 精确算法（Held-Karp / Branch-and-Bound）：能得最优，但大规模成本高。

本条目把 Christofides 与 Held-Karp 并置，强调“有保证近似”和“可验证最优”的关系。

## R16

应用场景：
- 配送路径规划（满足近似度量成本时）；
- 巡检、采样、测绘等回路规划；
- 组合优化课程中“近似算法 + 保证证明”教学；
- 作为更复杂局部搜索/元启发式（如 Lin-Kernighan）的初始化解。

## R17

可扩展方向：
- 将完美匹配从指数 DP 替换为 Blossom（支持更大奇度点集合）；
- 使用 KD-tree/稀疏候选边减少构图与后处理成本；
- 叠加 2-opt/3-opt 做后优化；
- 扩展到非欧氏但满足三角不等式的业务成本矩阵。

## R18

源码级算法流（`demo.py`，8 步）：
1. `main` 读取固定 `configs`，逐个调用 `run_one_instance` 构建欧氏 TSP。  
2. `make_euclidean_instance` 生成点集并由 `euclidean_distance_matrix` 得到完整度量矩阵。  
3. `christofides_tsp` 内先调用 `prim_mst` 得到 MST，再由 `odd_degree_vertices` 提取奇度顶点。  
4. `min_weight_perfect_matching_dp` 在奇度顶点诱导完全图上做 bitmask 记忆化搜索，返回精确最小完美匹配。  
5. `build_multigraph` 合并 MST 与匹配边得到欧拉多重图，`eulerian_tour`（Hierholzer）求欧拉回路。  
6. `shortcut_to_hamiltonian` 跳过欧拉回路中的重复顶点并闭环，得到 Christofides 哈密顿回路；`tour_length` 计算长度。  
7. 同一实例上 `held_karp_exact_tsp` 求精确最优回路，并用 `nearest_neighbor_tsp` 给出基线启发式。  
8. `run_one_instance` 做可行性与 `1.5` 比值断言，`print_report` 输出关键中间量与最终对比指标。  
