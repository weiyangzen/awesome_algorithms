# Delaunay三角剖分

- UID: `MATH-0189`
- 学科: `数学`
- 分类: `计算几何`
- 源序号: `189`
- 目标目录: `Algorithms/数学-计算几何-0189-Delaunay三角剖分`

## R01

Delaunay 三角剖分用于把二维散点连接成“不瘦长”的三角网，常见定义是：
- 任意三角形的外接圆内部不包含其它输入点（空圆性质）；
- 与 Voronoi 图互为对偶结构。

本条目提供一个可运行 MVP：
- 优先用 `scipy.spatial.Delaunay`（Qhull）生成三角网；
- 若运行环境无 SciPy，则自动回退到内置 Bowyer-Watson 增量法；
- 用显式几何谓词复核空圆性质（不把库结果当不可审计黑盒）；
- 输出三组固定样例的网格规模与质量摘要。

## R02

问题定义（本实现）：
- 输入：二维点集 `P = {(x_i, y_i)}`，数组形状 `n x 2`。
- 输出：三角形索引集合 `T = {(i, j, k)}`，其中每个索引指向去重后的点集。

实现约束：
- 自动去重并处理退化输入（点数不足、全共线）。
- 三角形统一整理为逆时针（CCW）顶点顺序。
- 脚本无交互输入，运行后直接打印结果。

## R03

核心数学性质：
1. 空圆性质：对任意三角形 \(\triangle abc\)，其外接圆内部不得包含其它点。
2. 最大化最小角直觉：Delaunay 倾向避免“极瘦三角形”，有利于数值计算稳定性。
3. Voronoi 对偶：Delaunay 边连接对应 Voronoi 邻区的生成点。
4. 局部合法性：在凸四边形中，若对角线满足空圆条件，则对应局部连接是 Delaunay 合法连接。

## R04

算法策略（本 MVP）：
1. 输入校验 + 去重。
2. 若点集退化（`n < 3` 或全共线）直接返回空结果。
3. 若 SciPy 可用，调用 `scipy.spatial.Delaunay`（底层 Qhull）生成 `simplices`；否则执行内置 Bowyer-Watson。
4. 将每个三角形整理成 CCW 并剔除近零面积退化片。
5. 统计边集、顶点度、空圆性质校验结果。

这样保持代码简洁，同时保留几何结果可验证性。

## R05

数据结构：
- `Triangle = (i, j, k)`：三角形索引三元组。
- `DelaunayResult`：
  - `unique_points: np.ndarray` 去重后点集；
  - `triangles: list[Triangle]` 三角形列表；
  - `removed_duplicates: int` 去重数量；
  - `used_qhull: bool` 是否实际调用到 Qhull。

## R06

`demo.py` 函数职责：
- `check_points`：输入形状与有限性检查。
- `deduplicate_points`：按容差去重。
- `are_all_collinear`：全共线检测。
- `build_delaunay`：调用 SciPy Delaunay 并规范化输出。
- `circumcircle_contains` / `verify_empty_circumcircle`：空圆性质复核。
- `build_edge_set` / `compute_vertex_degree_stats`：网格统计。
- `summarize_case` / `main`：组织样例并打印摘要。

## R07

复杂度分析：
- Qhull 在二维 Delaunay 的平均复杂度通常可视作接近 `O(n log n)`；
- 本脚本额外空圆复核为朴素 `O(|T| * n)`，用于教学与审计；
- 边统计与度统计约 `O(|T|)`。

说明：MVP 优先“可读 + 可验证”，没有实现大规模场景下的加速校验结构。

## R08

数值稳定性处理：
- `eps` 同时用于去重、共线阈值、退化三角形过滤、空圆判定容差。
- 对局部近共线三角形，按面积阈值剔除，避免污染统计。
- 输入坐标尺度差异很大时，建议先做归一化再三角剖分。

## R09

异常与边界处理：
- 非 `n x 2` 输入或含 `nan/inf`：抛 `ValueError`。
- 去重后点数 `< 3`：返回空三角网。
- 全共线：返回空三角网。
- Qhull 报错（如近退化配置）：自动切换到 Bowyer-Watson 路径继续求解。

## R10

运行方式：

```bash
python3 demo.py
```

脚本不读取命令行参数，不需要交互输入。

## R11

输出字段说明：
- `input points`：原始输入点数。
- `unique points`：去重后点数及去重数量。
- `triangles`：三角形数量。
- `edges`：无向边数量。
- `vertex degree`：顶点度最小/均值/最大。
- `all triangles CCW`：方向一致性检查。
- `empty circumcircle check`：空圆性质是否通过。
- `max incircle determinant`：复核中最大内点行列式（越小越安全）。

## R12

内置测试样例：
- `Uniform random`：24 个均匀随机点。
- `Noisy grid`：规则网格加小扰动，考察接近结构化分布。
- `With duplicates`：包含重复点与近邻点，检验去重流程。
- `Collinear fallback`：共线样例，预期返回 0 个三角形。

## R13

结果解读建议：
- `empty circumcircle check=True` 说明输出满足 Delaunay 核心判据（在给定容差下）。
- 顶点平均度通常在平面三角网中保持较稳定数量级，过高可能暗示输入分布过密局部聚簇。
- 共线样例返回空网格是正确行为，不应视为失败。

## R14

参数调优建议：
- `eps` 偏大：可能误删合法近邻点，三角形数量偏少。
- `eps` 偏小：近退化配置更容易触发数值噪声或 Qhull 异常。
- `qhull_options` 可按数据特点调整（例如共圆/退化数据可加入 `QJ` 抖动策略）；无 SciPy 时该参数不生效。

## R15

与相关方法对比：
- 对比 `Bowyer-Watson` 手写增量法：
  - 手写法流程透明，便于教学；
  - Qhull 工程鲁棒性和性能更成熟。
- 对比仅做普通三角化：
  - Delaunay 的空圆性质通常带来更好的三角形形状质量。

## R16

典型应用：
- 空间插值（TIN 地形、传感器场重建）。
- 网格生成与有限元前处理。
- 最近邻关系近似图构造（与 Voronoi 结构联动）。
- 计算机图形中的表面离散化基础模块。

## R17

当前实现限制：
- 仅处理二维欧式平面点集。
- 空圆复核采用朴素全点扫描，适合中小规模数据。
- 不含约束边（CDT）与带洞区域约束。
- 非 CAD 级稳健内核；若需强鲁棒性，应接入精确几何谓词库。

## R18

`demo.py` 源码级流程拆解（双路径，9 步）：
1. `main` 构造固定测试点集并逐个调用 `summarize_case`。
2. `summarize_case` 先进入 `build_delaunay`，执行输入检查、去重、共线判定。
3. `build_delaunay` 优先走 `triangulate_with_qhull`；若 SciPy 不可用或 Qhull 抛错，则切换 `bowyer_watson_triangulation`。
4. Qhull 路径中，`scipy.spatial.Delaunay(points, qhull_options=...)` 在 `_qhull.pyx` 将点阵转为连续内存并组装命令参数。
5. SciPy 包装层调用 Qhull 核心入口（`qh_new_qhull`）；Qhull 通过“抛物面提升 + 凸包”提取下凸壳，映射回二维 Delaunay 单形。
6. Qhull 返回的 `simplices` 被逐个 CCW 规范化，过滤近零面积三角形。
7. 回退路径中，Bowyer-Watson 从超三角形启动，逐点删除坏三角形、提取空腔边界并局部重建。
8. 两条路径最终都产出统一的 `DelaunayResult`，并记录 `used_qhull` 标记。
9. `verify_empty_circumcircle` 使用显式 `incircle` 行列式逐三角形复核空圆性质，最后打印质量统计。
