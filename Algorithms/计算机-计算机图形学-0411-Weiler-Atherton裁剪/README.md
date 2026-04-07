# Weiler-Atherton裁剪

- UID: `CS-0254`
- 学科: `计算机`
- 分类: `计算机图形学`
- 源序号: `411`
- 目标目录: `Algorithms/计算机-计算机图形学-0411-Weiler-Atherton裁剪`

## R01

Weiler-Atherton 裁剪算法用于**多边形与多边形**的裁剪，尤其适合处理凹多边形场景。与只面向凸裁剪窗口的算法相比，它通过“交点插入 + 双链表遍历”的方式构造交集边界，能够输出 0 个、1 个或多个结果多边形。

## R02

本题目标是实现 `subject polygon ∩ clip polygon`：

- 输入: 两个简单多边形顶点序列（默认首尾隐式相连）。
- 输出: 交集多边形列表，每个多边形是一个顶点序列。
- 允许输出为空（无交集）。

`demo.py` 给出一个可运行 MVP，不依赖交互输入。

## R03

核心思想（Weiler-Atherton 交集模式）：

1. 计算主体多边形与裁剪多边形边段之间的所有交点。
2. 将交点按边参数顺序插回两个多边形边界序列。
3. 每个交点在两个多边形中各有一个节点，并互相建立 `neighbor` 映射。
4. 给主体多边形上的交点标注 `entry/exit`（进入/离开裁剪区）。
5. 从未访问的 `entry` 交点出发，沿主体边界与裁剪边界交替追踪，得到闭合交集环。

## R04

几何判定基础：

- 线段求交: 使用二维叉积参数方程，求 `t,u`，当 `t,u ∈ [0,1]` 时相交。
- 点在多边形内: 使用 ray casting，并将“点在线段上”视为 inside（边界内含）。
- 交点前后 inside 状态变化: 用交点前后微小扰动点判断 entry/exit。

## R05

实现数据结构（`demo.py`）：

- `Node`: 多边形边界节点
  - `point`: 坐标
  - `is_intersection`: 是否交点
  - `neighbor`: 指向另一个多边形中的同一交点
  - `next/prev`: 环形双向链
  - `entry`: 仅主体交点使用
  - `visited`: 遍历标记
- `IntersectionRecord`: 交点记录
  - 保存交点坐标、所在边编号、参数值、以及两侧节点引用。

## R06

算法流程（MVP 版本）：

1. 枚举所有主体边与裁剪边，求线段交点。
2. 按边参数排序并把交点插入两侧边界链表。
3. 建立交点对 (`s_node <-> c_node`)。
4. 在主体链上标注 entry。
5. 以 `entry && 未访问` 的交点为起点追踪闭环。
6. 去除连续重复点，收集有效多边形（顶点数 >= 3）。
7. 若无交点，执行包含关系回退：
   - subject 在 clip 内 -> 输出 subject
   - clip 在 subject 内 -> 输出 clip
   - 否则空集

## R07

伪代码：

```text
records = intersect_all_edges(subject, clip)
if records empty:
    return containment_fallback(subject, clip)

subject_nodes = insert_intersections(subject, records_on_subject_edges)
clip_nodes    = insert_intersections(clip, records_on_clip_edges)
pair_neighbors(records)
mark_entry_flags(subject_nodes, clip)

result = []
for start in subject_nodes where start.is_intersection and start.entry and not visited:
    loop = trace_by_switching(start)
    if valid(loop):
        result.append(loop)

if result empty:
    return containment_fallback(subject, clip)
return result
```

## R08

正确性直觉：

- 交点插入后，边界拓扑由“几何边段”转成“可遍历节点链”。
- entry 标注保证从“进入裁剪区”的点开始追踪，走到下一个交点时切换边界，始终沿交集边界前进。
- `visited` 防止重复生成同一闭环。
- 无交点时交集仅可能是“包含”或“相离”，由包含测试覆盖。

## R09

复杂度（`n`=subject 顶点数, `m`=clip 顶点数, `k`=交点数）：

- 求交: `O(nm)`
- 每条边内交点排序: 总计约 `O(k log k)`
- 构图与遍历: `O(n + m + k)`
- 总体: `O(nm + k log k)`

## R10

数值稳定性处理：

- 统一使用 `EPS` 比较，避免浮点边界抖动。
- `point_in_polygon` 先判“在线段上”，再 ray casting。
- 参数 `t,u` 会夹取到 `[0,1]`，减少微小超界。
- 连续重复点通过 `dedupe_polygon` 清理。

## R11

`demo.py` 的工程取舍：

- 使用 `numpy` 做向量运算，代码短且可读。
- 不使用 shapely / clipper 等黑盒库，核心算法可见。
- 保持最小可运行：单文件、无命令行参数、固定示例直接打印结果。

## R12

运行方式：

```bash
uv run python Algorithms/计算机-计算机图形学-0411-Weiler-Atherton裁剪/demo.py
```

脚本会输出：输入顶点数量、结果多边形数量、每个结果多边形顶点与有向面积。

## R13

示例输出解读：

- `output polygons: 0` 表示无交集。
- `output polygons: 1` 是常见情况。
- `output polygons > 1` 表示交集被裁剪成多个不连通区域。
- `signed area` 的正负仅代表顶点方向（逆时针为正、顺时针为负），面积绝对值才是大小。

## R14

建议最小测试集：

1. 相离：两个多边形完全分开，应输出空。
2. 包含：subject 完全在 clip 内，应输出 subject。
3. 包含（反向）：clip 完全在 subject 内，应输出 clip。
4. 凹多边形 + 矩形裁剪：应得到合理折线边界。
5. 多交点场景：检查是否能输出多个交集多边形。

## R15

与常见裁剪算法对比：

- Sutherland-Hodgman: 实现更简洁，但经典版本常用于凸裁剪窗口。
- Weiler-Atherton: 实现更复杂，但对一般多边形（尤其凹形）表达能力更强。
- 扫描线/布尔库: 更强大更鲁棒，但工程复杂度与依赖更高。

## R16

可扩展方向：

- 支持并运算 / 差运算 / 异或运算。
- 支持孔洞（外环 + 内环）与多轮廓输入。
- 更完善的退化处理（共线重叠、顶点相切、重复边）。
- 输出标准化（统一方向、去共线点、拓扑校验）。

## R17

当前 MVP 已知限制：

- 未完整处理“共线重叠边”布尔运算细节。
- 对“恰好在顶点相切”的退化情况，仅做基础容错，未做完整拓扑分类。
- 默认输入为简单多边形（无自交）。

这些限制是为了保持最小实现可读、可跑、可验证。

## R18

源码级算法追踪（`demo.py`，非黑盒）可拆为 8 步：

1. `segment_intersection`：对每对边计算参数交点 `t,u` 与几何点坐标。
2. `weiler_atherton_intersection`：双层循环收集全部 `IntersectionRecord`，并按边归档到 `s_hits/c_hits`。
3. `build_polygon_nodes`：在每条边上按参数升序插入交点节点，构建环形双向链。
4. 交点配对：把同一个几何交点在两条链中的节点互相设为 `neighbor`。
5. `classify_entry_exit`：在主体链上比较交点前后 inside 状态，打 `entry` 标记。
6. 从每个未访问 `entry` 出发沿链前进；遇交点时通过 `neighbor` 切换到另一条链继续追踪。
7. `dedupe_polygon`：清理连续重复点，过滤无效环，形成交集多边形集合。
8. 若没有可追踪环，则回退到包含关系判定，覆盖“无交点但存在包含”的情况。
