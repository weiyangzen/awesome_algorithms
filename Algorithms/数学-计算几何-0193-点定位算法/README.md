# 点定位算法

- UID: `MATH-0193`
- 学科: `数学`
- 分类: `计算几何`
- 源序号: `193`
- 目标目录: `Algorithms/数学-计算几何-0193-点定位算法`

## R01

**问题定义**：给定一组平面多边形（通常视作平面细分的面），对查询点 `q=(x,y)` 判断它属于哪个区域。

本条目采用三类输出关系：
- `INSIDE`：点在某个多边形内部。
- `BOUNDARY`：点落在多边形边界上。
- `OUTSIDE`：不在任意多边形内。

这就是计算几何中的经典 **Point Location（点定位）** 问题。

## R02

**输入/输出约定（MVP）**：
- 输入：
  - `polygons`: `List[List[Tuple[float,float]]]`，每个多边形按顶点顺序给出。
  - `labels`: 与 `polygons` 等长的区域名称。
  - `query point`: 一个二维点。
- 输出：`LocateResult(label, relation, polygon_index)`
  - `label`: 匹配区域名，若失败则为 `OUTSIDE`
  - `relation`: `INSIDE | BOUNDARY | OUTSIDE`
  - `polygon_index`: 命中多边形下标，失败为 `None`

## R03

**核心思路**：
1. 预处理每个多边形的轴对齐包围盒（AABB）。
2. 用全局包围盒构建均匀网格，将每个多边形按包围盒覆盖到若干网格单元。
3. 查询时先定位点所在网格，拿到少量候选多边形。
4. 对候选多边形执行精确几何判定：
   - 先做“点在线段上”边界判断。
   - 再做射线法（奇偶规则）判断 inside/outside。

这样把“先粗筛、再精判”组合起来，降低平均查询代价。

## R04

**数据结构设计**：
- `self.polygons`: `np.ndarray` 列表，形状 `(n_i, 2)`。
- `self.bboxes`: `shape=(m,4)`，每行 `[min_x, min_y, max_x, max_y]`。
- `self.global_bbox`: 全体多边形外包盒。
- `self._grid`: `Dict[(gx,gy), List[polygon_idx]]`，网格到候选面列表。
- `LocateResult`: 统一返回结构，便于批量评估和一致性检查。

## R05

**预处理阶段**：
- 校验输入合法性（至少一个多边形、每个多边形顶点数 >= 3、标签数量一致）。
- 计算每个多边形包围盒与全局包围盒。
- 按 `grid_resolution` 将包围盒映射到网格索引区间 `[gx0..gx1] × [gy0..gy1]`。
- 在对应格子中登记该多边形索引。

预处理时间复杂度近似 `O(m + C)`：
- `m` 为多边形数；
- `C` 为包围盒覆盖网格单元总数。

## R06

**查询阶段**：
1. 若点在全局包围盒外，直接可判为 `OUTSIDE`（实现中也允许回退扫描）。
2. 若在范围内，映射到 `(gx, gy)` 网格，取候选列表。
3. 逐个候选做精确判断：
   - `point_on_segment`：判边界；
   - `point_in_polygon`：射线奇偶翻转。
4. 找到第一个非 `OUTSIDE` 即返回；若都未命中则 `OUTSIDE`。

## R07

**正确性要点**：
- “点在线段上”通过二维叉积接近 0 + 投影范围约束共同判定，避免只靠距离误判。
- 射线法使用 `(yi > y) != (yj > y)` 控制跨越，规避顶点重复计数。
- 先判边界再判内部，避免边界点被归类为 inside/outside 的歧义。

## R08

**复杂度分析**：
- 单次查询（均摊）：`O(k * e_bar)`
  - `k` = 网格候选多边形数（通常远小于 `m`）
  - `e_bar` = 候选多边形平均边数
- 最坏情况：`O(m * e_bar)`（例如网格退化或候选过密）
- 空间复杂度：`O(m + C)`

## R09

**数值稳定性处理**：
- 使用 `eps=1e-9` 处理浮点比较。
- 包围盒宽高取 `max(width, eps)`，避免除零。
- 交点计算分母使用 `yj - yi + eps`，减少水平边附近的数值异常。

说明：这是工程化 MVP 的稳健处理，不是严格符号几何证明级别的鲁棒核。

## R10

**`demo.py` 模块结构**：
- `LocateResult`: 返回对象。
- `PointLocator`:
  - `_build_grid_index`：网格索引预处理
  - `_point_on_segment`：边界判断
  - `_point_in_polygon`：射线法
  - `locate`：网格候选查询
  - `locate_bruteforce`：全扫描基线
- `build_demo_map`: 构造 4 个示例区域。
- `run_demo`: 输出定位结果 + 20,000 次随机点性能对比。

## R11

**运行方式**：

```bash
python3 Algorithms/数学-计算几何-0193-点定位算法/demo.py
```

无需交互输入，直接打印示例查询与基准统计。

## R12

**示例场景说明**：
- 构造 `Zone-A ~ Zone-D` 四个非重叠多边形（含凸与凹形）。
- 查询点覆盖：
  - 内部点（如 `(2.0, 1.0)`）
  - 边界点（如 `(4.0, 2.0)`）
  - 外部点（如 `(11.0, 11.0)`）
- 随机基准点范围：`[-2, 12] x [-2, 12]`。

## R13

**输出解读**：
- `label=Zone-X, relation=INSIDE` 表示点在某一区域内。
- `relation=BOUNDARY` 表示点在边界上。
- `label=OUTSIDE, relation=OUTSIDE` 表示未命中任何多边形。
- `benchmark` 中：
  - `indexed search` 是网格加速版
  - `brute force` 是全扫描版
  - `result mismatch` 应为 `0`，用于一致性校验

## R14

**工程取舍（为何是这个 MVP）**：
- 没有引入复杂平面细分结构（如 trapezoidal map / Kirkpatrick hierarchy）。
- 采用“均匀网格 + 射线法”是因为：
  - 实现短小、可读、易验证；
  - 对中小规模数据有明显提速；
  - 能直接体现点定位算法的核心流程。

## R15

**可扩展方向**：
- 自适应空间索引：R-tree / quadtree 替代固定网格。
- 复杂场景支持：带洞多边形、多层边界、共享边拓扑。
- 批处理优化：向量化或并行化大量查询。
- 严格鲁棒几何：使用有理数核或成熟几何库的稳健谓词。

## R16

**建议测试清单**：
- 正常点：位于凸多边形内部、凹多边形内部。
- 边界点：在线段中点、顶点、水平边附近。
- 外部点：全局包围盒外、包围盒内但不在任何面内。
- 极端参数：超小 `eps`、不同 `grid_resolution`。
- 一致性：`locate` 与 `locate_bruteforce` 结果逐点比对。

## R17

**当前实现边界与限制**：
- 假设多边形面积不重叠；若存在重叠，返回顺序依赖候选扫描顺序。
- 未显式处理“多边形洞”（内环）语义。
- 该实现为教学/工程 MVP，不替代 CAD/GIS 级几何内核。

## R18

**源码级算法流程拆解（非黑盒，8 步）**：
1. `__init__` 接收多边形和标签，做结构校验并转成 `numpy` 数组。
2. 遍历每个多边形顶点，计算 `bboxes[i]=[min_x,min_y,max_x,max_y]`。
3. 汇总得到 `global_bbox`，据此确定网格归一化坐标范围。
4. `_build_grid_index` 将每个多边形包围盒覆盖到网格区间，写入 `_grid[(gx,gy)] -> [idx...]`。
5. 查询时 `locate(p)` 先用 `_cell_of_point` 找到所在网格，取候选多边形索引列表。
6. 对每个候选先调用 `_point_on_segment`：
   - 叉积近零判断共线；
   - 点积判断是否在线段投影范围内；
   - 若命中立即返回 `BOUNDARY`。
7. 若非边界，调用 `_point_in_polygon` 执行射线奇偶法：对每条边判断是否跨过水平射线并翻转 `inside`。
8. 首个非 `OUTSIDE` 结果即返回；若候选都失败则返回 `OUTSIDE`，并可用 `locate_bruteforce` 做基线一致性验证。
