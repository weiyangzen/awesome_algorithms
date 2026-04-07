# 凸包算法 - Jarvis步进

- UID: `MATH-0184`
- 学科: `数学`
- 分类: `计算几何`
- 源序号: `184`
- 目标目录: `Algorithms/数学-计算几何-0184-凸包算法_-_Jarvis步进`

## R01

Jarvis 步进（Gift Wrapping）用于求二维点集的凸包边界：
从最左点出发，每一步选择“最逆时针”的下一个点，直到回到起点。

本目录给出一个可运行 MVP：
- `demo.py` 手写 Jarvis 主流程，不调用 `scipy.spatial.ConvexHull` 等黑盒；
- 覆盖重复点、全共线等退化情形；
- 输出 3 组固定样例的凸包结果与一致性检查。

## R02

问题定义（本实现）：
- 输入：二维点集 `P = {(x_i, y_i)}`，数组形状为 `n x 2`。
- 输出：凸包顶点序列 `H = [p_0, p_1, ..., p_{h-1}]`，满足：
  - `H` 是输入去重点集上的极点序列；
  - 一般情形下按逆时针顺序给出；
  - 退化情形下：
    - 0 点返回空；
    - 1 点返回该点；
    - 全共线返回两端点。

## R03

数学基础：
1. 方向谓词（orientation）：
   `cross((b-a), (c-a)) > 0` 表示 `a->b->c` 为逆时针。  
2. 凸包定义：包含全部点的最小凸多边形。  
3. Jarvis 核心策略：
   - 当前位置 `p` 下，选择点 `q`，使得其它任意点都不在有向边 `p->q` 的左侧（等价地，`q` 是“最外层”候选）。  
4. 共线处理：若多个候选与当前边共线，保留距离当前点最远者，避免丢失极点。

## R04

算法流程（高层）：
1. 检查输入是否为有限数值的 `n x 2` 数组。  
2. 用容差 `eps` 去重。  
3. 处理退化情况（空点集、单点、全共线）。  
4. 选取起点：最左点（同 `x` 时取较小 `y`）。  
5. 固定当前点 `current`，扫描所有点选出下一顶点 `candidate`：
   - 按叉积符号选择更外侧候选（实现中是更顺时针候选）；
   - 若共线则选更远点。  
6. 将 `candidate` 追加到凸包并迭代。  
7. 当回到起点时结束。  
8. 输出顶点索引、顶点坐标和去重统计。

## R05

核心数据结构：
- `HullResult`：
  - `unique_points: np.ndarray` 去重后的点集；
  - `hull_indices: list[int]` 凸包点在去重点集中的索引序；
  - `hull_points: np.ndarray` 凸包坐标序列；
  - `removed_duplicates: int` 去除的重复点数量。
- 几何函数：
  - `orientation(a, b, c)`：计算二维叉积符号；
  - `squared_distance(a, b)`：用于共线时“最远点”判定。

## R06

正确性要点：
- `orientation` 保证每轮选择的都是边界“最外”方向。  
- 共线候选取最远点，确保边界端点不被中间点替代。  
- 从最左点出发并循环至起点，得到闭合边界。  
- `verify_hull` 在结果后验检查：
  - `convex`：边转向不出现明显右拐；
  - `contains_all_points`：所有点都在凸包边界内或边上；
  - `ccw`：面积为正（非退化情形）。

## R07

复杂度分析：
- 设 `n` 为去重后点数，`h` 为凸包顶点数。  
- 每确定一个凸包顶点需线性扫描全部点一次，时间复杂度为 `O(nh)`。  
- 空间复杂度主要为点集和结果索引，约 `O(n)`。

说明：当 `h` 远小于 `n` 时，Jarvis 在工程上有可读性优势；当 `h` 接近 `n` 时会趋近 `O(n^2)`。

## R08

边界与异常处理：
- 输入非 `n x 2`：抛 `ValueError`。  
- 输入含 `nan/inf`：抛 `ValueError`。  
- 去重后点数为 0/1：直接返回。  
- 全共线：返回两个端点。  
- 存在重复点：自动合并并记录 `removed_duplicates`。

## R09

MVP 取舍：
- 只做二维凸包，不扩展到 3D。  
- 仅依赖 `numpy`，保持最小工具栈。  
- 不做图形绘制，默认输出文本摘要以保证可在纯终端运行。  
- 不调用黑盒库凸包 API，确保算法过程可逐行审查。

## R10

`demo.py` 主要函数职责：
- `check_points`：输入合法性检查。  
- `deduplicate_points`：按容差去重。  
- `orientation` / `are_all_collinear`：方向与共线判断。  
- `jarvis_march`：Jarvis 主算法。  
- `polygon_area` / `point_on_segment` / `verify_hull`：结果后验验证。  
- `summarize_case` / `main`：组织样例并打印输出。

## R11

运行方式：

```bash
cd Algorithms/数学-计算几何-0184-凸包算法_-_Jarvis步进
python3 demo.py
```

脚本不需要命令行参数，不读取交互输入。

## R12

输出字段说明：
- `input points`：原始点数。  
- `unique points`：去重后点数及被去除重复数量。  
- `hull vertex count`：凸包顶点数。  
- `hull indices`：凸包点在去重点集中的索引。  
- `hull points`：凸包顶点坐标。  
- `checks`：后验校验结果（凸性、包含性、逆时针方向）。

## R13

内置测试样例：
- `Uniform random`：20 个随机点，验证一般情形。  
- `Skewed rectangle cloud`：近矩形边界 + 内部点，验证极点选择。  
- `Degenerate collinear + duplicates`：共线且重复，验证退化处理。

可扩展测试建议：
- 大量近重复点（测试去重容差稳定性）；
- 坐标尺度极不均衡的点集（测试数值稳定性）；
- 已经是凸多边形顶点顺序输入（检查不误删顶点）。

## R14

可调参数：
- `eps`（默认 `1e-12`）：
  - 控制去重量化尺度；
  - 控制 orientation 共线阈值。

调参建议：
- 若重复点没有合并：增大 `eps`。  
- 若不同点被误并：减小 `eps`。  
- 若数据尺度特别大或特别小：先做归一化再运行。

## R15

方法对比：
- 对比 Graham Scan：
  - Graham 常见复杂度 `O(n log n)`；
  - Jarvis 为 `O(nh)`，在 `h` 很小时有优势。  
- 对比 Andrew Monotone Chain：
  - Andrew 实现更短、复杂度稳定；
  - Jarvis 更直观展示“逐边包裹”的几何思想。  
- 对比库函数：
  - 库函数更快、更健壮；
  - 本实现更适合教学和源码级审计。

## R16

典型应用场景：
- 点云外边界提取。  
- GIS/地图中的区域轮廓粗提。  
- 碰撞检测前的包围多边形构造。  
- 计算几何教学中 orientation 谓词与凸包概念演示。

## R17

可扩展方向：
- 增加可视化（例如 matplotlib）输出包裹过程。  
- 使用更鲁棒的几何谓词（自适应精度/精确算术）。  
- 加入批量基准测试，与 Graham/Andrew 算法做性能对照。  
- 提供流式增量更新（动态点集）。

## R18

`demo.py` 的源码级流程拆解（8 步）：
1. `main` 构造三组固定二维点集（随机、近矩形、共线重复）。  
2. 每组数据进入 `summarize_case`，调用 `jarvis_march`。  
3. `jarvis_march` 先执行 `check_points` 与 `deduplicate_points`。  
4. 若是空集、单点或全共线，直接按退化规则返回。  
5. 否则选最左点作为起点 `start`，初始化 `current = start`。  
6. 在每一轮中遍历全部点，利用 `orientation(current, candidate, i)` 按“更顺时针候选”规则更新（配合最左起点实现整体 CCW 包裹）；若共线则用 `squared_distance` 保留更远点。  
7. 将选中的 `candidate` 作为下一凸包顶点，更新 `current`，直到回到 `start` 停止。  
8. `summarize_case` 调 `verify_hull`（凸性/包含性/方向）并打印结果摘要。
