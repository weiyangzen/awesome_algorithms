# de Casteljau算法

- UID: `MATH-0180`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `180`
- 目标目录: `Algorithms/数学-数值分析-0180-de_Casteljau算法`

## R01

本条目实现 de Casteljau 算法的最小可运行版本，用于稳定计算 Bezier 曲线点，并演示曲线采样与控制多边形细分。

核心目标：
- 给定控制点和参数 `t in [0,1]`，计算 Bezier 曲线点；
- 显式输出 de Casteljau 递推金字塔（pyramid）中间层，保证可解释性；
- 用 Bernstein 形式交叉验证数值一致性，确认实现正确。

## R02

问题定义（有限精度数值计算）：
- 输入：
  - 控制点 `P_0, P_1, ..., P_n`，每个点是 `d` 维实向量；
  - 参数 `t in [0,1]`；
  - 采样数量 `m`（用于整条曲线离散采样）。
- 输出：
  - 曲线点 `B(t)`；
  - de Casteljau 每一层中间点（可选，便于调试/教学）；
  - 多个 `t` 上的采样点序列。

## R03

数学基础：

1) Bezier 曲线定义（Bernstein 形式）
- `B(t) = sum_{i=0}^n C(n,i) (1-t)^(n-i) t^i P_i`。

2) de Casteljau 递推定义
- 初始层：`b_i^(0) = P_i`；
- 递推层：`b_i^(r) = (1-t) b_i^(r-1) + t b_{i+1}^(r-1)`，其中 `r=1..n`；
- 终值：`B(t) = b_0^(n)`。

3) 数值性质
- 递推只涉及凸组合，稳定性通常优于直接高次 Bernstein 展开；
- 曲线点始终落在控制点凸包内（凸包性质）。

## R04

算法总览（单点求值）：
1. 检查控制点数组形状，保证至少 2 个点且维度一致。
2. 检查 `t` 在 `[0,1]` 且有限。
3. 将控制点作为第 0 层。
4. 对层数 `r = 1..n`，按相邻点线性插值生成下一层。
5. 重复直到只剩一个点。
6. 该唯一点即 `B(t)`；若开启调试则返回全部层。

## R05

核心数据结构：
- `numpy.ndarray`：
  - `control_points: shape=(n+1, d)`；
  - `levels[k]: shape=(n+1-k, d)`，表示第 `k` 层；
  - `curve_points: shape=(m, d)`，整条曲线采样结果。
- `DeCasteljauResult`（`dataclass`）：
  - 记录 `t`、最终曲线点、所有层中间结果，便于打印与后续扩展。

## R06

正确性要点：
- 每次递推都与定义 `b_i^(r) = (1-t)b_i^(r-1) + t b_{i+1}^(r-1)` 一致；
- 递推层数从 `n+1` 点缩减为 `1` 点，满足构造终止条件；
- de Casteljau 与 Bernstein 形式等价，可用同一组控制点与 `t` 交叉验证；
- 全过程是凸组合，因此结果满足凸包约束。

## R07

复杂度分析：
- 单次 `B(t)` 计算：
  - 时间 `O(n^2 * d)`（共 `n(n+1)/2` 次向量线性插值）；
  - 空间 `O(n^2 * d)`（若保留全部层），或 `O(n * d)`（仅保留当前层）。
- 采样 `m` 个参数点：时间 `O(m * n^2 * d)`。

## R08

边界与异常处理：
- 控制点不是二维数组或点数小于 2：抛出 `ValueError`；
- 控制点含 `nan/inf`：抛出 `ValueError`；
- `t` 非有限或不在 `[0,1]`：抛出 `ValueError`；
- `num_samples < 2`：抛出 `ValueError`。

## R09

MVP 取舍：
- 只使用 `numpy + 标准库`，避免重依赖；
- 不调用现成 Bezier 黑盒接口，显式实现递推过程；
- 以可读性和算法可追踪性优先，不做并行化/矢量化极限优化；
- 额外提供 Bernstein 验证函数用于结果对照。

## R10

`demo.py` 函数职责：
- `validate_control_points`：检查控制点矩阵合法性；
- `validate_t`：检查参数范围；
- `de_casteljau_point`：计算单个 `t` 的曲线点并返回中间层；
- `bernstein_point`：用组合数公式独立计算同一点（交叉验证）；
- `sample_bezier_curve`：在 `[0,1]` 上均匀采样曲线；
- `subdivide_control_polygon`：在给定 `t` 下输出左右子曲线控制点；
- `run_demo`：组织示例并打印误差、采样、细分信息；
- `main`：无交互入口。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0180-de_Casteljau算法
python3 demo.py
```

脚本无需输入，会自动执行固定样例并打印结果。

## R12

输出解读：
- `degree`：Bezier 阶数（控制点数减 1）；
- `t`：参数值；
- `de_casteljau_point` 与 `bernstein_point`：两种计算路径的结果；
- `l2_error`：两者欧氏距离误差，应接近机器精度；
- `sampled_points_head`：前几个采样点；
- `left_subcurve_control_points` / `right_subcurve_control_points`：细分后的控制多边形。

## R13

建议最小测试集：
- 三次曲线（4 个控制点）在 `t=0,0.25,0.5,0.75,1` 的一致性测试；
- 端点测试：`B(0)=P_0`，`B(1)=P_n`；
- 非法输入测试：空控制点、`t=-0.1`、`t=1.1`、`nan` 控制点；
- 采样稳定性测试：`num_samples=2` 与较大样本数对比。

## R14

可调参数：
- `control_points`：控制多边形形状；
- `t_values`：单点评估参数序列；
- `num_samples`：整条曲线采样密度；
- `subdivide_t`：细分参数（常用 `0.5`）。

建议：先用低阶曲线验证公式，再增大阶数观察数值行为。

## R15

方法对比：
- 与直接 Bernstein 求和相比：
  - de Casteljau 在高阶或极端 `t` 时通常更数值稳定；
  - Bernstein 形式更适合表达闭式公式与理论推导。
- 与样条曲线（如 B-spline）相比：
  - Bezier/de Casteljau 简洁直接；
  - 但高阶全局控制较弱，复杂形状常用分段曲线。

## R16

应用场景：
- CAD/CAM 曲线与曲面建模；
- 字体轮廓（TrueType/CFF）与矢量图形；
- 动画路径插值与运动轨迹平滑；
- 数值几何课程中的稳定插值算法教学。

## R17

后续扩展方向：
- 扩展到有理 Bezier（加入权重）和齐次坐标版本；
- 用向量化或 JIT 优化大量采样性能；
- 支持曲率/切向量计算与弧长近似；
- 增加单元测试并与几何库输出做系统比对。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `run_demo` 定义控制点、`t_values`、采样数与细分参数。  
2. 对每个 `t` 先调用 `de_casteljau_point`：将控制点作为第 0 层。  
3. 在 `de_casteljau_point` 内循环层号 `r=1..n`，执行 `next=(1-t)*prev[:-1]+t*prev[1:]` 构造下一层。  
4. 当层长度降到 1 时取 `levels[-1][0]` 作为 `de_casteljau_point`。  
5. 同一 `t` 调用 `bernstein_point`，用 `comb(n,i)*(1-t)^(n-i)*t^i` 权重求和得到独立参考值。  
6. 计算两结果的 `L2` 误差并打印，验证递推实现与闭式公式一致。  
7. 调用 `sample_bezier_curve` 在 `[0,1]` 生成均匀 `t` 网格，逐点复用 de Casteljau 得到曲线采样数组。  
8. 调用 `subdivide_control_polygon` 从递推金字塔提取左右子控制点，展示同一曲线在 `t` 处的几何细分结果。  
