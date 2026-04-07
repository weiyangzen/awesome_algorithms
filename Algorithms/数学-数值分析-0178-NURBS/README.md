# NURBS

- UID: `MATH-0178`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `178`
- 目标目录: `Algorithms/数学-数值分析-0178-NURBS`

## R01

NURBS（Non-Uniform Rational B-Splines，非均匀有理 B 样条）是 CAD/CAM 与几何建模中的核心曲线表示方法。

本条目目标是交付一个最小可运行 MVP：
- 从零实现 NURBS 曲线求值（不依赖黑箱几何库）；
- 给出可验证的数值性质（分片基函数归一、端点插值）；
- 给出经典“二次 NURBS 精确表示四分之一圆”验证。

## R02

问题定义（曲线版）：

给定：
- 控制点 `P_i in R^d, i=0..n`；
- 权重 `w_i > 0`；
- 次数 `p`；
- 非递减节点向量 `U = {u_0, ..., u_m}`，其中 `m = n + p + 1`。

求：
- 参数 `u` 下曲线点 `C(u)`；
- 一组采样参数上的离散曲线点序列。

## R03

NURBS 曲线定义：

`C(u) = sum_{i=0..n} R_{i,p}(u) * P_i`

其中有理基函数为：

`R_{i,p}(u) = N_{i,p}(u) * w_i / sum_{j=0..n}(N_{j,p}(u) * w_j)`

`N_{i,p}(u)` 是 B 样条基函数（Cox-de Boor 递推定义）：

- `N_{i,0}(u) = 1` 当 `u_i <= u < u_{i+1}`，否则 `0`；
- `N_{i,p}(u) = ((u-u_i)/(u_{i+p}-u_i))*N_{i,p-1}(u) + ((u_{i+p+1}-u)/(u_{i+p+1}-u_{i+1}))*N_{i+1,p-1}(u)`。

## R04

数值算法选型（MVP）：
- 使用 The NURBS Book 常见的两步局部算法：
  1. `find_span`：二分查找参数所在节点区间；
  2. `basis_funs`：只计算当前区间内 `p+1` 个非零基函数。
- 再结合权重得到有理基函数 `R_i`，最终线性组合控制点。

该方案优点：
- 时间和空间局部性好；
- 代码短、可审计；
- 适合教学和最小工程实现。

## R05

本目录 `demo.py` 的核心数据结构：

- `NURBSCurve`（`dataclass`）
  - `control_points`: `(n+1, dim)`
  - `weights`: `(n+1,)`
  - `degree`: `p`
  - `knots`: `(n+p+2,)`
- `numpy.ndarray`
  - 参数向量与采样点矩阵。

`NURBSCurve.__post_init__` 负责一致性校验：维度、正权重、节点非递减、参数域有效等。

## R06

正确性关键点：

- 有理基函数归一性：`sum_i R_i(u) = 1`（分母非零时）；
- 开区间夹持节点（clamped/open）下，曲线通过首尾控制点；
- 正权重保证分母严格正，从而避免奇异。

MVP 中用 `verify_partition_of_unity` 与 `verify_endpoints` 做数值断言。

## R07

复杂度（单个参数 `u` 的求值）：

- `find_span`：二分查找，`O(log n)`；
- `basis_funs`：局部递推，`O(p^2)`；
- 组装有理基并线性组合控制点：`O(p + dim*(p+1))`。

整体采样 `S` 个参数点时，总体近似 `O(S*(log n + p^2 + dim*p))`。

## R08

边界与异常处理（代码已覆盖）：

- 控制点不是 2D 数组、权重长度不匹配、次数非法：抛出 `ValueError`；
- 权重非正、节点非递减条件不满足：抛出 `ValueError`；
- 参数 `u` 超出 `[U_p, U_{n+1}]`：抛出 `ValueError`；
- 有理分母非正：抛出 `ZeroDivisionError`。

## R09

MVP 取舍说明：

- 只实现 NURBS 曲线，不扩展曲面；
- 不实现导数、曲率、反求参数等高级功能；
- 不依赖专用几何库（如 `geomdl`），确保算法细节可见；
- 使用 `numpy` 即可运行，依赖最小化。

## R10

`demo.py` 函数分工：

- `make_open_uniform_knot`：构造夹持开区间均匀节点；
- `find_span`：定位参数所在 span；
- `basis_funs`：计算局部非零 B 样条基函数；
- `rational_basis_row`：把 B 样条基函数加权归一为 NURBS 有理基；
- `curve_point`：单点求值 `C(u)`；
- `sample_curve`：批量采样；
- `build_general_demo_curve` / `build_quarter_circle_curve`：构造两个演示曲线；
- `verify_*`：执行数值性质校验；
- `run_demo`：打印关键结果。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0178-NURBS
python3 demo.py
```

脚本无交互输入，直接输出：
- 一般三次 NURBS 的采样摘要；
- 四分之一圆 NURBS 的半径误差验证；
- 若断言失败会直接抛异常。

## R12

输出解读：

- `max_partition_unity_error`：`max |sum_i R_i(u)-1|`，应接近机器精度；
- `max_endpoint_error`：端点插值误差，夹持节点下应接近 0；
- `max_radius_error`：四分之一圆样本半径相对 1 的最大误差，越小越好；
- `sample_general[...]` / `sample_circle[...]`：参数点和坐标预览。

## R13

最小测试建议：

- 参数域端点：`u=U_p`、`u=U_{n+1}`；
- 内点归一性：随机多个 `u` 验证 `sum R_i(u) ~ 1`；
- 非法输入：负权重、降序节点、`u` 越界；
- 经典构形：二次三控制点四分之一圆半径一致性。

## R14

可调参数：

- `degree`：曲线次数（示例含 2 次和 3 次）；
- 控制点布局与权重分布；
- `num_samples`：采样密度；
- 校验阈值 `tol`（如 `1e-10`、`5e-12`）。

## R15

与相近表示方法对比：

- 相比普通 B 样条：NURBS 通过权重引入有理形式，可精确表示圆锥曲线；
- 相比 Bézier：NURBS 支持非均匀节点与局部控制，更适合复杂 CAD 曲线；
- 相比直接多项式插值：NURBS 的局部支撑与形状可控性更强，数值稳定性更工程化。

## R16

典型应用：

- CAD/CAE 几何建模（轮廓、曲线边界）；
- CAM 刀具路径与轨迹光顺；
- 字体/矢量形状建模；
- 等几何分析（IGA）中的几何表示基础。

## R17

后续扩展方向：

- NURBS 曲面（张量积）与体；
- 一阶/二阶导数、法向、曲率计算；
- 节点插入（knot insertion）与细分；
- 曲线拟合/逼近（最小二乘求控制点与权重）。

## R18

源码级算法流（对应 `demo.py`，8 步）：

1. `build_general_demo_curve` 与 `build_quarter_circle_curve` 构造两条曲线，分别覆盖通用形状和解析可验证形状。  
2. 在 `NURBSCurve.__post_init__` 中统一完成维度、权重、节点向量、参数域的合法性检查。  
3. `sample_curve` 先在有效参数域 `[U_p, U_{n+1}]` 上生成等距参数。  
4. 对每个参数 `u`，`find_span` 用二分法定位节点区间 `span`。  
5. `basis_funs` 在该局部区间递推得到 `p+1` 个非零 B 样条基函数值 `N_{i,p}(u)`。  
6. `rational_basis_row` 将局部基函数乘权重并归一化，形成全局有理基向量 `R_i(u)`。  
7. `curve_point` 用 `R_i(u)` 对控制点做线性组合，得到曲线点 `C(u)`。  
8. `verify_partition_of_unity`、`verify_endpoints`、`verify_quarter_circle` 分别检查基函数归一性、端点插值和圆弧半径一致性，并输出误差指标。
