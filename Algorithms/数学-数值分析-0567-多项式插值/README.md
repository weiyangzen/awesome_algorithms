# 多项式插值

- UID: `MATH-0567`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `567`
- 目标目录: `Algorithms/数学-数值分析-0567-多项式插值`

## R01

本条目实现“多项式插值（Polynomial Interpolation）”的最小可运行版本，重点展示重心形式（Barycentric form）的拉格朗日插值。

目标：
- 给定离散节点 `(x_i, y_i)`，构造次数不超过 `n-1` 的插值多项式；
- 在一组查询点上稳定评估插值结果；
- 对比等距节点与 Chebyshev 节点在 Runge 函数上的误差差异；
- 给出与 `numpy.polyfit` 的误差对照，并可选与 `scipy` 实现做一致性检查。

## R02

问题定义：
- 输入：
  - 节点坐标数组 `x = [x_0, ..., x_{n-1}]`（互异）；
  - 节点函数值数组 `y = [y_0, ..., y_{n-1}]`；
  - 查询点数组 `xq`；
  - 节点生成策略（等距 / Chebyshev）。
- 输出：
  - 查询点处的插值值 `p(xq)`；
  - 误差统计（对已知真值样例输出最大绝对误差）；
  - 一致性指标（插值在节点上是否回到原值、与 SciPy 重心实现差异）。

## R03

核心数学公式：

1. 拉格朗日形式  
`p(x) = \sum_{i=0}^{n-1} y_i L_i(x)`，  
`L_i(x) = \prod_{j \ne i}\frac{x - x_j}{x_i - x_j}`。

2. 重心权重  
`w_i = 1 / \prod_{j \ne i}(x_i - x_j)`。

3. 重心评估公式（`x` 不等于任何节点时）  
`p(x) = \frac{\sum_i \frac{w_i y_i}{x - x_i}}{\sum_i \frac{w_i}{x - x_i}}`。

4. 节点命中规则  
若 `x = x_k`，则直接令 `p(x)=y_k`，避免数值除零并保持插值精确性。

## R04

算法流程：
1. 生成节点（等距或 Chebyshev）并计算函数值。
2. 校验输入（维度、长度、节点互异性）。
3. 根据节点计算重心权重 `w_i`。
4. 对每个查询点：
   - 若与某节点重合，直接返回该节点值；
   - 否则按重心公式计算插值值。
5. 在稠密网格上与真值比较，得到最大绝对误差。
6. 用 `numpy.polyfit` 构造同阶多项式做对照误差。
7. 若环境有 `scipy.interpolate.BarycentricInterpolator`，比较两者输出差异。
8. 打印分场景报告（函数、节点策略、误差、稳定性指标）。

## R05

核心数据结构：
- `ExperimentResult`（`dataclass`）：
  - `case_name`：测试函数名称；
  - `node_kind`：`equidistant` 或 `chebyshev`；
  - `n_nodes`：节点数；
  - `node_consistency`：`max|p(x_i)-y_i|`；
  - `max_abs_error_barycentric`：重心插值最大绝对误差；
  - `max_abs_error_polyfit`：`numpy.polyfit` 结果最大绝对误差；
  - `max_abs_diff_vs_scipy`：与 SciPy 重心实现最大差异（可空）。
- `demo.py` 中固定样例列表：包含函数、区间、节点数与评估网格数。

## R06

正确性要点：
- 多项式插值在节点处应满足 `p(x_i)=y_i`，代码通过“节点命中直返”保证该性质；
- 重心公式与拉格朗日形式代数等价，但数值评估更稳定；
- 节点互异性校验确保权重定义合法；
- 使用节点回代误差 `node_consistency` 作为实现正确性的直接证据；
- 用解析函数真值在稠密网格上比较，验证“实现正确 + 数值行为合理”。

## R07

复杂度分析（`n` 为节点数，`m` 为查询点数）：
- 权重计算：双重循环，时间 `O(n^2)`，空间 `O(n)`；
- 单次批量评估：主要为 `m x n` 级矩阵运算，时间 `O(mn)`，空间 `O(mn)`；
- `polyfit` 对照：内部近似 `O(n^3)`（依赖线性代数求解）；
- 在教学与小规模原型中，`n` 一般较小，整体开销可接受。

## R08

边界与异常处理：
- `x`、`y` 非一维或长度不一致时抛出 `ValueError`；
- `n < 2`（无法体现一般插值）抛出 `ValueError`；
- 节点重复（非互异）抛出 `ValueError`；
- 查询点允许为标量或数组，统一转为 `numpy.ndarray` 处理；
- SciPy 不可用时仅跳过一致性对照，不影响主流程运行。

## R09

MVP 取舍：
- 主算法自行实现（不把三方库当黑盒）；
- 使用 `numpy` 做数值数组运算与 `polyfit` 对照；
- `scipy` 仅用于“可选一致性验证”，不是算法依赖；
- 不引入图形绘制与命令行参数解析，保持脚本最小、可直接运行。

## R10

`demo.py` 函数职责：
- `build_nodes`：生成等距或 Chebyshev 节点；
- `validate_inputs`：输入合法性检查；
- `compute_barycentric_weights`：按定义计算重心权重；
- `barycentric_interpolate`：执行重心插值评估（含节点命中处理）；
- `run_experiment`：单场景运行与误差统计；
- `format_float_or_na`：统一打印格式；
- `main`：组织固定实验并输出摘要。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0567-多项式插值
python3 demo.py
```

脚本无交互输入，会直接执行预设样例并打印结果表。

## R12

输出字段说明：
- `Case`：测试函数与区间场景；
- `Nodes`：节点类型；
- `N`：节点数量；
- `NodeConsistency`：节点回代误差（越接近 0 越好）；
- `MaxErr(Bary)`：重心插值在评估网格的最大绝对误差；
- `MaxErr(polyfit)`：`numpy.polyfit` 同阶拟合误差；
- `Max|Bary-SciPy|`：自实现与 SciPy 重心实现最大差异（不可用时显示 `N/A`）。

## R13

最小测试集（`demo.py` 已内置）：
- Case 1：Runge 函数 `f(x)=1/(1+25x^2)`，区间 `[-1,1]`，`n=15`；
- Case 2：平滑振荡函数 `f(x)=exp(x)cos(3x)`，区间 `[-1,1]`，`n=15`；
- 每个 case 分别测试：
  - 等距节点（更容易出现 Runge 现象）；
  - Chebyshev 节点（通常误差更稳健）。

## R14

可调参数：
- `n_nodes`：插值节点个数（默认 15）；
- `n_eval`：误差评估网格点数（默认 2000）；
- `node_kind`：`equidistant` / `chebyshev`。

调参建议：
- 若想更明显观察 Runge 现象，可增大等距节点数；
- 若想提升整体稳定性，可优先采用 Chebyshev 节点分布。

## R15

方法对比：
- 与牛顿插值相比：
  - 牛顿形式便于增量加点；
  - 重心拉格朗日在“给定固定节点批量评估”场景实现更直接。
- 与样条插值相比：
  - 多项式插值是全局高阶模型，节点多时可能振荡；
  - 样条是分段低阶模型，工程上通常更稳。
- 与 `numpy.polyfit` 相比：
  - 二者都可得到插值多项式；
  - 重心评估通常更强调数值稳定性与节点命中精确性。

## R16

应用场景：
- 数值分析课程中的插值与误差行为演示；
- 已知少量采样点时的函数近似与重采样；
- 谱方法、求积规则等算法中的基础构件；
- 需要解释性较强的数值原型验证任务。

## R17

后续扩展方向：
- 加入分段插值策略，与全局多项式误差进行自动对比；
- 增加导数插值（Hermite）与误差上界估计；
- 引入自适应节点选择或最优实验设计策略；
- 增加可视化（误差曲线与插值曲线）用于教学演示。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `main` 定义两个解析函数、区间与节点配置，并遍历 `equidistant/chebyshev` 两种节点策略。  
2. `run_experiment` 调用 `build_nodes` 生成插值节点 `x_i`，并计算 `y_i=f(x_i)`。  
3. `validate_inputs` 检查维度、长度、节点数与互异性，保证权重公式可计算。  
4. `compute_barycentric_weights` 通过双重循环计算 `w_i = 1 / Π_{j!=i}(x_i-x_j)`。  
5. `barycentric_interpolate` 对查询点构造 `diff = xq - x_i`；若命中节点则直接回填 `y_i`。  
6. 对非命中点按重心公式计算分子 `Σ(w_i y_i/(x-x_i))` 与分母 `Σ(w_i/(x-x_i))`，得到 `p(x)`。  
7. `run_experiment` 在稠密网格上计算真值误差，并用 `numpy.polyfit` 构造同阶多项式做对照误差。  
8. 若安装了 SciPy，则调用 `BarycentricInterpolator` 做一致性比对并输出 `max|self-scipy|`，验证实现与库算法流程等价。  
