# Gauss-Legendre求积

- UID: `MATH-0132`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `132`
- 目标目录: `Algorithms/数学-数值分析-0132-Gauss-Legendre求积`

## R01

本条目实现 Gauss-Legendre 求积（Gaussian Quadrature with Legendre polynomials）的最小可运行版本。

核心目标：
- 在给定区间 `[a, b]` 上近似计算定积分 `I = \int_a^b f(x) dx`；
- 明确展示节点（nodes）和权重（weights）如何从正交多项式递推关系构造出来；
- 通过多组样例验证“对低阶多项式的高精度/精确性”和对光滑函数的快速收敛。

## R02

问题定义（有限精度计算版）：
- 输入：
  - 被积函数 `f: R -> R`；
  - 积分区间 `a < b`；
  - 求积阶数 `n >= 1`（节点数）；
- 输出：
  - 积分近似值 `Q_n`；
  - 标准区间 `[-1, 1]` 的节点 `t_i` 和权重 `w_i`；
  - 映射到 `[a, b]` 后的节点 `x_i` 和权重 `w_i'`；
  - 误差指标（若有解析真值，则输出绝对误差）。

## R03

数学基础与公式：

1) 标准区间公式
- 对 `[-1, 1]`，Gauss-Legendre 求积写成
  `\int_{-1}^{1} g(t) dt \approx \sum_{i=1}^{n} w_i g(t_i)`。
- 节点 `t_i` 是 `n` 次 Legendre 多项式 `P_n(t)` 的根。
- 该公式对所有次数 `<= 2n-1` 的多项式是精确的。

2) 区间变换
- 将 `x in [a,b]` 映射为 `t in [-1,1]`：
  `x = ((b-a)/2) t + (a+b)/2`。
- 得到
  `\int_a^b f(x) dx = (b-a)/2 \int_{-1}^{1} f(((b-a)/2)t + (a+b)/2) dt`。

3) Golub-Welsch 构造
- Legendre 多项式三项递推对应一个对称三对角 Jacobi 矩阵 `J`；
- `J` 的特征值就是节点 `t_i`；
- 若 `v_i` 是单位特征向量，则权重 `w_i = 2 * (v_i[0])^2`。

## R04

算法总览（MVP）：
1. 检查输入：`n >= 1`、`a < b`、`a,b` 有限。
2. 根据 Legendre 递推系数构造 `n x n` 的对称三对角 Jacobi 矩阵。
3. 对 Jacobi 矩阵做特征分解（`numpy.linalg.eigh`）。
4. 读取特征值作为 `[-1,1]` 上节点；读取特征向量首行构造权重。
5. 把节点和权重线性映射到 `[a,b]`。
6. 在映射后的节点上采样 `f(x_i)`，计算加权和 `sum(w_i' * f(x_i))`。
7. 输出近似结果，并在可得真值时报告误差。

## R05

核心数据结构：
- `numpy.ndarray`：
  - `nodes_std: shape=(n,)`，标准区间节点；
  - `weights_std: shape=(n,)`，标准区间权重；
  - `nodes_mapped: shape=(n,)`，目标区间节点；
  - `weights_mapped: shape=(n,)`，目标区间权重；
  - `fx: shape=(n,)`，函数值采样向量；
- `IntegrationResult`（`dataclass`）：
  - 汇总 `n`、节点、权重、函数采样、积分近似值，便于打印和后续扩展。

## R06

正确性要点：
- Legendre 多项式族在 `[-1,1]` 上关于权函数 `1` 正交；
- 基于该正交族构造的 Gauss 型公式，对 `2n-1` 次以内多项式精确；
- Golub-Welsch 定理保证：Jacobi 矩阵特征值/特征向量与 Gauss 节点/权重一一对应；
- 线性区间变换只引入常数雅可比因子 `(b-a)/2`，不会破坏求积结构。

## R07

复杂度分析：
- 构造 Jacobi 矩阵：`O(n)` 非零元素（若按稠密矩阵存储则空间 `O(n^2)`）；
- 对称特征分解（稠密）：时间 `O(n^3)`，空间 `O(n^2)`；
- 函数采样与加权和：时间 `O(n)`；
- 总体瓶颈在特征分解。

说明：本 MVP 重点在算法透明性，采用稠密 `eigh`；若 `n` 很大，可改用专门三对角本征算法降低常数和内存。

## R08

边界与异常处理：
- `n < 1`：抛出 `ValueError`；
- `a >= b`：抛出 `ValueError`；
- `a` 或 `b` 非有限：抛出 `ValueError`；
- 采样函数返回非有限值（`nan/inf`）：抛出 `ValueError`。

## R09

MVP 取舍：
- 选用 `numpy` + 标准库，不依赖更重框架；
- 不直接调用 `numpy.polynomial.legendre.leggauss` 黑盒接口，而是显式实现 Golub-Welsch 路径；
- 保留中间量（节点/权重/采样）用于可解释输出；
- 优先“可读+可验证”，而非极限性能优化。

## R10

`demo.py` 函数职责：
- `check_interval`：检查区间合法性；
- `gauss_legendre_nodes_weights`：由 Jacobi 矩阵求标准区间节点与权重；
- `map_nodes_weights`：把标准区间节点/权重映射到 `[a,b]`；
- `gauss_legendre_integrate`：执行一次完整求积并返回 `IntegrationResult`；
- `print_result_summary`：按样例打印近似值、真值和误差；
- `run_examples`：组织多组函数与阶数进行演示；
- `main`：无交互入口。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0132-Gauss-Legendre求积
python3 demo.py
```

脚本无需输入，会自动打印多组函数在不同阶数下的积分近似与误差。

## R12

输出解读：
- `n`：Gauss-Legendre 节点数；
- `estimate`：求积近似值；
- `reference`：解析真值（示例中提供）；
- `abs_error`：`|estimate-reference|`；
- `sample_nodes`：映射后的前几个节点；
- `sample_weights`：映射后的前几个权重。

观察点：
- 对多项式样例，`n` 充分时误差会接近机器精度；
- 对光滑非多项式，`n` 增大时误差通常快速下降。

## R13

建议最小测试集：
- 精确性测试：`f(x)=x^5-2x^3+x+1`，区间 `[-1,1]`，真值 `2`；
- 平滑函数：`f(x)=exp(x)`，区间 `[0,1]`，真值 `e-1`；
- 振荡函数：`f(x)=cos(5x)`，区间 `[0,1]`，真值 `sin(5)/5`；
- 逆向测试：`n=0`、`a>=b`、`nan/inf` 区间、函数返回 `nan`。

## R14

可调参数：
- `n_values`：测试使用的阶数列表（如 `[2,3,4,8,16]`）；
- 区间端点 `a,b`：可替换为其他任务区间；
- `preview_k`：输出时展示前 `k` 个节点和权重，控制日志长度。

实践建议：先用小 `n` 验证管线，再逐步增大 `n` 观察误差收敛。

## R15

方法对比：
- 与复合梯形/辛普森相比：
  - Gauss-Legendre 在同等采样点数量下通常精度更高；
  - 但其节点并非等距，不直接复用等距采样数据。
- 与 `scipy.integrate.quad` 相比：
  - `quad` 更通用且自适应；
  - 本实现更适合教学、可解释分析与固定节点批处理。

## R16

应用场景：
- 有限元/谱方法中的单元积分；
- 物理与工程中的高精度定积分近似；
- 概率统计中期望积分的数值近似；
- 数值分析课程中正交多项式与高斯求积教学。

## R17

后续扩展方向：
- 用三对角专用本征分解替代稠密 `eigh` 提升大 `n` 性能；
- 扩展到 Gauss-Chebyshev / Gauss-Laguerre / Gauss-Hermite；
- 增加自适应分区策略，处理局部剧烈变化函数；
- 增加与 SciPy 的交叉验证与单元测试。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `run_examples` 组装样例函数、积分区间、解析真值与阶数列表。  
2. 每次求积先调用 `check_interval(a,b)`，保证区间有效且端点有限。  
3. `gauss_legendre_nodes_weights(n)` 按 `beta_k = k/sqrt(4k^2-1)` 构造 Legendre 的 Jacobi 三对角矩阵 `J`。  
4. 调用 `numpy.linalg.eigh(J)`，取特征值作为标准区间节点 `t_i`。  
5. 从特征向量矩阵首行提取 `v_0i`，按 `w_i = 2*(v_0i)^2` 计算标准区间权重。  
6. `map_nodes_weights` 用仿射变换把 `(t_i,w_i)` 映射到目标区间 `(x_i,w_i')`。  
7. 在每个 `x_i` 上采样函数，做加权求和 `Q_n = sum(w_i' * f(x_i))` 得到积分近似。  
8. 汇总 `IntegrationResult` 并打印 `estimate/reference/abs_error` 及节点权重样本，完成可验证输出。  
