# Gauss-Laguerre求积

- UID: `MATH-0133`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `133`
- 目标目录: `Algorithms/数学-数值分析-0133-Gauss-Laguerre求积`

## R01

本条目实现 Gauss-Laguerre 求积（Gaussian Quadrature with Laguerre polynomials）的最小可运行版本。

核心目标：
- 近似计算半无穷区间上的带权积分 `I = \int_0^{+\infty} e^{-x} f(x) dx`；
- 显式展示节点（nodes）和权重（weights）如何由 Laguerre 正交多项式的 Jacobi 矩阵构造；
- 用有解析真值的样例验证“多项式精确性 + 非多项式收敛性”。

## R02

问题定义（MVP 范围）：
- 输入：
  - 函数 `f: R -> R`；
  - 节点数 `n >= 1`。
- 输出：
  - Gauss-Laguerre 近似值 `Q_n`；
  - 节点 `x_i` 与权重 `w_i`；
  - 若已知真值，输出绝对误差 `|Q_n - I|`。

注意：该算法直接对应带权积分 `\int_0^{+\infty} e^{-x} f(x) dx`，不是无权积分 `\int_0^{+\infty} g(x)dx` 的直接公式。

## R03

数学基础：

1) 基本公式（`alpha=0`）
- `\int_0^{+\infty} e^{-x} f(x) dx \approx \sum_{i=1}^{n} w_i f(x_i)`。
- 节点 `x_i` 是 `n` 次 Laguerre 多项式 `L_n(x)` 的根。

2) 代数精确性
- 对所有次数 `<= 2n-1` 的多项式 `p(x)`，有
  `\int_0^{+\infty} e^{-x} p(x) dx = \sum_{i=1}^{n} w_i p(x_i)`（忽略浮点舍入）。

3) 经典矩积分
- `\int_0^{+\infty} e^{-x} x^k dx = k!`。
- 这提供了验证 Gauss-Laguerre 实现正确性的直接基准。

4) Golub-Welsch 构造（Laguerre, `alpha=0`）
- 构造对称三对角 Jacobi 矩阵 `J`：
  - 对角元 `J_{k,k} = 2k+1`（`k=0,1,...,n-1`）；
  - 次对角元 `J_{k,k+1}=J_{k+1,k}=k+1`。
- `J` 的特征值是节点 `x_i`；若 `v_i` 是对应单位特征向量，则
  `w_i = mu_0 * (v_i[0])^2`，其中 `mu_0 = \int_0^{+\infty} e^{-x} dx = 1`，所以 `w_i = (v_i[0])^2`。

## R04

算法流程（MVP）：
1. 校验 `n >= 1`。
2. 构造 Laguerre(`alpha=0`) 的 Jacobi 三对角矩阵 `J`。
3. 使用 `numpy.linalg.eigh(J)` 求特征值与特征向量。
4. 读取特征值作为节点 `x_i`（自动升序）。
5. 用特征向量首行构造权重 `w_i = (v_i[0])^2`。
6. 在节点上采样 `f(x_i)`，检查函数值有限性。
7. 计算加权和 `Q_n = sum(w_i * f(x_i))`。
8. 打印估计值、真值、误差与节点/权重样本。

## R05

核心数据结构：
- `numpy.ndarray`：
  - `nodes: shape=(n,)`，Gauss-Laguerre 节点；
  - `weights: shape=(n,)`，对应权重；
  - `fx: shape=(n,)`，节点处函数值。
- `IntegrationResult`（`dataclass`）：
  - 字段：`n`、`estimate`、`nodes`、`weights`、`fx`。

这保证了输出既可直接读，也可用于后续扩展测试。

## R06

正确性要点：
- Laguerre 多项式在 `[0,+\infty)` 上关于权函数 `e^{-x}` 正交；
- Gauss 正交求积理论保证 `n` 节点对 `2n-1` 次多项式精确；
- Golub-Welsch 定理给出“节点/权重 <- Jacobi 矩阵特征分解”的严格对应；
- 代码中权重由特征向量首分量平方生成，满足 `sum(w_i)=1`（对应 `f(x)=1` 的积分真值）。

## R07

复杂度分析：
- 构造 Jacobi 矩阵：时间 `O(n)`（非零元），按稠密存储空间 `O(n^2)`；
- 对称特征分解（稠密 `eigh`）：时间 `O(n^3)`，空间 `O(n^2)`；
- 函数采样 + 加权求和：时间 `O(n)`；
- 总体瓶颈：特征分解。

MVP 优先可读性与可解释性，未做三对角专用本征算法优化。

## R08

边界与异常处理：
- `n < 1`：抛出 `ValueError`；
- 函数值出现 `nan/inf`：抛出 `ValueError`；
- 如果用户函数本身抛错，异常向上传播，方便定位业务函数问题。

## R09

MVP 取舍：
- 只实现 `alpha=0` 的标准 Gauss-Laguerre；
- 不直接调用 `scipy.special.roots_laguerre` 黑盒，而是显式写出 Golub-Welsch 路径；
- 不实现广义 Laguerre (`alpha>-1`) 与自适应策略；
- 目标是“可运行 + 可解释 + 可验证”，而不是覆盖全部变体。

## R10

`demo.py` 职责划分：
- `gauss_laguerre_nodes_weights`：构造 Jacobi 矩阵并求节点/权重；
- `gauss_laguerre_integrate`：执行一次积分并返回结构化结果；
- `print_result_summary`：打印估计、真值、误差、节点/权重样本；
- `run_examples`：组织多组示例（多项式、振荡函数、指数函数）；
- `main`：无交互入口。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0133-Gauss-Laguerre求积
python3 demo.py
```

脚本无交互输入，直接输出多组结果。

## R12

输出字段解读：
- `estimate`：Gauss-Laguerre 近似值；
- `reference`：解析真值；
- `abs_error`：绝对误差；
- `sample_nodes`：前若干节点（均为正，随 `n` 扩展到更大 `x`）；
- `sample_weights`：前若干权重（总和约为 1）。

观察点：
- `f(x)=x^5` 在 `n>=3` 时应接近机器精度；
- 对 `sin(x)`、`exp(-x)`，随 `n` 增大误差通常下降。

## R13

建议最小测试集：
- 精确性测试：
  - `f(x)=x^5`，真值 `5!=120`，`n=3` 应高精度；
  - `f(x)=x^7`，真值 `7!=5040`，`n=4` 应高精度。
- 光滑非多项式：
  - `f(x)=exp(-x)`，真值 `1/2`；
  - `f(x)=sin(x)`，真值 `1/2`。
- 异常测试：
  - `n=0`；
  - 函数返回 `nan`。

## R14

可调参数：
- `n_values`：每个示例的节点数列表（如 `[2,4,8,16]`）；
- `preview_k`：打印时展示的节点/权重前缀长度；
- 示例函数集合：可替换为目标业务函数以观察收敛行为。

建议：先小 `n` 检查流程，再增加 `n` 观察误差趋势。

## R15

方法对比：
- 相比 Gauss-Legendre：
  - Gauss-Legendre 适合有限区间无权积分；
  - Gauss-Laguerre天然适合 `e^{-x}` 权的半无穷积分。
- 相比截断区间 + 复合求积：
  - 截断法需要额外处理尾部误差；
  - Gauss-Laguerre把尾部衰减结构直接编码到权函数中，通常更稳定高效。
- 相比自适应黑盒积分器：
  - 黑盒通用性更高；
  - 本实现更适合教学与固定节点批处理。

## R16

应用场景：
- Gamma 函数与矩积分相关数值计算；
- 拉普拉斯变换型积分近似；
- 衰减核（指数权）下的期望/响应估计；
- 量子与统计物理中半无穷积分离散化。

## R17

后续扩展方向：
- 扩展到广义 Gauss-Laguerre（`alpha > -1`，权 `x^alpha e^{-x}`）；
- 增加“无权半无穷积分 `\int_0^\infty g(x)dx`”的安全变换辅助函数；
- 引入三对角专用本征算法提升大规模 `n` 性能；
- 增加单元测试与对照（如与 SciPy `roots_laguerre` 对比）。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `run_examples` 定义示例函数、解析真值与每个示例的 `n_values`。  
2. 每次调用 `gauss_laguerre_integrate` 时先进入 `gauss_laguerre_nodes_weights(n)`，检查 `n>=1`。  
3. `gauss_laguerre_nodes_weights` 按 `diag_k=2k+1`、`offdiag_k=k+1` 构造 Laguerre 的对称 Jacobi 三对角矩阵 `J`。  
4. 调用 `numpy.linalg.eigh(J)` 进行对称本征分解，取本征值作为节点 `x_i`。  
5. 从本征向量矩阵首行取分量 `v_{0i}`，按 `w_i=(v_{0i})^2` 计算权重（`mu_0=1`）。  
6. 回到 `gauss_laguerre_integrate`，在节点上逐点计算 `f(x_i)` 并检查是否有限。  
7. 计算加权和 `Q_n = sum_i w_i f(x_i)`，得到 `\int_0^{\infty} e^{-x} f(x) dx` 近似值。  
8. `print_result_summary` 输出 `estimate/reference/abs_error` 与节点权重样本，完成可复验的端到端演示。  
