# Gauss-Hermite求积

- UID: `MATH-0134`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `134`
- 目标目录: `Algorithms/数学-数值分析-0134-Gauss-Hermite求积`

## R01

本条目实现 Gauss-Hermite 求积（Gaussian Quadrature with Hermite polynomials）的最小可运行版本。

核心目标：
- 近似计算带权无界积分 `I = \int_{-\infty}^{\infty} e^{-x^2} f(x) dx`；
- 显式展示节点（nodes）与权重（weights）如何由 Hermite 对应 Jacobi 矩阵的特征分解得到；
- 通过多组样例验证多项式精确性与一般光滑函数的收敛表现。

## R02

问题定义（有限精度计算版）：
- 输入：
  - 被积函数 `f: R -> R`；
  - 求积阶数 `n >= 1`（节点数）。
- 输出：
  - 积分近似值 `Q_n`；
  - 节点数组 `x_i` 与权重数组 `w_i`；
  - 若有解析真值，输出绝对误差 `|Q_n - I|`。

积分模型采用经典 Hermite 权函数：
`I = \int_{-\infty}^{\infty} e^{-x^2} f(x) dx`。

## R03

数学基础与公式：

1) Gauss-Hermite 公式
- `\int_{-\infty}^{\infty} e^{-x^2} f(x) dx \approx \sum_{i=1}^{n} w_i f(x_i)`。
- `x_i` 是 `n` 次 Hermite 多项式（物理学家版本）的根。
- 对任意次数 `<= 2n-1` 的多项式 `p(x)`，上述公式精确。

2) Golub-Welsch 构造
- Hermite 对应 Jacobi 矩阵 `J` 为对称三对角，主对角为 0；
- 次对角/超对角元素为 `\beta_k = sqrt(k/2)`，`k=1,...,n-1`；
- `J` 的特征值即节点 `x_i`。

3) 权重公式
- 设 `v_i` 为 `J` 的单位特征向量（对应 `x_i`），其首分量为 `v_i[0]`；
- `\mu_0 = \int_{-\infty}^{\infty} e^{-x^2} dx = sqrt(pi)`；
- 权重 `w_i = \mu_0 * (v_i[0])^2 = sqrt(pi) * (v_i[0])^2`。

4) 与标准正态期望的关系
- 若 `Z ~ N(0,1)`，则
  `E[g(Z)] = (1/sqrt(pi)) \int_{-\infty}^{\infty} e^{-x^2} g(\sqrt{2}x) dx`
  `\approx (1/sqrt(pi)) \sum_i w_i g(\sqrt{2}x_i)`。

## R04

算法总览（MVP）：
1. 校验 `n >= 1`。
2. 构造 Hermite 的 `n x n` 对称三对角 Jacobi 矩阵。
3. 调用 `numpy.linalg.eigh` 做特征分解。
4. 取特征值作为节点 `x_i`。
5. 用特征向量首行计算权重 `w_i = sqrt(pi)*(v_i[0])^2`。
6. 在节点上采样 `f(x_i)` 并计算 `Q_n = sum(w_i * f(x_i))`。
7. 输出近似值、真值（若提供）和误差。

## R05

核心数据结构：
- `numpy.ndarray`：
  - `nodes: shape=(n,)`，Gauss-Hermite 节点；
  - `weights: shape=(n,)`，对应权重；
  - `fx: shape=(n,)`，节点上的函数值采样。
- `IntegrationResult`（`dataclass`）：
  - 汇总 `n`、`estimate`、`nodes`、`weights`、`fx`。

## R06

正确性要点：
- Hermite 多项式族在 `(-\infty,\infty)` 上关于权函数 `e^{-x^2}` 正交；
- 高斯求积理论保证：使用 `n` 个节点时，对 `2n-1` 次以内多项式精确；
- Golub-Welsch 定理保证 Jacobi 矩阵特征值/特征向量与节点/权重的对应关系；
- 权重由零阶矩 `sqrt(pi)` 与特征向量首分量平方构成，保证权重非负且总和为 `sqrt(pi)`。

## R07

复杂度分析：
- 构造 Jacobi 矩阵：`O(n)` 非零元素（按稠密实现占 `O(n^2)` 空间）；
- 对称特征分解（稠密 `eigh`）：时间 `O(n^3)`，空间 `O(n^2)`；
- 函数采样与加权求和：`O(n)`；
- 总体瓶颈在特征分解。

## R08

边界与异常处理：
- `n < 1`：抛出 `ValueError`；
- 被积函数在节点处返回 `nan/inf`：抛出 `ValueError`；
- 样例只使用确定性函数，避免随机扰动导致不可复现。

## R09

MVP 取舍：
- 使用 `numpy` + 标准库，工具栈最小化；
- 不直接调用 `numpy.polynomial.hermite.hermgauss` 黑盒接口，而是显式实现 Golub-Welsch 主流程；
- 保留节点、权重与采样向量，优先可解释性与可验证性；
- 不做自适应阶数选择，聚焦固定 `n` 下的算法透明实现。

## R10

`demo.py` 函数职责：
- `gauss_hermite_nodes_weights`：构造 Jacobi 矩阵并计算节点权重；
- `gauss_hermite_integrate`：执行一次带权积分并返回 `IntegrationResult`；
- `gauss_hermite_normal_expectation`：利用 Gauss-Hermite 近似标准正态期望；
- `print_weighted_integral_summary`：打印带权积分结果；
- `print_normal_expectation_summary`：打印标准正态期望结果；
- `run_examples`：组织并运行全部样例；
- `main`：无交互入口。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0134-Gauss-Hermite求积
python3 demo.py
```

脚本无输入参数，运行后会自动输出多组 `n` 下的近似值和误差。

## R12

输出解读：
- `n`：Gauss-Hermite 节点数；
- `estimate`：数值积分（或期望）近似值；
- `reference`：解析真值；
- `abs_error`：绝对误差；
- `sample_nodes`：前几个节点样本；
- `sample_weights`：前几个权重样本。

可观测现象：
- 对低阶多项式，达到理论阶数后误差可接近机器精度；
- 对光滑函数（如 `cos(x)`），随着 `n` 增大，误差通常快速下降。

## R13

建议最小测试集：
- 多项式精确性：`f(x)=x^4`，真值 `3*sqrt(pi)/4`（`n>=3` 应精确）；
- 光滑函数：`f(x)=cos(x)`，真值 `sqrt(pi)*exp(-1/4)`；
- 概率期望：`E[Z^4]` (`Z~N(0,1)`)，真值 `3`；
- 异常测试：`n=0`、函数返回非有限值。

## R14

可调参数：
- `n_values`：每组样例的阶数列表（如 `[2,3,4,8,16]`）；
- `preview_k`：日志中展示的节点/权重个数；
- 样例函数集合：可替换为业务实际积分函数。

实践建议：
- 先用小 `n` 验证流程，再逐步增大 `n` 观察误差趋势；
- 若函数增长过快，注意 `e^{-x^2}f(x)` 的数值稳定性。

## R15

方法对比：
- 与 Gauss-Legendre 相比：
  - Gauss-Legendre 适合有限区间 `[a,b]`；
  - Gauss-Hermite天然适配 `(-\infty,\infty)` 且权函数为 `e^{-x^2}` 的积分。
- 与蒙特卡洛估计正态期望相比：
  - Gauss-Hermite 在低维光滑场景通常收敛更快、噪声更低；
  - 但维度很高时会遭遇张量积节点爆炸。

## R16

应用场景：
- 统计与机器学习中的高斯加权积分与期望计算；
- 量子力学、热传导等出现 Hermite 权函数的模型；
- 金融工程中对正态分布函数的高精度确定性积分近似。

## R17

后续扩展方向：
- 使用三对角专用本征算法提升大 `n` 性能；
- 扩展到 Gauss-Laguerre / Gauss-Jacobi 等其他权函数家族；
- 引入张量积 Gauss-Hermite 处理低维多元高斯期望；
- 增加与 SciPy 参考实现的单元测试对照。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `run_examples` 组装三类样例：带权多项式积分、带权三角函数积分、标准正态期望。  
2. 每次计算先调用 `gauss_hermite_nodes_weights(n)` 校验 `n`，并创建 Hermite Jacobi 三对角矩阵 `J`。  
3. 在 `gauss_hermite_nodes_weights` 内使用 `numpy.linalg.eigh(J)` 求本征对。  
4. 取本征值作为节点 `x_i`，取特征向量首行 `v_0i` 计算权重 `w_i = sqrt(pi) * v_0i^2`。  
5. `gauss_hermite_integrate` 在节点上计算 `f(x_i)`，并做点积 `sum(w_i * f(x_i))` 得到积分估计。  
6. 若用于正态期望，`gauss_hermite_normal_expectation` 先做变量替换 `z = sqrt(2)x`，再把积分结果除以 `sqrt(pi)`。  
7. 结果打包为 `IntegrationResult`，保留节点、权重、采样值，避免“只有最终数值、缺少过程”。  
8. 打印 `estimate/reference/abs_error` 以及节点权重样本，直接验证精度与收敛行为。  
