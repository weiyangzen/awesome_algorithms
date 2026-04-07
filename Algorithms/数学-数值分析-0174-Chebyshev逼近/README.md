# Chebyshev逼近

- UID: `MATH-0174`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `174`
- 目标目录: `Algorithms/数学-数值分析-0174-Chebyshev逼近`

## R01

本条目实现 Chebyshev 逼近（Chebyshev polynomial approximation）的最小可运行版本：

- 在区间 `[-1, 1]` 上把函数近似为有限阶 Chebyshev 级数；
- 显式展示“系数拟合 + 级数求值”的核心数值流程；
- 用可复现实验输出误差指标（`max_abs_error` / `rmse`）。

## R02

问题定义（数值计算版）：

- 输入：
  - 函数 `f(x)`（在 `[-1,1]` 上可采样）；
  - 近似阶数 `n >= 0`；
  - 采样点数 `N >= n+1`。
- 输出：
  - Chebyshev 系数 `c_0...c_n`；
  - 近似多项式 `p_n(x) = sum(c_j T_j(x), j=0..n)`；
  - 误差评估（稠密网格上最大绝对误差和 RMSE）。

## R03

数学基础：

1. Chebyshev 多项式定义：`T_j(x) = cos(j arccos x)`，`x in [-1,1]`。  
2. 在 Chebyshev-Gauss 点 `x_k = cos(theta_k)`、`theta_k=(k+1/2)pi/N` 上，离散正交关系成立。  
3. 系数离散公式（MVP 使用）：  
   `c_j = (2/N) * sum_{k=0}^{N-1} f(x_k) cos(j theta_k)`，最终 `c_0` 再乘 `1/2`。  
4. 级数求值采用 Clenshaw 递推，避免直接构造高次幂带来的不稳定性。

## R04

算法总览（MVP）：

1. 校验参数：`degree >= 0` 且 `sample_count >= degree+1`。
2. 生成 Chebyshev-Gauss 采样点 `x_k` 与角度 `theta_k`。
3. 计算函数采样值 `f(x_k)`。
4. 用离散正交公式计算 `c_0...c_n`。
5. 用 Clenshaw 递推在稠密网格上计算 `p_n(x)`。
6. 统计 `max_abs_error` 与 `rmse` 并打印结果表。

## R05

核心数据结构：

- `numpy.ndarray`：
  - `x_nodes/theta`：采样点与对应角度；
  - `coeffs`：Chebyshev 系数向量；
  - `x_grid/y_true/y_pred`：评估网格及真值、预测值。
- `ApproximationReport`（`dataclass`）：
  - `name, degree, sample_count, max_abs_error, rmse, coeffs`。

## R06

正确性要点：

- 对于足够平滑函数，Chebyshev 级数截断通常能快速收敛；
- 离散正交公式在 Chebyshev-Gauss 采样上提供稳定的系数估计；
- Clenshaw 递推与 Chebyshev 基底匹配，数值稳定性优于直接展开到幂基；
- `demo.py` 内置“已知 Chebyshev 多项式系数恢复”自检，验证拟合与求值流程。

## R07

复杂度分析：

- 拟合阶段：
  - 对每个阶 `j` 计算一次长度 `N` 的余弦向量并做内积；
  - 时间复杂度 `O(nN)`，额外空间复杂度 `O(N)`（除系数输出外）。
- 评估阶段（网格大小 `G`）：
  - Clenshaw 求值时间 `O(nG)`，额外空间 `O(G)`。
- 本 MVP 侧重透明性，不做 FFT/DCT 加速。

## R08

边界与异常处理：

- `degree < 0`：抛出 `ValueError`；
- `sample_count < degree+1`：抛出 `ValueError`；
- 函数返回形状不匹配：抛出 `ValueError`；
- 函数值含 `nan/inf`：抛出 `ValueError`。

## R09

MVP 取舍：

- 只依赖 `numpy` + 标准库；
- 不调用 `numpy.polynomial.chebyshev.chebfit` 作为黑箱；
- 系数公式与 Clenshaw 递推都在源码中显式实现；
- 以教学可读与可验证为先，不追求极限性能。

## R10

`demo.py` 结构：

- `validate_hyperparams`：参数合法性检查；
- `chebyshev_gauss_nodes`：生成 Chebyshev-Gauss 节点；
- `fit_chebyshev_by_discrete_orthogonality`：离散正交法拟合系数；
- `eval_chebyshev_clenshaw`：Clenshaw 递推求值；
- `evaluate_report`：计算误差指标并返回报告；
- `run_polynomial_exactness_check`：多项式精确恢复自检；
- `run_examples`：执行固定样例并打印表格；
- `main`：无交互入口。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0174-Chebyshev逼近
python3 demo.py
```

脚本不需要交互输入，会直接输出自检结果与三组函数的逼近误差表。

## R12

输出字段解释：

- `degree`：Chebyshev 近似阶数；
- `sample_count`：拟合系数时采用的 Chebyshev-Gauss 采样点数；
- `max_abs_error`：稠密网格上的最大绝对误差；
- `rmse`：稠密网格上的均方根误差；
- `coeff preview`：高阶实验下前 6 个系数，便于观察系数衰减。

## R13

最小验证集（`demo.py` 已覆盖）：

- `exp(x)`：解析且平滑，观察快速收敛；
- `1/(1+25x^2)`（Runge 型）：观察端点附近逼近表现；
- `abs(x)`：非光滑函数，观察收敛速度变慢；
- 已知 Chebyshev 系数多项式：验证系数恢复精度应接近机器精度。

## R14

可调参数：

- `degree`：近似阶数，决定多项式复杂度；
- `sample_count`：系数拟合采样点数；
- `sample_factor`：示例中按阶数自动放大采样点数；
- `grid_size`：误差评估网格密度。

经验：先用中小阶验证流程，再增大 `degree` 和 `sample_count` 观察误差收敛趋势。

## R15

方法对比：

- 与等距节点幂基多项式拟合相比：
  - Chebyshev 节点更抑制端点振荡；
  - Clenshaw 求值更稳健。
- 与最小二乘黑箱接口相比：
  - 黑箱更省代码；
  - 本实现更利于理解“节点-系数-递推”完整链路。

## R16

应用场景：

- 函数逼近与模型降阶；
- 数值积分/微分前的函数预拟合；
- 谱方法与伪谱方法中的函数表示；
- 科学计算课程中的正交多项式实验。

## R17

后续扩展：

- 用 DCT/FFT 形式重写系数计算以降低常数开销；
- 扩展到带权逼近和分段 Chebyshev 逼近；
- 增加自动选阶（按误差阈值停止）；
- 增加与 `chebfit` / `Chebyshev.fit` 的交叉验证测试。

## R18

源码级算法流（对应 `demo.py`，8 步）：

1. `run_examples` 组装示例函数、阶数列表和采样倍数。  
2. 每组参数先进入 `fit_chebyshev_by_discrete_orthogonality`，由 `validate_hyperparams` 做参数校验。  
3. `chebyshev_gauss_nodes` 生成 `x_k=cos((k+1/2)pi/N)` 和 `theta_k`。  
4. 在节点采样 `f(x_k)`，按阶 `j` 逐次计算 `cos(j*theta_k)` 并做内积。  
5. 按离散正交公式计算 `c_j`，最后将 `c_0` 乘 `1/2` 得到 Chebyshev 系数。  
6. `eval_chebyshev_clenshaw` 用反向递推计算 `p_n(x)`，避免直接展开高次多项式。  
7. `evaluate_report` 在稠密网格上对比 `p_n(x)` 与 `f(x)`，计算 `max_abs_error` 与 `rmse`。  
8. 打印误差表与系数预览；同时 `run_polynomial_exactness_check` 断言系数恢复与网格误差达到机器精度量级。  
