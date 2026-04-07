# ϵ-展开 (Epsilon Expansion)

- UID: `PHYS-0298`
- 学科: `物理`
- 分类: `统计力学`
- 源序号: `301`
- 目标目录: `Algorithms/物理-统计力学-0301-ϵ-展开_(Epsilon_Expansion)`

## R01

ϵ-展开（epsilon expansion）是重整化群（RG）中处理临界现象的经典近似方法：从可控维度 \(d=4\) 出发，把空间维数写成 \(d=4-\epsilon\)，再把临界指数、固定点耦合等量按 \(\epsilon\) 做级数展开。它的核心价值在于把强耦合临界问题转为“近 4 维的微扰问题”。

## R02

本条目聚焦 \(O(N)\) 对称 \(\phi^4\) 场论在二阶相变附近的最小可运行计算：

- 输入：分量数 \(N\)、维度偏离参数 \(\epsilon=4-d\)。
- 输出：Wilson-Fisher 非平凡固定点 \(g^*\) 与一组临界指数近似（\(\nu,\eta,\gamma,\beta,\alpha,\delta\)）。
- 范围：只做低阶（以一回路为主，\(\eta\) 用常见 \(\epsilon^2\) 近似项）MVP，不做高阶重求和。

## R03

采用无量纲化耦合 \(g\) 后，\(O(N)\) \(\phi^4\) 理论在低阶下常写为

\[
\beta(g) = \frac{dg}{d\ln \mu} = -\epsilon g + \frac{N+8}{6}g^2 + O(g^3).
\]

其中 \(\mu\) 是重整化尺度。固定点满足 \(\beta(g^*)=0\)。

## R04

由上式可得两个固定点：

- 高斯固定点：\(g^*=0\)。
- Wilson-Fisher 固定点（\(\epsilon>0\)）：

\[
g^* = \frac{6\epsilon}{N+8} + O(\epsilon^2).
\]

这说明当 \(d<4\)（即 \(\epsilon>0\)）时，非平凡固定点出现并主导临界普适行为。

## R05

在该固定点附近，可得到常用临界指数近似：

\[
\nu^{-1} = 2 - \frac{N+2}{N+8}\epsilon + O(\epsilon^2),
\]

\[
\eta = \frac{N+2}{2(N+8)^2}\epsilon^2 + O(\epsilon^3).
\]

MVP 中其余指数通过标度关系由 \(\nu,\eta\) 与 \(d=4-\epsilon\) 推出。

## R06

为什么 ϵ-展开在统计力学里实用：

- 直接连接“微观模型 -> RG 流 -> 普适类指数”。
- 给出可解析的维度依赖趋势（例如 \(d=3\) 对应 \(\epsilon=1\)）。
- 可系统提升阶数（两回路、三回路…）并结合 Borel/Pade 等重求和改善精度。

局限也很明确：\(\epsilon=1\) 并不小，低阶结果通常只能给“数量级与趋势正确”的近似。

## R07

本目录算法流程（概念级）如下：

1. 读入 \(N\) 与 \(\epsilon\)（对应维度 \(d=4-\epsilon\)）。
2. 构造一回路 \(\beta(g)\) 与约化方程 \(\beta(g)/g=0\)。
3. 数值求解非平凡固定点 \(g^*\)（同时保留解析解做交叉检查）。
4. 代入低阶公式计算 \(\nu\) 与 \(\eta\)。
5. 用标度关系计算 \(\gamma,\beta,\alpha,\delta\)。
6. 汇总为表格并输出。

## R08

复杂度（设参数组合数为 \(M\)）：

- 每个 \((N,\epsilon)\) 组合只涉及常数规模计算与一次标量根求解，时间约 \(O(1)\)。
- 总时间复杂度约 \(O(M)\)。
- 空间复杂度约 \(O(M)\)（主要是结果表存储）。

## R09

数值实现注意事项：

- \(\epsilon\le 0\) 时不应求 Wilson-Fisher 正固定点（只剩高斯点）。
- \(N\to -8\) 会使公式分母奇异，必须显式拦截。
- 当 \(\nu^{-1}\) 接近 0 时，\(\nu\) 会变得很大，需做稳定性检查。
- 根求解采用带区间的 `brentq`，避免牛顿法初值不稳问题。

## R10

适用场景：

- 教学/科研中的 RG 入门验证。
- 快速比较不同 \(O(N)\) 普适类的指数趋势。
- 为后续高阶重求和、蒙特卡洛或实验数据拟合提供初值与先验。

## R11

与其他方案对比：

- 对比均值场理论：ϵ-展开能给出非平凡维度修正；均值场在 \(d<4\) 常偏差较大。
- 对比高温展开/数值模拟：ϵ-展开更解析、成本低，但精度常不如高精度数值。
- 对比功能 RG：ϵ-展开更轻量透明；功能 RG 更通用但实现复杂度更高。

## R12

`demo.py` 结构：

- `EpsilonExpansionConfig`：参数集合与默认实验设置。
- `beta_phi4_one_loop`：一回路 \(\beta(g)\) 函数。
- `find_wilson_fisher_fixed_point`：数值求固定点并与解析解对照。
- `critical_exponents_low_order`：计算 \(\nu,\eta\) 与标度导出指数。
- `run_mvp`：批量计算、生成 `pandas.DataFrame`。
- `main`：打印结果并执行最小质量断言。

## R13

MVP 实验设置：

- 默认 \(N\in\{1,2,3\}\)（分别对应 Ising / XY / Heisenberg 常见普适类）。
- 默认 \(\epsilon\in\{0.1,0.5,1.0\}\)，覆盖“近 4 维”到 3 维。
- 输出两张表：
  - 固定点与指数计算表。
  - \(\epsilon=1\) 时与常见 3D 参考值的误差对照表（定性检查）。

## R14

运行方式：

```bash
cd Algorithms/物理-统计力学-0301-ϵ-展开_(Epsilon_Expansion)
uv run python demo.py
```

主要依赖：

- `numpy`
- `scipy`
- `pandas`

## R15

输出解读：

- `g_star_numeric` 与 `g_star_analytic`：应高度一致（同一低阶模型的数值/解析交叉验证）。
- `nu, eta, gamma, beta, alpha, delta`：给出对应 \((N,\epsilon)\) 的低阶预测。
- `abs_err_*_vs_ref`：与常见 3D 文献数值的差距，通常显示“趋势正确、精度有限”的特征。

## R16

边界条件与异常处理：

- `N <= -8`：抛出 `ValueError`（分母奇异）。
- `epsilon <= 0`：抛出 `ValueError`（非平凡固定点不在正耦合）。
- `nu_inv <= 0`：抛出 `ValueError`（超出该低阶公式可解释范围）。
- 数值根求解失败：抛出 `RuntimeError`。

## R17

可扩展方向：

- 引入两回路/三回路 \(\beta\) 与指数公式。
- 增加 Borel-Pade 重求和模块，提高 \(\epsilon=1\) 预测精度。
- 接入蒙特卡洛或高温展开数据，做参数反演与误差标定。
- 对更多 \(N\) 与分数量子维度做相图扫描。

## R18

`demo.py` 源码级算法流（8 步，含第三方调用细节）：

1. `main` 构建默认 `EpsilonExpansionConfig`，触发 `run_mvp` 批量计算 \((N,\epsilon)\)。
2. `run_mvp` 对每组参数调用 `find_wilson_fisher_fixed_point`，先定义约化方程 \(-\epsilon + \frac{N+8}{6}g = 0\)。
3. `find_wilson_fisher_fixed_point` 调用 `scipy.optimize.root_scalar(..., method="brentq")`，在给定区间内做有界求根。
4. `root_scalar` 的 `brentq` 路径按“二分保区间 + 割线/逆二次插值加速”的混合策略迭代，直到函数值或区间宽度满足容差。
5. 求得 `g_star_numeric` 后，函数同时返回解析表达式 `g_star_analytic=6\epsilon/(N+8)` 与相对误差用于交叉验证。
6. `critical_exponents_low_order` 用 \(\nu^{-1}\)、\(\eta\) 低阶式计算 \(\nu,\eta\)，再用标度关系推导 \(\gamma,\beta,\alpha,\delta\)。
7. `run_mvp` 将所有结果聚合为 `pandas.DataFrame`，并额外生成 \(\epsilon=1\) 对照误差表（对常见 3D 参考指数）。
8. `main` 打印两张表并执行断言（固定点数值/解析一致性、指数正性等），确保脚本可作为稳定 MVP 运行。
