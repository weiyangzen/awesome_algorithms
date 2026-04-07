# 正规化 (Regularization)

- UID: `PHYS-0385`
- 学科: `物理`
- 分类: `量子场论`
- 源序号: `404`
- 目标目录: `Algorithms/物理-量子场论-0404-正规化_(Regularization)`

## R01

正规化（Regularization）是在量子场论中把发散积分临时变成“可计算量”的步骤。  
本条目用一个最小可运行例子演示 UV 发散的处理：

`I(Λ,m) = ∫_{|k|<Λ} d^4k_E/(2π)^4 * 1/(k^2 + m^2)`。

这里 `Λ` 是硬截止（hard cutoff），`m` 是质量参数。

## R02

本目录解决的核心问题：

1. 用数值积分与闭式公式同时计算一圈 tadpole 积分；
2. 直观看到原始积分随 `Λ` 增大而发散；
3. 显式减去发散项后，得到随 `Λ` 收敛到有限值的“重整化后量”；
4. 用断言做自动化正确性检查，保证脚本可复用。

## R03

采用的数学对象（欧氏动量空间）为：

`I(Λ,m) = ∫_{|k|<Λ} d^4k_E/(2π)^4 * 1/(k^2 + m^2)`。

在 4 维球坐标下：

- `d^4k = 2π^2 k^3 dk`；
- 所以 `I(Λ,m) = (1/(8π^2)) ∫_0^Λ k^3/(k^2+m^2) dk`。

这一定义在代码里由 `tadpole_integrand_radial` 与径向 `quad` 数值积分直接实现。

## R04

硬截止正规化的闭式结果：

`I(Λ,m) = [Λ^2 - m^2 ln(1 + Λ^2/m^2)] / (16π^2)`。

这给出两种重要信息：

1. 存在 `Λ^2` 的幂发散；
2. 同时有 `m^2 ln(Λ^2)` 的对数发散。

`demo.py` 中 `tadpole_cutoff_analytic` 就是这条公式。

## R05

大截止展开（`Λ >> m`）使用：

`I(Λ,m) ≈ [Λ^2 - m^2 ln(Λ^2/m^2)]/(16π^2)`。

这一步的作用：

- 解释“发散来自哪里”；
- 和精确闭式比较误差，验证渐近展开在大 `Λ` 的有效性。

`demo.py` 中由 `tadpole_cutoff_asymptotic` 给出并进入结果表。

## R06

减法重整化（本 MVP 使用显式减去发散项的 toy scheme）：

`I_R(Λ,m;μ) = I(Λ,m) - [Λ^2 - m^2 ln(Λ^2/μ^2)]/(16π^2)`。

其中 `μ` 是重整化尺度。此时有

`lim_{Λ→∞} I_R(Λ,m;μ) = [m^2 ln(m^2/μ^2)]/(16π^2)`。

脚本会计算 `I_R` 并对照该极限，验证有限性。

## R07

`demo.py` 的功能闭环：

1. 固定 `m=0.7`、`μ=1.0`、`Λ ∈ {2,4,8,16,32,64}`；
2. 对每个 `Λ` 同时计算：数值积分、闭式结果、渐近结果、减法后结果；
3. 组装 `pandas.DataFrame` 输出；
4. 做四个断言：数值-闭式一致、原始量单调发散、减法后趋于极限、误差随 `Λ` 下降。

## R08

复杂度（`N` 为 cutoff 网格点数）：

- 每个 cutoff 一次 1D 数值积分，成本记为 `Q`；
- 总时间复杂度约 `O(N*Q)`；
- 其余闭式/渐近/减法计算是 `O(N)`；
- 空间复杂度 `O(N)`（结果表存储）。

本例 `N` 很小，运行时间主要由 `scipy.integrate.quad` 决定。

## R09

关键数据结构：

- `RegularizationConfig`：保存 `mass`、`mu_ren`、`cutoffs`、积分容差；
- `pandas.DataFrame`：统一承载每个 cutoff 的观测量；
- 列字段包括：
  `I_numeric`, `I_analytic`, `I_asymptotic`, `I_ren_subtracted`, 以及对应误差列。

## R10

运行方式（无交互输入）：

```bash
cd Algorithms/物理-量子场论-0404-正规化_(Regularization)
uv run python demo.py
```

或从仓库根目录：

```bash
uv run python Algorithms/物理-量子场论-0404-正规化_(Regularization)/demo.py
```

## R11

脚本输出解释：

1. 配置参数：`m`、`μ` 与理论极限值；
2. 主结果表（按 cutoff 行展示）：
   - `I_numeric`：径向积分数值值；
   - `I_analytic`：闭式精确值；
   - `abs_err_numeric_vs_analytic`：数值实现误差；
   - `I_asymptotic` 与其误差；
   - `I_ren_subtracted` 与有限极限的偏差；
3. 若断言全部通过，打印 `All checks passed.`。

## R12

正确性验证策略：

1. 数值 vs 闭式：`max abs_err_numeric_vs_analytic < 1e-10`；
2. 发散行为：`I_analytic` 随 `Λ` 严格上升；
3. 有限化行为：最大 cutoff 处减法结果接近极限（阈值 `5e-7`）；
4. 收敛趋势：减法误差随 cutoff 增大而降低。

这四条覆盖了“算对了”和“物理行为对了”两类检查。

## R13

适用场景：

- 量子场论课程中解释“为什么要正规化”；
- 作为更复杂重整化流程（MS-bar、维数正规化、RGE）的入门基线；
- 验证数值程序里的发散结构与 counterterm 逻辑。

不适用场景：

- 真实多圈图、张量结构复杂的振幅计算；
- 需要方案无关物理可观测量的完整现象学分析；
- 需要格点 QCD 或非微扰方法的任务。

## R14

常见失效模式与排查：

1. `mass<=0`、`μ<=0` 或 cutoff 非递增，触发输入校验异常；
2. 把体积因子写错（`2π^2` 或 `(2π)^4`），会导致整体数值偏差；
3. 把 `ln(1+x)` 写成 `ln(x)`，小 cutoff 区域误差显著；
4. 只看原始积分不做减法，会误以为“算法不收敛”；
5. 容差过宽会使数值-闭式误差变大，断言失败。

## R15

可扩展方向：

1. 改为维数正规化形式，加入 `1/ε` 极点与 MS-bar 减法；
2. 对比不同正规化方案（硬截止、Pauli-Villars、dim-reg）在有限部分上的差异；
3. 推广到两点函数自能并提取波函数重整化常数；
4. 把 counterterm 参数化后做拟合，连接“重整化条件”。

## R16

与相关条目的关系：

- 与“跑动耦合常数/渐近自由”互补：本条目解决的是发散积分的可计算化；
- 与“重整化群”关系：正规化引入尺度，重整化处理尺度依赖；
- 与“有效场论 cutoff”关系：硬截止也可看作 EFT 的物理 UV 截断。

## R17

最小可交付能力清单（本条目已覆盖）：

1. 提供至少一种明确正规化方案（硬截止）；
2. 同时给出数值实现与闭式对照；
3. 显式展示发散项与减法后的有限量；
4. 输出结构化结果表（`pandas`）；
5. 提供可复现断言，保证 `uv run python demo.py` 自动通过。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 构造 `RegularizationConfig`（质量、重整化尺度、cutoff 网格、积分容差）。  
2. `build_report` 逐个 cutoff 循环，调用 `tadpole_cutoff_numeric` 做径向数值积分。  
3. `tadpole_cutoff_numeric` 先做输入检查，再把 `k^3/(k^2+m^2)` 送入 `quad`，最后乘 `1/(8π^2)` 角向因子。  
4. 同一 cutoff 上调用 `tadpole_cutoff_analytic` 计算闭式硬截止结果。  
5. 调用 `tadpole_cutoff_asymptotic` 计算大 cutoff 渐近式，并和闭式比较误差。  
6. 调用 `renormalized_tadpole_subtracted` 显式减去 `Λ^2` 与 `m^2 ln Λ^2` 发散项，再与 `renormalized_limit` 对照。  
7. 将每个 cutoff 的全部标量写入 `pandas.DataFrame`，形成可审计结果表。  
8. `main` 对结果执行四个断言并打印 `All checks passed.`。

说明：`scipy.integrate.quad` 只承担通用数值积分；发散结构、角向归一化、反项减法和验证逻辑都在源码中逐步展开，不是黑盒调用。
