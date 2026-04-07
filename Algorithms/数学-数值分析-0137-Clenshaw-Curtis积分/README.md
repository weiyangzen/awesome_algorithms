# Clenshaw-Curtis积分

- UID: `MATH-0137`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `137`
- 目标目录: `Algorithms/数学-数值分析-0137-Clenshaw-Curtis积分`

## R01

Clenshaw-Curtis 积分（Clenshaw-Curtis Quadrature）用于近似定积分：

`I = ∫[a,b] f(x) dx`

它先在标准区间 `[-1,1]` 上选择 Chebyshev-Lobatto 节点

`x_k = cos(kπ/n), k=0..n`

再把节点映射到 `[a,b]`，以预先计算的权重 `w_k` 做加权求和。该方法在光滑函数上通常收敛很快，并且节点天然在端点聚集，对端点附近变化较快的函数比较友好。

## R02

本条目 MVP 目标：

- 输入：可计算函数 `f`、积分区间 `[a,b]`、阶数 `n`（节点数为 `n+1`）。
- 输出：积分近似 `I_n`，并在 demo 中打印误差与经验收敛阶。
- 约束：不调用黑盒积分器，手写节点/权重生成与求积流程。
- 验证：用多组存在解析解的积分问题检查精度与收敛趋势。

## R03

核心公式（标准区间 `[-1,1]`）：

1. 节点：

`x_k = cos(kπ/n), k=0..n`

2. 求积近似：

`∫[-1,1] f(x) dx ≈ Σ_{k=0}^n w_k f(x_k)`

3. 区间映射：

令 `x = (a+b)/2 + (b-a)/2 * t`，则

`∫[a,b] f(x) dx = (b-a)/2 * ∫[-1,1] f((a+b)/2 + (b-a)/2*t) dt`

4. 最终离散形式：

`I_n = (b-a)/2 * Σ_{k=0}^n w_k f((a+b)/2 + (b-a)/2*x_k)`

本实现使用 cosine-series 公式计算 `w_k`，与经典 `clencurt` 实现一致。

## R04

`demo.py` 的单次积分流程：

1. 校验 `a,b,n`（有限数、`n>=0`）。
2. 在 `[-1,1]` 上构造 Chebyshev-Lobatto 节点 `x_k`。
3. 按奇偶 `n` 的不同公式计算端点权重与内部权重。
4. 把标准节点映射到目标区间 `[a,b]`。
5. 在映射点上评估函数值（向量化优先，必要时回退逐点）。
6. 计算 `(b-a)/2 * dot(w, f(x_mapped))` 并返回近似积分值。

## R05

核心数据结构：

- `IntegralCase`：单个测试问题，包含 `name/func/a/b/exact`。
- `n_values`：阶数序列，例如 `2,4,8,...,128`。
- 结果列：`estimate`、`abs_error`、`rel_error`、`obs_order`。

这种结构将“求积实现”和“实验输出”分离，便于替换测试函数或扩展统计指标。

## R06

正确性要点：

- 节点使用 `cos(kπ/n)`，保证是 Clenshaw-Curtis 的标准采样点。
- 权重使用显式余弦级数公式，不依赖外部黑盒求积器。
- 对有解析值的函数（`sin`、`exp`、`1/(1+25x^2)`、`|x|`）直接比较绝对误差。
- 通过 `n` 逐步加倍，观察误差下降和经验阶 `obs_order`，验证实现行为与理论一致。

## R07

复杂度（单次积分）：

- 时间复杂度：`O(n^2)`（权重计算阶段含按 `k` 的余弦累加；积分加权求和为 `O(n)`）。
- 空间复杂度：`O(n)`（存储节点、权重、函数值）。

说明：若进一步用 FFT/DCT 可把权重构造降到更优复杂度，但 MVP 先强调清晰可审计实现。

## R08

边界与异常处理：

- `n < 0`：抛出 `ValueError`；
- `a` 或 `b` 非有限数：抛出 `ValueError`；
- `a == b`：直接返回 `0.0`；
- 函数向量化调用失败时自动回退逐点计算；
- 若函数返回非有限值（`nan/inf`），主动报错，避免静默传播。

## R09

MVP 取舍：

- 选择“手写节点+权重+映射”的最小闭环，而不是直接调用 `scipy.integrate.quad`；
- 权重计算采用经典显式公式，优先可读性与可验证性；
- 不引入自适应细分、误差控制器等额外机制，先保证基础算法正确可运行。

## R10

`demo.py` 函数职责：

- `_check_finite`：输入有限性校验；
- `_evaluate_function`：函数值评估（向量化优先，逐点回退）；
- `clenshaw_curtis_nodes_weights`：构造标准节点与权重；
- `clenshaw_curtis_integrate`：在 `[a,b]` 上完成一次积分；
- `_relative_error` / `_observed_order`：误差与收敛阶统计；
- `run_case`：执行并打印单个案例结果表；
- `main`：组织全部案例并批量运行。

## R11

运行方式：

```bash
cd Algorithms/数学-数值分析-0137-Clenshaw-Curtis积分
python3 demo.py
```

脚本无需任何交互输入，会自动输出 4 组积分问题在多种 `n` 下的对比结果。

## R12

输出字段说明：

- `n`：Clenshaw-Curtis 阶数（节点数为 `n+1`）；
- `estimate`：积分近似值；
- `abs_error`：绝对误差 `|estimate - exact|`；
- `rel_error`：相对误差（`exact=0` 时退化为绝对误差）；
- `obs_order`：相邻两层误差的经验阶 `log2(e_prev/e_curr)`。

`obs_order` 越接近理论趋势，说明实现越符合预期收敛行为。

## R13

当前最小测试集：

- `∫[0,π] sin(x) dx = 2`
- `∫[-1,1] exp(x) dx = e - e^{-1}`
- `∫[-1,1] 1/(1+25x^2) dx = (2/5) arctan(5)`
- `∫[-1,1] |x| dx = 1`（非光滑基准）

这些案例覆盖光滑函数与非光滑函数，便于观察不同收敛特征。

## R14

可调参数：

- `n_values`：阶数列表，决定精度/计算量；
- `cases`：测试函数与区间集合；
- 权重构造函数本身可替换为 FFT/DCT 版本以优化性能。

调参建议：先从小 `n` 验证正确性，再提高到 `64/128` 观察收敛稳定性。

## R15

方法对比（简要）：

- 相比复合梯形：Clenshaw-Curtis 常在同等节点数下更高精度；
- 相比 Gaussian Quadrature：实现更直观、节点可复用，但理论最优性通常弱于高斯求积；
- 相比黑盒自适应积分：可解释性更强，适合教学与底层模块构建。

## R16

适用场景：

- 中高精度的一维定积分近似；
- 端点附近变化更明显的函数积分；
- 谱方法、Chebyshev 相关数值算法中的积分子模块；
- 需要“可审计、可复现”数值流程的工程或教学任务。

## R17

可扩展方向：

- 加入基于 `n -> 2n` 的自适应阶数选择；
- 增加 Richardson 外推或谱系数后处理提升精度；
- 使用 FFT/DCT 快速构造权重并支持更大 `n`；
- 扩展到分段积分、奇异点处理或加权积分版本。

## R18

`demo.py` 源码级算法流（8 步）：

1. `main` 构建 `IntegralCase` 列表和统一 `n_values`，作为批量实验入口。  
2. 每个案例进入 `run_case`，对 `n` 序列逐个调用 `clenshaw_curtis_integrate`。  
3. `clenshaw_curtis_integrate` 先做输入合法性检查，再调用 `clenshaw_curtis_nodes_weights(n)`。  
4. `clenshaw_curtis_nodes_weights` 用 `theta_k = kπ/n` 生成节点 `x_k = cos(theta_k)`。  
5. 权重构造按 `n` 奇偶分支：先设端点权重，再对内部点累加余弦项，得到全部 `w_k`。  
6. 将标准节点仿射映射到 `[a,b]`：`x = (a+b)/2 + (b-a)/2 * x_k`。  
7. `_evaluate_function` 优先向量化评估 `f(x)`，失败时逐点回退，之后执行 `dot(w, f(x))` 并乘缩放因子 `(b-a)/2`。  
8. `run_case` 汇总 `estimate/abs_error/rel_error/obs_order` 并打印表格，用解析值直接验证算法正确性与收敛表现。  
