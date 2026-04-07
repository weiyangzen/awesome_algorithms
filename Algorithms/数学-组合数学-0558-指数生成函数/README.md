# 指数生成函数

- UID: `MATH-0558`
- 学科: `数学`
- 分类: `组合数学`
- 源序号: `558`
- 目标目录: `Algorithms/数学-组合数学-0558-指数生成函数`

## R01

指数生成函数（Exponential Generating Function, EGF）是带标签组合计数中的核心工具。  
若序列是 `a_n`，其 EGF 定义为
\[
A(x)=\sum_{n\ge 0} a_n\frac{x^n}{n!}.
\]

与普通生成函数（OGF）相比，EGF 的 `n!` 缩放天然匹配“元素带标签”的排列方式，常用于集合分拆、排列类计数、结构组合构造等问题。

## R02

本条目用经典问题“Bell 数（集合分拆计数）”作为 MVP：
- `B_n`：将 `n` 个带标签元素划分为任意个非空无序块的方案数。
- 其 EGF 为
\[
\sum_{n\ge 0} B_n\frac{x^n}{n!}=\exp(\exp(x)-1).
\]

目标：给定 `n_max`，通过截断 EGF 计算 `B_0...B_{n_max}`，并用 Stirling 数递推进行真值校验。

## R03

记
\[
g(x)=\sum_{n=0}^{N} g_n x^n,\quad f(x)=\exp(g(x))=\sum_{n=0}^{N} f_n x^n.
\]

对幂级数有系数递推（由 `f' = g' f` 得）：
\[
n f_n = \sum_{k=1}^{n} k g_k f_{n-k},\quad n\ge1.
\]

在本问题中，
\[
g(x)=\exp(x)-1=\sum_{n\ge1}\frac{x^n}{n!},
\]
故 `g_n = 1/n! (n>=1)`，`g_0=0`。求得 `f_n` 后可恢复
\[
B_n = f_n\cdot n!.
\]

## R04

为什么这个递推适合做 MVP：
- 不依赖符号系统，纯数值/有理数运算即可。
- 只需要一次双层循环即可得到到 `N` 阶全部系数。
- 能清晰展示 EGF 里“先做函数构造，再读系数”的算法化流程。

`demo.py` 用 `Fraction` 做有理数计算，避免浮点误差对整数序列恢复造成干扰。

## R05

边界与初始化：
- 截断上界 `N >= 0`。
- `f_0 = exp(g_0)`；本例 `g_0=0`，因此 `f_0=1`。
- `n=0` 时直接得到 `B_0 = 1`。
- 递推顺序必须从低阶到高阶，保证 `f_{n-k}` 先已就绪。

本实现显式检查 `g_0==0`，保证演示路径保持精确有理数。

## R06

校验策略：
- 主算法：EGF 系数递推得到 `B_n`。
- 对照算法：Stirling 数第二类递推
\[
S(n,k)=S(n-1,k-1)+kS(n-1,k),\quad B_n=\sum_k S(n,k).
\]
- 若两者不一致则抛错，阻止“看似运行成功但数学不正确”。

## R07

伪代码：

```text
input N
fact[n] = n! for n=0..N
g[0]=0; g[n]=1/fact[n] for n>=1

f[0]=1
for n in 1..N:
    s = 0
    for k in 1..n:
        s += k * g[k] * f[n-k]
    f[n] = s / n

bell_egf[n] = f[n] * fact[n]
bell_ref[n] = sum_k S(n,k)  (Stirling DP)
assert bell_egf == bell_ref
return bell_egf
```

## R08

正确性要点：
1. `f=exp(g)` 满足微分关系 `f'=g'f`。  
2. 比较 `x^{n-1}` 系数得到 `n f_n = Σ_{k=1}^n k g_k f_{n-k}`。  
3. 本算法逐阶计算该递推，故得到唯一正确的截断系数 `f_0..f_N`。  
4. EGF 定义决定 `a_n = n! [x^n]A(x)`，因此 `B_n = f_n n!`。  
5. Stirling 递推是独立已知计数公式，对拍一致即为强校验。

## R09

复杂度：
- EGF 递推部分：时间 `O(N^2)`，空间 `O(N)`。
- Stirling 校验部分：时间 `O(N^2)`，空间 `O(N^2)`（本实现保留整张表便于说明）。

在本 MVP 里 `N=10`，运行开销极小，重点在算法可解释性与可靠性。

## R10

常见错误：
- 把 OGF 系数当成 EGF 系数，忘记 `n!` 缩放。
- 直接用浮点做高阶系数恢复，得到“接近整数但不等于整数”的误判。
- 写成 `f_n = Σ g_k f_{n-k}` 这种缺失 `k/n` 因子的错误递推。
- 只打印序列不做独立对拍，难以及时发现实现偏差。

## R11

`demo.py` 结构：
- `factorial_table`：生成 `0..N` 的阶乘。
- `exp_series`：实现 `f=exp(g)` 的截断幂级数递推。
- `bell_numbers_via_egf`：构造 `g(x)=exp(x)-1` 并恢复 Bell 数。
- `bell_numbers_via_stirling`：作为独立真值基准。
- `main`：执行、打印结果表、断言一致性。

## R12

运行方式（无交互）：

```bash
cd Algorithms/数学-组合数学-0558-指数生成函数
uv run python demo.py
```

脚本会输出：
- `n`、`[x^n]A(x)=B_n/n!`（有理数）
- `B_n`（EGF 计算）
- `B_n`（Stirling 校验）
并在末尾报告是否完全一致。

## R13

内置样例参数：
- `n_max = 10`
- 目标 EGF：`A(x)=exp(exp(x)-1)`

对应 Bell 数前几项应为：
`1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975`。

`demo.py` 会自动给出完整表格并验证该序列。

## R14

可调项：
- 改 `n_max` 可控制截断阶数与输出长度。
- 若只关心速度可把 `Fraction` 改成浮点；若关心严格整数恢复建议保持 `Fraction`。
- 若要扩展到其它 EGF，只需替换 `g` 的系数构造，再复用 `exp_series`。

## R15

与相关方法的关系：
- OGF：更适合“无标签”计数；EGF：更适合“有标签”结构。
- Bell 数也可由 Dobinski 公式或 Bell 三角形计算；本条目强调“从 EGF 出发”的可程序化路径。
- Stirling 数递推在这里扮演“独立验证器”，不是主求解路径。

## R16

应用场景（EGF 思想层面）：
- 带标签对象的结构计数（集合分拆、置换结构、组合类构造）。
- 在算法竞赛或研究原型中，把复杂计数问题转成“函数构造 + 读系数”。
- 在概率与统计组合中，用生成函数做矩、累积量与分布近似分析。

## R17

可扩展方向：
- 实现更多幂级数算子：`log`、`inverse`、`power`，形成通用 EGF 工具箱。
- 用 FFT/NTT 加速卷积，把某些 `O(N^2)` 步骤优化到准线性。
- 加入更多基准序列（如错排数、 involution 数）验证框架通用性。
- 对大 `N` 增加高精数值与误差控制策略。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 设定 `n_max`，调用 `bell_numbers_via_egf` 与 `bell_numbers_via_stirling` 两条路径。  
2. `bell_numbers_via_egf` 先用 `factorial_table` 构造 `0..N` 阶乘，并组装 `g_n=1/n!`（`g_0=0`）。  
3. 进入 `exp_series(g, N)`，初始化 `f_0=1`，准备逐阶计算 `f_n`。  
4. 对每个 `n=1..N`，按 `n f_n = Σ_{k=1}^n k g_k f_{n-k}` 做双层循环累加并除以 `n`。  
5. 将得到的 `f_n=[x^n]exp(exp(x)-1)` 乘以 `n!`，恢复整数 `B_n`。  
6. `bell_numbers_via_stirling` 构建 `S(n,k)` 表，并用 `B_n=Σ_k S(n,k)` 生成独立真值序列。  
7. `main` 用 `numpy.array_equal` 断言两序列完全一致；不一致立即抛出异常。  
8. 使用 `pandas.DataFrame` 输出 `n`、有理系数 `B_n/n!`、两条路径的 `B_n`，给出最终结论。
