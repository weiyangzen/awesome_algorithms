# Catalan数计算

- UID: `MATH-0551`
- 学科: `数学`
- 分类: `组合数学`
- 源序号: `551`
- 目标目录: `Algorithms/数学-组合数学-0551-Catalan数计算`

## R01

Catalan 数是组合数学中的经典计数序列，前几项为：
`1, 1, 2, 5, 14, 42, 132, ...`。

它出现在大量等价问题中，例如：
- 含 `n` 对括号的合法括号串个数；
- `n+1` 个叶子的满二叉树结构数；
- 凸 `n+2` 边形的三角剖分方案数；
- 不越过对角线的 Dyck 路径数。

本条目实现一个最小但可验证的 MVP，给出多种算法并进行交叉校验。

## R02

标准定义：

`C_0 = 1`

`C_n = sum_{i=0}^{n-1} C_i * C_{n-1-i}, n >= 1`

这一定义对应“把结构按根处分解为左右两个子结构”的组合拆分思想。

## R03

闭式公式：

`C_n = (1 / (n + 1)) * binom(2n, n)`

也可写为：

`C_n = binom(2n, n) - binom(2n, n + 1)`

闭式适合直接计算单点 `C_n`，并且在 Python 大整数支持下可以得到精确整数结果。

## R04

本 MVP 在 `demo.py` 中实现 4 条计算路径：

1. `catalan_dp`：`O(n^2)` 动态规划递推；
2. `catalan_binomial`：`math.comb` + 闭式；
3. `catalan_multiplicative`：整数安全的乘法递推；
4. `catalan_scipy_exact`：`scipy.special.comb(exact=True)` 作为独立校验通道。

四路结果按 `n=0..25` 全量比对，任何不一致都会抛出异常。

## R05

乘法递推使用关系：

`C_{k+1} = C_k * 2*(2k+1)/(k+2)`

实现时每一步都先乘后整除，且该除法在 Catalan 数上总是整除，因此不引入浮点误差，适合较大 `n` 的稳定整数计算。

## R06

为何保留多种实现而非单一闭式：

- DP 体现组合递推定义，教学直观；
- 闭式方法速度快，写法简洁；
- 乘法递推避免中间出现超大二项式再除法；
- SciPy 通道提供“外部实现”对照，降低同源错误风险。

## R07

数据组织：

- 用 `CatalanRecord` 记录每个 `n` 下四种计算值；
- 属性 `all_equal` 统一判断是否完全一致；
- `records_to_dataframe` 把结果转换成 `pandas.DataFrame`，包含：
  - `n`
  - `Catalan`
  - `digits`
  - 三个一致性布尔列。

## R08

输入输出约定：

- 无命令行交互输入；
- `main` 固定 `max_n = 25`；
- 标准输出打印完整结果表。

因此可直接被自动化验证命令调用：

`uv run python demo.py`

## R09

复杂度（求单个 `C_n`）：

- DP：时间 `O(n^2)`，空间 `O(n)`；
- 闭式（`math.comb`）：取决于大整数乘除实现，实践中通常快于 DP；
- 乘法递推：时间 `O(n)` 次大整数运算，空间 `O(1)`；
- SciPy 精确组合数：时间复杂度受其内部实现影响，外部调用接口为常数个步骤。

## R10

正确性校验策略：

1. 对 `n=0..25` 全量生成记录；
2. 逐项检查 `dp == binomial == multiplicative == scipy_exact`；
3. 若任一失败，直接抛出 `RuntimeError` 并给出失败 `n` 列表；
4. 全通过后输出表格。

该策略比“只信任某一个实现”更稳健。

## R11

边界情况处理：

- 对所有公开函数统一约束 `n >= 0`，否则抛 `ValueError`；
- `n=0` 返回 `1`；
- `digits` 列使用 `floor(log10(C_n))+1`，对于 `C_0=1` 结果为 `1`。

## R12

运行方式：

```bash
cd Algorithms/数学-组合数学-0551-Catalan数计算
uv run python demo.py
```

脚本不会等待用户输入，执行后即打印结果。

## R13

输出解释：

- `Catalan`：主值（使用 DP 列展示）；
- `digits`：十进制位数，反映增长速度；
- `dp==binomial`、`dp==multiplicative`、`dp==scipy`：一致性校验标记。

理想情况下这三列应全为 `True`。

## R14

MVP 的范围与取舍：

- 优先保证“数学正确 + 可执行 + 可校验”；
- 不追求超大规模基准测试；
- 不引入复杂 CLI、文件读写或可视化框架。

这符合当前 Stage0 条目“最小可运行实现”的目标。

## R15

常见错误与规避：

- 错把 Catalan 当作浮点计算，导致大 `n` 舍入误差；
- 忘记闭式中的除以 `n+1`；
- 递推索引写错成 `C_{n-i}`（应为 `C_{n-1-i}`）；
- 仅做单算法输出，不做交叉校验。

本实现通过“全整数 + 四路比对”规避上述问题。

## R16

可扩展方向：

1. 增加模数版本（如 `C_n mod p`）用于竞赛场景；
2. 增加生成函数法与 FFT 卷积法用于批量大规模计算；
3. 增加性能基准（不同方法在 `n` 规模上的耗时曲线）；
4. 扩展到 Narayana 数等 Catalan 相关细分计数。

## R17

与实际应用连接：

- 编译原理：合法括号结构计数；
- 数据结构：二叉搜索树结构数；
- 几何算法：多边形三角剖分计数；
- 路径计数：Dyck path 与 ballot 类问题。

因此 Catalan 数计算是组合计数基础能力之一。

## R18

`demo.py` 的源码级算法流程（8 步）如下：

1. `main` 固定 `max_n=25`，调用 `build_records` 生成 `n=0..25` 的结果集合。  
2. `build_records` 对每个 `n` 分别调用 `catalan_dp`、`catalan_binomial`、`catalan_multiplicative`、`catalan_scipy_exact`。  
3. `catalan_dp` 按定义递推：外层遍历 `k=1..n`，内层累加 `values[i]*values[k-1-i]`，得到 `C_k`。  
4. `catalan_binomial` 调用 `math.comb(2n,n)` 得到精确二项式，再整除 `n+1` 得到 `C_n`。  
5. `catalan_multiplicative` 从 `C_0=1` 出发，循环执行 `C_{k+1}=C_k*2*(2k+1)//(k+2)`，逐步推进到目标 `n`。  
6. `catalan_scipy_exact` 调用 `scipy.special.comb(2n,n, exact=True)` 走 SciPy 的精确整数分支，再整除 `n+1`。  
7. 主流程检查每条记录的 `all_equal`；若有不一致，立即抛错并列出失败下标，阻止错误结果继续传播。  
8. 全部一致后，`records_to_dataframe` 计算位数列并输出表格，形成可读且可审计的最终结果。  
