# 有理标准形算法

- UID: `MATH-0094`
- 学科: `数学`
- 分类: `线性代数`
- 源序号: `94`
- 目标目录: `Algorithms/数学-线性代数-0094-有理标准形算法`

## R01

有理标准形（Rational Canonical Form, RCF）是矩阵在给定域（本条目取 `Q`）上的标准相似分类形式。

对任意 `A in Q^{n x n}`，存在可逆矩阵 `P` 使得：

`P^{-1} A P = diag(C(f1), C(f2), ..., C(ft))`

其中：
- `C(fi)` 是多项式 `fi(x)` 的 companion 矩阵；
- `f1 | f2 | ... | ft`（整除链）；
- 这些 `fi` 就是不变因子（invariant factors）。

## R02

本条目 MVP 的目标：

1. 输入一个有理矩阵 `A`（脚本内置，不需交互）；
2. 计算 `xI - A` 的行列式因子（determinantal divisors）；
3. 由 `Δk/Δk-1` 恢复不变因子；
4. 用不变因子构造 RCF 的 companion 块对角矩阵；
5. 自动做一致性校验并输出报告。

## R03

核心数学关系（本实现直接使用）：

- 令 `M(x) = xI - A`。
- `Δk(x)`：`M(x)` 全部 `k x k` 子式行列式的 gcd（取 monic）。
- 不变因子：`dk(x) = Δk(x) / Δk-1(x)`，`Δ0 = 1`。
- 非单位（非常数）`dk` 按顺序即 `f1, ..., ft`。
- RCF 由 `diag(C(f1), ..., C(ft))` 给出。

## R04

为什么这个路径是“有理标准形算法”的直接版本：

- 理论上，不变因子来自 `xI-A` 在 `Q[x]` 上的 Smith 结构；
- 行列式因子 `Δk` 是 Smith 不变量的等价刻画；
- 因而用 `Δk` 恢复不变因子，再转 companion 块，就是标准可审计流程；
- 该流程比“直接调用黑盒库函数”更透明，适合教学与验证。

## R05

输入输出约定（`demo.py`）：

- 输入：脚本内部构造一个 `5x5` 有理矩阵 `A`。该 `A` 由已知 RCF 通过相似变换生成，保证样例非平凡。
- 输出：
  - 恢复到的不变因子链；
  - 由其构造出的 canonical 矩阵对应不变因子；
  - 整除链检查、特征多项式一致性检查；
  - 最终 `All checks passed.`。

## R06

示例数据设计：

- 设不变因子链：
  - `f1(x) = x^2 + 1`
  - `f2(x) = x^3 + 2x^2 + x + 2 = (x^2+1)(x+2)`
- 先构造 `F = diag(C(f1), C(f2))`；
- 再取上三角幺模矩阵 `P`，令 `A = P F P^{-1}`。

这样 `A` 与 `F` 相似，但通常 `A != F`，能检验算法是否真的恢复结构而非“碰巧输入已标准化”。

## R07

实现中的数值/代数策略：

- 全程使用 `fractions.Fraction`，避免浮点误差污染 gcd 与整除判定；
- 多项式表示为“低到高”系数元组；
- 所有关键对象（多项式、行列式、gcd、除法）都在源码里显式实现；
- `numpy` 仅用于非核心的展示比较（例如确认示例确实非平凡）。

## R08

时间复杂度（本 MVP）：

- 对每个 `k`，需枚举 `C(n,k)^2` 个子式；
- 每个 `k x k` 子式行列式采用排列展开，代价约 `O(k! * poly_mul_cost)`；
- 总体复杂度随 `n` 增长很快，属于教学/小规模验证型实现。

本实现选择 `n=5`，可在普通环境内快速运行。

## R09

空间复杂度：

- 主要开销来自多项式矩阵 `xI-A` 与中间子式行列式；
- 矩阵存储规模 `O(n^2)`，但每个元素是多项式对象；
- 子式计算产生的临时多项式较多，实际内存与 `n`、多项式次数共同增长。

## R10

技术栈：

- Python 3
- 标准库：`fractions`, `itertools`, `dataclasses`, `typing`
- 第三方：`numpy`（仅辅助展示/比对，不承担核心算法）

说明：环境中无 `sympy/scipy`，因此核心代数流程完全手写，保证可运行与可解释。

## R11

运行方式：

```bash
cd Algorithms/数学-线性代数-0094-有理标准形算法
python3 demo.py
```

脚本不接收命令行参数，不会请求交互输入。

## R12

输出字段解读：

- `recovered_invariant_factors`：从 `A` 直接恢复的不变因子链；
- `canonical_invariant_factors`：对构造出的 canonical 矩阵再次计算得到的因子链；
- `divisibility_chain_ok`：是否满足 `f1 | f2 | ...`；
- `same_characteristic_polynomial`：`A` 与 canonical 矩阵特征多项式是否一致；
- `nontrivial_similarity_example`：示例是否确实 `A != F`。

## R13

正确性校验（`run_checks`）：

1. 恢复因子与预设 ground truth 一致；
2. 因子链满足整除关系；
3. canonical 化后再提取因子不变；
4. `det(xI-A)` 与 `det(xI-F)` 一致，且等于因子乘积；
5. 样例不是平凡输入（`A` 不等于 `F`）。

任一失败都会抛出异常，避免静默错误。

## R14

局限与边界：

- 当前实现以“精确代数 + 小规模”优先，不适合大规模矩阵；
- 行列式用排列展开，复杂度高；
- 未实现高性能的 Bareiss/多项式矩阵消元/SNF 完整分解；
- 仅覆盖有理域 `Q` 路径，不处理数值近似域上的稳定性问题。

## R15

可扩展方向：

- 用 Bareiss 或 fraction-free 消元替换排列展开行列式；
- 直接实现 `Q[x]` 上 Smith Normal Form，减少子式枚举成本；
- 引入更高效多项式表示与缓存；
- 增加从任意外部输入矩阵读取与批量验证。

## R16

与相关标准形的关系：

- Jordan 标准形依赖特征值分裂域，RCF 不需要；
- RCF 只依赖域上不变因子，适用性更广；
- 对工程实现，RCF 常用于“代数结构证明/分类”，Jordan 常用于“几何重数与链结构分析”。

## R17

最小测试建议：

1. 用不同可整除因子链（如 `f1|f2|f3`）构造相似样例反复验证；
2. 增加 `n=3/4/5` 小规模随机幺模相似变换测试；
3. 验证 `A` 与恢复 canonical 的 `det(xI-.)` 一致；
4. 验证 canonical 再 canonical 化不变（幂等性测试）。

## R18

`demo.py` 源码级流程（9 步）：

1. `build_demo_input()` 先指定不变因子 `f1,f2`，构造块对角 companion 矩阵 `F`。  
2. 构造幺模上三角 `P`，计算 `A = PFP^{-1}` 形成非平凡输入。  
3. `invariant_factors_from_matrix(A)` 先建立多项式矩阵 `M(x)=xI-A`。  
4. 对 `k=1..n`，`all_k_minors_det()` 枚举所有 `k x k` 子式并用 `poly_det()` 求其多项式行列式。  
5. 对同一 `k` 的全部子式行列式做 `poly_gcd()`，得到 `Δk(x)`。  
6. 通过 `dk = Δk / Δk-1`（`poly_exact_div`）恢复不变因子序列，并过滤单位因子。  
7. `rational_canonical_form_from_invariant_factors()` 对每个非单位因子调用 `companion_matrix()`，再块对角拼接成 `F_rcf`。  
8. `run_checks()` 验证整除链、因子保持、特征多项式一致性与样例非平凡性。  
9. `main()` 打印恢复结果、canonical 矩阵与最终通过标记 `All checks passed.`。  

说明：`numpy` 在本实现中不负责“求有理标准形”，核心代数流程全部在源码中显式实现。
