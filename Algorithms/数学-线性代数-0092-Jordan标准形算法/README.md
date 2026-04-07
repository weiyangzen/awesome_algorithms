# Jordan标准形算法

- UID: `MATH-0092`
- 学科: `数学`
- 分类: `线性代数`
- 源序号: `92`
- 目标目录: `Algorithms/数学-线性代数-0092-Jordan标准形算法`

## R01

Jordan 标准形（Jordan Canonical Form, JCF）用于给出复/代数闭域上线性算子的相似分类。

对矩阵 `A`，若特征多项式可完全分裂，则存在可逆矩阵 `P` 使得：

`P^{-1} A P = J`

其中 `J` 是若干 Jordan 块的块对角矩阵，每个 Jordan 块形如：

`J_k(λ) = λI + N`，`N` 在超对角线上为 1，其余为 0。

## R02

本条目 MVP 目标：

1. 构造一个非平凡相似样例矩阵 `A`（不是直接把 Jordan 形当输入）。
2. 从 `A` 精确恢复每个特征值对应的 Jordan 块尺寸。
3. 组装恢复得到的 Jordan 标准形矩阵 `J`。
4. 自动完成一致性校验并输出可读报告。

## R03

`demo.py` 的输入输出约定：

- 输入：脚本内部固定构造的 `6x6` 有理矩阵 `A`。
- 输出：
  - `p_A(x)`（特征多项式）；
  - 特征值及代数重数；
  - 每个特征值对应的 Jordan 块尺寸列表；
  - 恢复得到的 Jordan 矩阵 `J`；
  - `All checks passed.` 校验通过标记。

脚本无交互输入，`python3 demo.py` 直接运行。

## R04

本实现采用的核心判据是“核维数增长法”（nullity growth）：

记 `N_λ = A - λI`，则对 `k >= 1`：

`d_k(λ) = dim ker(N_λ^k) - dim ker(N_λ^{k-1})`

其中 `d_k(λ)` 等于“大小至少为 `k` 的 Jordan 块个数”。

进一步有：

`e_k(λ) = d_k(λ) - d_{k+1}(λ)`

其中 `e_k(λ)` 即“大小恰为 `k` 的 Jordan 块个数”。

## R05

该关系直接连接“线性代数可计算量”与“Jordan 结构”：

- 可计算量：`rank/nullity`（高斯消元可得）；
- 结构量：Jordan 块大小分布；
- 结论：只要已知特征值和代数重数，就可由 `ker(N_λ^k)` 序列恢复块尺寸。

这也是本 MVP 的主算法路线。

## R06

整体算法流程（与 `demo.py` 一致）：

1. 用 Faddeev-LeVerrier 公式精确计算特征多项式；
2. 用有理根定理分解特征多项式，得到特征值与代数重数；
3. 对每个 `λ` 计算 `N_λ^k` 的核维数序列；
4. 由差分恢复 Jordan 块尺寸；
5. 按恢复结果拼成块对角 Jordan 矩阵。

## R07

示例矩阵设计为“已知 Jordan 形 + 相似变换”：

- 目标 Jordan 块：
  - `λ=-1`：一个 `2x2` 块；
  - `λ=2`：一个 `3x3` 块和一个 `1x1` 块。
- 先构造 `J_true`；
- 再取上三角幺模矩阵 `P`，令 `A = P J_true P^{-1}`。

这样可确保输入不是标准形本身，能验证算法确实“从结构恢复”，不是“读答案”。

## R08

复杂度（`n x n` 矩阵）：

- 特征多项式（Faddeev-LeVerrier）：约 `O(n^4)`（朴素矩阵乘法下）。
- 每个特征值的核维数链：需要多次矩阵幂和秩计算，整体约 `O(n^4)` 到 `O(n^5)` 的教学级成本。
- 本条目样例 `n=6`，运行耗时很小。

## R09

精度策略：

- 全程使用 `fractions.Fraction` 做有理数精确运算；
- 避免浮点误差影响秩、核维数和根判定；
- `numpy` 仅用于非核心检查（例如 Frobenius 范数判断“样例非平凡”）。

## R10

技术栈：

- Python 3
- 标准库：`fractions`, `collections`, `math`, `typing`
- 第三方：`numpy`（辅助展示与轻量数值检查）

说明：核心 Jordan 恢复流程没有依赖黑盒 `jordan_form` API。

## R11

运行方式：

```bash
cd Algorithms/数学-线性代数-0092-Jordan标准形算法
python3 demo.py
```

## R12

输出字段解读：

- `Characteristic polynomial p_A(x)`：矩阵 `A` 的特征多项式；
- `Recovered eigenvalue multiplicities`：特征值与代数重数；
- `Recovered Jordan block sizes per eigenvalue`：每个特征值下 Jordan 块尺寸；
- `Recovered Jordan matrix J`：拼装后的 Jordan 标准形矩阵。

## R13

正确性校验（`run_checks`）包含：

1. 恢复块尺寸与构造样例真值一致；
2. `A` 与恢复 `J` 的特征多项式一致；
3. 对每个特征值，`dim ker((A-λI)^k)` 与 `dim ker((J-λI)^k)` 的序列一致；
4. 输入矩阵 `A` 与 `J_true` 明确不相等（非平凡相似样例）。

任一失败会抛异常，不会静默通过。

## R14

局限性：

- 当前实现假设特征多项式可在 `Q` 上完全分裂；
- 有理根分解路径不覆盖不可约高次因子与复根情形；
- 未实现通用 Jordan 基向量构造（即未显式给出相似变换矩阵 `P_est`）；
- 复杂度适合教学与小规模验证，不适合大规模工业计算。

## R15

可扩展方向：

1. 增加多项式因式分解后端（如 `sympy`）以覆盖复特征值；
2. 实现 generalized eigenvector 链构造，输出 `P^{-1}AP=J` 的 `P`；
3. 引入更高效的秩与矩阵幂算法；
4. 增加随机回归测试和批量样例。

## R16

与相关分解的关系：

- 与 Schur 分解相比：Schur 更数值稳定，Jordan 更强调代数结构；
- 与有理标准形相比：有理标准形不要求特征多项式分裂，适用域更广；
- Jordan 形在“重根 + 不可对角化”结构分析上最直观。

## R17

最小测试建议：

1. 纯对角可对角化样例（所有块均 `1x1`）；
2. 单特征值大 Jordan 块样例（如一个 `k x k` 块）；
3. 多特征值混合块样例（本条目即此类）；
4. 人工扰动相似变换矩阵 `P`，验证恢复结果不变。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `build_demo_input()` 先构造目标 Jordan 矩阵 `J_true`，再用可逆矩阵 `P` 生成 `A=PJ_trueP^{-1}`。  
2. `characteristic_polynomial_coeffs_q()` 用 Faddeev-LeVerrier 公式计算 `p_A(x)` 的精确有理系数。  
3. `factor_over_rationals_from_charpoly()` 基于有理根定理逐次试根并线性除法，得到特征值多重集。  
4. 用 `Counter` 汇总代数重数，形成 `λ -> multiplicity` 映射。  
5. 对每个特征值 `λ`，在 `jordan_block_sizes_from_nullity_profile()` 中计算 `nullity((A-λI)^k)` 序列。  
6. 对核维数序列做差分：先得“块大小至少为 k 的个数”，再得“块大小恰为 k 的个数”。  
7. `jordan_block()` + `block_diag_q()` 将恢复出的块尺寸重建为 Jordan 标准形矩阵 `J`。  
8. `run_checks()` 校验块结构、特征多项式和核维数链一致性，最终输出 `All checks passed.`。  

说明：本 MVP 没有把第三方“Jordan 形一键函数”当黑箱，核心推导与计算路径都在源码中显式实现。
