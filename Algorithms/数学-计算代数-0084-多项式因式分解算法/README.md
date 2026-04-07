# 多项式因式分解算法

- UID: `MATH-0084`
- 学科: `数学`
- 分类: `计算代数`
- 源序号: `84`
- 目标目录: `Algorithms/数学-计算代数-0084-多项式因式分解算法`

## R01

问题：给定一元整系数多项式
\[
f(x)=a_nx^n+a_{n-1}x^{n-1}+\cdots+a_0,\quad a_i\in\mathbb Z
\]
把它分解为有理数域 \(\mathbb Q\) 上的不可约因子乘积。

本目录的 MVP 目标不是实现完整 CAS（计算机代数系统），而是实现一个可运行、可验证、结构清晰的最小版本：
- 提取内容因子（系数最大公因数）
- 用有理根定理寻找线性因子
- 用综合除法逐步降阶
- 对剩余二次项做判别式分解

## R02

核心数学工具：
- 内容因子分离：\(f(x)=c\cdot g(x)\)，其中 \(c=\gcd(a_0,\dots,a_n)\)，\(g\) 为 primitive 多项式。
- 有理根定理：若 \(p/q\)（既约）是 \(g(x)\) 的有理根，则 \(p\mid a_0,\ q\mid a_n\)。
- 因子定理：\(g(r)=0\iff (x-r)\mid g(x)\)。
- 二次多项式判别式：\(ax^2+bx+c\) 在 \(\mathbb Q\) 上可分解当且仅当
  \(\Delta=b^2-4ac\) 为完全平方数。

## R03

输入（代码中使用系数数组）：
- `coeffs=[a_n, a_{n-1}, ..., a_0]`，高次到低次。

输出（`FactorizationResult`）：
- `content`：提取出的整数内容因子
- `linear_factors`：找到的有理根列表 `r`，对应线性因子 `(x-r)`
- `irreducible_remainder`：无法继续按当前策略分解的剩余多项式系数

## R04

算法思想（自顶向下）：
1. 先把整体系数公因子抽出来，避免后续候选根过多。
2. 对当前多项式生成有理根候选集合 \(\pm p/q\)。
3. 逐个代入 Horner 评估，命中根就做一次综合除法并继续。
4. 若没有有理根且当前次数为 2，则检查判别式是否可再分。
5. 仍不可分则停止，作为在 \(\mathbb Q\) 下的剩余不可约部分。

## R05

伪代码：

```text
factor(coeffs):
  poly <- trim_leading_zeros(coeffs)
  content, poly <- extract_content(poly)
  roots <- []

  while degree(poly) > 0:
    if constant_term(poly) == 0:
      roots.append(0)
      poly <- synthetic_divide(poly, 0)
      continue

    candidates <- rational_candidates(poly)
    r <- first candidate with poly_eval(poly, r) == 0

    if r exists:
      roots.append(r)
      poly <- synthetic_divide(poly, r)
      continue

    if degree(poly) == 2 and discriminant is perfect square:
      roots.extend(two roots)
      poly <- [1]
    break

  return (content, roots, poly)
```

## R06

正确性要点：
- `extract_content` 只做整数公因子提取，不改变零点结构。
- 有理根定理保证候选集合完备（对有理根而言）。
- 每次 `poly_eval(r)==0` 后再综合除法，保证降阶操作合法。
- 二次判别式分支只在判别式完全平方时拆成两个有理线性因子。
- 最终输出满足：
  \[
  f(x)=\text{content}\cdot\prod_i (x-r_i)\cdot h(x)
  \]
  其中 `h(x)` 为算法停止时的剩余项。

## R07

时间复杂度（设次数为 \(n\)，常数项和首项约数数目分别为 \(d_0,d_n\)）：
- 单轮候选根数量约 \(2d_0d_n\)
- 每个候选评估用 Horner，复杂度 \(O(n)\)
- 最坏情况下总复杂度可近似写成 \(O(n^2\cdot d_0d_n)\)

空间复杂度：
- 主要是系数与候选集合，\(O(n+d_0d_n)\)

## R08

实现层面的稳定性与精度策略：
- 使用 `fractions.Fraction` 全程有理数计算，避免浮点误差。
- 综合除法和多项式求值都在精确算术下进行。
- 展示字符串时再格式化为 `a/b`，不参与运算。

## R09

边界与异常情况：
- 全零多项式：定义上不可标准分解，代码返回 `content=0, remainder=[0]`。
- 常数多项式：没有线性因子，直接作为结果。
- 常数项为 0：先提取根 `x=0`，反复除到常数项非零。
- 判别式非完全平方的二次项：保留为不可约剩余项（在 \(\mathbb Q\) 上）。

## R10

MVP 代码结构：
- 数据结构：`FactorizationResult`
- 核心函数：
  - `factor_polynomial_over_q`
  - `rational_root_candidates`
  - `synthetic_division`
  - `factor_quadratic_over_q`
- 辅助函数：格式化输出、约数生成、去前导零等

该结构便于后续替换为更强算法（如 Berlekamp/Zassenhaus），同时保留演示入口。

## R11

`demo.py` 展示了 5 个样例：
- 完全线性分解（三次）
- 含首项系数不为 1 的三次
- 四次可完全拆成线性因子
- 三次仅提取一个线性因子后剩不可约二次
- 二次在 \(\mathbb Q\) 上不可约（`x^2+1`）

这样可以覆盖「可分解」「部分可分解」「不可分解」三种常见情形。

## R12

运行方式（无交互）：

```bash
cd /Users/wangweiyang/GitHub/awesome_algorithms/.cron/stage0_exec_repo_slot20
python3 Algorithms/数学-计算代数-0084-多项式因式分解算法/demo.py
```

## R13

一次典型输出（不同 Python 版本仅格式细节可能略有差异）：

```text
MATH-0084 多项式因式分解算法 Demo

[1] f(x) = x^3 - 6*x^2 + 11*x - 6
    因式分解: (x - 1) * (x - 2) * (x - 3)

[5] f(x) = x^2 + 1
    因式分解: (x^2 + 1)
```

## R14

当前 MVP 的能力边界：
- 仅处理一元多项式。
- 主目标域是 \(\mathbb Q\)（有理数域）。
- 不包含有限域因式分解、模提升、LLL 等高级流程。
- 对高次稠密多项式的效率不如专业 CAS。

## R15

可扩展方向：
- 增加平方因子分解（square-free decomposition）。
- 在 \(\mathbb F_p[x]\) 上加入 Berlekamp/Cantor-Zassenhaus。
- 基于 Hensel lifting + Zassenhaus 做整系数高次分解。
- 提供符号验证：把分解结果重新乘回原多项式。

## R16

建议最小测试集：
- `x^3-6x^2+11x-6`（三个整数根）
- `2x^3+3x^2-11x-6`（非 monic）
- `x^4-5x^2+4`（四个整数根）
- `3x^3-12`（部分分解）
- `x^2+1`（在 \(\mathbb Q\) 上不可约）

## R17

与常见方案对比：
- 本实现：
  - 优点：短小、可读、易验证、依赖极少
  - 缺点：覆盖范围有限，复杂度对大规模问题不友好
- CAS 黑盒（如直接调用大库）：
  - 优点：功能强、算法完善
  - 缺点：学习和审计门槛更高，不利于算法教学的「透明路径」

## R18

`demo.py` 的源码级算法流（8 步）：
1. `factor_polynomial_over_q` 接收系数，转为 `Fraction` 并去前导零。
2. `extract_content` 计算整数系数 gcd，得到 `content` 和 primitive 多项式。
3. 若常数项为 0，直接记录根 `0`，调用 `synthetic_division` 降阶。
4. 调 `rational_root_candidates`：根据常数项与首项约数枚举 \(\pm p/q\)。
5. 用 `poly_eval`（Horner）逐个验证候选根是否为真根。
6. 命中根后再次 `synthetic_division`，并把该根加入 `linear_factors`。
7. 若没有有理根且次数为 2，调用 `factor_quadratic_over_q` 用判别式判断是否可拆。
8. 输出 `FactorizationResult`，再由 `format_factorization` 拼接成可读因式分解字符串。
