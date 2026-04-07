# Buchberger算法

- UID: `MATH-0082`
- 学科: `数学`
- 分类: `计算代数`
- 源序号: `82`
- 目标目录: `Algorithms/数学-计算代数-0082-Buchberger算法`

## R01

Buchberger 算法用于把一组多项式生成元转换为该理想在指定单项式序下的 Groebner 基。

对理想 `I=<f1,...,fm> ⊂ K[x1,...,xn]`，算法输出 `G={g1,...,gt}`，满足：

- `ideal(G)=I`（生成同一个理想）
- 对任意 `h in I`，`LM(h)` 可被某个 `LM(gi)` 整除（`LM` 为首项单项式）
- 等价地，使用 `G` 做多项式除法得到的余项具有唯一正规形

本目录 MVP 采用有理数域 `Q` 上的精确算术（`fractions.Fraction`），实现了经典 Buchberger 主循环，不调用 CAS 的黑箱 Groebner 接口。

## R02

有限计算版问题定义：

- 输入：
  - 变量集合（本实现演示为 `x,y`）
  - 多项式生成元列表 `F=[f1,...,fm]`
  - 单项式序（本实现使用 `lex`，即字典序，`x > y > ...`）
- 输出：
  - `F` 的一个 Groebner 基 `G`
  - 可选的 reduced Groebner 基（通过后处理得到）
- 判定标准：
  - Buchberger 判据：任意 `S(gi,gj)` 对 `G` 约简后余项为 0

## R03

背景与意义：

- Buchberger 在 1965 年提出该算法，它是计算代数中处理多项式理想的核心方法。
- Groebner 基可视为“多元多项式系统的消元标准形”，在代数几何、机器人运动学、自动定理证明、密码分析等场景都常见。
- 从工程角度看，Buchberger 是最易于手写和验证的 Groebner 基基线实现，便于教学和最小可行原型开发。

## R04

算法依赖的核心代数对象：

1. 单项式表示：`x1^a1 ... xn^an`，本实现编码为指数元组 `(a1,...,an)`。
2. 首项：
   - `LM(f)`：按单项式序取最大单项式。
   - `LC(f)`：`LM(f)` 对应系数。
3. S-多项式：
   - `S(f,g) = lcm(LM(f),LM(g))/LT(f) * f - lcm(LM(f),LM(g))/LT(g) * g`
4. 多项式约简（division）：持续用基中可整除当前首项的多项式做消元，直到不可约。
5. Buchberger 判据：`G` 是 Groebner 基当且仅当所有 `S(gi,gj)` 约简余项都为 0。

## R05

`demo.py` 的数据结构设计：

- `Monomial = tuple[int,...]`：单项式指数向量。
- `Polynomial = dict[Monomial, Fraction]`：稀疏多项式。
- `Case`：演示案例（名称、变量名、生成元、期望基大小）。
- `BuchbergerStats`：运行统计（处理 pair 数、新增多项式数、约简步数）。

该结构使每一步代数操作（`LM/LC/S/remainder`）都可显式追踪。

## R06

示例（`curve_intersection_case`）：

- 输入生成元：
  - `f1 = x*y - 1`
  - `f2 = y^2 - x`
- 在 `lex(x>y)` 下运行 Buchberger 后得到 reduced 基：
  - `G1 = x - y^2`
  - `G2 = y^3 - 1`

这体现了消元效果：从原联立系统中得到仅含 `y` 的约束 `y^3-1=0`，再回代可得 `x`。

## R07

时间复杂度（理论与实践）：

- 单步多项式运算复杂度取决于项数 `T`，约简一次常见为 `O(T^2)` 量级（稀疏实现、排序策略相关）。
- Buchberger 需要处理成对 `S` 多项式，pair 数可增长到 `O(|G|^2)`。
- Groebner 基计算已知可能出现双指数级最坏复杂度（相对变量数/次数）。

因此本算法适合作为小规模系统的精确基线，而非大规模工业求解器替代品。

## R08

空间复杂度：

- 主要由中间多项式项数和基大小决定。
- 以稀疏字典表示时，空间可近似看作 `O(sum_i terms(gi))`。
- 在复杂实例中，中间项爆炸（expression swell）会显著增加内存占用。

## R09

正确性要点（对应本实现）：

1. 每轮从 pair `(gi,gj)` 构造 `S(gi,gj)`。
2. 对当前基 `G` 求余 `h = NF_G(S)`（正规形）。
3. 若 `h != 0`，将 `h`（首一化后）加入基，扩大首项理想。
4. 迭代至所有 pair 处理完且不再产生新余项。
5. 最终满足 Buchberger 判据，因此得到 Groebner 基。

脚本中还额外做了独立验证：对最终基再检查所有 `S`-余项均为零。

## R10

边界与异常处理：

- 零多项式会在预处理被过滤，不参与生成元集合。
- 若输入全为零多项式，`buchberger` 抛出 `ValueError`。
- `poly_from_terms` 会检查单项式维度是否等于变量数，防止畸形输入。
- 所有运算使用有理数，避免浮点误差导致的“伪非零余项”。

## R11

运行方式：

```bash
cd Algorithms/数学-计算代数-0082-Buchberger算法
python3 demo.py
```

特性：

- 无交互输入
- 固定案例可复现
- 自动断言验证（若失败会抛异常）

## R12

输出字段解读：

- `raw Buchberger basis`：主循环直接产出的基（通常非最简）。
- `reduced Groebner basis`：经后处理后更紧凑、可读性更高的基。
- `Buchberger criterion passed`：是否通过 `S`-多项式零余项验证。
- `generator fi reduced-to-zero`：原生成元是否可被最终基约简到 0。
- `stats`：
  - `processed pairs`：处理过的 pair 数
  - `added polynomials`：新增基元素数量
  - `reduction steps`：约简主循环步数

## R13

最小测试集（`demo.py` 已覆盖）：

1. `curve_intersection_case`
   - 生成元：`x*y-1`, `y^2-x`
   - 期望 reduced 基规模：2
2. `line_circle_case`
   - 生成元：`x^2+y^2-1`, `x-y`
   - 期望 reduced 基规模：2

每个案例都检查：

- Buchberger 判据通过；
- 原始生成元对最终基约简余项为 0；
- reduced 基大小符合预期。

## R14

可调参数与扩展方向：

- 单项式序：可从 `lex` 扩展到 `grlex` / `grevlex`。
- 变量维度：当前实现对 `n` 变量通用，示例为二维。
- Pair 选择策略：可替换 FIFO 为启发式策略。
- Buchberger Criteria（乘积判据、链判据）可加入以减少无效 `S`-pair。
- 系数域可从 `Q` 扩展到有限域 `GF(p)`（需模运算实现）。

## R15

与“直接调用 CAS 接口”对比：

- 本实现优势：
  - 算法路径透明，适合教学与审计；
  - 精确有理算术，结果可解释；
  - 不依赖第三方黑箱求解器。
- 本实现劣势：
  - 缺少高级优化（判据、矩阵法、F4/F5）；
  - 规模上限远低于成熟 CAS（Singular/Magma/SymPy 内核等）。

因此它是“诚实 MVP”，定位为可运行、可验证、可扩展的最小基线。

## R16

典型应用场景：

- 多项式方程组消元与变量消去；
- 代数几何中理想成员判定/零点结构分析；
- 机器人逆运动学中的符号约束求解；
- 计算代数课程中算法演示与作业验证。

## R17

`demo.py` 关键函数映射：

- 基础代数：
  - `leading_monomial`, `leading_term`, `monomial_lcm`
  - `multiply_by_term`, `poly_sub`, `normalize_poly`
- Buchberger 核心：
  - `s_polynomial`
  - `remainder_by_basis`
  - `buchberger`
- 结果整理与验证：
  - `reduce_groebner_basis`
  - `verify_groebner_basis`
  - `run_case` / `main`

整体代码只依赖 Python 标准库（`dataclasses`, `fractions`, `itertools`, `typing`）。

## R18

源码级算法流程（对应 `demo.py`，8 步）：

1. `main` 构造固定测试案例，并逐个调用 `run_case`。  
2. `run_case` 把输入生成元传入 `buchberger`，启动 Groebner 基主循环。  
3. `buchberger` 初始化 pair 列表，对每个 pair 调用 `s_polynomial` 计算 `S(gi,gj)`。  
4. 对每个 `S`，`remainder_by_basis` 执行多元除法：若某个 `LM(g)` 可整除当前 `LM(p)`，则做首项消元；否则把首项移入余项。  
5. 若余项 `h` 非零，则首一化后加入基，并把与新基元素相关的新 pair 追加到待处理列表。  
6. 循环直到 pair 耗尽，得到一个原始 Groebner 基；随后 `reduce_groebner_basis` 对每个基元素做“用其余元素再约简”得到更简洁基。  
7. `verify_groebner_basis` 对最终基的所有 pair 再做一次 `S`-余项检查，确保满足 Buchberger 判据。  
8. `run_case` 输出基、验证结果和统计信息，并通过断言保证脚本可用于自动化校验。  

这一路径是对 Buchberger 算法的显式实现，而不是对第三方 `groebner(...)` 的封装调用。
