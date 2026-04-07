# F4/F5算法

- UID: `MATH-0083`
- 学科: `数学`
- 分类: `计算代数`
- 源序号: `83`
- 目标目录: `Algorithms/数学-计算代数-0083-F4／F5算法`

## R01

F4/F5 算法是 Groebner 基计算中的两代关键优化思想：

- F4（Faugere 1999）：把多个 `S`-多项式放入同一批次，用矩阵行消元统一做大量约简。
- F5（Faugere 2002）：引入签名（signature）机制，尽量避免无效约简和冗余 `S`-pair。

本目录给出的是一个可运行、可审计的 MVP：

- 保留 F4 的“批处理 + 线性代数行消元”主干；
- 保留 F5 的“签名键去重”思想（采用保守版本，主要避免重复 pair 处理）；
- 全程使用有理数精确算术，不依赖 CAS 黑箱接口。

## R02

有限计算版问题定义：

- 输入：
  - 多项式生成元 `F=[f1,...,fm]`，定义在 `Q[x1,...,xn]`
  - 单项式序（本实现固定为 `lex`，示例里 `x > y`）
- 输出：
  - 一个 Groebner 基 `G`
  - 一个后处理得到的 reduced Groebner 基
- 验证条件：
  - Buchberger 判据：任意 `S(gi,gj)` 对最终基约简余项都为 0
  - 原始生成元都能被最终基约简到 0

## R03

背景与意义：

- 经典 Buchberger 算法易懂但在规模稍大时会慢，F4/F5 是工业级求解器常用的核心优化路线。
- F4 的价值是把符号代数问题部分转成线性代数批处理；F5 的价值是从“需要算什么”层面减少无意义计算。
- 在教学/工程最小原型场景中，F4/F5 的“思想级实现”比直接调用 `groebner(...)` 更利于调试、审计与二次开发。

## R04

算法依赖的核心对象：

1. 稀疏多项式表示：`Polynomial = dict[Monomial, Fraction]`。
2. 首项信息：`LM/LC/LT`（首单项式、首系数、首项）。
3. `S`-多项式：由两基元素首项最小公倍单项式构造。
4. 多项式除法余项：`remainder_by_basis`。
5. F4 批处理矩阵：将同批 `S`-多项式映射为共享单项式列空间上的系数矩阵并做 RREF。
6. F5-lite 签名键：对 pair 建立保守签名键，避免重复 pair 路径。

## R05

`demo.py` 中的数据结构：

- `Monomial = tuple[int,...]`：指数向量。
- `Polynomial = dict[Monomial, Fraction]`：稀疏多项式。
- `Case`：测试案例（名称、变量名、生成元、可选期望基大小）。
- `F4F5Stats`：统计信息，包含：
  - `processed_pairs`
  - `signature_skips`
  - `batches`
  - `row_reduction_calls`
  - `matrix_rows_total`
  - `matrix_cols_peak`
  - `added_polynomials`
  - `remainder_steps`

## R06

示例（`curve_intersection_case`）输入：

- `f1 = x*y - 1`
- `f2 = y^2 - x`

在 `lex(x>y)` 下，reduced 基可化为：

- `x - y^2`
- `y^3 - 1`

这体现了消元效果：先得到只含 `y` 的方程，再回代得到 `x`。

## R07

时间复杂度（定性）：

- 最坏情况下 Groebner 基计算仍可能出现双指数级复杂度（相对变量数/次数）。
- 本实现里每轮包含：
  - pair 选择与 `S`-构造；
  - F4 批量矩阵消元（高斯消元约 `O(r*c*min(r,c))` 量级）；
  - 对候选多项式做除法约简。
- 相比逐个 `S`-多项式串行约简，批量线性代数通常能减少重复消元工作，但在小样例上收益有限。

## R08

空间复杂度：

- 主体来自基集合与中间多项式项数（稀疏字典存储）。
- F4 批处理中还需要矩阵空间 `O(rows * cols)`。
- 当中间单项式数增长时，`matrix_cols_peak` 会明显上升，内存压力主要在这部分。

## R09

正确性要点（对应本实现）：

1. 主循环始终围绕 `S`-pair 推进，不改变 Groebner 理论框架。
2. F4 行消元只作为同批 `S`-多项式的线性预处理，不直接替代最终余项判定。
3. 每个候选结果都通过 `remainder_by_basis` 求正规形，只有非零正规形才并入基。
4. 最终使用 `verify_groebner_basis` 对全部 pair 做独立 Buchberger 判据检查。
5. 额外检查原始生成元是否都能约到 0，确认 `ideal(F) ⊆ ideal(G)` 且基构造一致。

## R10

边界与异常处理：

- 零多项式输入会被过滤；若全为零则抛 `ValueError`。
- `poly_from_terms` 会校验单项式维度与变量维度一致。
- 所有系数使用 `fractions.Fraction`，避免浮点误差造成伪非零余项。
- 通过 `canonical_poly_key` 去重，减少重复基元素导致的循环膨胀。

## R11

运行方式：

```bash
cd Algorithms/数学-计算代数-0083-F4／F5算法
python3 demo.py
```

脚本特性：

- 无交互输入；
- 固定案例，结果可复现；
- 内置断言，验证失败会直接抛异常。

## R12

输出字段解读：

- `raw F4/F5-like basis`：主循环直接产出的基（通常不最简）。
- `reduced Groebner basis`：后处理后的更简形式。
- `Buchberger criterion passed`：最终基是否通过判据。
- `generator fi reduced-to-zero`：原生成元是否属于最终基生成理想。
- `stats`：
  - `processed pairs`：处理 pair 总数
  - `signature skips`：签名键命中的跳过次数
  - `batches`：最小 lcm 次数批处理轮数
  - `row-reduction calls`：矩阵行消元调用次数
  - `matrix rows total / cols peak`：线性代数规模观测
  - `added polynomials`：新增基元素数
  - `remainder steps`：余项约简步数

## R13

最小测试集（`demo.py` 已内置）：

1. `curve_intersection_case`
   - `f1=x*y-1, f2=y^2-x`
   - 期望 reduced 基规模为 2
2. `line_circle_case`
   - `f1=x^2+y^2-1, f2=x-y`
   - 期望 reduced 基规模为 2
3. `three_generators_case`
   - `f1=x*y-1, f2=y^2-x, f3=x^2-y`
   - 用于触发更丰富的 pair/batch 过程

每个案例都做判据验证与生成元回代验证。

## R14

可调参数与扩展方向：

- 单项式序：可扩展到 `grlex` / `grevlex`。
- F4 预处理：可加入更完整的 symbolic preprocessing（当前是轻量实现）。
- F5 判据：可从“保守签名去重”升级为更完整的 rewritable / syzygy criterion。
- 系数域：可扩展到有限域 `GF(p)`。
- 线性代数内核：可替换为更高效稀疏矩阵实现。

## R15

与“直接调用第三方 Groebner 接口”对比：

- 本实现优势：
  - 算法路径透明，可逐函数审计；
  - 结果由精确有理算术得到；
  - 便于教学和做定制化改造。
- 本实现限制：
  - 不是完整工业版 F4/F5；
  - 未实现高级判据与高性能稀疏线代优化；
  - 大规模实例性能无法与专业 CAS 对齐。

定位是“可运行、可验证、可扩展”的诚实 MVP。

## R16

典型应用场景：

- 计算代数教学中的 F4/F5 思想演示；
- 小规模多项式系统的符号消元原型；
- 新判据或新 pair 选择策略的实验基线；
- 对 Groebner 求解流程做可解释性审计。

## R17

`demo.py` 关键函数映射：

- 基础多项式代数：
  - `normalize_poly`, `leading_monomial`, `monomial_lcm`, `s_polynomial`
  - `remainder_by_basis`, `make_monic`
- F4 相关：
  - `build_matrix`, `row_reduce_rref`, `f4_batch_row_reduce`
- F5-lite 相关：
  - `pair_signature_key`
- 主循环与验证：
  - `split_min_degree_pairs`, `f4_f5_mvp`
  - `reduce_groebner_basis`, `verify_groebner_basis`
  - `run_case`, `main`

依赖仅为 Python 标准库（`dataclasses`, `fractions`, `itertools`, `typing`）。

## R18

源码级算法流程（对应 `demo.py`，8 步）：

1. `main` 构造固定案例并逐个进入 `run_case`。  
2. `run_case` 调用 `f4_f5_mvp`，以输入生成元初始化基与 pair 集合。  
3. `f4_f5_mvp` 通过 `split_min_degree_pairs` 选出同一最小 lcm 次数的 pair 批次。  
4. 对批次中的每个 pair，先用 `pair_signature_key` 做保守签名去重，再构造 `S`-多项式。  
5. 将同批 `S`-多项式送入 `f4_batch_row_reduce`：先 `build_matrix`，再 `row_reduce_rref` 做矩阵行消元，得到批量线代预约简结果。  
6. 对“矩阵预约简结果 + 原始 `S`-多项式”统一调用 `remainder_by_basis`，得到正规形候选；非零候选首一化后加入基，并扩展新 pair。  
7. 当 pair 集合耗尽后，`reduce_groebner_basis` 做后处理，再由 `verify_groebner_basis` 检查所有 `S`-余项是否为 0。  
8. `run_case` 打印基与统计信息，并断言“判据通过 + 原生成元可约到零”，保证脚本可用于自动化验证。  

这条路径明确展示了 F4/F5 的关键思想如何落地为可执行代码，而不是把求解完全交给黑箱接口。
