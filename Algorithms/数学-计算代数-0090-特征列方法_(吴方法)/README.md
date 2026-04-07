# 特征列方法 (吴方法)

- UID: `MATH-0090`
- 学科: `数学`
- 分类: `计算代数`
- 源序号: `90`
- 目标目录: `Algorithms/数学-计算代数-0090-特征列方法_(吴方法)`

## R01

特征列方法（吴方法）用于把多项式方程组转化为“按主变量分层”的三角结构（characteristic set），从而把原问题拆成逐级消元的问题。

本目录 MVP 的目标是实现一个可运行、可审计的最小版本：

- 稀疏多元多项式表示（有理数域 `Q`）
- 按变量类（class）选择枢轴多项式
- 用伪余式（pseudo-remainder）做消元
- 构造 Wu 风格特征列并检测一致/不一致

## R02

有限计算版问题定义：

- 输入：多元多项式集合 `F = {f1,...,fm}`，以及变量次序（本实现固定演示 `z < y < x`）
- 输出：
  - `chain`：特征列（按 class 递增的三角多项式链）
  - `residuals`：用 `chain` 约化后仍未消去的剩余方程
  - 运行统计（枢轴个数、伪余式调用次数、伪余式迭代步数）
- 判定：
  - 若全部输入生成元都可约化到 0，则该链对原系统是完整解释之一
  - 若出现非零常数余项（如 `1`），则系统不一致（无解）

## R03

背景与意义：

- 吴方法是中国计算机代数领域的重要里程碑之一，核心思想是把方程系统组织成“可逐级处理”的特征列。
- 与 Groebner 基相比，吴方法强调 pseudo-division 与三角结构，在自动几何定理证明和符号消元中具有代表性。
- 工程上，本 MVP 适合作为教学和算法透明化基线：每一步代数变换都在源码里可追踪。

## R04

本实现使用的核心概念：

1. `class(f)`：多项式 `f` 中出现的最高变量下标。
2. `deg_{x_k}(f)`：`f` 关于变量 `x_k` 的次数。
3. 主变量（leading variable）：`class(f)` 对应变量。
4. 伪余式：
   \[
   R \leftarrow lc(B)\cdot R - lc(R)\cdot x_k^{d_R-d_B}\cdot B
   \]
   反复执行至 `deg_{x_k}(R) < deg_{x_k}(B)`。
5. 特征列：按 class 严格递增的一组多项式链，形成三角结构。

## R05

`demo.py` 的数据结构：

- `Monomial = tuple[int, ...]`：指数向量（与变量顺序一一对应）
- `Polynomial = dict[Monomial, Fraction]`：稀疏多项式
- `Case`：固定演示用例
- `WuStats`：运行计数器
- `CharacteristicSetResult`：`chain + residuals + stats`

实现只依赖 Python 标准库：`dataclasses`, `fractions`, `math`, `typing`。

## R06

主示例（`consistent_triangularizable_system`）：

- 变量顺序：`z < y < x`
- 输入生成元：
  - `x^3 - x*y`
  - `x^2 - y`
  - `y^2 - z`
  - `z - 1`
- 算法输出特征列：
  - `C1 = z - 1`
  - `C2 = y^2 - 1`
  - `C3 = x^2 - y`

该链满足 class 递增 `0 < 1 < 2`，且原生成元都可被该链伪约化到 0。

## R07

时间复杂度（MVP 量级分析）：

- 伪余式单次迭代包含若干多项式乘减，设当前项数规模约 `T`，一次迭代常见 `O(T^2)`。
- 每次伪余式最多执行 `deg_{x_k}(A)-deg_{x_k}(B)+1` 次有效降阶迭代。
- 总复杂度受三个因素共同影响：
  - 多项式项数膨胀（expression swell）
  - 各变量次数差
  - 枢轴选择后触发的约化次数

因此该实现适合小规模教学/验证，不定位为大规模工业求解器。

## R08

空间复杂度：

- 核心开销来自稀疏多项式字典（`chain`、`pool`、中间伪余式）。
- 可粗略记为 `O(sum terms(poly_i))`。
- 在复杂系统中，伪除中间项可能增长显著，导致内存快速放大。

## R09

正确性要点（对应本实现）：

1. 每个 class 先用已选低 class 链对候选方程做伪约化，保证层级一致性。
2. 在该 class 里选低秩枢轴（低次数、较稀疏）加入链。
3. 对同 class 可约化方程做伪余式消元，使其在该主变量次数下降。
4. 最终链按 class 严格递增，形成三角结构。
5. 对原始生成元做 `pseudo_reduce_by_chain`：
   - 一致案例应全部得到 0
   - 若得到非零常数，则判定系统不一致

脚本中用断言固定验证这两类行为。

## R10

边界与异常处理：

- 零多项式会被规范化阶段过滤。
- 伪余式除数若为零，抛出 `ValueError`。
- 伪余式迭代设 `max_steps` 上限（默认 200），超过后抛 `RuntimeError`，防止异常输入导致长循环。
- 多项式在每步都做规范化（去零项、去内容因子、符号标准化），保证可比性与输出稳定性。

## R11

运行方式（无交互）：

```bash
cd Algorithms/数学-计算代数-0090-特征列方法_(吴方法)
python3 demo.py
```

脚本将自动运行两个固定案例并打印链、余项、判定和统计信息。

## R12

输出字段解读：

- `characteristic chain`：构造出的特征列。
- `class=k, lv=...`：链元素的变量层级与主变量。
- `residual equations after chain-reduction`：链约化后的剩余约束。
- `triangular_chain_ok`：是否满足 class 严格递增。
- `all_generators_reduce_to_zero`：原输入是否全约为 0。
- `detected_inconsistency`：是否检测到非零常数余项。
- `stats`：运行计数（`pivots`, `pseudo_remainder_calls`, `pseudo_remainder_steps`）。

## R13

最小测试集（`demo.py` 已内置）：

1. `consistent_triangularizable_system`
   - 期望：三角链成立，全部生成元约化为 0
2. `inconsistent_system`
   - 期望：出现非零常数余项（示例中为 `1`），判定不一致

每个案例都通过断言保证结果与预期一致。

## R14

可调参数与扩展方向：

- 变量顺序：目前演示 `z < y < x`，可改为任意顺序。
- 枢轴策略：可引入更完整 rank（class、次数、项序）和启发式。
- 约化策略：可加入更严格的自约化（autoreduction）循环。
- 系数域：当前是 `Q`（`Fraction`），可扩展到 `GF(p)`。
- 结果形态：可进一步输出分支分解（regular chain / triangular decomposition）。

## R15

与“直接调用黑盒 CAS”对比：

- 本实现优势：
  - 算法路径完全透明；
  - 每一步伪除和选枢轴都可审计；
  - 适合教学与单元测试。
- 本实现局限：
  - 没有工业级优化（判据、重排策略、矩阵化加速）；
  - 大规模系统上效率远低于成熟 CAS。

因此定位是“诚实 MVP”，而非性能求解器。

## R16

典型应用场景：

- 计算代数课程中的特征列法教学演示
- 自动定理证明中的符号消元原型
- 小规模多项式约束系统的一致性检查
- 构建更高级三角分解算法前的基线模块

## R17

`demo.py` 关键函数映射：

- 多项式基础：
  - `normalize_poly`, `canonicalize_poly`, `poly_add`, `poly_sub`, `poly_mul`
  - `poly_class`, `degree_in_var`, `leading_coeff_poly`
- Wu 核心：
  - `pseudo_remainder`
  - `pseudo_reduce_by_chain`
  - `wu_characteristic_set`
- 验证与输出：
  - `is_triangular_chain`, `has_nonzero_constant`
  - `polynomial_to_str`, `run_case`, `main`

## R18

源码级算法流程（对应 `demo.py`，8 步）：

1. `main` 调用 `build_cases`，构造一致/不一致两个固定方程组。  
2. `run_case` 把生成元传入 `wu_characteristic_set`，开始按 class 构造特征列。  
3. `wu_characteristic_set` 先用 `pseudo_reduce_by_chain` 把当前候选方程对已有低 class 链做伪约化。  
4. 在当前 class 候选中，`select_pivot` 选择低秩枢轴（低次数、较少项）。  
5. 对其余可约化方程调用 `pseudo_remainder`：循环执行 `lc(B)*R - lc(R)*x^t*B`，直到该变量次数下降到阈值以下。  
6. 完成所有 class 后，得到 `chain`；再对剩余方程做一次链约化，得到 `residuals`。  
7. `run_case` 对每个原始生成元再次 `pseudo_reduce_by_chain`，检查是否全部约化为 0，并检查是否出现非零常数余项。  
8. 打印链结构、余项和统计信息，并用断言固定验证：一致案例必须全零，不一致案例必须检测到矛盾常数。  

该实现不是对第三方 CAS 的一行封装，而是把 Wu 方法关键步骤（选列、伪除、成链、判定）直接写在源码中。
