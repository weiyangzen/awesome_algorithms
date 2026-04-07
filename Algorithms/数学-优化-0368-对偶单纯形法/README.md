# 对偶单纯形法

- UID: `MATH-0368`
- 学科: `数学`
- 分类: `优化`
- 源序号: `368`
- 目标目录: `Algorithms/数学-优化-0368-对偶单纯形法`

## R01

对偶单纯形法（Dual Simplex Method）是线性规划中与原始单纯形法互补的一类基迭代算法。它从“对偶可行但原始不可行”的基解出发，在保持对偶可行性的同时逐步修复原始可行性，最终达到原始与对偶同时可行，即最优。

本目录 MVP 采用的线性规划形式是：

\[
\min\ c^\top x\quad
\text{s.t. } A x=b,\ x\ge 0
\]

## R02

适用场景：
- 已有一个基满足对偶可行（约化成本满足符号条件），但基本变量中存在负值（原始不可行）。
- 在线性规划求解中，约束右端 `b` 发生小幅变化后，旧最优基可能失去原始可行性，此时对偶单纯形常比“从头跑原始单纯形”更快。
- 作为割平面法、分支定界法等流程中的重优化子程序。

## R03

数学条件（最小化版本）：
- 给定基 `B` 与非基 `N`，基本解 `x_B = B^{-1}b`，非基变量 `x_N=0`。
- 对偶变量（乘子）满足：
  \[
  B^\top \lambda = c_B
  \]
- 非基约化成本：
  \[
  \bar c_N = c_N - A_N^\top\lambda
  \]
- 对偶可行条件：`\bar c_N \ge 0`。
- 原始可行条件：`x_B \ge 0`。

对偶单纯形每步都维持 `\bar c_N \ge 0`，直到 `x_B \ge 0`。

## R04

核心 pivot 思路（最小化版本）：
1. 选离基行：从 `x_B` 中选最负分量所在行 `r`（`x_{B_r}<0`）。
2. 行系数：`\bar A = B^{-1}A_N`，取 `r` 行 `\bar a_{rj}`。
3. 入基候选：仅考虑 `\bar a_{rj} < 0` 的非基变量（这样增大该非基变量才可能提升负的基本变量）。
4. 比值检验：
   \[
   j^* = \arg\min_{\bar a_{rj}<0} \frac{\bar c_j}{-\bar a_{rj}}
   \]
5. 执行基交换（pivot），进入下一轮。

若某次离基行没有任何 `\bar a_{rj}<0`，则判定原问题不可行。

## R05

伪代码：

```text
input: A, b, c, initial basis B_idx
repeat:
    build B, N from basis/nonbasis
    x_B = B^{-1} b
    lambda = solve(B^T lambda = c_B)
    reduced = c_N - A_N^T lambda

    if any reduced < 0:
        error "initial basis is not dual-feasible"

    if all x_B >= 0:
        return optimal solution

    r = argmin(x_B)  # most negative basic variable
    row = (B^{-1} A_N)[r, :]
    candidates = {j | row[j] < 0}
    if candidates empty:
        return infeasible

    j* = argmin reduced[j] / (-row[j]) over candidates
    pivot: replace B_idx[r] with nonbasis[j*]
until max_iter
```

## R06

正确性要点（工程视角）：
- 由于每步检查并维持 `\bar c_N \ge 0`，对偶可行性不被破坏。
- 比值检验保证 pivot 后新约化成本仍满足符号约束。
- 当 `x_B \ge 0` 且 `\bar c_N \ge 0` 同时成立时，满足线性规划 KKT 条件，故达到最优。
- 若负基本变量所在行无法找到 `\bar a_{rj}<0` 候选，则无法通过任何非基变量修复该负值，原问题不可行。

## R07

复杂度分析（稠密实现）：
- 设约束数 `m`，变量数 `n`。
- 每轮主要计算：
  - `B^{-1}` 或等价线性代数操作，约 `O(m^3)`；
  - 约化成本与行系数计算约 `O(m(n-m))`。
- 单轮复杂度近似 `O(m^3 + m(n-m))`，本 MVP 以可读性为主，直接重算逆矩阵。
- 空间复杂度：`O(mn)`（存储 `A`） + `O(m^2)`（临时矩阵）。

## R08

`demo.py` 数据结构：
- `IterationLog`：单轮日志，字段包括 `objective`、最小基本变量、离基/入基变量编号、比值。
- `DualSimplexResult`：最终结果，含状态、最优解向量、目标值、迭代数、最终基、日志。
- `basis`：长度为 `m` 的变量索引列表。
- `nonbasis`：由补集动态生成，保证与 `basis` 不重叠。

## R09

MVP 输入输出约定：
- 输入（在 `main()` 中固定，无需交互）：
  - `A`：`3x5` 约束矩阵；
  - `b`：`[-1, -2, 4]`；
  - `c`：`[1, 2, 0, 0, 0]`；
  - 初始基 `basis=[2,3,4]`（三个松弛变量）。
- 输出：
  - 求解状态（`optimal/infeasible/max_iter_reached`）；
  - 解向量 `x` 与目标值 `c^T x`；
  - pivot 历史表；
  - 末尾自动校验结果。

## R10

边界与异常处理：
- `A/b/c` 形状不匹配、`basis` 非法时抛 `ValueError`。
- 基矩阵奇异时抛 `RuntimeError`。
- 若出现负约化成本（违背对偶可行前提）直接报错。
- 若某离基行无可入基候选，返回 `infeasible`。
- 迭代超限返回 `max_iter_reached`。

## R11

实现策略与取舍：
- 仅用 `numpy`，不调用 `scipy.optimize.linprog` 黑箱。
- 用“重算 `B^{-1}`”换取代码直观性，便于审计 pivot 逻辑。
- 固定一个可验证的小型 LP，强调算法路径可追踪、输出可复现。

## R12

运行方式：

```bash
cd Algorithms/数学-优化-0368-对偶单纯形法
python3 demo.py
```

脚本无需任何命令行参数或交互输入。

## R13

预期输出特征：
- `Status: optimal`。
- `Optimal x` 约为 `[1.0, 0.0, 0.0, 0.0, 3.0]`。
- `Optimal objective` 约为 `1.0`。
- 输出若干轮 pivot 记录（迭代号、离基/入基变量、ratio）。
- 最后打印 `Validation checks passed.`。

## R14

常见实现错误：
- 把最小化与最大化的约化成本符号条件混淆。
- 离基行选取不当（没选负基本变量）导致无法修复原始不可行。
- 比值检验分母符号写反，破坏对偶可行性。
- pivot 后未同步更新基/非基索引，造成维度错配或死循环。

## R15

最小测试清单：
- 功能测试：当前内置 LP 应得到 `x=[1,0,0,0,3]`、`z=1`。
- 可行性测试：检查 `Ax=b` 与 `x>=0`。
- 稳定性测试：减小 `tol` 仍可收敛到同一解。
- 异常测试：故意给出重复 basis 索引，应抛输入错误。

## R16

可扩展方向：
- 用 LU 分解增量更新代替每轮显式逆矩阵，提高效率与稳定性。
- 增加 Bland 规则等防循环策略。
- 支持从不等式自动建模（自动加松弛/剩余变量）。
- 增加多案例 benchmark 与 CSV 日志导出。

## R17

局限与工程权衡：
- 当前实现面向教学和审计，不追求大规模稀疏 LP 性能。
- 未实现抗退化（degeneracy）高级策略，极端案例可能慢。
- 示例问题规模较小，主要用于验证流程正确性而非性能上限。

## R18

`demo.py` 源码级算法流程（8 步，非黑箱）：
1. `main` 构造标准形 LP 的 `A,b,c` 与初始基 `basis=[2,3,4]`，调用 `dual_simplex`。  
2. `dual_simplex` 每轮由 `basis` 生成 `B`，由补集生成 `N`，计算 `x_B=B^{-1}b`。  
3. 解 `B^T\lambda=c_B`，再算约化成本 `\bar c_N=c_N-A_N^T\lambda`，若有负值立即报错（初始对偶不可行）。  
4. 若 `x_B` 全非负，构造完整解向量 `x` 并返回 `optimal`。  
5. 否则选最负基本变量所在行 `r` 作为离基行，并取该行系数 `row=(B^{-1}A_N)[r,:]`。  
6. 在 `row<0` 的候选中做比值检验 `\bar c_j/(-row_j)`，取最小者作为入基变量；无候选则返回 `infeasible`。  
7. 记录本轮日志（目标值、离基/入基变量、ratio），执行基交换 `basis[r]=entering_var`。  
8. 循环直至最优、不可行或超出 `max_iter`；`main` 对返回解做 `Ax=b`、`x>=0` 与目标值断言。
