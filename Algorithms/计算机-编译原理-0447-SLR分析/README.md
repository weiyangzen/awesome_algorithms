# SLR分析

- UID: `CS-0287`
- 学科: `计算机`
- 分类: `编译原理`
- 源序号: `447`
- 目标目录: `Algorithms/计算机-编译原理-0447-SLR分析`

## R01

SLR(1)（Simple LR）是介于 LR(0) 与 LR(1) 之间的自底向上分析方法：
- 状态构造仍使用 LR(0) 项目集（不携带 lookahead）；
- 归约时不再像 LR(0) 那样对所有终结符都归约，而是只在 `FOLLOW(A)` 上对 `A -> α` 进行归约；
- 相比 LR(0) 冲突更少、状态规模更小于 LR(1)。

本条目实现一个可运行 MVP：从零构造规范 LR(0) 项目集族，结合 FIRST/FOLLOW 生成 SLR(1) 分析表，并执行可追踪移进-归约过程。

## R02

问题定义（本实现输入/输出）：
- 输入 1：上下文无关文法（demo 内置表达式文法）；
- 输入 2：若干 token 序列（demo 内置，非交互）；
- 输出：
  - 产生式编号（用于 `rN`）；
  - FIRST/FOLLOW 集合；
  - 规范 LR(0) 状态（项目集）；
  - SLR(1) ACTION/GOTO 表；
  - 每个样例的逐步解析轨迹与接受/拒绝结论；
  - 状态数、ACTION 表填充率、步数统计。

## R03

选择 SLR(1) 的原因：
- 是从 LR(0) 到 LR(1) 的关键过渡，最能体现“FOLLOW 集限制归约”的思想；
- 代码复杂度显著低于 LR(1)，但能力通常强于 LR(0)；
- 非常适合教学场景：可以把冲突成因、信息粒度、状态规模三者的平衡说清楚。

## R04

核心对象定义：
- LR(0) 项：`[A -> α · β]`；
- 状态：一个 LR(0) 项目集；
- ACTION：`shift s`、`reduce r`、`accept`；
- GOTO：`(state, nonterminal) -> state`。

`demo.py` 中的关键数据结构：
- `LR0Item(lhs, rhs, dot)`；
- `action_table[(state, terminal)]`；
- `goto_table[(state, nonterminal)]`；
- `ParseStep/ParseResult`（轨迹与结果）。

## R05

FIRST/FOLLOW 的作用（SLR 核心）：
- FIRST：用于 FOLLOW 的迭代传播；
- FOLLOW：决定 reduce 的落点。

在 `build_slr_tables` 中，若某状态含完成项 `A -> α ·` 且 `A` 不是增广开始符，只在 `FOLLOW(A)` 中的终结符上填写 `reduce A -> α`，而不是像 LR(0) 那样“全终结符归约”。

## R06

LR(0) 状态机构造：
1. `closure_lr0`：若出现 `A -> α · B β`，则加入 `B -> · γ`；
2. `goto_lr0(I, X)`：对点前为 `X` 的项右移并闭包；
3. `canonical_lr0_collection`：从 `S' -> · S` 的闭包出发，遍历符号构造完整项目集族与转移。

SLR 的状态层与 LR(0) 完全一致，区别只在“怎么填归约表项”。

## R07

SLR(1) 造表规则（`build_slr_tables`）：
- 若项 `A -> α · a β`（`a` 为终结符），填 `ACTION[state, a] = shift`；
- 若项 `A -> α · B β`（`B` 为非终结符），填 `GOTO[state, B]`；
- 若项 `A -> α ·`：
  - 若 `A` 为增广开始符，填 `ACTION[state, $] = accept`；
  - 否则仅对 `FOLLOW(A)` 中终结符填 `reduce`。

若同一单元被写入不同动作，立即记录冲突并报错。

## R08

表驱动解析流程（`parse_tokens`）：
1. 状态栈初始化为 `[0]`，符号栈初始化为 `[$]`；
2. 读取当前状态与输入符号查询 ACTION；
3. `shift`：压入输入符号与目标状态；
4. `reduce`：按产生式右部长度弹栈，再用 GOTO 压入左部；
5. `accept`：成功结束；
6. 无动作：失败并给出该状态下可接受终结符。

全过程记录 `ParseStep`，可逐步审计。

## R09

MVP 使用文法（表达式文法）：
- `S -> E`
- `E -> E + T | T`
- `T -> T * F | F`
- `F -> ( E ) | id`

它体现：
- `*` 优先级高于 `+`；
- 左递归带来左结合；
- 是 SLR(1) 经典可处理样例。

## R10

内置样例（含断言）：
- 正例：
  - `id + id * id`
  - `( id + id ) * id`
  - `id * ( id + id )`
  - `id`
- 反例：
  - `id + * id`
  - `( id + id * id`（缺右括号）
  - `id id + id`

脚本会对每例断言 `accepted == expected`。

## R11

实现边界：
- 不含词法分析器，token 直接以字符串数组给定；
- 不做错误恢复（如 panic-mode），表项缺失即失败返回；
- 目标是“可审计 SLR 核心链路”，不是完整编译器前端。

## R12

复杂度说明（`|I|` 为状态数，`|Σ|` 为文法符号总数，`n` 为输入长度）：
- FIRST/FOLLOW 迭代：约 `O(iter * productions * rhs_len)`；
- 项目集构造：约 `O(|I| * |Σ| * C_closure)`；
- 造表：约 `O(total_items + transitions + reduce_fill)`；
- 单次解析：通常近似 `O(n)`（每步移进或归约，栈操作常数级）。

## R13

工程取舍：
- 不调用 parser generator 黑盒，FIRST/FOLLOW、闭包、GOTO、造表、解析均显式实现；
- 使用 `numpy` 仅做统计汇总（填充率、步数），核心算法保持纯 Python；
- 输出“集合 + 状态 + 表 + 轨迹 + 汇总”，便于教学与调试。

## R14

方法对比：
- 对比 LR(0)：SLR 用 FOLLOW 约束归约，冲突更少；
- 对比 LR(1)：SLR 状态更小，但归约条件更粗粒度，可能仍有冲突；
- 对比 LALR(1)：LALR 在很多工业场景更常见，精度与规模更折中。

可理解为：SLR 是“低成本增强版 LR(0)”。

## R15

常见坑与本实现处理：
1. 把 SLR 误写成 LR(0)，归约填满所有终结符。
   - 处理：严格按 `FOLLOW(lhs)` 填 reduce。
2. FOLLOW 传播漏掉“后缀可空时继承 FOLLOW(lhs)”规则。
   - 处理：`compute_follow_sets` 中显式处理 `not beta or EPS in FIRST(beta)`。
3. 表项冲突被静默覆盖。
   - 处理：`set_action/set_goto` 统一冲突检测。
4. reduce 弹栈长度错误导致状态错乱。
   - 处理：按右部长度精确弹栈（保留 `ε` 分支通用性）。

## R16

运行方式：

```bash
cd Algorithms/计算机-编译原理-0447-SLR分析
uv run python demo.py
```

预期输出包括：
- 产生式编号；
- FIRST/FOLLOW 集；
- 规范 LR(0) 状态（截断展示）；
- SLR ACTION/GOTO 表；
- 7 个样例解析轨迹；
- `All SLR checks passed.`。

## R17

交付文件说明：
- `README.md`：R01-R18 完整说明（定义、流程、复杂度、边界、运行）；
- `demo.py`：可直接运行的 SLR 最小 MVP；
- `meta.json`：保持任务元数据一致（UID/分类/源序号/目录信息一致）。

## R18

`demo.py` 源码级算法流程（9 步）：
1. `build_expression_grammar` 定义表达式文法，`augment_grammar` 构造增广开始符。  
2. `collect_terminals` 与 `enumerate_productions` 整理终结符集合和产生式索引。  
3. `compute_first_sets` 先求 FIRST，再由 `compute_follow_sets` 迭代得到 FOLLOW。  
4. `closure_lr0` 按“点后非终结符展开”做闭包，保证状态内部项目完备。  
5. `goto_lr0` 进行“点右移 + 闭包”，`canonical_lr0_collection` 构建完整 LR(0) 项目集族。  
6. `build_slr_tables` 先填 shift/goto，再把完成项的 reduce 仅写入 `FOLLOW(lhs)`。  
7. `parse_tokens` 执行移进-归约循环：查 ACTION、shift/reduce、经 GOTO 回到新状态。  
8. `run_demo` 对内置正反例执行断言，验证接受与拒绝行为是否符合预期。  
9. 用 `numpy` 汇总状态规模、ACTION 填充率与步数分布，输出可复核统计。

说明：实现没有把 SLR 交给第三方解析器“一步生成”，而是把核心链路拆成可审计的源码步骤。
