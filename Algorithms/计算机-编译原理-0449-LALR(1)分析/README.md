# LALR(1)分析

- UID: `CS-0289`
- 学科: `计算机`
- 分类: `编译原理`
- 源序号: `449`
- 目标目录: `Algorithms/计算机-编译原理-0449-LALR(1)分析`

## R01

LALR(1)（LookAhead LR）是 LR 语法分析家族中的折中方案：
- 与 LR(1) 一样，使用 1 个向前看符号（lookahead）驱动移进/归约决策；
- 与 SLR(1) 不同，归约条件来自项目项的精确 lookahead，而非粗粒度 FOLLOW 集；
- 与 LR(1) 不同，LALR(1) 会把“同 LR(0) 核”的状态合并，以明显减少状态数。

本条目实现一个可运行 MVP：先构造规范 LR(1) 项目集，再按同核合并为 LALR(1)，最后生成 ACTION/GOTO 表并执行移进-归约分析。

## R02

问题定义（本实现输入/输出）：
- 输入 1：上下文无关文法（demo 中内置算术表达式文法）；
- 输入 2：若干 token 序列（同样在 demo 中内置，非交互）；
- 输出：
  - 产生式编号（用于 `rN` 归约动作）；
  - 规范 LR(1) 状态与合并后的 LALR(1) 状态（部分展示）；
  - LALR(1) ACTION/GOTO 分析表；
  - 每个样例的逐步分析轨迹与接受/拒绝结论；
  - 状态压缩率、表填充率、步骤统计。

## R03

核心对象定义：
- LR(1) 项：`[A -> α · β, a]`，其中 `a` 是 lookahead；
- LR(0) 核：去掉 lookahead 后的三元组 `(A, αβ, dot_pos)`；
- ACTION：`shift s`、`reduce r`、`accept`；
- GOTO：`(state, nonterminal) -> state`。

`demo.py` 对应结构：
- `LR1Item(lhs, rhs, dot, lookahead)` 存项目；
- `core_of_state` 提取 LR(0) 核；
- `action_table` / `goto_table` 存分析表。

## R04

FIRST 集用于 LR(1) 闭包时计算传播 lookahead：
- 若项为 `[A -> α · B β, a]`，则对每个 `B -> γ` 引入
  `[B -> · γ, b]`，其中 `b ∈ FIRST(βa)`。

实现细节：
- `compute_first_sets` 做不动点迭代；
- `first_of_sequence` 负责 `FIRST(βa)` 的序列求值；
- 即便本示例文法无 `ε` 产生式，代码仍保留 `ε` 分支以保持通用性。

## R05

`closure_lr1` 算法：
1. 以初始项目集为起点；
2. 不断扫描可展开项（点后是非终结符）；
3. 依据 `FIRST(βa)` 生成新项目；
4. 直到集合不再增大（收敛）。

这一步保证了状态内所有“潜在可归约候选”及其 lookahead 都被完整补全。

## R06

`goto_lr1(I, X)` 算法：
- 对 `I` 中所有点后符号为 `X` 的项目，把点右移一位；
- 对右移结果再做 `closure_lr1`。

规范 LR(1) 自动机构造（`canonical_lr1_collection`）：
- 初态为 `closure({[S' -> · S, $]})`；
- 对每个状态和每个文法符号计算 GOTO；
- 新状态入队，直到没有新增状态。

## R07

LALR 合并原则：
- 若两个 LR(1) 状态的 LR(0) 核完全相同，则可合并；
- 合并后同一核项目的 lookahead 取并集。

实现函数 `merge_lalr_states`：
- `core_to_gid` 把同核状态归组；
- 在组内聚合 lookahead 构成新状态；
- 把规范 LR(1) 转移映射到合并后转移，并检查是否出现目标不一致（异常保护）。

## R08

ACTION/GOTO 表构造（`build_lalr_tables`）：
- 对项 `[A -> α · a β, b]` 且 `a` 终结符，填 `ACTION[state, a] = shift target`；
- 对项 `[A -> α · B β, b]` 且 `B` 非终结符，填 `GOTO[state, B] = target`；
- 对项 `[A -> α ·, a]`，填 `ACTION[state, a] = reduce A -> α`；
- 对项 `[S' -> S ·, $]`，填 `ACTION[state, $] = accept`。

若同一格出现不同动作，记录冲突并终止，避免误把非 LALR(1) 文法继续执行。

## R09

移进-归约解析过程（`parse_tokens`）：
1. 状态栈初始化 `[0]`，符号栈初始化 `[$]`；
2. 读当前状态 `s` 与 lookahead `a`，查 `ACTION[s, a]`；
3. `shift t`：压入符号 `a` 与状态 `t`；
4. `reduce A -> β`：弹出 `|β|` 个符号/状态，再按 `GOTO[top, A]` 转移；
5. `accept`：接受；
6. 查表失败：拒绝并报告期望符号集合。

每步都会记录 `ParseStep`，输出可追踪轨迹。

## R10

本 MVP 的实现边界：
- 聚焦语法分析，不包含词法分析器；
- token 直接用字符串列表给定（如 `id + id * id`）；
- 不做错误恢复（panic-mode），遇错即报；
- 目标是“可解释可复现的最小实现”，而非完整编译器前端。

## R11

复杂度（记）：
- `|I|`：规范 LR(1) 状态数；
- `|P|`：产生式条数；
- `|Σ|`：文法符号总数；
- `n`：输入 token 数。

主要开销：
- 规范 LR(1) 自动机构造：近似 `O(|I| * |Σ| * C_closure)`；
- 同核合并：`O(|I| * avg_items_per_state)`；
- 表构造：`O(total_items + transitions)`；
- 单次解析：`O(n)`（每步必移进或归约推进）。

## R12

示例文法（左递归表达式文法）：
- `S -> E`
- `E -> E + T | T`
- `T -> T * F | F`
- `F -> ( E ) | id`

特性：
- `*` 优先级高于 `+`；
- 通过左递归表达左结合；
- 是经典 LR/LALR 友好文法，适合展示移进-归约行为。

## R13

样例集合：
- `valid_1`: `id + id * id`（应接受）
- `valid_2`: `( id + id ) * id`（应接受）
- `valid_3`: `id * ( id + id )`（应接受）
- `invalid_1`: `id + * id`（应拒绝）
- `invalid_2`: `( id + id * id`（应拒绝，缺右括号）

脚本会逐条断言期望结果，防止“打印看似正常但语义错误”。

## R14

MVP 的工程取舍：
- 不依赖 parser generator 黑盒库；
- 所有关键过程（闭包、GOTO、合并、表构造、解析）都在源码显式实现；
- 仅引入 `numpy` 做统计（压缩率、填充率、步数均值），主算法纯 Python；
- 输出包含状态和轨迹，便于教学与调试。

## R15

与相关方法对比：
- 对比 SLR(1)：LALR(1) 利用项目项 lookahead，更少误归约冲突；
- 对比 LR(1)：LALR(1) 合并同核状态，通常大幅减少表规模；
- 对比 LL(1)：LALR(1) 能处理更广文法（含常见左递归表达式文法）。

因此，LALR(1) 常作为工程实践中的表驱动语法分析折中点。

## R16

常见坑与本实现处理：
1. 合并状态时只合并“项目全集”而非“同核”。
   - 处理：`core_of_state` 明确只取 `(lhs,rhs,dot)` 作为判据。
2. 合并后转移不一致未被检测。
   - 处理：`merge_lalr_states` 检查并报告 `inconsistencies`。
3. reduce 时栈回退长度错误。
   - 处理：按产生式右部长度精确弹栈；`ε` 时弹 0。
4. 失败信息不透明。
   - 处理：报错中给出当前状态、lookahead 与 expected 集。

## R17

运行方式：

```bash
cd Algorithms/计算机-编译原理-0449-LALR(1)分析
uv run python demo.py
```

预期行为：
- 打印产生式编号、状态样例、ACTION/GOTO 表；
- 显示 canonical LR(1) 到 LALR 的状态映射；
- 对 5 个样例输出 step-by-step 轨迹；
- 最后输出 `All LALR(1) checks passed.`。

## R18

`demo.py` 源码级算法流程（9 步）：
1. `build_expression_grammar` 定义表达式文法；`augment_grammar` 增广得到 `S' -> S`。  
2. `compute_first_sets` + `first_of_sequence` 预计算 FIRST，为闭包 lookahead 传播准备基础。  
3. `canonical_lr1_collection` 以 `closure_lr1` / `goto_lr1` 构造规范 LR(1) 项目集与转移图。  
4. `merge_lalr_states` 按 `core_of_state` 将同 LR(0) 核状态合并，并聚合同核项 lookahead。  
5. 同时把原 LR(1) 转移映射到合并后状态，若同一 `(state,symbol)` 指向不一致则立即报错。  
6. `build_lalr_tables` 从合并后项目集生成 ACTION/GOTO，并在冲突时记录并终止。  
7. `parse_tokens` 执行表驱动移进-归约，逐步记录 `ParseStep`（状态栈/符号栈/剩余输入/动作）。  
8. `run_demo` 对正例与反例做断言，验证接受与拒绝行为和预期一致。  
9. 使用 `numpy` 计算状态压缩率、ACTION 表填充率和步骤统计，输出最终总结。

说明：本实现没有把 LALR(1) 交给第三方库“一步生成”，而是把 LR(1) 构造、同核合并和分析表填充完整拆成可审计源码流程。
