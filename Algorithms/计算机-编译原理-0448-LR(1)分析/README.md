# LR(1)分析

- UID: `CS-0288`
- 学科: `计算机`
- 分类: `编译原理`
- 源序号: `448`
- 目标目录: `Algorithms/计算机-编译原理-0448-LR(1)分析`

## R01

LR(1)（LookAhead LR）是自底向上语法分析方法中的经典“全信息”表驱动版本：
- 以 LR 项和 1 个向前看符号（lookahead）做动作判定；
- 相比 SLR(1) 的 FOLLOW 近似，LR(1) 的归约条件更精细；
- 相比 LALR(1)，LR(1) 不做同核状态合并，状态通常更多但冲突更少。

本条目实现一个教学级可运行 MVP：从零构造规范 LR(1) 项目集族、生成 ACTION/GOTO 表，并对固定 token 序列执行移进-归约分析。

## R02

问题定义（本实现输入/输出）：
- 输入 1：上下文无关文法（demo 内置表达式文法）；
- 输入 2：若干 token 序列（demo 内置，非交互）；
- 输出：
  - 产生式编号（用于 `rN` 归约动作）；
  - 规范 LR(1) 项目集状态（部分展示）；
  - LR(1) ACTION/GOTO 分析表；
  - 每个样例的逐步分析轨迹与接受/拒绝结论；
  - 状态规模、表填充率与步骤统计。

## R03

核心对象定义：
- LR(1) 项：`[A -> α · β, a]`，其中 `a` 为 lookahead；
- ACTION：`shift s`、`reduce r`、`accept`；
- GOTO：`(state, nonterminal) -> state`；
- 解析状态：由“状态栈 + 符号栈 + 当前输入指针”组成。

`demo.py` 中对应数据结构：
- `LR1Item(lhs, rhs, dot, lookahead)`；
- `action_table[(state, terminal)]`；
- `goto_table[(state, nonterminal)]`；
- `ParseStep/ParseResult` 用于可追踪执行轨迹。

## R04

FIRST 集用于 LR(1) 闭包中的 lookahead 传播：
- 对项 `[A -> α · B β, a]`，闭包新增 `[B -> · γ, b]`；
- 其中 `b ∈ FIRST(βa)`。

实现细节：
- `compute_first_sets` 不动点迭代求 FIRST；
- `first_of_sequence` 支持序列 FIRST 计算；
- 即便示例文法无 `ε` 产生式，代码仍保留 `ε` 分支，保证通用性。

## R05

`closure_lr1` 算法：
1. 以初始项目集作为闭包种子；
2. 扫描每个可展开项（点后是非终结符）；
3. 根据 `FIRST(βa)` 生成新初始项；
4. 直到集合不再增长（收敛）。

这一过程保证状态内项目及其 lookahead 完备，不依赖外部黑盒库。

## R06

`goto_lr1(I, X)` 算法：
- 将 `I` 中点后为 `X` 的项目统一右移一位；
- 对右移结果执行 `closure_lr1`。

规范 LR(1) 自动机构造（`canonical_lr1_collection`）：
- 初态：`closure({[S' -> · S, $]})`；
- 对每个状态和每个文法符号计算 GOTO；
- 新状态入队，直到无新增状态。

## R07

ACTION/GOTO 表构造（`build_lr1_tables`）：
- 若项 `[A -> α · a β, b]` 且 `a` 为终结符，填 `ACTION = shift`；
- 若项 `[A -> α · B β, b]` 且 `B` 为非终结符，填 `GOTO`；
- 若项 `[A -> α ·, a]`，填 `ACTION = reduce A -> α`；
- 若项 `[S' -> S ·, $]`，填 `ACTION = accept`。

若同一单元出现不一致动作，立即记录冲突并终止，避免错误表继续参与解析。

## R08

移进-归约执行（`parse_tokens`）：
1. 初始化状态栈 `[0]`、符号栈 `[$]`；
2. 读取 `(state, lookahead)` 查询 ACTION；
3. `shift`：压入符号和目标状态；
4. `reduce`：按产生式右部长度弹栈，再查 GOTO 压入左部；
5. `accept`：接受并结束；
6. 无动作：拒绝并报告该状态可期待终结符集合。

每一步生成 `ParseStep`，可直接复盘。

## R09

示例文法（表达式文法）：
- `S -> E`
- `E -> E + T | T`
- `T -> T * F | F`
- `F -> ( E ) | id`

语义特性：
- `*` 优先级高于 `+`；
- `E/T` 的左递归实现左结合；
- 是典型 LR 家族友好文法。

## R10

样例集合（内置断言）：
- `valid_1`: `id + id * id`（应接受）
- `valid_2`: `( id + id ) * id`（应接受）
- `valid_3`: `id * ( id + id )`（应接受）
- `invalid_1`: `id + * id`（应拒绝）
- `invalid_2`: `( id + id * id`（应拒绝，缺右括号）

脚本对每条样例做 `assert`，防止“看起来运行了但结论不对”。

## R11

实现边界：
- 仅覆盖语法分析，不含词法分析器；
- token 以字符串序列直接给定；
- 不实现错误恢复（panic-mode），出错立即返回；
- 目标是最小可复现 LR(1) 核心链路，不是完整编译器前端。

## R12

复杂度说明（记 `|I|` 为 LR(1) 状态数，`|Σ|` 为符号总数，`n` 为输入长度）：
- 项目集构造约为 `O(|I| * |Σ| * C_closure)`；
- 分析表构造约为 `O(total_items + transitions)`；
- 单次解析时间通常线性于输入长度，近似 `O(n)`（每步要么移进要么归约）。

## R13

工程取舍：
- 不依赖 parser generator 或编译器工具链黑盒；
- `closure/goto/FIRST/表构造/解析` 全在源码显式实现；
- 仅引入 `numpy` 做统计（表填充率、步骤均值），主流程保持纯 Python；
- 输出“状态 + 表 + 轨迹”三层信息，便于教学调试。

## R14

与相邻方法对比：
- 对比 SLR(1)：LR(1) 用精确 lookahead，冲突更少但状态更多；
- 对比 LALR(1)：LR(1) 不合并状态，表规模更大但信息保真；
- 对比 LL(1)：LR(1) 能自然处理常见左递归表达式文法。

结论：LR(1) 是“精确性优先”的标准构造。

## R15

常见坑与本实现对应处理：
1. 闭包 lookahead 传播写成 `FIRST(β)`，漏掉尾随 `a`。
   - 处理：使用 `FIRST(βa)`（`lookahead_seed = beta + (item.lookahead,)`）。
2. 表构造时 shift/reduce 覆盖无检测。
   - 处理：`set_action/set_goto` 统一做冲突检测。
3. reduce 弹栈长度错误。
   - 处理：按右部长度精确弹栈，`ε` 右部弹 0。
4. 失败信息不可定位。
   - 处理：错误中输出状态、lookahead、expected 集。

## R16

运行方式：

```bash
cd Algorithms/计算机-编译原理-0448-LR(1)分析
uv run python demo.py
```

预期输出：
- 产生式编号；
- 规范 LR(1) 状态（截断展示）；
- ACTION/GOTO 表；
- 5 条样例的 step-by-step 轨迹；
- `All LR(1) checks passed.`。

## R17

交付文件说明：
- `README.md`：R01-R18 完整说明（算法、边界、复杂度、运行与审计要点）；
- `demo.py`：可直接运行的 LR(1) MVP；
- `meta.json`：任务元数据（UID、学科、分类、源序号、目录）保持一致。

## R18

`demo.py` 源码级算法流程（9 步）：
1. `build_expression_grammar` 定义文法，`augment_grammar` 生成增广开始符。  
2. `compute_first_sets` 与 `first_of_sequence` 计算 FIRST，用于闭包 lookahead 传播。  
3. `closure_lr1` 从种子项反复展开非终结符，按 `FIRST(βa)` 追加项目直至收敛。  
4. `goto_lr1` 对指定符号执行“点右移 + 闭包”，得到状态转移目标。  
5. `canonical_lr1_collection` 从初态出发遍历所有符号，构造规范 LR(1) 项目集族与 DFA 转移。  
6. `build_lr1_tables` 按项目形态填充 ACTION/GOTO，并在冲突时立即记录。  
7. `parse_tokens` 按表驱动执行移进-归约，逐步生成 `ParseStep` 轨迹。  
8. `run_demo` 对正例/反例执行断言，验证接受与拒绝行为符合预期。  
9. 使用 `numpy` 统计状态数量、ACTION 表填充率与步骤分布并输出汇总。

说明：实现没有把 LR(1) 交给第三方库“一步生成”，而是把闭包、GOTO、自动机构造、表填充和解析过程完整拆解为可审计源码步骤。
