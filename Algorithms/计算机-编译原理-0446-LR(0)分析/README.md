# LR(0)分析

- UID: `CS-0286`
- 学科: `计算机`
- 分类: `编译原理`
- 源序号: `446`
- 目标目录: `Algorithms/计算机-编译原理-0446-LR(0)分析`

## R01

LR(0)（LookAhead=0 的 LR）是最基础的自底向上语法分析方法：
- 用“项目（item）+ 状态自动机”表示分析上下文；
- 用 ACTION/GOTO 表驱动移进-归约过程；
- 不使用任何向前看符号，归约动作仅由“完成项目”触发。

本条目给出一个教学级最小 MVP：从零构造规范 LR(0) 项目集族、生成分析表、并对固定 token 序列执行可追踪解析。

## R02

本实现的问题定义：
- 输入 1：上下文无关文法（脚本内置）；
- 输入 2：若干 token 序列（脚本内置，无交互）；
- 输出：
  - 产生式编号（用于 `rN` 归约）；
  - 规范 LR(0) 状态集（项目集族）；
  - ACTION/GOTO 表；
  - 每个样例的逐步分析轨迹与接受/拒绝结论；
  - 状态数量、表填充率、步数统计。

## R03

选择 LR(0) 的原因：
- 是 LR 家族最小原型，最适合展示“闭包-转移-造表-解析”的完整链路；
- 可直接说明为何某些文法会出现 shift/reduce 冲突；
- 便于与 LR(1)、LALR(1) 做后续对照（信息量 vs 状态规模）。

## R04

核心定义（与 `demo.py` 对应）：
- LR(0) 项：`[A -> α · β]`，不含 lookahead；
- 状态：一组 LR(0) 项（一个项目集）；
- ACTION：`shift s` / `reduce r` / `accept`；
- GOTO：`(state, nonterminal) -> state`。

代码中的关键数据结构：
- `LR0Item(lhs, rhs, dot)`；
- `action_table[(state, terminal)]`；
- `goto_table[(state, nonterminal)]`；
- `ParseStep/ParseResult` 保存逐步轨迹与结果。

## R05

`closure_lr0` 算法：
1. 从初始项目集开始；
2. 若存在项 `A -> α · B β` 且 `B` 是非终结符，则加入 `B` 的所有初始项 `B -> · γ`；
3. 重复直到项目集不再增长。

该闭包过程只依赖文法结构，不依赖 FOLLOW/FIRST 或 lookahead。

## R06

`goto_lr0(I, X)` 与规范项目集族构造：
- `goto_lr0`：把 `I` 中点前为 `X` 的项统一右移一位，再做闭包；
- `canonical_lr0_collection`：
  - 初态：`closure({S' -> · S})`；
  - 枚举每个状态对每个文法符号的 GOTO；
  - 若产生新项目集就分配新状态编号；
  - 直到无新状态。

结果是一个 DFA（状态图），其边即 ACTION/GOTO 的来源。

## R07

LR(0) 分析表生成规则（`build_lr0_tables`）：
- 若项形如 `A -> α · a β`，且 `a` 为终结符，填 `ACTION[state, a] = shift`；
- 若项形如 `A -> α · B β`，且 `B` 为非终结符，填 `GOTO[state, B]`；
- 若项形如 `A -> α ·`：
  - 若 `A` 是增广开始符，填 `ACTION[state, $] = accept`；
  - 否则对所有终结符和 `$` 填 `reduce A -> α`（这是 LR(0) 的关键特征）。

若同一格被写入不一致动作，立即记录冲突并报错。

## R08

表驱动移进-归约流程（`parse_tokens`）：
1. 初始化状态栈 `[0]`、符号栈 `[$]`；
2. 取当前状态和输入符号查询 ACTION；
3. `shift`：压入符号与目标状态，输入前进；
4. `reduce`：按产生式右部长度弹栈，随后按 GOTO 压入左部；
5. `accept`：成功结束；
6. 无动作：报错并给出该状态可接受终结符集合。

每一步都会写入 `ParseStep`，可复盘整个决策序列。

## R09

MVP 使用的示例文法：
- `S -> C C`
- `C -> c C | d`

该文法是经典可 LR(0) 处理示例，语言形式可理解为：
- 两段 `C` 串联；
- 每段 `C` 由若干个前缀 `c`，最后接一个 `d`。

## R10

内置样例集（带断言）：
- 正例：
  - `d d`
  - `c d d`
  - `c c d d`
  - `d c d`
- 反例：
  - `d`
  - `c d c`
  - `c c d`

脚本会逐例打印轨迹，并断言 `accepted == expected`。

## R11

实现边界：
- 只实现语法分析，不包含词法分析器；
- 输入 token 直接给定为字符串列表；
- 不实现错误恢复（panic-mode），遇到表项缺失直接返回失败；
- 目标是“可审计 LR(0) 核心逻辑”，不是完整编译器前端。

## R12

复杂度（记状态数为 `|I|`，符号总数为 `|V|+|T|`，输入长度为 `n`）：
- 规范项目集构造：约 `O(|I| * (|V|+|T|) * C_closure)`；
- ACTION/GOTO 表构造：约 `O(total_items + |I| * |T|)`；
- 单次解析：通常 `O(n)`（每步移进或归约，栈操作为常数级）。

空间复杂度主要由状态项目集与分析表决定，约 `O(total_items + table_size)`。

## R13

工程取舍：
- 不依赖 parser generator 黑盒，`closure/goto/造表/解析` 全源码实现；
- 代码保持纯 Python 主流程，仅用 `numpy` 做统计汇总（步数与填充率）；
- 输出“产生式 + 状态 + 表 + 轨迹 + 汇总”，便于教学与调试。

## R14

与相邻方法对比：
- 对比 LL(1)：LR(0) 是自底向上，天然更适配左递归文法；
- 对比 SLR(1)：LR(0) 归约不看 FOLLOW，冲突更易出现；
- 对比 LR(1)：LR(1) 用 lookahead 精细区分归约条件，表达能力更强；
- 对比 LALR(1)：LALR(1) 兼顾状态规模与能力，工程里更常见。

LR(0) 的价值是“最简可执行模型”，不是覆盖面最大的方法。

## R15

常见坑与对应处理：
1. `closure` 漏掉递归展开，导致状态不完整。
   - 处理：不动点迭代直到无新增项。
2. `goto` 只做点右移不做闭包。
   - 处理：`goto_lr0` 统一调用 `closure_lr0`。
3. 归约动作只填部分终结符。
   - 处理：严格按 LR(0) 规则，对所有终结符与 `$` 填入 reduce。
4. 表项覆盖不报错。
   - 处理：`set_action/set_goto` 统一冲突检测。

## R16

运行方式：

```bash
cd Algorithms/计算机-编译原理-0446-LR(0)分析
uv run python demo.py
```

预期可见：
- 产生式编号；
- 规范 LR(0) 状态（项目）；
- ACTION/GOTO 表；
- 7 个样例的解析轨迹；
- `All LR(0) checks passed.`。

## R17

交付文件说明：
- `README.md`：R01-R18 全部完成，覆盖定义、流程、复杂度、边界与运行指引；
- `demo.py`：可直接运行的 LR(0) 最小 MVP；
- `meta.json`：保持任务元数据一致（UID/分类/源序号/目录信息不变）。

## R18

`demo.py` 源码级算法流（9 步）：
1. `build_lr0_demo_grammar` 定义文法，`augment_grammar` 生成增广开始符 `S'`。  
2. `collect_terminals` 与 `enumerate_productions` 整理终结符集合和产生式索引。  
3. `closure_lr0` 通过“不动点扩展点后非终结符”得到闭包项目集。  
4. `goto_lr0` 对指定符号执行“点右移 + 闭包”得到目标状态。  
5. `canonical_lr0_collection` 从初态出发遍历所有符号，构造完整项目集族与 DFA 转移。  
6. `build_lr0_tables` 按 LR(0) 规则填 ACTION/GOTO，并在写表冲突时记录错误。  
7. `parse_tokens` 执行移进-归约循环：查 ACTION、执行 shift/reduce、经 GOTO 回到新状态。  
8. `run_demo` 跑内置正反例并断言期望结果，确保“可运行且结论正确”。  
9. 使用 `numpy` 统计状态数、ACTION 填充率、步数分布，输出最终汇总。

说明：本实现没有调用第三方解析器“一键生成”，而是把 LR(0) 全流程拆成可审计源码步骤。
