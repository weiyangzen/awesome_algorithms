# Yacc/Bison算法

- UID: `CS-0290`
- 学科: `计算机`
- 分类: `编译原理`
- 源序号: `450`
- 目标目录: `Algorithms/计算机-编译原理-0450-Yacc／Bison算法`

## R01

Yacc/Bison 本质上是“**基于文法自动生成 shift-reduce 语法分析器**”的算法体系。它把上下文无关文法（CFG）转化为可执行的解析表（`ACTION/GOTO`），再用状态栈驱动输入 token 流完成语法分析。

本条目实现的是一个教学级最小 MVP：手写 `SLR(1)` 版本来复现这条核心算法链路，而不是直接调用第三方黑盒 parser。

## R02

Yacc（及 GNU Bison）在工程中通常承担两件事：

- 根据文法和优先级规则构建 LR 家族解析表；
- 生成一个可执行的 shift/reduce 驱动器，并在 reduce 时触发语义动作。

`demo.py` 与其对应关系：

- `build_canonical_lr0_collection` 对应“项集自动机生成”；
- `build_slr_parser` 对应“表项填充与冲突检测”；
- `parse_and_evaluate` 对应“运行期 shift/reduce 执行器”；
- `apply_semantic_action` 对应“归约语义动作”。

## R03

MVP 问题定义：

- 输入：算术表达式字符串（支持 `+`、`*`、括号、数字和变量）；
- 输出：表达式求值结果，且解析过程必须走 LR 表驱动而非递归下降。

使用文法（增广后）：

- `S' -> E`
- `E -> E + T | T`
- `T -> T * F | F`
- `F -> ( E ) | id`

该文法体现了典型的左递归表达式文法，也是 Yacc/Bison 示例中最常见的一类。

## R04

核心算法分解：

1. 从文法计算终结符/非终结符集合；
2. 构造 `FIRST/FOLLOW` 集；
3. 用 `closure/goto` 生成 LR(0) 规范项集族；
4. 用 FOLLOW 把完成项写入 reduce 动作，形成 SLR 的 `ACTION` 表；
5. 为非终结符迁移填充 `GOTO` 表；
6. 运行时维护状态栈，执行 `shift/reduce/accept`。

这就是 Yacc/Bison 在“生成 + 执行”两个阶段中的算法骨架。

## R05

复杂度（设文法产生式数为 `P`，项集状态数为 `N`，输入 token 数为 `L`）：

- 规范项集构建：通常在 `N` 与迁移边数上主导，近似 `O(N * P)` 量级（教学实现）；
- 表构建：`O(N * (|T| + |V|))`；
- 解析执行：`O(L)`（每个 token 触发有限次栈操作）。

因此，生成阶段离线成本更高，解析阶段在线成本线性。

## R06

正确性要点（本实现级别）：

- `set_action` 强制冲突检测，出现同一 `(state, lookahead)` 的不同动作会直接报错；
- reduce 仅在 FOLLOW 集允许的 lookahead 上触发，符合 SLR 定义；
- 解析过程只依赖 `ACTION/GOTO` 和状态栈，不做额外“猜测式”分支；
- 通过固定用例与 `numpy.allclose` 做数值一致性校验。

这保证了“可重复、可验证、可追踪”的最小正确性。

## R07

与真实 Yacc/Bison 的关系：

- 真实 Bison 默认是 LALR(1)（并支持 GLR 等模式），比本 MVP 的 SLR(1) 更强；
- 真实工具支持优先级与结合性声明来解决部分冲突；
- 本 MVP 不做代码生成，而是在 Python 中直接执行表驱动流程。

换言之，本实现覆盖“算法原理主干”，但不是完整工业编译前端。

## R08

依赖与环境：

- Python 3.10+
- `numpy`

工具栈保持最小化：`numpy` 只用于批量断言，不承担解析核心逻辑；`closure/goto/FOLLOW/shift-reduce` 都在源码中显式实现。

## R09

运行方式：

```bash
cd Algorithms/计算机-编译原理-0450-Yacc／Bison算法
uv run python demo.py
```

脚本无需交互输入，会自动：

- 构建 SLR 表；
- 执行正例表达式；
- 打印第一条表达式的解析轨迹；
- 执行一条反例并验证报错路径。

## R10

预期输出结构：

- `states/action entries/goto entries`：表规模概览；
- `[Case Results]`：每个表达式求值；
- `[Trace of first expression]`：逐步 `shift/reduce/accept` 轨迹；
- `[Negative Case]`：非法输入触发 `SyntaxError` 的确认。

若任何断言失败，脚本会直接异常退出，不会静默通过。

## R11

冲突与失败处理：

- 解析表构建冲突：`ValueError`（含 state/lookahead 冲突位点）；
- 词法层非法字符或数字：`ValueError`；
- 标识符未在变量映射里提供：`ValueError`；
- 解析过程中查不到动作或 goto：`SyntaxError`。

这种“失败即显式报错”比静默回退更适合算法教学和调试。

## R12

边界与鲁棒性：

- 支持空白字符跳过；
- 支持浮点数字面量；
- 支持变量表达式（例如 `x*(y+2)`）；
- 对坏样例 `2+*3` 明确走错误路径验证。

MVP 的目标不是“尽可能容错”，而是“尽可能暴露语法错误定位”。

## R13

当前实现限制：

- 仅覆盖 `+/*/()/id` 这组表达式文法；
- 不支持一元负号、函数调用、逗号参数列表等复杂语法；
- 不实现 Bison 的 `%left/%right/%prec` 指令；
- 不输出抽象语法树（AST），仅做语义值归约求值。

这些限制是为了保持“单文件、可读、可运行”的最小形态。

## R14

可扩展方向：

1. 把 `SLR(1)` 升级为 `LALR(1)`（核心是 LR(1) 项与核合并）；
2. 引入优先级/结合性声明，处理算符冲突；
3. 归约结果从数值切换为 AST 节点，接入后续语义分析；
4. 增加错误恢复策略（panic-mode / phrase-level recovery）。

## R15

工程适用场景：

- DSL 解释器原型；
- 配置语言解析器；
- 编译原理课程中的 LR 表驱动演示；
- 对黑盒 parser generator 结果做可视化/可解释验证。

当语法规模上升后，通常应切换到 Bison/ANTLR 等成熟工具链。

## R16

`demo.py` 关键函数导图：

- 文法与集合：`build_expression_grammar`、`compute_first_sets`、`compute_follow_sets`
- 自动机：`closure`、`goto`、`build_canonical_lr0_collection`
- 表构建：`build_slr_parser`、`set_action`
- 执行器：`tokenize`、`parse_and_evaluate`、`apply_semantic_action`
- 验证：`run_fixed_cases`、`run_negative_case`

阅读顺序建议：先看 `build_slr_parser`，再看 `parse_and_evaluate`。

## R17

交付内容：

- `README.md`：R01-R18 完整说明；
- `demo.py`：可直接运行的 Yacc/Bison 核心算法 MVP（SLR 版本）；
- `meta.json`：保留并对齐任务元数据（UID/学科/分类/源序号/目录）。

目录内文件是自包含的，不依赖外部输入。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `main` 调用 `build_slr_parser`，先把文法转成 `FIRST/FOLLOW` 集与 LR(0) 项集自动机。  
2. 在 `build_slr_parser` 中，遍历每个状态和项目，按 SLR 规则填 `ACTION/GOTO`；若出现冲突由 `set_action` 立即抛错。  
3. `run_fixed_cases` 逐条表达式调用 `parse_and_evaluate`，并收集预测值。  
4. `tokenize` 把输入字符串转成 token 序列（末尾附加 `$`），数字和变量统一映射到 `id` 终结符。  
5. `parse_and_evaluate` 维护状态栈：查 `ACTION[state, lookahead]`，遇到 `shift` 就压入新状态与 token 值。  
6. 若动作为 `reduce`，按产生式右部长度弹栈，执行 `apply_semantic_action`，再用 `GOTO[top, head]` 压回归约结果。  
7. 当动作为 `accept` 时返回最终值，同时输出完整 `shift/reduce/accept` 轨迹。  
8. `run_fixed_cases` 用 `numpy.allclose` 校验数值；`run_negative_case` 验证非法串 `2+*3` 必须走 `SyntaxError`，完成正反两向验证。

本实现未把第三方解析库当作黑盒调用；Yacc/Bison 的关键算法步骤都在源码中可逐行跟踪。
