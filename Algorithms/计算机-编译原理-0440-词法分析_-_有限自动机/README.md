# 词法分析 - 有限自动机

- UID: `CS-0280`
- 学科: `计算机`
- 分类: `编译原理`
- 源序号: `440`
- 目标目录: `Algorithms/计算机-编译原理-0440-词法分析_-_有限自动机`

## R01

词法分析（Lexical Analysis）是编译前端第一步：把源代码字符流切分成 token 流。有限自动机（这里用 DFA）是最常见的实现模型，因为它能在线性时间内逐字符扫描，并天然支持“最长匹配（maximal munch）”。

本条目实现一个可运行 MVP：
- 显式定义状态集合、字符类别、转移表；
- 支持常见 token（标识符、关键字、整数、运算符、括号与分隔符）；
- 输出 token 序列并做固定样例自检。

## R02

问题定义（本实现语义）：
- 输入：`text: str`（源码字符串）。
- 输出：`List[Token]`，每个 token 含 `token_type/lexeme/start/end`。
- 扫描策略：左到右、最长匹配、忽略空白 token（`WS` 不进入最终输出）。

脚本运行时使用内置样例，不依赖交互输入。

## R03

支持的词法单元：
- 标识符：`[A-Za-z_][A-Za-z0-9_]*`
- 关键字：`if/else/while/return/int/float/for`（由 `IDENT` 后处理提升）
- 整数：`[0-9]+`
- 运算符：`= == ! != < <= > >= + - * /`
- 界符：`(` `)` `{` `}` `;` `,`
- 空白：空格、制表符、换行（识别但丢弃）

## R04

DFA 状态设计（22 个状态）：
- `S_START`：起始态；
- `S_IDENT`、`S_INT`：多字符词法单元；
- `S_EQ/S_EQEQ`、`S_BANG/S_NEQ`、`S_LT/S_LE`、`S_GT/S_GE`：双字符运算符分流；
- `S_PLUS/S_MINUS/S_STAR/S_SLASH/...`：单字符 token；
- `S_WS`：空白连续吞并状态。

其中多数状态是接受态，`S_START` 不是接受态。

## R05

字符先映射为有限的类别，再查表转移。类别包括：
- `LETTER/DIGIT/UNDERSCORE/WS`
- `EQ/EXCL/LT/GT/PLUS/MINUS/STAR/SLASH`
- `LPAREN/RPAREN/LBRACE/RBRACE/SEMI/COMMA`
- `OTHER`

`OTHER` 在任何状态都无有效转移，触发词法错误。

## R06

核心数据结构（`demo.py`）：
- `transitions: np.ndarray`，形状 `(N_STATES, N_CLASSES)`，默认 `-1`（无边）；
- `accept_token: np.ndarray`，长度 `N_STATES`，记录“该状态接受时对应 token id”；
- `TOKEN_NAMES`：token id 到字符串名称映射。

这使得 DFA 运行时只做常数时间索引：`next_state = transitions[state, class]`。

## R07

扫描算法采用“最长匹配”：
1. 每次从当前位置 `pos` 和 `S_START` 出发；
2. 持续读取字符并按表迁移；
3. 维护最近一次到达的接受态与位置 `last_accept_state/last_accept_pos`；
4. 当下一步无转移时，回退到最近接受位置产出 token；
5. 若从未到达接受态，则该位置为非法词法输入。

该策略可正确区分 `=` 与 `==`、`<` 与 `<=` 等前缀冲突 token。

## R08

关键字识别采用“两阶段”策略：
- DFA 先统一识别成 `IDENT`；
- 产出 token 前检查词素是否在关键字集合中，若在则改写为 `KW_XXX`。

这样避免为每个关键字单独扩展大量状态，保持自动机简洁。

## R09

复杂度分析：
- 设输入长度为 `n`。
- 时间复杂度：`O(n)`。每个字符最多被常数次处理（推进与分词边界判断）。
- 空间复杂度：
  - 运行时额外状态为 `O(1)`；
  - 输出 token 列表为 `O(k)`（`k` 为 token 数）。

DFA 表规模固定，为常量开销。

## R10

正确性要点：
- 转移完备性：支持语法集合内每类字符在可达状态都有定义；
- 最长匹配：通过“记录最后接受态”保证前缀冲突时取最长合法词素；
- 边界无歧义：双字符运算符由中间态细化（如 `S_EQ -> S_EQEQ`）；
- 关键字一致性：`IDENT` 后处理保证关键字不会误判为普通标识符。

## R11

异常处理：
- 当某位置无法走出任何合法 token（`last_accept_state < 0`）时抛 `ValueError`；
- 错误信息包含当前位置和上下文片段，便于定位非法字符。

当前 MVP 专注于核心词法机制，不处理注释、字符串字面量和浮点数。

## R12

实现取舍：
- 依赖栈尽量小，仅使用 `numpy` 来表达 DFA 表（非黑盒，状态机逻辑完全手写）；
- 不使用 `re` 直接做整段 tokenization，确保“有限自动机实现路径”可审计；
- 无命令行参数，脚本运行即展示样例与测试结果，便于自动验证。

## R13

`demo.py` 代码结构：
- `classify_char`：字符分类器；
- `build_dfa_tables`：构建转移表与接受态表；
- `normalize_token_type`：关键字提升；
- `tokenize`：DFA 分词主流程（maximal munch）；
- `run_demo_cases`：固定测试集断言；
- `pretty_print_tokens`：打印 token 表；
- `main`：串联演示。

## R14

运行方式：

```bash
cd Algorithms/计算机-编译原理-0440-词法分析_-_有限自动机
uv run python demo.py
```

或在仓库根目录执行：

```bash
uv run python Algorithms/计算机-编译原理-0440-词法分析_-_有限自动机/demo.py
```

## R15

预期输出特征：
- 首行打印 DFA 规模（状态数、字符类别数、接受态个数）；
- 展示一段多行源码的分词表（token 类型、词素、字符区间）；
- 执行 3 组确定性测试并打印 `pass=True/False`；
- 全部通过后打印 `All lexer DFA demo cases passed.`。

## R16

常见坑点：
- 忽略最长匹配，导致 `==` 被拆成两个 `=`；
- 未把空白单独建模，导致 token 边界混乱；
- `IDENT` 与关键字混在 DFA 中硬编码，状态爆炸；
- 漏掉错误分支，非法字符被静默吞掉。

## R17

可扩展方向：
- 增加字符串/字符字面量与转义序列状态；
- 增加注释（`//`、`/*...*/`）状态机；
- 增加浮点数、科学计数法；
- 增加 token 位置信息中的行列号（而不仅是绝对索引）；
- 与后续语法分析器（LL/LR）直接对接。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 调用 `build_dfa_tables`，创建 `transitions` 与 `accept_token` 两张核心表。  
2. `main` 准备示例源码字符串并调用 `tokenize` 进入词法扫描。  
3. `tokenize` 在每个 `pos` 从 `S_START` 出发，循环读取字符并用 `classify_char` 得到类别列索引。  
4. 通过 `next_state = transitions[state, class]` 推进 DFA；每到达接受态就更新 `last_accept_state/last_accept_pos`。  
5. 当无可用转移时：若存在最近接受态，则按 `[pos:last_accept_pos)` 切出词素；若不存在，抛出词法错误。  
6. 根据 `accept_token[last_accept_state]` 取得基础 token 类型，再由 `normalize_token_type` 把关键字从 `IDENT` 提升为 `KW_*`。  
7. 空白 token（`WS`）被丢弃，其余 token 追加到结果列表；`pos` 前移到 `last_accept_pos` 继续扫描。  
8. 输入结束后追加 `EOF`，`main` 打印 token 表并调用 `run_demo_cases` 验证固定样例全部通过。
