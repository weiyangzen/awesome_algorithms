# 死代码消除

- UID: `CS-0296`
- 学科: `计算机`
- 分类: `编译原理`
- 源序号: `456`
- 目标目录: `Algorithms/计算机-编译原理-0456-死代码消除`

## R01

死代码消除（Dead Code Elimination, DCE）是编译优化中的基础步骤，目标是在不改变程序可观察语义的前提下删除“不会影响最终结果”的代码。

本目录实现的是**基于活跃变量（Liveness）分析的赋值死代码消除**：
- 删除无副作用、且定义结果不再被使用的赋值语句；
- 保留 `print`、`return`、`call` 这类有可观察效果的指令；
- 在含分支的控制流图（CFG）上工作，而不是只做线性代码扫描。

## R02

问题输入与输出（MVP 语义）：
- 输入：一个由基本块组成的简化三地址代码 CFG；
- 输出：删除死代码后的新 CFG 与删除报告（每轮删除了哪些语句）。

死代码判定标准：
- 指令定义变量 `x`；
- 指令本身没有副作用；
- 在该指令执行点之后，`x` 不活跃（`x ∉ live_out_at_inst`）。

满足以上条件可删除该指令。

## R03

核心数据流方程（按基本块）：

- `live_out[B] = ⋃ live_in[S]`，其中 `S` 是 `B` 的后继块；
- `live_in[B] = use[B] ∪ (live_out[B] - def[B])`。

其中：
- `use[B]`：块内“先使用后定义”的变量集合；
- `def[B]`：块内被定义过的变量集合。

在实现上通过迭代直到不再变化得到全局不动点。

## R04

MVP 算法框架：
1. 对当前 CFG 做活跃变量分析；
2. 以 `live_out` 为起点，对每个基本块自底向上逆序扫描；
3. 遇到“纯赋值且定义变量不活跃”的指令则删除；
4. 若本轮有删除，重新做活跃变量分析并继续下一轮；
5. 直到某轮删除数为 0。

这样可以覆盖“删掉后又触发新的死代码”的链式场景。

## R05

`demo.py` 的 IR 抽象：
- `Instruction`：单条指令，字段包括 `op/dst/args/target` 等；
- `BasicBlock`：基本块，包含 `instructions` 与 `successors`；
- `DceReport`：优化报告（轮数、总删除数、逐轮删除列表）；
- `RunResult`：解释执行结果（返回值、print 输出、trace 日志）。

支持的关键操作：
- 算术：`add/sub/mul/div/copy/const`；
- 控制流：`jump/cjump/return`；
- 副作用：`print/call`（其中 `call trace(...)` 被定义为有副作用）。

## R06

正确性关键点：
- 只删除“无副作用 + 结果不活跃”的赋值；
- 副作用语句即使结果未使用也保留（例如 `dbg = call trace(v)`）；
- 优化后用同一组输入执行“优化前/优化后”程序，比较：
  - 返回值；
  - `print` 输出序列；
  - `trace` 日志序列。

三者一致才认为语义保持。

## R07

复杂度（设总指令数 `N`，边数 `E`，迭代轮数 `K`）：
- 一次活跃变量不动点迭代复杂度近似 `O(E * I)`（`I` 为收敛迭代次数）；
- 一轮指令逆序扫描复杂度 `O(N)`；
- 全流程复杂度约 `O(K * (E * I + N))`。

在教学级小 CFG 上，该成本非常可控。

## R08

边界与异常处理：
- `max_passes <= 0` 直接抛 `ValueError`；
- 解释执行阶段若读取未定义变量，抛 `KeyError`；
- 若块执行越界（无终结语句）或步数超限，抛 `RuntimeError`；
- 未支持的操作符会抛 `ValueError`。

这些保护让失败模式显式化，便于调试。

## R09

MVP 取舍：
- 采用简化 IR 与手写数据流分析，重点展示 DCE 原理；
- 不依赖编译框架（如 LLVM）黑盒优化；
- 不实现 SSA、别名分析、全局值编号等进阶优化；
- 使用最小工具栈：`Python + numpy`（`numpy` 仅用于数值一致性断言）。

## R10

`demo.py` 函数职责：
- `build_sample_cfg`：构造含分支和副作用的示例 CFG；
- `compute_use_def` / `liveness_analysis`：数据流分析核心；
- `dce_one_pass` / `eliminate_dead_code`：逐轮删除死赋值直到稳定；
- `execute_program`：解释执行简化 IR；
- `run_equivalence_tests`：验证优化前后语义等价；
- `ensure_no_pure_dead_assignments`：二次检查“无可删纯赋值残留”；
- `main`：串联流程并打印优化报告。

## R11

运行方式：

```bash
cd Algorithms/计算机-编译原理-0456-死代码消除
uv run python demo.py
```

脚本不读取交互输入，运行后会自动完成优化、等价性校验和结果打印。

## R12

示例程序结构（`build_sample_cfg`）：
- `entry`：计算 `t0/t1`，包含 `dead_entry`（可删），再条件跳转；
- `then`：计算 `u`，包含 `dead_then`（可删）；
- `else`：计算 `u`，包含 `dead_else`（可删）；
- `join`：包含
  - `tmp = v * 2`（可删），
  - `dbg = call trace(v)`（有副作用，保留），
  - `return ret` 后的 `dead_after_return`（可删）。

该示例覆盖了：分支合流、链式死代码、以及“有副作用但结果未使用”的判定边界。

## R13

预期输出要点：
- 打印优化前后指令总数；
- 打印删除比例与优化轮次；
- 按轮列出被删除的具体 IR 语句；
- 打印优化前后的 CFG 文本；
- 最终输出 `Semantic equivalence checks passed.`。

若断言失败，脚本会直接抛异常停止，避免静默错误。

## R14

常见误区与本实现的处理：
- 误区 1：只看“变量是否被读”，忽略副作用。
  - 处理：`call/print/return` 一律视为不可删。
- 误区 2：只做一轮删除。
  - 处理：`eliminate_dead_code` 做多轮直到稳定。
- 误区 3：只在基本块内做局部分析。
  - 处理：使用 CFG 级别的 `live_in/live_out` 方程。

## R15

与相关优化的关系：
- 与常量折叠不同：DCE 不改变表达式值，只删除无效语句；
- 与复制传播不同：DCE 不主动改写使用点，只做删除；
- 与不可达代码消除有交集：`return` 后无效赋值在本模型中也会被删掉。

实际编译器通常将这些优化组合迭代执行。

## R16

工程化扩展方向：
- 在 SSA 形式上执行 DCE（变量版本更清晰）；
- 引入内存别名分析，提升对 `store/load` 可删性的判断精度；
- 将“副作用模型”从硬编码扩展为函数属性表（pure/impure）；
- 增加真实 IR 解析器（如三地址文本或字节码）并导出优化前后 diff。

## R17

交付文件说明：
- `README.md`：R01-R18 的算法说明与实现细节；
- `demo.py`：可直接运行的 DCE MVP；
- `meta.json`：任务元信息（UID、学科、子类、源序号、目录路径）。

目录是自包含的，不依赖外部输入文件。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 构建示例 CFG（含 `entry/then/else/join` 四个基本块）。
2. 调用 `eliminate_dead_code` 开始优化循环。
3. 每轮先用 `liveness_analysis` 计算全图 `live_in/live_out` 不动点。
4. `dce_one_pass` 对每个块逆序扫描：若指令定义变量且无副作用，并且定义变量不在当前 `live` 集合中，则删除。
5. 对保留指令执行 `live = (live - def) ∪ use` 更新，继续向前扫描。
6. 若本轮有删除则进入下一轮，否则停止并返回 `DceReport`。
7. `run_equivalence_tests` 在多组输入下分别执行优化前后 CFG，比对返回值、打印输出与 trace 日志。
8. `ensure_no_pure_dead_assignments` 再次静态检查无可删纯赋值残留，最后输出报告与优化前后 CFG。
