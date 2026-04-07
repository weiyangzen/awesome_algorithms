# 寄存器分配 - 图着色

- UID: `CS-0299`
- 学科: `计算机`
- 分类: `编译原理`
- 源序号: `459`
- 目标目录: `Algorithms/计算机-编译原理-0459-寄存器分配_-_图着色`

## R01

寄存器分配（Register Allocation）要解决的问题是：把无限数量的虚拟寄存器映射到有限数量的物理寄存器，同时尽量减少内存读写（spill/reload）。

“图着色法”是经典方案之一：先构建变量间的干涉图（不能同时占同一寄存器的变量之间连边），再把图的节点着色为有限颜色（颜色即物理寄存器）。

## R02

本条目实现的是一个最小可运行 MVP，覆盖以下关键链路：

1. 在简化 CFG 上做活跃变量分析（Liveness Analysis）；
2. 用指令级 live-out 构造干涉图；
3. 采用简化版 Chaitin/Briggs 的 `Simplify + Select` 图着色流程；
4. 在寄存器不足时给出 spill 变量和栈槽编号。

不依赖 LLVM 等黑盒框架，算法细节全部在 `demo.py` 可追踪。

## R03

输入是一个手工构造的简化三地址风格 CFG：

- 基本块：`entry / then / else / merge`
- 指令类型：`const/add/sub/mul/mov/cjump/jump/ret`
- 变量：视作虚拟寄存器（如 `v1`、`v2`、`retv`）

输出包括：

- 每个块的 `live_in/live_out`
- 干涉图邻接表与节点度统计
- 颜色分配（虚拟寄存器 -> 物理寄存器）
- spill 清单（虚拟寄存器 -> `stack[i]`）
- 重写后的伪代码位置映射

## R04

活跃变量分析使用经典逆向数据流方程：

- `live_out[B] = ⋃ live_in[S]`（`S` 为后继）
- `live_in[B] = use[B] ∪ (live_out[B] - def[B])`

其中 `use/def` 基于块内指令顺序计算，直到全图不动点收敛。

## R05

干涉图构造规则（核心）：

- 对每条指令，取其定义变量 `d` 与该指令后活跃集合 `live_after`；
- 对 `live_after` 中每个变量 `x` 加边 `d -- x`（`d != x`）；
- 对 `mov dst, src`，按经典启发式跳过 `dst -- src` 这条边，保留后续合并机会。

这样得到的无向图即“不能共用寄存器”的约束图。

## R06

着色算法采用简化版 Chaitin/Briggs：

1. `Simplify`：反复删除度 `< K` 的节点并压栈；
2. 若不存在低度节点，则按 `spill_cost / degree` 最小者选择潜在 spill 节点压栈；
3. `Select`：逆序弹栈，为节点分配未被邻居占用的寄存器颜色；
4. 若无可用颜色，则该节点记为 spill。

这里 `K` 是物理寄存器数量，示例中使用 `K=3`（`r0/r1/r2`）。

## R07

spill 代价（spill cost）在本 MVP 中定义为“变量被使用/定义的总次数”，即引用频率启发式：

- 频率越高，通常越不希望被 spill；
- 与度数组合为 `cost/degree`，用于在高压阶段挑选更可接受的溢出候选。

该启发式简单但可解释，适合教学和原型验证。

## R08

`demo.py` 的样例 CFG 设计有两个目标：

- 覆盖分支合流，确保 liveness 不是纯线性局部问题；
- 制造寄存器压力，触发“着色 + 可能 spill”的完整流程。

即使某次运行样例恰好无 spill，算法仍完整执行 spill 候选选择逻辑。

## R09

复杂度直觉（设虚拟寄存器数 `V`，边数 `E`，指令数 `N`）：

- 活跃变量分析：迭代形式下近似 `O(iter * (N + E))`；
- 干涉图构建：近似 `O(N * avg_live)`；
- 图着色（简化实现）：约 `O(V^2)` 量级（集合操作与邻接更新）。

对于课程规模和中小 CFG，开销可控。

## R10

正确性要点：

- 干涉图边表达“同时活跃冲突”，保证相邻节点不能拿同色；
- `assert_valid_coloring` 会检查所有已着色相邻节点颜色不同；
- 若无可用颜色即标记 spill，避免错误强行分配造成冲突。

因此结果是“保守可执行”的分配方案。

## R11

边界与异常处理：

- CFG 后继块不存在时抛 `ValueError`；
- 物理寄存器集合为空时抛 `ValueError`；
- 指令类型不支持时抛 `ValueError`；
- 多处 `assert` 用于验证分配结果覆盖所有虚拟寄存器。

这能避免 silent failure，便于定位实现问题。

## R12

与工业编译器的差异（有意简化）：

- 未实现迭代“真实重写后再分配”（真实系统会插入 load/store 后重试）；
- 未实现保守合并、冻结、偏好寄存器、预着色等完整 Briggs 细节；
- 未建模调用约定（caller-saved/callee-saved）与寄存器类别约束。

本条目聚焦“图着色分配主干流程”的透明实现。

## R13

最小工具栈说明：

- 语言：Python 3.10+
- 第三方：`numpy`（仅用于图度统计展示）
- 其余逻辑全部标准库实现

符合“小而诚实”的 MVP 原则，避免把核心算法交给黑盒库。

## R14

运行方式：

```bash
cd Algorithms/计算机-编译原理-0459-寄存器分配_-_图着色
uv run python demo.py
```

脚本不读取任何交互输入，运行后会依次打印：

- 输入 CFG
- 块级 liveness
- 干涉图与度统计
- 寄存器分配结果
- 重写后的伪代码位置映射
- 通过断言信息

## R15

常见实现误区与本实现处理：

1. 只做块级活跃，不回推到指令级就建图。
   - 处理：`instruction_live_out_sets` 生成指令级 live-out。
2. 忽略 `mov` 特殊性，导致过多干涉边。
   - 处理：对 `mov dst, src` 跳过 `dst--src`。
3. 只在高压阶段标记 spill，不在着色阶段复核。
   - 处理：真正 spill 判定在 `Select` 阶段“无色可用”时完成。

## R16

可扩展方向：

- 增加“插入 load/store 后重新分配”的完整 spill-rewrite 循环；
- 引入预着色和寄存器类约束（整数/浮点分离）；
- 增加 move coalescing / freezing / optimistic coloring；
- 接入 SSA IR，把 phi 处理纳入分配流程；
- 输出可视化干涉图（如 Graphviz）。

## R17

文件职责：

- `README.md`：R01-R18 说明图着色寄存器分配的理论、工程取舍与代码映射；
- `demo.py`：可运行 MVP，包含 liveness、建图、着色、spill 和重写输出；
- `meta.json`：任务元信息（UID、学科、子类、源序号、路径）。

目录自包含，可直接用于验证。

## R18

`demo.py` 源码级算法流程（9 步）：

1. `build_sample_cfg` 构造带分支的示例 CFG，定义虚拟寄存器与控制流。  
2. `liveness_analysis` 基于 `use/def` 和逆向数据流方程迭代求每个基本块的 `live_in/live_out`。  
3. `instruction_live_out_sets` 从块级 `live_out` 反向扫描到指令级，得到每条指令执行后的活跃集合。  
4. `build_interference_graph` 按“定义变量 vs 指令后活跃变量”加无向边，构建干涉图；对 `mov` 跳过 `dst--src` 干涉边。  
5. `compute_spill_costs` 统计每个虚拟寄存器的引用频次，作为 spill 启发式成本。  
6. `graph_coloring_allocate` 进入 `Simplify` 阶段：优先移除度 `< K` 节点，否则按 `cost/degree` 最小策略选择潜在 spill 节点压栈。  
7. `graph_coloring_allocate` 进入 `Select` 阶段：逆序弹栈并分配可用物理寄存器；若无可用颜色则标记 spill 并分配 `stack[i]`。  
8. `assert_valid_coloring` 验证所有已着色相邻节点颜色不同，确保分配满足冲突约束。  
9. `print_rewritten_program` 将虚拟寄存器替换为 `r0/r1/r2` 或 `stack[i]`，展示最终位置分配结果。

第三方库没有替代核心算法逻辑；`numpy` 仅用于干涉图度统计展示，寄存器分配主流程完全由本地源码逐步实现。
