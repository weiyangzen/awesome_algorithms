# 活跃区间分析

- UID: `CS-0300`
- 学科: `计算机`
- 分类: `编译原理`
- 源序号: `460`
- 目标目录: `Algorithms/计算机-编译原理-0460-活跃区间分析`

## R01

活跃区间分析（Live Interval Analysis）用于估计每个变量在指令线性序中的“存活范围”，通常写成区间 `[start, end]`。  
它是线性扫描寄存器分配（Linear Scan Register Allocation）的核心前置步骤：如果两个变量的活跃区间重叠，就不能安全地映射到同一物理寄存器。

## R02

本条目实现一个可运行的最小 MVP，覆盖三件事：

1. 在简化 CFG 上做经典活跃变量分析（块级数据流）；
2. 回推到指令级 `live_before/live_after`；
3. 基于线性化指令位置生成每个变量的活跃区间与分段信息。

脚本运行后会直接打印结果，无需交互输入。

## R03

使用的核心定义如下：

- `live_out[B] = ⋃ live_in[S]`，`S` 是 `B` 的后继块；
- `live_in[B] = use[B] ∪ (live_out[B] - def[B])`；
- 指令级逆向方程：`live_before[i] = use[i] ∪ (live_after[i] - def[i])`；
- 活跃区间：变量在全部相关程序点上的最小覆盖区间 `[min_pos, max_pos]`；
- 活跃分段：把该变量出现的离散程序点压缩为若干连续段，便于观察“区间洞”。

## R04

`demo.py` 的算法流程：

1. 构造示例 CFG（`entry / then / else / merge`）；
2. 计算每个基本块的 `use/def`；
3. 迭代求块级 `live_in/live_out` 直至不动点；
4. 对每个块逆序扫描，得到指令级 `live_before/live_after`；
5. 将所有指令按固定块顺序线性化并编号；
6. 汇总每个变量的程序点集合，构造区间与分段；
7. 输出峰值寄存器压力与基本一致性校验。

## R05

为什么要先做块级再做指令级：

- 仅靠单块内逆向扫描无法处理跨分支传播；
- 块级数据流先给出每个块出口的活跃集合；
- 指令级回推以该出口集合作为边界条件，才能得到正确的每条指令活跃信息。

这也是多数编译器后端的标准分层做法。

## R06

示例 IR 结构（简化三地址风格）：

- `entry`：定义 `p/q/x/cond`，再条件跳转到 `then/else`；
- `then`：计算 `t` 与 `y`；
- `else`：计算 `u` 与 `y`；
- `merge`：使用 `y` 与 `x` 计算 `z`，最后 `ret z`。

该结构能覆盖“分支 + 合流”场景，展示变量 `x`、`y` 在不同路径上的生命周期。

## R07

复杂度分析（设基本块数 `B`、指令数 `N`、变量数 `V`）：

- 块级数据流：`O(iter * (B + E) * V)`，`E` 为 CFG 边数；
- 指令级回推：`O(N * V)`（集合操作按变量规模计）；
- 区间构造：`O(N * V)`；
- 空间主要是活跃集合与区间映射，约 `O((B + N) * V)`。

在教学规模 IR 上，这一开销很小且可解释性高。

## R08

正确性直觉：

- 数据流方程保证每个块入口/出口的活跃集合满足全图一致性；
- 指令级方程保证“当前指令前活跃 = 当前使用 + 后继活跃去掉本条定义”；
- 区间来自指令级活跃与定义/使用点的并集，因此不会漏掉真实生命周期端点；
- 若变量从未被定义或使用，不会被纳入区间集合（避免伪变量）。

## R09

边界与异常处理：

- 空区间集合会触发断言（防止分析链路失效）；
- 区间覆盖集合必须与程序中出现过的变量集合一致，否则抛错；
- 每个区间必须满足 `start <= end`；
- 示例无交互输入，运行结果稳定可复现。

## R10

`demo.py` 模块职责：

- `Instruction/BasicBlock/LineRecord/Interval`：数据模型；
- `build_sample_cfg`：构造演示 CFG；
- `block_use_def`：计算块级 `use/def`；
- `liveness_analysis`：块级不动点求解；
- `instruction_liveness`：块内指令级回推；
- `linearize_records`：线性化与位置编号；
- `live_intervals`：区间与分段构造；
- `peak_register_pressure`：峰值活跃数统计；
- `sanity_checks`：一致性检查。

## R11

运行方式：

```bash
cd Algorithms/计算机-编译原理-0460-活跃区间分析
uv run python demo.py
```

脚本会直接打印块级活跃结果、指令级活跃结果、活跃区间表和峰值寄存器压力。

## R12

输出解读重点：

- `Block Liveness`：每个块的 `use/def/in/out`；
- `Instruction-Level Liveness`：每条指令前后活跃集合；
- `Live Intervals`：
  - `start/end`：线性位置上的区间端点；
  - `span`：区间长度；
  - `holes`：分段数量减一（越大说明区间内有更多不连续空洞）；
  - `segments`：每个连续活跃片段。

## R13

验证策略：

1. 结构验证：区间变量集合必须等于 IR 中实际出现变量集合；
2. 端点验证：每个区间必须满足有序端点；
3. 运行验证：脚本末尾输出 `Sanity checks passed.` 才算通过。

这能快速发现“数据流没收敛/回推漏变量/区间构造漏点”等常见错误。

## R14

常见误区与规避：

1. 只做线性代码活跃分析，忽略 CFG 分支。  
   - 规避：先做块级方程并迭代到不动点。
2. 仅用定义和使用点构造区间，忽略中间传播。  
   - 规避：把 `live_before/live_after` 一并纳入区间程序点。
3. 只输出 `[start,end]`，看不到区间空洞。  
   - 规避：额外输出 `segments` 与 `holes`。

## R15

在编译器中的位置关系：

- 前置：IR 构建、CFG 建立；
- 本条目：活跃分析 + 区间抽取；
- 后续：线性扫描寄存器分配（按区间起点排序，维护 active 集合）；
- 再后续：必要时 spill/reload 与指令重写。

因此“活跃区间分析”是寄存器分配链路中的关键中间层。

## R16

可扩展方向：

- 支持 SSA 形式并处理 `phi` 的特殊活跃传播；
- 区间构造从“单区间”升级为“多区间（split interval）”；
- 加入 spill 代价估计（使用频次、循环深度加权）；
- 接入线性扫描分配器，形成端到端可执行后端原型。

## R17

交付物说明：

- `README.md`：R01-R18 全部填写完毕；
- `demo.py`：可直接运行的活跃区间分析 MVP；
- `meta.json`：保留并保持任务元信息一致（UID/学科/子类/源序号/目录）。

目录自包含，可直接交给验证脚本执行。

## R18

`demo.py` 源码级算法流程（9 步）：

1. `build_sample_cfg` 构造四个基本块的简化三地址 IR 与 CFG 后继关系。  
2. `block_use_def` 扫描每个块内指令，按“先用后定才算 use”规则得到 `use/def`。  
3. `liveness_analysis` 根据 `live_out = succ.live_in` 与 `live_in = use ∪ (out - def)` 迭代到不动点。  
4. `instruction_liveness` 以块级 `live_out` 为边界，自底向上回推每条指令的 `live_before/live_after`。  
5. `linearize_records` 按固定块顺序给全部指令分配线性位置 `pos`，形成统一程序点坐标。  
6. `live_intervals` 将每个变量在 `defs/uses/live_before/live_after` 中出现的位置汇总为点集。  
7. `build_segments` 把点集压缩成连续片段，并得到总区间 `[start, end]` 与 `holes`。  
8. `peak_register_pressure` 统计各程序点 `|live_before|`，给出峰值寄存器压力与位置。  
9. `sanity_checks` 校验区间集合与变量集合一致且端点合法，全部通过后打印成功信息。
