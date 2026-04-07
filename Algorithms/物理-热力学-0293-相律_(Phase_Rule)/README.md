# 相律 (Phase Rule)

- UID: `PHYS-0290`
- 学科: `物理`
- 分类: `热力学`
- 源序号: `293`
- 目标目录: `Algorithms/物理-热力学-0293-相律_(Phase_Rule)`

## R01

相律（Gibbs Phase Rule）用于刻画多相平衡系统在给定约束下可独立调节变量的数量（自由度）。

本条目采用广义表达式：

`F = C - P + 2 - R - M - S`

其中：

- `F`：自由度（variance）
- `C`：独立组分数
- `P`：共存相数
- `R`：独立化学反应数
- `M`：被固定的强度变量个数（如固定 `T`、固定 `p`）
- `S`：额外独立约束数量

## R02

MVP 目标是给出一个可运行、可审计的“相律计算器”，而不是完整相图求解器：

1. 对多个确定性场景计算 `F`；
2. 判别平衡可行性（`F >= 0` 可行，`F < 0` 不可行）；
3. 给出最大可共存相数 `P_max`；
4. 用纯物质与二元体系扫相数表，验证经典结论（如纯水三相点 `F=0`）。

## R03

`demo.py` 的输入输出（全内置、无交互）：

- 输入（脚本内置）：
1. 场景列表（`C/P/R` 与固定 `T/p` 约束）
2. 两组相数扫描配置：`C=1` 与 `C=2, fixed_pressure=True`

- 输出：
1. 场景汇总表：`F`、variance 分类、可行性、`P_max`；
2. 相数扫描表：展示 `P` 从可行区进入不可行区时 `F` 的变化；
3. 断言通过信息 `All checks passed.`。

## R04

核心数学关系：

1. 广义相律：
`F = C - P + 2 - R - M - S`

2. 非反应、无额外约束系统（`R=S=0`）：
`F = C - P + 2 - M`

3. 常见特例：
- 全开放（`M=0`）：`F = C - P + 2`
- 定压（`M=1`）：`F = C - P + 1`
- 定温定压（`M=2`）：`F = C - P`

4. 最大共存相数来自 `F >= 0`：
`P_max = C + 2 - R - M - S`

## R05

算法高层流程：

1. 用 `PhaseRuleScenario` 描述热力学场景；
2. 对每个场景计算 `F`、variance 分类与 `P_max`；
3. 使用 `enumerate_phase_counts` 对给定 `C` 自动生成 `P=1..P_upper` 扫描表；
4. 输出场景汇总和扫描结果；
5. 通过断言校验经典结果与边界行为。

## R06

正确性与实现对应关系：

- 公式落地：`phase_rule_degrees_of_freedom` 直接实现 `F = C - P + 2 - R - M - S`；
- `P_max` 推导：`max_coexisting_phases` 由 `F >= 0` 直接解得；
- 可行性判定：`analyze_scenario` 和扫描表中统一使用 `F >= 0`；
- 分类语义：`classify_variance` 将 `F` 映射到 invariant/univariant/bivariant/multivariant/infeasible。

## R07

复杂度分析：

设场景数为 `N`、相数扫描上限为 `K`。

- 场景评估：`O(N)` 时间，`O(N)` 空间；
- 相数扫描：`O(K)` 时间，`O(K)` 空间；
- 整体复杂度：`O(N + K)` 时间，`O(N + K)` 空间。

本 MVP 不含迭代求根或优化器，复杂度透明、可预测。

## R08

边界与异常处理：

- `components`、`phases` 必须为正整数；
- `independent_reactions`、`fixed_intensive`、`extra_constraints` 必须为非负整数；
- 输入不满足约束时抛出 `TypeError/ValueError`；
- 当 `F < 0` 时标记为 `infeasible`，明确表示该相数组合在给定约束下不可平衡。

## R09

MVP 取舍说明：

- 只做“相律计数逻辑”，不做完整相图构建；
- 不引入 EOS、活度系数模型、化学势联立方程；
- 优先保证结论可解释、代码可追溯；
- 工具栈保持最小：`numpy + pandas`。

## R10

`demo.py` 主要函数职责：

- `phase_rule_degrees_of_freedom`：计算自由度 `F`；
- `max_coexisting_phases`：计算 `P_max`；
- `classify_variance`：自由度语义化分类；
- `analyze_scenario`：单场景汇总；
- `enumerate_phase_counts`：对 `P` 扫描并构建 DataFrame；
- `main`：组织场景、打印结果、执行断言验收。

## R11

运行方式：

```bash
cd Algorithms/物理-热力学-0293-相律_(Phase_Rule)
uv run python demo.py
```

脚本会直接输出表格并执行断言，无需输入参数。

## R12

输出字段说明：

- `C/P/R`：组分数、相数、独立反应数；
- `M_fixed`：固定强度变量数量（`T/p` 固定计数）；
- `S_extra`：额外独立约束数量；
- `F`：自由度；
- `variance`：自由度分类；
- `equilibrium_possible`：是否满足 `F >= 0`；
- `P_max_from_rule`：由相律推导的最大可共存相数。

## R13

脚本内最小验收项（断言）：

1. 纯物质三相点：`phase_rule_degrees_of_freedom(1,3) == 0`；
2. 纯物质两相且定压：`phase_rule_degrees_of_freedom(1,2,fixed_intensive=1) == 0`；
3. 纯物质最大共存相数：`max_coexisting_phases(1) == 3`；
4. 二元定压最大共存相数：`max_coexisting_phases(2,fixed_intensive=1) == 3`；
5. 纯物质 `P=4` 时 `F=-1` 且不可行。

## R14

关键参数与调参建议：

- `independent_reactions (R)`：反应数增加会降低 `F`；
- `fixed_temperature/fixed_pressure -> M`：固定强度变量越多，可调自由度越少；
- `extra_constraints (S)`：用于建模组成路径或工艺锁定条件；
- 若用于工程前筛选，可先快速枚举 `P` 再进入高成本模型求解。

## R15

与“黑盒相图软件”方式对比：

- 黑盒工具通常直接给结果点，解释链条较弱；
- 本 MVP 明确拆分为公式计算、分类、可行性判断与上界推导；
- 适合作为相平衡建模前的先验可行性检查层。

## R16

适用场景：

- 热力学教学（相律、三相点、变量约束）；
- 工程建模前快速判断相数是否可能；
- 反应-相平衡问题的第一层可行性筛查。

不适用场景：

- 需要定量相组成和相分率计算；
- 需要真实物性模型（EOS/活度模型）与实验回归；
- 强耦合非平衡过程的动态模拟。

## R17

可扩展方向：

- 接入真实物性模型后自动生成“相律 + 物性”联合校验；
- 将 `R` 与 `S` 从人工输入扩展为从反应网络自动提取；
- 增加对批量工况表的向量化评估与 CSV 报告导出；
- 在相图计算前自动做“不可行工况剪枝”。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 构造 `PhaseRuleScenario` 列表，定义纯物质、二元体系和反应体系的代表场景。  
2. `analyze_scenario` 读取每个场景的 `C/P/R/M/S`，调用 `phase_rule_degrees_of_freedom` 计算 `F`。  
3. 同一函数调用 `max_coexisting_phases` 推导 `P_max = C + 2 - R - M - S`。  
4. `classify_variance` 把 `F` 映射为 `invariant/univariant/bivariant/multivariant/infeasible`。  
5. `enumerate_phase_counts` 用 `numpy.arange` 扫描 `P`，批量计算 `F(P)`，生成可行-不可行分区表。  
6. `pandas.DataFrame` 组织场景汇总和扫描表，并在 `main` 中格式化打印。  
7. 断言检查经典结论（如纯水三相点 `F=0`、纯水 `P=4` 不可行）验证实现正确性。  
8. 所有断言通过后输出 `All checks passed.`，形成可重复的最小验证闭环。  

补充：脚本仅使用 `numpy/pandas` 做数组与表格处理；相律公式本身在源码中显式实现，不存在“第三方库一键黑盒求解”。
