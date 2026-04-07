# 卡诺定理 (Carnot's Theorem)

- UID: `PHYS-0283`
- 学科: `物理`
- 分类: `热力学`
- 源序号: `286`
- 目标目录: `Algorithms/物理-热力学-0286-卡诺定理_(Carnot's_Theorem)`

## R01

卡诺定理给出热机效率的上界规律，核心有两句：

1. 在相同高温热源 `Th` 与低温热源 `Tc` 之间工作的任意热机，其效率都不可能超过可逆卡诺热机；
2. 在同一对热源之间，所有可逆热机效率相同，仅由 `Th` 与 `Tc` 决定，而与工质细节无关。

在理想条件（温度用开尔文）下：

`eta_carnot = 1 - Tc / Th`，并且任何可实现热机都满足 `eta_actual <= eta_carnot`。

本条目把该定理写成一个可执行的“效率边界检查器”。

## R02

该定理在工程上是一个“不可突破上限”：

- 用于热机方案早期筛选，快速识别物理上不可能的高效率宣称；
- 用于判定某方案离理论极限还有多远（`eta_margin = eta_carnot - eta_actual`）；
- 用于教学中解释“可逆极限”“不可逆损失”“违反第二定律”三类情形。

MVP 的目标是可审计性而不是复杂仿真：输入若干热机工况，输出是否违反卡诺上界。

## R03

`demo.py` 的输入输出（均为脚本内置样例，无交互）：

- 输入：
1. `EngineCase` 列表，每个样例包含 `Th, Tc, q_in, w_out, claimed_reversible`；
2. 数值判定容差 `tol`。
- 输出：
1. `eta_actual = w_out / q_in`；
2. `eta_carnot = 1 - Tc/Th`；
3. `eta_margin = eta_carnot - eta_actual`；
4. 分类结果：`reversible_consistent` / `irreversible_allowed` / `violates_carnot_bound` / `reversible_claim_inconsistent`；
5. 断言全部通过后打印 `All checks passed.`。

## R04

MVP 使用的核心公式：

1. 热机效率定义：
`eta_actual = W_out / Q_in`，其中 `Q_in > 0`。

2. 卡诺效率：
`eta_carnot = 1 - Tc / Th`，要求 `Th > Tc > 0`。

3. 卡诺定理上界判据：
`eta_actual <= eta_carnot`。

4. 误差容差后的程序判据：
若 `eta_actual - eta_carnot > tol`，判为违反上界。

5. 热量闭合（用于报告）：
`Q_out = Q_in - W_out`。

## R05

设样例数为 `N`：

- 每个样例计算复杂度：`O(1)`；
- 全量评估复杂度：`O(N)`；
- 空间复杂度：`O(N)`（保存结果表）；
- 断言与分类均为线性时间扫描。

该 MVP 的重点是物理约束判定，不涉及高维数值优化。

## R06

`demo.py` 的三个最小闭环：

- 闭环 A：`EngineCase.validate` 执行物理可行输入检查（`Th>Tc>0`、`q_in>0`、`0<=w_out<q_in`）。
- 闭环 B：`actual_efficiency` 与 `carnot_efficiency` 计算实测效率与理论上界。
- 闭环 C：`classify_case` 根据 `eta_actual - eta_carnot` 的符号和 `claimed_reversible` 给出可解释结论。

## R07

优点：

- 公式与代码一一映射，便于审计与复核；
- 不依赖黑盒热力学求解器，透明度高；
- 通过固定样例和断言保证回归稳定性。

局限：

- 不从状态方程推导 `Q_in`、`W_out`，而是把它们当作输入数据；
- 不模拟真实传热、摩擦、压降等过程细节；
- 只验证卡诺上界，不进行多目标设计优化。

## R08

前置知识与运行环境：

- 热力学第二定律与热机效率定义；
- Python `>=3.10`；
- 依赖：`numpy`、`pandas`。

运行命令：

```bash
cd Algorithms/物理-热力学-0286-卡诺定理_(Carnot's_Theorem)
uv run python demo.py
```

## R09

适用场景：

- 对热机实验/仿真结果做快速物理一致性检查；
- 审核“超高效率”宣传数据的基础可信度；
- 作为更复杂循环设计器的前置约束过滤。

不适用场景：

- 需要求解完整 P-V-T 状态轨迹；
- 需要传热传质耦合、相变或化学反应模型；
- 需要实验级不确定度传播与参数识别。

## R10

正确性直觉：

1. 若某热机效率超过可逆卡诺热机，则可构造与第二定律矛盾的复合装置；
2. 因而真实可行热机必须满足 `eta_actual <= eta_carnot`；
3. 对同一 `Th, Tc`，可逆过程不存在额外熵产生，效率只由热源温度比决定；
4. 所以两个不同“工质细节”的可逆样例，应得到相同效率且等于 `1 - Tc/Th`。

脚本中的断言正对应这三条逻辑。

## R11

数值稳定性策略：

- 对输入温度做严格正值与顺序检查，避免非物理参数；
- 使用 `float64`（Python `float` + `numpy`）；
- 采用容差 `tol` 处理浮点舍入误差；
- 用“可逆等号”“不可逆小于”“违规大于”三类样例覆盖全部分支。

## R12

关键参数：

- `Th`, `Tc`：热源温度，直接决定卡诺上界；
- `q_in`, `w_out`：决定实测效率 `eta_actual`；
- `tol`：分类阈值，影响等号附近判定。

调参建议：

- 理论构造样例可用 `1e-10` 或更小；
- 实验数据可把 `tol` 设为测量误差同量级；
- 若跨量纲数据混合，先归一化再比较效率差值。

## R13

- 近似比保证：N/A（非优化近似算法）。
- 随机成功率保证：N/A（无随机采样过程）。

可验证保证：

- 若样例是可逆并满足理想关系，分类应为 `reversible_consistent`；
- 若样例效率低于上界，分类应为 `irreversible_allowed`；
- 若样例效率高于上界，分类必为 `violates_carnot_bound`。

## R14

常见失效模式：

1. 温度误用摄氏度而非开尔文；
2. 把输入功和输出功的符号约定混淆；
3. 以局部瞬时效率代替整循环效率；
4. 忽略测量误差，导致等号附近误判；
5. 使用与当前热源不一致的 `Th/Tc` 计算理论上界。

## R15

扩展方向：

- 增加制冷机/热泵版本，验证 `COP` 的卡诺上限；
- 接入实验 CSV 数据批处理，自动给出违规报告；
- 将 `eta_actual <= eta_carnot` 作为优化约束集成到设计搜索；
- 加入不确定度区间，输出“违规概率”而非硬阈值。

## R16

相关主题：

- 克劳修斯不等式；
- Kelvin-Planck 表述与第二定律等价形式；
- 熵产生与不可逆损失；
- 热机效率上界、制冷循环 `COP` 上界。

## R17

`demo.py` MVP 功能清单：

- 定义 `EngineCase` 数据结构与输入校验；
- 计算 `eta_actual` 与 `eta_carnot`；
- 计算 `q_out` 与 `eta_margin` 作为解释量；
- 输出 `pandas.DataFrame` 结果表；
- 构造四个固定样例覆盖主要物理分支；
- 通过断言检查卡诺定理两条核心结论并验证违规分支。

运行方式：

```bash
cd Algorithms/物理-热力学-0286-卡诺定理_(Carnot's_Theorem)
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程（8 步）：

1. `EngineCase` 定义热机样例字段，并在 `validate` 中执行 `Th>Tc>0`、`q_in>0`、`0<=w_out<q_in` 检查。  
2. `carnot_efficiency` 根据 `1 - Tc/Th` 计算理论效率上界。  
3. `actual_efficiency` 用 `w_out/q_in` 计算样例实际效率。  
4. `classify_case` 比较 `eta_actual` 与 `eta_carnot`：超上界则判 `violates_carnot_bound`，可逆且贴合等号判 `reversible_consistent`，其余按规则归类。  
5. `evaluate_case` 汇总每个样例的 `q_out`、`eta_margin`、分类等结构化指标。  
6. `build_demo_cases` 构造四个确定性样例：两个同温区可逆样例、一个可行不可逆样例、一个故意违规样例。  
7. `main` 批量计算并打印 `DataFrame`，形成可直接审阅的结果表。  
8. `main` 末尾断言两类定理结论：同温区可逆效率相等且等于卡诺值；违规样例必须被识别为超上界。  

说明：实现未调用热力学黑盒函数，公式到判定逻辑全部在源码中显式展开。
