# 卡诺循环 (Carnot Cycle)

- UID: `PHYS-0282`
- 学科: `物理`
- 分类: `热力学`
- 源序号: `285`
- 目标目录: `Algorithms/物理-热力学-0285-卡诺循环_(Carnot_Cycle)`

## R01

卡诺循环（Carnot Cycle）是理想可逆热机循环，由四个准静态过程组成：

1. `1 -> 2`：高温热源 `Th` 下等温膨胀（吸热）；
2. `2 -> 3`：绝热膨胀（温度从 `Th` 降到 `Tc`）；
3. `3 -> 4`：低温热源 `Tc` 下等温压缩（放热）；
4. `4 -> 1`：绝热压缩（温度从 `Tc` 升回 `Th`）。

在理想可逆条件下，热效率只由热源温度决定：

`eta = 1 - Tc / Th`。

本条目给出一个可运行 MVP：直接由理想气体与可逆过程公式计算四状态点、热量、功与效率，并用断言验证卡诺循环的核心等式。

## R02

工程与教学意义：

- 给出热机效率的理论上限参考，作为方案评估基线；
- 用最小模型展示“效率只依赖热源温度，而不依赖工质细节”；
- 把热力学概念（等温热量、绝热关系、克劳修斯等式）落成可审计代码。

本 MVP 目标是“公式到代码一一映射”，不是高保真 CFD/传热仿真。

## R03

`demo.py` 输入输出（全部为脚本内置样例，无交互）：

- 输入（`CarnotCase`）：
1. `th`, `tc`：高低温热源温度（K）；
2. `gamma`：`Cp/Cv`（需 `>1`）；
3. `n_mol`：物质的量（mol）；
4. `v1`：状态 1 体积；
5. `r_iso = V2/V1`：高温等温膨胀比（`>1`）。
- 输出：
1. 四状态点 `T, V, P`；
2. `Q_hot`, `Q_cold_mag`, `W_net`；
3. `eta_actual` 与 `eta_carnot`；
4. `clausius_integral = Q_hot/Th - Q_cold_mag/Tc` 与 `entropy_generation`；
5. 断言通过后打印 `All checks passed.`。

## R04

MVP 用到的核心关系（理想气体、可逆过程）：

1. 状态方程：
`pV = nRT`。

2. 可逆绝热关系：
`T * V^(gamma-1) = const`。

3. 等温过程热量（理想气体）：
`Q = nRT * ln(V_out / V_in)`。

4. 卡诺循环净功与效率：
`W_net = Q_hot - Q_cold_mag`，  
`eta_actual = W_net / Q_hot`，  
`eta_carnot = 1 - Tc/Th`。

5. 可逆循环克劳修斯等式：
`Q_hot/Th - Q_cold_mag/Tc = 0`（数值上允许浮点容差）。

## R05

复杂度（单个循环样例）：

- 状态与热量计算均为常数次代数运算；
- 时间复杂度：`O(1)`；
- 空间复杂度：`O(1)`（若把结果存表，则为样例数线性 `O(N)`）。

本条目重点在物理关系验证，不涉及迭代优化或网格离散求解。

## R06

`demo.py` 的三个最小闭环：

- 闭环 A（状态闭环）：
  `build_cycle_states` 用等温 + 绝热公式生成 1-4 四个状态点。
- 闭环 B（能量闭环）：
  `evaluate_carnot_cycle` 计算 `Q_hot/Q_cold_mag/W_net/eta`。
- 闭环 C（可逆性闭环）：
  `main` 断言 `eta_actual == eta_carnot` 且 `clausius_integral ~= 0`。

## R07

优点：

- 全流程透明：没有调用热力学黑盒求解器；
- 直接呈现卡诺循环四过程对应的数学表达；
- 通过多样例断言验证“效率只依赖 `Th/Tc`”。

局限：

- 工质限定为理想气体；
- 过程默认准静态可逆，不含摩擦、有限温差传热等不可逆因素；
- 不处理真实设备中的压降、泄漏、换热器效率与动态控制。

## R08

前置知识与环境：

- 热力学第一/第二定律、理想气体状态方程、可逆绝热关系；
- Python `>=3.10`；
- 依赖：`numpy`、`pandas`。

运行命令：

```bash
cd Algorithms/物理-热力学-0285-卡诺循环_(Carnot_Cycle)
uv run python demo.py
```

## R09

适用场景：

- 热机课程中做卡诺循环定量演示；
- 作为更复杂热机模型（Rankine/Brayton 等）的理论上限基准；
- 在算法管线中作为“热力学一致性单元测试”模板。

不适用场景：

- 需要真实工质性质库（如水蒸气表）；
- 需要不可逆传热/流动损失模型；
- 需要实验数据反演和参数辨识。

## R10

正确性直觉（为何该实现能表达卡诺循环核心性质）：

1. 等温段热量由 `nRT ln(V_ratio)` 决定，直接给出吸热与放热；
2. 绝热段用 `T*V^(gamma-1)=const` 连接高低温等温线，形成闭合循环；
3. 两段等温过程的体积比满足 `V2/V1 = V3/V4`，因此  
   `Q_cold_mag / Q_hot = Tc / Th`；
4. 代入效率定义得到  
   `eta = 1 - Q_cold_mag/Q_hot = 1 - Tc/Th`；
5. 对可逆循环，`Q_hot/Th - Q_cold_mag/Tc = 0`，对应克劳修斯等式。

脚本中的断言分别对应以上第 3-5 条。

## R11

数值稳定性策略：

- 对 `Th/Tc/gamma/n_mol/v1/r_iso` 做严格正值与顺序检查；
- 统一使用 `float64`（`numpy` 默认双精度）；
- 比较等式时使用 `tol=1e-10`，避免浮点舍入误判；
- 对同热源组样例做聚合断言，防止单例偶然通过。

## R12

关键参数与影响：

- `th`, `tc`：决定理论效率上限 `1 - tc/th`；
- `r_iso`：改变热量规模和净功大小，但不改变效率；
- `gamma`：影响绝热段体积变化与状态点位置，但不改变理想卡诺效率；
- `n_mol`、`v1`：缩放状态与能量量级，不改变效率表达式。

调参建议：

- 若要观察“效率不变、功变化”，固定 `Th/Tc` 改 `r_iso`；
- 若要观察温差影响，固定 `r_iso` 改 `Th/Tc`；
- 若要比较工质模型差异，可在理想假设下先改 `gamma` 做敏感性分析。

## R13

- 近似比保证：N/A（非近似优化算法）。
- 随机成功率保证：N/A（无随机采样，完全确定性计算）。

可验证保证（在当前模型假设内）：

- 每个样例都应满足 `eta_actual` 与 `eta_carnot` 在容差内相等；
- 每个样例都应满足 `clausius_integral` 在容差内为零；
- 同一对热源 `Th/Tc` 下，不同 `r_iso` 与 `gamma` 的效率应一致。

## R14

常见失效模式：

1. 温度误用摄氏度而非开尔文；
2. 把 `r_iso` 设为 `<=1`，导致“等温膨胀”不成立；
3. `gamma <= 1` 使绝热关系指数无意义；
4. 体积单位不一致导致压力数值错误；
5. 将不可逆真实过程误套到本可逆理想模型并据此做工程结论。

## R15

扩展方向：

- 引入不可逆项（有限温差传热、机械摩擦）并比较效率退化；
- 增加 `P-V`、`T-S` 曲线绘图；
- 接入真实工质性质库，替代理想气体假设；
- 将卡诺循环作为约束上限，嵌入热机参数优化任务；
- 加入批量 CSV 输入，做多工况自动评估与报告生成。

## R16

相关主题：

- 卡诺定理（效率上界）；
- 克劳修斯不等式与熵产生；
- 可逆/不可逆过程判据；
- 热机循环对比：Otto、Diesel、Brayton、Rankine。

## R17

`demo.py` MVP 功能清单：

- 定义 `CarnotCase` 与输入合法性检查；
- 按四过程显式计算状态点 `1,2,3,4`；
- 计算 `Q_hot/Q_cold_mag/W_net/eta`；
- 计算可逆循环 `clausius_integral` 与 `entropy_generation`；
- 输出指标表与状态点表（`pandas.DataFrame`）；
- 通过断言验证卡诺循环关键等式与同热源效率一致性。

运行方式：

```bash
cd Algorithms/物理-热力学-0285-卡诺循环_(Carnot_Cycle)
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程（8 步）：

1. `build_demo_cases` 构造多组确定性参数，覆盖“同热源不同 `r_iso/gamma`”与“不同温差”两类对照。  
2. `CarnotCase.validate` 先检查物理前提：`Th>Tc>0`、`gamma>1`、`n_mol>0`、`v1>0`、`r_iso>1`。  
3. `build_cycle_states` 先算 `V2=r_iso*V1`，再用 `adiabatic_end_volume` 根据 `T*V^(gamma-1)=const` 求 `V3`、`V4`。  
4. `ideal_gas_pressure` 用 `p=nRT/V` 计算四状态压力，得到完整状态点字典。  
5. `evaluate_carnot_cycle` 用等温公式 `Q=nRT ln(V_out/V_in)` 计算 `Q_hot` 与 `Q_cold_mag`。  
6. 同函数内继续计算 `W_net`、`eta_actual`、`eta_carnot`，并显式给出 `eta_gap`。  
7. 继续计算 `clausius_integral = Q_hot/Th - Q_cold_mag/Tc` 与 `entropy_generation`，用于验证可逆性。  
8. `main` 汇总并打印结果表，然后断言：每例 `eta_actual==eta_carnot`、`clausius_integral≈0`，且同热源样例效率完全一致。  

说明：实现只使用 `numpy/pandas` 做数值与表格输出，热力学关系均在源码中显式展开，无黑盒循环求解器。  
