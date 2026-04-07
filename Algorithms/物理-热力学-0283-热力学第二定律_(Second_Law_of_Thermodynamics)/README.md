# 热力学第二定律 (Second Law of Thermodynamics)

- UID: `PHYS-0280`
- 学科: `物理`
- 分类: `热力学`
- 源序号: `283`
- 目标目录: `Algorithms/物理-热力学-0283-热力学第二定律_(Second_Law_of_Thermodynamics)`

## R01

热力学第二定律给出“过程方向”和“可实现性”约束。对热机循环，本条目采用以下等价表述：

1. 熵增表述（宇宙）：`ΔS_universe >= 0`；
2. Clausius 不等式（循环）：`∮(δQ/T) <= 0`；
3. 效率上限：`eta <= eta_Carnot = 1 - Tc/Th`（`Th > Tc > 0`）。

其中 `Th`、`Tc` 分别是高温/低温热源温度（K），`eta` 为热机效率。

## R02

该定律在工程建模中的核心作用：

- 判定“看起来能量守恒但物理上不可能”的方案（例如超卡诺效率）；
- 给制冷、热机、换热网络提供理论上限；
- 提供可直接编码的可行性约束（熵产生非负）。

本 MVP 目标是把第二定律转成可运行的最小判定器，而不是构建复杂循环仿真器。

## R03

`demo.py` 输入输出（脚本内置参数，无交互）：

- 输入：
1. 多个热机案例 `HeatEngineCase(name, Th, Tc, Qh, eta_actual)`；
2. 每个案例都给定吸热量 `Qh` 和宣称效率 `eta_actual`。
- 输出：
1. 每个案例的 `W_out`、`Qc`、`ΔS_universe`、`∮δQ/T`；
2. 解析 Carnot 上限和数值求根得到的上限对照；
3. 是否满足第二定律的布尔判定；
4. 断言通过后打印 `All checks passed.`。

## R04

MVP 使用的核心关系：

1. 热机能量分配：
`W_out = eta * Qh`，`Qc = Qh - W_out = Qh * (1 - eta)`。

2. 宇宙熵变（系统完成循环，`ΔS_engine = 0`）：
`ΔS_universe = -Qh/Th + Qc/Tc`。

3. Clausius 循环积分（以“流入工质的热量”为正）：
`∮(δQ/T) = Qh/Th - Qc/Tc = -ΔS_universe <= 0`。

4. 效率上限（由 `ΔS_universe >= 0` 推得）：
`eta <= 1 - Tc/Th`。

5. 数值求根验证：
求 `f(eta) = -1/Th + (1-eta)/Tc = 0` 的根，得到临界效率。

## R05

复杂度（单次脚本运行）：

- 每个案例的代数评估为 `O(1)`；
- `scipy.optimize.brentq` 求根在固定精度下迭代步数很小，记为 `K`，则每案例 `O(K)`；
- 总时间复杂度 `O(N * K)`，`N` 为案例数；
- 空间复杂度 `O(N)`（用于输出汇总表）。

## R06

`demo.py` 的最小算法闭环：

- 闭环 A（物理量计算）：由 `Qh` 和 `eta` 计算 `W_out/Qc`；
- 闭环 B（第二定律判定）：用 `ΔS_universe` 与 `∮δQ/T` 双重等价判据交叉验证；
- 闭环 C（上限一致性）：解析 `eta_Carnot` 与数值求根上限 `eta_limit_numeric` 互相校验。

## R07

优点：

- 公式与代码一一对应，审计路径短；
- 既可展示“可逆极限”也能检测“超卡诺违规”；
- 用统一 DataFrame 输出，便于后续批量校验。

局限：

- 仅覆盖两热源热机模型，未包含多热源或再生循环细节；
- 不涉及传热动力学、流动压降和控制器；
- `Th`/`Tc` 视为定温热源，忽略热容有限引起的温度漂移。

## R08

前置知识与环境：

- 基本热力学、熵与可逆过程概念；
- Python `>=3.10`；
- 依赖：`numpy`、`pandas`、`scipy`。

运行命令：

```bash
cd Algorithms/物理-热力学-0283-热力学第二定律_(Second_Law_of_Thermodynamics)
uv run python demo.py
```

## R09

适用场景：

- 教学演示第二定律与 Carnot 上限的数值验证；
- 对热机方案做“物理可行性”快速筛查；
- 作为更复杂循环仿真的前置单元测试。

不适用场景：

- 需要真实工质性质库的高精度设备设计；
- 强非平衡、瞬态传热、多相流耦合问题；
- 需要与实验控制系统实时联动的在线优化。

## R10

正确性直觉：

1. 热机循环后工质回到初态，因此工质自身熵变为 0；
2. 全部不可逆性体现在“宇宙熵增”上，即 `ΔS_universe >= 0`；
3. 若某方案宣称效率过高，会导致 `Qc` 过低，从而使 `ΔS_universe < 0`；
4. 这与 Clausius 不等式等价地表现为 `∮δQ/T > 0`；
5. 可逆极限正好对应 `ΔS_universe = 0`，即 Carnot 上限。

## R11

数值稳定性策略：

- 输入检查：`Th > Tc > 0`、`Qh > 0`、`eta >= 0`；
- 统一使用 `float64`；
- 所有物理判定都带容差 `tol`，避免把浮点舍入误差误判为定律违背；
- 对 Carnot 上限采用“解析式 + `brentq` 求根”双通道验证，降低实现错误风险。

## R12

关键参数及影响：

- `Th`（高温热源）：越高，理论效率上限越高；
- `Tc`（低温热源）：越低，理论效率上限越高；
- `Qh`（吸热量）：线性缩放 `W_out/Qc/ΔS_universe` 的绝对量级；
- `eta_actual`（实际效率）：直接决定是否越过第二定律边界。

调参建议：

- 教学演示建议固定 `Th,Tc`，仅扫 `eta_actual` 观察边界；
- 方案对比建议固定 `eta_actual`，改变 `Th/Tc` 查看可行域变化。

## R13

- 近似比保证：N/A（非优化近似算法）。
- 随机成功率保证：N/A（脚本确定性，无随机采样）。

可验证保证（在模型假设内）：

- `ΔS_universe >= 0` 与 `eta <= eta_Carnot` 判定等价；
- `∮δQ/T = -ΔS_universe` 数值一致（容差内）；
- 解析 Carnot 上限与 `brentq` 数值根一致（容差内）。

## R14

常见失效模式：

1. 温度使用摄氏度而不是开尔文；
2. 混淆 `Qh`、`Qc` 的符号约定导致熵项正负颠倒；
3. 只检查能量守恒却忽略熵约束；
4. 把可逆极限当作可长期达到的工程常态；
5. 忽略测量误差与参数不确定性，过度解读临界点附近结果。

## R15

可扩展方向：

- 增加制冷机/热泵 COP 的第二定律边界；
- 引入有限温差换热模型，显式计算局部熵产生分布；
- 扩展为多热源网络并加入优化求解；
- 接入实验数据做“热机工况自动合规审计”。

## R16

相关主题：

- 熵的克劳修斯定义与熵平衡；
- Carnot 循环；
- 火用（Exergy）分析；
- 不可逆热力学与熵产生最小化。

## R17

`demo.py` MVP 功能清单：

- 定义 `HeatEngineCase` 数据结构与输入合法性检查；
- 计算 `W_out`、`Qc`、`ΔS_universe`、`∮δQ/T`；
- 解析计算 `eta_Carnot`；
- 用 `scipy.optimize.brentq` 通过熵平衡方程求数值效率上限；
- 构建三类案例：可逆参考、不可逆可行、超卡诺违规；
- 输出 `pandas` 表格并执行断言自检。

运行方式：

```bash
cd Algorithms/物理-热力学-0283-热力学第二定律_(Second_Law_of_Thermodynamics)
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 构造多个 `HeatEngineCase`，每个包含 `Th/Tc/Qh/eta_actual`。  
2. `HeatEngineCase.validate` 先检查定义域（`Th > Tc > 0`、`Qh > 0`、`eta >= 0`），保证后续公式可用。  
3. `evaluate_case` 用 `W_out = eta*Qh` 与 `Qc = Qh*(1-eta)` 计算循环能量分配。  
4. 同函数内计算 `ΔS_universe = -Qh/Th + Qc/Tc`，并计算 `∮δQ/T = Qh/Th - Qc/Tc`。  
5. `carnot_efficiency` 给出解析效率边界 `1 - Tc/Th`。  
6. `max_efficiency_from_entropy_balance` 构造目标函数 `f(eta) = -1/Th + (1-eta)/Tc`，调用 `scipy.optimize.brentq` 在区间 `[0, 1)` 内求根。  
7. `evaluate_case` 对比 `eta_actual` 与解析/数值两种边界，并给出 `second_law_ok`。  
8. `main` 汇总成 `pandas.DataFrame`，执行断言：等价关系成立、可逆案例熵产约为 0、不可逆案例熵产为正、超卡诺案例被识别。  

补充说明：`brentq` 只是执行一维有界求根，第二定律判据函数 `f(eta)` 的物理构造、区间选择、结果解释均在源码中显式展开，没有把物理逻辑交给黑盒。
