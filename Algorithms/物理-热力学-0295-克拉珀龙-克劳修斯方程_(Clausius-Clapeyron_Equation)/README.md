# 克拉珀龙-克劳修斯方程 (Clausius-Clapeyron Equation)

- UID: `PHYS-0292`
- 学科: `物理`
- 分类: `热力学`
- 源序号: `295`
- 目标目录: `Algorithms/物理-热力学-0295-克拉珀龙-克劳修斯方程_(Clausius-Clapeyron_Equation)`

## R01

克拉珀龙-克劳修斯方程描述相平衡曲线的斜率与潜热之间的关系。常用形式有两层：

1. 一般克拉珀龙形式：
`dP/dT = L / (T * Δv)`
2. 对液-汽相变常见近似（蒸汽近似理想气体、液相比容可忽略）：
`d ln(P) / d(1/T) = -L/R`

当潜热 `L` 在温区内近似常数时，可积分为：
`ln(P) = -L/(R*T) + C`。

## R02

本条目给出一个可运行的最小 MVP，用来把公式直接变成“可估参 + 可校验”的数值流程，而不只是推导。

MVP 覆盖三件事：

- 正向：由参考点 `(T_ref, P_ref)` 和潜热 `L` 生成蒸汽压曲线；
- 反向：由 `(T, P)` 数据估计 `L`（线性化拟合 + 非线性拟合）；
- 一致性：用微分形式估计局部潜热，并做邻接温度点预测误差检查。

## R03

`demo.py` 输入输出（脚本内置，无交互）：

- 输入：
1. 参考点 `T_ref=373.15 K`、`P_ref=101325 Pa`；
2. 真实潜热 `L_true=40700 J/mol`；
3. 温度区间 `[335, 390] K` 的离散采样；
4. 确定性扰动（正弦噪声）用于模拟测量误差。
- 输出：
1. 估计结果汇总表（`L_linear`, `L_nonlinear`, `R^2`, 相对误差等）；
2. 每个温度点的测量/拟合明细表；
3. 邻接点两点积分预测误差表；
4. 全部断言通过后输出 `All checks passed.`。

## R04

MVP 使用的核心关系与程序映射：

1. 参考点积分式：
`P(T) = P_ref * exp[-L/R * (1/T - 1/T_ref)]`  
对应 `vapor_pressure_integrated`。

2. 线性化回归：
令 `x=1/T, y=ln(P)`，则 `y = m*x + b`，且 `m=-L/R`，`b=C`。  
对应 `fit_linearized_clausius_clapeyron`。

3. 非线性回归：
`P(T)=exp(-L/(R*T)+C)` 直接拟合 `L,C`。  
对应 `fit_nonlinear_clausius_clapeyron`。

4. 微分式局部潜热：
`L_local = -R * dln(P)/d(1/T)`，用数值梯度近似。  
对应 `estimate_local_latent_heat`。

5. 两点积分预测：
`P2 = P1 * exp[-L/R * (1/T2 - 1/T1)]`。  
对应 `pairwise_prediction_table`。

## R05

设采样点数为 `n`：

- 数据生成：`O(n)`；
- 线性拟合：`O(n)`；
- 非线性拟合（两参数）：迭代 `k` 次，约 `O(k*n)`；
- 局部潜热与邻接预测：`O(n)`；
- 空间复杂度：`O(n)`（保存表格列）。

默认 `n=18`，因此运行成本很低。

## R06

`demo.py` 的最小闭环：

- **闭环 A（正向建模）**：`generate_dataset` 通过积分式生成 `P_true` 并构造可控“测量值”；
- **闭环 B（反向估参）**：线性化拟合和非线性拟合各自给出潜热估计；
- **闭环 C（一致性校验）**：局部潜热统计 + 邻接点预测残差 + 断言阈值检查。

## R07

优点：

- 公式与代码一一对应，便于审查与教学；
- 同时覆盖“正向预测 + 反向估参 + 一致性检查”；
- 无黑盒物性库依赖，最小可运行。

局限：

- 假设潜热近似常数，不适合宽温区高精度任务；
- 未显式建模真实工质在高压下的非理想性；
- 合成数据是可控示例，不等同于实验测量流程。

## R08

前置知识与环境：

- 相平衡、蒸汽压、对数线性化回归；
- Python `>=3.10`；
- 依赖：`numpy`、`pandas`、`scipy`。

运行命令：

```bash
cd Algorithms/物理-热力学-0295-克拉珀龙-克劳修斯方程_(Clausius-Clapeyron_Equation)
uv run python demo.py
```

## R09

适用场景：

- 热力学课程中演示 Clausius-Clapeyron 线性化估参；
- 作为蒸汽压数据处理管线的最小测试基线；
- 对新实现的回归与导数计算模块做单元验证。

不适用场景：

- 需要跨大温区、变潜热的精细物性预测；
- 需要多组分相平衡或强非理想状态方程耦合；
- 需要直接对真实噪声实验数据做严谨统计推断。

## R10

正确性直觉：

1. 若积分公式正确，则 `ln(P)` 对 `1/T` 应接近直线；
2. 直线斜率只由 `-L/R` 决定，所以可反推出 `L`；
3. 用独立的非线性压力拟合作交叉验证，估计值应接近；
4. 微分形式算出的 `L_local` 若围绕同一均值波动，说明方程与数据一致；
5. 两点积分预测误差小，进一步说明估计的 `L` 有解释力。

## R11

数值稳定策略：

- 全程使用 `float64`；
- 强制 `T>0, P>0`，避免 `ln(P)` 非法；
- 在线性拟合与非线性拟合间交叉校验，降低单模型偶然偏差；
- 局部潜热用 `np.gradient(..., edge_order=2)`，提高导数近似稳定性；
- 断言使用相对误差阈值，而非绝对值硬编码。

## R12

关键参数：

- `latent_heat_true_j_per_mol`：合成数据的真实潜热；
- `t_ref_k`, `p_ref_pa`：积分参考点；
- `t_min_k`, `t_max_k`, `n_points`：温度采样范围与密度；
- `noise_amplitude`：测量扰动幅度；
- `latent_heat_rel_tol`、`linear_rel_error_tol`、`pairwise_rel_error_tol`：通过阈值。

调参建议：

- 若想模拟更困难估计任务，可增大 `noise_amplitude`；
- 若想减少导数离散误差，可提高 `n_points`；
- 若断言过紧导致失败，应先确认噪声与采样密度是否匹配再调阈值。

## R13

- 近似比保证：N/A（非组合优化算法）。
- 随机成功率：N/A（本脚本为确定性流程）。

可验证保证：

- 线性化拟合 `R^2` 接近 1；
- 线性/非线性两种估计得到的潜热都接近真实值；
- 邻接点积分预测误差受控在阈值内；
- 若故意破坏公式符号或单位，断言会快速失败。

## R14

常见失效模式：

1. 温度单位误用（摄氏温度直接代替 K）；
2. 压力单位不一致（Pa 与 kPa/bar 混用）；
3. 忽略 `P>0` 约束导致 `ln(P)` 无定义；
4. 在宽温区强行假设常潜热，导致系统偏差；
5. 噪声较大但采样点过少，拟合结果不稳定。

## R15

工程扩展方向：

- 引入温度相关潜热 `L(T)`（分段或多项式模型）；
- 对真实实验数据增加加权回归与不确定度区间；
- 与 Antoine 方程或更高阶蒸汽压经验式做对比；
- 增加多工质批处理与自动报告输出。

## R16

相关概念：

- Clapeyron 方程（一般形式）；
- 相变潜热与 Gibbs 自由能平衡；
- Antoine 方程与经验蒸汽压模型；
- 回归诊断指标（`R^2`、残差分析）。

## R17

`demo.py` MVP 功能清单：

- `ExperimentConfig`：实验配置与输入有效性检查；
- `vapor_pressure_integrated`：参考点积分计算蒸汽压；
- `fit_linearized_clausius_clapeyron`：线性化估计潜热；
- `fit_nonlinear_clausius_clapeyron`：非线性压力空间拟合；
- `estimate_local_latent_heat`：由微分式恢复局部潜热；
- `pairwise_prediction_table`：两点积分关系逐对验证；
- `main`：打印三类表格并执行断言。

运行方式：

```bash
cd Algorithms/物理-热力学-0295-克拉珀龙-克劳修斯方程_(Clausius-Clapeyron_Equation)
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程（8 步）：

1. `ExperimentConfig.validate` 先检查潜热、温度、压力、噪声幅度等输入边界。  
2. `generate_dataset` 用 `vapor_pressure_integrated` 生成 `P_true(T)`，并叠加确定性扰动得到 `P_measured(T)`。  
3. `fit_linearized_clausius_clapeyron` 对 `ln(P)` 与 `1/T` 做线性回归，利用斜率恢复 `L_linear`。  
4. `fit_nonlinear_clausius_clapeyron` 以 `P(T)=exp(-L/(RT)+C)` 直接做 `curve_fit`，得到 `L_nonlinear`。  
5. `estimate_local_latent_heat` 计算 `dlnP/d(1/T)` 的数值梯度，得到每个采样点的 `L_local`。  
6. `pairwise_prediction_table` 用估计潜热在相邻温度点执行两点积分预测，计算逐对相对误差。  
7. `main` 汇总并打印 summary、明细表、邻接预测表，形成可审计输出。  
8. 断言检查 `R^2`、潜热相对误差、拟合误差与两点预测误差，全部通过后输出 `All checks passed.`。
