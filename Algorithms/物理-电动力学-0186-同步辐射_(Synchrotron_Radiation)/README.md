# 同步辐射 (Synchrotron Radiation)

- UID: `PHYS-0185`
- 学科: `物理`
- 分类: `电动力学`
- 源序号: `186`
- 目标目录: `Algorithms/物理-电动力学-0186-同步辐射_(Synchrotron_Radiation)`

## R01

同步辐射描述的是相对论带电粒子在弯转轨道（典型是磁场约束的圆弧轨道）上产生的强辐射。  
其核心特征是：

- 总辐射功率随洛伦兹因子 `gamma` 快速增长（约 `gamma^4`）；
- 频谱不是单频，而是宽带连续谱；
- 频谱主峰附近由无量纲变量 `x = omega/omega_c` 的通用函数控制。

## R02

本条目关注“最小可验证算法链”，而非完整加速器工程模型：

1. 用相对论 Lienard 公式计算总辐射功率；
2. 用同步辐射通用核 `F(x)` 构造频谱形状；
3. 将频谱积分回总功率，做守恒一致性检查；
4. 对不同 `beta`（不同 `gamma`）验证标度律。

## R03

MVP 目标是把公式变成可跑、可核验的脚本：

- 计算 `P_total`、`omega_c`；
- 数值构造 `F(x) = x * int_x^inf K_{5/3}(xi) dxi`；
- 归一化后得到 `g(x)`，并定义
  `dP/domega = (P_total/omega_c) * g(omega/omega_c)`；
- 验证 `int (dP/domega) domega` 与 `P_total` 一致。

## R04

`demo.py` 使用的物理方程（SI）：

1. 洛伦兹因子  
   `gamma = 1/sqrt(1-beta^2)`

2. 圆轨道同步辐射总功率  
   `P = (q^2 c / (6*pi*epsilon0*rho^2)) * beta^4 * gamma^4`

3. 临界角频率  
   `omega_c = (3/2) * gamma^3 * c / rho`

4. 通用核函数  
   `F(x) = x * int_x^inf K_{5/3}(xi) dxi`

5. 归一化频谱形状  
   `g(x) = F(x) / int_0^inf F(x) dx`

6. 角频率谱功率  
   `dP/domega = (P/omega_c) * g(omega/omega_c)`

这样构造后，理论上应满足 `int_0^inf (dP/domega) domega = P`。

## R05

设谱网格点数为 `N`：

- `K_{5/3}` 采样：`O(N)`；
- 尾积分（反向累积梯形）：`O(N)`；
- 归一化与谱生成：`O(N)`；
- 积分与诊断：`O(N)`。

总时间复杂度 `O(N)`，空间复杂度 `O(N)`。

## R06

脚本输出四类结果：

- 核函数诊断：`int F dx` 的数值值、解析常数对比误差；
- 单案例汇总：`beta, gamma, omega_c, P_formula, P_integrated, x_peak`；
- 双案例标度：`omega_c` 的 `gamma^3` 比例检查、功率 `beta^4*gamma^4` 比例检查；
- 阈值清单与最终 `Validation: PASS/FAIL`。

## R07

优点：

- 物理链路完整：总功率、谱形、积分闭环全部可见；
- 依赖轻量：`numpy + scipy + pandas`；
- 核心逻辑在源码显式实现，无黑盒“直接给结论”。

局限：

- 只覆盖理想圆轨道、真空、单粒子模型；
- 未包含发射角分布、偏振分解、束流集体效应；
- 频谱积分上限以有限 `x_max` 近似无穷上限。

## R08

前置知识：

- 相对论动力学中的 `beta/gamma`；
- 贝塞尔函数 `K_nu` 与数值积分；
- 频谱归一化与量纲一致性检查。

运行依赖：

- Python `>= 3.10`
- `numpy`
- `scipy`
- `pandas`

## R09

适用场景：

- 同步辐射课程中的“公式到算法”演示；
- 加速器辐射模块的前置 sanity check；
- 对谱形积分和标度律做自动回归测试。

不适用场景：

- 需要完整束流动力学与机器参数耦合建模；
- 需要角-频联合谱、偏振细节、屏蔽/边界效应；
- 直接用于工程级光束线设计。

## R10

正确性直觉：

1. 高频电子弯转越剧烈，辐射功率应快速升高（`gamma^4`）；
2. `omega_c` 设定谱的特征频率尺度（`~gamma^3`）；
3. 若 `g(x)` 是正确归一化的形状函数，那么按 `dP/domega=(P/omega_c)g` 构造的谱积分必回到 `P`；
4. 因此“归一化正确 + 标度正确 + 积分闭环正确”可作为 MVP 的核心正确性证据。

## R11

数值稳定性处理：

- `x` 采用对数网格（`geomspace`），兼顾低频与高频区分辨率；
- `int_x^inf` 通过“反向累积梯形”实现，避免每个点嵌套积分；
- `x_max=40` 利用 `K_nu(x)` 的指数衰减，控制截断尾误差；
- 阈值检查都在脚本中自动判定，失败即非零退出。

## R12

关键参数与影响：

- `rho_meter`：轨道曲率半径，越小辐射越强；
- `beta`：速度参数，决定 `gamma`，强烈影响功率与临界频率；
- `x_min/x_max`：频谱覆盖范围，过窄会损失积分精度；
- `num_x`：谱离散密度，影响峰值与积分误差；
- `beta_high`：第二组对照参数，用于标度律检验。

调参建议：

- 若积分误差偏大，优先增大 `num_x`，其次提高 `x_max`；
- 若峰值位置不稳定，可适度减小 `x_min` 并提升网格密度。

## R13

保证类型说明：

- 近似比保证：N/A（不是优化问题）；
- 随机成功率：N/A（流程确定性，无随机采样）；
- 可验证保证：
  1) `int g(x) dx ~= 1`；
  2) `int (dP/domega) domega ~= P`；
  3) `omega_c` 与 `gamma^3` 标度一致；
  4) 功率与 `beta^4*gamma^4` 标度一致。

## R14

常见失效模式：

1. `beta` 接近 1 但写成 `>=1` 导致 `gamma` 发散或 NaN；
2. 频谱网格过粗导致峰值偏移、积分误差增大；
3. `x_max` 太小截断高频尾，导致总功率回积分偏低；
4. 忘记归一化 `F(x)` 直接当 `g(x)` 使用，造成量纲和总功率不一致；
5. 将 `rho` 单位混用（m 与 mm）导致功率数量级错误。

## R15

可扩展方向：

- 加入角分布 `d^2P/(domega dOmega)`；
- 加入偏振分量（`sigma/pi`）谱分解；
- 接入电子束能散与角散，做卷积后的束流谱；
- 与储存环参数联动，输出每圈能量损失与临界光子能量。

## R16

相关主题：

- Larmor / Lienard 辐射公式；
- 曲率辐射（curvature radiation）；
- 磁制动辐射与同步辐射的联系；
- 贝塞尔函数与渐近谱分析。

## R17

`demo.py` 的 MVP 功能清单：

- 计算总功率 `P` 与临界频率 `omega_c`；
- 数值生成同步辐射通用核 `F(x)`；
- 归一化后构建 `dP/domega`；
- 对两组 `beta` 输出物理标度对比；
- 自动执行阈值检查并给出 PASS/FAIL。

运行方式：

```bash
cd Algorithms/物理-电动力学-0186-同步辐射_(Synchrotron_Radiation)
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程（8 步）：

1. `lorentz_gamma` 计算相对论因子 `gamma`，并校验 `0<beta<1`。  
2. `total_synchrotron_power` 按 `P ~ beta^4*gamma^4/rho^2` 公式计算总辐射功率。  
3. `critical_angular_frequency` 计算 `omega_c=(3/2)gamma^3 c/rho`，建立频谱尺度。  
4. `build_synchrotron_kernel_table` 在对数网格上计算 `K_{5/3}(x)`，并用反向累积梯形得到 `int_x^inf K_{5/3}`，再构造 `F(x)`。  
5. `normalized_shape_from_kernel` 计算 `int F dx` 并归一化得到 `g(x)`，确保谱形可积且无量纲。  
6. `spectral_power_distribution` 用 `dP/domega=(P/omega_c)g(omega/omega_c)` 构建物理量纲正确的谱表。  
7. `summarize_case` 对每个 `beta` 做谱积分回收 `P`、提取峰值 `x_peak`，形成单案例诊断。  
8. `main` 汇总核函数解析常数对照、双案例标度律对照与阈值检查，全部通过则输出 `Validation: PASS`。

第三方库没有被当作黑盒求解器：

- `scipy.special.kv` 仅提供特殊函数值；
- 积分、归一化、谱构造、标度检验与通过判据都在脚本源码中显式实现。
