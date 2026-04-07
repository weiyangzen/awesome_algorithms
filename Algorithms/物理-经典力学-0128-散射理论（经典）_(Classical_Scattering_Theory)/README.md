# 散射理论（经典） (Classical Scattering Theory)

- UID: `PHYS-0128`
- 学科: `物理`
- 分类: `经典力学`
- 源序号: `128`
- 目标目录: `Algorithms/物理-经典力学-0128-散射理论（经典）_(Classical_Scattering_Theory)`

## R01

经典散射理论研究的是入射粒子在势场中偏转后的角分布与截面分布。  
在中心势 `V(r)` 下，散射过程可以由冲量参数 `b`、总能量 `E`、偏转角 `theta` 三者关系刻画，核心任务是求解 `theta(b)` 与微分截面 `dσ/dΩ`。

## R02

本条目关注经典力学语境（非量子）下的两体散射等效问题：  
把两体系统约化为约化质量在中心势中的平面运动，利用角动量守恒与能量守恒，将轨道问题转写为一维径向积分问题。

## R03

MVP 目标是给出一个可运行、可核验的最小实现，而不是只写公式：

- 选取排斥库仑势 `V(r)=kappa/r`；
- 数值计算 `theta(b)`；
- 与 Rutherford 解析公式逐点对比；
- 再由数值 `b-theta` 关系恢复 `dσ/dΩ`，与解析截面比较误差。

## R04

`demo.py` 采用的模型与方程：

- 势能：`V(r)=kappa/r`（`kappa>0`，排斥散射）；
- 无量纲化后偏转角公式：
  `theta = pi - 2 * integral_{r_min}^{∞} [ b / (r^2 * sqrt(1 - b^2/r^2 - kappa/(E r))) ] dr`
- 转折点 `r_min` 由
  `1 - b^2/r^2 - kappa/(E r) = 0`
  数值求根得到。

## R05

解析对照使用 Rutherford 结果：

- 偏转角：
  `theta_analytic(b) = 2 * arctan(kappa / (2 E b))`
- 微分截面：
  `dσ/dΩ = (kappa/(4E))^2 * csc^4(theta/2)`

数值截面通过雅可比关系恢复：

- `dσ/dΩ = (b/sin(theta)) * |db/dtheta|`

其中 `db/dtheta` 由离散网格上的数值梯度估计。

## R06

脚本输出两组报告：

- 角度报告：`theta_numeric` 与 `theta_analytic` 的最大/平均绝对误差；
- 截面报告：数值截面相对解析截面的中位误差和 90 分位误差；
- 阈值检查：单调性与误差门槛；
- 末尾给出 `Validation: PASS/FAIL`。

## R07

优点：

- 直接把经典散射理论落成“可计算+可验证”流程；
- 数值角度积分与解析结果双重对照；
- 没有依赖重型框架，结构透明。

局限：

- 当前仅覆盖排斥库仑势，不代表所有中心势；
- 截面数值导数受网格分辨率影响，端点附近精度较弱；
- 不涉及多次散射、吸收或量子效应。

## R08

前置知识：

- 中心力散射、冲量参数、偏转角；
- 数值积分与一维求根；
- 微分截面的雅可比变换。

依赖环境：

- Python 3.10+
- `numpy`
- `scipy`
- `pandas`

## R09

设 `N` 为冲量参数采样点数，`Q` 为单次 `quad` 的自适应评估次数。

- `theta(b)` 计算：每个 `b` 做一次求根 + 一次积分，总体约 `O(N*Q)`；
- 截面恢复：一次梯度与逐点公式，`O(N)`；
- 总内存开销：`O(N)`。

## R10

数值稳定性处理：

- 对转折点使用 `brentq` 并显式扩张 bracket，避免漏根；
- 将 `r` 积分改写为有限区间 `t in [0,1]` 的变换积分，移除端点平方根奇异性；
- 截面比较剔除端点区间，减少数值导数在边界处放大误差；
- 所有阈值检查都在脚本内自动执行，失败即退出非零状态。

## R11

默认参数（`demo.py`）：

- `energy = 3.0`
- `kappa = 1.2`
- `b_min = 0.08`
- `b_max = 3.6`
- `num_b = 160`
- `quad_epsabs = 1e-10`
- `quad_epsrel = 1e-10`

这组参数在速度和精度间做了平衡，可在秒级完成验证。

## R12

正确性检查（内置断言阈值）：

1. `theta(b)` 必须随 `b` 单调递减；
2. 偏转角最大绝对误差 `< 2e-6`；
3. 截面相对误差中位数 `< 2%`；
4. 截面相对误差 90 分位 `< 8%`。

四项全部满足时输出 `Validation: PASS`。

## R13

保证类型说明：

- 这是物理模型数值验证，不是组合优化问题，因此无近似比指标；
- 对给定参数，流程是确定性的（无随机输入）；
- “保证”体现在与解析解的一致性阈值，而不是概率性成功率。

## R14

常见失效模式：

1. `b` 采样太稀导致 `db/dtheta` 噪声大，截面误差升高；
2. 求根 bracket 选取不当会导致 `r_min` 求解失败；
3. 若直接在 `r` 空间硬积分，端点奇异性会引入较大误差；
4. 在 `theta -> 0` 小角区截面发散，数值比较必须限制角域。

## R15

可扩展方向：

- 支持更多势函数（如 Yukawa、幂律势、硬球势）；
- 增加“给定角度反求冲量参数”的反问题模块；
- 对参数网格批量输出 CSV，构建误差相图；
- 增加与轨道积分法（直接解牛顿方程）的对照实验。

## R16

相关主题：

- Rutherford 散射公式；
- 有效势与转折点分析；
- 经典微分截面与总截面；
- 量子散射中的相移方法（作为后续对照方向）。

## R17

运行方式：

```bash
cd Algorithms/物理-经典力学-0128-散射理论（经典）_(Classical_Scattering_Theory)
uv run python demo.py
```

预期输出包括：

- 角度对照误差摘要；
- 截面对照误差摘要；
- 多项阈值检查；
- 最终 `Validation: PASS`。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `radial_equation_residual` 定义径向方程残差 `1 - b^2/r^2 - kappa/(E r)`。  
2. `find_turning_point` 对每个 `b` 用 bracket 扩张 + `brentq` 求根，得到最近接距离 `r_min`。  
3. `deflection_angle_numeric_single` 将半无限积分通过 `u=1/r` 与 `u=u_max*(1-t^2)` 变换到 `[0,1]`，再用 `quad` 数值积分得到 `theta_numeric`。  
4. `deflection_angle_analytic` 计算 Rutherford 解析偏转角，形成逐点基准。  
5. `compute_angle_table` 批量生成 `b, r_min, theta_numeric, theta_analytic, abs_error` 表。  
6. `numerical_dsigma_domega` 用 `np.gradient` 估计 `db/dtheta`，按 `dσ/dΩ=(b/sinθ)|db/dθ|` 构建数值微分截面。  
7. `rutherford_dsigma_domega` 与 `compute_cross_section_table` 计算解析截面并给出相对误差表，同时过滤端点不稳定区。  
8. `main` 汇总误差报告并执行阈值检查，全部通过则输出 `Validation: PASS`，否则退出失败。  

第三方库没有被当作“黑盒结论器”：`scipy` 仅用于底层求根与积分数值器，物理方程、变量变换、截面重构与验证逻辑均在源码中显式实现。
