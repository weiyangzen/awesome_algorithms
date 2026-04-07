# 耗散粒子动力学 (Dissipative Particle Dynamics)

- UID: `PHYS-0336`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `343`
- 目标目录: `Algorithms/物理-计算物理-0343-耗散粒子动力学_(Dissipative_Particle_Dynamics)`

## R01

耗散粒子动力学（DPD）是一种介观粒子法，用软势粗粒化粒子表示“流体团簇”，并通过三类成对作用力同时编码：

- 保守力（控制排斥与结构）
- 耗散力（速度相关阻尼）
- 随机力（热涨落）

它的核心价值是：在比原子分子动力学更大时空尺度上，仍保留动量守恒与流体动力学行为。

## R02

本条目 MVP 采用二维周期盒中的均匀 DPD 流体作为最小可运行示例：

- 维度：`2D`
- 边界：`x/y` 双向周期边界
- 初始态：随机位置 + 高斯速度（去除质心漂移）
- 目标：观察体系在 DPD 热浴作用下的温度稳定、成对相互作用统计、质心动量守恒趋势

此设定避免复杂几何与外场，专注验证 DPD 基础算法链路。

## R03

`demo.py` 中单对粒子 `i-j` 的作用力写为：

`F_ij = (F_C + F_D + F_R) * e_ij`

其中：

- `e_ij = r_ij / |r_ij|`
- `r = |r_ij| < r_c` 时才有相互作用
- `w_R(r) = 1 - r/r_c`
- `w_D(r) = [w_R(r)]^2`

三项分别为：

- `F_C = a * w_R`
- `F_D = -gamma * w_D * (v_ij · e_ij)`
- `F_R = sigma * w_R * theta_ij / sqrt(dt)`

这里 `theta_ij ~ N(0,1)`，并在代码里按“每对粒子每步一个随机数”实现反对称力更新。

## R04

DPD 的温控关键是涨落-耗散关系（Fluctuation-Dissipation）：

`sigma^2 = 2 * gamma * kBT`

MVP 参数：

- `gamma = 6.0`
- `kBT = 1.0`
- 因而 `sigma = sqrt(12) ≈ 3.464`

这样随机注热与耗散阻尼在统计上平衡，使体系温度围绕目标量级波动。

## R05

时间推进采用轻量的显式 Euler-Maruyama（随机微分方程一阶离散）：

1. 用当前 `x, v` 计算总力 `F(x,v)`（含随机项）
2. 速度更新：`v <- v + (F/m) * dt`
3. 位置更新：`x <- x + v * dt`
4. 位置按周期边界取模回盒内

该积分器实现简单、可读性高，适合教学型 MVP；高精度生产模拟通常会改用 DPD-VV / Shardlow splitting。

## R06

边界与邻域处理策略：

- 周期边界：`x = x % L`
- 距离计算：最小镜像规则
  - `dr = dr - L * round(dr / L)`

这保证任意两粒子交互时都使用周期盒中的最近镜像距离，避免边界处非物理跳变。

## R07

`demo.py` 主流程：

1. `DPDConfig` 校验参数合法性。
2. `initialize_state` 初始化位置和速度，并去除初始质心速度。
3. `compute_pair_forces` 双循环遍历 `i<j`，累加保守/耗散/随机力与保守势能。
4. `run_simulation` 执行 `steps` 次积分推进。
5. 周期记录 `temperature / pair_count / rms_speed / com_speed`。
6. 结束后计算一次快照的 `g(r)`（径向分布函数）。
7. 生成历史表与摘要表并打印。
8. 执行数值断言，全部通过则输出 `Validation: PASS`。

## R08

复杂度估计（`N` 粒子，步数 `T`）：

- 力计算为全对遍历：每步 `O(N^2)`
- 总时间复杂度：`O(T * N^2)`
- 空间复杂度：`O(N)`（主要是位置/速度/力数组）

本 MVP 配置 `N=64, T=3000`，CPU 单进程可在合理时间内运行。

## R09

数值稳定与可解释性约束：

- 使用较小步长 `dt=0.004` 降低显式积分离散误差
- 采用软排斥参数 `a=5`，避免硬碰撞刚性
- 通过 `temperature` 监控热平衡是否偏离
- 通过 `com_speed` 监控总动量漂移
- 通过 `pair_count` 监控体系交互是否充分

## R10

MVP 技术栈：

- Python 3
- `numpy`：粒子状态与力计算
- `pandas`：历史与摘要表格输出

未依赖专用 DPD 引擎。核心算法（成对力、最小镜像、积分推进、`g(r)`）均在源码中显式实现。

## R11

运行方式：

```bash
cd Algorithms/物理-计算物理-0343-耗散粒子动力学_(Dissipative_Particle_Dynamics)
uv run python demo.py
```

脚本无交互输入，直接输出历史指标、汇总指标和最终校验结果。

## R12

主要输出指标含义：

- `temperature`：瞬时动温（去质心自由度近似）
- `potential_energy`：仅保守力对应势能
- `pair_count`：每次报告时刻截断半径内的相互作用粒子对数量
- `rms_speed`：粒子速度均方根
- `com_speed`：体系质心速度模长
- `g_peak_near_rc`：`g(r)` 在 `r≈rc` 附近的峰值代理，粗略反映局域结构

## R13

`demo.py` 内置验收阈值：

1. 位置与速度数组必须全为有限值（无 NaN/Inf）
2. `0.50 <= temperature_mean <= 1.80`
3. `com_speed_max < 0.25`
4. `pair_count_mean > 20`

满足后输出 `Validation: PASS`。

## R14

当前 MVP 局限：

- 仅二维、单组分、同参数 `a/gamma` 的简化体系
- 力计算为 `O(N^2)` 全对遍历，未用邻居链表/Verlet list
- 仅实现显式 Euler-Maruyama，一阶精度
- 未引入剪切流、壁面、外场、反应或多相耦合
- `g(r)` 仅用最终快照估计，统计收敛性有限

## R15

可扩展方向：

- 引入 cell list / Verlet 邻居表，将每步复杂度降至近线性
- 升级到 DPD velocity-Verlet 或 Shardlow splitting 提升时间离散精度
- 扩展多组分参数矩阵 `a_ij, gamma_ij` 支持复杂流体混合
- 增加壁面边界与驱动力，模拟 Poiseuille/Couette 流
- 增加多时刻 `g(r)` 与输运系数（扩散系数、粘度）统计

## R16

DPD 常见应用：

- 聚合物/表面活性剂介观自组装
- 胶体悬浮液与复杂流体流变
- 微流控中的软物质输运
- 生物膜与粗粒化生物流体模拟
- 需要“热涨落 + 动量守恒”并重的介观流体场景

## R17

与相关方法对比：

- 相对原子 MD：DPD 粒子更粗粒，允许更大时空尺度，但微观细节减少
- 相对 Brownian Dynamics：DPD 保留成对反作用力，天然更利于恢复流体动量输运
- 相对 LBM：DPD 更偏拉格朗日粒子描述，处理多体软物质界面直观；LBM 在规则网格流场求解上更高效

本条目聚焦“可运行且可读的 DPD 最小闭环”。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `DPDConfig` 定义物理与数值参数，并在 `validate` 中做边界检查。  
2. `initialize_state` 生成随机位置和高斯速度，去除初始质心速度。  
3. `minimum_image` 对任意位移向量应用最小镜像规则。  
4. `compute_pair_forces` 对所有 `i<j`：
   - 计算 `r_ij, v_ij`；
   - 评估 `F_C/F_D/F_R`；
   - 以反对称形式累加到 `forces[i], forces[j]`；
   - 同时累计保守势能与相互作用对数。  
5. `run_simulation` 每步执行 Euler-Maruyama 更新：先 `v` 后 `x`，并施加周期取模。  
6. 每 `report_every` 步记录 `temperature/pair_count/rms_speed/com_speed` 到 `pandas` 历史表。  
7. `radial_distribution_function` 基于最终快照统计 2D `g(r)`，提取 `r≈rc` 区域峰值代理。  
8. `main` 汇总输出指标并执行断言（有限值、温度区间、质心漂移、交互对数），通过则打印 `Validation: PASS`。  

说明：`numpy/pandas` 只承担基础数组与表格功能，DPD 的力学更新逻辑均由源码显式实现，无第三方黑盒求解。
