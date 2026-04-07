# Andersen恒温器 (Andersen Thermostat)

- UID: `PHYS-0328`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `335`
- 目标目录: `Algorithms/物理-计算物理-0335-Andersen恒温器_(Andersen_Thermostat)`

## R01

Andersen 恒温器是一种用于分子动力学（MD）中的随机恒温方法。其思想是：
在常规牛顿积分过程中，以泊松过程触发“碰撞事件”，一旦触发就把粒子速度重采样为目标温度 `T` 下的 Maxwell-Boltzmann 分布，从而驱动系统采样正则系综（NVT）。

本条目 MVP 采用一维谐振子体系：
`U(x)=0.5*k*x^2`，并把 Andersen 随机碰撞嵌入 velocity Verlet 积分中。

## R02

该算法的重要性在于：

- 实现简单：只需在积分步骤后加入伯努利碰撞判断与速度重采样；
- 温控直接：目标温度通过速度分布方差显式控制；
- 对教学友好：非常适合演示“确定性动力学 + 随机热浴”的混合机制。

相比纯 deterministic 的恒温器，Andersen 通过随机碰撞更容易破除部分动力学相关性，但也会扰动真实动力学时间关联。

## R03

本 MVP 的输入输出（无需交互，参数写在 `AndersenConfig`）：

- 输入参数：
1. 质量 `mass`、弹簧常数 `spring_k`；
2. 目标温度 `target_temperature`、玻尔兹曼常数 `boltzmann_k`；
3. 碰撞频率 `collision_frequency=nu`；
4. 时间步长 `dt`、总步数 `n_steps`、烧入步数 `burn_in`；
5. 随机种子 `seed`。

- 输出结果：
1. 时间序列 `positions`, `velocities`, `kinetic_energy`, `potential_energy`；
2. 每步是否碰撞的布尔序列 `collision_events`；
3. `pandas.DataFrame` 汇总表 `summary`，包含温度、方差、碰撞频率与能量均值；
4. 终端打印误差并执行断言，成功时输出 `All checks passed.`。

## R04

核心数学关系：

1. 力与加速度（谐振子）：
`F(x) = -k*x`，`a(x)=F/m=-(k/m)*x`。

2. velocity Verlet 漂移步骤：
`v_{n+1/2}=v_n + 0.5*dt*a(x_n)`
`x_{n+1}=x_n + dt*v_{n+1/2}`
`v_{n+1}^{det}=v_{n+1/2}+0.5*dt*a(x_{n+1})`

3. Andersen 碰撞机制：
每步以概率 `p = nu*dt` 触发碰撞；若触发，则
`v_{n+1} ~ N(0, sigma_v^2)`，其中
`sigma_v^2 = k_B*T/m`。

4. 平衡态检验（谐振子正则分布）：
`<v^2> = k_B*T/m`
`<x^2> = k_B*T/k`
并据此估计样本温度 `T_hat = m*<v^2>/k_B`。

## R05

设时间步数为 `N=n_steps`：

- 时间复杂度：`O(N)`（每步常数成本积分 + 一次随机判定）；
- 空间复杂度：`O(N)`（保存完整轨迹与能量序列）；
- 若只保留在线统计量，可降到 `O(1)` 额外空间。

## R06

MVP 的三个闭环：

- 闭环 A（动力学推进）：velocity Verlet 更新 `x,v`；
- 闭环 B（恒温碰撞）：按 `nu*dt` 执行伯努利抽样并重采样速度；
- 闭环 C（统计验证）：烧入后统计 `<x^2>`,`<v^2>`,`T_hat` 与 `nu` 经验值并与理论目标对比。

## R07

优点：

- 算法结构清晰，源代码和公式一一对应；
- 恒温机制直接、可解释性强；
- 使用固定随机种子可复现结果，便于回归测试。

局限：

- 随机碰撞会破坏真实动力学相关函数，不适合保真动力学输运性质估计；
- 当前仅示范一维单粒子谐振子，不代表复杂多体体系全部数值挑战；
- 统计正确性依赖样本长度与烧入设置，单次短轨迹会有波动。

## R08

前置知识与环境：

- 统计物理基础：Maxwell-Boltzmann 分布、NVT 概念；
- 数值积分基础：velocity Verlet；
- Python `>=3.10`；
- 依赖：`numpy`, `pandas`。

## R09

适用场景：

- 教学演示 Andersen 恒温器原理；
- 为更复杂 MD 代码做最小可运行原型与单元测试基准；
- 快速检验温控参数（`nu`, `dt`）对平衡统计量的影响。

不适用场景：

- 需要严格保留真实动力学时间关联的研究；
- 大规模多体生产模拟（需更完整邻域、边界、并行等基础设施）；
- 对输运系数或时间相关函数做高保真估计。

## R10

正确性直觉：

1. 纯 velocity Verlet 负责在势能面上做稳定的确定性推进；
2. Andersen 碰撞把速度分布周期性拉回目标温度对应高斯分布；
3. 长时间下，位置与速度边缘分布应收敛到正则系综结果；
4. 对谐振子可用 `k_B*T/k` 与 `k_B*T/m` 的闭式二阶矩做直接交叉验证。

## R11

数值稳定与可复现策略：

- 配置校验：确保 `mass>0`、`spring_k>0`、`T>0`、`dt>0`；
- 约束 `nu*dt<=1`，保证伯努利碰撞概率合法；
- 使用 `np.random.default_rng(seed)` 固定随机源，避免漂移式不可复现；
- 设置 `burn_in` 丢弃初态偏置，再评估平衡统计量。

## R12

关键参数与调参建议：

- `collision_frequency (nu)`：
  太小会温控弱、收敛慢；太大则随机化过强，动力学连续性下降。
- `dt`：
  太大可能造成积分误差；太小则计算成本上升。
- `n_steps` 与 `burn_in`：
  `n_steps` 决定统计误差，`burn_in` 决定是否剔除初态过渡段。
- `spring_k` 与 `target_temperature`：
  共同决定理论位置方差 `k_B*T/k`，可据此检查参数是否合理。

## R13

理论保证（本 MVP 语境）：

- 近似比保证：N/A（非组合优化算法）；
- 确定性精确保证：N/A（含随机抽样）；
- 可验证保证：通过固定种子和足够长轨迹，脚本要求以下相对误差阈值：
1. 温度误差 `< 6%`；
2. 位置方差误差 `< 8%`；
3. 速度方差误差 `< 6%`；
4. 碰撞频率误差 `< 8%`。

## R14

常见失效模式：

1. `nu*dt>1` 导致碰撞概率非法；
2. `n_steps` 太小或 `burn_in` 不足，统计误差偏大；
3. `dt` 过大导致积分偏差积累；
4. `nu` 极低时温控弱，样本可能长时间未接近平衡；
5. 误把“温度统计正确”当作“动力学相关函数也正确”。

## R15

可扩展方向：

- 从单粒子扩展到多粒子、多维度；
- 加入周期边界、粒子间相互作用与邻域搜索；
- 输出直方图并与理论高斯进行更细粒度拟合检验；
- 与 Langevin、Nosé-Hoover 等恒温器并行对比温控质量与动力学扰动。

## R16

相关算法与概念：

- Langevin 恒温器（连续阻尼 + 随机噪声）；
- Nosé-Hoover 恒温器（扩展系统变量耦合）；
- Berendsen / velocity rescaling（工程常用温控策略）；
- Metropolis Monte Carlo（同样可采样正则分布，但动力学解释不同）。

## R17

`demo.py` 功能清单：

- 定义并校验 `AndersenConfig`；
- 实现 `verlet_andersen_step`（Verlet + 碰撞重采样）；
- 运行全轨迹并收集 `x/v/E` 序列；
- 烧入后统计温度、方差、碰撞频率；
- 生成 `summary` 表并打印关键指标；
- 执行断言，作为可自动化校验的最小闭环。

运行方式：

```bash
cd Algorithms/物理-计算物理-0335-Andersen恒温器_(Andersen_Thermostat)
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程（8 步）：

1. `AndersenConfig.validate` 对物理参数和离散参数做边界检查，并显式验证 `nu*dt<=1`。  
2. `run_simulation` 初始化 RNG、初始状态 `(x,v)`，并创建轨迹数组。  
3. 每个时间步调用 `verlet_andersen_step`：先按 `acceleration` 做 velocity Verlet 的确定性推进。  
4. 在同一步中执行伯努利判定 `rng.random() < nu*dt`；命中后从 `N(0, sigma_v^2)` 重采样速度。  
5. 将 `x,v,Ek,Ep,collision_flag` 写入序列，形成完整时域数据。  
6. 丢弃 `burn_in` 后计算 `sample_var_x`、`sample_var_v`、`sample_temperature` 与经验碰撞频率。  
7. 组装 `summary`（`pandas.DataFrame`）并由 `value_from_summary` 提取指标，计算相对误差。  
8. `main` 打印结果并执行四个误差断言，全部满足则输出 `All checks passed.`。  

说明：第三方库只承担基础数值容器/统计功能（`numpy`, `pandas`）；Andersen 恒温器的核心流程（Verlet 漂移、伯努利碰撞、Maxwell 重采样、平衡统计检验）均在源码中显式实现，不是黑盒调用。
