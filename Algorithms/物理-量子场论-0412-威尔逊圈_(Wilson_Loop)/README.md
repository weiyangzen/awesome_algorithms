# 威尔逊圈 (Wilson Loop)

- UID: `PHYS-0393`
- 学科: `物理`
- 分类: `量子场论`
- 源序号: `412`
- 目标目录: `Algorithms/物理-量子场论-0412-威尔逊圈_(Wilson_Loop)`

## R01

威尔逊圈（Wilson Loop）是规范场论里最核心的规范不变观测量之一：

`W(C) = < Tr P exp(i g ∮_C A_mu dx^mu) >`

在格点表述中，它对应闭合回路上链变量的有序乘积平均。其物理意义是用一个可计算量诊断禁闭行为：
- 若满足面积律 `W(C) ~ exp(-sigma * Area)`，通常对应禁闭相；
- 若更接近周长律，则对应非禁闭行为。

## R02

本 MVP 的问题定义：
- 输入：二维 `U(1)` 周期格点上的链变量（角度表示）以及耦合参数 `beta`；
- 计算：使用 Wilson plaquette 作用量进行 Metropolis 采样，并测量多个矩形回路 `W(R,T)`；
- 输出：
1. 平均 plaquette `<cos(theta_p)>`；
2. 多个 `W(R,T)` 的实部/虚部均值与误差；
3. 基于面积律的字符串张力近似 `sigma` 与 Creutz ratio。

## R03

离散模型与记号：
- 链变量：`U_mu(x) = exp(i * theta_mu(x))`，`theta_mu in [-pi, pi)`；
- plaquette 角：
`theta_p(x) = theta_0(x) + theta_1(x+e0) - theta_0(x+e1) - theta_1(x)`；
- Wilson 作用量（二维纯规范）：
`S = -beta * sum_p cos(theta_p)`；
- Metropolis 接受概率：`min(1, exp(-Delta S))`；
- 矩形 Wilson loop：`W(R,T) = < exp(i * sum_{l in C_{R,T}} theta_l) >`。

## R04

`demo.py` 的整体流程：
1. 初始化热启动链角 `theta ~ Uniform(-pi, pi)`；
2. 进行若干热化 sweep；
3. 在测量阶段执行“若干 sweep -> 记录观测量”；
4. 对每个 `(R,T)` 累积 Wilson loop 样本；
5. 统计均值/标准误差；
6. 用 `-ln(W)/Area` 与线性拟合给出 `sigma` 近似；
7. 输出 Creutz ratio `chi(2,2)`；
8. 执行内置断言检查。

## R05

核心数据结构：
- `WilsonLoopConfig`：集中管理格点规模、`beta`、采样步数、回路形状等配置；
- `links`：`shape=(L, L, 2)` 的 `numpy` 数组，存储二维每个方向的链角；
- `loop_samples`：`dict[(R,T)] -> list[complex]`，保存每次测量得到的回路值；
- `loop_df`：`pandas.DataFrame`，输出汇总表（均值、误差、局部 `sigma`）。

## R06

正确性关键点：
- 局部更新时只重算与该链相关的两个 plaquette，保证 `Delta S` 计算与 Wilson 作用量一致；
- 回路沿边积分采用“正向加、反向减”的有向和，保证闭合回路方向一致；
- 周期边界通过 `% L` 实现，避免边界丢链；
- 虚部均值应接近 0（统计涨落下非零但应很小），是实现正确性的额外信号。

## R07

复杂度（二维 `L x L`）：
- 单次 sweep 更新 `2L^2` 条链，每次更新常数成本，故约 `O(L^2)`；
- 每次回路测量对每个 `(R,T)` 遍历 `L^2` 起点，单起点路径长约 `O(R+T)`，故约 `O(|S| * L^2 * (R+T))`，`|S|` 为回路数量；
- 总体时间复杂度约：
`O((N_therm + N_meas * gap) * L^2 + N_meas * |S| * L^2 * (R+T))`；
- 空间复杂度主要是链场与样本缓存：`O(L^2 + N_meas * |S|)`。

## R08

边界与异常处理：
- 配置参数做显式校验：`beta>0`、步数为正、`proposal_width in (0, pi]`；
- 回路边长必须 `< L` 且为正；
- 接受率若过低/过高会触发断言，提示提案宽度不合适；
- 对 `log(W)` 使用下界截断 `clip(..., 1e-12)`，避免数值下溢。

## R09

MVP 取舍说明：
- 保留：最核心的格点 Wilson 作用量 + Metropolis + Wilson loop 观测链路；
- 省略：多重网格、混合更新、热浴算法、SU(2)/SU(3) 群结构与并行化；
- 目标是“可审计、可复现、秒级可跑”的教学与验证版本，而不是大规模生产级格点计算。

## R10

`demo.py` 主要函数职责：
- `validate_config`：参数合法性检查；
- `plaquette_angle / affected_plaquettes`：局部作用量几何关系；
- `metropolis_sweep`：核心 Markov 更新；
- `average_plaquette`：测量平均 plaquette；
- `loop_phase_at_start / measure_wilson_loop`：矩形回路相位与平均值；
- `build_loop_report`：整理观测统计表；
- `estimate_string_tension / creutz_ratio`：导出物理量；
- `run_quality_checks`：验收断言；
- `main`：完整执行入口。

## R11

运行方式：

```bash
cd Algorithms/物理-量子场论-0412-威尔逊圈_(Wilson_Loop)
uv run python demo.py
```

脚本无交互输入，直接打印配置、观测表和检查结果。

## R12

输出字段解读：
- `thermal_acceptance / measurement_acceptance`：热化阶段与测量阶段接受率；
- `<plaquette>`：平均 plaquette 值及标准误差；
- `W_real_mean`：`W(R,T)` 的实部均值（物理上主要信号）；
- `W_real_sem`：实部标准误差；
- `W_imag_mean`：虚部均值（应接近 0）；
- `sigma_local = -ln(W)/Area`：局部面积律斜率估计；
- `sigma_fit`：通过原点线性拟合得到的全局字符串张力近似；
- `chi(2,2)`：Creutz ratio，对应有限尺寸下的张力估计指标。

## R13

内置最小验证门槛：
1. 接受率必须在合理区间 `(0.05, 0.95)`；
2. `<plaquette>` 必须落在 `[-1, 1]`；
3. `W` 的均值与误差必须是有限数；
4. 若测得 `(1,1)` 与 `(2,2)`，要求 `W(1,1) > W(2,2)`（面积增大，回路值衰减）；
5. `max |Im W|` 不能过大（默认阈值 `0.08`）。

## R14

关键参数与调参建议：
- `beta`：控制涨落强弱；
- `proposal_width`：控制接受率，过小混合慢，过大接受率低；
- `thermalization_sweeps`：热化长度；
- `measurement_sweeps` 与 `sweeps_between_measurements`：统计精度与样本相关性折中；
- `loop_shapes`：回路集合，建议同时覆盖多个面积与长宽比。

经验建议：先把接受率调到 `0.2~0.7`，再加大测量步数降低误差。

## R15

与相关方法对比：
- 相比连续路径积分解析推导：本实现直接在离散格点上数值采样，更直观但有有限尺寸/离散误差；
- 相比“直接调用现成格点库”：本代码没有黑盒更新器，局部作用量、回路测量都显式可查；
- 相比 SU(3) 真实 QCD：`U(1)` 二维模型更简化，主要用于演示 Wilson loop 工作流。

## R16

典型应用场景：
- 量子场论/格点场论课程中的 Wilson loop 教学演示；
- 作为更复杂规范群（SU(2)/SU(3)）实现前的最小验证基线；
- 验证 Monte Carlo 更新、回路测量、误差统计三个环节是否打通。

## R17

可扩展方向：
- 升级到 3D/4D 格点与 SU(2)/SU(3) 链变量；
- 引入热浴、over-relaxation、HMC 等更高效更新；
- 做自相关时间估计与 binning 误差分析；
- 扫描 `beta` 研究相图与连续极限外推；
- 输出到 CSV/Parquet 并配合可视化脚本做系统分析。

## R18

`demo.py` 的源码级算法流程（9 步，非黑盒）：

1. `main` 创建 `WilsonLoopConfig`，并由 `validate_config` 校验格点规模、步数、提案宽度、回路尺寸。  
2. 随机初始化 `links[L,L,2]`（热启动），每个元素是一个 `U(1)` 链角。  
3. 热化阶段反复调用 `metropolis_sweep`：逐链提出角度扰动，按局部 `Delta S` 执行 Metropolis 接受/拒绝。  
4. `metropolis_sweep` 内部通过 `affected_plaquettes` 找到受影响的两个 plaquette，再用 `plaquette_angle` 计算更新前后 `cos(theta_p)` 差值。  
5. 测量阶段按“若干 sweep 后测一次”的节奏运行，以减弱样本相关性。  
6. 每次测量时先用 `average_plaquette` 记录 `<cos(theta_p)>`，再对每个 `(R,T)` 调 `measure_wilson_loop`。  
7. `measure_wilson_loop` 会遍历所有起点 `(x,y)`，由 `loop_phase_at_start` 沿矩形边界做有向链角求和并取 `exp(i*phase)`。  
8. 测量结束后 `build_loop_report` 统计 `W` 的实虚部均值与标准误差，并计算局部 `sigma_local=-ln(W)/Area`；`estimate_string_tension` 与 `creutz_ratio` 提供派生量。  
9. `run_quality_checks` 对接受率、虚部污染、回路衰减关系等做断言，全部通过后输出 `All checks passed.`。
