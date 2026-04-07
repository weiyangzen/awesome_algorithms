# 格点QCD (Lattice QCD)

- UID: `PHYS-0392`
- 学科: `物理`
- 分类: `计算粒子物理`
- 源序号: `411`
- 目标目录: `Algorithms/物理-计算粒子物理-0411-格点QCD_(Lattice_QCD)`

## R01

格点QCD的核心是把连续规范场理论离散到欧氏时空格点上，再用蒙特卡洛采样计算路径积分期望值。

本条目提供一个“可运行、可审计”的最小原型（MVP）：
在二维周期格点上实现 Wilson 规范作用量的 Metropolis 采样，并测量：

- 平均 plaquette
- 矩形 Wilson loop `W(R,T)`
- 由 `W(R,1)` 估计的有效静态势 `V_eff(R)`

## R02

真实生产级格点QCD通常包含：

- `SU(3)` 链变量
- 费米子行列式（如 HMC / RHMC）
- 大规模并行与长链统计误差控制

为保证本仓库单文件 MVP 的可解释性与可运行性，`demo.py` 采用紧致 `U(1)` 规范场作为教学替身（toy model），但完整保留“格点作用量 -> 局域更新 -> 热化 -> 测量”的算法骨架。

## R03

本实现使用二维 Wilson 规范作用量：

- 链变量：`U_mu(x) = exp(i * theta_mu(x))`
- plaquette 角：`theta_p = theta_0(t,x) + theta_1(t+1,x) - theta_0(t,x+1) - theta_1(t,x)`
- 作用量：`S = -beta * sum_p cos(theta_p)`

Metropolis 接受率：

- 提议 `theta -> theta'`
- `Delta S = S' - S`
- 接受概率 `P_acc = min(1, exp(-Delta S))`

可观测量：

- 平均 plaquette：`<P> = <cos(theta_p)>`
- Wilson loop：`W(R,T) = <cos(theta_loop(R,T))>`
- 有效势：`V_eff(R) = -ln(max(<W(R,1)>, eps))`

## R04

离散与边界设置：

- 维度：`2D`（时间 + 空间）
- 周期边界：`(t + Lt) mod Lt`, `(x + Lx) mod Lx`
- 每条链变量只影响两个相邻 plaquette，因此单链更新可局域计算 `Delta S`
- 角变量用 `wrap_angle` 约束到 `(-pi, pi]`

这使得源码中每个更新步骤都可直接追踪，无黑箱求解器。

## R05

关键假设与边界：

- 这是格点QCD流程教学原型，不是物理发表级仿真
- 使用 `U(1)` 替代 `SU(3)`，不含夸克动力学
- 只做二维格点，不做 4D 真实时空
- 使用 Metropolis，不实现 HMC / RHMC / over-relaxation
- 统计误差仅给出基础 `SEM`，未做自相关时间修正

## R06

`demo.py` 输入输出约定：

- 输入：脚本内固定配置参数（格点大小、`beta`、扫描步数、随机种子）
- 无交互输入，可直接 `uv run python demo.py`
- 输出：
1. 末尾测量表（tail）
2. 汇总统计（均值/标准误）
3. `W(R,1)` 反演的有效静态势表
4. 多项一致性检查与 `Validation: PASS/FAIL`

## R07

高层算法流程：

1. 初始化随机链变量 `theta[mu,t,x]`。
2. 热化阶段执行若干 sweep，仅更新不测量。
3. 进入测量阶段：每次测量前执行 `sweeps_per_measure` 个 sweep。
4. 每个 sweep 对全部链变量做局域 Metropolis 更新。
5. 每次测量记录 plaquette 与多组 `W(R,T)`。
6. 汇总样本均值、标准误（SEM）、Creutz 比与有效势。
7. 执行数值与物理范围检查，输出 PASS/FAIL。

## R08

设：

- 晶格体积 `V = Lt * Lx`
- 每次测量前 sweep 数 `s`
- 测量次数 `M`
- 测量的 Wilson loop 组合数 `L`
- 典型环面积尺度 `A`（由 `R*T` 控制）

复杂度估计：

- 单次 sweep：`O(V)`（每条链局域更新常数开销）
- 单次测量：`O(V * L * A)`（遍历全部起点计算 loop）
- 总时间：`O(thermal*V + M*(s*V + V*L*A))`
- 空间：`O(V)`（链变量 + 少量观测缓存）

## R09

数值稳定策略：

- 角变量每步 `wrap`，避免累积漂移
- 接受率区间检查（过低混合差，过高步长过小）
- Wilson loop 与 plaquette 的物理范围检查（绝对值不超过 1）
- `V_eff` 与 Creutz 比做有限值检查，防止 `log(0)`
- 固定随机种子，保证复现性

## R10

最小工具栈：

- `numpy`：链变量数组、随机采样、数值计算
- `pandas`：测量历史表与汇总展示
- `scipy.stats.sem`：基础标准误估计

未调用任何高层格点QCD库，算法路径在源码中可逐行追溯。

## R11

运行方式：

```bash
cd Algorithms/物理-计算粒子物理-0411-格点QCD_(Lattice_QCD)
uv run python demo.py
```

当所有检查通过时，末行显示 `Validation: PASS`；否则脚本以非零状态退出。

## R12

关键输出字段：

- `plaquette`：每次测量的平均 plaquette
- `W_R{r}_T{t}`：矩形 Wilson loop 样本值
- `acc_ratio`：该测量窗口内的 Metropolis 接受率
- `plaquette_mean / plaquette_sem`：平均 plaquette 与标准误
- `W_R1_T1_mean / W_R1_T1_sem`：最小环的均值与标准误
- `chi_22`：`chi(2,2)` Creutz 比
- `V_eff(R)`：由 `W(R,1)` 估计的有效静态势

## R13

内置验证条件：

1. 测量表无 NaN/Inf
2. 热化接受率在 `[0.10, 0.95]`
3. 测量接受率在 `[0.10, 0.95]`
4. 平均 plaquette 在 `[-1, 1]`
5. `|W_R1_T1| <= 1`
6. 所有 `V_eff(R)` 有限
7. `chi_22` 有限
8. 测量样本数等于配置值

全部满足时输出 `Validation: PASS`。

## R14

当前原型局限：

- 非 `SU(3)`，无法直接对应真实QCD数值结果
- 无费米子自由度，不能研究强子谱或手征性质
- 没有自相关时间估计与 block/binning 误差分析
- 没有 continuum extrapolation 与 finite-volume 系统误差研究

## R15

可扩展方向：

- 把 `U(1)` 升级为 `SU(2)`/`SU(3)` 链变量
- 引入 HMC（含伪费米子）替代纯 Metropolis
- 做多 `beta` 扫描并拟合尺度设定曲线
- 增加 Polyakov loop、关联函数与质量提取
- 做自相关时间估计、binning 与 jackknife/bootstrap 误差条

## R16

适用场景：

- 计算粒子物理课程中讲解格点路径积分流程
- 新成员快速理解“局域更新 + 可观测量测量”代码骨架
- 作为后续 `SU(3)+HMC` 工程化实现前的最小验证基线
- 作为 CI 中的快速 smoke test（可重复、无交互、秒级可运行）

## R17

与相关方法简对比：

- 直接连续理论解析：表达紧凑，但非微扰区通常不可解
- 本实现（2D U(1) Wilson + Metropolis）：实现最小、透明、易教学
- 生产级 Lattice QCD（4D SU(3)+HMC）：物理真实性强，但工程复杂度和算力成本极高

本条目的定位是“先把算法主干跑通并可审计”，不是替代生产级QCD求解器。

## R18

`demo.py` 源码级算法流（9 步）：

1. `LatticeConfig` 与 `LoopRequest` 定义格点、耦合、采样步数与 Wilson loop 请求集合。
2. `run_simulation` 用固定种子初始化 `theta[2, Lt, Lx]`，每条链变量是 `(-pi, pi]` 上的角。
3. `plaquette_angle` 按 Wilson 定义计算局域 plaquette 角；`affected_plaquettes` 返回受某条链影响的两个 plaquette。
4. `metropolis_link_update` 对单链做提议、计算局域 `Delta S`、按 `exp(-Delta S)` 接受或拒绝。
5. `single_sweep` 遍历全部链并累计接受率，先执行热化 sweep，再进入测量 sweep。
6. 每次测量调用 `average_plaquette` 和 `wilson_loop`（内部用 `wilson_loop_angle`）计算 `W(R,T)` 观测量。
7. 所有测量样本进入 `pandas.DataFrame`，形成可追溯的 Monte Carlo 历史表。
8. `estimate_static_potential` 用 `<W(R,1)>` 计算 `V_eff(R)`，`estimate_creutz_ratio` 计算 `chi(2,2)`。
9. `main` 汇总均值/SEM并执行 8 项检查，最终输出 `Validation: PASS/FAIL`。
