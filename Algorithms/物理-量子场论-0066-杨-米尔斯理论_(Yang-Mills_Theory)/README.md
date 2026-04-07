# 杨-米尔斯理论 (Yang-Mills Theory)

- UID: `PHYS-0066`
- 学科: `物理`
- 分类: `量子场论`
- 源序号: `66`
- 目标目录: `Algorithms/物理-量子场论-0066-杨-米尔斯理论_(Yang-Mills_Theory)`

## R01

杨-米尔斯理论是非阿贝尔规范场论的核心框架。与电磁场（阿贝尔 `U(1)`）不同，杨-米尔斯场的规范群可取 `SU(N)`，规范场自身带“色荷”，因此场之间会自相互作用。

连续理论中其动力学由场强张量 `F_{mu nu}` 与拉氏量决定；在数值上，常用离散化是格点 Wilson 作用量。这个条目给出一个最小可运行数值 MVP，演示杨-米尔斯在格点上的基本计算闭环。

## R02

本条目要解决的问题：

- 如何在不依赖大型框架的前提下，写出可运行的 `SU(2)` 杨-米尔斯格点模拟；
- 如何显式实现 Wilson plaquette 作用量与 Metropolis 更新，而不是黑盒调用；
- 如何在脚本内验证一个关键物理性质：规范不变性（平均 plaquette 在局域规范变换下不变）。

MVP 目标：

- 构建 2D 周期边界 `L x L` 晶格的 `SU(2)` 链变量；
- 用 Metropolis 扫掠热化与采样；
- 输出 `pandas` 观测表（接受率、平均 plaquette、作用量密度）；
- 对同一构型施加随机局域规范变换并验证可观测量不变。

## R03

`demo.py` 的输入输出约定（无交互）：

- 输入（脚本内固定参数）：
1. `seed=20260407`；
2. 晶格尺寸 `L=8`；
3. 耦合参数 `beta=2.20`；
4. 提案步长 `epsilon=0.30`；
5. 热化 `30` sweeps，生产 `20` sweeps。
- 输出：
1. 最近观测记录表（`phase/sweep/accept_rate/avg_plaquette/action_density`）；
2. 汇总指标表（生产期均值、标准差、规范不变性误差）；
3. 断言全部通过后打印 `All checks passed.`。

## R04

连续理论与离散模型：

1. 连续杨-米尔斯场强：
`F^a_{mu nu} = d_mu A^a_nu - d_nu A^a_mu + g f^{abc} A^b_mu A^c_nu`。

2. 连续拉氏量：
`L = -(1/4) F^a_{mu nu} F^{a,mu nu}`。

3. 格点表示：链变量 `U_mu(x) in SU(2)`，基本回路（plaquette）
`U_p = U_x(x) U_y(x+hat{x}) U_x^dagger(x+hat{y}) U_y^dagger(x)`。

4. Wilson 作用量（2D 版本）：
`S = beta * sum_p [1 - (1/2) Re Tr(U_p)]`。

`demo.py` 直接实现以上第 3-4 点，并通过 Metropolis 对 `exp(-S)` 进行采样。

## R05

复杂度分析（`L` 为边长，`N=L^2`，总 sweeps 记为 `T`）：

- 单次 sweeps 更新链接数约 `2N`；
- 每次链接提案只重算 2 个相关 plaquette（局部代价 `O(1)`）；
- 单 sweeps 时间复杂度 `O(N)`；
- 总时间复杂度 `O(TN)`；
- 空间复杂度 `O(N)`（存储全部链变量和少量观测历史）。

## R06

MVP 算法闭环：

1. 用四元数归一化生成 Haar 随机 `SU(2)` 链变量；
2. 计算定向 plaquette 与平均 plaquette；
3. 对每条链做局部随机群提案 `U' = R U`；
4. 只重算该链相邻 2 个 plaquette 得到局部 `Delta S`；
5. 以 `min(1, exp(-Delta S))` 接受/拒绝；
6. 执行热化 sweeps，再执行生产 sweeps；
7. 统计接受率与平均 plaquette；
8. 对最终构型施加随机局域规范变换 `U'_mu(x)=G(x)U_mu(x)G^dagger(x+mu)`；
9. 比较变换前后平均 plaquette 差值并断言通过。

## R07

优点：

- 公式和代码一一对应，可审计；
- 显式实现局部 `Delta S`，不是黑盒 ODE/MC 封装；
- 直接展示规范不变性这一杨-米尔斯核心结构。

局限：

- 仅为 2D `SU(2)` 教学 MVP，不是 4D 物理精确模拟；
- 未做自相关时间评估、误差条完整统计与外推；
- 未覆盖改进 action、热浴算法、过松弛等高效技术。

## R08

前置知识与环境：

- 规范场论基本概念：`SU(N)`、链变量、Wilson loop；
- MCMC / Metropolis 接受-拒绝机制；
- Python `>=3.10`；
- 依赖：`numpy`, `pandas`。

## R09

适用场景：

- 杨-米尔斯格点入门教学；
- 需要“最小可复现”的规范不变性数值演示；
- 为后续更复杂格点代码做骨架验证。

不适用场景：

- 高精度物理量提取（如真实 QCD 标度量）；
- 需要 4D、大体积、长链统计的研究级计算；
- 需要并行加速/GPU/HMC 的高性能任务。

## R10

正确性直觉：

1. Wilson 作用量由 plaquette 的迹构成，迹在相似变换下不变；
2. 局域规范变换只会把链变量左右乘 `G` 与 `G^dagger`，闭合回路的整体迹保持不变；
3. Metropolis 以 `exp(-S)` 为平衡分布，`Delta S` 决定接受概率；
4. 因此同一构型变换前后的平均 plaquette 应在数值误差内一致。

## R11

数值稳定策略：

- `SU(2)` 元素用单位四元数生成，避免直接矩阵随机化导致非幺正漂移；
- 提案步长 `epsilon` 控制在中等范围，避免接受率过低或过高；
- `exp(-Delta S)` 计算时对指数上界做截断，防止浮点溢出；
- 断言接受率、plaquette 区间与规范不变性误差，防止静默失败。

## R12

关键参数与影响：

- `L`：体积大小，增大后统计更接近热力学极限但计算更慢；
- `beta`：耦合强度参数，影响平均 plaquette 与涨落结构；
- `epsilon`：提案扰动大小，直接控制接受率；
- `thermal_sweeps`：热化时长，不足会残留初值偏差；
- `production_sweeps`：样本数，不足会导致统计不稳。

调参建议：

- 目标接受率可先瞄准 `0.2 ~ 0.8`；
- 若观测波动过大，先增加生产 sweeps，再增大晶格尺寸；
- 若热化不足，增加 thermal sweeps 并观察稳定平台。

## R13

- 近似比保证：N/A（非组合优化近似算法）。
- 随机成功率保证：N/A（不是概率正确性算法条目）。

本 MVP 提供的可验证保证：

- 输出构型的平均 plaquette 始终在物理合理区间 `(0,1)`；
- 生产期接受率处于健康范围（防止链冻结或无效随机游走）；
- 规范变换前后平均 plaquette 差值低于阈值 `1e-10`。

## R14

常见失效模式：

1. `epsilon` 过大导致接受率接近 0，链几乎不动；
2. `epsilon` 过小导致接受率接近 1，但混合过慢；
3. 更新时若误算相邻 plaquette 集合，会破坏 `Delta S` 正确性；
4. 忘记周期边界索引 `% L` 会导致错误拓扑；
5. 规范变换实现若邻点索引错位，规范不变性检查会失败。

## R15

工程扩展方向：

- 从 2D 扩展到 4D Euclidean lattice；
- 从 `SU(2)` 推广到 `SU(3)`（QCD 更相关）；
- 增加 Wilson loop、Creutz ratio、相关长度估计；
- 引入热浴/过松弛/HMC 提升采样效率；
- 加入自动相关分析与误差条（jackknife / bootstrap）。

## R16

相关条目：

- 非阿贝尔规范理论与规范对称性；
- Wilson 作用量与格点规范场论；
- MCMC 采样与详细平衡；
- 规范不变可观测量（plaquette、Wilson loop）。

## R17

`demo.py` 交付能力清单：

- 显式构造 `SU(2)` 链变量并计算 plaquette；
- 实现局部 `Delta S` 的 Metropolis sweeps；
- 输出热化与生产阶段观测表；
- 实施随机局域规范变换并做不变性断言；
- 无交互、单命令可运行。

运行方式：

```bash
cd Algorithms/物理-量子场论-0066-杨-米尔斯理论_(Yang-Mills_Theory)
uv run python demo.py
```

## R18

`demo.py` 的源码级算法流程（9 步）：

1. `su2_from_quaternion` 把单位四元数映射到 `2x2` 复矩阵，保证元素属于 `SU(2)`。  
2. `random_su2` 与 `small_random_su2` 分别生成初始链变量与 Metropolis 小步提案。  
3. `initialize_links` 创建 `links[mu, x, y]`，形成带方向的周期晶格规范场。  
4. `plaquette_xy` 计算最小闭合回路 `U_p`，`average_plaquette` 对全晶格求平均可观测量。  
5. `local_action_contribution` 只重算某条链相邻的 2 个 plaquette，得到局部 Wilson 作用量。  
6. `metropolis_sweep` 对每条链执行提案、计算 `Delta S`、按 `exp(-Delta S)` 接受或回退。  
7. `run_simulation` 先热化后采样，并把每次观测写入 `pandas` 表。  
8. `gauge_transform_links` 实现 `U'_mu(x)=G(x)U_mu(x)G^dagger(x+mu)`，构造规范等价构型。  
9. `main` 比较变换前后平均 plaquette，结合接受率与统计波动断言，全部通过后输出 `All checks passed.`。

说明：`demo.py` 没有把 Yang-Mills 关键部分交给第三方黑盒；群元素、作用量、局部更新、规范变换和验证指标都在源码中逐步展开。
