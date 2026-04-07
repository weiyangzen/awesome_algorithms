# 临界现象 (Critical Phenomena)

- UID: `PHYS-0294`
- 学科: `物理`
- 分类: `统计力学`
- 源序号: `297`
- 目标目录: `Algorithms/物理-统计力学-0297-临界现象_(Critical_Phenomena)`

## R01

临界现象（Critical Phenomena）指系统在二级相变临界点附近出现的普适行为：关联长度急剧增大、涨落增强、热力学量出现幂律或尖峰特征。  
在统计力学里，二维 Ising 模型是最经典的可计算范式之一，可用来演示：
- 序参量（磁化强度）从有序相到无序相的变化；
- 磁化率与热容在临界温度附近的峰值；
- 临界指数的幂律近似关系。

## R02

本任务的 MVP 目标不是“高精度发表级计算”，而是用最小可运行代码复现实验上最核心的临界签名：
- 当 `T < Tc`，自发磁化 `|m|` 较大；
- 当 `T > Tc`，`|m|` 快速减小；
- 磁化率 `χ` 与比热 `C` 在 `Tc` 附近出现峰值（有限尺寸下为圆滑峰）。

`demo.py` 采用二维正方晶格 Ising 模型 + Metropolis 单自旋翻转采样，属于“简单但诚实”的基线方案。

## R03

模型定义（无外场、耦合常数 `J=1`）：

- 自旋变量：`s_i ∈ {+1, -1}`
- 哈密顿量：`H = - Σ_{<i,j>} s_i s_j`
- 周期边界条件：晶格上下左右首尾相接

关键观测量：
- 单位自旋能量：`e = E / N`
- 单位自旋磁化：`m = M / N`
- 序参量：`|m|`
- 磁化率：`χ = N ( <m^2> - <m>^2 ) / T`
- 比热：`C = N ( <e^2> - <e>^2 ) / T^2`

## R04

临界区常用标度关系（无限系统极限）：
- `|m| ~ (Tc - T)^β`, `T < Tc`
- `χ ~ |T - Tc|^{-γ}`
- `C ~ |T - Tc|^{-α}`
- `ξ ~ |T - Tc|^{-ν}`（关联长度）

二维 Ising 的理论临界温度（Onsager 精确结果）：
`Tc = 2 / ln(1 + sqrt(2)) ≈ 2.2692`（单位取 `k_B=J=1`）。

有限晶格上不会真正发散，表现为“有限高度峰 + 峰位随尺寸漂移”。

## R05

算法选择：Metropolis Monte Carlo（MCMC）

单步翻转规则：
1. 随机挑选一个格点；
2. 计算翻转能量差 `ΔE`；
3. 若 `ΔE <= 0` 则接受；
4. 否则以概率 `exp(-βΔE)` 接受（`β=1/T`）。

该过程满足细致平衡，平衡态采样逼近玻尔兹曼分布 `p(s) ∝ exp(-βH)`。

## R06

`demo.py` 的统计估计流程：
- 预热（warmup）：先运行若干 sweep，让初始随机态靠近平衡；
- 采样（sample）：每隔 `sample_interval` 个 sweep 记录一次 `E` 与 `M`；
- 通过样本均值估算 `e, |m|, χ, C`；
- 在温度网格上扫描，取 `χ` 与 `C` 峰位来估计有限尺寸 `Tc(L)`。

这对应“先热化，再测量”的标准 MCMC 物理流程。

## R07

复杂度（设晶格边长 `L`，自旋数 `N=L^2`）：
- 一次 sweep：`O(N)` 次局部更新；
- 单温度模拟：`O((warmup + sample_steps * sample_interval) * N)`；
- 全温度扫描：再乘以温度点数 `K`。

空间复杂度：
- 自旋矩阵 `O(N)`；
- 采样数组 `O(sample_steps)`。

## R08

代码结构（`demo.py`）：
- `SimulationConfig`：集中管理参数（尺寸、温度网格、采样步数、随机种子）。
- `Ising2D`：模型状态与动力学
  - `metropolis_sweep`
  - `total_energy`
  - `magnetization`
- `simulate_one_temperature`：单温度统计估计。
- `run_temperature_scan`：温度扫描并返回 `pandas.DataFrame`。
- `estimate_critical_temperature`：基于峰值给出 `Tc(L)` 估计。
- `estimate_beta_exponent`：在 `T<Tc` 子区间做对数线性拟合，给出 `β` 的粗估计。

## R09

MVP 默认参数（兼顾可读性与运行速度）：
- `L = 24`
- 温度网格：`[1.8, 2.0, 2.15, 2.25, 2.35, 2.5, 2.8, 3.2]`
- `warmup_sweeps = 300`
- `sample_steps = 240`
- `sample_interval = 3`
- `seed = 20260407`

这些参数足以在数秒级别给出清晰趋势；如果需要更平滑曲线，可提高 `sample_steps` 或加密温度网格。

## R10

伪代码：

```text
for T in temperature_grid:
    init random spins
    beta = 1/T
    repeat warmup_sweeps:
        metropolis_sweep()

    for k in 1..sample_steps:
        repeat sample_interval:
            metropolis_sweep()
        record E, M

    compute e, |m|, χ, C from samples

Tc_chi = argmax_T χ(T)
Tc_cv  = argmax_T C(T)
Tc_est = (Tc_chi + Tc_cv) / 2
```

## R11

可复现性与确定性：
- 使用固定随机种子；
- 每个温度点用 `seed + 7919 * idx` 派生子种子，避免温度间随机流完全重叠；
- 输出包含完整参数与表格，方便复现实验。

程序无需外部输入，直接运行即可完成从采样到摘要结论的全流程。

## R12

输出解释（运行后会打印）：
- `temperature`：扫描温度；
- `energy_per_spin`：平均能量密度；
- `abs_magnetization`：序参量；
- `susceptibility`：磁化率；
- `heat_capacity`：比热；
- `T_chi*`、`T_cv*`：分别对应 `χ` 与 `C` 峰位；
- `Tc(L)`：二者均值形成的有限尺寸临界温度估计。

随后程序会打印 Onsager 理论值 `Tc=2.2692` 作为参照。

## R13

数值注意事项：
- 临界慢化：临界区自相关时间增长，采样方差会变大；
- 有限尺寸效应：`Tc(L)` 与无限体系 `Tc` 存在偏差；
- 网格粗糙：温度点太稀疏会使峰位估计偏粗；
- 初态影响：预热不足会引入偏差。

MVP 的目标是“展示机制”，非“高精度临界指数测量”；精度可通过更长链、更大尺寸和误差分析进一步提升。

## R14

与常见替代方案比较：
- Metropolis（当前实现）
  - 优点：实现最简单、机制透明；
  - 缺点：临界附近混合慢。
- Wolff / Swendsen-Wang 团簇算法
  - 优点：显著缓解临界慢化；
  - 缺点：实现更复杂，不适合作为最小入门 MVP。

因此本目录选择 Metropolis 作为教学与验证的第一步实现。

## R15

可扩展路线：
1. 做有限尺寸标度（多 `L`）拟合 `β/ν, γ/ν`；
2. 使用自助法/分块法估计统计误差条；
3. 用 Wolff 团簇更新替代单自旋翻转；
4. 增加 Binder cumulant 交点估计 `Tc`；
5. 加入外场 `h` 与响应函数分析。

这些扩展都可以在当前 `Ising2D + scan` 框架上增量实现。

## R16

运行方式（无交互输入）：

```bash
uv run python Algorithms/物理-统计力学-0297-临界现象_(Critical_Phenomena)/demo.py
```

若当前目录已在该算法目录下：

```bash
uv run python demo.py
```

## R17

交付核对：
- `README.md`：`R01-R18` 全部完成；
- `demo.py`：可直接运行，输出临界现象相关统计表；
- `meta.json`：与任务元数据一致（UID、学科、分类、源序号、目录信息）；
- 目录自包含，不依赖交互输入，不修改外部共享文件。

## R18

`demo.py` 源码级算法流程（8 步，对应具体函数）：

1. `SimulationConfig` 定义晶格尺寸、温度网格、采样参数与随机种子。  
2. `run_temperature_scan` 遍历温度列表，并为每个温度派生独立子种子。  
3. `simulate_one_temperature` 初始化 `Ising2D` 随机自旋矩阵，设置 `beta=1/T`。  
4. 进入预热阶段：循环调用 `Ising2D.metropolis_sweep`，将初始态推进到平衡附近。  
5. 进入采样阶段：每次先执行 `sample_interval` 个 sweep，再调用 `total_energy` 与 `magnetization` 记录样本。  
6. 用样本数组计算 `energy_per_spin`、`abs_magnetization`、`susceptibility`、`heat_capacity`，形成单温度统计行。  
7. 扫描结束后，`estimate_critical_temperature` 通过 `χ` 与 `C` 的峰值温度计算有限尺寸 `Tc(L)`。  
8. `estimate_beta_exponent` 在 `T<Tc` 数据上执行对数线性拟合，给出 `β` 粗估计并在 `main` 中打印完整摘要。  

实现未调用第三方“黑盒临界现象函数”；核心马尔可夫更新、观测量计算与峰值判定都在本文件显式展开。
