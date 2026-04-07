# 标度律 (Scaling Laws)

- UID: `PHYS-0296`
- 学科: `物理`
- 分类: `统计力学`
- 源序号: `299`
- 目标目录: `Algorithms/物理-统计力学-0299-标度律_(Scaling_Laws)`

## R01

标度律（Scaling Laws）描述系统在临界区附近的“无量纲幂律关系”：当微观细节被粗粒化后，宏观量只保留指数结构与普适函数。  
在统计力学中，它解释了为什么不同材料在二级相变附近会呈现相同的临界指数组合。

本任务聚焦有限尺寸系统的标度律演示：
- `|m|(Tc, L) ~ L^{-β/ν}`
- `χ(Tc, L) ~ L^{γ/ν}`
- `|m|(T, L) L^{β/ν}` 与 `χ(T, L) L^{-γ/ν}` 对 `x=(T-Tc)L^{1/ν}` 发生数据塌缩。

## R02

MVP 目标是用最小可运行程序，直接从 Monte Carlo 采样数据中恢复“幂律 + 塌缩”两类标度证据，而不是依赖黑盒库函数。  
`demo.py` 会：
- 在多个晶格尺寸 `L` 和温度 `T` 上运行 2D Ising Metropolis 采样；
- 提取接近 `Tc` 的截面数据并做对数线性拟合；
- 用理论指数 `β/ν=1/8, γ/ν=7/4, ν=1` 计算塌缩变量并给出定量质量指标。

## R03

模型（二维 Ising，无外场，`J=1, k_B=1`）：
- 自旋：`s_i ∈ {+1,-1}`
- 哈密顿量：`H = - Σ_{<i,j>} s_i s_j`
- 周期边界条件

观测量定义：
- `m = M/N`，其中 `M=Σ_i s_i`, `N=L^2`
- `|m|` 作为序参量
- `χ = N ( <m^2> - <m>^2 ) / T`
- `e = E/N` 仅用于辅助监控能量趋势

## R04

本目录使用的核心标度关系：

1. 临界截面（`T=Tc`）有限尺寸幂律：
- `|m|(Tc, L) = A_m L^{-β/ν}`
- `χ(Tc, L) = A_χ L^{γ/ν}`
- `χ_max(L) = A'_χ L^{γ/ν}`（数值上常用于更稳健估计 `γ/ν`）

2. 有限尺寸标度（FSS）塌缩形式：
- `|m|(T, L) = L^{-β/ν} F_m((T-Tc)L^{1/ν})`
- `χ(T, L) = L^{γ/ν} F_χ((T-Tc)L^{1/ν})`

二维 Ising 理论值：
- `Tc = 2 / ln(1 + sqrt(2)) ≈ 2.269185`
- `β = 1/8`, `γ = 7/4`, `ν = 1`

## R05

算法路线：
- 采样器：Metropolis 单自旋翻转（最小且透明）；
- 扫描维度：多 `L` × 多 `T`；
- 统计阶段：预热后按固定间隔采样，估计 `|m|` 与 `χ`；
- 拟合阶段：在近 `Tc` 截面上执行 `log-log` 线性回归获取指数比值估计。

该路线优先保证“机制可追踪”，而非一次性追求高精度指数。

## R06

数值流程：
1. 为每组 `(L,T)` 初始化随机自旋；
2. 进行 `replicas_per_point` 条独立链采样并分别预热 `warmup_sweeps`；
3. 每条链每隔 `sample_interval` 个 sweep 记录一次 `M,E`；
4. 合并多链样本后计算 `|m|`、`χ`、`e`；
5. 汇总为 `DataFrame`；
6. 对每个 `L` 选取最接近 `Tc` 的点，拟合 `|m|` 与 `χ` 的尺寸幂律；
7. 对每个 `L` 提取温度扫描中的 `χ_max`，拟合 `χ_max(L)` 获取更稳健的 `γ/ν`；
8. 计算塌缩变量 `x, y_m, y_χ`，并用分箱变异系数评估塌缩质量。

## R07

复杂度估计：
- 单次 sweep：`O(N)`（`N=L^2` 次局部尝试）；
- 单组 `(L,T)`：`O((warmup + sample_steps*sample_interval)*N)`；
- 全扫描：乘以尺寸数 `K_L` 与温度数 `K_T`。

空间复杂度：
- 自旋矩阵 `O(N)`；
- 采样数组 `O(sample_steps)`；
- 汇总表 `O(K_L * K_T)`。

## R08

`demo.py` 模块结构：
- `SimulationConfig`：统一配置尺寸、温度、采样步数与随机种子；
- `Ising2D`：模型状态、Metropolis 更新、能量与磁化计算；
- `simulate_point`：单个 `(L,T)` 采样并返回观测量；
- `run_scan`：多尺寸多温度批量扫描；
- `extract_near_tc_slice`：抽取近 `Tc` 截面；
- `fit_log_log`：幂律拟合与 `R^2` 计算；
- `compute_scaled_frame`：构造塌缩变量与质量指标；
- `main`：打印结果与基本自检。

## R09

MVP 默认参数（速度与稳定性折中）：
- `lattice_sizes = (10, 14, 18, 22)`
- `temperatures = (2.15, 2.22, 2.27, 2.32, 2.39)`
- `warmup_sweeps = 140`
- `sample_steps = 140`
- `sample_interval = 2`
- `replicas_per_point = 3`
- `seed = 20260407`

这些参数通常可在较短时间内展示清晰标度趋势；若需更平滑结果，可增加采样长度或扩展尺寸序列。

## R10

伪代码：

```text
for L in lattice_sizes:
    for T in temperatures:
        init random Ising state
        warm up using Metropolis sweeps
        sample M,E along the Markov chain
        compute |m|, χ, e

collect all rows into table
select near-Tc row for each L
fit log(|m|) ~ a_m + b_m log(L)      => β/ν = -b_m
select peak-χ row for each L
fit log(χ_max) ~ a_x + b_x log(L)    => γ/ν =  b_x

build collapse variables:
x   = (T - Tc) * L^(1/ν)
ym  = |m| * L^(β/ν)
ychi= χ / L^(γ/ν)
measure collapse CV in x-bins
```

## R11

可复现性策略：
- 顶层固定 `seed`；
- 每个 `(L,T)` 用可重复偏移生成子种子，并在点内再派生多副本链种子；
- 全流程无交互输入，运行即输出完整配置与数据表。

因此同环境重复运行可得到统计上接近的结果与相同执行路径。

## R12

输出包含三部分：
1. 原始扫描表：`lattice_size, temperature, abs_magnetization, susceptibility, energy_per_spin`；
2. 近 `Tc` 尺寸截面表：每个 `L` 一行，用于幂律拟合；
3. `χ` 峰值截面表：每个 `L` 一行，用于 `γ/ν` 的稳健拟合；
4. 拟合与塌缩摘要：
- `beta_over_nu_est`, `gamma_over_nu_est`
- `collapse_cv_m`, `collapse_cv_chi`（越小通常表示塌缩越好）
- 趋势自检 `PASS/CHECK_MANUALLY`。

## R13

数值注意事项：
- 临界慢化会增大自相关，短链会让拟合波动；
- 近 `Tc` 取点若太少，指数估计会有系统偏差；
- 有限尺寸较小（如 `L<=22`）时，指数更应视作“趋势验证”；
- `χ` 对采样噪声敏感，需适度增加 `sample_steps` 才更稳。

MVP 以“展示标度机制”为目标，不等同于高精度临界指数论文级估计。

## R14

替代算法对比：
- Metropolis（当前）
  - 优点：实现短、可读、每一步物理意义清晰；
  - 缺点：临界区混合慢。
- Wolff/Swendsen-Wang 团簇算法
  - 优点：显著减轻临界慢化，指数估计更高效；
  - 缺点：实现复杂度更高，不适合作为最小起步版本。

因此本目录优先用 Metropolis 建立可验证基线。

## R15

可扩展方向：
1. 增加 `L` 点数并做有限尺寸外推；
2. 引入自助法或分块法估计误差条；
3. 用 Binder cumulant 交点联合估计 `Tc` 与 `ν`；
4. 切换到 Wolff 更新比较临界慢化改善；
5. 对 `F_m, F_χ` 做参数化拟合而非仅靠 CV 指标。

## R16

运行方式（无交互）：

```bash
uv run python Algorithms/物理-统计力学-0299-标度律_(Scaling_Laws)/demo.py
```

若当前已在该目录：

```bash
uv run python demo.py
```

## R17

交付核对：
- `README.md`：`R01-R18` 已完整填充；
- `demo.py`：已替换全部占位符，直接运行输出结果；
- `meta.json`：保留并与任务元数据一致（UID/学科/分类/源序号/目录信息）；
- 仅修改本算法自有目录内文件。

## R18

`demo.py` 源码级算法流（8 步）：

1. `SimulationConfig` 固定 `L` 列表、温度网格、采样长度、理论 `Tc` 与指数参数。  
2. `run_scan` 双层循环遍历 `(L,T)`，为每个点生成确定性子种子并调用 `simulate_point`。  
3. `simulate_point` 在每个 `(L,T)` 上运行多条独立链：每条链创建 `Ising2D`，先预热再按间隔采样 `M,E`。  
4. `simulate_point` 合并多链样本并计算 `|m|`、`χ`、`e`，形成单点统计记录。  
5. `extract_near_tc_slice` 为每个 `L` 选取最接近 `Tc` 的点，用于 `|m|` 的临界截面拟合。  
6. `extract_peak_susceptibility_slice` 为每个 `L` 选取 `χ` 峰值点，用于 `χ_max(L)` 幂律拟合。  
7. `fit_log_log` 计算 `β/ν`、`γ/ν` 估计及 `R^2`，`compute_scaled_frame` 计算塌缩变量与 CV。  
8. `main` 打印扫描表、两类截面表、拟合结果、塌缩质量和趋势自检，完成端到端最小验证。  

流程中没有调用第三方“标度律黑盒”；幂律拟合、塌缩变量构造与质量度量都在源码中显式实现。
