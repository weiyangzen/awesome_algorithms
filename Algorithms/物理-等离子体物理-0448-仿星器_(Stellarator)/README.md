# 仿星器 (Stellarator)

- UID: `PHYS-0428`
- 学科: `物理`
- 分类: `等离子体物理`
- 源序号: `448`
- 目标目录: `Algorithms/物理-等离子体物理-0448-仿星器_(Stellarator)`

## R01

仿星器（Stellarator）是一类通过三维外部线圈直接产生扭转磁场、实现等离子体稳态磁约束的装置。  
和托卡马克相比，仿星器不依赖强环向等离子体电流来建立旋转变换（rotational transform），因此天然具备稳态运行与减弱电流驱动不稳定性的潜力。

本条目实现的是“磁力线追踪”最小模型：把磁力线方程简化到 `r-θ-φ` 坐标中的非轴对称摄动系统，用来演示磁面保持、径向漂移与有效旋转变换的数值估计。

## R02

问题设定（MVP 范围）：

- 采用约化哈密顿形式，只追踪磁力线，不解完整 MHD 平衡；
- 自变量选取环向角 `φ`，状态变量为归一化小半径 `r` 与极向角 `θ`；
- 在轴对称背景上叠加一个螺旋摄动项，模拟仿星器三维场效应；
- 通过多条初值磁力线的长时间追踪，评估约束质量。

这保证了模型足够小、可复现、可解释，同时仍保留“非轴对称磁几何导致磁面结构变化”的核心物理要点。

## R03

MVP 目标：

- 给出可运行的仿星器磁力线 ODE 模型并数值积分；
- 输出每条磁力线的径向波动范围（`r_span`）和是否越过边界阈值；
- 用线性回归拟合 `θ(φ)` 斜率，估计每条磁力线的有效旋转变换 `iota_eff`；
- 用 PyTorch 自动微分评估“约束损失对摄动幅值 `epsilon` 的梯度”；
- 脚本可直接通过 `uv run python demo.py` 运行，无需交互输入。

## R04

`demo.py` 使用的约化模型：

设相位 `phase = m*θ - n*Nfp*φ`，其中：

- `Nfp`：场周期数（field periods）
- `m, n`：螺旋模数
- `epsilon`：非轴对称摄动强度

磁力线方程写为：

1. `dr/dφ = 0.5 * epsilon * m * r * sin(phase)`
2. `dθ/dφ = iota0 + iota1 * r^2 + epsilon * cos(phase)`

其中 `iota0 + iota1*r^2` 给出背景旋转变换剖面，`epsilon` 项刻画仿星器三维线圈引入的空间谐波影响。

## R05

数值离散与复杂度：

- 每条磁力线使用 `scipy.integrate.solve_ivp(RK45)` 在 `φ ∈ [0, 2π * n_toroidal_turns]` 上积分；
- 每个整环向周回（`Δφ = 2π`）记录一次截面点，形成 Poincaré 风格序列；
- 共追踪 `n_lines` 条磁力线，主复杂度约 `O(n_lines * n_steps)`；
- 存储“逐周回样本 + 汇总表”，空间复杂度约 `O(n_lines * n_turns)`。

## R06

运行输出包括：

- 全局统计：
  - `confined_fraction`（未越界磁力线比例）
  - `avg_iota_eff`（平均有效旋转变换）
  - `avg_radial_span`（平均径向摆幅）
- PyTorch 诊断：
  - `torch_loss = mean((r_end - r0)^2)`
  - `d(loss)/d(eps)`
  - `suggested_epsilon`（单步梯度下降建议值）
- 每条磁力线表格：`r0, r_mean, r_span, r_max, iota_eff, edge_crossed`
- 自动验证项：`finite_values / confinement_ratio / iota_range / linear_fit_quality / autograd_sensitivity`

## R07

优点：

- 保留仿星器非轴对称磁几何的关键影响，同时模型规模很小；
- 物理方程、积分、诊断指标都在源码中显式实现，可直接审计；
- 同时包含“前向模拟（SciPy）+ 参数敏感性（PyTorch）”两类分析。

局限：

- 不是 VMEC/BOOZER 级平衡重建，不含真实线圈几何；
- 未包含碰撞、输运、压力剖面自洽与湍流效应；
- `torch` 部分采用可微 Euler 近似，仅用于灵敏度示例。

## R08

前置知识：

- 磁约束基本概念（磁面、磁力线、旋转变换）；
- 常微分方程初值问题与 Runge-Kutta 积分；
- 回归斜率的物理解释（`dθ/dφ` 对应 `iota` 估计）；
- 自动微分的链式法则思想。

依赖环境：

- Python `>=3.10`
- `numpy`
- `scipy`
- `pandas`
- `scikit-learn`
- `torch`

## R09

适用场景：

- 教学或原型验证中的“仿星器磁力线追踪最小样例”；
- 比较不同 `epsilon`、`iota` 剖面对磁面保持能力的影响；
- 作为更高保真磁场工具链前的快速 sanity check。

不适用场景：

- 需要工程级三维平衡和线圈反算设计；
- 需要定量预测热输运、功率负载、壁相互作用；
- 需要与实验诊断做高精度一一对应。

## R10

正确性直觉：

1. 当 `epsilon` 较小，`r` 的波动应有限，多数磁力线不应越过边界阈值；
2. `θ` 随 `φ` 总体近线性增长，线性拟合斜率可作为 `iota_eff` 估计；
3. 外层磁力线通常比内层更易出现更大 `r_span`；
4. 若输出出现 NaN/Inf 或大量越界，通常是参数过激或积分设置不合理；
5. `d(loss)/d(eps)` 非零说明模型对三维摄动强度具有可辨识敏感性。

## R11

稳定性与鲁棒性处理：

- `validate_config` 对关键参数做范围约束（模数、线数、转数、半径区间、摄动幅度）；
- `solve_ivp` 使用严格容差（`rtol=1e-8`, `atol=1e-10`）并限制 `max_step`；
- 每条线若积分失败立即抛错，避免静默污染结果；
- 验证阶段检查数值有限性和拟合质量（`R^2 > 0.95`）。

## R12

关键参数（`StellaratorConfig`）：

- `n_field_periods`：场周期数，决定三维场重复节奏；
- `m_pol, n_tor`：螺旋谐波模式数；
- `iota0, iota1`：背景旋转变换剖面参数；
- `helical_perturbation`：非轴对称摄动强度（核心扫描参数）；
- `n_lines`：同时追踪的磁力线数量；
- `n_toroidal_turns`：追踪时长（周回数）；
- `r0_min, r0_max`：初始半径采样区间；
- `edge_threshold`：判定“越界/丢失”的径向阈值。

## R13

保证类型说明：

- 近似比保证：N/A（连续动力系统模拟，不是离散优化近似算法）；
- 随机成功率保证：N/A（默认流程确定性，无随机采样）。

可执行保证：

- 固定参数、无交互输入、可重复运行；
- 内置 PASS/FAIL 检查，失败时返回非零退出码。

## R14

常见失效模式：

1. `helical_perturbation` 过大导致径向振荡剧烈、约束显著恶化；
2. `n_toroidal_turns` 太小，统计不足，难判断长期行为；
3. 初始半径区间过靠边界，易出现“看似失稳”的偏置结论；
4. 拟合 `iota_eff` 时若数据点过少，回归不稳定；
5. 过松积分精度可能把数值误差误判为物理漂移。

## R15

可扩展方向：

1. 用真实磁场数据表（如 Boozer/VMEC 导出）替换约化解析场；
2. 计算更标准的 Poincaré 截面统计（岛宽、混沌层厚度）；
3. 对 `epsilon`、`iota0/iota1` 做网格扫描并形成相图；
4. 把 PyTorch 可微模块扩展为小规模参数反演；
5. 加入简单碰撞/扩散项，构造输运近似。

## R16

相关主题：

- 托卡马克与仿星器磁约束差异；
- 旋转变换、磁剪切与有理面共振；
- 磁岛与随机磁场线；
- Hamiltonian field-line tracing；
- 仿星器优化（准对称、准等动力学）思想。

## R17

运行方式（无交互）：

```bash
cd Algorithms/物理-等离子体物理-0448-仿星器_(Stellarator)
uv run python demo.py
```

交付核对：

- `README.md` 的 `R01-R18` 已完整填写；
- `demo.py` 不含占位符并可直接运行；
- `meta.json` 与任务元数据保持一致；
- 目录可独立用于验证。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `StellaratorConfig` 固化 `Nfp, (m,n), iota0/iota1, epsilon, 线数, 周回数, 初值范围`。  
2. `validate_config` 检查参数物理/数值范围，提前阻断无效输入。  
3. `field_line_rhs` 根据约化方程计算 `dr/dφ` 与 `dθ/dφ`，显式编码非轴对称相位 `mθ-nNfpφ`。  
4. `trace_single_line` 调用 `solve_ivp(RK45)` 对单条线积分，并在每个环向周回采样截面点。  
5. `estimate_effective_iota` 用 `LinearRegression` 拟合连续 `θ(φ)` 斜率，得到 `iota_eff` 与 `R^2`。  
6. `run_experiment` 批量追踪多条初始线，汇总 `r_span/r_max/edge_crossed/iota_eff` 指标与总体统计。  
7. `torch_confinement_sensitivity` 用可微 Euler rollout 计算约束损失 `mean((r_end-r0)^2)`，并反向传播得到 `d(loss)/d(epsilon)`。  
8. `main` 打印结构化表格与验证项，全部通过则输出 `Validation: PASS`，否则非零退出。

第三方库没有被当作黑盒求解器：

- `numpy`：数组与基础数值运算；
- `scipy`：通用 ODE 积分器（被明确提供微分方程）；
- `pandas`：表格化结果展示；
- `scikit-learn`：线性回归估计 `iota_eff`；
- `torch`：自动微分求敏感度，核心动力学更新在源码显式编写。
