# 量子霍尔效应 (Quantum Hall Effect)

- UID: `PHYS-0194`
- 学科: `物理`
- 分类: `低维物理`
- 源序号: `195`
- 目标目录: `Algorithms/物理-低维物理-0195-量子霍尔效应_(Quantum_Hall_Effect)`

## R01

问题定义：将“量子霍尔效应（QHE）”从概念描述落地为可运行的最小数值 MVP，展示二维电子气在强磁场下的 Landau 量子化与霍尔台阶输运。

本条目的 MVP 目标闭环：
- 固定电子面密度 `n_e`，扫描磁场 `B`；
- 构造 Landau 能级并加入无序展宽，数值求解每个 `B` 对应化学势 `mu`；
- 计算 `sigma_xy, sigma_xx` 以及 `rho_xy, rho_xx` 曲线；
- 通过 `pandas` 输出样本表；
- 用 `sklearn` 在低磁场段拟合经典 Hall 斜率反推载流子密度；
- 用 `PyTorch` 对跃迁宽度参数做小型反演校准，形成“正向模拟 + 逆向估计”闭环。

## R02

MVP 采用的物理核心是整数 QHE 的简化输运图景：
- 磁场下能谱离散为 Landau 能级 `E_n = hbar*omega_c*(n+1/2)`，`omega_c = eB/m*`；
- 每个 Landau 能级每单位面积简并度 `g(B) = eB/h`；
- 无序导致态密度展宽（此处用高斯展宽近似）；
- 固定总电子密度时，化学势 `mu(B)` 随磁场变化；
- `sigma_xy` 随 `mu` 穿越 Landau 中心形成近整数量子台阶，`sigma_xx` 在台阶过渡处出现峰值。

该实现刻意追求“可算、可解释、可验证”，不追求真实器件级误差预算。

## R03

计算任务拆解：
1. 在给定 `B` 上生成有限个 Landau 能级 `E_n(B)`；
2. 用高斯核把每个能级展宽并叠加，得到 `DOS(E;B)`；
3. 通过 `n_e = ∫ DOS(E;B) f(E,mu,T) dE` 数值求根得到 `mu(B)`；
4. 用平滑阶跃函数累计得到 `sigma_xy/(e^2/h)`；
5. 用 Landau 中心附近的高斯峰模型得到 `sigma_xx/(e^2/h)`；
6. 由电导张量反演电阻率 `rho_xy, rho_xx`；
7. 在低磁场段线性拟合 `rho_xy(B)` 并估计 `n_e`；
8. 构造带噪“观测”，用 `torch` 反演 `gamma_xy` 与 `sigma_xx0`。

## R04

模型假设（有意简化）：
- 只处理二维电子气的整数 QHE 形态，不含分数量子霍尔多体关联；
- 忽略 Zeeman 劈裂与自旋分辨结构（等效并入模型参数）；
- 无序影响由固定高斯宽度 `gamma_dos` 表示；
- `sigma_xy` 用平滑阶跃近似拓扑台阶转换，`sigma_xx` 用 Landau 中心附近峰函数近似；
- 温度通过费米分布进入，电子-电子相互作用不显式建模。

## R05

`demo.py` 中关键公式：

1. Landau 能级：
`E_n(B) = (hbar*e*B/(m* m_e)/e)*(n+1/2)`（单位 eV）

2. 展宽 DOS：
`DOS(E) = Σ_n g(B) * G(E - E_n, gamma_dos)`，
`G(x,s) = exp(-x^2/(2s^2)) / (s*sqrt(2*pi))`

3. 密度约束求 `mu`：
`n_e = ∫ DOS(E) * f(E,mu,T) dE`

4. Hall 电导（量子单位）：
`sigma_xy_q = Σ_n sigmoid((mu - E_n)/gamma_xy)`

5. 纵向电导（量子单位）：
`sigma_xx_q = sigma_xx0 * Σ_n exp(-(mu-E_n)^2/(2*gamma_xx^2))`

6. 张量反演：
`rho_xy = sigma_xy / (sigma_xy^2 + sigma_xx^2)`，
`rho_xx = sigma_xx / (sigma_xy^2 + sigma_xx^2)`

## R06

算法流程：
1. 校验输入参数（网格大小、磁场区间、温度、展宽参数）。
2. 在 `B` 网格循环，构造 Landau 能级与 DOS。
3. 用 `scipy.optimize.brentq` 对密度方程求根得到 `mu(B)`。
4. 计算 `sigma_xy_q, sigma_xx_q`，再转成 SI 电导/电阻率。
5. 聚合为 `pandas.DataFrame`，补充台阶整数与台阶误差列。
6. 用 `sklearn.linear_model.LinearRegression` 在低场段拟合 `rho_xy(B)`，估计密度与 `R^2`。
7. 合成带噪观测，用 `torch.optim.Adam` 反演 `gamma_xy` 与 `sigma_xx0`。
8. 执行断言门槛，打印 summary、曲线样例与台阶区域样例。

## R07

复杂度估计（`N_B=n_B`, `N_E=n_energy`, `N_L=max_landau_level+1`, `I_root` 为根求解迭代次数，`E_t=torch_epochs`）：
- 正向主循环：约 `O(N_B * (N_E*N_L + I_root*N_E))`；
- 低场线性拟合：`O(N_B)`；
- Torch 反演：`O(E_t * N_B * N_L)`。

默认配置（`N_B=180`, `N_E=2400`, `N_L=29`, `E_t=500`）在本地可秒级运行，适合 MVP 验证。

## R08

数值稳定策略：
- 费米分布指数输入裁剪到 `[-80, 80]`，避免指数溢出；
- 密度方程采用显式 bracket + `brentq`，保证一维单调方程稳健求根；
- 电阻率分母加极小正数，避免极端点除零；
- PyTorch 待拟合参数通过 `softplus` 强制为正；
- 带噪观测的 `rho_xx` 做下截断，避免噪声导致非物理负值。

## R09

适用场景：
- 低维物理课程中 QHE 数值链路教学；
- 输运曲线分析流程原型验证（台阶检测、参数敏感性）；
- 在无实验数据时进行“物理一致”的合成数据基准测试。

不适用场景：
- 分数量子霍尔、边缘态干涉、强关联多体效应；
- 真实样品中复杂散射机制、接触电阻、几何非理想的精确拟合；
- 需要严格误差条与实验标定流程的论文级反演。

## R10

脚本内置质量门槛：
1. 台阶候选点数 `plateau_count >= 25`；
2. 台阶误差均值 `plateau_mae <= 0.12`；
3. `sigma_xy(B)` 随 `B` 递增的比例 `<= 0.06`（应整体随 `B` 增大而下降）；
4. 低场线性拟合 `R^2 >= 0.97`；
5. 低场拟合反推密度相对误差 `<= 0.18`；
6. Torch 反演最终损失 `<= 2.5e-2`；
7. Torch 反演的 `gamma_xy` 误差 `<= 0.45 meV`。

这些阈值用于保证“流程正确 + 指标合理”，不是器件实验验收标准。

## R11

默认参数（`QHEParams`）：
- `electron_density_m2 = 3.2e15`
- `effective_mass_ratio = 0.067`（GaAs 常用量级）
- `temperature_K = 1.6`
- 磁场：`2.0 ~ 14.0 T`，`180` 点
- Landau 阶数：`0~28`
- 能量网格：`-6 ~ 190 meV`，`2400` 点
- 展宽参数：`gamma_dos=1.8 meV`, `gamma_xy=0.55 meV`, `gamma_xx=1.6 meV`
- `sigma_xx0_quantum = 0.33`
- Torch：`epochs=500`, `lr=0.035`, 固定随机种子 `7`

## R12

本地实测（命令：`uv run python Algorithms/物理-低维物理-0195-量子霍尔效应_(Quantum_Hall_Effect)/demo.py`）：

Summary：
- `n_B_points = 180`
- `B_range_T = (2.0, 14.0)`
- `mu_meV_range = (15.004319393353406, 29.903084607382304)`
- `sigma_xy_q_range = (0.9949807469974359, 6.677755180622068)`
- `rho_xx_q_peak = 0.19776957092741795`
- `low_field_slope_ohm_per_T = 1858.2013164049424`
- `low_field_r2 = 0.9812979449073103`
- `density_est_m2 = 3.358898209444925e15`
- `torch_gamma_xy_fit_meV = 0.5473197128308577`
- `torch_sigma_xx0_fit_q = 0.32999098512987485`
- `torch_final_loss = 1.2763991950307674e-04`
- `plateau_count = 79`
- `plateau_mae = 0.09368647527095252`

前 10 行曲线样例显示：`sigma_xy_q` 随 `B` 增大整体下降，`rho_xx_q` 在过渡区域附近抬升，符合台阶-峰值互补特征。

## R13

结果解释：
- `sigma_xy_q` 覆盖约 `~1` 到 `~6.68`，与给定磁场区间对应的填充因子尺度一致；
- `sigma_xy_positive_jump_ratio = 0.0`，说明曲线在离散采样上严格单调下降，无异常反跳；
- 低场反推密度相对误差约 `4.97%`，表明数值输运与经典 Hall 关系在低场段基本一致；
- Torch 反演得到 `gamma_xy_fit = 0.5473 meV`（真值 `0.55 meV`），误差约 `0.00268 meV`，说明参数可辨识性良好；
- `torch_final_loss ~ 1.28e-4`，验证了“带噪观测 -> 参数回归”链路是闭合且稳定的。

## R14

常见失败模式与修复：
- 失败：`solve_mu_for_density` 报错 root not bracketed。
  - 修复：扩大能量窗口（提高 `energy_max_meV` 或降低 `energy_min_meV`），或增加 Landau 阶数。
- 失败：台阶误差过大（`plateau_mae` 超阈值）。
  - 修复：减小 `gamma_xy_meV`、降低温度或增加磁场上限。
- 失败：低场拟合 `R^2` 过低。
  - 修复：将低场拟合区间扩展到更线性的区域，或减小展宽参数。
- 失败：Torch 反演损失震荡。
  - 修复：减小学习率、增大 epoch，必要时缩小观测噪声幅度。

## R15

工程实践建议：
- 将“正向模拟指标”和“反演指标”同时纳入回归测试，避免只盯单一曲线形态；
- 保持单位统一（eV、meV、SI）并在函数边界显式转换；
- 对 `B` 网格和能量网格做参数化，便于以后做分辨率与性能权衡；
- 若接入真实实验数据，可直接替换 DataFrame 中观测列，再复用低场拟合与 Torch 反演模块。

## R16

可扩展方向：
- 加入自旋劈裂（Zeeman）与简并度拆分，构造更细台阶结构；
- 扩展到分数量子霍尔经验模型（例如复合费米子有效填充框架）；
- 引入边缘态通道和 Landauer-Büttiker 多端口输运描述；
- 对参数反演改用贝叶斯方法，输出置信区间而非点估计；
- 增加温度扫描与相图输出（`B-T` 平面）。

## R17

本目录交付物：
- `README.md`：R01-R18 完整技术说明；
- `demo.py`：可运行 MVP（`numpy + scipy + pandas + scikit-learn + torch`）；
- `meta.json`：与任务元数据保持一致（UID、学科、分类、源序号、目录路径）。

运行方式：

```bash
cd Algorithms/物理-低维物理-0195-量子霍尔效应_(Quantum_Hall_Effect)
uv run python demo.py
```

无需交互输入，运行后直接输出 summary 和样例表。

## R18

`demo.py` 源码级算法流拆解（8 步，非黑盒）：
1. `QHEParams` 固定物理常数、网格和优化超参数；`main` 初始化执行链路。
2. `qhe_forward` 对每个 `B` 调用 `landau_levels_eV` 构造 `E_n(B)`，并用 `dos_from_landau` 叠加高斯展宽 DOS。
3. `solve_mu_for_density` 通过 `density_error(mu)=∫DOS*f - n_e` 建立标量方程，再用 `brentq` 在有符号区间中求根，得到 `mu(B)`。
4. 在同一循环内显式计算 `sigma_xy_q`（sigmoid 台阶和）与 `sigma_xx_q`（Landau 中心高斯峰和）。
5. 把量子单位电导换算成 SI 电导，并通过张量反演得到 `rho_xy, rho_xx`，收集为 `DataFrame`。
6. `fit_low_field_density` 在低场子集上做线性回归 `rho_xy = slope*B + intercept`，按 `n=1/(e*slope)` 回推载流子密度。
7. `torch_fit_transition` 先对正向曲线加噪得到“观测”，再把 `gamma_xy` 与 `sigma_xx0` 设为可训练参数，最小化 `sigma_xy` 与 `rho_xx` 的联合 MSE。
8. `validate` 执行一组物理与数值门槛断言，`main` 打印 summary 与样例表完成交付。

第三方库仅承担数值求解、回归与优化器功能；Landau 建模、密度守恒求解、输运构造和评估逻辑均在源码中展开实现。
