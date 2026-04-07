# 角分辨光电子能谱 (ARPES)

- UID: `PHYS-0452`
- 学科: `物理`
- 分类: `凝聚态物理`
- 源序号: `475`
- 目标目录: `Algorithms/物理-凝聚态物理-0475-角分辨光电子能谱_(ARPES)`

## R01

问题定义：把 ARPES（Angle-Resolved Photoemission Spectroscopy）从“概念说明”落成一个可运行的最小数值链路。

本条目交付的 MVP 覆盖以下闭环：
- 生成含分辨率与噪声的合成 ARPES 强度图 `I(omega, theta)`；
- 按选定能量切片做 MDC（momentum distribution curve）洛伦兹拟合，提取峰位 `k_peak(omega)`；
- 用线性模型拟合近费米能处色散，估计 `k_F` 与 `v_F`；
- 用 PyTorch 对简单紧束缚参数 `(t, mu, gamma)` 做反演校准，形成“正向模拟 + 逆向估计”闭环。

## R02

ARPES 的核心物理关系（MVP 视角）是：

`I(k, omega) ~ |M(k)|^2 * A(k, omega) * f(omega, T) + background`

其中：
- `A(k, omega)` 是单粒子谱函数，决定峰位置和宽度；
- `f(omega,T)` 是费米分布，负责占据截断；
- `M(k)` 是矩阵元，体现光电发射几何和轨道选择规则；
- `k` 来自发射角与光电子动能换算。

本实现聚焦单带、常数散射宽度与近费米面色散提取，目标是数值透明和可复现，而非材料级高精度拟合。

## R03

计算任务拆解：
1. 在角度与能量网格上构造一维紧束缚色散 `epsilon(k)`；
2. 用洛伦兹型谱函数构造 `A(k,omega)`，乘费米分布和矩阵元得到理论 `I`；
3. 引入高斯分辨率卷积与高斯噪声，得到更接近实验形态的谱图；
4. 对多条 MDC 切片做非线性拟合，得到 `k_peak(omega)` 与拟合质量 `R^2`；
5. 用 `scikit-learn` 在线性窗口拟合 `omega = v_F * k + b`，反推 `k_F`；
6. 用 `torch` 最小化谱图均方误差，反演 `(t, mu, gamma)` 并与真值对照。

## R04

模型假设（有意简化）：
- 单带一维紧束缚近似，忽略多轨道/多带混合；
- 自能宽度取常数 `gamma`，不显式构造 `Sigma(k,omega)` 频率依赖；
- 用参考动能 `hν-φ` 做角度到动量换算，忽略最终态细节与内势修正；
- 矩阵元采用光滑经验函数，而非从轨道波函数严格计算；
- 近费米面色散用线性回归，适用于局部窗口，不代表全带精确形式。

## R05

`demo.py` 实现的关键公式：

1. 动量换算（自由电子最终态近似）：
`k_parallel(theta) = 0.5123 * sqrt(hv - phi) * sin(theta)`

2. 紧束缚能带：
`epsilon(k) = -2 t cos(k a) - mu`

3. 谱函数（洛伦兹型）：
`A(k,omega) = (gamma/pi) / ((omega - epsilon(k))^2 + gamma^2)`

4. ARPES 强度模型：
`I ~ M(k) * A(k,omega) * f(omega,T) + background`

5. MDC 拟合模型：
`I_MDC(k) = bg + amp * gamma_mdc^2 / ((k-k0)^2 + gamma_mdc^2)`

6. 近费米面线性拟合：
`omega = v_F * k + b`, `k_F = -b / v_F`

## R06

算法流程：
1. 参数合法性检查（能量窗、网格密度、温度、角度范围）。
2. 构造 `theta/omega` 网格并换算 `k_parallel`。
3. 计算 `epsilon(k)`、`A(k,omega)`、费米分布和矩阵元调制，生成理想谱图。
4. 用 `scipy.ndimage.gaussian_filter` 施加能量/角度分辨率；叠加随机噪声并归一化。
5. 在 12 条能量切片上用 `scipy.optimize.curve_fit` 做 MDC 洛伦兹拟合，得到峰位序列。
6. 用 `sklearn.linear_model.LinearRegression` 拟合近费米面色散并提取 `k_F, v_F`。
7. 用 `torch` 构建可微谱图并用 Adam 优化 `t, mu, gamma, scale, bg`。
8. 打印 `summary`、MDC 表头和长表头，并执行质量断言。

## R07

复杂度估计（`Nw=n_omega`, `Nk=n_theta`, `Nm=mdc_count`, `E=torch_epochs`）：
- 光谱生成：`O(Nw * Nk)`；
- 高斯卷积：约 `O(Nw * Nk)`（常数核宽度）；
- MDC 拟合：`O(Nm * Nk * I_fit)`，`I_fit` 为拟合迭代次数；
- 线性回归：`O(Nm)`；
- Torch 反演：`O(E * Nw_ds * Nk_ds)`（下采样后执行）。

本默认参数规模下（`241 x 181` 谱图，`500` 轮优化）可在桌面环境几秒内运行。

## R08

数值稳定策略：
- 费米分布指数输入做 `clip[-80,80]`，避免上溢/下溢；
- PyTorch 正参数 (`t/gamma/scale/bg`) 通过 `softplus` 约束为正；
- MDC 拟合加入边界约束（峰宽下限、背景范围）避免非物理解；
- 卷积后再加噪并截断到非负，最后全图归一化到 `[0,1]`；
- 反演使用下采样网格，降低优化噪声与算力开销。

## R09

适用场景：
- ARPES 数据分析流程教学（从谱图到 `k_F/v_F`）；
- 算法原型验证（拟合流程、反演稳定性、回归指标）；
- 新手快速理解 `A(k,omega)`、MDC 与近费米面线性化关系。

不适用场景：
- 需要材料级精确拟合的真实实验（多体自能、矩阵元、背景模型更复杂）；
- 多带强关联体系或显著各向异性问题；
- 需要绝对能标/动量标定误差建模的高精度分析。

## R10

脚本内置正确性门槛：
1. 成功 MDC 拟合点数不少于 6（默认应为 12）；
2. `mean(MDC R^2) >= 0.85`；
3. 近费米面线性拟合 `R^2 >= 0.95`；
4. 反推出的 `k_F` 必须落在采样动量窗口内；
5. Torch 反演 `final_mse <= 1.2e-2`；
6. 反演 `t` 与真值偏差不超过 `0.25 eV`；
7. 提取峰位与真值色散的平均绝对误差 `MAE <= 0.06 eV`。

这些门槛强调“数值链路闭环可用”，不是实验误差预算标准。

## R11

默认参数（`ARPESParams`）：
- 光子能量与功函数：`hv=21.2 eV`, `phi=4.6 eV`
- 晶格常数：`a=3.2 A`
- 真值紧束缚参数：`t=0.90 eV`, `mu=0.25 eV`, `gamma=0.030 eV`
- 温度：`35 K`
- 角度网格：`0~18 deg`, `181` 点
- 能量网格：`-0.22~0.08 eV`, `241` 点
- MDC 取样：`-0.14~-0.02 eV` 共 `12` 条切片
- Torch 优化：`epochs=500`, `lr=0.04`

## R12

本地实测（命令：`uv run python demo.py`）：

Summary（关键指标）：
- `n_mdc_points = 12`
- `mean_mdc_fit_r2 = 0.988647`
- `linear_fit_r2 = 0.999839`
- `kF_from_linear_fit_Ainv = 0.534463`
- `vF_from_linear_fit_eV_A = 5.771796`
- `torch_t_fit_eV = 0.901535`
- `torch_mu_fit_eV = 0.249953`
- `torch_gamma_fit_eV = 0.037360`
- `torch_final_mse = 2.176901e-04`
- `t_relative_error = 1.706e-03`
- `mu_absolute_error_eV = 4.680e-05`

MDC 前 8 行（节选）显示 `k_peak` 随 `omega` 单调变化且 `R^2` 均接近 0.99，说明拟合切片质量较高。

## R13

结果一致性解释：
- 线性拟合 `R^2=0.999839`，说明所选能量窗内近费米面线性化成立；
- `k_F` 位于采样窗口内部，避免了外推导致的不稳定；
- Torch 反演得到 `t`、`mu` 与真值高度接近，验证了正向模拟与逆向估计链路一致；
- `gamma` 反演略高于真值（`0.037 > 0.030 eV`）符合“卷积分辨率 + 噪声会把峰宽等效放大”的常见现象。

## R14

常见失败模式与修复：
- 失败：`hv <= phi` 导致动能非正、动量换算失效。
  - 修复：增大 `hv` 或减小 `phi`，保证 `hv-phi>0`。
- 失败：MDC 拟合不收敛或 `R^2` 低。
  - 修复：减小 `noise_std`、放宽 `maxfev`、缩窄能量窗口到峰更明显区域。
- 失败：`k_F` 超出采样范围。
  - 修复：扩大角度窗口或调节 `(t,mu)` 让过零点落入观测区。
- 失败：Torch 反演震荡或损失高。
  - 修复：降低学习率、增加 epoch、适度增加下采样密度。

## R15

工程实践建议：
- 保留“真值模拟 + 反演估计”双链路，方便做单元回归测试；
- 报告中同时给出 `MDC R^2`、`linear R^2`、`final_mse` 三类指标，避免单指标误判；
- 把噪声幅度、卷积宽度作为可控参数，做灵敏度扫描；
- 如需对接真实实验数据，可先替换 `simulate_arpes` 输出为数据读取，再复用后续拟合与反演模块。

## R16

可扩展方向：
- 引入频率依赖自能 `Sigma(omega)`，替代常数 `gamma`；
- 从单带扩展到多带并加入轨道矩阵元；
- 引入二维动量切片（`k_x, k_y`）与等能面拟合；
- 增加背景模型（Shirley / polynomial）与仪器分辨函数联合拟合；
- 用贝叶斯或不确定性传播评估 `k_F/v_F` 置信区间。

## R17

本目录交付内容：
- `demo.py`：可运行 MVP（`numpy + scipy + pandas + scikit-learn + torch`）；
- `README.md`：R01-R18 完整说明、公式、指标、限制与扩展；
- `meta.json`：任务元数据（UID/学科/分类/源序号/目录路径）。

运行方式：

```bash
cd Algorithms/物理-凝聚态物理-0475-角分辨光电子能谱_(ARPES)
uv run python demo.py
```

无需交互输入，程序会直接输出摘要表、MDC 表头和检查结果。

## R18

`demo.py` 源码级算法流拆解（8 步，非黑盒）：
1. `ARPESParams` 与 `check_params` 固定并校验实验与数值参数（光子能量、角度窗、网格、噪声、优化超参）。
2. `angle_to_k_parallel` 将角度网格映射为动量网格；`tight_binding_dispersion` 计算带结构 `epsilon(k)`。
3. `simulate_arpes` 显式构造谱函数 `A(k,omega)`、费米分布 `f(omega,T)` 与矩阵元 `M(k)`，合成理想强度。
4. `simulate_arpes` 内部进一步执行高斯卷积（能量/角度分辨率）与噪声叠加，得到可测量形态的归一化谱图。
5. `extract_mdc_peaks` 对每个目标能量切片调用 `curve_fit` 拟合洛伦兹峰，输出 `k_peak(omega)`、峰宽与拟合 `R^2`。
6. `fit_linear_dispersion` 用 `LinearRegression` 拟合 `omega-k` 关系，提取 `v_F` 与 `k_F`。
7. `torch_refine_tb_params` 把同一物理模型写成可微计算图，用 Adam 优化 `(t,mu,gamma,scale,bg)`，最小化谱图 MSE。
8. `main` 汇总 `summary` 与表头输出，并执行质量门槛断言（拟合质量、窗口覆盖、反演误差、能量一致性）。

第三方库仅承担数值求解与优化器角色；ARPES 的核心物理流程（谱函数构造、MDC 拟合、色散提取、参数反演）均在源码中展开实现。
