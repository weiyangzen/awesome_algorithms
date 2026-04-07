# 爱因斯坦关系 (Einstein Relation)

- UID: `PHYS-0309`
- 学科: `物理`
- 分类: `统计力学`
- 源序号: `312`
- 目标目录: `Algorithms/物理-统计力学-0312-爱因斯坦关系_(Einstein_Relation)`

## R01

爱因斯坦关系（Einstein Relation）连接了扩散与线性响应：

`D = mu * k_B * T`

其中 `D` 是扩散系数，`mu` 是迁移率（单位力下的漂移速度系数），`T` 是温度，`k_B` 是玻尔兹曼常数。该关系是涨落-耗散定理在布朗运动场景下的经典结果。

## R02

本条目的 MVP 目标是“用一个可运行、可审计的最小仿真同时估计 `D` 与 `mu`，并在数值上验证上式”，而不是只写理论公式。  
`demo.py` 通过两组过阻尼朗之万模拟完成：

- 零外力实验：由 MSD（均方位移）斜率估计 `D`；
- 常力驱动实验：由平均位移斜率估计 `mu`；
- 对比 `D` 与 `mu*kBT` 的一致性。

## R03

采用一维过阻尼布朗粒子模型（Euler-Maruyama 离散）：

`x_{n+1} = x_n + mu*F*dt + sqrt(2*D*dt)*eta_n`

- `eta_n ~ N(0,1)`；
- 当 `F=0` 时仅扩散；
- 当 `F!=0` 时叠加确定性漂移。

在该模型下：

- `MSD(t) = <(x(t)-x(0))^2> = 2Dt`；
- `<x(t)-x(0)> = mu*F*t`。

## R04

本实现的估计链路：

1. 由零外力轨道拟合 `MSD(t) = a_msd*t + b_msd`，取 `D_hat = a_msd/2`；
2. 由恒力轨道拟合 `mean_disp(t) = a_drift*t + b_drift`，取 `v_hat = a_drift`；
3. 迁移率估计：`mu_hat = v_hat/F`；
4. 爱因斯坦预测：`D_from_relation = mu_hat*kBT`；
5. 评估比值：`ratio = D_hat / D_from_relation`，理论上应接近 1。

## R05

算法设计强调“最小且透明”：

- 轨道生成：手写 Euler-Maruyama，不调用黑盒 SDE 求解器；
- 统计提取：在固定时间点记录 `mean displacement` 与 `MSD`；
- 参数拟合：使用显式线性最小二乘 `numpy.linalg.lstsq`；
- 结果验证：检查相对误差、`ratio` 偏差和线性拟合 `R^2`。

## R06

复杂度分析（`N` 粒子数，`T` 时间步数）：

- 单次仿真：每步一次长度 `N` 的向量更新，时间复杂度 `O(NT)`；
- 内存：仅保留当前粒子位置 `O(N)`，外加采样记录 `O(T/stride)`；
- 两组实验（`F=0` 与 `F!=0`）总体仍为同阶线性复杂度。

在默认参数（`N=6000`, `T=5000`）下可快速运行。

## R07

`demo.py` 输出三部分：

- 配置参数（`kbt`, `mobility_true`, `external_force`, `dt`, `n_steps` 等）；
- 汇总表（`D_true`, `D_from_MSD`, `mu_true`, `mu_from_drift`, `ratio`, `R^2`）；
- 时间序列表头/表尾（便于确认轨道统计的线性趋势）。

脚本末尾有自动断言，成功时打印 `All checks passed.`。

## R08

前置知识：

- 过阻尼朗之万方程；
- MSD 与扩散系数关系；
- 线性响应与迁移率定义。

依赖环境：

- Python `>=3.10`
- `numpy`
- `pandas`

本 MVP 未依赖 SciPy/Sklearn/PyTorch 黑盒模块。

## R09

适用场景：

- 统计力学教学中演示涨落-耗散关系；
- 对随机动力学代码做最小 sanity check；
- 在更复杂模型前验证仿真管线是否守住基本物理量纲关系。

不适用场景：

- 强非线性势场或状态相关迁移率问题；
- 需要高精度不确定度估计（置信区间、贝叶斯后验）；
- 需要实验数据拟合与系统辨识的生产科研流程。

## R10

正确性直觉：

1. 若噪声项实现正确，零外力下 `MSD` 应近似线性随时间增长；
2. 若漂移项实现正确，恒力下平均位移应近似线性随时间增长；
3. 两者斜率分别给出 `D` 与 `mu` 后，应满足 `D≈mu*kBT`；
4. 因此，线性 `R^2` 与 Einstein 比值同时通过，才说明“统计涨落”和“线性响应”两条链路都成立。

## R11

数值与统计稳健性措施：

- 使用较大粒子数（默认 6000）降低样本噪声；
- 用固定随机种子保证可复现；
- 跳过 `t=0` 点再做回归，避免平凡点过度影响拟合；
- 使用相对误差与 `R^2` 双指标，而非仅看某一个数值。

## R12

关键参数（`EinsteinConfig`）：

- `kbt`：热噪声强度；
- `mobility_true`：真实迁移率；
- `external_force`：驱动实验中的常力；
- `dt`、`n_steps`：离散时间步与总时长；
- `record_stride`：采样间隔；
- `n_particles`：统计样本规模；
- `seed`：随机种子。

调参建议：若误差偏大，优先增大 `n_particles` 或 `n_steps`。

## R13

保证说明：

- 近似比保证：N/A（非组合优化问题）；
- 概率成功闭式下界：N/A（属于 Monte Carlo 统计估计）。

本实现提供的工程保证：

- 无交互输入，单命令可运行；
- 内置断言会在误差超阈时立即失败；
- 所有估计链路在源码中显式展开，可逐行审计。

## R14

常见失效模式：

1. 把 `MSD` 斜率直接当 `D`（漏掉 `1/2` 因子）；
2. 漂移实验忘记除以外力 `F`，导致 `mu` 维度错误；
3. 粒子数太少使拟合噪声过大，误判 Einstein 关系失效；
4. 同时改 `kbt` 与噪声系数但公式不同步，导致系统性偏差；
5. 仅看终点值，不做全时域线性拟合，容易受偶然波动影响。

## R15

可扩展方向：

1. 给 `D_hat`、`mu_hat` 增加 bootstrap 置信区间；
2. 扫描多个温度点，验证 `D/mu` 对 `T` 的线性关系；
3. 引入空间势场 `U(x)`，比较局域/有效 Einstein 关系；
4. 扩展到二维或三维并比较各向异性扩散；
5. 使用 PyTorch 自动微分反演 `mu` 或噪声参数。

## R16

相关主题：

- 涨落-耗散定理（Fluctuation-Dissipation Theorem）；
- Fokker-Planck / Smoluchowski 方程；
- 线性响应理论与 Kubo 公式；
- 布朗运动与随机微分方程离散化。

## R17

运行方式（无交互）：

```bash
cd Algorithms/物理-统计力学-0312-爱因斯坦关系_(Einstein_Relation)
uv run python demo.py
```

交付核对：

- `README.md` 的 `R01-R18` 已完整填写；
- `demo.py` 可直接运行并完成自检；
- `meta.json` 的 UID/学科/分类/源序号/目录信息与任务一致；
- 仅修改本算法目录内文件。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 构造 `EinsteinConfig` 并调用 `run_einstein_relation_demo`。  
2. 在 `run_einstein_relation_demo` 中按 `D_true = mobility_true * kbt` 计算理论扩散系数。  
3. 分别执行两次 `simulate_overdamped_ensemble`：一次 `force=0`（扩散实验），一次 `force=external_force`（漂移实验）。  
4. 每次仿真使用 Euler-Maruyama 更新 `x`，并按 `record_stride` 记录 `time/mean_displacement/msd`。  
5. 对零力实验调用 `fit_linear(t, msd)`，由斜率得到 `D_from_MSD = slope/2`。  
6. 对恒力实验调用 `fit_linear(t, mean_disp)`，由斜率得到 `drift_velocity`，再算 `mu_from_drift = drift_velocity / force`。  
7. 计算 `D_from_mu_kBT = mu_from_drift * kbt` 与 `ratio = D_from_MSD / D_from_mu_kBT`，并汇总成 `pandas.DataFrame`。  
8. `main` 打印摘要与时间序列样本，执行断言（`D`、`mu` 相对误差、Einstein 比值偏差、两条拟合的 `R^2`），全部通过后输出 `All checks passed.`。

第三方库边界说明：`numpy` 仅用于随机数、向量更新和线性代数，`pandas` 仅用于结果表展示；Einstein 关系验证流程（仿真、估计、比值检验）均为源码显式实现，不依赖专用黑盒求解器。
