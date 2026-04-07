# 超导体电动力学 (Electrodynamics of Superconductors)

- UID: `PHYS-0197`
- 学科: `物理`
- 分类: `超导物理`
- 源序号: `198`
- 目标目录: `Algorithms/物理-超导物理-0198-超导体电动力学_(Electrodynamics_of_Superconductors)`

## R01

超导体电动力学关注超导态下电磁场与电流的耦合行为。该问题的最小可运行目标是：
- 在一维半空间模型中重现实验可观测的迈斯纳效应磁场衰减；
- 通过带噪声数据反演伦敦穿透深度 `lambda_L`；
- 在有限频率下用两流体模型估计复电导率与表面阻抗。

## R02

核心连续介质方程（各向同性近似）：
- 伦敦方程与麦克斯韦方程组合后得到  
  `∇^2 B = B / lambda_L^2`
- 对一维几何（表面法向为 `x`）解为  
  `B(x) = B0 * exp(-x / lambda_L)`
- 两流体复电导率：
  `sigma_tilde(omega, T) = sigma1(T) - i * sigma2(T)`
- 表面阻抗（局域极限）：
  `Zs = sqrt(i * omega * mu0 / sigma_tilde)`

## R03

MVP 输入（`demo.py` 内置，无交互）：
- 物理常数：`mu0, e, m_e`
- 超导参数：`Tc, n_total, tau, lambda_true`
- 频率：`f`（转化为 `omega = 2*pi*f`）
- 磁场数据：空间采样点 `x` 与带噪观测 `B_obs`

## R04

MVP 输出：
- 反演得到的 `lambda_fit` 与估计标准差 `lambda_std`
- 磁场拟合质量指标 `R^2`
- 温度扫描表（`pandas.DataFrame`）：
  - `f_s, f_n`（超流/正常流体积分数）
  - `sigma1, sigma2`
  - `Re(Zs), Im(Zs)`
  - 近似有效穿透深度 `lambda_eff`

## R05

关键建模假设：
- 材料均匀、各向同性，忽略晶格各向异性；
- 采用局域伦敦极限，不引入 Pippard 非局域核；
- 两流体模型 `f_s = 1 - (T/Tc)^4`（`T<Tc`）；
- 频域稳态响应，忽略强非线性与涡旋动力学。

## R06

算法流程概述：
1. 用真值 `lambda_true` 生成无噪声 `B_clean(x)`；
2. 叠加高斯噪声形成模拟实验数据 `B_obs`；
3. 固定 `B0`，用 `scipy.optimize.curve_fit` 拟合 `lambda_L`；
4. 计算不同温度下 `f_s, f_n`；
5. 构造 `sigma1(T), sigma2(T)` 并得到 `Zs(T)`；
6. 将结果整理为 `DataFrame` 并打印。

## R07

`demo.py` 中主要函数：
- `meissner_profile(x, B0, lambda_l)`：指数衰减磁场；
- `fit_penetration_depth(x, b_obs, b0)`：拟合穿透深度；
- `superfluid_fraction(T, Tc)`：两流体分数；
- `two_fluid_conductivity(T, omega, Tc, n_total, tau)`：复电导率分量；
- `surface_impedance(sigma1, sigma2, omega)`：表面阻抗；
- `main()`：参数设定、计算、输出。

## R08

时间复杂度（`N_x` 为空间采样点数，`N_T` 为温度点数）：
- 磁场生成与噪声叠加：`O(N_x)`
- 非线性拟合：每次迭代 `O(N_x)`，总计约 `O(k*N_x)`（`k` 为迭代数）
- 温度扫描电动力学计算：`O(N_T)`
- 总体：`O(k*N_x + N_T)`，空间复杂度 `O(N_x + N_T)`。

## R09

数值稳定性处理：
- 拟合时对 `lambda_L` 施加边界 `1e-9 ~ 1e-5 m`，避免非物理解；
- `f_s` 下限裁剪到 `1e-12` 级别用于 `lambda_eff` 计算，防止除零；
- 统一 SI 单位，避免隐式单位换算误差；
- 固定随机种子确保可复现。

## R10

参数量纲检查：
- `lambda_L, x`：米（m）
- `B`：特斯拉（T）
- `sigma1, sigma2`：西门子每米（S/m）
- `Zs`：欧姆（Ohm）
- `omega`：弧度每秒（rad/s）
  
MVP 中所有公式按 SI 制执行，避免单位混杂。

## R11

依赖栈（最小且透明）：
- `numpy`：向量化数值计算；
- `scipy`：`curve_fit` 非线性参数反演；
- `pandas`：结构化结果展示；
- `scikit-learn`：`r2_score` 评估拟合质量。
  
未引入额外框架，保持可读与可维护。

## R12

运行方式（仓库根目录）：

```bash
uv run python "Algorithms/物理-超导物理-0198-超导体电动力学_(Electrodynamics_of_Superconductors)/demo.py"
```

脚本不会请求任何交互输入。

## R13

结果解读要点：
- `lambda_fit` 应接近 `lambda_true`（噪声水平允许小偏差）；
- 低温下 `f_s` 大，`sigma2` 主导，体现超导感性响应；
- 温度接近 `Tc` 时 `f_s` 下降，`Re(Zs)` 上升，耗散增强。

## R14

可验证的物理一致性：
- `B(x)` 随 `x` 单调衰减；
- 对固定频率，`sigma2(T)` 随温度升高总体下降；
- `f_s + f_n = 1`（数值上在浮点误差范围内成立）；
- `Re(Zs) >= 0`（被动系统耗散非负）。

## R15

边界与异常情形：
- 当 `T >= Tc`，两流体模型退化为正常导体（`f_s=0`）；
- 当测量噪声过大时，`lambda_fit` 方差增大；
- 当频率极低时，`sigma2 ~ 1/omega` 可能变大，需谨慎解释近零频结果；
- 当 `tau` 取值异常时，`sigma1` 的数量级会显著变化。

## R16

MVP 局限性：
- 未建模混合态磁通涡旋与 pinning；
- 未考虑非局域效应、强耦合效应与各向异性超导；
- 未包含时域脉冲响应，仅做稳态频域分析；
- 参数使用合成数据演示，不等同真实材料标定。

## R17

可扩展方向：
- 用真实实验 `B(x)` 数据替换合成数据进行参数反演；
- 引入贝叶斯/Bootstrap 给出 `lambda_L` 置信区间；
- 扩展到 Mattis-Bardeen 频率依赖电导模型；
- 增加温度-频率二维扫描与可视化输出（图或 CSV）。

## R18

第三方函数不是黑箱，本项目中的源级算法流可拆为 8 步：
1. 用 `numpy` 在离散网格上构造 `x`，按解析式计算 `B_clean(x)`。
2. 用随机噪声模型生成 `B_obs(x)`，形成“测量数据”。
3. 将拟合目标写成单参数函数 `B_model(x; lambda_L)=B0*exp(-x/lambda_L)`。
4. `scipy.curve_fit` 内部执行有界非线性最小二乘：重复调用 `B_model`，比较残差 `r=B_obs-B_model`，并更新 `lambda_L` 直到收敛。
5. 读取协方差矩阵主对角线，得到参数估计方差并开方得到 `lambda_std`。
6. 对每个温度点计算 `f_s, f_n`，再按两流体公式显式计算 `sigma1, sigma2`（非黑箱，逐项可追踪）。
7. 构造 `sigma_tilde=sigma1-i*sigma2`，按电动力学公式计算 `Zs=sqrt(i*omega*mu0/sigma_tilde)`。
8. 将全部中间量和结果整理为 `pandas` 表格并输出，便于审计每一步数值来源。
