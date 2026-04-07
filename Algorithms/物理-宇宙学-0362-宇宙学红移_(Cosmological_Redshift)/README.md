# 宇宙学红移 (Cosmological Redshift)

- UID: `PHYS-0344`
- 学科: `物理`
- 分类: `宇宙学`
- 源序号: `362`
- 目标目录: `Algorithms/物理-宇宙学-0362-宇宙学红移_(Cosmological_Redshift)`

## R01

本条目实现一个“最小但可追踪”的宇宙学红移 MVP，覆盖以下主链路：
- 红移与尺度因子关系：`1+z = a_obs / a_emit`。
- 在平直 `ΛCDM` 中从 `z` 计算 `H(z)`、共动距离 `χ(z)`、光度距离 `d_L(z)`、角径距离 `d_A(z)`、回望时间。
- 从谱线波长比与光度距离反演红移。
- 结合 `scikit-learn` 与 `PyTorch` 做参数诊断（低红移 `H0` 回归、`Ωm0` 拟合）。

## R02

问题对象是“宇宙膨胀导致的光谱红移与距离-时间关系”。MVP 目标不是高精度观测学管线，而是：
- 方程透明。
- 数值步骤可复现。
- 依赖库调用可解释，不把核心计算当黑箱。

## R03

核心关系式：
- 红移定义：`z = (λ_obs - λ_emit) / λ_emit`。
- 尺度因子：`a(z)=1/(1+z)`。
- 平直 `ΛCDM`：`E(z)=sqrt(Ωm0(1+z)^3+ΩΛ0)`，`H(z)=H0 E(z)`。
- 共动距离：`χ(z)= (c/H0) ∫[0->z] dz'/E(z')`。
- 光度距离：`d_L=(1+z)χ`。
- 角径距离：`d_A=χ/(1+z)`。
- 回望时间：`t_L(z)= (1/H0) ∫[0->z] dz'/((1+z')E(z'))`。
- 距离模数：`μ=5 log10(d_L/Mpc)+25`。

## R04

`demo.py` 无交互输入，脚本内固定参数并直接打印三类结果：
- 红移-距离-时间表（含谱线波长反演校验）。
- 低红移样本上的 `H0` 线性回归估计。
- 合成超新星样本上的 `Ωm0` 拟合结果与残差预览。

## R05

默认宇宙学参数（`FlatLambdaCDM`）：
- `h = 0.674`
- `Ωm0 = 0.315`
- `ΩΛ0 = 0.685`
- `H0 = 100h km/s/Mpc`
- `c = 299792.458 km/s`

数值设置：
- 积分网格 `n_grid=4096`（距离与时间积分）。
- `PyTorch` 拟合迭代 `700` 步。

## R06

实现策略：
- 用 `scipy.integrate.cumulative_trapezoid` 预积分 `1/E(z)` 与 `1/[(1+z)E(z)]`，再用插值得到 `χ(z)` 与 `t_L(z)`。
- 用 `pandas` 组织可读报告表。
- 用 `scipy.optimize.root_scalar` 做 `d_L -> z` 的数值反演。
- 用 `LinearRegression` 在低红移近似 `v≈H0 d_L` 下估计 `H0`。
- 用 `PyTorch` 自动微分拟合 `Ωm0`，目标是最小化 `μ(z)` 残差均方。

## R07

时间复杂度（主导项）：
- 距离/时间积分：`O(N_grid)`。
- 低红移回归：`O(N_lowz)`。
- `Ωm0` 拟合：`O(N_iter * N_sn * N_int)`，其中 `N_int` 是每个红移点的积分离散数。

默认参数下在 CPU 可秒级运行。

## R08

数值稳定设计：
- 限制红移 `z>=0`，非法输入直接报错。
- 对 `z=0` 特判，避免不必要积分与 `log10(0)` 问题。
- `Ωm0` 用 `sigmoid` 重参数化到 `[0.05, 0.60]`，防止训练跑到非物理区间。
- 距离反演使用有括区 `Brent` 方法，避免开放迭代发散。

## R09

边界与失败条件：
- 本实现只覆盖平直 `ΛCDM`（要求 `Ωm0+ΩΛ0=1`）。
- 不含辐射、曲率、中微子、暗能量状态方程演化（`w(z)`）。
- 若目标光度距离超出 `z∈[0,10]` 括区会显式报错。
- 若 `root_scalar` 未收敛，抛出 `RuntimeError`。

## R10

结果应按“算法演示”理解：
- 红移越大，`a(z)` 越小，`d_L` 与回望时间总体增大。
- `d_A` 在某一红移后会下降，体现角径距离的宇宙学行为。
- 低红移回归可近似恢复 `H0`，但受噪声与近似误差影响。
- `Ωm0` 拟合是合成数据示例，不代表真实观测约束。

## R11

运行方式：

```bash
uv run python Algorithms/物理-宇宙学-0362-宇宙学红移_(Cosmological_Redshift)/demo.py
```

脚本无需输入参数，直接打印报告。

## R12

输出解读建议：
- 先看 `[Redshift-Distance-Time Report]`，确认 `z_from_line` 与输入 `z` 一致。
- 再看 `[Low-z Hubble Regression]`，`H0` 估计应接近真值、`R^2` 应较高。
- 最后看 `[Omega_m0 Fit]`，比较 `True Omega_m0` 与 `Fitted Omega_m0` 及 `RMSE`。
- `[Inverse Checks]` 提供尺度因子和光度距离反演的闭环验证。

## R13

最小验收清单：
- `README.md` 与 `demo.py` 的占位符已全部清除。
- `uv run python demo.py` 可直接完成执行。
- 输出中 `z_ref=1` 的距离反演 `z_inv` 与 1 足够接近。
- `H0` 回归值与设定 `H0=67.4` 同量级且方向正确。

## R14

第三方库的使用不是黑箱：
- `scipy` 只负责基础数值原语（梯形积分累积、Brent 求根），核心物理公式在源码显式实现。
- `scikit-learn` 仅执行线性回归 `v` 对 `d_L`，输入与目标都由脚本物理量构造。
- `PyTorch` 只做 `Ωm0` 的单参数梯度优化，`μ(z)` 公式和积分离散在 `torch_distance_modulus_model` 明确展开。

## R15

当前局限：
- 没有接入真实观测数据（仅合成样本）。
- 没有误差协方差、系统误差、贝叶斯后验等严谨统计流程。
- 高红移处未纳入辐射项，早期宇宙不适用。

## R16

可扩展方向：
- 将 `Ωm0` 单参数拟合扩展到 `h, Ωm0, w0` 多参数联合拟合。
- 接入超新星或 BAO 公开数据集，替换合成数据。
- 增加不确定性传播（MCMC 或 Fisher 近似）。
- 对比 `ΛCDM` 与 `wCDM` 模型的赤经向距离差异。

## R17

工具栈与职责：
- `numpy`：数组网格、插值、随机噪声。
- `scipy`：累计积分与根求解。
- `pandas`：表格化报告输出。
- `scikit-learn`：低红移线性回归估计 `H0`。
- `PyTorch`：自动微分拟合 `Ωm0`。

## R18

`demo.py` 的源码级算法流程（8 步）：
1. `FlatLambdaCDM` 定义宇宙学参数，并在 `main` 校验平直性 `Ωm0+ΩΛ0=1`。
2. `e_of_z` 与 `hubble_of_z` 计算背景膨胀率，`scale_factor_from_redshift` 给出 `a(z)`。
3. `comoving_distance_mpc` 用 `cumulative_trapezoid` 对 `1/E(z)` 积分并插值得到 `χ(z)`。
4. 基于 `χ(z)` 推导 `d_L(z)`、`d_A(z)`，并在 `lookback_time_gyr` 中积分得到回望时间。
5. `build_redshift_report` 组装 `z`、距离、时间、谱线波长映射表，验证 `z = λ_obs/λ_emit - 1`。
6. `low_z_hubble_fit` 生成低红移合成样本，使用 `LinearRegression` 回归估计 `H0`。
7. `fit_omega_m0_with_torch` 调用 `torch_distance_modulus_model`，通过梯度下降拟合 `Ωm0` 使 `μ(z)` 误差最小。
8. `infer_redshift_from_luminosity_distance` 用 `root_scalar` 反演红移，最后打印全链路报告与闭环校验。
