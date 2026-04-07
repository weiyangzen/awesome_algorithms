# 暗物质 (Dark Matter)

- UID: `PHYS-0056`
- 学科: `物理`
- 分类: `宇宙学`
- 源序号: `56`
- 目标目录: `Algorithms/物理-宇宙学-0056-暗物质_(Dark_Matter)`

## R01

暗物质问题的一个核心观测证据是星系旋转曲线“外盘不按开普勒定律快速下降”，即在可见物质难以解释的半径范围，圆周速度依然较高。  
本条目把“暗物质”落到一个可执行算法任务：给定星系半径-速度数据，拟合一个参数化暗晕模型，并与“仅重子成分”基线比较拟合质量。

## R02

本 MVP 的输入与输出：
- 输入：半径 `r`、观测旋转速度 `v_obs`、观测误差 `sigma`（脚本内生成可复现合成数据）。
- 输出：NFW 暗晕参数估计 `rho_s, r_s`，重子缩放参数 `upsilon`，以及模型对比指标（`chi2/rmse/r2/aic/bic`）。

`demo.py` 无需交互输入，可直接运行并输出结果表。

## R03

数学模型：

1. 总旋转速度分解
\[
v_{\text{tot}}^2(r)=v_{\text{bary}}^2(r)+v_{\text{halo}}^2(r)
\]

2. 重子项采用简化的盘+核球速度剖面（用于教学与算法验证）：
\[
v_{\text{bary,scaled}}(r)=\sqrt{\Upsilon}\,v_{\text{bary,base}}(r)
\]
其中 `\Upsilon` 为质量-光度缩放代理参数。

3. NFW 密度与包围质量：
\[
\rho(r)=\frac{\rho_s}{(r/r_s)(1+r/r_s)^2}
\]
\[
M(<r)=4\pi\rho_s r_s^3\left[\ln(1+r/r_s)-\frac{r/r_s}{1+r/r_s}\right]
\]

4. 暗晕速度项：
\[
v_{\text{halo}}(r)=\sqrt{\frac{G M(<r)}{r}}
\]

## R04

参数估计目标：最小化带误差权重的残差平方和
\[
\chi^2=\sum_i\left(\frac{v_{\text{pred},i}-v_{\text{obs},i}}{\sigma_i}\right)^2
\]

使用 `scipy.optimize.least_squares`：
- 基线模型（无暗晕）：只拟合 `log10_upsilon`；
- 暗晕模型：联合拟合 `log10_upsilon, log10_rho_s, log10_r_s`；
- 通过参数取对数并设置边界，保证物理参数为正且搜索稳定。

## R05

算法高层流程：
1. 生成固定随机种子的合成旋转曲线数据。
2. 计算重子基准速度 `v_bary_base(r)`。
3. 构建“仅重子”残差函数并做最小二乘拟合。
4. 构建“重子+NFW 暗晕”残差函数并做最小二乘拟合。
5. 计算两模型的 `chi2/rmse/r2/aic/bic`。
6. 输出拟合参数与前 10 行预测对比表。
7. 用 PyTorch 复算 NFW 速度公式，与 NumPy 结果做一致性检查。
8. 通过断言确保暗晕模型在该数据上显著优于基线。

## R06

数据与单位约定：
- 半径单位：`kpc`；
- 速度单位：`km/s`；
- 质量单位：`Msun`；
- 引力常数：`G = 4.30091e-6 kpc*(km/s)^2/Msun`。

合成数据默认配置：
- 半径范围 `0.8 ~ 28.0 kpc`，共 `64` 点；
- 观测噪声标准差 `4 km/s`；
- 真值参数：`log10_rho_s=7.1`、`log10_r_s=1.08`、`log10_upsilon=-0.05`。

## R07

复杂度（`n` 为采样半径点数，`t` 为优化迭代步数）：
- 单次模型评估：`O(n)`；
- 单次残差向量构建：`O(n)`；
- 最小二乘拟合总复杂度：约 `O(t*n)`；
- 空间复杂度：`O(n)`（存储半径、速度、残差等向量）。

## R08

边界与异常处理：
- `r <= 0`、噪声标准差非正、半径区间非法：抛 `ValueError`；
- 优化器不收敛：抛 `RuntimeError`；
- 质量参数和尺度半径通过对数参数化确保始终为正；
- 参数边界限制在物理可解释范围，避免无约束搜索导致病态解。

## R09

MVP 取舍：
- 不接入外部观测数据库，先用可复现合成数据完成闭环。
- 使用 NFW 作为最常见暗晕参数化之一，避免过度复杂模型。
- 仅实现最关键的拟合与对比，不引入 MCMC/HMC 等重型采样器。
- 虽调用 `least_squares` 求解器，但物理模型、残差构造、指标计算全部显式实现，非黑箱端到端调用。

## R10

`demo.py` 主要函数职责：
- `baryon_velocity_base`：计算盘+核球基准速度剖面；
- `nfw_enclosed_mass / nfw_halo_velocity`：计算 NFW 包围质量与暗晕速度；
- `total_velocity_from_params`：合成总旋转速度；
- `simulate_rotation_curve`：生成可复现合成观测表；
- `fit_baryon_only / fit_with_halo`：执行两类拟合；
- `model_metrics`：产出误差与信息准则；
- `torch_consistency_check`：检查 torch 与 numpy 公式一致性；
- `main`：组织流程、打印结果、执行断言。

## R11

运行方式：

```bash
cd Algorithms/物理-宇宙学-0056-暗物质_(Dark_Matter)
uv run python demo.py
```

脚本不读取命令行参数，不请求用户输入。

## R12

输出结果包含：
- 真值参数与拟合参数对照；
- 模型比较表：
  - `chi2`：加权残差平方和；
  - `rmse`：速度预测均方根误差；
  - `r2`：拟合优度；
  - `aic/bic`：考虑参数数目的模型比较指标；
- 前 10 行观测与预测对照；
- `Torch vs NumPy` 公式一致性最大误差。

## R13

最小验证策略：
1. 基线模型与暗晕模型同数据对比，应见暗晕模型 `rmse` 更低；
2. 拟合参数应接近生成真值（允许一定噪声偏差）；
3. Torch 与 NumPy 的 NFW 速度实现应数值一致；
4. 固定随机种子后，每次运行结果稳定可复现。

## R14

关键可调参数：
- `n_points`：采样密度，影响参数可辨识度；
- `noise_sigma_kms`：噪声强度，影响反演难度；
- `true_log10_rho_s / true_log10_r_s / true_log10_upsilon`：合成真值；
- 最小二乘参数边界：决定搜索空间；
- 半径范围：决定对内盘/外盘信息的覆盖程度。

实务建议：若要更真实，可替换成观测数据并引入半径相关误差模型。

## R15

与其他暗物质推断路线对比：
- 与“仅重子拟合”相比：暗晕模型能解释外盘高速度平台。
- 与 MOND 等修正引力范式相比：本条目仅实现“加暗晕质量”路线，不做理论裁决。
- 与贝叶斯/MCMC 参数推断相比：最小二乘更轻量、可快速完成工程 MVP，但后验不确定性表达较弱。

## R16

适用范围与限制：
- 适用：教学演示、算法原型、模型比较流程验证。
- 限制：
  - 重子剖面为简化经验函数；
  - 未纳入气体盘、非轴对称结构、速度各向异性等复杂物理；
  - 合成数据的“真模型”与拟合模型同族，难度低于真实观测场景。

## R17

可扩展方向：
1. 使用真实旋转曲线观测数据（如 SPARC）替代合成数据；
2. 把 NFW 扩展为 Burkert、Einasto 等剖面并做模型选择；
3. 使用 `emcee` 或 PyTorch 自动微分做后验采样；
4. 联合透镜数据与动力学数据做多证据约束；
5. 在批量星系上做参数分布统计与层次建模。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 创建 `GalaxyConfig`，固定随机种子与真值参数，调用 `simulate_rotation_curve` 生成半径、观测速度、误差。  
2. `simulate_rotation_curve` 用 `total_velocity_from_params` 生成无噪声真值曲线，再叠加高斯噪声，封装为 `pandas.DataFrame`。  
3. `fit_baryon_only` 构建仅重子残差 `residual_baryon_only`，优化单参数 `log10_upsilon`，得到基线预测。  
4. `fit_with_halo` 构建含暗晕残差 `residual_with_halo`，联合优化 `log10_upsilon/log10_rho_s/log10_r_s`，得到暗晕预测。  
5. 两类残差都通过 `least_squares` 最小化加权残差；参数在对数空间优化并施加边界。  
6. `model_metrics` 计算 `chi2/rmse/r2/aic/bic`，在同一数据集上比较两模型拟合质量。  
7. `torch_consistency_check` 用 PyTorch 重写 NFW 公式并与 NumPy 实现逐点比对，验证公式实现一致。  
8. `main` 打印参数与表格，并执行四个断言（性能提升、参数回收精度、公式一致性），全部通过后输出 `All checks passed.`。  
