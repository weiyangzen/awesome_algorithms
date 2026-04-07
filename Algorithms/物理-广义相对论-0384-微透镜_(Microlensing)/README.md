# 微透镜 (Microlensing)

- UID: `PHYS-0365`
- 学科: `物理`
- 分类: `广义相对论`
- 源序号: `384`
- 目标目录: `Algorithms/物理-广义相对论-0384-微透镜_(Microlensing)`

## R01

微透镜是引力透镜中的“时间域”版本：当一个前景致密天体经过背景恒星视线附近时，背景源并不一定被空间分辨成多个像，但总亮度会随时间出现典型增亮-回落曲线。

本条目 MVP 目标是把该现象落成可执行算法：
- 用点源-点透镜（PSPL）模型实现 Paczynski 光变曲线；
- 生成含噪声合成观测数据；
- 用最小二乘反演事件参数并输出拟合质量指标。

## R02

问题定义（MVP 范围）：
- 输入：
  - 时间序列 `t`；
  - 观测通量 `F_obs(t)` 与测量误差 `sigma_F`；
  - （演示脚本中）真值参数用于合成数据：`t0, u0, tE, fs, fb`。
- 输出：
  - 拟合参数 `t0_hat, u0_hat, tE_hat, fs_hat, fb_hat`；
  - 统计指标：`RMSE`、`MAE`、`reduced_chi2`；
  - 参数恢复误差（相对误差）；
  - 公式一致性检查（NumPy 与 PyTorch 计算差异）。

脚本内置完整样例，`uv run python demo.py` 可直接运行，无需交互输入。

## R03

核心模型（PSPL）：

1. 归一化角距（以爱因斯坦角为单位）

`u(t) = sqrt(u0^2 + ((t - t0)/tE)^2)`

2. 点源点透镜放大率（Paczynski 公式）

`A(u) = (u^2 + 2) / (u * sqrt(u^2 + 4))`

3. 观测通量模型（含 blending）

`F(t) = fs * A(u(t)) + fb`

其中：
- `t0`：最近接时刻；
- `u0`：最小冲击参数；
- `tE`：爱因斯坦时间尺度；
- `fs`：源星通量；
- `fb`：混合背景通量。

## R04

算法高层流程：

1. 设定一组真值微透镜参数并生成时间网格。
2. 用 PSPL 公式计算理想光变曲线 `F_clean(t)`。
3. 叠加高斯噪声得到观测数据 `F_obs(t)`。
4. 由观测曲线估计初始参数（峰时、基线、初始 `u0/tE`）。
5. 构造归一化残差 `r = (F_model - F_obs)/sigma_F`。
6. 调用 `scipy.optimize.least_squares` 拟合 5 个参数。
7. 输出参数恢复误差和拟合质量（`RMSE`、`reduced_chi2`）。
8. 执行阈值检查，确保该 MVP 在合成数据上稳定可复现。

## R05

核心数据结构：

- `MicrolensingParams`（`dataclass`）：
  - `t0, u0, tE, fs, fb` 五个物理参数。
- `numpy.ndarray`：
  - `time`：观测时刻数组；
  - `flux_obs`：带噪声观测通量；
  - `flux_err`：观测误差；
  - `flux_clean`：无噪声真值曲线。
- `dict` 指标：
  - `rmse_flux`、`mae_flux`、`reduced_chi2`；
  - `rel_err_t0/u0/tE/fs/fb`；
  - `max_rel_param_error`。

## R06

正确性设计：

- 物理关系透明：放大率、冲击参数、通量方程都在源码中显式实现。
- 参数合法性强约束：`u0>0, tE>0, fs>0, fb>=0`，并检查有限值。
- 拟合目标可解释：最小化加权残差，等价于高斯误差下的最大似然。
- 结果闭环：
  - 参数误差直接对照真值；
  - `reduced_chi2` 检验拟合与噪声水平匹配程度；
  - 峰值通量拟合值对照真值峰值。
- 数值一致性：额外用 PyTorch 张量公式对照 NumPy 放大率实现。

## R07

复杂度（`N` 为观测点数，`K` 为优化迭代次数）：

- 单次模型前向计算：`O(N)`；
- 单次残差计算：`O(N)`；
- 最小二乘拟合总体：`O(KN)`；
- 空间复杂度：`O(N)`。

对 MVP 默认 `N=240`，运行开销很小，适合快速回归测试。

## R08

边界与异常处理：

- `n_points < 32`、`noise_sigma <= 0`、`span_factor <= 0`：抛 `ValueError`。
- 参数数组维度不一致、长度不足：抛 `ValueError`。
- 误差数组中存在非正值：抛 `ValueError`。
- 优化失败或指标超阈值：`run_checks` 抛 `RuntimeError`。
- `torch` 不可用时不终止流程，相关一致性指标返回 `NA`。

## R09

MVP 取舍说明：

- 仅覆盖单透镜单源 PSPL，不含双星透镜、视差、有限源效应、xallarap。
- 使用单波段通量模型，不做多色联合拟合。
- 不做 MCMC 后验采样，只做确定性最小二乘参数恢复。
- 不输出图像文件，终端输出参数表和诊断指标，保持最小实现与高可审计性。

## R10

`demo.py` 函数职责：

- `validate_params`：检查参数合法性与数值稳定边界。
- `impact_parameter`：计算 `u(t)`。
- `magnification_pspl`：实现 Paczynski 放大率公式。
- `flux_model`：组合出观测通量模型 `F(t)`。
- `synthesize_dataset`：生成含噪声演示数据。
- `initial_guess`：从观测数据构造优化初值。
- `fit_microlensing_curve`：执行约束最小二乘拟合。
- `evaluate_fit`：计算误差和拟合质量。
- `torch_formula_consistency_check`：NumPy 与 Torch 公式一致性核对。
- `run_checks`：对拟合成功与指标阈值进行通过判定。

## R11

运行方式：

```bash
cd Algorithms/物理-广义相对论-0384-微透镜_(Microlensing)
uv run python demo.py
```

脚本无交互输入、无网络调用，运行后直接打印参数恢复结果和诊断信息。

## R12

输出字段解释：

- 参数表：
  - `param`：参数名；
  - `true`：真值；
  - `fitted`：拟合值；
  - `rel_error`：相对误差。
- 指标表：
  - `rmse_flux` / `mae_flux`：通量拟合误差；
  - `reduced_chi2`：归一化卡方；
  - `max_rel_param_error`：5 个参数最大相对误差；
  - `peak_flux_true` / `peak_flux_fit`：峰值通量真值与拟合值；
  - `torch_formula_max_abs_diff`：NumPy/Torch 公式最大绝对差；
  - `optimizer_nfev`：优化器函数评估次数。

## R13

建议最小验证集（脚本内置）：

- 单事件合成数据：
  - 真值参数：`t0=60, u0=0.23, tE=24, fs=1.2, fb=0.35`；
  - 观测点：`240`；
  - 噪声：`sigma=0.02`。
- 通过阈值：
  - `rmse_flux < 0.08`；
  - `reduced_chi2 < 3.0`；
  - `max_rel_param_error < 0.15`。

可扩展异常测试：
- `u0 <= 0`；
- `tE <= 0`；
- `flux_err` 含零或负值；
- `n_points < 32`。

## R14

关键可调参数：

- `u0`：控制峰值放大率，越小峰越尖、越高。
- `tE`：控制事件持续时间，越大曲线越宽。
- `fs` 与 `fb`：控制峰值高度和基线水平（blending 程度）。
- `noise_sigma`：观测噪声强度，直接影响参数可识别性。
- `n_points` 与 `span_factor`：控制采样密度和覆盖时窗。

## R15

方法对比：

- 对比“只做峰值估计”的启发式方法：
  - 启发式快但不能稳定恢复 `fs/fb` 等耦合参数；
  - 本方法用全曲线拟合，信息利用更完整。
- 对比黑盒天文拟合工具：
  - 黑盒易用但内部目标函数与约束不透明；
  - 本 MVP 从公式到残差再到优化流程完全显式。
- 对比 MCMC 全后验方法：
  - MCMC 更全面但计算更重；
  - 本 MVP 适合作为前处理和 sanity check 基线。

## R16

典型应用场景：

- 微透镜教学：快速展示 Paczynski 曲线形态与参数意义。
- 管线前置校验：在真实天文测光前验证拟合链路可用性。
- 算法基线：作为复杂效应（视差、有限源、双透镜）扩展前的基线模块。
- 数据仿真：生成可控事件用于回归测试与误差传播实验。

## R17

可扩展方向：

- 加入地球/卫星视差项，处理长时间尺度事件。
- 加入有限源与 limb-darkening，提高高放大率事件真实性。
- 扩展到双透镜模型，支持 caustic crossing 结构。
- 引入多波段联合拟合，提升 `fs/fb` 退化分离能力。
- 用 MCMC 或变分方法输出参数后验区间而非点估计。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 构建真值参数 `MicrolensingParams(t0, u0, tE, fs, fb)`，并调用 `synthesize_dataset` 生成带噪声观测。
2. `synthesize_dataset` 先用 `flux_model` 计算无噪声曲线，再叠加高斯噪声得到 `flux_obs` 与 `flux_err`。
3. `initial_guess` 从数据的峰时和两端基线估计 `t0/fs/fb`，并给出 `u0/tE` 初值。
4. `fit_microlensing_curve` 构建参数向量与上下界，定义残差 `r=(F_model-F_obs)/sigma_F`。
5. `least_squares` 迭代调用 `residuals`；每次 `residuals` 内部通过 `impact_parameter -> magnification_pspl -> flux_model` 计算预测曲线。
6. 拟合结束后，`evaluate_fit` 计算 `RMSE`、`MAE`、`reduced_chi2` 和每个参数的相对误差。
7. `torch_formula_consistency_check` 用 Torch 张量复算放大率公式，与 NumPy 结果做最大绝对差对比。
8. `run_checks` 执行质量门限，随后打印参数表与诊断表并给出 `All checks passed.`。

第三方库没有被当作物理黑盒：`scipy` 只提供通用最小二乘优化器，微透镜核心物理公式（`u(t)`、`A(u)`、`F(t)`）和全部诊断逻辑都在源码中显式展开。
