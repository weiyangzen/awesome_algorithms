# 啁啾质量 (Chirp Mass)

- UID: `PHYS-0374`
- 学科: `物理`
- 分类: `引力波物理`
- 源序号: `393`
- 目标目录: `Algorithms/物理-引力波物理-0393-啁啾质量_(Chirp_Mass)`

## R01

啁啾质量（Chirp Mass, `M_c`）是双致密天体引力波信号里最先、也最稳健可测的组合质量参数。对于双星分量质量 `m1,m2`，定义为：

`M_c = (m1*m2)^(3/5) / (m1+m2)^(1/5)`。

在低阶后牛顿近似（leading-order PN）下，引力波频率演化率 `df/dt` 由 `M_c` 主导，因此观测到的“啁啾”轨迹（频率随时间上升）可以直接反推出 `M_c`。

## R02

本条目要解决的问题：

- 输入：一组观测频率 `f` 与对应的频率导数 `fdot=df/dt`（可含噪声）。
- 输出：啁啾质量估计值 `M_c`（太阳质量单位）。
- 要求：给出最小可运行、可审计实现，而不是调用黑盒天体物理框架。

`demo.py` 采用“合成数据 + 两种估计器 + 自动校验”的 MVP 结构，确保端到端可复现。

## R03

核心物理公式（leading-order inspiral）：

`fdot = (96/5) * pi^(8/3) * (G*M_c/c^3)^(5/3) * f^(11/3)`

其中 `G` 为万有引力常数，`c` 为光速。

反解得到点估计公式：

`M_c = (c^3/G) * [ (5/96) * pi^(-8/3) * fdot * f^(-11/3) ]^(3/5)`

本实现用这两个公式分别做：
- 正向模拟（由 `M_c` 生成 `fdot`）；
- 反向估计（由 `(f,fdot)` 估计 `M_c`）。

## R04

估计任务分两层：

1. 点反演估计：
- 对每个采样点 `(f_i, fdot_i)` 反解 `M_c,i`。

2. 全局拟合估计：
- 用 `scipy.optimize.minimize` 在全体样本上最小化相对残差平方和，得到单个 `M_c`。

这样可以同时得到：
- 可解释的逐点估计分布；
- 更稳健的全局一致估计。

## R05

数据结构设计：

- `ChirpFitResult`（dataclass）
- `chirp_mass_true_solar`
- `chirp_mass_est_point_solar`
- `chirp_mass_est_opt_solar`
- `point_rel_error`, `opt_rel_error`
- `objective_value`, `n_iter`

- `pandas.DataFrame` 列：
- `f_hz`
- `fdot_true_hz_per_s`
- `fdot_obs_hz_per_s`
- `chirp_mass_point_est_solar`
- `fdot_pred_opt_hz_per_s`
- `point_abs_err_solar`

## R06

数值与正确性要点：

- 物理量正性约束：`f>0`, `fdot>0`, `M_c>0`。
- 噪声后 `fdot` 可能接近 0，代码做最小截断避免无效幂运算。
- 全局拟合在 `log(M_c)` 空间优化，天然保证 `M_c` 正值。
- 点估计汇总使用 `scipy.stats.trim_mean`，降低异常点影响。

## R07

复杂度分析（`n` 个频率采样点）：

- 点估计反演：`O(n)`
- 每次目标函数评估：`O(n)`
- 设优化迭代 `T` 次，则全局拟合：`O(T*n)`
- 空间复杂度：`O(n)`（主要是数据表与中间向量）

该复杂度非常轻量，适合作为引力波参数估计的教学级起点。

## R08

边界与异常处理：

- 组件质量 `m1,m2 <= 0` 抛 `ValueError`。
- 频率数组含非正值抛 `ValueError`。
- 观测 `fdot` 含非正值抛 `ValueError`。
- 样本点数过少（`n_points < 20`）抛 `ValueError`。
- 非线性拟合未收敛时抛 `RuntimeError`，避免静默失败。

## R09

MVP 取舍范围：

- 包含：
- leading-order PN 啁啾方程
- 噪声观测下的 `M_c` 估计
- 非线性最小二乘拟合

- 不包含：
- 自旋、偏心率、高阶 PN 项
- 完整贝叶斯后验采样（如 MCMC）
- 探测器响应函数与模板匹配全流程

因此这是“参数反演核心逻辑”的最小诚实实现。

## R10

`demo.py` 函数职责：

- `chirp_mass_from_component_masses`：由 `(m1,m2)` 计算真值 `M_c`。
- `fdot_from_chirp_mass`：正向模型 `M_c,f -> fdot`。
- `chirp_mass_from_f_and_fdot`：点反演 `f,fdot -> M_c`。
- `make_synthetic_observation`：合成观测数据并加入噪声。
- `estimate_chirp_mass_pointwise`：修剪均值聚合点估计。
- `estimate_chirp_mass_nonlinear_ls`：全局非线性拟合 `M_c`。
- `fit_chirp_mass_demo`：串联模拟与估计流程。
- `main`：打印结果、做最小质量门槛检查。

## R11

运行方式：

```bash
cd Algorithms/物理-引力波物理-0393-啁啾质量_(Chirp_Mass)
uv run python demo.py
```

脚本无交互输入，直接输出估计结果与样例表。

## R12

输出解读：

- `true_chirp_mass_solar`：合成数据使用的真实啁啾质量。
- `point_estimate_solar`：逐点反演后修剪均值估计。
- `nonlinear_ls_estimate_solar`：全局拟合估计（推荐值）。
- `point_relative_error / ls_relative_error`：两种估计相对误差。
- `optimizer_objective / optimizer_iterations`：拟合收敛诊断。
- `Sample observations`：展示局部观测、点估计与预测 `fdot`。

## R13

最小验证项：

- 固定随机种子，保证结果可复现。
- 全局估计误差阈值：`opt_rel_error <= 8%`，否则抛错。
- 观察点估计与全局拟合差异，确认拟合有稳定增益。

可扩展验证：

- 改变噪声强度 `rel_noise_std`，观察误差趋势。
- 改变频段 `f_min/f_max`，验证信息量变化。
- 增加样本点，检验估计方差收缩。

## R14

关键超参数与影响：

- `n_points`：采样点数；越大越稳健。
- `f_min_hz, f_max_hz`：观测频段；影响 `fdot` 动态范围。
- `rel_noise_std`：观测噪声强度；越大越难估计。
- `trim_ratio`：点估计修剪比例；抑制离群值。
- 拟合边界 `log(M_c) in [log(1e-3), log(300)]`：防止无意义极端解。

## R15

与其他方案的关系：

- 对比直接点反演：
- 点反演简单但易受噪声影响；
- 全局拟合能利用全体样本一致性，通常更稳健。

- 对比完整贝叶斯推断：
- 本 MVP 速度快、实现短；
- 但不输出完整后验分布与置信区间。

- 对比黑盒 GW 参数估计工具：
- 本实现强调公式可审计与源码透明；
- 工程精度和物理完整度低于专业管线。

## R16

应用场景：

- 引力波参数估计教学中的“第一性原理”演示。
- 对更复杂推断框架的 sanity check 基线。
- 快速验证观测噪声和频段对 `M_c` 可识别性的影响。

## R17

可扩展方向：

1. 加入高阶 PN 修正（相位与振幅）。
2. 同时估计质量比 `q`、自旋参数等。
3. 引入测量协方差与加权最小二乘。
4. 用 MCMC/NUTS 输出 `M_c` 后验区间。
5. 接入真实 strain 数据和模板匹配预处理。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `fit_chirp_mass_demo` 先设定双星分量质量 `(m1,m2)`，调用 `chirp_mass_from_component_masses` 得到真实 `M_c`。  
2. `make_synthetic_observation` 在频段 `[f_min,f_max]` 生成频率网格，用 `fdot_from_chirp_mass` 计算无噪声 `fdot_true`。  
3. 在 `fdot_true` 上施加乘性高斯噪声得到 `fdot_obs`，并做正值截断避免非物理值。  
4. `chirp_mass_from_f_and_fdot` 对每个观测点反解 `M_c,i`，得到逐点啁啾质量样本分布。  
5. `estimate_chirp_mass_pointwise` 对 `M_c,i` 做修剪均值（trimmed mean），得到稳健初值 `M_c_init`。  
6. `estimate_chirp_mass_nonlinear_ls` 在 `log(M_c)` 空间用 `scipy.optimize.minimize(L-BFGS-B)` 最小化相对残差平方和，得到全局最优 `M_c_hat`。  
7. 用 `M_c_hat` 再次前向计算 `fdot_pred`，并构建 `DataFrame` 汇总观测、点估计、预测与误差列。  
8. `main` 打印真值/估计值/误差与样例表；若 `ls_relative_error > 8%` 则抛错，否则输出 `All checks passed.`。  
