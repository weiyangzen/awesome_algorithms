# 张量-标量比 (Tensor-to-Scalar Ratio)

- UID: `PHYS-0354`
- 学科: `物理`
- 分类: `宇宙学`
- 源序号: `372`
- 目标目录: `Algorithms/物理-宇宙学-0372-张量-标量比_(Tensor-to-Scalar_Ratio)`

## R01

张量-标量比 `r` 是暴胀宇宙学中的核心可观测参数，定义为标量和张量原初扰动功率谱振幅之比：
`r(k*) = A_t(k*) / A_s(k*)`（通常取枢轴尺度 `k*=0.05 Mpc^-1`）。

物理意义：
- `A_s` 控制温度各向异性与大尺度结构的主导标量扰动；
- `A_t` 对应原初引力波背景，最直接体现在 CMB B-mode 极化；
- 因此 `r` 连接“早期宇宙量子涨落”与“可测 CMB 信号”的桥梁。

## R02

本目录 MVP 的目标是实现一个可运行、可审计的最小推断流程，而非完整 Boltzmann 求解器：
1. 构造 toy CMB B-mode 模型：`D_ell^BB = r*T_ell + L_ell + n_ell`。
2. 生成带噪声模拟观测 bandpower。
3. 用 `chi^2` 最小化估计 `r_hat`。
4. 给出 95% 单侧上限 `r_95`。
5. 从 `r_hat` 派生慢滚参数与暴胀能标估计。

这样可以直接展示“如何从 B 模数据约束 r”。

## R03

符号与参数（见 `demo.py` 中 `TensorScalarParams`）：
- `ell_min, ell_max, n_bins`：多极矩 bin 配置（默认 `20~200`, 12 个 bin）。
- `r_true`：生成模拟数据的真实值（默认 `0.035`）。
- `lensing_amp_true`：透镜 B 模幅度因子（默认 `1.0`）。
- `noise_floor_uk2, frac_noise`：噪声底与相对噪声强度。
- `a_s`：标量振幅（默认 `2.1e-9`）。
- `m_pl_reduced_gev`：约化普朗克质量（`2.435e18 GeV`）。
- `r_min, r_max`：参数估计区间（`[0, 0.3]`）。

## R04

MVP 使用的核心公式：

1. 观测模型（toy）
`D_ell^BB(obs) = r * T_ell + L_ell + n_ell`

2. 似然（高斯噪声）对应的 `chi^2`
`chi^2(r) = Σ_i [ (D_i(obs) - (r*T_i + L_i)) / sigma_i ]^2`

3. 单参数 95% 单侧上限判据
`chi^2(r_95) - chi^2_min = 2.71`

4. 慢滚关系（最低阶）
`epsilon = r / 16`, `n_t = -r / 8`

5. 暴胀能标估计
`V = (3*pi^2/2) * A_s * r * M_pl^4`
`V^(1/4) = [(3*pi^2/2) * A_s * r]^(1/4) * M_pl`

## R05

本实现输出以下可验证结果：
- `r_hat`：`chi^2` 最优拟合值；
- `sigma_r`：由曲率近似得到的 1σ 误差；
- `r_95`：95% 单侧上限；
- `chi2_min` 与自由度 `dof`；
- `epsilon, n_t` 与 `V^(1/4)`（由 `r` 派生的物理量）；
- `chi^2` 采样剖面表和模拟 bandpower 表。

## R06

`demo.py` 的计算流程：
1. 用 `make_ell_grid` 生成 `ell` bin 中心。
2. 构造 `T_ell`（原初张量模板）与 `L_ell`（透镜模板）。
3. 生成 `bb_obs_uk2` 与异方差 `sigma_uk2` 的模拟观测。
4. 定义 `chi2_of_r`。
5. 用有界标量最优化得到 `r_hat`。
6. 在 `r_hat` 附近用二阶差分估计曲率误差 `sigma_r`。
7. 用 `brentq` 解 `Δchi^2=2.71` 得 `r_95`。
8. 打印拟合结果、派生宇宙学量和审计表格。

## R07

正确性依据：
- 统计上：高斯噪声下最小化 `chi^2` 等价于最大似然估计。
- 物理上：`r` 只缩放原初张量模板 `T_ell`，符合参数定义。
- 数值上：`r` 被显式限制在物理区间 `r>=0`，避免无物理意义解。
- 结果上：`r_95` 通过 `Δchi^2` 判据独立计算，可与 `r_hat ± sigma_r` 互相校验趋势。

## R08

复杂度（`n` 为 bandpower bin 数，`k` 为优化迭代步数）：
- 单次 `chi^2` 评估：`O(n)`。
- `r` 拟合：`O(k*n)`。
- 上限根求解：`O(k*n)`。
- 曲率估计：固定 3 次 `chi^2` 调用，`O(n)`。
- 总体：`O(k*n)`，空间复杂度 `O(n)`。

## R09

数值稳定性与工程处理：
- `minimize_scalar(method="bounded")` 使用封闭区间，避免参数发散。
- `brentq` 只在保证异号区间时使用；若 `r_max` 仍未达到 `Δchi^2=2.71`，返回边界值 `r_max`。
- 曲率法若出现非正二阶导（噪声导致近似失败），返回 `NaN`，避免误导性误差条。
- 噪声模型采用 `noise_floor + frac * signal`，避免 `sigma` 过小导致数值爆炸。

## R10

代码模块划分（`demo.py`）：
- `TensorScalarParams`：集中管理可复现实验参数。
- `primordial_bb_template_r1` / `lensing_bb_template`：构造 toy 理论模板。
- `build_mock_dataset`：生成模拟观测数据表。
- `chi2_of_r`：定义目标函数。
- `fit_r_bounded`：估计 `r_hat`。
- `estimate_sigma_from_curvature`：误差近似。
- `upper_limit_r_95`：95% 上限求解。
- `slow_roll_from_r` / `inflation_energy_scale_gev`：物理解释量映射。
- `run_demo`：组织并输出完整结果。

## R11

最小依赖栈：
- `numpy`：数组与数值计算；
- `scipy.optimize`：有界标量最小化与一维根求解；
- `pandas`：表格化输出（便于检查与后续导出）。

未引入 CAMB/CLASS 等大型库，保持 MVP 轻量与可读。

## R12

运行方式（仓库根目录）：

```bash
uv run python "Algorithms/物理-宇宙学-0372-张量-标量比_(Tensor-to-Scalar_Ratio)/demo.py"
```

或在该目录中运行：

```bash
uv run python demo.py
```

脚本无需交互输入。

## R13

输出字段解释：
- `r_true`：模拟数据的输入真值；
- `r_hat`：拟合得到的最优 `r`；
- `Approx 1-sigma uncertainty`：局部曲率误差估计；
- `r_95`：95% 单侧上限；
- `chi2_min, dof`：拟合优度快速参考；
- `epsilon, n_t`：慢滚一致性关系的派生参数；
- `V^(1/4)`：对应暴胀能量尺度估计；
- `Chi-square profile sample`：用于检查似然形状是否单峰；
- `Synthetic bandpower table`：输入观测构造细节。

## R14

自检建议：
1. 将 `r_true` 调大到 `0.08`，应看到 `r_hat` 与 `V^(1/4)` 上升。
2. 将 `frac_noise` 提高到 `0.35`，应看到 `sigma_r` 变大、`r_95` 变宽松。
3. 将 `r_true` 设到 `0.0`，应看到 `r_hat` 更靠近 0 且上限主导。
4. 将 `r_max` 缩到 `0.08`，若边界过紧，`r_95` 可能碰到上边界。

## R15

模型边界与局限：
- 模板 `T_ell, L_ell` 为教学型 toy 形状，不等价于真实 Boltzmann 解。
- 未联合拟合透镜幅度、前景残差、系统学偏差与协方差非对角项。
- 仅示范单参数 `r` 估计流程，不代表真实实验完整统计管线。
- `sigma_r` 由局部抛物近似给出，在强非高斯/边界主导情形下会失真。

## R16

可扩展方向：
- 把 `lensing_amp` 加入联合拟合，形成二维参数后验。
- 用真实实验噪声/波束/天空覆盖模型替换 toy 噪声。
- 接入 CAMB/CLASS 生成真实 `C_ell` 模板。
- 从 profile 扩展到 MCMC 或网格后验，输出可信区间而非仅上限。

## R17

与宇宙学框架的关系：
- `r` 是检验暴胀模型最直接的参数之一；
- 若 `r` 被显著测得，可反推暴胀能标并约束势函数形状；
- 若仅得到更严格上限，则排除高 `r` 的暴胀模型分支；
- 因此 `r` 约束与 `n_s`、非高斯度等观测一起构成早期宇宙模型筛选核心。

## R18

本实现的源码级算法流（9 步）：
1. `run_demo` 创建参数对象并调用 `build_mock_dataset`。
2. `build_mock_dataset` 先生成 `ell` 网格，再计算 `T_ell` 与 `L_ell` 两套模板。
3. 根据 `signal = r_true*T_ell + L_ell` 与异方差 `sigma` 生成高斯噪声观测 `bb_obs`。
4. `fit_r_bounded` 将 `chi2_of_r` 作为目标函数，交给 `scipy.optimize.minimize_scalar(method="bounded")`。
5. `bounded` 算法在 `[r_min, r_max]` 内迭代：先给定区间与试探点，按函数值比较缩小区间；当抛物线插值不可靠时回退到黄金分割步进，直到区间宽度小于容差，得到 `r_hat`。
6. `estimate_sigma_from_curvature` 在 `r_hat±h` 处做二阶有限差分，利用 `d2chi2/dr2 = 2/sigma_r^2` 得到 `sigma_r`。
7. `upper_limit_r_95` 构造方程 `chi2(r)-chi2_min-2.71=0`，用 `brentq` 在 `[max(r_hat,0), r_max]` 区间做保号求根得到 `r_95`。
8. `slow_roll_from_r` 与 `inflation_energy_scale_gev` 把 `r_hat` 转换为 `epsilon`、`n_t`、`V^(1/4)`。
9. 最后输出 `chi^2` 剖面采样和完整 bandpower 表，形成可复现实验审计链路。
