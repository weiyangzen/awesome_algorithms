# 弱透镜 (Weak Lensing)

- UID: `PHYS-0364`
- 学科: `物理`
- 分类: `广义相对论`
- 源序号: `383`
- 目标目录: `Algorithms/物理-广义相对论-0383-弱透镜_(Weak_Lensing)`

## R01

本条目实现一个“最小但诚实”的弱透镜算法 MVP，目标是把从收敛场 `κ` 到剪切场 `γ`、再回到 `κ` 的核心链路跑通。包含：
- 合成弱透镜收敛图（`|κ| << 1` 的弱场近似）。
- 在平坦天空近似下，用傅里叶核把 `κ -> (γ1, γ2)`。
- 注入形状噪声并做高斯平滑。
- 用 Kaiser-Squires 反演重建 `κ`。
- 用 `sklearn` 与 `PyTorch` 给出可解释校准诊断。

## R02

弱透镜观测通常获得背景星系椭率估计，在弱场近似中可视为剪切 `γ` 的带噪观测。该 MVP 聚焦的问题是：
- 已知（或测得）`γ1, γ2`，如何重建投影质量场 `κ`。
- 噪声与平滑对重建质量有何影响。
- 如何用简单线性/仿射校准减小系统偏差。

## R03

核心关系（平坦天空近似）：
- `κ = 1/2 (ψ,11 + ψ,22)`
- `γ1 = 1/2 (ψ,11 - ψ,22)`
- `γ2 = ψ,12`

在傅里叶空间（`k != 0`）：
- `γ1_hat = D1 * κ_hat`, `D1 = (kx^2 - ky^2) / (kx^2 + ky^2)`
- `γ2_hat = D2 * κ_hat`, `D2 = 2 kx ky / (kx^2 + ky^2)`

Kaiser-Squires 反演：
- `κ_hat = D1 * γ1_hat + D2 * γ2_hat`
- 零模 `κ_hat(0,0)` 不可观测，设为 `0`。

## R04

`demo.py` 不读取外部文件、无需交互输入。脚本内固定参数后直接输出三类结果：
- 地图统计表：`κ_true / γ_true / γ_obs` 的均值、标准差、最大绝对值。
- 重建质量表：`RMSE / MAE / Pearson`（raw、平滑、校准后）。
- 频域斜率与校准参数：`dlogP/dlogk`、`sklearn` 和 `torch` 的仿射系数。

## R05

默认配置（见 `WeakLensingConfig`）：
- `n_grid = 128`
- `pixel_scale_arcmin = 0.4`
- `sigma_e = 0.30`（单分量本征椭率弥散）
- `n_gal_per_pix = 30`
- `smooth_sigma_pix = 1.2`
- `rng_seed = 42`

其中噪声标准差采用 `sigma_e / sqrt(n_gal_per_pix)`。

## R06

实现策略：
- 先构造“真值”收敛图 `κ_true`（多个高斯团块 + 大尺度模式）。
- 用同一组傅里叶核完成前向模型 `κ -> γ` 和反演 `γ -> κ`，保证链路一致。
- 对观测剪切施加高斯平滑后再反演，抑制高频噪声放大。
- 用线性回归和 PyTorch 仿射拟合分别做偏差诊断与校准。

## R07

时间复杂度（`N = n_grid`）：
- FFT 主导：每次正反变换 `O(N^2 log N)`。
- 本脚本包含常数次 FFT（前向 + 多次反演），总复杂度仍为 `O(N^2 log N)`。
- 其余步骤（噪声、平滑、指标统计、回归）约为 `O(N^2)`。

默认 `128x128` 可在 CPU 秒级完成。

## R08

数值稳定措施：
- `k=0` 处核函数显式置零，避免除零。
- 强制 `κ_hat[0,0]=0`，与质量片退化（mass-sheet degeneracy 的零模）一致。
- 对带噪剪切先平滑再反演，降低高频噪声泄漏。
- 指标中相关系数在方差近零时返回 `NaN`，避免伪数值。

## R09

边界与失败条件：
- 该实现仅覆盖弱透镜近似，不适用于强透镜（`|κ|` 或 `|γ|` 接近 1）。
- 周期边界由 FFT 隐含引入，边缘效应被简化处理。
- 若输入网格过小，径向功率谱拟合点不足会返回 `NaN` 斜率。

## R10

MVP 诚实范围：
- 这是算法演示，不是完整观测管线。
- 未包含真实 survey 的掩膜、PSF、photo-z、不规则采样与 E/B 模分解。
- 未做宇宙学参数反演，仅展示“场重建 + 基础诊断”。

## R11

运行方式：

```bash
uv run python Algorithms/物理-广义相对论-0383-弱透镜_(Weak_Lensing)/demo.py
```

或在该目录内运行：

```bash
uv run python demo.py
```

## R12

输出读法：
- `weak-regime check=True` 说明合成场满足弱场前提。
- `smoothed_KS` 相比 `raw_KS` 通常有更低 `RMSE/MAE`、更高相关性。
- `sklearn` 与 `torch` 仿射系数接近时，表示校准结果一致性较好。
- 功率谱斜率可用于比较重建后与真值的尺度依赖是否偏软/偏硬。

## R13

最小验证清单：
- `README.md` 与 `demo.py` 不含待填充占位符。
- `uv run python demo.py` 可直接运行，无交互。
- 输出中 `max|kappa_true| < 0.1` 且 `max|gamma_true| < 0.1`。
- 平滑重建指标优于或接近原始重建，方向符合噪声抑制预期。

## R14

物理解释建议：
- `κ` 代表视线方向投影质量分布，峰值对应高密度结构。
- 剪切噪声反映有限星系数与本征形状散布造成的观测不确定性。
- Kaiser-Squires 反演本质是一个线性反卷积，易受高频噪声影响，因此平滑是常见第一步。

## R15

当前局限：
- 只使用规则网格与周期边界，未处理真实掩膜和不完整天空覆盖。
- 未显式区分 E/B 模污染。
- 未包含非高斯噪声、系统误差传播和协方差估计。

## R16

可扩展方向：
- 加入掩膜下的稀疏重建（如稀疏先验 / Wiener / MAP）。
- 在球面上实现 spin-2 谐波版本，替代平坦天空近似。
- 增加 tomographic bins，连接到 `P(k,z)` 与宇宙学参数拟合。
- 引入更真实的噪声与系统项（PSF、photo-z bias）。

## R17

工具栈与职责：
- `numpy`: 网格构造、FFT、核函数、统计量。
- `scipy` (`ndimage.gaussian_filter`): 剪切场平滑。
- `pandas`: 表格化输出，便于人工核验。
- `scikit-learn`: 线性回归与 `R2` 诊断。
- `PyTorch`: 通过自动微分拟合仿射校准参数 `m, b`。

## R18

`demo.py` 的源码级算法流程（8 步）：
1. `synthetic_kappa_map` 构造弱场 `κ_true`（多团块 + 大尺度模式）并去均值。
2. `fourier_kernels` 生成 `D1, D2` 核，显式处理 `k=0`。
3. `forward_shear_from_kappa` 做 FFT 前向映射，得到 `γ1_true, γ2_true`。
4. `add_shape_noise` 按 `sigma_e/sqrt(n_gal)` 注入观测噪声，形成 `γ_obs`。
5. `kaiser_squires_inversion` 对 `γ_obs` 直接反演得 `kappa_raw`，并对平滑后 `γ` 反演得 `kappa_smooth`。
6. `LinearRegression` 在像素级拟合 `kappa_true ≈ m*kappa_smooth + c`，得到 `kappa_skl` 与 `R2`。
7. `fit_affine_torch` 用梯度下降拟合同一仿射模型，得到 `kappa_torch` 与拟合误差。
8. `reconstruction_metrics` 与 `radial_power_slope` 汇总误差、相关性、频域斜率并打印报表。
