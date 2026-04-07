# 电子自旋共振 (ESR/EPR)

- UID: `PHYS-0456`
- 学科: `物理`
- 分类: `磁共振`
- 源序号: `479`
- 目标目录: `Algorithms/物理-磁共振-0479-电子自旋共振_(ESR／EPR)`

## R01

电子自旋共振（ESR/EPR）在连续波（CW）实验中常用“固定微波频率 + 扫磁场”的方式采样，仪器输出通常是吸收线型对磁场的一阶导数信号。本条目提供一个最小可运行 MVP：
- 合成多组分 Lorentz 吸收导数谱（含高斯噪声）；
- 在导数谱中自动定位共振线中心场 `B0`；
- 由 `h*nu = g*mu_B*B0` 反演 `g` 因子；
- 由导数峰谷间距估计线宽（`DeltaB_pp` 与 `FWHM`）；
- 输出真值-估计值误差表与聚合指标。

## R02

问题定义（对应 `demo.py`）：
- 输入：磁场网格 `B (mT)` 与一维导数谱 `dI/dB`（MVP 内部合成，无交互输入）。
- 输出：
  - 共振中心场 `B0 (mT)`；
  - `g` 因子估计；
  - 线宽 `DeltaB_pp`（导数峰谷距）与 `FWHM`（吸收半高全宽）。
- 目标：在可控噪声下，恢复多组分 ESR 参数，并给出场误差、`g` 误差、线宽误差。

## R03

简化物理模型：
- 共振条件：`h*nu = g*mu_B*B0`。
- 吸收线型（Lorentz）：
  `L(B) = A / (1 + ((B-B0)/gamma)^2)`，其中 `gamma` 为半高半宽（HWHM）。
- CW-ESR 常测导数谱：
  `dL/dB = A * (-2x) / (gamma * (1 + x^2)^2)`，`x=(B-B0)/gamma`。
- 对 Lorentz 导数：`FWHM = 2*gamma`，且 `DeltaB_pp = 2*gamma/sqrt(3)`，故
  `FWHM = sqrt(3) * DeltaB_pp`。

## R04

MVP 合成数据设置：
- 微波频率：`nu = 9.50 GHz`（X-band 典型量级）。
- 扫场区间：`332 ~ 347 mT`，采样点 `4096`。
- 三个自旋组分真值：
  - `radical_A`: `g=2.0035`, `A=1.00`, `HWHM=0.19 mT`；
  - `radical_B`: `g=2.0082`, `A=0.72`, `HWHM=0.26 mT`；
  - `radical_C`: `g=1.9986`, `A=0.55`, `HWHM=0.17 mT`。
- 噪声：零均值高斯噪声，`noise_std=0.035`。

## R05

核心计算关系：
- 由 `g` 求中心场：`B0 = h*nu/(g*mu_B)`，并从 Tesla 转为 mT。
- 由中心场反推 `g`：`g = h*nu/(mu_B*B0)`。
- 导数线宽反演：先测 `DeltaB_pp = B_trough - B_peak`，再得
  `FWHM = sqrt(3) * DeltaB_pp`。
- 参数评价：
  - `|B0_est - B0_true|`；
  - `g_error_ppm = 1e6*|g_est-g_true|/g_true`；
  - `FWHM` 相对误差（%）。

## R06

算法流程（高层）：
1. 用 `ESRComponent` 配置多组分参数（`g`、幅值、线宽）。
2. `generate_synthetic_cw_esr` 生成干净导数谱与加噪导数谱。
3. 对导数谱做 `moving_average` 平滑以抑制高频噪声。
4. 在平滑信号中检测局部峰/谷与 `+ -> -` 零交叉锚点。
5. 用“左峰 + 右谷”对构成候选谱线，估计 `B0` 与 `DeltaB_pp`。
6. 把 `DeltaB_pp` 换算成 `FWHM`，并由 `B0` 反演 `g`。
7. 与真值做最近邻匹配，构建误差表。
8. 输出聚合指标（平均场误差、平均 `g` 误差 ppm、平均线宽相对误差、SNR）。

## R07

复杂度分析（设采样点 `N`，谱线数 `K`）：
- 数据生成：`O(KN)`。
- 平滑卷积（固定窗口）：`O(N)`。
- 峰/谷检测：`O(N)`。
- 零交叉配对与候选构建：`O(N)`（搜索窗口有上限）。
- 真值匹配：`O(K^2)`（本 MVP 的 `K` 很小，可视为常数）。
- 总体：`O(KN + N)`，空间复杂度 `O(N)`。

## R08

正确性直觉：
- `B0` 对应导数谱的过零点（同一条线左正右负）。
- Lorentz 导数的峰谷间距与原吸收线宽存在解析关系，可直接闭式反演。
- `g` 与 `B0` 通过共振条件一一对应，因此只要场轴标定正确，`g` 可稳定恢复。
- 加平滑与最小间距约束能减少噪声引入的伪峰影响。

## R09

伪代码：

```text
given field grid B, microwave frequency nu, true components

signal <- sum_j d/dB Lorentz(B; B0_j, gamma_j, A_j) + noise
smooth <- moving_average(signal)

peaks <- local maxima(smooth)
troughs <- local minima(smooth)
zeros <- indices where smooth[i] > 0 and smooth[i+1] <= 0

for each zero crossing z:
    choose nearest left peak (within search window)
    choose nearest right trough (within search window)
    DeltaB_pp <- B_trough - B_peak
    B0_est <- interpolated zero crossing field
    FWHM_est <- sqrt(3) * DeltaB_pp
    g_est <- h*nu/(mu_B*B0_est)

select top-K by score and enforce minimum line separation
match each truth line with nearest unmatched estimated line
report absolute/relative errors and aggregate metrics
```

## R10

数值与边界处理：
- 若 `scipy` 不可用，使用 `_fallback_find_peaks` 完成本地极值检测。
- `moving_average` 自动把偶数窗口修正为奇数，保证对称性。
- `_interp_zero_cross` 使用线性插值减小栅格误差。
- 若某个零交叉附近找不到合法左峰/右谷组合，该候选会被跳过。
- `estimate_snr` 计算时排除主峰附近区间，避免把信号当噪声。

## R11

默认超参数：
- 随机种子：`479`。
- 线数上限：`max_lines=3`。
- 线分离阈值：`min_separation_mt=0.45`。
- 极值显著性比例：`prominence_ratio=0.11`。
- 配对搜索窗：`search_window_mt=0.75`。
- 平滑窗口：`smoothing_window=11`。

## R12

`demo.py` 覆盖范围与依赖：
- 工具栈：`numpy + pandas + scipy.signal.find_peaks(可选)`。
- 非黑盒实现：
  - 谱线合成、导数模型、零交叉插值、线宽反演、`g` 反演都在源码中显式实现；
  - `scipy` 仅用于辅助极值候选搜索，并带纯 `numpy` 兜底路径。
- 输出格式结构化（`DataFrame`），便于后续自动验证。

## R13

运行方式：

```bash
cd Algorithms/物理-磁共振-0479-电子自旋共振_(ESR／EPR)
uv run python demo.py
```

脚本无交互输入，直接打印真值参数、检测结果、匹配误差与聚合质量指标。

## R14

示例输出结构（固定 seed 下可复现，浮点末位可能有微小差异）：

```text
=== Synthetic CW-ESR setup ===
...

=== Ground-truth ESR components ===
component  amplitude  g_true  center_field_mt_true  ...
...

=== Detected ESR lines ===
rank  center_field_mt  g_estimate  linewidth_pp_mt  linewidth_fwhm_mt  ...
...

=== Matching: truth vs estimate ===
component  field_true_mt  field_est_mt  field_abs_error_mt  g_true  g_est  g_error_ppm  ...
...

=== Aggregate quality metrics ===
mean_abs_field_error_mt: ...
mean_g_error_ppm: ...
mean_fwhm_relative_error_pct: ...
dominant_abs_signal_snr_db: ...
```

## R15

最小验收清单：
- `README.md` 与 `demo.py` 不含模板占位符。
- `uv run python demo.py` 可直接运行且无交互。
- 输出至少包含：
  - 真值 `g/B0/线宽`；
  - 检测线的 `B0`、`g`、`DeltaB_pp/FWHM`；
  - 真值-估计误差表；
  - 聚合指标与 SNR。

## R16

当前 MVP 局限：
- 只建模了单导数 Lorentz 线型，未覆盖 Gaussian/Voigt 混合展宽。
- 未显式模拟超精细分裂、多重峰耦合与各向异性粉末谱。
- 采用启发式配对与最近邻匹配，复杂重叠谱场景下可能失配。
- 未建模实验系统误差（场漂移、调制幅度过大导致畸变、基线漂移）。

## R17

可扩展方向：
- 加入 Voigt 或多组分联合非线性拟合，提高重叠峰下稳健性。
- 引入超精细耦合常数 `A` 的自动识别与同位素模式匹配。
- 在参数估计中加入不确定度评估（bootstrap/蒙特卡洛）。
- 增加真实数据读入与仪器校准链路（场标定、频率漂移修正）。
- 扩展到时域脉冲 EPR 场景（例如回波衰减参数反演）。

## R18

`demo.py` 源码级算法流（8 步，非黑盒）：
1. `main` 定义微波频率、扫场网格与 3 个 `ESRComponent`（`g`、幅值、HWHM）。
2. `generate_synthetic_cw_esr` 通过 `resonance_field_mt` 把每个 `g` 转成 `B0`，再叠加 `lorentzian_derivative` 得到干净导数谱并注入噪声。
3. `detect_esr_lines` 先对导数谱执行 `moving_average` 平滑，再调用 `find_local_extrema` 提取峰/谷候选（`scipy` 或 fallback）。
4. 在平滑信号中定位 `+ -> -` 零交叉锚点，使用 `_interp_zero_cross` 做线性插值求更精细的中心场 `B0_est`。
5. 针对每个锚点，搜索窗内按最近邻配对“左峰 + 右谷”，得到 `DeltaB_pp`，并按 `FWHM = sqrt(3)*DeltaB_pp` 反演线宽。
6. 通过 `estimate_g_factor` 按 `g = h*nu/(mu_B*B0_est)` 计算 `g_est`，并按峰谷幅值构造候选评分 `score`。
7. 对候选线按 `score` 排序并施加最小线间距约束，选择前 `K` 条输出为检测结果。
8. `build_comparison_table` 将真值与估计线按最近中心场匹配，计算场误差、`g` 误差 ppm、线宽相对误差，并在 `main` 汇总打印。
