# 核磁共振 (NMR)

- UID: `PHYS-0455`
- 学科: `物理`
- 分类: `磁共振`
- 源序号: `478`
- 目标目录: `Algorithms/物理-磁共振-0478-核磁共振_(NMR)`

## R01

核磁共振（NMR）的基础观测量通常是时域自由感应衰减信号（FID）与其频域谱峰。本条目提供一个最小可运行 MVP：
- 构造多组分复数 FID（含噪声）；
- 通过加窗 + FFT 重建频谱并检测共振峰；
- 对每个峰基于功率谱半高宽（FWHM）反演 `T2*` 衰减时间；
- 输出“真值 vs 估计值”的误差表与 SNR 指标。

## R02

问题定义（对应 `demo.py`）：
- 输入：采样时间序列 `t` 与复数信号 `s(t)`（MVP 中由合成器生成）。
- 输出：
  - 主共振频率 `f_k`（Hz）；
  - 峰强度（谱幅值）；
  - 每个共振分量的 `T2*` 估计（秒）。
- 目标：在可控噪声下恢复多峰共振参数，并量化频率/弛豫参数误差。

## R03

简化物理模型：
- 单个共振分量可写为
  `s_k(t) = A_k * exp(-t/T2*_k) * exp(i*(2*pi*f_k*t + phi_k))`。
- 总 FID 是多个分量叠加再加复高斯噪声：
  `s(t) = sum_k s_k(t) + n(t)`。
- 频谱峰位置对应拉莫尔频率偏移，峰宽与 `T2*` 成反比（近似 `linewidth ~ 1/(pi*T2*)`）。

## R04

MVP 的合成数据设置：
- 采样率：`2048 Hz`，采样点：`2048`（时长约 1 秒）。
- 共振分量（真值）：
  - `peak_A`: `f=120 Hz`, `T2*=0.300 s`, `A=1.00`；
  - `peak_B`: `f=-210 Hz`, `T2*=0.140 s`, `A=0.70`；
  - `peak_C`: `f=410 Hz`, `T2*=0.090 s`, `A=0.45`。
- 噪声：复高斯噪声，`noise_std=0.035`。

## R05

核心计算公式：
- 加窗（指数展宽）：`w(t)=exp(-pi*lb*t)`。
- 频谱：`S(f)=FFT(s(t)*w(t))`，脚本使用零填充提升频率采样密度。
- 峰检测：在 `|S(f)|` 上做局部极大值搜索并按峰强排序。
- `T2*` 估计：对每个峰在功率谱 `|S(f)|^2` 上找半高宽 `FWHM`，并扣除人为加窗展宽 `lb`：
  `T2* ~= 1 / (pi * (FWHM - lb))`（若 `FWHM <= lb` 则返回 `NaN`）。

## R06

算法流程（高层）：
1. `generate_synthetic_fid` 生成多峰复数 FID。
2. `apodize_and_fft` 对 FID 加窗、零填充并 FFT 得到频谱。
3. `detect_resonance_peaks` 在幅度谱中找主峰。
4. 对每个峰调用 `estimate_t2_star_from_linewidth`，从半高宽反演 `T2*`。
5. `build_comparison_table` 将估计峰和真值按最邻近频率匹配。
6. 计算平均频率误差、`T2*` 相对误差、主峰 SNR 并输出表格。

## R07

复杂度分析（设采样点数 `N`，峰数 `K`）：
- 合成 FID：`O(KN)`。
- FFT：`O(N log N)`（零填充后按 `N_fft` 计）。
- 峰检测：`O(N)`。
- 每个峰的半高宽搜索（双向扫描）：`O(N)`，总计 `O(KN)`。
- 总体近似：`O(N log N + KN)`，空间复杂度 `O(N)`。

## R08

正确性直觉：
- 频率恢复：若采样率满足 Nyquist 且峰间隔大于分辨率，FFT 幅度谱峰位与真实频率近似一致。
- 弛豫恢复：指数衰减在频域表现为 Lorentz 峰型，功率谱半高宽与 `1/T2*` 成正比；从每个峰的 `FWHM` 可直接反演 `T2*`。
- 噪声鲁棒性：阈值筛选 + 中位数噪声底估计可减轻低 SNR 区域对拟合和 SNR 评估的干扰。

## R09

伪代码：

```text
time <- arange(N) / fs
fid <- sum_k A_k * exp(-time/T2_k) * exp(i*(2*pi*f_k*time + phi_k)) + complex_noise

freq, spec <- FFT_shift( FFT( fid * exp(-pi*lb*time), zero_filled ) )
peaks <- detect_local_maxima(abs(spec))
keep top-K by magnitude

for each detected peak idx:
    locate left/right half-power crossing around idx
    FWHM <- right_cross - left_cross
    intrinsic_width <- FWHM - line_broadening_hz
    T2_est <- 1 / (pi * intrinsic_width)

match each true component with nearest unmatched estimated frequency
report frequency error, T2 relative error, SNR
```

## R10

数值与边界处理：
- `zero_fill_factor` 强制 `>=1`，避免非法 FFT 长度。
- 峰检测若无候选，退化为全局最大峰，保证流程可继续。
- 半高宽左右侧 crossing 使用线性插值，减少频率栅格误差。
- 若峰无法找到半高宽交点，或 `FWHM <= line_broadening_hz`，`T2*` 返回 `NaN`。
- SNR 使用中位数噪声底并排除主峰邻域，降低泄漏对噪声估计的污染。

## R11

默认超参数：
- 随机种子：`478`（可复现）。
- 频谱处理：`line_broadening_hz=1.5`，`zero_fill_factor=4`。
- 峰检测：`max_peaks=3`，`min_separation_hz=40`，`prominence_ratio=0.08`。
- `T2*` 估计：功率谱半高宽 `FWHM`，并扣除 `line_broadening_hz=1.5` 的展宽项。

## R12

`demo.py` 实现范围：
- 覆盖从信号生成到参数恢复的完整闭环。
- 工具栈保持最小：`numpy + pandas + scipy(signal.find_peaks)`。
- 未把任务交给“单函数黑盒”：
  - FID 合成、加窗、FFT、半高宽搜索与 `T2*` 反演都在源码中显式展开；
  - `scipy` 仅用于峰候选定位，且提供 `_fallback_find_peaks` 兜底逻辑。

## R13

运行方式：

```bash
cd Algorithms/物理-磁共振-0478-核磁共振_(NMR)
uv run python demo.py
```

脚本无交互输入，执行后直接输出真值表、检测峰表、匹配误差表和聚合指标。

## R14

示例输出结构（固定 seed 下可复现，具体小数可能因环境浮点实现有细微差异）：

```text
=== Synthetic NMR FID setup ===
...

=== Ground-truth resonance components ===
component  amplitude  frequency_hz  t2_star_s  linewidth_hz_approx
...

=== Detected spectrum peaks ===
rank  frequency_hz  magnitude  t2_star_estimate_s
...

=== Matching: truth vs estimate ===
component  true_freq_hz  est_freq_hz  abs_freq_error_hz  true_t2_star_s  est_t2_star_s  t2_rel_error_pct
...

=== Aggregate quality metrics ===
mean_abs_frequency_error_hz: ...
mean_t2_relative_error_pct: ...
dominant_peak_snr_db: ...
```

## R15

最小验收清单：
- `README.md` 与 `demo.py` 不含任何模板占位符残留。
- `uv run python demo.py` 可直接完成运行。
- 输出至少包含：
  - 真值共振参数；
  - 检测到的峰频和 `T2*` 估计；
  - 频率误差与 `T2*` 相对误差；
  - 主峰 SNR（线性与 dB）。

## R16

当前 MVP 局限：
- 仅为单次 FID 的简化模拟，不包含真实仪器脉冲序列、相位循环、场不均匀补偿。
- 未建模化学位移标定（ppm）、J-coupling、多重峰精细结构与基线漂移。
- `T2*` 反演依赖 Lorentz 峰型与半高宽近似，对复杂线型（Voigt/强耦合）不够稳健。
- 峰匹配策略是“最近频率”贪心法，复杂谱图下可能产生错配。

## R17

可扩展方向：
- 引入真实 FID 数据读取与 ppm 轴标定（参考频率/磁场强度）。
- 在频域加基线校正、去卷积和非线性峰型拟合（Lorentz/Gauss/Voigt）。
- 用多峰联合优化替代逐峰独立估计，提高重叠峰条件下鲁棒性。
- 增加不确定度估计（bootstrap/蒙特卡洛）并输出置信区间。
- 扩展至 `T1/T2` 脉冲序列（如 IR/CPMG）参数反演。

## R18

`demo.py` 源码级算法流（8 步，非黑盒）：
1. `main` 设置采样率、时长、噪声和三组真值峰参数，构造时间网格 `time_s`。  
2. `generate_synthetic_fid` 对每个峰按 `A*exp(-t/T2*)*exp(i(2*pi*f*t+phi))` 叠加，并注入复高斯噪声，得到 FID。  
3. `apodize_and_fft` 施加指数窗 `exp(-pi*lb*t)`，零填充到 2 的幂次长度，再做 `fftshift(fft())` 获得中心化频谱。  
4. `detect_resonance_peaks` 在 `|spectrum|` 上做局部极值搜索（优先 `scipy.signal.find_peaks`，否则 fallback），并按峰高选前 `K` 个。  
5. 对每个检测峰，`estimate_t2_star_from_linewidth` 在功率谱中向左右扫描半高点，并用 `_interpolate_half_height_crossing` 线性插值交点频率。  
6. `estimate_snr` 以主峰高度除以去主峰邻域后的中位数噪声底，得到线性 SNR 与 dB。  
7. `build_comparison_table` 使用“每个真值峰匹配最近未使用估计峰”的策略，生成频率误差和 `T2*` 相对误差表。  
8. `main` 汇总 `truth_df/peak_df/comparison_df` 与聚合误差指标并打印，形成完整可验证输出。
