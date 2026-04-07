# 重子声学振荡 (Baryon Acoustic Oscillations)

- UID: `PHYS-0355`
- 学科: `物理`
- 分类: `宇宙学`
- 源序号: `373`
- 目标目录: `Algorithms/物理-宇宙学-0373-重子声学振荡_(Baryon_Acoustic_Oscillations)`

## R01

重子声学振荡（BAO）是早期光子-重子流体中的声波印记，在今天星系分布中表现为一个近固定“标准尺”尺度。
在频域（功率谱）中，BAO 表现为相对平滑谱 `P_nw(k)` 的振荡修饰；在实空间（相关函数）中，BAO 表现为 `xi(r)` 上约百 `Mpc/h` 的凸起峰。

## R02

本条目 MVP 目标是实现一个最小且可审计的 BAO 数值闭环：

1. 构造平滑无振荡基线 `P_nw(k)`；
2. 注入阻尼振荡得到 mock `P_obs(k)`；
3. 从 `P_obs/P_nw-1` 反演声学标尺 `r_s`；
4. 再将 `P(k)` 傅里叶变换到 `xi(r)` 并检测 BAO 峰位；
5. 用断言验证“恢复的尺度与输入真值一致”。

## R03

本实现采用的建模假设与单位：

- 单位：`k` 用 `h/Mpc`，`r` 与 `r_s` 用 `Mpc/h`；
- `P_nw(k)` 用教学型平滑形状，不追求 CAMB/CLASS 级精度；
- BAO 振荡采用阻尼正弦近似：`sin(k r_s) * exp[-(k Sigma_nl)^2/2]`；
- 使用固定随机种子生成噪声，确保输出可复现。

## R04

核心公式：

1. 平滑谱（教学近似）
`P_nw(k) = k^{n_s} / [1 + (k/k_turn)^alpha]`

2. 含 BAO 的功率谱
`P(k) = P_nw(k) * [1 + A_bao * sin(k r_s) * exp(-(k Sigma_nl)^2/2)]`

3. 相关函数变换
`xi(r) = (1/(2pi^2)) * int dk k^2 P(k) j0(kr)`

4. 比值拟合目标
`y(k)=P_obs/P_nw-1 ~= a * sin(k r_s) * exp(-(k Sigma_nl)^2/2)`

## R05

`demo.py` 的主流程：

1. `generate_mock_spectrum` 生成 `k, P_nw, P_true, P_obs`；
2. `estimate_sound_horizon_from_ratio` 扫描候选 `r_s` 并做最小二乘振幅拟合；
3. `power_to_correlation_function` 把 `P_obs(k)` 变换成 `xi(r)`；
4. `remove_broadband_trend` 用多项式去掉宽带背景；
5. `detect_bao_peak` 在搜索窗口内定位 BAO 峰；
6. `run_checks` 断言恢复精度和物理范围。

## R06

声学尺度反演算法（频域）细节：

- 对每个候选 `r_s` 构建模板
`t(k)=sin(k r_s) * exp(-(k Sigma_nl)^2/2)`；
- 闭式最小二乘求振幅
`a_hat = <y,t>/<t,t>`；
- 用 `MSE = mean((y-a_hat t)^2)` 作为评分；
- 取最小 MSE 的 `r_s` 为估计值。

该过程完全在源码中显式实现，不依赖外部 BAO 拟合黑盒。

## R07

BAO 峰位提取算法（实空间）细节：

1. 数值积分得到 `xi(r)`；
2. 在非峰区拟合低阶多项式，得到宽带基线；
3. 残差 `xi_res = xi - baseline` 保留局部 BAO bump；
4. 在 `[80,130] Mpc/h` 窗口内用 `find_peaks` 选最显著峰；
5. 若无局部峰则退化为窗口内最大值。

## R08

复杂度（`N_k` 为波数点数，`N_r` 为距离点数，`N_s` 为 `r_s` 搜索点数）：

- 生成 mock 谱：`O(N_k)`；
- `r_s` 扫描拟合：`O(N_s * N_k)`；
- `P(k)->xi(r)` 变换：`O(N_r * N_k)`；
- 峰检测与表格输出：`O(N_r)`。

默认参数 (`N_k=2500, N_r=400, N_s=601`) 在普通 CPU 可快速运行。

## R09

数值稳定性处理：

- `s_tt` 分母使用 `clip` 防止极端情况下除零；
- 噪声采用乘性小扰动并固定随机种子，避免不稳定回归；
- 相关函数积分使用 `scipy.integrate.simpson`，对振荡积分更稳健；
- 峰提取采用“`find_peaks` + fallback `argmax`”双保险。

## R10

代码结构：

- `BAOParams`：统一管理物理与数值参数；
- `smooth_no_wiggle_power` / `bao_wiggle_factor`：构造谱模型；
- `estimate_sound_horizon_from_ratio`：频域标尺反演核心；
- `power_to_correlation_function`：球贝塞尔核积分变换；
- `remove_broadband_trend` + `detect_bao_peak`：实空间峰提取；
- `run_checks`：自动化物理 sanity checks；
- `run_demo`：组织输出。

## R11

最小依赖栈：

- `numpy`：向量化数值计算；
- `scipy.integrate.simpson`：`k` 方向数值积分；
- `scipy.special.spherical_jn`：球贝塞尔函数 `j0`；
- `scipy.signal.find_peaks`：峰检测；
- `pandas`：结构化诊断表输出。

## R12

运行方式：

```bash
uv run python "Algorithms/物理-宇宙学-0373-重子声学振荡_(Baryon_Acoustic_Oscillations)/demo.py"
```

或在目标目录内：

```bash
uv run python demo.py
```

脚本无需交互输入，直接打印恢复结果与诊断表。

## R13

输出字段说明：

- `True r_s (input)`：mock 生成时设定的真值；
- `Estimated r_s from ratio fit`：频域拟合恢复的标尺；
- `Estimated wiggle amplitude`：最优模板对应振幅 `a_hat`；
- `Best-fit ratio MSE`：拟合残差强度；
- `Detected xi(r) peak`：实空间 BAO bump 位置；
- `Sample points of observed wiggle ratio`：用于快速检查振荡行为的样本表。

## R14

自检建议：

1. 把 `noise_fraction` 提高到 `0.05`，观察 `r_s` 误差增大；
2. 把 `bao_amplitude` 降到 `0.03`，检查峰检测是否仍稳定；
3. 把 `sigma_nl` 增大到 `10`，应看到高 `k` 振荡被更强阻尼；
4. 改 `sound_horizon_true`（如 `102`），验证拟合结果同步平移。

## R15

模型边界与局限：

- `P_nw(k)` 为教学近似，不包含完整转移函数物理；
- 未接入真实星系窗口函数、红移误差和系统学；
- 只做单参数 `r_s` 模板扫描，未联合拟合偏置/红移畸变；
- 因为是 MVP，结果用于流程验证，不用于精确宇宙参数推断。

## R16

可扩展方向：

1. 将平滑谱替换为 Eisenstein-Hu 或 CAMB/CLASS 输出；
2. 在拟合中加入 dilation 参数 `alpha`（`r_s -> alpha r_s`）并估计其后验；
3. 使用协方差矩阵做加权最小二乘而非均匀 MSE；
4. 接入真实巡天数据（BOSS/eBOSS/DESI）并进行 chi-square 拟合。

## R17

应用场景：

- 作为“标准尺”教学演示，解释 BAO 如何约束宇宙学距离；
- 作为更复杂 BAO 管线前的单元验证基线；
- 用于比较不同阻尼、噪声与拟合窗口对标尺恢复的影响；
- 与 SNe/CMB 等距离信息组合前的前处理原型。

## R18

`demo.py` 的源码级算法流（8 步，展开第三方调用）：

1. `run_demo` 构造 `BAOParams`，固定 `k/r` 网格、噪声幅度、真值 `r_s`。  
2. `generate_mock_spectrum` 先调用 `smooth_no_wiggle_power` 得到 `P_nw(k)`，再显式乘上 `bao_wiggle_factor` 生成 `P_true(k)`，最后用 `numpy` 伪随机数加噪得到 `P_obs(k)`。  
3. `estimate_sound_horizon_from_ratio` 构造候选 `r_s` 网格和模板矩阵 `sin(k r_s) * damping`，用向量化内积计算每个候选的闭式振幅 `a_hat`。  
4. 同一函数中用代数化 `SSE = <y,y> - <y,t>^2/<t,t>` 计算 `MSE`，避免逐候选循环构建残差矩阵；选最小 `MSE` 的候选为 `r_s_est`。  
5. `power_to_correlation_function` 使用 `scipy.special.spherical_jn(0,kr)` 生成球贝塞尔核，再把被积函数送入 `scipy.integrate.simpson`，按 `k` 轴积分得到 `xi(r)`。  
6. `remove_broadband_trend` 在非峰区执行 `numpy.polyfit/polyval`，得到宽带基线并形成残差 `xi_res`。  
7. `detect_bao_peak` 在峰搜索窗口内调用 `scipy.signal.find_peaks` 获取局部峰，再按峰高选择窗口内最高峰；若无峰则回退到 `argmax`。  
8. `run_checks` 对 `|r_s_est-r_s_true|`、峰位区间和振幅正性执行断言；通过后输出摘要与 `pandas` 诊断表。
