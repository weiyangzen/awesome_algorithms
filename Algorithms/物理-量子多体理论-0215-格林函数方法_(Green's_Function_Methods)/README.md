# 格林函数方法 (Green's Function Methods)

- UID: `PHYS-0214`
- 学科: `物理`
- 分类: `量子多体理论`
- 源序号: `215`
- 目标目录: `Algorithms/物理-量子多体理论-0215-格林函数方法_(Green's_Function_Methods)`

## R01

格林函数方法是量子多体理论的核心语言之一：把“相互作用体系的激发与响应”编码进传播子 `G(k, \omega)`。  
对电子体系而言，单粒子 retarded 格林函数 `G^R` 直接关联可观测谱函数：

`A(k, \omega) = -Im G^R(k, \omega) / \pi`

从 `A(k, \omega)` 可得到：
- 动量分辨光谱线形（ARPES 语境）；
- 态密度 `DOS(\omega)`；
- 占据数 `n_k` 与准粒子寿命信息。

本条目给出一个“最小可运行且可审计”的 Dyson 方程 MVP，用参数化自能展示相互作用如何重塑谱峰。

## R02

本条目的目标是实现一个教学级、可验证的 Green's function pipeline：

1. 在 1D 晶格上构造非相互作用色散 `\varepsilon_k`；
2. 构造满足因果性的 retarded 自能 `\Sigma^R(\omega)`；
3. 用 Dyson 方程求 `G^R(k,\omega)`；
4. 计算 `A(k,\omega)`、`DOS(\omega)`、`n_k`；
5. 对比 interacting 与 non-interacting 的谱峰位置，量化峰位偏移；
6. 加入数值阈值检查（和规则、因果性、占据范围等），输出 `PASS/FAIL`。

## R03

MVP 使用的核心方程：

1. 非相互作用色散：

`\varepsilon_k = -2t\cos k - \mu`

2. 模型 retarded 自能（单极点，含阻尼）：

`\Sigma^R(\omega) = U^2 / (\omega + \omega_0 + i\gamma)`

其中 `\gamma > 0`，保证 `Im\Sigma^R(\omega) \le 0`。

3. Dyson 方程：

`G^R(k,\omega) = 1 / (\omega + i\eta - \varepsilon_k - \Sigma^R(\omega))`

4. 谱函数、态密度、占据数：

`A(k,\omega) = -Im G^R(k,\omega)/\pi`

`DOS(\omega) = (1/N_k) \sum_k A(k,\omega)`

`n_k = \int d\omega\, A(k,\omega) f(\omega, T)`

其中 `f` 为费米分布。

## R04

模型设定与近似：

- 单带 1D 晶格模型；
- 自能只依赖频率，不依赖动量（`local/self-energy-only` 风格近似）；
- 自能采用参数化极点结构，不从第一性原理图展开逐项求和；
- 使用 retarded 形式，显式保留 `i\eta`；
- 频率积分在有限窗口离散完成（非无限区间解析积分）。

该设定适合教学演示 Dyson + 谱函数机制，不用于材料级精确预测。

## R05

`demo.py` 默认参数（`GreenConfig`）：

- 网格：`nk=81`, `k_max=\pi`, `nw=2201`, `omega\in[-8,8]`；
- 色散：`t=1.0`, `\mu=0.0`；
- 正则化：`eta=0.04`；
- 相互作用自能：`U=1.45`, `omega_0=1.20`, `gamma=0.35`；
- 温度：`T=0.10`；
- 峰值比较窗口：`|\omega|<=2.5`。

这组参数会产生可见的谱峰位移与有限散射率，同时保持数值稳定。

## R06

输入输出约定：

- 输入：脚本内部固定参数（无交互、无命令行必需参数）；
- 输出：
1. 代表性 `k` 点诊断表（`\epsilon_k`, `n_k`, 和规则误差）；
2. 汇总表（峰位、峰高、`Z` 因子、散射率、积分误差等）；
3. 阈值检查列表与 `Validation: PASS/FAIL`。

若任一关键阈值失败，程序以非零退出码结束。

## R07

算法流程（高层）：

1. 构造 `k` 网格与 `\omega` 网格；
2. 计算 `\varepsilon_k`；
3. 计算 `\Sigma^R(\omega)`；
4. 分别求 interacting 与 non-interacting 的 `G^R`；
5. 计算 `A(k,\omega)` 与 `DOS(\omega)`；
6. 进行频率积分得到 `n_k` 与谱和规则误差；
7. 在近费米窗口提取主峰并比较峰位偏移；
8. 估算 `Z=[1-\partial_\omega Re\Sigma|_{\omega=0}]^{-1}` 与散射率；
9. 汇总指标并执行阈值校验。

## R08

复杂度分析（`N_k = nk`, `N_\omega = nw`）：

- 构造 `G^R(k,\omega)`：`O(N_\omega N_k)`；
- 构造谱函数与 DOS：`O(N_\omega N_k)`；
- 数值积分（对每个 `k`）：`O(N_\omega N_k)`；
- 峰值检测：`O(N_\omega)`。

总体时间复杂度 `O(N_\omega N_k)`，空间复杂度同样为 `O(N_\omega N_k)`（主耗在 `G` 与 `A` 矩阵）。

## R09

数值稳定性策略：

- 强制 `nk` 为奇数，确保 `k=0` 对称采样；
- 强制 `eta>0`, `gamma>0`, `T>0`，避免非物理极点；
- Fermi 指数输入做 `clip`，防止溢出；
- 对峰值检测设置最低阈值与 `argmax` 回退路径；
- 对 `\omega=0` 处导数使用中心差分并检查索引在内部；
- 用有限性检查 + 因果性检查（`Im G^R <= 0`）做后验约束。

## R10

MVP 技术栈：

- `numpy`：向量化复数格林函数与谱函数计算；
- `scipy.integrate.simpson`：频率积分（和规则、占据数、DOS 积分）；
- `scipy.signal.find_peaks`：谱峰定位；
- `pandas`：结构化诊断表输出。

实现没有调用“黑盒多体求解器”；Dyson、自能、谱映射和验收逻辑均在源码显式展开。

## R11

运行方式：

```bash
cd Algorithms/物理-量子多体理论-0215-格林函数方法_(Green's_Function_Methods)
uv run python demo.py
```

脚本不需要交互输入。

## R12

关键输出字段说明：

- `peak_energy_interacting / noninteracting`：近 `k_F` 处主谱峰能量；
- `peak_shift_interacting_minus_noninteracting`：相互作用导致的峰位偏移；
- `quasiparticle_residue_Z`：准粒子权重估计；
- `im_sigma_at_fermi`：`\omega=0` 处自能虚部；
- `fermi_scattering_rate`：`-2 Z Im\Sigma(0)`；
- `mean_sum_rule_error`, `max_sum_rule_error`：`\int A d\omega = 1` 的离散误差；
- `dos_integral`：`\int DOS(\omega)d\omega`；
- `occupation_min/max`：`n_k` 物理范围检查；
- `max_imag_green`：retarded 因果性检查量；
- `min_spectral_value`：谱函数非负性检查量。

## R13

`demo.py` 内置验收阈值：

1. 全部关键数组有限；
2. `min A(k,\omega) > -1e-7`；
3. `max Im G^R <= 1e-8`；
4. `mean` 和规则误差 `< 5e-2`；
5. `max` 和规则误差 `< 1.2e-1`；
6. `DOS` 积分位于 `[0.90, 1.10]`；
7. `n_k` 保持在 `[0,1]`（含微小数值容差）；
8. 费米面散射率 `> 0`；
9. interacting 与 non-interacting 主峰有可分辨偏移（`|\Delta\omega| > 2e-2`）。

全部通过时输出 `Validation: PASS`。

## R14

当前实现局限：

- 仅单带模型，不含自旋/轨道/多带耦合；
- 自能是参数模型，不是从费曼图自洽求解（如 GW/DMFT 自洽环）；
- 未做 `k` 依赖自能与顶点修正；
- 未包含实材料矩阵元、晶格结构细节；
- 峰位与宽度仅用于机制演示，不能直接对实验定量拟合。

## R15

可扩展方向：

- 升级到自洽 Dyson 循环（更新 `\Sigma` 与 `G`）；
- 引入动量依赖自能 `\Sigma(k,\omega)`；
- 接入 Matsubara 轴并做解析延拓；
- 结合 DMFT impurity solver 或 GW 自能作为输入；
- 扩展到两粒子格林函数与响应函数（极化、磁化率）。

## R16

典型应用语境：

- 多体课程中讲解“自能如何改写谱函数”；
- 从理论角度理解 ARPES 峰移与展宽；
- 建立从 `G` 到 `A/DOS/n_k` 的可复现实验前处理流程；
- 在更复杂求解器前，做参数敏感性与数值管线验证。

## R17

与相关方法关系：

- 非相互作用能带法：计算简单，但无法给出寿命/展宽；
- 微扰图展开：可系统改进，但实现复杂且常需重整化技巧；
- DMFT/GW：更接近生产级，但工程成本高。

本条目选用“参数化自能 + Dyson”的最小路径，保留 Green's function 方法的核心机制，同时保证脚本可直接运行与审计。

## R18

`demo.py` 源码级算法流（9 步）：

1. `GreenConfig` 固定离散规模、色散参数、`\Sigma` 参数和阈值相关数值，保证实验可复现。  
2. `build_k_grid` 与 `build_omega_grid` 构造对称 `k` 网格和频率网格，并对规模与边界做合法性检查。  
3. `dispersion` 计算 `\varepsilon_k=-2t\cos k-\mu`；`retarded_self_energy` 计算 `\Sigma^R(\omega)=U^2/(\omega+\omega_0+i\gamma)`。  
4. `retarded_green` 用 Dyson 方程分别构造 interacting 与 non-interacting 的 `G^R(k,\omega)`。  
5. `spectral_function` 把 `G^R` 映射为 `A(k,\omega)`，随后对 `k` 平均得到 `DOS(\omega)`。  
6. `simpson` 在频率轴执行数值积分，得到 `\int A d\omega` 和规则误差、`n_k` 与总占据。  
7. `dominant_peak` 基于 `find_peaks`（无峰时回退 `argmax`）提取近 `k_F` 主峰，并计算 interacting 与 non-interacting 峰位偏移。  
8. 在 `\omega=0` 邻域用中心差分估计 `\partial_\omega Re\Sigma`，计算 `Z` 因子与散射率 `-2ZIm\Sigma(0)`。  
9. `main` 汇总表格与 9 条阈值检查，逐条输出 `PASS/FAIL`，若任一失败则 `SystemExit(1)`。

说明：本实现只把 `scipy` 用作通用积分/峰检工具；多体物理逻辑（自能建模、Dyson、谱函数与验收指标）在源码中显式展开，不是黑盒封装。
