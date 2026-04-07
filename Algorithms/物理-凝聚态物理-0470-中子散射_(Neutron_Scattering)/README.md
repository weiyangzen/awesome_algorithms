# 中子散射 (Neutron Scattering)

- UID: `PHYS-0447`
- 学科: `物理`
- 分类: `凝聚态物理`
- 源序号: `470`
- 目标目录: `Algorithms/物理-凝聚态物理-0470-中子散射_(Neutron_Scattering)`

## R01

中子散射（这里聚焦非弹性中子散射, INS）常用于测量晶格/自旋激发的色散关系与线宽。
本条目给出一个最小可运行 MVP，完成以下闭环：

1. 生成带计数噪声的合成谱图 `I(Q, omega)`；
2. 对每个 `Q` 切片拟合峰位与线宽；
3. 回归恢复色散参数 `c` 与 `gap`；
4. 用 `PyTorch` 对整张谱图做全局参数反演一致性校验。

## R02

本 MVP 采用一个可解释的简化模型：

- 色散关系：`omega(Q) = sqrt(gap^2 + c^2 * xi(Q)^2)`；
- 约化动量：`xi(Q) = 2 sin(Qa/2)`；
- 线宽：`gamma(Q) = gamma0 + gamma1 * xi(Q)^2`；
- 光谱函数：`S ~ (n+1)L(omega,+omega0,gamma_eff) + nL(omega,-omega0,gamma_eff)`；
- 其中 `n` 为 Bose 因子，`L` 为洛伦兹峰，`gamma_eff = gamma + sigma_res` 近似包含仪器展宽。

同时加入 `Q` 相关强度包络、弹性线（零能转移高斯峰）和常数背景。

## R03

核心物理量与可观测量：

- 输入网格：`Q in [0.25, 2.55] A^-1`, `omega in [-25, 25] meV`；
- 真值参数：`c_true=6.8 meV`, `gap_true=3.2 meV`, `gamma0_true=0.45 meV`, `gamma1_true=0.30 meV`；
- 拟合输出：`omega0_fit(Q)`, `gamma_fit(Q)`, `slice_r2`, `slice_mae`；
- 回归输出：`c_fit`, `gap_fit`, `dispersion_r2`；
- 全局优化输出：`torch_*` 参数与 `torch_nmse`。

## R04

算法拆分为四段：

1. 前向合成：按每个 `Q` 计算 `I_clean(Q,omega)`，再用泊松采样生成 `I_noisy`；
2. 局部拟合：用 `scipy.optimize.curve_fit` 对每条 `Q` 切片拟合 `amp, omega0, gamma, elastic, bg`；
3. 色散回归：将 `omega0_fit(Q)^2` 对 `xi(Q)^2` 做线性回归恢复 `c, gap`；
4. 全局反演：用 `torch` 对整张谱图联合优化参数并计算 NMSE。

## R05

最小工具栈及职责：

- `numpy`：网格、前向模型、噪声采样、数学运算；
- `scipy`：`curve_fit` 执行每个 `Q` 切片非线性拟合；
- `pandas`：拟合结果表与谱图表导出 CSV；
- `scikit-learn`：线性回归与 `MAE/MSE/R2` 指标计算；
- `PyTorch`：整图可微模型与全局参数优化。

第三方库只提供数值工具，模型公式、特征构造和验证门槛都在 `demo.py` 显式实现。

## R06

`demo.py` 运行后产出：

1. 终端打印配置与指标摘要；
2. `q_cut_fit_results.csv`：每个 `Q` 切片的拟合参数与质量指标；
3. `map_summary.csv`：完整 `Q-omega` 网格下的 clean/noisy 强度；
4. 内置质量门槛全部通过后输出 `All checks passed.`。

## R07

脚本中的正确性门槛（质量闸）：

1. `success_ratio >= 0.85`（切片拟合成功率）；
2. `slice_r2_mean >= 0.80`（平均切片拟合质量）；
3. `dispersion_r2 >= 0.92`（色散回归质量）；
4. `|c_fit - c_true| <= 1.1 meV`；
5. `|gap_fit - gap_true| <= 1.2 meV`；
6. `torch_nmse <= 0.15`（全局反演归一化误差）。

这些门槛兼顾了计数噪声下的稳定性和参数可恢复性。

## R08

默认参数（`NeutronScatteringConfig`）：

- 温度：`140 K`
- `Q` 点数：`22`
- `omega` 点数：`281`
- 晶格常数：`a = 3.35 A`
- 分辨率：`sigma_res = 0.75 meV`
- 真值：`c=6.8`, `gap=3.2`, `gamma0=0.45`, `gamma1=0.30`
- 强度：`amp_scale=165`, `elastic_scale=34`, `background=0.9`
- Torch 优化：`epochs=420`, `lr=0.05`

参数取值保证 `uv run python demo.py` 在普通 CPU 上数秒内完成。

## R09

复杂度（`Nq=n_q`, `Nw=n_omega`, `I=curve_fit 迭代步`, `E=torch_epochs`）：

- 前向合成：`O(Nq * Nw)`；
- 切片拟合：`O(Nq * I * Nw)`；
- 色散回归：`O(Nq)`；
- 全局优化：`O(E * Nq * Nw)`。

默认配置 (`22 x 281`, `E=420`) 下运行时间约 2-3 秒。

## R10

运行方式（无交互）：

```bash
cd Algorithms/物理-凝聚态物理-0470-中子散射_(Neutron_Scattering)
uv run python demo.py
```

或在仓库根目录执行：

```bash
uv run python Algorithms/物理-凝聚态物理-0470-中子散射_(Neutron_Scattering)/demo.py
```

## R11

输出解读建议：

1. `slice_r2_mean` 反映逐 `Q` 谱线拟合整体质量；
2. `omega_mae_meV` 直接衡量峰位恢复误差；
3. `c_fit_meV` / `gap_fit_meV` 对应色散参数恢复精度；
4. `torch_nmse` 检查整图层面的全局一致性；
5. 若 `c/gap` 恢复准确但 `torch_gamma*` 偏差较大，通常表示参数相关性导致的可辨识性限制。

## R12

本地实测（命令：`uv run python demo.py`）关键结果：

- `success_ratio = 1.000000`
- `slice_r2_mean = 0.834501`
- `omega_mae_meV = 0.088539`
- `c_fit_meV = 6.817257`（真值 `6.8`）
- `gap_fit_meV = 3.122291`（真值 `3.2`）
- `dispersion_r2 = 0.997848`
- `torch_c_fit_meV = 6.808309`
- `torch_gap_fit_meV = 3.096462`
- `torch_nmse = 0.090969`

结论：在泊松计数噪声下，色散参数可稳定恢复，整图全局误差也处于可接受范围。

## R13

适用范围：

- 中子散射数据分析流程教学；
- 反演算法回归测试（噪声下参数恢复能力）；
- 从前向模拟到逆向拟合的端到端原型验证。

局限：

- 使用单分支、各向同性、经验线宽模型；
- 未包含多模耦合、详细仪器分辨函数、磁矩矩阵元、背景系统误差；
- 不能直接替代真实束线数据的最终科学结论。

## R14

常见问题与排查：

1. 峰位拟合不稳：通常是噪声过高或初值偏离过大，可增加计数或调整初值范围；
2. 回归 `R2` 异常低：检查 `omega0_fit` 是否出现大量失败值；
3. Torch 发散：降低学习率或增加 `epochs`；
4. 低能区偏差：多由弹性峰参数不匹配引起；
5. `c_fit/gap_fit` 偏差大：扩大 `Q` 覆盖区间能改善可辨识性。

## R15

可扩展方向：

1. 引入多分支激发（如声子+磁振子）联合拟合；
2. 将 `gamma(Q)` 从经验式替换为微观散射机制模型；
3. 使用更真实的分辨函数卷积（非高斯尾部、能量依赖分辨率）；
4. 将全局优化扩展为贝叶斯后验估计，输出参数置信区间；
5. 对接实验数据文件格式（HDF5/NeXus）实现真实数据流程。

## R16

与相关方法关系：

- 与 IXS（非弹性 X 射线散射）流程类似，都是从 `S(Q,omega)` 反演色散与线宽；
- 与中子衍射不同，本条目关注动态响应而非仅弹性结构峰；
- 与完整第一性原理谱函数计算相比，本实现是可审计、可快速迭代的工程原型；
- 与黑盒拟合不同，核心公式与参数流在源码中可逐步追踪。

## R17

本目录最小交付状态：

1. `README.md`：R01-R18 已完整填写；
2. `demo.py`：可运行且无需交互输入；
3. `meta.json`：与任务元数据一致；
4. 运行后自动生成 `q_cut_fit_results.csv`、`map_summary.csv`；
5. 脚本内置质量门槛，满足后明确输出 `All checks passed.`。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `NeutronScatteringConfig` 固定温度、`Q-omega` 网格和真值参数，`check_config` 做边界检查。  
2. `simulate_dataset` 逐 `Q` 调用 `intensity_slice`：先算 `xi(Q)`、`omega(Q)`、`gamma(Q)`，再构造 Stokes/anti-Stokes 洛伦兹峰、弹性高斯峰与背景。  
3. 在同一函数中显式计算 Bose 因子 `n(E,T)` 与 `Q` 包络函数 `q_form_factor(Q)`，得到 clean map。  
4. 对 clean map 做逐点泊松采样，得到 noisy map，形成“实验观测”。  
5. `fit_single_q_cut` 使用 `curve_fit` 对每个 `Q` 切片拟合 `amp, omega0, gamma, elastic_amp, bg`，并计算 `slice_r2/slice_mae`。  
6. `regress_dispersion` 将拟合得到的 `omega0_fit^2` 对 `xi^2` 做线性回归，显式恢复 `c_fit=sqrt(slope)` 与 `gap_fit=sqrt(intercept)`。  
7. `torch_global_refinement` 复现同一谱图方程的可微版本，在整图上联合优化参数并计算 `torch_mse/torch_nmse`。  
8. `main` 汇总指标并执行质量闸断言，通过后导出 `q_cut_fit_results.csv`、`map_summary.csv` 并打印结果。
