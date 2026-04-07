# 非弹性X射线散射 (Inelastic X-ray Scattering)

- UID: `PHYS-0461`
- 学科: `物理`
- 分类: `凝聚态物理`
- 源序号: `484`
- 目标目录: `Algorithms/物理-凝聚态物理-0484-非弹性X射线散射_(Inelastic_X-ray_Scattering)`

## R01

非弹性 X 射线散射（IXS）用于测量材料的动态结构因子 `S(Q, omega)`，可解析声子、磁激发等低能激发的色散与寿命。

本条目把 IXS 从概念落成一个可运行最小闭环：
- 生成带仪器分辨率和计数噪声的合成 IXS 谱图 `I(Q, omega)`；
- 对每个 `Q` 切片做非线性拟合，提取 `Omega(Q)` 和 `Gamma(Q)`；
- 用 `scikit-learn` 回归色散关系 `Omega^2 ~ Q^2`；
- 用 `PyTorch` 做全局参数反演，验证“前向模拟 + 逆向估计”一致性。

## R02

IXS 在凝聚态中的核心观测量可写为：

`I(Q, omega) ~ S(Q, omega) ⊗ R(omega) + background`

其中：
- `S(Q, omega)` 是物理信号（本 MVP 用单声子 Stokes/anti-Stokes 近似）；
- `R(omega)` 是能量分辨函数（本实现用高斯）；
- `⊗` 表示卷积；
- `background` 代表非结构背景。

在有限温度下，Stokes 与 anti-Stokes 强度受玻色占据数 `n(omega,T)` 调制，因此同一声子模在 `+Omega` 与 `-Omega` 侧峰强不对称。

## R03

本任务的计算拆解：
1. 构建 `Q`-`omega` 网格与真值参数（色散、阻尼、强度包络）。
2. 逐 `Q` 生成理论单切片谱 `I_Q(omega)`。
3. 对每条切片施加高斯分辨率展宽，并加入泊松计数噪声。
4. 逐 `Q` 用受约束非线性最小二乘拟合 `Omega/Gamma/Amp/Elastic/Bg`。
5. 汇总为 `pandas` 表并评估 `R^2`、`MAE`。
6. 用线性回归拟合 `Omega(Q)^2 = c^2 Q^2 + gap^2`。
7. 用 `torch` 在整张谱图上联合优化 `c, gap, g0, g2` 等全局参数。

## R04

模型简化假设：
- 只保留一个有效声子分支，忽略多声子与多分支耦合；
- 线宽采用经验形式 `Gamma(Q)=g0+g2*Q^2`；
- 强度包络用经验函数 `A(Q)` 和 `Elastic(Q)`，不显式求矩阵元；
- 仪器分辨率仅建模为高斯核，不含非高斯尾部与系统漂移。

该简化适用于算法流程验证，不用于真实实验的最终物理结论。

## R05

`demo.py` 中的核心公式：

1. 玻色占据数：
`n(E,T) = 1 / (exp(E / (k_B T)) - 1)`

2. 洛伦兹线型：
`L(omega; omega0, gamma) = (gamma/pi) / ((omega-omega0)^2 + gamma^2)`

3. 单声子（未展宽）谱：
`I_raw = A * ((n+1)L(omega,+Omega,Gamma) + nL(omega,-Omega,Gamma)) + I_elastic + bg`

4. 色散关系：
`Omega(Q) = sqrt((cQ)^2 + gap^2)`

5. 阻尼关系：
`Gamma(Q) = g0 + g2 Q^2`

6. 仪器展宽：
`I_obs(Q,omega) = G_sigma ⊗ I_raw(Q,omega)`

## R06

算法流程（脚本级）：
1. 参数合法性检查（网格范围、温度、分辨率、真值参数）。
2. 生成 `Q` 与 `omega` 网格。
3. 用真值参数逐 `Q` 生成理论谱并卷积分辨率。
4. 叠加泊松噪声得到“实验态”谱图。
5. 对每个 `Q` 切片调用 `curve_fit` 拟合 5 参数模型。
6. 汇总拟合表并计算每切片 `R^2` 与 `MAE`。
7. 对 `Omega_fit^2` 与 `Q^2` 做线性回归，得到 `c` 与 `gap`。
8. 用 `torch` 在全图上全局优化参数并输出 MSE。
9. 执行质量门槛断言，确保 MVP 闭环有效。

## R07

复杂度估计（`Nq=n_q`, `Nw=n_omega`, `I_fit=切片拟合迭代数`, `E=torch_epochs`）：
- 谱图生成：`O(Nq * Nw)`；
- 逐切片拟合：约 `O(Nq * I_fit * Nw)`；
- 色散线性回归：`O(Nq)`；
- Torch 全局优化：`O(E * Nq_ds * Nw_ds)`。

默认参数（`24 x 351` 网格）下，`uv run python demo.py` 在桌面 CPU 环境可在数秒内完成。

## R08

数值稳定措施：
- 玻色因子指数输入做 `clip`，避免上溢；
- 洛伦兹宽度下限限制为 `1e-6` 防止除零；
- 逐切片拟合设置有界参数空间，避免非物理解；
- 泊松计数前对期望计数裁剪到正数；
- Torch 参数通过 `softplus` 保证正值约束；
- Torch 损失在归一化谱图上计算，减少数值尺度不平衡。

## R09

适用场景：
- IXS 数据分析流程教学和方法学原型验证；
- 拟合器稳定性测试（不同噪声/分辨率下参数恢复）；
- 新手理解 `S(Q,omega)`、分辨率卷积与色散反演关系。

不适用场景：
- 真实束线数据的最终物理参数发布；
- 多分支耦合、强关联系统、强非高斯分辨率情形；
- 需要完整误差传播与系统误差预算的正式分析。

## R10

`demo.py` 质量门槛（内置断言）：
1. `Q` 切片成功拟合数至少占 75%。
2. 平均逐切片拟合质量 `mean_fit_r2 >= 0.90`。
3. 色散回归质量 `dispersion_r2 >= 0.94`。
4. 提取声子能量平均误差 `MAE <= 2.5 meV`。
5. Torch 反演得到的 `c` 与真值偏差不超过 `3 meV·A`。
6. Torch 全局拟合 `MSE <= 4e-3`。

## R11

默认参数（`IXSParams`）：
- 温度：`280 K`
- 动量网格：`Q in [0.7, 3.2] A^-1`，`n_q=24`
- 能量网格：`omega in [-35, 35] meV`，`n_omega=351`
- 分辨率高斯宽度：`sigma=1.15 meV`
- 真值色散：`c=11.0 meV·A`, `gap=4.8 meV`
- 真值阻尼：`g0=1.3 meV`, `g2=0.34 meV/A^2`
- 强度参数：`amp_scale=950`, `elastic_scale=150`, `background=1.15`
- Torch 优化：`epochs=420`, `lr=0.05`

## R12

本地实测（命令：`uv run python demo.py`）关键输出：

- `n_q_total = 24`
- `n_q_success = 24`
- `mean_fit_r2 = 0.989339`
- `dispersion_r2 = 0.999926`
- `omega_fit_mae_meV = 0.038973`
- `regressed_c_meV_A = 11.017928`（真值 `11.0`）
- `regressed_gap_meV = 4.693763`（真值 `4.8`）
- `torch_c_fit_meV_A = 11.106040`
- `torch_gap_fit_meV = 1.190282`
- `torch_g0_fit_meV = 0.871577`
- `torch_g2_fit_meV_A2 = 0.239166`
- `torch_final_mse = 0.003135`

逐 `Q` 拟合表头显示每个切片 `R^2` 都在较高区间（约 0.986-0.992）。

## R13

结果解读：
- 切片拟合和色散回归均接近真值，说明局部拟合链路稳定。
- `regressed c/gap` 与真值相近，表明 `Omega(Q)^2` 回归对合成数据恢复能力强。
- Torch 全图优化能较好恢复 `c`，但 `gap/g0/g2` 偏差较大，说明在当前简化模型与下采样条件下存在参数相关性（可辨识性不足）。
- 尽管如此，Torch 的 `MSE` 仍较低，说明其作为“全局一致性校验器”是可用的。

## R14

常见失败模式与修复：
- 失败：分辨率设置过大导致峰完全展宽，拟合不稳定。
  - 修复：减小 `resolution_sigma_meV` 或增大能量窗密度。
- 失败：噪声过强导致 `curve_fit` 失败。
  - 修复：提高信号幅度（`amp_scale_true`）或降低背景噪声。
- 失败：参数触边界。
  - 修复：放宽 `bounds`，并改进初值（`omega_guess/bg0`）。
- 失败：Torch 收敛到局部最优。
  - 修复：调整 `torch_lr`、`epochs`，或加入先验正则项。

## R15

工程实践建议：
- 把“逐切片拟合”和“全图拟合”同时保留，二者互相校验；
- 对每次模型改动固定随机种子，保证可回归比较；
- 报告至少同时输出 `R^2`、`MAE`、`MSE` 三类指标；
- 真数据接入时先替换 `simulate_ixs_map`，后处理逻辑可直接复用。

## R16

可扩展方向：
- 引入多声子分支或磁激发分支，做多峰联合拟合；
- 将经验 `A(Q)` 替换为更物理的散射截面表达；
- 引入贝叶斯后验估计，给出参数区间而非点估计；
- 加入分辨率非高斯尾、零点漂移、系统误差项；
- 对接实验文件格式（如 HDF5/NeXus）完成端到端分析。

## R17

本目录交付：
- `README.md`：R01-R18 完整填写；
- `demo.py`：可运行 MVP（`numpy + scipy + pandas + scikit-learn + torch`）；
- `meta.json`：与任务元数据保持一致。

运行方式：

```bash
cd Algorithms/物理-凝聚态物理-0484-非弹性X射线散射_(Inelastic_X-ray_Scattering)
uv run python demo.py
```

脚本无需交互输入。

## R18

`demo.py` 源码级算法流拆解（8 步，非黑盒）：
1. `IXSParams` + `check_params` 固定并校验物理/数值边界（温度、网格、分辨率、真值参数）。
2. `phonon_dispersion` 与 `phonon_damping` 显式定义 `Omega(Q)`、`Gamma(Q)`，`q_amplitude/q_elastic` 给出 `Q` 依赖强度包络。
3. `unresolved_ixs_spectrum` 按 Stokes/anti-Stokes 结构构造未展宽单切片谱，并显式计算玻色因子与洛伦兹峰。
4. `resolved_ixs_spectrum` 对未展宽谱做 `gaussian_filter1d` 卷积，得到带仪器分辨率的理论谱。
5. `simulate_ixs_map` 在整张 `Q-omega` 网格上逐切片生成期望计数并施加泊松采样，形成观测谱图。
6. `fit_single_q_spectrum` 对每个 `Q` 切片调用 `curve_fit`（5 个自由参数）执行有界非线性最小二乘，输出 `Omega_fit/Gamma_fit/R2/MAE`。
7. `regress_dispersion` 用 `LinearRegression` 对 `Omega_fit^2` 与 `Q^2` 回归，计算 `c`、`gap` 与回归质量。
8. `torch_global_refine` 把同一物理模型写成可微计算图：`softplus` 施加约束、`conv1d` 实现分辨率卷积、`Adam` 最小化全图 MSE，最后 `main` 汇总指标并执行质量断言。

第三方库只承担数值优化与线性代数角色；IXS 的物理模型、切片拟合流程、色散回归与全图反演均在源码中逐步展开。
