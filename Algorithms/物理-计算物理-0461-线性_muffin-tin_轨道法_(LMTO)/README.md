# 线性 muffin-tin 轨道法 (LMTO)

- UID: `PHYS-0440`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `461`
- 目标目录: `Algorithms/物理-计算物理-0461-线性_muffin-tin_轨道法_(LMTO)`

## R01

LMTO（Linear Muffin-Tin Orbital）是全电子结构方法中经典的“线性化局域轨道”路线。核心思想是：

1. 在每个原子 muffin-tin 球内，解径向方程得到 `phi(r, E_nu)`；
2. 计算能量导数 `phi_dot(r, E_nu)`；
3. 用线性组合 `phi + (E-E_nu)phi_dot` 近似不同能量附近的真实轨道；
4. 把基函数投到有限维空间，解广义本征值问题得到能级。

本条目给出一个可运行的 1D 教学版 MVP，突出 LMTO 计算链路，而非完整 DFT 自洽流程。

## R02

MVP 的问题设定（刻意简化但保持结构）：

1. 构建一个 1D 原子链，多个高斯型吸引势井模拟原子势；
2. 每个原子周围半径 `R_mt` 作为 muffin-tin 区；
3. 在单原子球内解径向方程并做线性化；
4. 组装两套基组并对比：
   - `MT_only`：仅 `phi`；
   - `LMTO`：`[phi, phi_dot]` 扩展基组；
5. 用高分辨率有限差分本征值作为参考答案。

## R03

实现采用的关键公式：

1. 径向方程（1D 教学类比）
   `-1/2 u''(r) + V(r)u(r) = E u(r)`
2. 线性化导数
   `phi_dot(r,E_nu) ~= [phi(r,E_nu+ΔE)-phi(r,E_nu-ΔE)]/(2ΔE)`
3. 广义本征值问题
   `Hc = ESc`
4. 矩阵元定义
   - `S_ij = ∫ b_i b_j dx`
   - `H_ij = ∫ [0.5 b_i' b_j' + V b_i b_j] dx`
5. 误差指标
   - 谱误差：`RMSE(E_model, E_ref)`
   - 波函数误差：核心区与全局 RMSE。

## R04

物理直觉：

1. 只用 `phi` 的基组通常“过硬”，对能量偏离线性化点的状态描述不足；
2. 加入 `phi_dot` 后，基组在能量方向多了一个一阶自由度；
3. 这种一阶补偿就是 LMTO 的核心价值：在较小基组下提升谱精度；
4. 在示例里，这会反映为 `LMTO` 对参考谱的 RMSE 下降。

## R05

`demo.py` 的最小实现目标：

1. 不调用任何 DFT 黑盒包；
2. 显式展示从径向 ODE 到 `H,S` 的构造过程；
3. 使用 `numpy/scipy/pandas/scikit-learn/torch` 五类基础工具；
4. 运行后自动打印能级对比表和质量指标表；
5. 内置断言，确保 LMTO 相比 `MT_only` 有实质改进。

## R06

复杂度分析（`N_g` 为实空间网格点，`M` 为原子数）：

1. 径向 ODE 解：`O(N_r)`；
2. 基函数构造：`O(M * N_g)`；
3. 矩阵组装：
   - `MT_only`：`O(M^2 * N_g)`；
   - `LMTO`：`O((2M)^2 * N_g)`；
4. 广义本征求解：
   - `MT_only`：`O(M^3)`；
   - `LMTO`：`O((2M)^3)`；
5. Torch 标量优化：每步一个小规模本征值问题，成本可控。

## R07

数值稳定性策略：

1. 配置检查：`R_mt < a/2`，防止 muffin-tin 球重叠；
2. 所有局域基函数做 `L2` 归一化；
3. 对重叠矩阵 `S` 检查最小特征值，避免奇异；
4. Torch 路径中对 `S` 添加小 `jitter` 再 Cholesky；
5. 最终以断言验证结果可信度。

## R08

代码结构总览：

1. `solve_linearized_radial_set`：得到 `u` 与 `u_dot`；
2. `build_lmto_basis`：把局域轨道放置到每个原子中心；
3. `assemble_hs_matrices`：计算 `H,S`；
4. `build_lmto_blocks` + `assemble_lmto_from_alpha`：构造 LMTO 分块矩阵；
5. `finite_difference_reference`：生成高分辨率参考谱；
6. `torch_optimize_alpha`：微调 `phi_dot` 缩放；
7. `run_lmto_mvp`：组织端到端流程；
8. `run_checks`：验收。

## R09

三组模型输出解释：

1. `MT_only`：只用 `phi`；
2. `LMTO_alpha1`：标准 LMTO 近似，`alpha=1`；
3. `LMTO_alpha_opt`：在 `alpha` 上做小幅可微调优。

输出表字段：

- `eig_rmse_vs_ref`：低能级对参考谱 RMSE；
- `wf_rmse_core`：核心区波函数 RMSE；
- `wf_rmse_global`：全域波函数 RMSE；
- `s_min_eig`、`s_condition`：重叠矩阵健康度。

## R10

运行方式：

```bash
cd Algorithms/物理-计算物理-0461-线性_muffin-tin_轨道法_(LMTO)
uv run python demo.py
```

脚本无交互输入，完成后会打印：

1. 前若干能级对比；
2. 质量指标表；
3. `All checks passed.`（通过信号）。

## R11

参数含义（`LMTOConfig`）：

1. 晶格与几何：`lattice_constant`, `n_atoms`, `muffin_tin_radius`；
2. 势能：`site_potential_depth`, `site_potential_sigma`；
3. 线性化：`linearization_energy`, `linearization_delta_e`；
4. 数值离散：`x_points`, `radial_points`, `reference_points`；
5. 拟合与优化：`n_reference_levels`, `torch_steps`, `torch_lr`。

## R12

验证逻辑（`run_checks`）：

1. 所有误差指标必须是有限值；
2. `LMTO_alpha1` 的谱 RMSE 必须优于 `MT_only`；
3. `LMTO_alpha1` 的核心区波函数 RMSE 必须优于 `MT_only`；
4. `LMTO_alpha_opt` 不得劣于 `LMTO_alpha1` 的谱 RMSE；
5. 三组模型的 `S` 都必须保持正定裕量。

## R13

为什么这个 MVP 对 LMTO 是“诚实”的：

1. 线性化不是口头描述，而是显式计算 `phi_dot`；
2. 不是直接调用高层电子结构软件，而是手工组装 `H,S`；
3. 明确提供非线性化基组（`MT_only`）作为对照；
4. 用独立数值参考（有限差分）衡量收益，避免自证循环。

## R14

当前局限：

1. 1D 教学模型，未包含真实 3D 角动量通道与球谐函数；
2. 未实现屏蔽常数与正交化体系的完整 LMTO 家族细节；
3. 不是自洽 Kohn-Sham 循环，只是固定势单次本征问题；
4. `alpha` 调整是教学辅助，不等价于完整 LMTO 参数化策略。

## R15

可扩展方向：

1. 扩展到 3D 晶体与 `k` 点采样；
2. 增加多 `l` 通道与自旋自由度；
3. 引入 SCF 密度迭代；
4. 替换 toy 势为真实原子势参数；
5. 与 LAPW/PAW 在同一基准上做误差-成本对比。

## R16

与相关方法对比（简述）：

1. 相比纯赝势平面波：LMTO 更偏局域轨道和全电子信息恢复；
2. 相比 LAPW：LMTO 在基函数组织上更紧凑，但参数化技巧要求高；
3. 相比紧束缚：LMTO 的矩阵元来自连续方程和线性化径向函数，不是纯经验参数。

## R17

最小验收清单：

1. `README.md` 无模板占位符；
2. `demo.py` 无模板占位符；
3. `uv run python demo.py` 可直接运行；
4. 输出包含三模型对比与 `All checks passed.`；
5. `meta.json` 与任务元数据一致。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `run_lmto_mvp` 调 `solve_linearized_radial_set`：在 `E_nu` 与 `E_nu±ΔE` 上三次解径向 ODE，中心差分得到 `phi_dot`。
2. `build_lmto_basis` 把每个原子中心的 `phi/phi_dot` 通过“球内数值解 + 球外指数尾”拼接成全局局域轨道，并归一化。
3. `assemble_hs_matrices` 直接按积分定义构造 `H,S`，得到 `MT_only` 广义本征问题；`scipy.linalg.eigh(H,S)` 求能级。
4. `build_lmto_blocks` 计算 `pp/pd/dd` 三类分块矩阵；`assemble_lmto_from_alpha` 组装 `LMTO` 的块结构矩阵。
5. `finite_difference_reference` 在更密网格上离散哈密顿量并求参考低能谱，形成外部对照基准。
6. `torch_optimize_alpha` 在源码中显式做广义本征值的可微变换：`S=LL^T`，构造 `L^{-1}HL^{-T}` 后 `torch.linalg.eigvalsh`，最小化参考谱 MSE 拟合 `alpha`。
7. 回到 NumPy/SciPy 路径，用最优 `alpha` 重建 LMTO 矩阵并求最终谱，计算 `eig_rmse_vs_ref` 与波函数核心区 RMSE（`sklearn.mean_squared_error`）。
8. `run_checks` 执行改进性与稳定性断言，`main` 打印能级表与指标表并输出 `All checks passed.`。
