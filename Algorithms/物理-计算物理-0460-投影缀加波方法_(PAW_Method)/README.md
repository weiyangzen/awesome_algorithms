# 投影缀加波方法 (PAW Method)

- UID: `PHYS-0439`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `460`
- 目标目录: `Algorithms/物理-计算物理-0460-投影缀加波方法_(PAW_Method)`

## R01

投影缀加波方法（PAW, Projector Augmented-Wave）是平面波第一性原理计算中的核心思想之一，用一个平滑的赝波函数 `|psi_tilde>` 配合局域投影与增广项，重建近似全电子波函数 `|psi>`。

本条目提供一个可运行的 1D 最小 MVP，聚焦 PAW 变换本身，而不是完整 DFT 自洽循环。核心目标是展示：

1. 如何构造 `|phi_i>、|phi_tilde_i>、|p_i>` 三元组。
2. 如何由投影系数 `c_i=<p_i|psi_tilde>` 做增广重建。
3. 如何用定量指标验证重建优于原始赝波函数。

## R02

MVP 范围（刻意收敛）：

1. 空间维度固定为 1D，采用有限差分网格与 Dirichlet 边界。
2. 构造两种势能：
   - 平滑赝势 `V_pseudo`；
   - 在核附近更“硬”的全电子参考势 `V_all_electron`。
3. 用 `scipy.linalg.eigh_tridiagonal` 求低能本征态，作为 `phi_tilde_i` 与 `phi_i` 的来源。
4. 选取两个偶宇称分波（索引 `0` 和 `2`）构造 PAW 局域子空间。
5. 输出 `pseudo / paw / paw_tuned` 三组结果并比较误差、重叠和能量。

## R03

本实现使用的 PAW 公式：

1. **PAW 线性变换**
   `|psi> = |psi_tilde> + sum_i (|phi_i> - |phi_tilde_i>) <p_i|psi_tilde>`

2. **投影器对偶条件**
   `<p_i|phi_tilde_j> = delta_ij`

3. **离散内积（网格积分）**
   `<f|g> ~= sum_k f(x_k) g(x_k) dx`

4. **重建误差指标**
   - 全局 RMSE
   - 核区 RMSE（`|x| <= r_aug`）
   - 与全电子参考态重叠 `int psi_ref * psi dx`

5. **可选系数微调（Torch）**
   在固定 `Delta phi_i = phi_i - phi_tilde_i` 下，仅优化 `c_i` 以最小化核区加权 MSE。

## R04

物理直觉：

1. 赝波函数在核区被“抹平”，计算更稳定，但丢失尖峰细节。
2. 全电子波函数在核区变化更快，数值上更难直接用平面波/粗网格表示。
3. PAW 的做法不是全局替换，而是在局域增广球（这里是 1D 区域）内补回细节：
   - 外部保持赝波函数；
   - 内部用 `Delta phi` 修正。
4. 投影器把“赝态的局域特征”压缩成少量系数，再驱动增广。

## R05

正确性关键点（对应 `demo.py`）：

1. `build_projectors` 显式构造对偶关系，并计算 `dual_overlap=<p_i|phi_tilde_j>` 验证接近单位阵。
2. `reconstruct_paw` 严格按 PAW 公式实现，不依赖黑盒重建器。
3. 所有波函数都做 `L2` 归一化，避免幅值漂移造成虚假改进。
4. `run_checks` 至少验证：
   - `rmse_core(paw) < rmse_core(pseudo)`；
   - `overlap(paw) > overlap(pseudo)`；
   - 投影器对偶误差足够小。
5. `paw_tuned` 仅调系数，不改基函数，便于区分“基函数能力”与“系数估计”两类误差来源。

## R06

复杂度（`N` 为网格点数，`m` 为分波数，本例 `m=2`）：

1. 三对角本征求解（取前 `k` 个低能态）主成本约 `O(Nk)` 到 `O(Nk + k^2)`，远低于稠密 `O(N^3)`。
2. 投影器构造涉及 `m x m` 线性代数，成本 `O(m^3)`，本例几乎常数。
3. 每次 PAW 重建为 `O(mN)`。
4. Torch 系数微调每步 `O(mN)`，总计 `O(steps * mN)`。

在默认参数（`N=801, m=2, steps=400`）下可快速运行。

## R07

标准执行流程：

1. 建立 1D 网格并定义 `V_pseudo` 与 `V_all_electron`。
2. 求解两种势下的低能态集合。
3. 选定分波索引 `partial_indices=[0,2]` 形成 `phi_tilde_i`。
4. 在增广半径内注入 `phi_i-phi_tilde_i`，外部保持一致。
5. 构造投影器并计算系数 `c_i=<p_i|psi_tilde>`。
6. 重建 `psi_paw`，再用 Torch 微调得到 `psi_paw_tuned`。
7. 汇总指标表并执行断言检查。

## R08

`demo.py` 的 MVP 设计选择：

1. 不引入完整 DFT 软件栈，只保留 PAW 变换必要路径。
2. 依赖均为基础科学计算库：
   - `numpy`：网格、内积、矩阵运算；
   - `scipy`：三对角本征求解；
   - `pandas`：指标表输出；
   - `scikit-learn`：RMSE 计算；
   - `torch`：自动微分调参。
3. 脚本无交互输入，直接 `uv run python demo.py` 可复现。

## R09

核心函数接口：

- `solve_lowest_states(potential, dx, num_states) -> (eigvals, states)`
- `build_projectors(phi_tilde, cutoff, dx) -> (projectors, dual_overlap)`
- `projector_coefficients(projectors, psi_tilde, dx) -> coeffs`
- `reconstruct_paw(psi_tilde, coeffs, delta_phi, dx) -> psi`
- `refine_coefficients_torch(coeffs_init, psi_tilde, delta_phi, psi_target, core_mask, ...) -> coeffs_tuned`
- `build_metrics_table(...) -> pandas.DataFrame`
- `run_paw_mvp() -> PawResult`
- `run_checks(result) -> None`

## R10

测试策略：

1. **代数一致性**：`dual_overlap` 近似单位阵。
2. **物理改进性**：核区 RMSE 必须下降。
3. **波函数相干性**：与全电子参考态重叠必须提升。
4. **数值稳定性**：指标表中全部数值有限（非 NaN/Inf）。
5. **调优稳定性**：Torch 微调不能明显破坏 PAW 基线结果（容差约束）。

## R11

边界条件与异常处理：

1. 非法网格或势能维度错误时抛 `ValueError`。
2. 归一化范数非正或非有限时抛 `ValueError`。
3. 投影器重叠矩阵病态（条件数过大）时抛 `ValueError`。
4. 自动断言失败（例如 PAW 未改进）会抛 `AssertionError`，并阻止误导性结果通过。

## R12

运行方式：

```bash
cd Algorithms/物理-计算物理-0460-投影缀加波方法_(PAW_Method)
uv run python demo.py
```

运行后会打印：

1. 能级信息（pseudo 与 all-electron）。
2. PAW 系数与微调系数。
3. 对偶矩阵 `<p_i|phi_tilde_j>`。
4. 三模型对比表。
5. `All checks passed.` 作为验收信号。

## R13

输出表字段说明：

- `model`：`pseudo` / `paw` / `paw_tuned`
- `rmse_global`：全区间 RMSE（相对全电子参考）
- `rmse_core`：增广核心区 RMSE（`|x|<=r_aug`）
- `overlap_with_ae`：与参考态重叠积分，越接近 1 越好
- `energy_on_ae_hamiltonian`：在全电子哈密顿量下的期望能量（用于辅助诊断）

## R14

最小验收标准：

1. `README.md` 与 `demo.py` 无占位符残留。
2. `uv run python demo.py` 无需输入，能一次跑通。
3. 终端出现 `All checks passed.`。
4. 指标中 `paw` 相比 `pseudo` 至少在 `rmse_core` 和 `overlap_with_ae` 两项上改进。

## R15

关键参数与调节建议：

1. `augmentation_radius`：增广区域大小，过小会补偿不足，过大可能引入不必要耦合。
2. `partial_indices`：分波基的选择直接影响重建质量，本例使用偶宇称 `0,2`。
3. 势参数（如高斯核深度与宽度）控制 pseudo/AE 差异强度。
4. Torch 微调参数：
   - `steps`：迭代步数；
   - `lr`：学习率；
   - 损失中 core/global/regularization 权重。

## R16

与相关方法的关系：

1. 与纯赝势法相比：PAW 通过增广恢复核区细节，通常精度更高。
2. 与超软赝势（USPP）思路接近：都使用投影与增广；PAW可视作更系统的全电子重建框架。
3. 与 LAPW/APW 相比：PAW在平面波效率与全电子信息之间取得折中。
4. 本条目只覆盖“单次重建”环节，不含完整 Kohn-Sham 自洽迭代和交换关联泛函细节。

## R17

可扩展方向：

1. 从 1D 升级到 3D 晶体周期边界。
2. 引入多原子、多通道（不同角动量）投影器。
3. 加入自洽循环（密度更新、泊松求解、SCF 收敛判据）。
4. 用真实原子数据文件替代 toy 势，做更接近生产环境的验证。
5. 增加可视化（波函数、`Delta phi`、误差分布）与单元测试。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `run_paw_mvp` 建立网格与两套势能，调用 `solve_lowest_states` 分别求 pseudo/AE 低能本征态。
2. `solve_lowest_states` 将二阶导离散成三对角哈密顿量，调用 `scipy.linalg.eigh_tridiagonal(..., lapack_driver=\"stemr\")` 求指定低能本征对。
3. 在 SciPy/LAPACK 路径中，`stemr` 使用 MRRR（Multiple Relatively Robust Representations）策略：构造鲁棒表示、分离特征簇、逐簇提取特征值/向量，避免把问题当成稠密 `eig` 黑盒。
4. 代码选取 `partial_indices=[0,2]` 构造 `phi_tilde_i`，并在增广区注入 `phi_i-phi_tilde_i` 得到 `phi_i`，区外两者保持一致。
5. `build_projectors` 先形成局域基 `basis_i`，再解线性系统得到投影器，使 `<p_i|phi_tilde_j>≈delta_ij`；打印 `dual_overlap` 做直接验算。
6. `projector_coefficients` 计算 `c_i=<p_i|psi_tilde>`，`reconstruct_paw` 按 `psi=psi_tilde+sum_i c_i (phi_i-phi_tilde_i)` 重建 `psi_paw`。
7. `refine_coefficients_torch` 将 `c_i` 设为可训练张量，利用 autograd 对 core-weighted MSE（加全局项与正则）做 Adam 优化，得到 `psi_paw_tuned`。
8. `build_metrics_table` 汇总 RMSE/重叠/能量，`run_checks` 断言改进成立，`main` 打印报告并输出 `All checks passed.`。
