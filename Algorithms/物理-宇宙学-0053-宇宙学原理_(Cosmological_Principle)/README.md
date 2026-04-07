# 宇宙学原理 (Cosmological Principle)

- UID: `PHYS-0053`
- 学科: `物理`
- 分类: `宇宙学`
- 源序号: `53`
- 目标目录: `Algorithms/物理-宇宙学-0053-宇宙学原理_(Cosmological_Principle)`

## R01

宇宙学原理（Cosmological Principle）声明：在足够大尺度上，宇宙在统计意义下是各向同性（isotropic）且均匀（homogeneous）的。

本条目给出一个最小可运行算法 MVP：用点过程样本（模拟“星系目录”）分别检验这两个条件，并输出可审计的统计量与通过/拒绝结论。

## R02

MVP 采用观测者中心球体体积内的三维点目录：
- 理想目录：满足“体积内均匀 + 角向均匀”。
- 违背目录：人为注入角向偶极偏置与径向非均匀分布。

核心目标不是做精细宇宙学参数估计，而是演示“宇宙学原理可被算法化检验”的完整流程：采样 -> 统计量 -> 显著性检验 -> 结论。

## R03

统计化表达：
- 各向同性：方向单位向量 \(\hat n\) 的分布在球面上应均匀，等价于 \(\mu=\cos\theta\) 与 \(\phi\) 的联合分布在等面积分箱中近似常数。
- 均匀性：体密度在大尺度上接近常数。若半径上限为 \(R\)，则
  \[
  u=\left(\frac{r}{R}\right)^3
  \]
  在理想均匀体分布下应服从 \([0,1]\) 上均匀分布。

因此代码分别构建角向检验和径向（体积坐标）检验。

## R04

本实现使用两类显式统计量：

1. 角向卡方统计量（isotropy）  
将 \(\mu\in[-1,1]\)、\(\phi\in[0,2\pi)\) 离散为 `mu_bins × phi_bins`：
\[
\chi^2_{\text{iso}}=\sum_{i,j}\frac{(N_{ij}-E)^2}{E},\quad
E=\frac{N}{\text{mu\_bins}\cdot\text{phi\_bins}}
\]
并用 `scipy.stats.chi2.sf` 计算 \(p_{\text{iso}}\)。

2. 径向卡方统计量（homogeneity）  
对 \(u=(r/R)^3\) 做 `radial_bins` 等宽分箱：
\[
\chi^2_{\text{homo}}=\sum_k\frac{(N_k-E_r)^2}{E_r},\quad
E_r=\frac{N}{\text{radial\_bins}}
\]
得到 \(p_{\text{homo}}\)。

另外输出偶极振幅 \(\|\langle \hat n\rangle\|\) 与径向分箱变异系数 `CV` 作为辅助诊断。

## R05

前提与假设：
- 使用理想化全空间选择函数（无遮挡、无观测极限、无红移误差）。
- 样本点彼此独立抽样，不显式模拟真实大尺度结构相关函数。
- 统计检验阈值使用固定显著性水平 `alpha=0.01`。
- 该 MVP 旨在展示“原理检验路径”，不替代真实巡天分析流程。

## R06

`demo.py` 输入输出约定：
- 输入：脚本内置配置（样本数、球半径、分箱数、随机种子、显著性阈值）。
- 输出：
1. 两个目录（参考目录与违背目录）的统计表；
2. `p_iso`、`p_homo`、偶极振幅、径向 `CV`；
3. 阈值检查项及最终 `Validation: PASS/FAIL`。

脚本无交互输入，可直接运行复现。

## R07

算法主流程（高层）：
1. 构造参考目录：体积均匀（`r=R*u^(1/3)`）+ 角向均匀（`\mu` 均匀、`\phi` 均匀）。
2. 构造违背目录：径向使用 \(F(r)=(r/R)^\eta\)（\(\eta\neq 3\)）+ 角向偶极偏置 `p(mu) ∝ 1 + beta*mu`。
3. 对每个目录计算球坐标量 `r, mu, phi`。
4. 计算角向卡方统计量及 `p_iso`。
5. 计算体积坐标 `u=(r/R)^3` 的径向卡方统计量及 `p_homo`。
6. 计算辅助指标：偶极振幅、径向 shell 变异系数。
7. 判定是否接受宇宙学原理：`p_iso > alpha and p_homo > alpha`。
8. 输出表格与通过/失败结论。

## R08

复杂度（`N` 为样本点数，`B = mu_bins*phi_bins + radial_bins`）：
- 采样生成：`O(N)`。
- 球坐标与单位向量计算：`O(N)`。
- 统计直方图计算：`O(N + B)`。
- 卡方与辅助指标：`O(B)`。

总体时间复杂度 `O(N)`，空间复杂度 `O(N)`；适合教学与快速原型验证。

## R09

数值与统计注意事项：
- 用 `mu=cos(theta)` 分箱可保持等面积；直接按 `theta` 等宽会引入面积偏差。
- 对向量归一化使用 `clip(norm, EPS, None)` 防止除零。
- 卡方检验要求期望频数不宜过小；当前 `N=12000` 与分箱设置可保证每箱期望值充足。
- 随机种子固定，保证结果可复现实验。

## R10

MVP 技术栈：
- Python 3
- `numpy`：采样、向量化计算、直方图
- `scipy`：卡方分布生存函数 `chi2.sf`
- `pandas`：结果表格化输出
- `dataclasses`：配置与结果结构化

未依赖黑箱宇宙学分析框架；统计流程在源码中逐步展开。

## R11

运行方式：

```bash
cd Algorithms/物理-宇宙学-0053-宇宙学原理_(Cosmological_Principle)
uv run python demo.py
```

脚本不读取命令行参数，也不要求交互输入。

## R12

输出字段说明：
- `catalog`: 目录名称（参考或违背）。
- `N`: 样本点数量。
- `chi2_iso`, `p_iso`: 各向同性卡方统计量与 p 值。
- `dipole_amp`: 方向单位向量均值的模，越小越接近各向同性。
- `chi2_homo`, `p_homo`: 均匀性卡方统计量与 p 值。
- `shell_cv`: 径向等体积分箱计数的变异系数。
- `CP_accept`: 是否接受宇宙学原理（本 MVP 判据）。

## R13

正确性验证（脚本内置检查）：
1. 参考目录应被接受（`CP_accept=True`）。
2. 违背目录应被拒绝（`CP_accept=False`）。
3. 参考目录偶极振幅应足够小（阈值 `< 0.03`）。
4. 违背目录偶极振幅应显著偏大（阈值 `> 0.08`）。

以上检查同时通过时输出 `Validation: PASS`，否则退出非零状态。

## R14

当前实现局限：
- 未建模真实观测选择函数（遮挡、亮度极限、红移误差、体积完备性）。
- 未引入两点相关函数或功率谱等更高阶结构统计。
- 仅使用球对称观测窗口；未涉及真实巡天掩膜。
- 结果用于方法演示，不宜直接用于科学结论。

## R15

可扩展方向：
- 加入 survey mask 与 selection function，做加权统计。
- 用 `scipy.spatial` 或 FFT 方法估计两点相关函数，联合检验大尺度均匀性。
- 引入 `pandas`/`sklearn` 管线做分红移层统计与系统误差回归。
- 用 `torch` 实现可微统计量，便于端到端误差传播与仿真反演。

## R16

应用场景：
- 宇宙学原理教学演示与统计检验入门。
- 巡天数据预分析中的“快速 sanity check”。
- 仿真目录生成器（mock catalog）质量控制基线。
- 方法学研究中对“各向同性/均匀性”假设的可复现实验模板。

## R17

方案对比：
- 直接调用高层黑箱宇宙学库可以更快得到结论，但难以审计算法细节。
- 只做可视化（例如天球散点图）直观但缺少显著性量化。
- 本方案以最小代码实现明确统计检验链路，兼顾可解释性、可复现性与可扩展性。

因此它适合作为“从理论原理到可运行算法”的过渡实现。

## R18

`demo.py` 源码级流程拆解（8 步）：
1. `main` 创建 `PrincipleConfig`，分别生成参考目录与违背目录。
2. `sample_isotropic_homogeneous_catalog` 使用 `r=R*u^(1/3)`、`\mu`/`\phi` 均匀采样构造理想点云。
3. `sample_anisotropic_inhomogeneous_catalog` 使用 `eta!=3` 的径向 CDF 与 `sample_mu_with_dipole` 生成偶极偏置方向分布。
4. `compute_catalog_diagnostics` 把点云转为 `r, mu, phi`，并通过 `histogram2d` 构建角向分箱计数。
5. 计算角向卡方 `chi2_isotropy` 与 `p_isotropy`（`chi2.sf`），并给出偶极振幅 `||mean(unit)||`。
6. 计算 `u=(r/R)^3` 的径向分箱计数，得到 `chi2_homogeneity`、`p_homogeneity` 与 `shell_cv`。
7. 基于判据 `p_iso > alpha and p_homo > alpha` 生成 `accept_cosmological_principle` 布尔结论。
8. `diagnostics_to_frame` 汇总为 `pandas.DataFrame`，`main` 打印检查项并输出 `Validation: PASS/FAIL`。
