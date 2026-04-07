# 引力透镜 (Gravitational Lensing)

- UID: `PHYS-0362`
- 学科: `物理`
- 分类: `广义相对论`
- 源序号: `381`
- 目标目录: `Algorithms/物理-广义相对论-0381-引力透镜_(Gravitational_Lensing)`

## R01

引力透镜是广义相对论中“时空弯曲导致光线偏折”的直接观测效应。  
在弱场、薄透镜近似下，可将复杂几何传播压缩为一个二维映射：
`β = θ - α(θ)`，其中 `β` 是源角位置，`θ` 是像角位置，`α` 是偏折角。

本目录的 MVP 聚焦“点质量透镜（Point Lens）”：
- 用平直 LCDM 背景计算角径距离；
- 用点透镜解析式计算爱因斯坦角；
- 求解双像位置与放大率；
- 用一个小型 2D 光线映射验证“可形成放大与多像”。

## R02

求解目标（本实现）：
- 输入：
  - 宇宙学参数 `H0, Ωm, ΩΛ`（平直宇宙）；
  - 透镜质量 `M`；
  - 透镜与源红移 `z_l, z_s`（要求 `z_s > z_l`）；
  - 一组源偏移 `β`（角秒）用于演示。
- 输出：
  - `D_l, D_s, D_ls`（角径距离）；
  - 爱因斯坦角 `θ_E`；
  - 每个 `β` 对应的 `θ_+, θ_-` 与 `μ_+, μ_-`、总放大率；
  - 透镜方程残差与 2D 网格放大率估计。

`demo.py` 固定参数自动运行，无需交互输入。

## R03

核心数学关系：

1. 共动距离（平直 LCDM）：
`D_c(z) = (c/H0) * ∫[0→z] dz' / sqrt(Ωm(1+z')^3 + ΩΛ)`。
2. 角径距离：
`D_A(0,z) = D_c(z)/(1+z)`，  
`D_A(z_l,z_s) = (D_c(z_s)-D_c(z_l))/(1+z_s)`。
3. 点质量透镜爱因斯坦角：
`θ_E = sqrt((4GM/c^2) * (D_ls/(D_l D_s)))`。
4. 标量透镜方程（轴对称点透镜）：
`β = θ - θ_E^2/θ`。
5. 解析双像解：
`θ_± = 0.5 * (β ± sqrt(β^2 + 4θ_E^2))`。
6. 点透镜放大率（`u = β/θ_E`）：
`μ_± = 1/2 ± (u^2+2)/(2u*sqrt(u^2+4))`，  
`μ_total = |μ_+| + |μ_-|`。

## R04

算法流程（高层）：
1. 设定宇宙学参数、透镜质量、红移与源偏移样本。
2. 数值积分计算 `D_c(z_l), D_c(z_s)`，进而得到 `D_l, D_s, D_ls`。
3. 由距离与质量计算 `θ_E`。
4. 对每个 `β`：
  - 用解析式求 `θ_+, θ_-`；
  - 计算 `μ_+, μ_-` 与 `μ_total`；
  - 反代透镜方程计算残差。
5. 构造 2D 像平面网格，用向量式透镜映射得到 `β(θ)`。
6. 对高斯源比较“未透镜/透镜后”总通量，得到网格放大率估计。
7. 打印结果并做最小一致性断言（残差阈值、放大率应大于 1）。

## R05

核心数据结构：
- `Cosmology`（`@dataclass`）：`h0_km_s_mpc, omega_m, omega_lambda`。
- 标量距离函数：
  - `comoving_distance_m`；
  - `angular_diameter_distance_m`；
  - `angular_diameter_distance_between_m`。
- 透镜核心函数：
  - `einstein_radius_rad`；
  - `image_positions_point_lens`；
  - `magnifications_point_lens`；
  - `lens_equation_residual`。
- 结果表：
  - `simulate_point_lens_table` 返回 `pandas.DataFrame`，每行对应一个 `β` 样本。

## R06

正确性要点：
- 距离计算明确使用标准平直 LCDM 公式，非经验拟合。
- 点透镜像位置使用闭式解，避免数值根查找误收敛。
- 每个像都反代 `β = θ - θ_E^2/θ` 计算残差，直接检验方程正确性。
- 放大率使用已知解析表达式，且输出双像奇偶性（`μ_-` 为负宇称）。
- 2D 光线映射遵循向量透镜方程，验证“同一公式在二维网格上可工作”。

## R07

复杂度分析：
- 设红移积分自适应采样成本为 `Q`，`β` 样本数为 `N`，2D 网格为 `G x G`。
- 距离部分：`O(Q)`（对 `z_l, z_s` 各做一次积分，常数较小）。
- 表格部分：每个 `β` 是闭式计算，`O(N)`。
- 2D 光线映射：网格向量化计算 `O(G^2)`，是主要成本。
- 空间复杂度：
  - 结果表 `O(N)`；
  - 网格数组 `O(G^2)`。

## R08

边界与异常处理：
- 非法输入（负质量、非正距离、非有限数）会抛出 `ValueError`。
- 要求 `z_s > z_l`，否则拒绝计算 `D_ls`。
- `β=0` 时放大率理论发散，代码返回 `inf/-inf/inf`（符合点源极限）。
- 2D 映射在 `|θ|→0` 处对 `r^2` 做下限截断，避免除零。
- 对最终结果做运行时断言：
  - 透镜方程残差应小于阈值；
  - 网格估计总放大率应大于 1。

## R09

MVP 取舍说明：
- 仅实现“点质量 + 薄透镜 + 弱场 + 小角度”主干，不做全 GR 光线追踪。
- 不引入外部黑箱 lensing 软件包，核心公式都在 `demo.py` 显式给出。
- 用 `scipy.integrate.quad` 仅做一维宇宙学积分；透镜求解逻辑完全自写。
- 2D 部分只做数值演示，不输出图像文件，保持脚本轻量可快速验证。

## R10

`demo.py` 函数职责：
- `_validate_positive`：参数合法性检查。
- `comoving_distance_m`：`D_c(z)` 数值积分。
- `angular_diameter_distance_m / angular_diameter_distance_between_m`：距离转换。
- `einstein_radius_rad`：根据质量和距离求 `θ_E`。
- `image_positions_point_lens`：解析双像位置。
- `magnifications_point_lens`：双像及总放大率。
- `lens_equation_residual`：方程反代残差。
- `simulate_point_lens_table`：批量生成 `β` 样本结果表。
- `ray_map_gaussian_source`：二维网格源映射与通量比较。
- `main`：组织参数、打印结果、执行断言。

## R11

运行方式：

```bash
cd Algorithms/物理-广义相对论-0381-引力透镜_(Gravitational_Lensing)
uv run python demo.py
```

脚本不会读取交互输入，也不依赖外部数据文件。

## R12

输出字段说明：
- `D_l, D_s, D_ls`：角径距离（单位 Mpc）。
- `theta_E`：爱因斯坦角（角秒）。
- 表格列：
  - `beta_arcsec`：源偏移；
  - `theta_plus_arcsec / theta_minus_arcsec`：双像角位置；
  - `mu_plus / mu_minus`：双像放大率（`mu_minus` 常为负宇称）；
  - `mu_total_abs`：总放大率；
  - `max_abs_residual`：双像反代透镜方程的最大残差。
- `Estimated total magnification`：二维高斯源网格上的总通量比值。

## R13

最小测试集（内置）：
1. 固定透镜配置：
  - `M = 1e12 Msun`，`z_l = 0.5`，`z_s = 2.0`。
2. 源偏移扫描：
  - `β ∈ [0.05, 1.20] arcsec` 共 10 个样本。
3. 一致性断言：
  - 表格中 `max_abs_residual` 最大值小于 `1e-12`；
  - 二维网格估计总放大率应为有限且 `> 1`。

可扩展测试：
- 改变 `M` 检查 `θ_E ∝ sqrt(M)` 标度关系；
- 改变 `β` 检查 `β` 越大时总放大率趋近 1。

## R14

关键参数与建议：
- `h0_km_s_mpc`：哈勃常数；改变它会整体缩放距离并影响 `θ_E`。
- `omega_m, omega_lambda`：背景宇宙学；影响 `D_l, D_s, D_ls`。
- `lens_mass_msun`：透镜质量；`θ_E` 对它按平方根增长。
- `beta_samples`：源偏移采样范围；越接近 0，放大率越高。
- 2D 映射参数：
  - `grid_points`：网格分辨率（更高更准，但更慢）；
  - `grid_half_size_arcsec`：视场范围；
  - `source_sigma_arcsec`：源尺寸（越小越接近点源行为）。

## R15

方法对比：
- 对比“仅几何光学直线传播”：
  - 本算法包含 GR 偏折，可产生双像与放大。
- 对比“全数值测地线追踪”：
  - 本算法计算快、透明，适合教学和快速估算；
  - 但忽略强场高阶效应、复杂透镜质量分布、波动光学衍射等。
- 对比“现成 lensing 黑箱库”：
  - 本实现可直接审计每个公式与中间量；
  - 代价是功能覆盖范围更窄。

## R16

典型应用场景：
- 强引力透镜入门教学与公式验证。
- 快速估算给定质量与红移下的像分离尺度（`θ_E` 量级）。
- 作为更复杂透镜模型（SIS/NFW/多透镜面）前的基线单元测试。
- 在观测提案或模拟前做数量级 sanity check。

## R17

可扩展方向：
- 质量模型扩展：点质量 -> SIS/NFW/椭圆势。
- 数据维度扩展：加入时间延迟、费马势与哈勃常数反演示例。
- 数值扩展：2D 网格输出成图（临界线/象差曲线）并做参数扫描。
- 工程扩展：与 PyTorch 自动微分结合，做透镜参数拟合。
- 物理扩展：加入外剪切（external shear）与多透镜平面。

## R18

`demo.py` 源码级流程（8 步）：
1. `main` 固定宇宙学参数、透镜质量和红移，构造演示所需配置。  
2. 调用 `comoving_distance_m` 与角径距离函数得到 `D_l, D_s, D_ls`。  
3. `einstein_radius_rad` 根据 `M, D_l, D_s, D_ls` 计算 `θ_E` 并换算为角秒。  
4. `simulate_point_lens_table` 遍历 `β` 样本，对每个样本调用：
   `image_positions_point_lens` 求 `θ_+, θ_-`，再用 `magnifications_point_lens` 求 `μ_+, μ_-`。  
5. 对每个像调用 `lens_equation_residual` 反代透镜方程，记录最大绝对残差到 DataFrame。  
6. `ray_map_gaussian_source` 在二维像平面网格执行向量透镜映射 `β(θ)`，比较透镜前后高斯源总通量。  
7. `main` 打印距离、爱因斯坦角、样本表格以及 2D 网格总放大率估计。  
8. 最后执行两条断言：残差需小于阈值、网格放大率需大于 1；通过则脚本正常结束。  
