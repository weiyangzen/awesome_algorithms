# 强透镜 (Strong Lensing)

- UID: `PHYS-0363`
- 学科: `物理`
- 分类: `广义相对论`
- 源序号: `382`
- 目标目录: `Algorithms/物理-广义相对论-0382-强透镜_(Strong_Lensing)`

## R01

强透镜是引力透镜中的“多像可分辨”区间：背景源与透镜几何关系满足临界条件时，像平面会出现多重像、显著放大与典型角分离。

本条目用一个可审计的最小模型实现强透镜流程：
- 透镜质量模型采用 SIS（奇异等温球）；
- 背景宇宙采用平直 LCDM 距离关系；
- 给出像数判定、像位置、放大率与强透镜截面；
- 用 Monte Carlo 检查“强透镜概率 = 面积比”的可重复结论。

## R02

问题定义（MVP）：
- 输入：
  - 宇宙学参数 `H0, Ωm, ΩΛ`；
  - 透镜速度弥散 `sigma_v`；
  - 红移 `z_l, z_s`（要求 `z_s > z_l`）；
  - 一组源角偏移 `beta`。
- 输出：
  - `D_l, D_s, D_ls`（角径距离）；
  - SIS 爱因斯坦角 `theta_E`；
  - 每个 `beta` 的像数、像位置、像分离与放大率；
  - 强透镜截面；
  - Monte Carlo 强透镜概率与解析面积比对照。

运行方式固定为 `uv run python demo.py`，不需要交互输入。

## R03

核心公式：

1. 平直 LCDM 共动距离
`D_c(z) = (c/H0) * ∫[0→z] dz' / sqrt(Ωm(1+z')^3 + ΩΛ)`。

2. 角径距离
`D_A(0,z) = D_c(z)/(1+z)`，  
`D_A(z_l,z_s) = (D_c(z_s)-D_c(z_l))/(1+z_s)`。

3. SIS 爱因斯坦角
`theta_E = 4π (sigma_v/c)^2 * (D_ls/D_s)`。

4. SIS 一维签名透镜方程
`beta = theta - theta_E * sign(theta)`。

5. 像位置（`beta >= 0`）
- 外像：`theta_+ = beta + theta_E`（总存在）；
- 内像：`theta_- = beta - theta_E`（仅当 `beta < theta_E`）。

6. 放大率
- `mu_+ = 1 + theta_E/beta`；
- `mu_- = 1 - theta_E/beta`（内像负宇称）；
- `mu_total = |mu_+| + |mu_-|`（若无内像则为 `|mu_+|`）。

7. 强透镜判据（SIS）
`beta < theta_E` 等价于“两像区”。

## R04

算法流程（高层）：
1. 设置宇宙学与 SIS 透镜参数。
2. 用数值积分计算 `D_l, D_s, D_ls`。
3. 由 `sigma_v, D_ls, D_s` 计算 `theta_E`。
4. 对每个样本 `beta`：
  - 求像位置 `theta_+, theta_-`；
  - 计算像数、像分离、放大率；
  - 反代透镜方程得到残差。
5. 计算强透镜源平面截面 `pi * (theta_E * D_s)^2`。
6. 在 `beta_max` 圆盘中均匀采样源位置，估计强透镜概率。
7. 对照解析概率 `(theta_E/beta_max)^2`，检查 Monte Carlo 与理论一致性。
8. 输出表格并执行最小断言。

## R05

代码结构：
- `Cosmology`：LCDM 参数容器。
- `SISLensConfig`：透镜演示参数容器。
- 距离函数：
  - `comoving_distance_m`；
  - `angular_diameter_distance_m`；
  - `angular_diameter_distance_between_m`。
- 透镜核心函数：
  - `einstein_radius_sis_rad`；
  - `sis_image_positions`；
  - `sis_magnifications`；
  - `sis_lens_equation_residual`。
- 统计函数：
  - `simulate_sis_table`；
  - `strong_lensing_cross_section_source_plane`；
  - `monte_carlo_strong_fraction`。
- `main`：组织计算、打印结果、触发断言。

## R06

正确性保障：
- 距离由标准平直 LCDM 一维积分得到，不使用经验近似常数替代。
- 像位置按 SIS 闭式表达计算，避免数值求根歧义。
- 每个像都做透镜方程反代，`max_abs_residual` 直接验证方程闭环。
- 强透镜判据由像数与 `beta < theta_E` 双重一致性检查。
- 概率层面增加 Monte Carlo 与解析面积比交叉验证，避免仅看单样本。

## R07

复杂度分析：
- 记源偏移样本数为 `N`，Monte Carlo 采样数为 `M`，积分成本记为 `Q`。
- 距离计算：`O(Q)`（常数次积分）。
- 样本表计算：`O(N)`（每个 `beta` 常数时间）。
- Monte Carlo 概率估计：`O(M)`。
- 总体时间复杂度：`O(Q + N + M)`。
- 空间复杂度：`O(N + M)`（主要为输出表与采样数组）。

## R08

边界处理：
- 对所有物理量做正值与有限性检查。
- 若 `z_s <= z_l` 直接拒绝计算。
- `beta = 0` 时放大率理论发散，返回 `inf/-inf/inf`（点源极限）。
- `beta >= theta_E` 自动切换为单像分支，内像字段设为 `NaN`。
- Monte Carlo 要求 `n_samples >= 1000`，避免统计波动过大。

## R09

MVP 取舍：
- 只做单透镜、轴对称 SIS，不覆盖椭圆势、外剪切、多透镜平面。
- 只做几何光学级别的强透镜，不做全 GR 测地线追踪或波动光学。
- 不依赖天文 lensing 黑箱库；第三方库仅用于数值积分、数组运算与表格打印。
- 目标是“可验证的最小实现”，而非观测级全流程拟合框架。

## R10

`demo.py` 函数职责：
- `_validate_positive_finite`：通用参数检查。
- `_h0_si / _e_of_z`：宇宙学辅助函数。
- `comoving_distance_m`：计算共动距离。
- `angular_diameter_distance_m`：观测者到目标角径距离。
- `angular_diameter_distance_between_m`：两红移间角径距离。
- `einstein_radius_sis_rad`：SIS 爱因斯坦角。
- `sis_image_positions`：SIS 像位置与像数分支。
- `sis_lens_equation_residual`：方程残差。
- `sis_magnifications`：像放大率与总放大率。
- `simulate_sis_table`：批量生成结果表。
- `strong_lensing_cross_section_source_plane`：强透镜截面。
- `monte_carlo_strong_fraction`：概率估计与误差条。
- `main`：参数装配、打印和断言。

## R11

运行命令：

```bash
cd Algorithms/物理-广义相对论-0382-强透镜_(Strong_Lensing)
uv run python demo.py
```

脚本不读取外部数据文件，不需要命令行参数。

## R12

输出解读：
- 距离项：`D_l, D_s, D_ls`（Mpc）。
- `theta_E`：SIS 爱因斯坦角（arcsec）。
- 表格列：
  - `beta_arcsec`：源偏移；
  - `theta_plus_arcsec / theta_minus_arcsec`：像角位置；
  - `n_images`：像数（1 或 2）；
  - `image_separation_arcsec`：像间分离；
  - `mu_plus / mu_minus / mu_total_abs`：放大率；
  - `is_strong_lensing`：是否处在两像强透镜区；
  - `max_abs_residual`：透镜方程残差。
- 统计项：
  - 强透镜截面（`kpc^2`）；
  - Monte Carlo 强透镜概率；
  - 解析概率与 Monte Carlo 统计误差。

## R13

内置测试策略：
1. 参数固定：
  - `sigma_v = 260 km/s`，`z_l = 0.5`，`z_s = 2.0`。
2. 样本扫描：
  - `beta` 从 `0.1 theta_E` 到 `2.5 theta_E` 等间隔 12 点。
3. 断言：
  - 残差最大值 `< 1e-12`；
  - `beta < theta_E` 的样本必须两像；
  - `beta >= theta_E` 的样本必须单像；
  - Monte Carlo 与解析概率差在 `4σ + 0.005` 容忍内。

## R14

关键参数建议：
- `sigma_v_km_s`：最重要尺度参数，`theta_E ∝ sigma_v^2`。
- `z_lens, z_source`：改变几何因子 `D_ls/D_s`，直接影响多像阈值。
- `beta_samples`：决定展示区间，应覆盖 `theta_E` 两侧。
- `beta_max`：Monte Carlo 的采样圆盘半径，影响解析概率 `(theta_E/beta_max)^2`。
- `n_samples`：统计稳定性参数，越大误差越小但耗时增大。

## R15

方法对比：
- 对比点质量透镜：
  - 点质量在中心有不同偏折标度，SIS 更适合描述星系尺度速度弥散主导的情形。
- 对比黑箱透镜软件：
  - 黑箱可直接给图像但不易审计算法细节；
  - 本实现可逐行核对公式、分支和阈值判据。
- 对比全观测拟合管线：
  - 全管线可处理 PSF、噪声、复杂质量模型；
  - 本 MVP 更轻量，适合教学与基线校验。

## R16

应用场景：
- 强透镜基础教学：清晰展示“何时进入多像区”。
- 观测前数量级估算：给定 `sigma_v` 与红移快速估计像分离尺度。
- 拟合系统的单元测试基线：先用 SIS 保障几何与阈值逻辑正确。
- Monte Carlo 方案验证：检查概率推断模块是否符合解析面积规律。

## R17

扩展方向：
- 质量模型扩展：SIS -> 椭圆 SIS / NFW / 复合势。
- 几何扩展：加入外剪切与多透镜平面。
- 观测扩展：加入 PSF 卷积、噪声模型、像素采样。
- 反演扩展：用 `scipy.optimize` 或 PyTorch 自动微分做参数拟合。
- 数据扩展：引入时延、费马势、多波段联合约束。

## R18

`demo.py` 源码级流程（8 步）：
1. `main` 创建 `Cosmology` 和 `SISLensConfig`，并检查 `z_s > z_l`。  
2. 调用 `comoving_distance_m` 与角径距离函数，得到 `D_l, D_s, D_ls`。  
3. `einstein_radius_sis_rad` 用 `theta_E = 4π(sigma_v/c)^2(D_ls/D_s)` 计算强透镜阈值角。  
4. `simulate_sis_table` 逐个处理 `beta`：`sis_image_positions` 分支出单像/双像并给出像位置。  
5. 同一循环内 `sis_magnifications` 计算 `mu_+, mu_-` 与总放大率，`sis_lens_equation_residual` 回代求残差。  
6. `strong_lensing_cross_section_source_plane` 把 `theta_E` 转成源平面强透镜截面积。  
7. `monte_carlo_strong_fraction` 在 `beta_max` 圆盘随机采样，估计强透镜概率并与解析面积比对照。  
8. `main` 打印表格并执行四类断言（残差、像数分支、概率一致性）；全部通过则脚本正常结束。  

第三方库使用说明：`scipy` 仅用于一维积分，`numpy/pandas` 仅用于数值和结果展示；透镜物理模型与判据全部在源码中显式实现，没有调用黑箱强透镜求解器。
