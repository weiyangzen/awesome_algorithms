# 爱因斯坦环 (Einstein Ring)

- UID: `PHYS-0366`
- 学科: `物理`
- 分类: `广义相对论`
- 源序号: `385`
- 目标目录: `Algorithms/物理-广义相对论-0385-爱因斯坦环_(Einstein_Ring)`

## R01

爱因斯坦环是引力透镜中的典型几何结果：当背景光源、透镜天体、观测者几乎共线时，源像会在透镜周围形成近圆环。

本条目的 MVP 目标是把该现象转成可计算流程：
- 用点质量透镜方程计算像平面到源平面的映射；
- 在数值网格上渲染透镜后亮度分布；
- 从图像径向统计反推出环半径并对照理论爱因斯坦角 `theta_E`。

## R02

问题定义（MVP 范围）：
- 输入：
  - 透镜质量 `M`（太阳质量单位）；
  - 观测者-透镜距离 `D_l` 与观测者-光源距离 `D_s`；
  - 光源在源平面的角位置偏移 `beta_x, beta_y`；
  - 光源角尺度 `sigma`；
  - 数值网格参数 `npix, fov_arcsec`。
- 输出：
  - 理论爱因斯坦角 `theta_E`；
  - 数值估计环半径 `radius_hat`；
  - 误差指标（相对误差）；
  - 方位均匀性指标（`azimuthal_cv`）用于区分完整环与破碎弧像。

脚本内置两组样例（近共线 vs 偏轴），`uv run python demo.py` 可直接运行，无需交互输入。

## R03

核心物理公式：

1. 爱因斯坦角（点质量透镜）

`theta_E = sqrt((4GM/c^2) * (D_ls/(D_l D_s)))`

其中 `D_ls = D_s - D_l`（本 MVP 采用平直几何近似）。

2. 点质量透镜方程（二维角坐标）

`beta = theta - (theta_E^2 / |theta|^2) * theta`

3. 面亮度守恒下的像平面亮度构造

`I(theta) = I_s(beta(theta))`

4. 本实现采用高斯源模型

`I_s(beta) = exp(-|beta - beta_0|^2 / (2 sigma^2))`

## R04

算法高层流程：

1. 校验透镜系统参数（质量、距离、源尺度必须合法）。
2. 由 `M, D_l, D_s` 计算理论 `theta_E`（角秒）。
3. 在像平面构建二维角坐标网格 `theta_x, theta_y`。
4. 用点质量透镜方程映射到源平面 `beta_x, beta_y`。
5. 在源平面评估高斯亮度并回写像平面，得到 `image`。
6. 对 `image` 计算径向平均剖面 `profile(r)`，取峰值半径作为 `radius_hat`。
7. 在 `r=theta_E` 附近环带统计方位亮度均匀性（变异系数 `azimuthal_cv`）。
8. 输出每个案例的理论半径、估计半径、误差和环像判定。

## R05

核心数据结构：

- `LensSystem`（`dataclass`）：
  - `name`：案例名称；
  - `mass_msun`：透镜质量（太阳质量）；
  - `d_l_mpc`、`d_s_mpc`：距离（Mpc）；
  - `beta_x_arcsec`、`beta_y_arcsec`：源角偏移（角秒）；
  - `source_sigma_arcsec`：源角尺度（角秒）。
- `numpy.ndarray`：
  - `theta_x, theta_y`：像平面网格；
  - `image`：透镜后亮度；
  - `centers, profile`：径向中心和径向平均亮度。
- `dict` 结果：
  - `theta_E_arcsec`、`radius_hat_arcsec`、`rel_error`、`azimuthal_cv`、`ring_like`。

## R06

正确性要点：

- 物理关系透明：`theta_E` 与透镜方程都在源码中显式计算，不依赖黑盒天体物理包。
- 数值稳定：在 `|theta|^2` 分母处加入 `eps` 防止中心点除零。
- 几何可解释：源偏移为 0 时应出现近圆环，偏移增大时应转为非均匀弧像。
- 指标闭环：
  - `radius_hat` 检验“半径是否符合理论”；
  - `azimuthal_cv` 检验“方位是否均匀成环”。
- 脚本内置断言确保两类场景都满足预期行为。

## R07

复杂度分析（网格边长 `N=npix`，径向分箱数 `B`，方位分箱数 `P`）：

- 网格映射与亮度渲染：`O(N^2)`；
- 径向统计：`O(N^2 + B)`；
- 方位均匀性统计：`O(N^2 + P)`。

总体时间复杂度 `O(N^2)`，空间复杂度 `O(N^2)`。

## R08

边界与异常处理：

- `mass <= 0`、`D_l <= 0`、`D_s <= D_l`、`sigma <= 0`：抛 `ValueError`。
- 网格参数异常（`npix < 64`、`fov <= 0`）：抛 `ValueError`。
- 数组形状不一致：抛 `ValueError`。
- 径向分箱过小或约束无可用 bin：抛 `RuntimeError`。
- 环带像素不足导致方位统计失效：抛 `RuntimeError`。

## R09

MVP 取舍说明：

- 只实现“点质量透镜 + 单高斯源”，不覆盖椭圆势、剪切场、多透镜平面。
- 距离关系采用简化几何 `D_ls=D_s-D_l`，不做完整宇宙学距离模型。
- 输出以终端表格为主，不生成图像文件，保持最小依赖和快速验证。
- 重点在“机制清晰 + 可跑通 + 可审计”，不是科研级全功能透镜模拟器。

## R10

`demo.py` 函数职责：

- `validate_system`：校验透镜系统参数合法性。
- `einstein_radius_arcsec`：计算理论爱因斯坦角。
- `build_theta_grid`：创建像平面网格。
- `lens_equation_point_mass`：执行点质量透镜映射 `theta -> beta`。
- `gaussian_source_intensity`：定义源平面亮度分布。
- `simulate_lensed_image`：串联网格、透镜方程和亮度映射得到图像。
- `radial_profile`：计算径向平均亮度曲线。
- `estimate_ring_radius`：从径向峰值估计环半径。
- `azimuthal_uniformity_cv`：评估环带方位均匀性。
- `analyze_system`：输出单案例诊断指标。
- `run_checks`：执行两个基准案例的通过门限。

## R11

运行方式：

```bash
cd Algorithms/物理-广义相对论-0385-爱因斯坦环_(Einstein_Ring)
uv run python demo.py
```

脚本无交互输入、无网络依赖，运行后打印结果表并给出通过标记。

## R12

输出字段解读：

- `case`：案例名（近共线环像 / 偏轴弧像）。
- `theta_E_arcsec`：理论爱因斯坦角（角秒）。
- `radius_hat_arcsec`：由径向剖面反推的峰值半径（角秒）。
- `rel_error`：`|radius_hat-theta_E|/theta_E`。
- `azimuthal_cv`：环带方位亮度变异系数（越小越接近完整圆环）。
- `ring_like`：是否判定为“完整环像”。
- `beta_offset_arcsec`：源在源平面的总角偏移量。

## R13

建议最小测试集（已内置）：

- `near-aligned source (Einstein ring)`：
  - `beta=0`，应出现高均匀性的环像；
  - `rel_error` 小，`azimuthal_cv` 低。
- `misaligned source (broken arc)`：
  - `beta=0.7*theta_E`，应明显破坏方位均匀性；
  - `ring_like=False`。

可补充异常测试：
- `D_s <= D_l`；
- `mass <= 0`；
- `npix < 64`。

## R14

关键可调参数：

- `mass_msun`：增大将提升 `theta_E`（环半径变大）。
- `d_l_mpc, d_s_mpc`：改变透镜几何因子 `D_ls/(D_l D_s)`。
- `beta_x_arcsec, beta_y_arcsec`：控制“完整环”到“弧像”过渡。
- `source_sigma_arcsec`：控制环宽度，越大环越厚。
- `npix`：空间分辨率，越大越精确但耗时更高。
- `fov_arcsec`：视场范围，需覆盖环半径附近区域。

## R15

方法对比：

- 对比“只计算理论 `theta_E`”：
  - 理论值不能验证图像形态；
  - 本实现加入数值成像与统计，可同时检查半径与环形结构。
- 对比“黑盒透镜包一键渲染”：
  - 黑盒调用快但算法细节隐藏；
  - 本实现完整展开透镜映射和统计步骤，便于教学与审计。
- 对比复杂透镜模型（NFW、椭圆势、外剪切）：
  - 复杂模型更接近观测数据；
  - 本 MVP 计算量小、可解释性高，适合作为基线模块。

## R16

典型应用场景：

- 广义相对论与引力透镜教学演示；
- 天文数据处理前的简化模型 sanity check；
- 透镜参数反演算法的合成数据基准；
- 图像形态指标（环半径、方位均匀性）定义与验证。

## R17

可扩展方向：

- 引入椭圆透镜势和外剪切，模拟更真实的弧像结构；
- 使用宇宙学角径距离替代简化 `D_s-D_l`；
- 支持多源/多透镜联合建模；
- 增加噪声、PSF 和像素响应以接近观测成像流程；
- 加入参数拟合环节，从观测图反演 `M` 和几何距离组合。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 构建两个 `LensSystem`：一个近共线（目标是形成爱因斯坦环），一个偏轴（目标是形成非完整弧像）。
2. `einstein_radius_arcsec` 用 `M, D_l, D_s` 计算每个系统的理论 `theta_E`，并做输入合法性校验。
3. `build_theta_grid` 在像平面生成二维角坐标网格 `theta_x, theta_y`。
4. `lens_equation_point_mass` 按 `beta = theta - theta_E^2 * theta / |theta|^2` 把每个像素映射到源平面坐标。
5. `gaussian_source_intensity` 在源平面评估高斯亮度，`simulate_lensed_image` 返回透镜后亮度图 `image`。
6. `radial_profile` 对 `image` 做径向分箱平均，`estimate_ring_radius` 取峰值半径 `radius_hat`，并计算与 `theta_E` 的相对误差。
7. `azimuthal_uniformity_cv` 在 `r=theta_E` 附近环带按方位角分箱，计算亮度变异系数，量化“是否接近完整圆环”。
8. `run_checks` 验证近共线案例满足低误差+高均匀性、偏轴案例满足高非均匀性，然后打印结果表与通过信息。

第三方库没有作为物理黑盒使用：`numpy/pandas` 仅承担数值运算和表格展示，透镜物理关系、图像构造和判定逻辑都在源码中逐步展开。
