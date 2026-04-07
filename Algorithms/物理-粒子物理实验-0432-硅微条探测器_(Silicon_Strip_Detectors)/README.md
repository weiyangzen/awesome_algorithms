# 硅微条探测器 (Silicon Strip Detectors)

- UID: `PHYS-0412`
- 学科: `物理`
- 分类: `粒子物理实验`
- 源序号: `432`
- 目标目录: `Algorithms/物理-粒子物理实验-0432-硅微条探测器_(Silicon_Strip_Detectors)`

## R01

硅微条探测器用于高能粒子径迹重建：带电粒子穿过硅层后产生电荷，电荷被离散条带(strip)读出，形成多层命中点，再通过直线或曲线拟合恢复轨迹参数。本 MVP 聚焦最小可运行链路：
1. 生成直线穿越事件。
2. 模拟条带电荷共享、电子学噪声和阈值判别。
3. 聚类得到每层测量位置。
4. 用鲁棒线性拟合重建轨迹并评估分辨率。

## R02

物理直觉：
- 条带读出是“离散位置采样”，单条带分辨率约为 `pitch/sqrt(12)`。
- 若有电荷共享并做重心(centroid)插值，可优于单条带分辨率。
- 多层测量后，轨迹拟合会进一步降低参数误差（`x0` 与斜率）。

## R03

本实现的输入/输出：
- 输入：`DetectorConfig` 中的几何参数、噪声参数、阈值、事件数与随机种子。
- 输出：
  - 事件级 `pandas.DataFrame`（命中数、是否重建、拟合参数、误差）。
  - 汇总指标（命中效率、重建效率、位置分辨率、残差 RMS、`x0`/斜率 RMSE）。

## R04

建模假设（为保持 MVP 小而真实）：
- 粒子轨迹近似为 `x(z)=x0 + t*z` 的直线（不含磁场弯曲）。
- 每层只考虑 1D 条带坐标 `x`，忽略 `y` 与时间维。
- 电荷横向扩散用高斯近似。
- 使用固定阈值判别，不建模复杂前端电子学整形。
- 少量外点(outlier)用于测试鲁棒拟合。

## R05

轨迹模型：
- 真值：`x_true(z_i) = x0 + t*z_i`。
- 层内测量：`x_meas = centroid(cluster)`。
- 拟合：最小化内点残差后得到 `x_fit(z)=a+b*z`。
- 关键误差指标：
  - 位置误差：`(x_meas - x_true)`（um）。
  - 拟合残差：`(x_meas - x_fit)`（um）。
  - 参数 RMSE：`RMSE(a-x0)`、`RMSE(b-t)`。

## R06

条带信号模型（`simulate_layer_hit`）：
1. 在真值位置附近取 `±3*pitch` 的局部条带窗口。
2. 用 `scipy.stats.norm.pdf` 计算各条带的高斯权重。
3. 总电荷归一化到 `mip_charge_e`。
4. 叠加高斯电子噪声 `noise_sigma_e`。
5. 用 `threshold_e` 做阈值筛选。

## R07

聚类与位置重建：
- 对超过阈值的条带索引做“连续段分裂”(contiguous runs)。
- 选择总电荷最大的簇作为主簇。
- 用电荷加权重心求命中位置：
  `x_centroid = sum(q_i*x_i)/sum(q_i)`。

## R08

轨迹重建算法：
- 使用 `sklearn` 的 `RANSACRegressor(LinearRegression)`。
- `min_samples=2`，`residual_threshold=0.20 mm`。
- 需要至少 3 层命中才计为成功重建。
- 输出截距 `a`（对应 `z=0` 时的 `x0`）和斜率 `b`。

## R09

核心流程伪代码：

```text
for each event:
    sample true x0, slope
    for each layer z:
        x_true = x0 + slope*z
        x_meas = simulate_layer_hit(x_true)
        if hit: store (z, x_meas)
    fit = RANSAC line fit on stored hits
    compute event-level errors/flags
aggregate all events -> summary metrics
```

## R10

代码组织（`demo.py`）：
- `DetectorConfig`: 所有可调参数。
- `strip_centers_mm`: 构造几何中心位置。
- `find_contiguous_runs`: 条带聚类辅助函数。
- `simulate_layer_hit`: 层级信号与聚类重心。
- `fit_track_ransac`: 鲁棒直线拟合。
- `run_mvp`: 批量仿真与统计。
- `main`: 打印关键指标和前几行事件结果。

## R11

运行方式：

```bash
uv run python "Algorithms/物理-粒子物理实验-0432-硅微条探测器_(Silicon_Strip_Detectors)/demo.py"
```

脚本无交互输入，固定随机种子，结果可复现。

## R12

输出解释：
- `hit_efficiency`: 总命中层数 / 总可观测层数。
- `track_reco_efficiency`: 事件中成功完成拟合的比例。
- `position_resolution_um`: 命中位置相对真值的标准差。
- `fit_residual_rms_um`: 拟合后残差 RMS，反映重建质量。
- `x0_rmse_um` 与 `slope_rmse_mrad`: 轨迹参数精度。

## R13

复杂度（设事件数 `E`、层数 `L`、局部窗口条带数 `W`）：
- 信号仿真与阈值筛选：`O(E*L*W)`。
- 聚类与重心：`O(E*L*W)`。
- 直线拟合：每事件命中数很小（通常 <=6），可视作 `O(E)` 常数开销。
- 总体：`O(E*L*W)`，内存约 `O(E)`（事件表）。

## R14

数值与工程注意点：
- 阈值过低会增噪声命中，过高会降低效率。
- `charge_sigma_mm` 决定电荷共享程度，影响重心精度。
- `residual_threshold` 太小会误拒真命中，太大则抑制外点能力下降。
- 保持单位一致（mm、um、mrad）对结果解释很关键。

## R15

最小验证策略：
1. 检查是否能稳定输出非 NaN 指标。
2. 确认 `track_reco_efficiency` 在合理范围（通常接近 1，但受噪声和外点影响）。
3. 对比 `position_resolution_um` 与 `fit_residual_rms_um`，拟合残差应通常不劣于单点误差量级。
4. 调整阈值/噪声观察趋势是否符合物理直觉。

## R16

当前 MVP 的边界：
- 未建模磁场下曲率、Landau 能损、层间材料散射。
- 仅单轨事件，不含 pile-up 与多轨歧义。
- 聚类策略为“最大电荷簇”，未做高级模式识别。

可扩展方向：
- 加入卡尔曼滤波与多重散射过程噪声。
- 引入 2D/双面条带几何和真实通道坏道掩码。
- 将模拟改为更接近实验电子学响应链。

## R17

与真实实验软件链的对应关系：
- 本例相当于将 Geant4 能量沉积、digitization、clustering、track fit 做了高度压缩。
- `simulate_layer_hit` 对应 digitization + clustering 的简化版。
- `fit_track_ransac` 对应重建链中“pattern recognition + line fit”的最小替代。

## R18

第三方算法（`RANSACRegressor`）在本任务中的源码级流程可拆为 8 步：
1. 设定基学习器为普通线性回归 `x=a+bz`。
2. 随机抽取最小样本集（这里是 2 个命中点）拟合候选直线。
3. 用候选直线计算全部命中点残差 `|x_i-(a+bz_i)|`。
4. 依据 `residual_threshold` 标记内点集合。
5. 若当前内点数优于历史最佳，保存该模型为“最佳候选”。
6. 重复随机采样-评估循环，直到达到停止准则（迭代次数/置信度）。
7. 使用最佳内点集合重新训练线性回归，得到最终 `a,b`。
8. 输出最终模型参数并用于事件级残差和效率统计。

这保证了实现不是“黑盒一行调用不解释”，而是明确了鲁棒拟合在条带外点场景下如何工作。
