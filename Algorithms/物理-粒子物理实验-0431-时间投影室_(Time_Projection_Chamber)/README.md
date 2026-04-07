# 时间投影室 (Time Projection Chamber)

- UID: `PHYS-0411`
- 学科: `物理`
- 分类: `粒子物理实验`
- 源序号: `431`
- 目标目录: `Algorithms/物理-粒子物理实验-0431-时间投影室_(Time_Projection_Chamber)`

## R01

时间投影室（TPC）是通过“漂移时间 + 读出平面位置”来恢复三维空间电离轨迹的探测器。该条目聚焦最小可复现问题：给定含噪 `x/y` pad 读出与漂移时间 `t`，重建单条带电粒子轨迹的 3D 直线近似（用 `x(z), y(z)` 表示），并评估抗离群点能力。

## R02

该算法在工程上的价值：

- 把原始可观测量 `x_pad, y_pad, t_drift` 转为几何重建参数；
- 显式展示漂移速度标定如何进入重建公式；
- 对比 OLS 与鲁棒 Huber 在异常击中（噪声簇、假击中）下的行为差异；
- 提供一个可直接扩展到多事件批处理的基线实现。

## R03

MVP 输入输出（脚本内固定参数，无交互）：

- 输入：
1. TPC 几何与读出参数（`z` 范围、pad pitch、漂移速度）；
2. 真值轨迹参数 `x(z)=k_x z+b_x, y(z)=k_y z+b_y`；
3. 测量噪声模型（pad 量化/高斯噪声、时间扩散噪声）；
4. 一定比例离群击中。
- 输出：
1. `OLS`、`Huber`、`Huber+Torch` 三种重建参数；
2. 斜率与截距误差、横向 RMSE、离群召回率等指标表；
3. 断言通过信息 `All checks passed.`。

## R04

核心物理与数学关系：

1. 漂移时间到纵向坐标：`z = v_drift * t`。
2. 轨迹近似（无磁场曲率教学模型）：
`x(z)=k_x z+b_x`, `y(z)=k_y z+b_y`。
3. 联合残差（二维横向）：
`r = [(x_pred-x_meas)/s_xy, (y_pred-y_meas)/s_xy]`。
4. OLS 目标：最小化 `sum r^2`。
5. Huber 目标：小残差二次、大残差线性，以降低离群点影响。

## R05

设击中数为 `N`，鲁棒迭代次数为 `I`，Torch 微调步数为 `S`：

- OLS：`O(N)`（2 个一维线性回归）；
- Huber（`least_squares`）：约 `O(I*N)`；
- Torch 微调：`O(S*N)`；
- 空间复杂度：`O(N)`。

在本脚本默认规模（`N=84`）下，运行开销很小，适合教学与 CI 验证。

## R06

`demo.py` 的最小闭环：

- **闭环 A：事件模拟**
  生成真轨迹 -> pad 读出量化 + 时间噪声 -> 注入离群击中。
- **闭环 B：重建求解**
  `t -> z` 反演后做 OLS、Huber、Torch 鲁棒拟合。
- **闭环 C：质量评估**
  对比真值并输出 DataFrame，最后执行自动断言。

## R07

优点：

- 代码结构与探测器数据流一致，便于审计；
- 鲁棒拟合路径明确，能观察离群点抑制效果；
- 同时包含解析回归（OLS）与数值优化（Huber/Torch）。

局限：

- 仅单轨迹，未处理多轨迹 hit 关联；
- 仅直线近似，未建模磁场螺旋和多重散射；
- 漂移速度设为常数，未包含空间依赖标定图。

## R08

前置知识与环境：

- TPC 基本机理（电离、漂移、读出）；
- 最小二乘与鲁棒损失概念；
- Python `>=3.10`；
- 依赖：`numpy`、`scipy`、`pandas`、`scikit-learn`、`torch`。

## R09

适用场景：

- 粒子物理实验课程中的 TPC 重建入门；
- 算法原型验证（读出到轨迹参数）；
- 比较线性拟合与鲁棒拟合在异常点下的稳定性。

不适用场景：

- 真实大规模离线重建生产链；
- 需要电场畸变、气体非均匀、空间电荷校正的高保真分析；
- 多事件 pile-up 与复杂拓扑（簇分裂、共享击中）场景。

## R10

正确性直觉：

1. 漂移速度已知时，`t` 的一阶映射可恢复 `z`；
2. 对固定 `z` 数据，`x(z), y(z)` 的参数可用最小二乘估计；
3. 离群点会显著拉偏 OLS，而 Huber 对大残差减权；
4. Torch 的 pseudo-Huber 微调在 Huber 解附近做可导细化，通常不会破坏稳定解。

## R11

数值稳定策略：

- 固定随机种子保证复现；
- `z_meas` 做物理范围裁剪，避免非物理漂移结果；
- 回归与优化都使用 `float64`；
- Torch 仅做小步数微调，避免过拟合噪声；
- 指标断言以“鲁棒方法优于或不劣于 OLS”为主。

## R12

关键参数：

- `drift_velocity_cm_per_ns`：时间到 `z` 的比例，标定误差直接产生纵向尺度偏差；
- `pad_pitch_cm`：横向离散化尺度，影响空间分辨率；
- `time_sigma_ns` 与 `diffusion_sigma_ns`：时间噪声与随 `z` 增强的扩散项；
- `outlier_fraction`：离群点强度；
- `huber_f_scale`、`torch_delta`：鲁棒损失转折阈值。

调参建议：

- 离群点比例升高时，优先减小 `huber_f_scale`；
- 时间噪声偏大时，适当增加击中数 `n_hits`；
- 若 Torch 微调不稳定，降低 `torch_lr` 或减少 `torch_steps`。

## R13

- 近似比保证：N/A（非组合近似算法）。
- 随机成功率保证：N/A（固定随机种子的确定性流程）。

本脚本内置可验证阈值：

- `Huber` 的 `transverse_rmse_inlier_cm < 0.28`；
- `Huber` 的 `outlier_recall >= 0.70`；
- `Huber+Torch` 的 `slope_error_x < 0.010`；
- `Huber+Torch` 的 `slope_error_y < 0.010`。

## R14

常见失效模式：

1. 漂移速度错标导致整体 `z` 缩放错误；
2. pad 几何/对准参数错误造成系统性横向偏移；
3. 离群比例过高时，鲁棒损失也可能被污染；
4. 轨迹近乎垂直于 `z` 轴假设失效时，`x(z), y(z)` 参数化不稳定。

## R15

可扩展方向：

- 引入磁场下的螺旋轨迹参数化；
- 联合 `dE/dx`、簇宽等信息做 hit 质量加权；
- 使用 RANSAC/Kalman 做多轨迹与逐层更新；
- 将单事件扩展为批量统计，输出效率与分辨率曲线。

## R16

相关模块与算法：

- 线性最小二乘（解析基线）；
- `scipy.optimize.least_squares` 的 Huber 鲁棒求解；
- pseudo-Huber 可导损失与 PyTorch 自动微分；
- `sklearn.metrics.mean_squared_error` 用于评估量化。

## R17

`demo.py` MVP 功能清单：

- 生成含噪单轨迹 TPC 读出并注入离群击中；
- 完成 `t -> z` 反演与三种拟合（OLS/Huber/Huber+Torch）；
- 输出统一指标表（误差、残差、离群检出）；
- 执行断言，保证脚本可自动验收。

运行方式：

```bash
cd Algorithms/物理-粒子物理实验-0431-时间投影室_(Time_Projection_Chamber)
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程（9 步）：

1. `simulate_event` 采样真值 `z_true`，按直线模型生成 `x_true/y_true`。  
2. 对 `x/y` 做 pad 量化与高斯噪声叠加，得到 `x_meas/y_meas`。  
3. 依据 `t = z/v_drift + 噪声` 生成时间读出，并随机替换部分击中为离群点。  
4. 将时间读出反演为 `z_meas = v_drift * t_meas`，并裁剪到物理漂移范围。  
5. `linear_xy_fit` 分别对 `x(z)` 和 `y(z)` 执行 `np.linalg.lstsq`，得到 OLS 初值。  
6. `huber_xy_fit` 构造联合残差向量，调用 `scipy.optimize.least_squares(loss="huber")` 求鲁棒参数。  
7. `torch_refine` 以 Huber 结果为初值，在 pseudo-Huber 损失上用 Adam 做少量梯度步细化。  
8. `evaluate_method` 用 `sklearn.metrics.mean_squared_error` 计算 inlier 横向 RMSE，并以残差阈值统计离群召回率/误报率。  
9. `main` 汇总 `pandas.DataFrame`、打印结果并执行断言。

说明：第三方库仅承担“线性代数/优化器/指标计算”基础能力，漂移反演、残差构造、离群判定与评估逻辑均在源码中显式实现，非单函数黑盒。
