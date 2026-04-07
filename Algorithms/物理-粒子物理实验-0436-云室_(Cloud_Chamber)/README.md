# 云室 (Cloud Chamber)

- UID: `PHYS-0416`
- 学科: `物理`
- 分类: `粒子物理实验`
- 源序号: `436`
- 目标目录: `Algorithms/物理-粒子物理实验-0436-云室_(Cloud_Chamber)`

## R01

云室图像/点云分析的目标是从“液滴轨迹点”中恢复粒子轨迹，并给出基础物理量估计。这个 MVP 将问题抽象成二维点集上的三件事：
1. 轨迹点与噪声点分离。
2. 对每条候选轨迹做几何拟合（直线或圆弧）。
3. 对圆弧轨迹按磁场弯曲半径估计动量量级。

## R02

云室中带电粒子电离介质，过饱和蒸汽沿电离路径凝结，形成可见液滴链。若存在近似均匀磁场 `B`，洛伦兹力使轨迹近似圆弧，满足经验关系：

`p (GeV/c) ≈ 0.3 * B(T) * r(m)`

因此，几何半径 `r` 是可观测代理量，适合在算法练习中做参数反演。

## R03

MVP 输入不是外部文件，而是脚本内合成的点表 `DataFrame`：
- `x_mm, y_mm`: 液滴坐标（毫米）
- `true_label`: 仅用于合成时的可读标记，不参与拟合决策

输出为两部分：
- 聚类后点表（附 `cluster_id`）
- 每条轨迹的摘要表：`chosen_model, radius_mm, momentum_gev_c, line_bic, circle_bic`

## R04

几何模型采用“直线 + 圆”竞争：
- 直线模型：正交距离最小化（PCA 方向）
- 圆模型：`(x-cx)^2 + (y-cy)^2 = r^2`

模型选择使用 BIC：

`BIC = n * ln(MSE) + k * ln(n)`

其中 `k` 为参数个数（直线 2，圆 3），`MSE` 为对应残差均方。

## R05

合成数据包含三条典型轨迹与背景噪声：
- Track A: 近直线（高动量）
- Track B: 中等半径圆弧
- Track C: 小半径圆弧（低动量）
- 随机噪声滴点

这样可覆盖“直线/弯曲/噪声”三类基本模式，便于验证流程完整性。

## R06

预处理步骤：
1. 取坐标矩阵 `X = [[x_mm, y_mm], ...]`
2. 用 `StandardScaler` 标准化，消除不同轴尺度影响
3. 在标准化空间进行密度聚类

原因：DBSCAN 的 `eps` 对尺度敏感，先标准化可使参数更稳定。

## R07

聚类算法使用 `DBSCAN(eps=0.11, min_samples=10)`：
- 优点：无需预设簇数、天然支持噪声标记 `-1`
- 适配场景：云室轨迹通常是局部连通的高密度点链

聚类后仅对 `cluster_id >= 0` 且点数足够的簇进行拟合。

## R08

直线拟合细节：
- 对簇点中心化后做 SVD
- 第一主方向作为轨迹方向
- 用法向量计算所有点到直线的正交距离
- `line_mse = mean(distance^2)`

这种方式比普通 `y=ax+b` 更通用，可处理接近垂直方向的轨迹。

## R09

圆拟合分两阶段：
1. `scipy.optimize.least_squares` 做 soft-L1 鲁棒拟合，得初值。
2. `PyTorch` 用 `smooth_l1_loss` + Adam 继续优化 `(cx, cy, r)`。

这样既利用 SciPy 的稳定初值能力，又展示了可微优化在实验分析中的可扩展性。

## R10

模型判别逻辑：
1. 分别计算 `line_bic` 与 `circle_bic`。
2. 若圆半径异常大（`r > 300 mm`），按近直线处理。
3. 否则选 BIC 更小者。

这是一种小而实用的防过拟合规则，避免把几乎直线的簇硬解释成大圆。

## R11

对判定为圆弧的轨迹，动量估计为：

`p ≈ 0.3 * B * r`

MVP 固定 `B = 0.8 T`，并将 `r_mm -> r_m` 后输出 `momentum_gev_c`。该值用于量级示范，不代表完整实验校准结果。

## R12

时间复杂度（`N` 为总点数，`m_i` 为第 `i` 个簇点数）：
- DBSCAN: 近似 `O(N log N)`（依赖邻域索引实现）
- 线拟合: 每簇 `O(m_i)`（2D SVD 常数极小）
- 圆拟合: 每簇 `O(T * m_i)`，`T` 为迭代步数

总体在小型教学数据上可快速运行。

## R13

数值与工程注意点：
- BIC 中对 `MSE` 加下限 `1e-12`，避免 `log(0)`。
- 圆半径用 `softplus(raw_r)` 保证正值。
- 计算距离时加微小项避免 `sqrt(0)` 导致梯度不稳。
- 小簇（点太少）直接跳过，减少偶然噪声误判。

## R14

`demo.py` 结构：
- `simulate_cloud_chamber`: 生成合成点
- `fit_line_pca`: 直线拟合与残差
- `fit_circle_scipy`: SciPy 圆拟合
- `refine_circle_torch`: Torch 精修圆参数
- `analyze_tracks`: 聚类、拟合、模型选择、动量估计
- `main`: 输出簇规模和轨迹摘要表

全部逻辑集中在单文件，便于审阅与扩展。

## R15

运行方式：

```bash
uv run python Algorithms/物理-粒子物理实验-0436-云室_(Cloud_Chamber)/demo.py
```

脚本无需交互输入，直接在终端打印分析结果。

## R16

典型输出应包含：
- `Total droplets`: 总点数
- `Cluster sizes`: 每个轨迹簇点数
- `Track estimates`: 每簇最终模型与参数

若某簇为圆模型，会显示 `radius_mm` 与 `momentum_gev_c`；若为直线模型，这两列为 `None`。

## R17

当前 MVP 的边界：
- 用的是二维合成点，不含真实光学成像误差与时间维。
- 没有做轨迹交叉点（散射/衰变顶点）识别。
- 没有通过磁场方向和点序恢复电荷符号。

可扩展方向：加入图像分割前端、Hough/RANSAC 备选模型、3D 重建与不确定度传播。

## R18

`demo.py` 的源码级算法流（非黑箱）如下：

1. `simulate_cloud_chamber` 明确构造直线、圆弧和随机噪声点并合并成表。
2. `StandardScaler` 将 `(x_mm, y_mm)` 归一化，保证 DBSCAN 的半径阈值可比。
3. `DBSCAN.fit_predict` 输出每个点的 `cluster_id`，噪声标为 `-1`。
4. 对每个有效簇，`fit_line_pca` 通过 SVD 求主方向，并计算正交残差 `line_mse`。
5. 同一簇进入 `fit_circle_scipy`：定义几何残差 `distance_to_center - r`，由 `least_squares` 求初始 `(cx, cy, r)`。
6. 初值传给 `refine_circle_torch`：把 `r` 重新参数化为 `softplus(raw_r)`，用 Adam 最小化 smooth-L1 残差，得到更稳健圆参数。
7. 用 `bic()` 分别计算线/圆模型代价，并结合“大半径即近直线”规则做最终模型选择。
8. 若选圆模型，调用 `estimate_momentum` 按 `p≈0.3Br` 输出动量估计；最后汇总为 pandas 表并打印。

以上 8 步即该 MVP 的完整可追踪流程，第三方库只承担明确的数值子任务，不掩盖决策逻辑。
