# 径迹重建 (Track Reconstruction)

- UID: `PHYS-0408`
- 学科: `物理`
- 分类: `粒子物理实验`
- 源序号: `428`
- 目标目录: `Algorithms/物理-粒子物理实验-0428-径迹重建_(Track_Reconstruction)`

## R01

问题定义：在粒子物理实验中，径迹重建的核心任务是从探测器离散命中点（hits）恢复带电粒子的轨迹参数。本 MVP 聚焦 `x-y` 平面、均匀磁场下的弯曲径迹（圆弧近似），实现从“多轨真命中 + 假命中”中恢复轨道圆参数 `(cx, cy, R)`。

## R02

输入（`demo.py` 内部模拟生成）：
- 多层探测命中点：每层包含真实命中与随机假命中。
- 束斑先验（beam spot）：默认 `(0, 0)`。
- 噪声与几何超参数（层数、噪声、曲率范围、阈值等）。

输出：
- 重建出的多条轨迹圆参数 `(center_x, center_y, radius)`。
- 每条轨迹的命中数、平均残差、命中纯度。
- 事件级指标：重建效率、假轨率、半径/圆心 RMSE。

## R03

简化假设：
- 只考虑横向二维平面，不含 `z` 和时间信息。
- 轨迹在 `x-y` 平面可由圆描述（均匀 `Bz`）。
- 每层最多取一个命中关联到某条轨迹。
- 束斑位置已知，可作为几何约束筛掉大部分伪种子。

这些假设使代码保持最小可运行，同时保留径迹重建中的关键步骤：种子、关联、鲁棒精修、匹配评估。

## R04

轨迹模型（均匀磁场下的二维投影）：
- 参数：初始方向 `phi0`、曲率 `kappa`、束斑位置 `v0=(x0,y0)`。
- 轨迹点按弧长 `s` 生成：

`x(s)=x0 + (sin(phi0+kappa*s)-sin(phi0))/kappa`

`y(s)=y0 - (cos(phi0+kappa*s)-cos(phi0))/kappa`

- 对应圆心与半径：

`c=(x0-sin(phi0)/kappa, y0+cos(phi0)/kappa), R=|1/kappa|`

## R05

数据生成（`simulate_event`）：
1. 随机采样 `n_truth_tracks` 条真轨迹的 `phi0`、电荷符号、`|kappa|`。
2. 在固定层弧长位置生成理想命中点。
3. 叠加高斯测量噪声，得到真实测量命中。
4. 每层再注入若干随机假命中（fake hits），构成混合命中集合。

默认参数下：5 条真轨迹、7 层、每层 3 个假命中。

## R06

种子生成（triplet seeding）：
- 选定三层（默认层 `0,3,6`），枚举三层命中点组合。
- 对每个三点组合求过三点圆（`circle_from_three_points`）。
- 对退化情况（近共线）直接丢弃。

这是常见“几何三点种子”思想的最小实现。

## R07

命中关联与候选打分：
- 对一个候选圆，在每一层取“径向残差最小”的命中。
- 径向残差定义：

`r_i = | ||p_i-c|| - R |`

- 仅接受 `r_i < residual_cut` 的层命中。
- 候选得分：优先最大化命中层数，其次最小化平均残差。

并引入束斑一致性约束：
`| ||c-beamspot|| - R | < tol`，抑制不物理的伪轨迹。

## R08

多轨重建策略（贪心）：
- 每轮选当前最佳候选轨迹。
- 将该轨迹已关联命中从可用命中池移除。
- 重复下一轮种子搜索，直到没有可行候选或达到最大轨迹数。

该策略是教学级最简多轨提取框架，易懂且可复现。

## R09

鲁棒精修（SciPy）：
- 对候选轨迹命中点做非线性最小二乘圆拟合。
- 调用 `scipy.optimize.least_squares`，损失设为 `soft_l1`，降低外点影响。
- 变量为 `(cx, cy, R)`，残差为 `||p-c||-R`。

相比只用三点定圆，这一步显著提升参数稳定性。

## R10

自动微分精修（PyTorch）：
- 以 SciPy 输出为初始化。
- 使用 pseudo-Huber 损失：

`L = sum(delta^2 * (sqrt(1 + (r_i/delta)^2) - 1))`

- 用 Adam 迭代更新 `(cx, cy, R)`，做小步二次优化。

作用：展示传统重建与可微优化结合的最小落地。

## R11

真值匹配与指标评估：
- 构建 `reco x truth` 成本矩阵：每个元素为“真轨迹理想命中到重建圆”的 RMS 径向残差。
- 用 `linear_sum_assignment`（Hungarian）求一对一最优匹配。
- 低于阈值的匹配计为成功。

输出关键指标：
- `tracking_efficiency`
- `fake_track_rate`
- `assigned_hit_purity`
- `radius_rmse_mm`, `center_x_rmse_mm`, `center_y_rmse_mm`

## R12

默认超参数（`Config`）：
- 曲率范围：`kappa_abs_range=(0.004, 0.010) 1/mm`
- 命中噪声：`sigma_hit_mm=0.25`
- 种子残差阈值：`residual_cut_seed_mm=0.50`
- 最终残差阈值：`residual_cut_final_mm=0.34`
- 最小轨迹命中层数：`min_hits_per_track=5`
- 束斑一致性阈值：`beamspot_tolerance_mm=2.8`

## R13

运行方式（无交互）：

```bash
uv run python demo.py
```

程序会打印摘要指标和重建轨迹表。

## R14

结果解读建议：
- `tracking_efficiency` 高且 `fake_track_rate` 低，说明“找轨 + 选轨”平衡较好。
- `assigned_hit_purity` 反映命中关联质量，通常应接近 1。
- `radius_rmse_mm` 与 `center_*_rmse_mm` 体现参数精度，受噪声和假命中数量显著影响。

## R15

复杂度分析（设层数 `L`，每层可用命中 `H`，重建轨迹数 `K`）：
- 种子枚举（3 层三重组合）：`O(H^3)`。
- 每个种子打分需扫层命中：`O(L*H)`。
- 每轮重建近似：`O(H^3 * L * H)`，共 `K` 轮。

教学参数下规模较小（几十命中量级），可在秒级完成。

## R16

局限与失效模式：
- 真实探测器中轨迹是 3D 螺旋并含材料效应，本实现未覆盖。
- 贪心“先到先得”在高密度事件中可能导致次优全局解。
- 仅用几何残差，不含完整卡尔曼滤波状态传播和协方差更新。

## R17

可扩展方向：
- 升级到 3D 螺旋参数 + 卡尔曼滤波平滑。
- 增加时间信息做 4D track finding，降低 pile-up 混叠。
- 把贪心选轨替换为全局图优化/网络流。
- 使用学习型打分器（GNN/Transformer）提升种子筛选效果。

## R18

源码级算法流程拆解（对应 `demo.py`，非黑盒）：
1. `simulate_event` 按物理参数生成真轨迹理想命中，并叠加噪声与假命中。
2. `build_hit_indices` 将命中组织为 `hit_id -> hit` 与 `layer -> hit_ids`，供快速检索。
3. 在 `reconstruct_tracks` 中枚举三层命中组合，调用 `circle_from_three_points` 生成圆种子。
4. `assign_hits_by_layer` 对每个种子执行分层最近残差关联，按“命中层数优先、残差次优”评分。
5. 对最佳候选先用 `refine_circle_scipy`（`least_squares + soft_l1`）做鲁棒非线性拟合。
6. 再用 `refine_circle_torch`（pseudo-Huber + Adam）做可微小步精修，得到最终轨迹参数。
7. 采用贪心策略移除已使用命中并重复步骤 3-6，得到多条重建轨迹。
8. `match_reco_to_truth` 构建成本矩阵并用 Hungarian (`linear_sum_assignment`) 匹配真值，`evaluate` 计算效率、假轨率与 RMSE。

以上 8 步把第三方库调用拆到可追踪的几何与优化步骤，避免“一行 API 黑盒化”。
