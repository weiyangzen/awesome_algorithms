# 光子映射

- UID: `CS-0265`
- 学科: `计算机`
- 分类: `计算机图形学`
- 源序号: `424`
- 目标目录: `Algorithms/计算机-计算机图形学-0424-光子映射`

## R01

光子映射（Photon Mapping）是 Jensen 提出的双阶段全局光照算法：

1. 光子追踪阶段：从光源发射大量光子，沿路径与场景交互，并在表面记录光子信息；
2. 渲染估计阶段：在可见点附近收集光子，用密度估计近似入射辐照度，再转成出射辐亮度。

它擅长处理间接光与焦散等“路径追踪纯随机采样难以稳定命中”的现象。

## R02

本目录 MVP 使用“全局漫反射光子图”的最小可运行版本，场景由 3 个几何体组成：

- 地板平面（`y=0`，漫反射）；
- 背墙平面（`z=9`，漫反射）；
- 一个漫反射球体。

光源是点光源，位于场景上方，朝下半球发射光子。

## R03

单个光子在本实现中的状态：

- `position`：命中点位置；
- `incoming`：命中处入射方向（朝向表面）；
- `power`：RGB 功率（通量份额）；
- `normal`：命中点法线。

`demo.py` 里用 `PhotonMap` 数据类把这些数组批量保存，后续供 KD-Tree 查询。

## R04

漫反射表面的路径推进使用俄罗斯轮盘赌（Russian Roulette）：

- 命中后先存储光子；
- 以 `rr_prob = clip(mean(albedo), 0.10, 0.95)` 决定是否继续；
- 若继续，更新 `power = power * albedo / rr_prob` 保持无偏估计；
- 新方向采用“余弦加权半球采样”。

该流程避免固定深度造成的系统性偏差，同时控制路径数量。

## R05

查询点辐照度估计使用经典近邻密度形式：

- 收集查询点周围 `k` 个光子（并可加最大半径裁剪）；
- 估计

```text
E(x) ≈ (1 / (π r^2)) * Σ Φ_i * max(0, n · ω_i)
```

其中 `Φ_i` 为光子功率，`ω_i` 为入射方向，`n` 为查询点法线，`r` 为本次收集半径。

## R06

`demo.py` 的主要函数划分：

- 几何/采样：`normalize`、`build_onb`、`sample_cosine_hemisphere`；
- 求交：`intersect_floor`、`intersect_back_wall`、`intersect_sphere`、`intersect_scene`；
- 光子追踪：`emit_and_trace_photons`；
- 收集估计：`estimate_irradiance_knn`（`scipy.spatial.cKDTree`）；
- 查询网格：`build_floor_queries`；
- 入口：`main`（运行、断言、报表）。

## R07

实现中的关键不变量：

1. 方向向量经过归一化后用于采样与交点计算；
2. 求交只接受 `t > EPS` 以避免自相交；
3. 光子功率在每次幸存反射后按 `albedo/rr_prob` 重标定；
4. 辐照度结果应非负且有限；
5. 查询点大部分应能找到有效邻居（脚本要求比例至少 0.90）。

## R08

复杂度（`P` 为存储光子数，`Q` 为查询点数，`k` 为收集邻居数）：

- 构建光子图：约 `O(P)`（本 MVP 为逐光子追踪，常数由场景求交和反弹次数决定）；
- KD-Tree 构建：`O(P log P)`；
- 每个查询点 kNN：平均 `O(log P + k)`；
- 总收集：`O(Q(log P + k))`。

## R09

第三方组件说明（非黑盒）：

- 使用 `cKDTree` 仅负责“空间近邻索引”；
- 光子状态定义、功率传播、余弦项、`πr²` 归一化全部在本地手写；
- 收集阶段仍显式遍历每个查询点并计算 `max(0, n·ω)` 与通量求和。

因此核心光子映射逻辑可追溯，不依赖单行 API 隐藏算法本体。

## R10

当前实验参数（`main()`）：

- `n_emit = 26000`，`max_bounces = 4`；
- `gather_k = 60`，`max_radius = 1.35`；
- 点光源：`light_pos=[0.0, 5.8, 1.8]`，`light_power_rgb=[220,220,220]`；
- 查询网格：地板上 `28 x 22 = 616` 个点。

随机种子固定为 `2026`，便于复现实验输出。

## R11

运行方式（无交互）：

```bash
cd Algorithms/计算机-计算机图形学-0424-光子映射
uv run python demo.py
```

脚本会直接打印样本查询表和汇总指标表。

## R12

输出指标解释：

- `photons_emitted` / `photons_stored` / `storage_ratio`：发射与落库统计；
- `trace_ms` / `gather_ms`：追踪与收集耗时；
- `nonzero_neighbor_ratio`：有可用邻居的查询点比例；
- `avg_neighbors_used` / `avg_gather_radius`：收集有效邻居和半径统计；
- `mean_irradiance_rgb` / `max_irradiance_rgb`：辐照度统计；
- `mean_radiance` / `max_radiance`：按 Lambert 模型转换后的辐亮度统计。

## R13

脚本内置质量门禁：

1. 存储光子数量不得低于发射数的 20%；
2. 辐照度必须是有限值（无 `NaN/Inf`）；
3. 辐照度不能出现显著负值；
4. 有效邻居覆盖率必须达到 90% 以上。

若不满足，程序直接抛出 `AssertionError`。

## R14

边界与异常处理：

- `n_emit <= 0` 或 `max_bounces <= 0` 会抛 `ValueError`；
- `estimate_irradiance_knn` 会检查 `points/normals` 形状一致性；
- 空光子图会抛错，防止“无数据但继续计算”；
- `max_radius <= 0` 会抛 `ValueError`。

## R15

光子映射的典型应用：

- 离线渲染中的间接光近似；
- 焦散（玻璃/水面）能量集中区域估计；
- 室内全局照明预计算（如全局光子图 + 局部收集）；
- 与路径追踪混合的双向/混合管线中的辅助缓存。

## R16

本 MVP 的限制：

- 仅实现漫反射全局图，不含独立焦散图；
- 场景几何是手写解析求交，未接网格 BVH；
- 收集核使用简单圆盘密度估计，未做自适应核或渐进光子映射；
- 结果以数值报表展示，未输出图像缓冲。

## R17

可扩展方向：

1. 增加镜面/折射材质，拆分 caustic photon map；
2. 用 BVH/Embree 加速复杂网格求交；
3. 引入渐进光子映射（PPM/SPPM）降低偏差；
4. 输出二维图像并与路径追踪结果做误差对比；
5. 将追踪与收集改造为并行/GPU 版本。

## R18

源码级算法链路（9 步）：

1. `main()` 固定随机种子与参数，设置点光源位置、总功率、追踪深度和 kNN 收集参数。  
2. `emit_and_trace_photons()` 对每个光子执行初始化：从光源位置出发，按 `sample_cosine_hemisphere(down)` 采样下半球方向，并把光源总功率均分到单光子。  
3. 每次反弹先调用 `intersect_scene()`，后者分别测试地板、背墙、球体求交并返回最小正 `t` 命中。  
4. 命中后把 `position / incoming=-dir / power / normal` 写入缓冲；然后用 `rr_prob=clip(mean(albedo),0.10,0.95)` 做俄罗斯轮盘赌，决定终止或继续。  
5. 若继续，则执行 `power = power * albedo / rr_prob` 做无偏能量重标定，再按命中法线进行余弦半球采样得到新方向，并沿法线偏移一个 `EPS` 防止自相交。  
6. 全部光子追踪完成后，构建 `PhotonMap` 数组，并在 `estimate_irradiance_knn()` 里用 `cKDTree(photon_positions)` 建立空间索引。  
7. 对每个查询点，`tree.query(..., k)` 取得近邻候选，再应用 `dist <= max_radius` 的二次过滤，确定有效邻居集合与本点收集半径 `r`。  
8. 对有效邻居逐个计算余弦项 `max(0, n·ω_i)`，累加 `power_i * cos_i`，最后除以 `πr²` 得到 RGB 辐照度；再按 Lambert 关系 `L = albedo/pi * E` 得到辐亮度。  
9. `main()` 汇总样本点表与指标表，并执行四个断言门禁（落库率、有限性、非负性、覆盖率）确保 MVP 可运行且结果合理。
