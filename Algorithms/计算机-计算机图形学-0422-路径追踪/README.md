# 路径追踪

- UID: `CS-0263`
- 学科: `计算机`
- 分类: `计算机图形学`
- 源序号: `422`
- 目标目录: `Algorithms/计算机-计算机图形学-0422-路径追踪`

## R01

路径追踪（Path Tracing）是对渲染方程进行蒙特卡洛估计的光线追踪方法：
1. 从相机向场景发射主光线；
2. 命中表面后随机采样新的反射方向，形成路径；
3. 累积路径上的自发光与环境光贡献；
4. 通过多次采样像素平均得到近似全局光照。

本条目给出一个教学型、可运行 MVP：漫反射材质 + 余弦加权半球采样 + 多次反弹 + Russian Roulette 终止。

## R02

问题定义（MVP 范围）：
1. 输入（代码中固定，无交互）：
   - 相机：分辨率、FOV、相机原点；
   - 场景：两个漫反射球、一个地面平面、一个发光球；
   - 采样参数：每像素采样数 `SPP`、最大反弹深度、Russian Roulette 起始深度。
2. 输出：
   - 图像文件：`render_path_tracing.ppm`；
   - 控制台统计：主光线、路径段、求交测试等；
   - 断言通过后输出 `All checks passed.`。

## R03

核心数学关系：
1. 光线方程：`p(t) = o + t d, t > 0`。
2. 球体求交：解 `||o + td - c||^2 = r^2` 的二次方程，取最小正根。
3. 平面求交：`t = dot(p0 - o, n) / dot(d, n)`，并要求 `t > eps`。
4. 漫反射 BRDF：`f_r = albedo / pi`。
5. 余弦加权采样 PDF：`p(omega) = cos(theta) / pi`。
6. 因 `f_r * cos / p = albedo`，每次反弹可直接做 `throughput *= albedo`（本实现显式采用该化简）。
7. Russian Roulette：路径继续概率 `q = min(0.95, max(throughput))`，继续时 `throughput /= q` 保持无偏。

## R04

`demo.py` 执行流程：
1. 构造固定场景（3 球 + 1 平面，其中 1 球为发光体）；
2. 初始化随机数种子，确保结果可复现；
3. 对每个像素做 `SPP` 次抖动采样；
4. 每条主光线进入 `trace_path`，按反弹深度循环；
5. 每次命中将材质发光累积到 `radiance`；
6. 使用余弦加权半球采样生成下一跳方向；
7. 执行 Russian Roulette 进行随机截断与权重校正；
8. 路径逃逸时采样天空背景；
9. 对线性 HDR 结果做 Reinhard + Gamma 映射并写出 PPM。

## R05

关键数据结构：
1. `Material`：`albedo`（反照率）、`emission`（自发光）；
2. `Sphere` / `Plane`：几何体与 `intersect` 求交函数；
3. `HitRecord`：命中距离、位置、法线、材质、对象名；
4. `PathTracer.stats`：
   - `camera_rays`：主光线数量；
   - `path_segments`：路径段数量（循环迭代次数）；
   - `intersection_tests`：总求交测试次数；
   - `surface_hits`：命中表面次数；
   - `escaped_rays`：逃逸到背景次数；
   - `russian_roulette_terminated`：被轮盘赌终止次数。

## R06

正确性直觉：
1. 最近正距离交点保证“当前可见”几何正确；
2. 余弦加权与 Lambert BRDF 匹配，可把估计器权重化简为 `albedo`；
3. 多次反弹让间接光照自然出现，而非仅局部光照；
4. Russian Roulette 在保无偏前提下降低深路径计算成本；
5. `EPSILON` 偏移可抑制自相交伪影。

## R07

复杂度分析（像素数 `P=W*H`，样本数 `S`，平均路径长度 `B`，物体数 `O`）：
1. 时间复杂度：`O(P * S * B * O)`。
2. 空间复杂度：`O(P)`（图像缓存）+ `O(1)`（每条路径临时变量）。

在本 MVP 中，`B` 由 `max_depth` 与 Russian Roulette 共同控制。

## R08

边界与鲁棒性处理：
1. 使用 `EPSILON` 过滤近零命中，避免阴影痤疮式自相交；
2. 平面求交时 `|dot(d,n)|` 过小直接判平行；
3. 对零长度向量归一化做保护；
4. `throughput` 过小触发高概率终止，避免无效深反弹；
5. 色调映射后再做 `[0,1]` 裁剪，保证输出像素合法。

## R09

MVP 取舍：
1. 仅实现漫反射路径追踪，不含镜面/折射 BSDF；
2. 不引入 BVH/KD-Tree，采用线性遍历，强调算法透明度；
3. 不依赖大型渲染框架，核心依赖仅 `numpy + pandas`；
4. 使用固定随机种子，便于回归测试和文档复现实验。

## R10

`demo.py` 主要函数职责：
1. `normalize`：向量归一化；
2. `sample_cosine_hemisphere`：在命中法线半球上做余弦加权采样；
3. `Sphere.intersect / Plane.intersect`：基础几何求交；
4. `PathTracer.find_nearest_hit`：最近命中搜索与计数；
5. `PathTracer.trace_path`：单路径多反弹积分主逻辑；
6. `PathTracer.render`：逐像素、逐样本渲染；
7. `tonemap_and_gamma`：显示域转换；
8. `save_ppm`：输出图像；
9. `run_sanity_checks`：范围和统计断言。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机图形学-0422-路径追踪
uv run python demo.py
```

脚本无需输入参数，运行后自动生成图像和统计信息。

## R12

输出说明：
1. 图像：`render_path_tracing.ppm`（已做色调映射和 Gamma 校正）；
2. 统计字段：
   - `camera_rays`：总主光线数，理论值约为 `W*H*SPP`；
   - `path_segments`：总路径段数，反映平均反弹深度；
   - `intersection_tests`：几何求交调用规模；
   - `surface_hits`：表面命中次数；
   - `escaped_rays`：路径到达环境光次数；
   - `russian_roulette_terminated`：随机提前终止次数；
3. `Image summary`：三个探针像素 + 全局均值亮度。

## R13

最小验证清单：
1. `README.md` 与 `demo.py` 不含任何未替换占位符；
2. `uv run python demo.py` 能一次运行完成；
3. 输出图像像素全部落在 `[0,1]`；
4. `camera_rays / intersection_tests / surface_hits` 等关键统计量均大于 0；
5. 控制台出现 `All checks passed.`。

## R14

当前固定实验配置：
1. 分辨率：`200 x 132`；
2. 视场角：`56°`；
3. 每像素样本：`24`；
4. 最大深度：`6`；
5. Russian Roulette 起始深度：`3`；
6. 场景：红球 + 蓝球 + 发光球 + 地面平面。

该配置可在较短时间内得到有全局照明特征的稳定结果。

## R15

与相邻方法对比：
1. 相比基础光线追踪：路径追踪显式估计间接光照，不仅有直接光与硬阴影；
2. 相比 Whitted：路径追踪更通用（可覆盖复杂光传输），但噪声更高、收敛更慢；
3. 相比光子映射：路径追踪实现更统一简洁，但对焦散等场景可能更难高效收敛。

## R16

适用场景：
1. 图形学课程中的蒙特卡洛渲染入门；
2. 验证“采样-积分-无偏估计”链路；
3. 后续扩展到 MIS、NEE、BVH 的基础代码框架。

不适用场景：
1. 生产级电影渲染（需更完善的材质、采样器、加速结构）；
2. 实时高帧率渲染（需 GPU 并行与专门工程优化）。

## R17

可扩展方向：
1. 加入显式直接光采样（Next Event Estimation）降低噪声；
2. 引入 MIS（多重重要性采样）提升收敛效率；
3. 支持镜面/折射材质与 Fresnel；
4. 构建 BVH 以降低 `O` 对时间复杂度的影响；
5. 迁移到 PyTorch 张量并行或 CUDA 后端。

## R18

`demo.py` 源码级算法流（9 步，非黑箱）：
1. `main` 固定分辨率、FOV、`SPP`、随机种子，并调用 `build_default_scene` 构建场景对象。  
2. 初始化 `PathTracer`，设置 `max_depth=6` 与 `rr_start_depth=3`。  
3. `render` 遍历像素；每个像素做 `SPP` 次抖动采样，生成相机主光线方向。  
4. 每条主光线进入 `trace_path`，初始化 `radiance=0` 与 `throughput=1`。  
5. 每个 bounce 调用 `find_nearest_hit`，逐个执行 `Sphere.intersect` / `Plane.intersect`，取最近正交点。  
6. 若未命中物体，则把 `throughput * background(ray_dir)` 累积到 `radiance` 并结束路径。  
7. 若命中物体，则先累积 `throughput * emission`，再用 `sample_cosine_hemisphere` 采样下一跳方向，并执行 `throughput *= albedo`。  
8. 达到轮盘赌深度后，用 `survive_prob=min(0.95,max(throughput))` 决定是否终止；若继续则 `throughput /= survive_prob` 做无偏校正。  
9. 所有样本平均后得到线性 HDR 图，`tonemap_and_gamma` 做显示映射，`save_ppm` 输出图像，最后 `run_sanity_checks` 验证数值与统计。  
