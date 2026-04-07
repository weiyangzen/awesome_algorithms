# 各向异性过滤

- UID: `CS-0273`
- 学科: `计算机`
- 分类: `计算机图形学`
- 源序号: `432`
- 目标目录: `Algorithms/计算机-计算机图形学-0432-各向异性过滤`

## R01

各向异性过滤（Anisotropic Filtering）用于纹理采样时的抗锯齿与细节保留。相比仅按单一尺度选 mip 级别的各向同性过滤，它会识别像素在纹理空间的拉伸方向（主轴）与压缩方向（次轴），然后沿主轴做多点采样、沿次轴做 mip 选择，从而减少斜视角下的模糊与闪烁。

## R02

问题背景：当一个屏幕像素对应到纹理空间时，足迹通常不是圆而是椭圆。传统双线性/三线性采样默认“各方向同等缩放”，会在强透视时产生两类问题：
1. 沿拉伸方向细节被过度平均，图案发糊。
2. 若 mip 级别选取偏小，会出现闪烁、摩尔纹和走样。

## R03

本任务的 MVP 输入输出约定：
- 输入：程序生成的高频纹理、构造的透视 UV 映射、`max_aniso` 采样上限。
- 输出：
  1. `output_isotropic.ppm`（各向同性近似结果）
  2. `output_anisotropic.ppm`（各向异性结果）
  3. `output_reference.ppm`（更高采样上限的近似参考）
  4. `output_absdiff.ppm`（前两者差异热度图）
  5. 终端打印 MSE/PSNR 与误差改进倍数。

## R04

核心思想：
1. 用雅可比矩阵 `J = [[du/dx, du/dy], [dv/dx, dv/dy]]` 描述屏幕像素到 UV 的局部线性映射。
2. 对 `J` 做 SVD，得到主轴/次轴尺度 `sigma_max/sigma_min`。
3. 用 `sigma_min` 决定 LOD（避免把次轴细节过早抹掉）。
4. 用 `sigma_max / sigma_min` 决定采样点数（上限 `max_aniso`）。
5. 沿主轴方向分布采样并平均，近似椭圆足迹积分。

## R05

关键数学关系：
- 各向同性 LOD 估计：
  `lod_iso = log2(max(rho * tex_size, 1))`
  其中 `rho = max(||dUV/dx||, ||dUV/dy||)`。
- 各向异性比率：
  `aniso_ratio = sigma_max / sigma_min`。
- 各向异性 LOD 估计：
  `lod_minor = log2(max(sigma_min * tex_size, 1))`。
- 采样数：
  `taps = ceil(aniso_ratio)`，并截断到 `[1, max_aniso]`。

## R06

伪代码（与 `demo.py` 对应）：

```text
build_mipmaps(texture)
compute u_map, v_map
compute du/dx, dv/dx, du/dy, dv/dy
for each pixel:
    J <- [[du/dx, du/dy], [dv/dx, dv/dy]]
    [_, s, vh] <- svd(J)
    sigma_max <- s[0], sigma_min <- s[1]
    lod <- log2(sigma_min * tex_size)
    taps <- clamp(ceil(sigma_max/sigma_min), 1, max_aniso)
    major_dir <- normalize(J @ vh[0])
    color <- average_{k in taps}(trilinear(u + offset_k, v + offset_k, lod))
```

## R07

正确性直觉：
- 若局部缩放几乎各向同性（`sigma_max ≈ sigma_min`），`taps` 接近 1，算法退化到普通三线性采样。
- 若局部拉伸明显（`sigma_max >> sigma_min`），算法增加主轴方向采样密度，避免单点采样忽略长条足迹导致的 aliasing。
- 因为 LOD 由次轴驱动，能保留主轴可见细节而不过度模糊。

## R08

复杂度分析：
- 设输出分辨率为 `H*W`，平均采样点数为 `K`。
- 时间复杂度约 `O(H*W*K)`；其中 `K` 由局部各向异性比率决定。
- 空间复杂度约 `O(H*W + mip_total)`，`mip_total` 约为基纹理大小的 `4/3` 倍量级。

## R09

数值与边界处理：
- 对 `sigma_min` 与梯度长度使用 `1e-8` 下界，避免除零与 `log2(0)`。
- LOD 被裁剪到 mip 范围 `[0, level_max]`。
- UV 采用 repeat wrap（取模），避免越界访问。
- 主轴向量退化时回退到 `[1, 0]`。

## R10

工程实现要点：
- 纹理与 mip 统一用 `float64` 计算，减少累计误差。
- 图像输出用 PPM，避免额外依赖（例如 Pillow/matplotlib）。
- 演示里把“高 `max_aniso` 结果”作为近似参考，便于自动化比较不同配置。

## R11

`demo.py` 的模块结构：
1. 纹理构造：`make_procedural_texture`
2. mip 链：`downsample_2x`、`build_mipmaps`
3. 采样器：`bilinear_sample`、`trilinear_sample`
4. 几何场：`make_uv_map`、`jacobian_from_uv`
5. 渲染器：`render_isotropic`、`render_anisotropic`
6. 评估与导出：`psnr`、`write_ppm`、`main`

## R12

运行方式（无需交互输入）：

```bash
uv run python demo.py
```

执行后在当前目录生成 `output_*.ppm` 文件，并在终端输出误差指标。

## R13

结果解读建议：
- 观察 `output_isotropic.ppm`：在透视压缩严重区域会出现更多模糊/闪烁风险。
- 观察 `output_anisotropic.ppm`：斜向纹理线条更稳定，细节更连贯。
- 观察 `output_absdiff.ppm`：亮区表示两种方法差异显著，通常对应高各向异性区域。

## R14

与常见过滤方法对比：
- 双线性：只在单一 mip 层内插值，抗锯齿能力最弱。
- 三线性：在相邻 mip 层插值，缓解层级跳变，但仍假设各方向等价。
- 各向异性：引入方向性采样，处理“长条形像素足迹”更有效。

## R15

参数敏感性：
- `max_aniso` 越大，质量通常越高，但性能线性下降。
- 纹理频率越高，三线性与各向异性差距越明显。
- UV 透视压缩越强，算法收益越大。

## R16

典型应用：
- 3D 场景中道路、地面砖块、墙面贴图等斜视角纹理。
- 大规模室外场景（地形纹理）与高速相机运动场景。
- 任何需要在远处保留纹理方向细节的实时渲染管线。

## R17

局限与可扩展方向：
- 本 MVP 使用 CPU 循环实现，性能不面向实时渲染。
- 参考图使用同类算法高采样近似，不是严格连续积分真值。
- 可扩展到 EWA（Elliptical Weighted Average）核、SIMD/GPU 实现、自适应采样策略与缓存优化。

## R18

源码级算法流程（8 步）：
1. 生成高频程序纹理并构建 mipmap 金字塔。
2. 构造屏幕到 UV 的透视映射 `u_map/v_map`。
3. 通过 `np.gradient` 计算 `du/dx,dv/dx,du/dy,dv/dy`。
4. 逐像素构建雅可比 `J`，用 SVD 得到 `sigma_max/sigma_min` 与主方向。
5. 用 `sigma_min` 计算 mip LOD（次轴控制模糊程度）。
6. 用 `ceil(sigma_max/sigma_min)` 得到采样数并限制在 `max_aniso`。
7. 沿主轴在 `[-0.5,0.5]` 区间均匀分布采样点，逐点执行三线性采样并平均。
8. 输出各向同性图、各向异性图、近似参考图和差异图，并计算 MSE/PSNR。
