# Gouraud着色

- UID: `CS-0269`
- 学科: `计算机`
- 分类: `计算机图形学`
- 源序号: `428`
- 目标目录: `Algorithms/计算机-计算机图形学-0428-Gouraud着色`

## R01

Gouraud 着色（Gouraud Shading）是经典的逐顶点光照、逐像素插值着色方法：
- 每个顶点先计算一次光照强度；
- 三角形内部对顶点强度做重心插值；
- 视觉上比 Flat 着色更平滑，计算量又低于逐像素 Phong 着色。

本条目给出一个可运行的最小软件渲染 MVP：
- 使用 `numpy` 手写 3D 变换、透视投影、光栅化和 z-buffer；
- 使用 Lambert 漫反射在顶点处计算强度；
- 输出 `PPM` 图像文件，不依赖交互输入。

## R02

本实现的问题定义：
- 输入：
  - 固定网格（立方体，8 个顶点，12 个三角形）；
  - 相机参数（`eye/target/up`）；
  - 投影参数（`fov/aspect/near/far`）；
  - 光照参数（方向光向量，`ambient` 与 `diffuse`）。
- 输出：
  - 渲染图像 `gouraud_shading_output.ppm`；
  - 控制台统计信息：三角形数、像素着色数、顶点强度范围、平均像素强度、深度范围。

脚本参数全部写在 `demo.py` 内部，无需命令行输入。

## R03

核心数学关系：

1. 顶点法线估计（按邻接面平均）：
   - 面法线 `n_f = normalize((b-a) × (c-a))`；
   - 顶点法线 `n_v = normalize(sum(n_f))`（对连接到该顶点的面求和再归一化）。
2. 顶点 Lambert 光照：
   - `I_v = clamp(ambient + diffuse * max(0, n_v · L), 0, 1)`。
3. 重心插值（像素级强度）：
   - 对像素重心坐标 `w0,w1,w2`，`I_p = clamp(w0*I0 + w1*I1 + w2*I2, 0, 1)`。
4. 深度测试：
   - `z_p = w0*z0 + w1*z1 + w2*z2`，若 `z_p < z_buffer[x,y]` 则更新像素。

## R04

高层流程：

1. 构建立方体网格（顶点与三角形索引）。
2. 构建模型、观察、透视投影矩阵（`M/V/P`）。
3. 将顶点变换到 view、ndc、screen 空间。
4. 依据三角形邻接关系估计每个顶点的平滑法线。
5. 在 view 空间按 Lambert 公式计算每个顶点强度。
6. 光栅化每个三角形，在像素内插值得到强度和深度。
7. 执行 z-buffer 深度比较并写入颜色缓冲。
8. 输出 PPM 文件并打印统计结果。

## R05

关键数据结构：
- `vertices: ndarray[8,3]`：模型顶点。
- `triangles: ndarray[12,3]`：三角形索引。
- `vertex_normals_view: ndarray[8,3]`：顶点法线（view 空间）。
- `vertex_intensity: ndarray[8]`：顶点光照强度。
- `image: ndarray[H,W,3]`：颜色缓冲（`uint8`）。
- `depth_buffer: ndarray[H,W]`：深度缓冲（初值 `inf`）。
- `RenderStats`：渲染统计结构体。

## R06

正确性要点：
- 顶点法线必须来源于邻接面平均，才能体现 Gouraud 的平滑特性。
- 光照在顶点计算，像素阶段只做插值，不重复法线光照计算。
- 使用重心坐标确保三角形内部插值连续且边界一致。
- 深度值同样由重心插值得到，并以 z-buffer 保证遮挡关系正确。
- 以“至少有像素被着色”为运行后置条件，避免空渲染假成功。

## R07

复杂度分析：

设三角形数为 `F`，顶点数为 `V`，第 `i` 个三角形包围盒像素数为 `A_i`。
- 顶点法线累计：`O(F)`。
- 顶点光照：`O(V)`。
- 光栅化：`O(sum(A_i))`，最坏近似 `O(F*W*H)`。

空间复杂度：
- 颜色缓冲与深度缓冲：`O(W*H)`；
- 网格、法线、强度：`O(V+F)`。

## R08

边界与异常处理：
- `normalize` 处理零向量或非有限范数会抛 `ValueError`。
- `make_perspective` 对非法 `fov/aspect/near/far` 抛 `ValueError`。
- 齐次坐标 `w` 过小会抛 `RuntimeError`，避免除零。
- 分辨率 `width/height <= 0` 抛 `ValueError`。
- 退化三角形（面积接近 0）在光栅化时跳过。
- 若最终 `pixels_shaded == 0`，抛 `RuntimeError`。

## R09

MVP 范围与取舍：
- 仅实现 Gouraud（不实现 Phong 高光、阴影、纹理映射）。
- 光源为单一方向光，便于验证插值机制。
- 不实现显式裁剪器（clipper），通过场景参数保证模型可见。
- 输出格式选 `PPM`，减少第三方图像库依赖，保证可复现。

## R10

`demo.py` 关键函数：
- `make_rotation_x/make_rotation_y`：模型旋转矩阵。
- `make_look_at`：观察矩阵。
- `make_perspective`：透视投影矩阵。
- `apply_transform/project_to_ndc/ndc_to_screen`：坐标变换链。
- `compute_vertex_normals`：邻接面平均法估计顶点法线。
- `lambert_vertex_intensities`：计算顶点光照强度。
- `rasterize_triangle_gouraud`：像素内强度/深度插值 + z-test。
- `render_gouraud_shaded`：组织全流程渲染。
- `save_ppm`：保存图像。
- `main`：执行并输出统计。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机图形学-0428-Gouraud着色
uv run python demo.py
```

该命令不需要任何交互输入。

## R12

输出说明：
- 图像文件：
  - `gouraud_shading_output.ppm`（P6 二进制 PPM）。
- 终端输出字段：
  - `triangles_total`：网格三角形总数；
  - `triangles_rasterized`：实际有像素写入的三角形数；
  - `pixels_shaded`：通过 z-test 写入的像素总数；
  - `mean_pixel_intensity`：通过写入像素的平均插值强度；
  - `vertex_intensity_range`：顶点强度最小值/最大值；
  - `depth_range`：有效深度最小值/最大值。

## R13

建议最小测试集：
- 基线测试：直接运行并确认生成 `gouraud_shading_output.ppm`。
- 平滑性测试：将 `vertex_normals_view` 改为面法线重复值，对比图像会更“分面”。
- 参数测试：调整 `light_dir_view`，应观察到可解释的亮暗变化。
- 异常测试：手动设 `width=0` 或 `fov_y_deg=180`，应触发明确异常。

## R14

可调参数：
- 分辨率：`width`, `height`。
- 模型姿态：`make_rotation_x/y` 角度。
- 相机：`eye`, `target`, `up`。
- 投影：`fov_y_deg`, `z_near`, `z_far`。
- 光照：`light_dir_view`, `ambient`, `diffuse`。
- 物体基色：`base_rgb`。

调参建议：
- 画面过暗：提高 `ambient` 或调整光方向。
- 平滑感不足：检查顶点法线是否正确累计并归一化。
- 遮挡异常：检查 `depth` 映射区间与 z-test 逻辑。

## R15

与相关着色模型对比：
- Flat 着色：按面计算一次光照，边界清晰，计算最省。
- Gouraud 着色：按顶点计算光照，像素插值，平滑且开销低。
- Phong 着色：像素级法线插值并逐像素光照，细节最好但成本更高。

Gouraud 的典型局限：
- 镜面高光若不落在顶点附近，可能在像素插值中被“抹平”。

## R16

典型应用场景：
- 软件光栅化教学中的“平滑着色”阶段。
- 资源受限场景下的实时渲染基线。
- 对比 Flat/Phong 的渲染质量与性能实验。
- 作为后续加入纹理、镜面项、阴影的中间里程碑。

## R17

可扩展方向：
1. 实现 Phong 着色并与 Gouraud 做质量/耗时对比。
2. 增加 Blinn-Phong 镜面项，展示 Gouraud 高光丢失问题。
3. 加入背面剔除、裁剪与透视正确插值（`1/w`）。
4. 增加多光源与点光源衰减模型。
5. 增加输出深度图/法线图用于调试和回归测试。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 设定分辨率并调用 `render_gouraud_shaded`，最后写出 `gouraud_shading_output.ppm`。
2. `render_gouraud_shaded` 用 `generate_cube_mesh` 构造立方体，并用 `make_rotation_x/y`、`make_look_at`、`make_perspective` 构建 `M/V/P`。
3. 顶点经 `apply_transform`、`project_to_ndc`、`ndc_to_screen` 生成屏幕空间坐标与深度。
4. 调用 `compute_vertex_normals`：遍历每个三角形计算面法线并累加到邻接顶点，再逐顶点归一化。
5. 调用 `lambert_vertex_intensities`：把顶点法线与 `light_dir_view` 点乘，得到每个顶点的 `I_v`。
6. 遍历三角形并调用 `rasterize_triangle_gouraud`：先用边函数做 inside-test，再用重心坐标插值深度和强度。
7. 对每个覆盖像素执行 z-test；若更近则写入 `depth_buffer` 和 `image`，颜色为 `base_rgb * I_p`。
8. 渲染结束后统计像素强度与深度范围，返回 `RenderStats` 并在 `main` 中打印可审计结果。
