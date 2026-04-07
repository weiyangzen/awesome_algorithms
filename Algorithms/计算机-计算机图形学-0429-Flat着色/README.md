# Flat着色

- UID: `CS-0270`
- 学科: `计算机`
- 分类: `计算机图形学`
- 源序号: `429`
- 目标目录: `Algorithms/计算机-计算机图形学-0429-Flat着色`

## R01

Flat 着色（Flat Shading）是最基础的多边形光照模型之一：
- 以三角形面片为单位计算一次光照强度；
- 整个三角形内部使用同一颜色，不做顶点到像素的平滑插值；
- 视觉上会保留明显的“棱面感”，适合低成本实时渲染和教学演示。

本条目的 MVP 提供一个最小可运行的软件渲染管线：
- 使用 `numpy` 手写 3D 变换、透视投影、三角形光栅化和 z-buffer；
- 使用 Lambert 漫反射做面级别光照；
- 输出一张 `PPM` 图像文件，无需交互输入。

## R02

问题定义（本实现）：
- 输入：
  - 一个固定网格（立方体，8 顶点、12 三角形）；
  - 相机参数（`eye/target/up`）；
  - 投影参数（`fov/aspect/near/far`）；
  - 光照参数（方向光方向、`ambient`、`diffuse`）。
- 输出：
  - 渲染图像 `flat_shading_output.ppm`；
  - 控制台统计信息：总三角形数、成功光栅化三角形数、着色像素数、平均面亮度、深度范围。

脚本全部参数在 `demo.py` 内固定，不读取命令行参数，也不要求人工输入。

## R03

核心数学基础：

1. 面法线：
   - 对三角形顶点 `a,b,c`，法线方向 `n = normalize((b-a) x (c-a))`。
2. Flat 光照（Lambert）：
   - 设方向光向量 `L`（单位向量），则漫反射项为 `max(0, n·L)`；
   - 面强度 `I = ambient + diffuse * max(0, n·L)`。
3. 透视投影：
   - 顶点按 `clip = P * V * M * p_h`；
   - 再做齐次除法 `ndc = clip.xyz / clip.w`。
4. 光栅化与可见性：
   - 用边函数/重心坐标判定像素是否在三角形内部；
   - 使用 z-buffer，深度更小的片元覆盖更远片元。

## R04

算法流程（高层）：

1. 构建立方体顶点和三角形索引。  
2. 构建 `Model/View/Projection` 矩阵。  
3. 将顶点分别变换到 world/view/ndc/screen 空间。  
4. 初始化颜色缓冲和深度缓冲。  
5. 遍历每个三角形，计算面法线与 Flat 光照颜色。  
6. 对三角形做包围盒扫描，使用重心坐标判定像素覆盖。  
7. 通过 z-buffer 决定是否写入该像素颜色。  
8. 输出 PPM 文件并打印渲染统计。

## R05

核心数据结构：
- `vertices: ndarray[8,3]`：立方体顶点坐标。  
- `triangles: ndarray[12,3]`：每个三角形的顶点索引。  
- `image: ndarray[H,W,3]`：RGB 颜色缓冲（`uint8`）。  
- `depth_buffer: ndarray[H,W]`：深度缓冲（初值 `inf`）。  
- `RenderStats`：汇总渲染统计（面数、像素数、亮度、深度范围）。

## R06

正确性要点：
- 法线方向由叉乘定义，保证每个三角形获得一致的面方向。  
- Flat 着色按“每面一次光照”原则，仅用面法线参与亮度计算。  
- 像素覆盖使用重心坐标符号一致性判断，避免漏填与越界。  
- 深度插值后进行 z-test，确保前景三角形遮挡背景三角形。  
- 输出前检查至少有像素被着色，防止“空渲染”误通过。

## R07

复杂度分析：

记：
- 三角形数为 `F`；
- 第 `i` 个三角形包围盒像素数为 `A_i`。

则：
- 逐面法线和光照计算为 `O(F)`；
- 光栅化总复杂度约为 `O(sum(A_i))`；
- 在最坏情况下（所有三角形覆盖全屏）可近似上界为 `O(F*W*H)`。

空间复杂度：
- 颜色缓冲 `O(W*H)`；
- 深度缓冲 `O(W*H)`；
- 网格和中间顶点数据 `O(V+F)`。

## R08

边界与异常处理：
- `normalize` 若遇到零向量或非有限范数，抛 `ValueError`。  
- 透视/视角参数非法（如 `fov<=0`、`near>=far`）抛 `ValueError`。  
- 齐次坐标 `w` 过小（接近 0）抛 `RuntimeError`，避免除零。  
- `width/height <= 0` 抛 `ValueError`。  
- 三角形退化（面积接近 0）会被跳过，不参与着色。  
- 若最终没有任何像素通过着色，抛 `RuntimeError`。

## R09

MVP 取舍说明：
- 只实现 Flat 着色，不实现 Gouraud/Phong 插值。  
- 仅使用方向光，不实现点光源、阴影、镜面高光。  
- 不做三角形裁剪（clipper）；通过场景参数保证几何体在视锥内。  
- 输出 `PPM`（无第三方图像库依赖），保证最小工具链和可复现性。

## R10

`demo.py` 主要函数职责：
- `make_rotation_x/make_rotation_y`：构建模型旋转矩阵。  
- `make_look_at`：构建相机观察矩阵。  
- `make_perspective`：构建透视投影矩阵。  
- `apply_transform/project_to_ndc/ndc_to_screen`：完成空间坐标变换链路。  
- `face_normal`：计算三角形面法线。  
- `shade_face`：按 Flat 规则计算整面颜色。  
- `rasterize_triangle`：逐像素光栅化 + 深度测试。  
- `save_ppm`：写出二进制 PPM 图像。  
- `render_flat_shaded`：组织全流程渲染。  
- `main`：运行 MVP 并打印结果。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机图形学-0429-Flat着色
uv run python demo.py
```

该命令不会请求交互输入。

## R12

输出说明：
- 文件输出：
  - `flat_shading_output.ppm`：渲染结果图（P6 二进制 PPM）。
- 终端输出字段：
  - `output_image`：图像路径；
  - `resolution`：图像分辨率；
  - `triangles_total`：网格总三角形数；
  - `triangles_rasterized`：实际写入过像素的三角形数；
  - `pixels_shaded`：通过 z-test 的像素总数；
  - `mean_face_intensity`：参与绘制面片的平均亮度；
  - `depth_range`：已写入深度的最小/最大值。

## R13

建议最小测试集：
- 基线测试：直接运行脚本，检查是否生成 `flat_shading_output.ppm`。  
- 稳定性测试：将分辨率改为更高值（如 `960x720`），应仍可输出。  
- 参数测试：修改相机位置或光照方向，结果应出现可解释的亮暗变化。  
- 异常测试：将 `width=0` 或 `fov=180`，应触发明确异常。

## R14

可调参数（在 `main` 或 `render_flat_shaded` 内）：
- 分辨率：`width`、`height`。  
- 模型姿态：`make_rotation_x/y` 的角度。  
- 相机：`eye/target/up`。  
- 投影：`fov_y_deg`、`z_near`、`z_far`。  
- 光照：`light_dir_view`、`ambient`、`diffuse`。

调参建议：
- 画面过暗：提高 `ambient` 或调整光照方向。  
- 遮挡关系异常：检查 `z_near/z_far` 与深度映射区间。  
- 锯齿明显：提高分辨率（本 MVP 未实现抗锯齿）。

## R15

与常见着色模型对比：
- Flat 着色：
  - 每个三角形一次光照，计算量低；
  - 保留面感，边界明显。
- Gouraud 着色：
  - 顶点计算光照，片元插值；
  - 视觉更平滑，但高光细节可能丢失。
- Phong 着色：
  - 片元级法线插值并逐像素光照；
  - 质量更高，但计算成本更大。

## R16

典型应用场景：
- 教学：展示渲染管线与 z-buffer 的最小闭环。  
- 原型：快速验证网格、相机和可见性逻辑。  
- 低开销实时渲染：低多边形或风格化画面。  
- 作为 Gouraud/Phong/PBR 的基线实现。

## R17

可扩展方向：
- 添加背面剔除、视锥裁剪和齐次裁剪。  
- 增加线框叠加与法线可视化调试。  
- 实现 Gouraud/Phong 以对比插值策略。  
- 增加多光源与镜面反射项。  
- 输出深度图或法线图用于验证与后处理。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 固定分辨率后调用 `render_flat_shaded`，并准备输出文件路径。  
2. `render_flat_shaded` 通过 `generate_cube_mesh` 构建立方体几何，随后构建 `M/V/P` 矩阵。  
3. 调用 `apply_transform` 与 `project_to_ndc`/`ndc_to_screen`，得到 world/view/screen 空间顶点。  
4. 初始化 `image` 与 `depth_buffer`，设置方向光 `light_dir_view`。  
5. 遍历每个三角形，先用 `face_normal` 分别求 view/world 法线，再用 `shade_face` 得到该面的统一 RGB。  
6. 将三角形传入 `rasterize_triangle`：先算包围盒，再对像素中心用边函数求重心坐标并做内部性判断。  
7. 对内部像素插值深度并执行 z-test，若更近则更新颜色与深度，累计像素计数与统计量。  
8. 渲染结束后 `save_ppm` 输出图像，`main` 打印关键统计字段，形成可审计的最小 Flat 着色闭环。
