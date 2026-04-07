# 纹理映射

- UID: `CS-0271`
- 学科: `计算机`
- 分类: `计算机图形学`
- 源序号: `430`
- 目标目录: `Algorithms/计算机-计算机图形学-0430-纹理映射`

## R01

纹理映射（Texture Mapping）用于把二维纹理图像贴到三维几何表面。最常见流程是：
- 给网格顶点提供纹理坐标 `UV`；
- 光栅化时对像素插值 `UV`；
- 用插值后的 `UV` 在纹理上采样颜色；
- 将采样颜色作为该像素的表面颜色。

本条目聚焦单个三角形的最小可运行纹理映射链路。

## R02

本 MVP 解决的问题：
- 输入：
  1. 三角形屏幕坐标 `tri_xy`；
  2. 顶点纹理坐标 `tri_uv`；
  3. 顶点深度 `tri_z`；
  4. 一张程序化生成的 RGB 棋盘纹理。
- 输出：
  1. 仿射 UV 插值得到的贴图结果；
  2. 透视校正 UV 插值得到的贴图结果；
  3. 二者在覆盖像素上的 UV 与颜色误差统计；
  4. 样本像素表用于人工核查。

脚本无交互输入，直接运行并打印结果。

## R03

核心公式：

1. 重心坐标（对像素中心点 `p`）
\[
\lambda_0 + \lambda_1 + \lambda_2 = 1,
\quad p = \lambda_0 v_0 + \lambda_1 v_1 + \lambda_2 v_2
\]

2. 仿射 UV 插值
\[
(u, v)_{aff} = \lambda_0 (u_0,v_0) + \lambda_1 (u_1,v_1) + \lambda_2 (u_2,v_2)
\]

3. 透视校正 UV 插值（`z_i > 0`）
\[
(u,v)_{persp} =
\frac{\sum_i \lambda_i (u_i,v_i)/z_i}{\sum_i \lambda_i / z_i}
\]

4. 双线性采样（每个通道独立）
\[
C(u,v) = (1-t_y)\big((1-t_x)C_{00}+t_x C_{10}\big)
+ t_y\big((1-t_x)C_{01}+t_x C_{11}\big)
\]

## R04

算法主流程：
1. 生成棋盘纹理（高频，便于观察采样差异）。
2. 根据三角形顶点算出屏幕包围盒。
3. 在包围盒内枚举像素中心，计算重心坐标并做 inside 判断。
4. 对覆盖像素分别计算仿射 UV 与透视校正 UV。
5. 对两组 UV 均执行双线性采样，得到两张贴图结果。
6. 统计 UV 与颜色差异（`L1/L∞/MSE/MAE`）。
7. 打印指标与样本表，并执行断言保证结果可信。

## R05

核心数据结构：
- `texture: np.ndarray[H, W, 3]`：`[0,1]` 范围 RGB 纹理。
- `tri_xy: np.ndarray[3, 2]`：屏幕空间三角形顶点。
- `tri_uv: np.ndarray[3, 2]`：每个顶点对应 UV。
- `tri_z: np.ndarray[3]`：顶点深度（用于透视校正）。
- `inside_points: np.ndarray[N, 2]`：三角形覆盖像素中心。
- `inside_bary: np.ndarray[N, 3]`：覆盖像素重心权重。
- `uv_affine/uv_perspective: np.ndarray[N, 2]`：两种插值 UV。
- `color_affine/color_perspective: np.ndarray[N, 3]`：两种采样颜色。

## R06

正确性要点：
- 重心坐标求和应接近 `1`（脚本内断言最大误差）。
- 退化三角形（面积约为 0）必须拒绝处理。
- `tri_z` 必须为正，否则透视校正分母可能失效。
- 包围盒与 inside 过滤后，至少应覆盖一个像素。
- 使用同一纹理与采样器时，仿射与透视路径的差异只来自 UV 插值策略。

## R07

复杂度分析：
- 设包围盒像素数为 `B`，覆盖像素数为 `N`（`N <= B`）。
- 重心计算与 inside 测试：`O(B)`。
- UV 计算与双线性采样：`O(N)`。
- 总时间复杂度：`O(B + N)`，在单三角场景下近似 `O(B)`。
- 额外空间：主要为点集与中间数组，`O(B + N)`。

## R08

边界与异常处理：
- `texture` 必须是 `(H, W, 3)`。
- `tri_xy/tri_uv/tri_z` 形状必须分别为 `(3,2)/(3,2)/(3,)`。
- `width,height,size,checks` 必须为正且尺寸合理。
- 双线性采样采用 `repeat` 包裹（`u,v mod 1`），支持纹理平铺。
- 对 triangle bbox 做图像边界裁剪，避免越界访问。

## R09

MVP 取舍：
- 只处理单个三角形，避免引入完整渲染管线复杂度。
- 只比较“仿射 vs 透视校正”两条最核心路径。
- 只使用 `numpy + pandas`，不依赖 OpenGL/渲染引擎黑盒。
- 不输出图片文件，统一用数值指标和样本表验证。
- 纹理使用程序化棋盘，便于复现和控制频率。

## R10

`demo.py` 函数分工：
- `generate_checkerboard`：生成 RGB 棋盘纹理。
- `bilinear_sample_repeat`：向量化双线性采样（repeat 模式）。
- `barycentric_for_points`：批量计算重心坐标。
- `triangle_bbox`：三角形包围盒计算并裁剪到图像范围。
- `rasterize_triangle`：光栅化 + 两类 UV 插值 + 采样 + 指标汇总。
- `make_sample_table`：构造可读样本像素对照表。
- `main`：设置参数、运行流程、打印结果、执行断言。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机图形学-0430-纹理映射
uv run python demo.py
```

无需命令行参数，无交互输入。

## R12

输出解释：
- `covered_pixels`：三角形覆盖像素数。
- `barycentric_sum_max_error`：重心坐标归一性误差。
- `uv_l1_mean`：两种 UV 插值平均绝对差。
- `uv_linf_max`：两种 UV 插值最大绝对差。
- `color_mse_affine_vs_perspective`：颜色均方误差。
- `color_mae_affine_vs_perspective`：颜色平均绝对误差。
- 样本表中 `dR/dG/dB` 为透视校正结果减去仿射结果。

## R13

最小测试建议：
- 保持三角形不变，仅增大 `tri_z` 跨度，应看到误差增大。
- 把 `tri_z` 设为全相等，仿射与透视结果应接近一致。
- 调高 `checks`（更高频纹理）可放大采样差异。
- 更改 `tri_uv` 平铺倍数，观察 repeat 采样稳定性。
- 把三角形移到图像边缘，验证 bbox 裁剪与无越界。

## R14

可调参数（位于 `main`）：
- `width, height`：目标图像分辨率。
- `size, checks`：纹理尺寸与棋盘频率。
- `tri_xy`：三角形几何形状与位置。
- `tri_uv`：贴图展开方式与平铺强度。
- `tri_z`：透视深度分布，决定透视畸变强弱。

建议先固定几何形状，再逐步调 `tri_z` 观察仿射误差。

## R15

与相关方法对比：
- 纯仿射 UV：实现最简单，但在透视投影下会产生纹理“游走/拉伸”误差。
- 透视校正 UV：现代 GPU 标准做法，额外代价小，视觉正确性显著更好。
- 更进一步可加 Mipmap/各向异性过滤，主要解决缩小时的 aliasing，而非透视插值误差。

## R16

典型应用场景：
- 实时渲染中的网格贴图（地形、角色、建筑）。
- 软件光栅器教学与渲染管线验证。
- 可微渲染前端中 UV 插值与采样模块原型。
- CPU 端离线预览器或轻量图形实验环境。

## R17

可扩展方向：
1. 从单三角扩展到三角网格与深度缓冲。
2. 增加 Mipmap 与各向异性 footprint 估计。
3. 增加边界模式（clamp/mirror/wrap）切换。
4. 引入法线贴图、PBR 材质参数纹理。
5. 对接 `PyTorch` 实现可微版本并验证梯度。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 设置 `width/height`、棋盘纹理参数与三角形 `tri_xy/tri_uv/tri_z`，调用 `generate_checkerboard` 得到基础纹理。  
2. `rasterize_triangle` 先用 `triangle_bbox` 计算并裁剪包围盒，再枚举包围盒内像素中心点。  
3. `barycentric_for_points` 构造三角形齐次矩阵 `M`，求 `inv(M)` 后批量计算每个像素的重心权重。  
4. 通过 `inside = all(bary >= -eps)` 过滤覆盖像素，得到 `inside_points` 与 `inside_bary`。  
5. 用 `inside_bary @ tri_uv` 计算仿射 UV；同时按 `uv/z` 与 `1/z` 公式计算透视校正 UV。  
6. `bilinear_sample_repeat` 对两组 UV 分别执行双线性采样（`u,v mod 1` + 四邻域线性混合），得到两组颜色。  
7. 在覆盖像素上统计 `uv_l1_mean/uv_linf_max` 与颜色 `MSE/MAE`，并回填到结果字典。  
8. `main` 打印指标表和样本像素表，最后断言重心和约束、覆盖像素数和“仿射与透视有可见差异”三项质量门禁。  
