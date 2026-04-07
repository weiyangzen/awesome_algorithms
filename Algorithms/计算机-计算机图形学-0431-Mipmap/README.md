# Mipmap

- UID: `CS-0272`
- 学科: `计算机`
- 分类: `计算机图形学`
- 源序号: `431`
- 目标目录: `Algorithms/计算机-计算机图形学-0431-Mipmap`

## R01

Mipmap 是纹理映射中的层级预过滤技术。核心思想是：
- 预先把原始纹理逐层下采样为 `1/2, 1/4, ...` 尺寸的金字塔。
- 采样时根据屏幕像素对应的纹理覆盖面积（footprint）选择合适层级（LOD）。
- 在相邻层级之间做线性插值（三线性过滤的一部分），减少走样（aliasing）和闪烁。

## R02

本条目要解决的问题：
- 输入：一张高频纹理（示例为棋盘格）与屏幕到纹理的缩放关系。
- 输出：
  1. Mipmap 金字塔各层。
  2. 不使用 Mipmap 的采样结果。
  3. 使用 Mipmap（三线性层级插值）的采样结果。
  4. 二者相对超采样参考图的误差对比。

`demo.py` 直接运行，无需任何交互输入。

## R03

关键公式：
- 纹理覆盖尺度（texel footprint）记为 `rho`（单位：texel/像素）。
- LOD 估计：
\[
\text{lod} = \log_2(\max(\rho, \epsilon))
\]
- 将 `lod` 裁剪到 `[0, L-1]`，其中 `L` 是金字塔层数。
- 层内双线性采样：`c_l = bilinear(level_l, u, v)`。
- 层间线性插值（三线性）：
\[
c = (1-t)\,c_{\lfloor lod \rfloor} + t\,c_{\lceil lod \rceil},\quad t=lod-\lfloor lod \rfloor
\]

## R04

算法主流程：
1. 生成基础纹理（本示例为 RGB 棋盘格）。
2. 迭代执行 `2x2` 盒滤波下采样，构建 Mipmap 金字塔直到 `1x1`。
3. 设定输出分辨率与纹理重复次数，计算每个屏幕像素中心对应的 `(u,v)`。
4. 基线方法：仅在第 0 层做双线性采样（无 Mipmap）。
5. Mipmap 方法：根据 `rho` 计算 `lod`，做层内双线性 + 层间线性插值。
6. 使用多子像素超采样作为参考图。
7. 统计两种方法到参考图的 MSE，并打印改善比例。

## R05

核心数据结构：
- `Texture = np.ndarray[H, W, C]`：浮点 RGB 图像，范围 `[0,1]`。
- `levels: list[Texture]`：Mipmap 金字塔，`levels[0]` 为原图。
- `render_*: np.ndarray[H_out, W_out, 3]`：渲染结果。
- 标量指标：`mse_no_mipmap`、`mse_mipmap`、`improvement_ratio`。

## R06

正确性要点：
- 金字塔每层是上一层的低通近似（`2x2` 平均），可抑制高频混叠。
- `lod` 随 footprint 单调增长，缩小越厉害时会选到更低分辨率层。
- 三线性层间插值避免“层级跳变”导致的突变。
- 参考图采用更高采样率积分近似，可作为抗锯齿基准比较。

## R07

复杂度（设原图 `N x N`，输出 `M x M`）：
- 构建金字塔：
  \[
  O\left(N^2 + \frac{N^2}{4} + \frac{N^2}{16} + ...\right)=O(N^2)
  \]
- 渲染：`O(M^2)`（每像素常数次采样）。
- 超采样参考（`s x s` 子样本）：`O(M^2 s^2)`。
- 空间复杂度：金字塔总存储 `O(N^2)`。

## R08

边界与异常处理：
- 输入纹理必须是 `H x W x C` 且 `H,W,C > 0`。
- 允许奇数尺寸：下采样前使用边界复制补齐偶数维。
- `u,v` 采用 repeat 包裹（`mod 1`）支持纹理重复。
- `rho` 使用 `max(rho, 1e-8)` 防止 `log2(0)`。
- 输出脚本中包含断言：Mipmap 误差应不高于无 Mipmap 误差。

## R09

MVP 取舍：
- 仅用 `numpy` 实现，避免引入图像框架依赖。
- 采用经典 `2x2` 盒滤波构建层级，简单且可解释。
- 采样采用双线性 + 层间线性（即三线性核心），不扩展到各向异性过滤。
- 不写文件，直接在终端输出层级信息与误差指标。

## R10

`demo.py` 函数分工：
- `generate_checkerboard_texture`：生成可控频率的测试纹理。
- `downsample_2x2_box`：单层下采样。
- `build_mipmap`：构建完整金字塔。
- `bilinear_sample_repeat`：纹理重复模式下的双线性采样。
- `sample_no_mipmap`：仅采样基底层。
- `sample_with_mipmap`：按 `lod` 做三线性采样。
- `render_scene`：渲染无 Mipmap / 有 Mipmap / 参考超采样三种图。
- `main`：组织流程并打印量化结果。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机图形学-0431-Mipmap
uv run python demo.py
```

无参数、无交互输入。

## R12

输出解释：
- `Mipmap levels`：金字塔层数与每层尺寸。
- `MSE(no mipmap, reference)`：无 Mipmap 与参考图误差。
- `MSE(mipmap, reference)`：Mipmap 与参考图误差。
- `Improvement ratio`：误差改善比例，值越大越好。
- `Sample pixel`：挑选一处像素对比三种结果，便于直观看差异。

## R13

建议最小测试：
- 高频棋盘 + 大缩小倍率（本示例）：应看到 Mipmap 明显更接近参考图。
- 低缩小倍率（`rho≈1`）：两者接近，LOD 多在 0 附近。
- 不同重复次数（`tiling`）：频率越高，Mipmap 价值越明显。
- 奇数分辨率纹理：验证下采样补齐逻辑。

## R14

可调参数（位于 `main`）：
- `base_size`：基础纹理分辨率。
- `checks`：棋盘频率。
- `out_size`：输出分辨率。
- `tiling`：输出画面中纹理重复次数。
- `supersample`：参考图每维子采样数。

实践建议：先固定 `out_size`，增大 `tiling` 观察 aliasing 与 Mipmap 抑制效果。

## R15

与相关方法对比：
- 仅双线性（无 Mipmap）：实现简单，但强缩小时混叠明显。
- Mipmap（三线性）：成本低、稳定，是实时渲染的标准最小配置。
- 各向异性过滤：在斜视角更好，但计算与访存开销更高。

## R16

典型应用场景：
- 游戏引擎中的 2D/3D 纹理采样。
- 远景地形与建筑表面贴图。
- 虚拟相机缩放时减少闪烁和摩尔纹。
- GPU 纹理单元中的固定功能采样链路。

## R17

可扩展方向：
- 把盒滤波替换为高斯或 Lanczos 预过滤。
- 增加各向异性 footprint 估计（使用屏幕导数向量）。
- 引入 gamma-aware 下采样（在线性空间构建 mip）。
- 加入性能计时与矢量化优化，评估不同采样策略成本。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 设定 `base_size/checks/out_size/tiling/supersample` 并生成棋盘纹理。  
2. `build_mipmap` 从第 0 层开始循环调用 `downsample_2x2_box`，直到尺寸收敛为 `1x1`。  
3. `downsample_2x2_box` 对奇数宽高做 edge padding，再把四个子采样块做均值形成下一层。  
4. `render_scene` 逐像素计算纹理坐标 `(u,v)` 与覆盖尺度 `rho = tiling * base_width / out_size`。  
5. 无 Mipmap 路径调用 `sample_no_mipmap`，其内部直接在 `levels[0]` 上做 `bilinear_sample_repeat`。  
6. Mipmap 路径调用 `sample_with_mipmap`：先算 `lod=log2(rho)` 并裁剪，再对 `floor/ceil` 两层做双线性采样并线性混合。  
7. 参考路径在每个像素内做 `supersample x supersample` 子像素积分，近似理想低通结果。  
8. `main` 统计两条路径相对参考图的 MSE、改善比例，并断言 Mipmap 误差不高于无 Mipmap 误差。  
