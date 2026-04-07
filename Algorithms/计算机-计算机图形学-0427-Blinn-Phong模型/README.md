# Blinn-Phong模型

- UID: `CS-0268`
- 学科: `计算机`
- 分类: `计算机图形学`
- 源序号: `427`
- 目标目录: `Algorithms/计算机-计算机图形学-0427-Blinn-Phong模型`

## R01

Blinn-Phong 模型是局部光照模型中的经典镜面反射近似：
- 漫反射沿用 Lambert 项 `max(n·l, 0)`；
- 镜面项不直接使用反射向量 `r`，而是使用半角向量 `h = normalize(l + v)`；
- 高光强度写作 `max(n·h, 0)^shininess`，通常数值稳定且计算简洁。

本条目实现一个最小可运行 MVP：
1. 用解析射线-球体求交得到可见点；
2. 在命中点计算环境光 + 漫反射 + Blinn-Phong 高光；
3. 输出一张可审计的 `PPM` 图像并打印统计指标。

## R02

问题定义（MVP 范围）：
1. 输入（全部内置在 `demo.py` 中，无交互）：
   - 相机参数：分辨率、FOV、相机位置；
   - 场景参数：单球体中心与半径；
   - 材质参数：`albedo/ambient/diffuse/specular/shininess`；
   - 光源参数：两盏点光源位置、颜色、强度；
   - 全局环境光 `ambient_light`。
2. 输出：
   - 图像文件：`blinn_phong_sphere.ppm`；
   - 终端统计：命中像素占比、平均亮度、平均 `n·l`、平均 `n·h`、Blinn 与 Phong 高光强度对比；
   - 成功标志：`All checks passed.`。

## R03

核心数学关系：
1. 主光线：`p(t)=o+t*d, t>0`。
2. 球体求交：解 `||o+t*d-c||^2=r^2`，取最近正根。
3. 法线：`n = normalize(p_hit - c)`。
4. 漫反射：`I_diff = kd * max(n·l, 0)`。
5. Blinn-Phong 高光：
   - `h = normalize(l+v)`；
   - `I_spec = ks * max(n·h, 0)^s`，`s` 为高光指数。
6. 最终颜色：
   - `C = C_ambient + Σ(C_diff + C_spec)`，最后裁剪到 `[0,1]`。

## R04

`demo.py` 执行流程：
1. 构建固定相机、球体、材质与两盏点光源；
2. 遍历像素，反投影生成主光线；
3. 执行射线-球体求交；
4. 未命中写背景渐变；
5. 命中则计算法线和视线方向；
6. 对每盏灯计算 Blinn-Phong 光照贡献；
7. 汇总颜色写入图像缓冲并累计统计量；
8. 保存 PPM、打印统计表与探针像素，并做断言校验。

## R05

关键数据结构：
1. `Material`：材质颜色与三项系数（环境/漫反射/镜面）及 `shininess`；
2. `PointLight`：点光位置、颜色、强度；
3. `Camera`：相机位置与视场角；
4. `image: ndarray[H,W,3]`：颜色缓冲；
5. `stats: dict[str,float]`：渲染统计（命中率、平均亮度、平均镜面强度等）。

## R06

正确性直觉：
1. 球体解析求交保证几何可见点可解释、可复现；
2. 法线由几何定义直接给出，避免插值误差来源；
3. `n·l` 控制受光面，确保背光面不会产生漫反射；
4. `n·h` 控制高光集中度，`shininess` 越大高光越尖锐；
5. 颜色统一裁剪到 `[0,1]`，防止数值越界造成图像非法像素。

## R07

复杂度分析（设像素总数 `P=W*H`，光源数 `L`）：
1. 时间复杂度：`O(P * L)`。
   - 每个像素一次球体求交（常数时间）；
   - 命中像素对每个光源计算一次光照贡献。
2. 空间复杂度：`O(P)`（图像缓存）+ `O(1)`（常量临时变量）。

## R08

边界与鲁棒性处理：
1. `EPSILON` 过滤接近 0 的长度，避免归一化除零；
2. 球体求交仅接受正根，排除相机后方交点；
3. 光源距离过近时跳过该光源采样，避免数值不稳定；
4. 渲染后断言像素范围、命中像素数量、亮度范围与统计下界；
5. 探针像素表用于快速发现“全黑/全白/异常偏色”问题。

## R09

MVP 取舍：
1. 只实现局部光照，不做阴影、反射、折射、全局光照；
2. 场景固定为单球体，优先突出 Blinn-Phong 本体；
3. 输出选择 PPM，避免额外图像库依赖；
4. 保留一个 Phong 高光强度统计作对照，但渲染结果使用 Blinn-Phong。

## R10

`demo.py` 主要函数职责：
1. `ray_sphere_intersection`：解析球体求交；
2. `background_color`：背景渐变；
3. `blinn_phong_shade`：计算环境+漫反射+Blinn 高光，并记录诊断；
4. `render_blinn_phong_sphere`：逐像素发射主光线并汇总统计；
5. `save_ppm`：保存二进制 PPM；
6. `build_probe_table`：输出探针像素颜色表；
7. `run_checks`：最小正确性断言；
8. `main`：组装参数并执行全流程。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机图形学-0427-Blinn-Phong模型
uv run python demo.py
```

脚本不需要任何交互输入。

## R12

输出说明：
1. 文件输出：`blinn_phong_sphere.ppm`；
2. 统计字段：
   - `hit_ratio`：球体覆盖像素比例；
   - `mean_luminance`：图像平均亮度；
   - `mean_ndotl`：平均漫反射余弦项；
   - `mean_ndoth`：平均半角余弦项；
   - `mean_blinn_spec_strength`：Blinn 高光平均强度；
   - `mean_phong_spec_strength`：Phong 高光平均强度（对照）。
3. `Probe colors`：关键像素 RGB 与亮度，便于人工检查。

## R13

最小验证清单：
1. `README.md` 不含未替换占位符；
2. `demo.py` 不含未替换占位符；
3. `uv run python demo.py` 可一次运行完成；
4. 生成 `blinn_phong_sphere.ppm`；
5. 终端出现 `All checks passed.`；
6. `hit_pixels` 与 `background_pixels` 均大于 0。

## R14

当前固定实验参数：
1. 分辨率：`320 x 220`；
2. 相机 FOV：`52°`；
3. 球体中心/半径：`[0,0,3.7]` / `1.15`；
4. 材质：蓝色基底，`shininess=48`；
5. 光源：暖色主光 + 冷色辅光。

该参数能稳定看到镜面高光、明暗过渡与背景梯度。

## R15

与 Phong 镜面项对比：
1. Phong：`max(r·v,0)^s`，需先求反射向量 `r`；
2. Blinn-Phong：`max(n·h,0)^s`，需先求半角向量 `h`；
3. 在相同 `s` 下两者高光宽度通常不同，工程中常按经验映射指数；
4. 本实现输出两者平均高光强度，帮助观察差异但不改变最终渲染路径。

## R16

适用场景：
1. 图形学入门课程中讲解“局部光照 + 高光模型”；
2. 光照参数快速调试（`kd/ks/shininess`）；
3. 作为后续加入阴影、法线贴图、PBR 的前置基线。

不适用场景：
1. 追求物理真实感的离线渲染；
2. 需要间接光照、软阴影、体积散射的任务。

## R17

可扩展方向：
1. 增加阴影检测（shadow ray）；
2. 增加多物体与材质系统；
3. 引入法线贴图与纹理采样；
4. 对 `shininess` 与能量守恒做更严格校准；
5. 迁移为批量向量化/并行版本（如 NumPy 批处理或 PyTorch 张量）。

## R18

`demo.py` 源码级算法流（8 步）：
1. `main` 固定相机、球体、材质和光源，并调用 `render_blinn_phong_sphere`。
2. `render_blinn_phong_sphere` 逐像素反投影构建主光线方向 `d`。
3. 调用 `ray_sphere_intersection` 解二次方程，得到最近正交点距离 `t`。
4. 若无命中，使用 `background_color` 写入背景；若命中，计算命中点 `p` 与法线 `n`。
5. 调用 `blinn_phong_shade`，先计算环境光，再遍历每个点光源。
6. 对每个光源计算 `n·l` 的漫反射项，并构造半角向量 `h=normalize(l+v)`。
7. 用 `max(n·h,0)^shininess` 得到 Blinn 镜面强度，累加颜色并统计对照用 Phong 强度。
8. 全图渲染完成后 `save_ppm` 写文件，`run_checks` 执行断言，打印统计与探针表。
