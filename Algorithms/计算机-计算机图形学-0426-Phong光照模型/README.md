# Phong光照模型

- UID: `CS-0267`
- 学科: `计算机`
- 分类: `计算机图形学`
- 源序号: `426`
- 目标目录: `Algorithms/计算机-计算机图形学-0426-Phong光照模型`

## R01

Phong 光照模型是经典的局部光照经验模型，常用于实时渲染入门与教学。其核心思想：
1. 环境光项近似全局底光；
2. 漫反射项采用 Lambert 余弦定律 `max(n·l, 0)`；
3. 镜面高光项采用 Phong 指数项 `max(r·v, 0)^shininess`。

本条目给出一个最小可运行 MVP：用解析射线-球求交得到可见点，在命中点应用 Phong 光照并导出一张可审计图像。

## R02

问题定义（MVP 范围）：
1. 输入固定在 `demo.py` 中，无交互：
   - 相机参数：分辨率、FOV、相机原点；
   - 几何参数：单球体中心与半径；
   - 材质参数：`albedo/ambient/diffuse/specular/shininess`；
   - 光源参数：两盏点光源的位置、颜色、强度；
   - 全局环境光 `ambient_light`。
2. 输出：
   - 图像文件：`phong_sphere.ppm`；
   - 终端统计：命中率、平均亮度、平均 `n·l`、平均 `r·v`、平均 Phong 高光强度；
   - 关键像素探针表与成功标记 `All checks passed.`。

## R03

核心数学关系：
1. 主光线：`p(t)=o+t*d, t>0`。
2. 球体求交：解方程 `||o+t*d-c||^2=r^2`，取最近正根。
3. 几何法线：`n=normalize(p_hit-c)`。
4. 漫反射：`I_diff = kd * max(n·l, 0)`。
5. Phong 镜面反射：
   - 反射向量 `r = reflect(-l, n)`；
   - 高光强度 `I_spec = ks * max(r·v,0)^s`，`s` 即 `shininess`。
6. 最终颜色：
   - `C = C_ambient + Σ(C_diff + C_spec)`；
   - 最终裁剪到 `[0,1]`。

## R04

`demo.py` 执行流程：
1. 构建相机、球体、材质、点光源、环境光；
2. 逐像素反投影生成主光线；
3. 执行射线-球解析求交；
4. 未命中写入背景渐变；
5. 命中后计算命中点、法线、观察方向；
6. 对每盏灯计算漫反射与 Phong 镜面项；
7. 汇总颜色并累计诊断统计；
8. 保存 PPM，打印统计与探针，执行断言。

## R05

关键数据结构：
1. `Material`：材质底色与 `ambient/diffuse/specular/shininess`；
2. `PointLight`：点光位置、颜色、强度；
3. `Camera`：相机原点与视场角；
4. `image: ndarray[H,W,3]`：渲染颜色缓冲；
5. `stats: dict[str,float]`：渲染统计指标。

## R06

正确性直觉：
1. 解析求交避免光栅化插值细节干扰，几何关系更直接；
2. 背光面由 `n·l<=0` 自动剪除漫反射与高光贡献；
3. 高光由 `r·v` 决定，视线接近镜面反射方向时亮度提升；
4. `shininess` 越大高光越集中，越小高光越宽；
5. 颜色统一 `clip` 到 `[0,1]`，保证输出像素合法。

## R07

复杂度分析（像素数 `P=W*H`，光源数 `L`）：
1. 时间复杂度：`O(P*L)`；
2. 空间复杂度：`O(P)`（图像缓存）+ `O(1)`（每像素临时变量）。

## R08

边界与鲁棒性处理：
1. `EPSILON` 防止零长度向量归一化除零；
2. 球体求交仅接受正根，过滤相机后方交点；
3. 光源与命中点过近时跳过该光源项，避免数值爆炸；
4. 断言检查像素范围、命中比例、亮度范围和统计下界；
5. 通过探针像素表快速发现全黑、过曝或偏色异常。

## R09

MVP 取舍说明：
1. 只做局部光照，不包含阴影、反射、折射、全局光照；
2. 只渲染单球体，便于聚焦 Phong 模型本身；
3. 输出采用 PPM（二进制）以减少第三方图像依赖；
4. 参数固定在脚本内，保证复现实验结果稳定。

## R10

`demo.py` 函数职责：
1. `normalize`：向量归一化；
2. `reflect`：计算反射向量；
3. `ray_sphere_intersection`：解析求交；
4. `background_color`：生成背景渐变；
5. `phong_shade`：执行环境光 + 漫反射 + Phong 镜面；
6. `render_phong_sphere`：主渲染循环与统计汇总；
7. `save_ppm`：保存二进制 PPM；
8. `build_probe_table`：生成探针像素表；
9. `run_checks`：最小可验证断言；
10. `main`：组装参数并执行全流程。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机图形学-0426-Phong光照模型
uv run python demo.py
```

脚本不需要任何交互输入。

## R12

输出说明：
1. 文件：`phong_sphere.ppm`；
2. 统计字段：
   - `hit_ratio`：命中像素占比；
   - `mean_luminance`：图像平均亮度；
   - `mean_ndotl`：平均漫反射余弦项；
   - `mean_rdotv`：平均反射-视线夹角余弦项；
   - `mean_phong_spec_strength`：平均 Phong 高光强度。
3. `Probe colors`：关键位置 RGB 与亮度，便于人工审查。

## R13

最小验证清单：
1. `README.md` 不含占位符残留；
2. `demo.py` 不含占位符残留；
3. `uv run python demo.py` 可一次运行完成；
4. 生成 `phong_sphere.ppm`；
5. 终端出现 `All checks passed.`；
6. `hit_pixels` 与 `background_pixels` 均大于 0。

## R14

当前固定实验参数：
1. 分辨率：`320 x 220`；
2. 相机 FOV：`54°`；
3. 球体中心/半径：`[0.0, -0.03, 3.75] / 1.12`；
4. 材质：偏冷蓝色漫反射，`shininess=42`；
5. 光源：暖色主光 + 冷色辅光。

该参数组合可稳定观察到明暗面过渡与镜面高光。

## R15

与 Blinn-Phong 的关系：
1. Phong 使用 `r·v`，需显式构造反射向量 `r`；
2. Blinn-Phong 使用 `n·h`，需构造半角向量 `h=normalize(l+v)`；
3. 两者都属于经验镜面模型，但高光形状与指数映射并不完全一致；
4. 本条目严格采用 Phong 路径，保持概念与实现一致。

## R16

适用与不适用场景：
1. 适用：
   - 图形学教学中的局部光照入门；
   - 快速调试 `diffuse/specular/shininess` 参数；
   - 作为更复杂渲染器的基线模块。
2. 不适用：
   - 需要物理正确能量守恒的 PBR 场景；
   - 需要间接光照、软阴影、体积效应的高真实感渲染。

## R17

可扩展方向：
1. 加入阴影射线，屏蔽被遮挡光源贡献；
2. 扩展到多物体与多材质；
3. 加入纹理映射和法线贴图；
4. 将逐像素循环向量化（NumPy 批处理或 PyTorch）；
5. 迁移到 OpenGL/WebGPU 着色器版本做实时显示。

## R18

`demo.py` 源码级算法流（8 步）：
1. `main` 中定义相机、球体、材质、点光源与环境光，并调用 `render_phong_sphere`。
2. `render_phong_sphere` 逐像素反投影，计算主光线方向 `d`。
3. 调用 `ray_sphere_intersection` 解二次方程，获取最近正交点参数 `t`。
4. 若无命中则调用 `background_color` 写入背景；命中则计算命中点 `p`、法线 `n`、视线 `v`。
5. 调用 `phong_shade`，先叠加环境光，再遍历每个点光源。
6. 每盏灯先计算漫反射 `max(n·l,0)`，再通过 `reflect(-l,n)` 得到反射向量 `r`。
7. 计算镜面项 `max(r·v,0)^shininess`，累加颜色并记录 `ndotl/rdotv/spec` 诊断统计。
8. 渲染完成后 `save_ppm` 输出图像，`run_checks` 做断言，打印统计与探针表。
