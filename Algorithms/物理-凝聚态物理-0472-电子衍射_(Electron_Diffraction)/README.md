# 电子衍射 (Electron Diffraction)

- UID: `PHYS-0449`
- 学科: `物理`
- 分类: `凝聚态物理`
- 源序号: `472`
- 目标目录: `Algorithms/物理-凝聚态物理-0472-电子衍射_(Electron_Diffraction)`

## R01

电子衍射描述了电子波与晶体周期势相互作用后在远场形成衍射斑点的现象。其核心是“电子具有波动性”，满足德布罗意关系，且在晶体中由倒易点阵和结构因子决定可见反射。

本条目给出一个最小可运行模型：在透射电镜（TEM）小角近似下，构造单原子 fcc 晶体的区轴衍射图，并验证消光规则与几何半径比。

## R02

在凝聚态物理与材料表征里，电子衍射用于：

- 晶体结构鉴定（立方、六方、超结构）；
- 区轴判定与取向关系分析；
- 晶格常数、相变与缺陷（位错、孪晶）线索提取；
- 作为 HRTEM、CBED、4D-STEM 分析前的快速结构约束。

该 MVP 聚焦“可解释算法骨架”，不是工业级多重散射模拟器。

## R03

MVP 目标是把以下闭环打通：

1. 从加速电压计算相对论电子波长；
2. 枚举 `(h,k,l)` 并施加区轴定律；
3. 用 fcc 结构因子筛掉系统性消光反射；
4. 按 `R ≈ L * lambda * |g_perp|` 投影到探测器；
5. 光栅化为二维衍射图并执行自动断言验证。

## R04

本实现使用的主要公式：

1. 相对论电子波长：
`lambda = h / sqrt(2 m e V * (1 + eV/(2mc^2)))`
2. 区轴定律（zone law）：
`h*u + k*v + l*w = 0`
3. 立方晶体倒易矢量（不含 `2pi` 约定）：
`g = (h, k, l) / a`
4. fcc 单原子结构因子：
`F = sum_j exp(2pi i (h x_j + k y_j + l z_j))`
5. 运动学强度（含简化 Debye-Waller 衰减）：
`I ~ |F|^2 * exp(-B |g|^2)`
6. TEM 小角几何投影：
`x = L * lambda * g_x,  y = L * lambda * g_y,  R = sqrt(x^2+y^2)`

## R05

设 `H = max_miller_index`，反射枚举规模约 `M = O((2H+1)^3)`，图像边长 `N`：

- 枚举与区轴过滤：`O(M)`；
- 结构因子与强度计算：`O(M)`；
- 光栅化点沉积：`O(M)`；
- 高斯模糊（`scipy.ndimage.gaussian_filter`）：`O(N^2)` 量级；
- 径向平均：`O(N^2)`。

总时间复杂度可写为 `O(M + N^2)`，空间复杂度 `O(M + N^2)`。

## R06

`demo.py` 完成以下输出：

- 关键物理参数（电压、波长、晶格常数、区轴）；
- 反射表（`h,k,l,radius,g,intensity_rel`）的高强度子集；
- 按 `h^2+k^2+l^2` 分组的前几圈衍射环统计；
- 图像径向平均 profile 样本；
- 自动验证全部通过后打印 `All checks passed.`。

## R07

优点：

- 物理路径透明：每一步都在源码显式展开；
- 有结构因子消光和几何比例两类独立验证；
- 工具栈小（`numpy/scipy/pandas`）且可复现实验。

局限：

- 使用运动学近似，未包含多重散射（dynamical diffraction）；
- 晶体模型为单原子 fcc，未覆盖复杂基元；
- 探测器响应、像差、厚度效应仅以简化参数处理。

## R08

前置知识：

- 德布罗意波长与布拉格/倒易点阵语言；
- 区轴定律与结构因子消光规则；
- Python 数组运算与基础数据表。

运行环境：

- Python `>=3.10`
- `numpy`
- `scipy`
- `pandas`

## R09

适用场景：

- 教学展示“电压 -> 波长 -> 反射点 -> 衍射图”；
- 快速验证 fcc 区轴图样和消光规则；
- 更复杂 TEM/4D-STEM 管线中的最小可审计算法核。

不适用场景：

- 定量拟合厚样品多重散射；
- 高精度晶体势反演与原子级参数回归；
- 含强非弹性散射或复杂仪器函数的实验拟合。

## R10

正确性直觉：

1. 更高电压应给出更短电子波长；
2. fcc 结构中 `(100)/(110)` 应系统性消光；
3. 同一区轴下 `R` 与 `|g|` 成正比，因此 `R_220 / R_200 = sqrt(2)`；
4. 非磁、非手性简化下图样应满足中心反演对称（Friedel 对称近似）。

这些直觉在 `demo.py` 中全部对应到断言。

## R11

数值稳定与实现细节：

- 区轴条件使用整数点积 `==0`，避免浮点判定漂移；
- 结构因子虽经复指数计算，但消光判定保留数值容差；
- 图像渲染先点沉积再高斯模糊，减少离散像素锯齿；
- 对最终图归一化，避免不同参数下打印量纲失衡。

## R12

关键参数与影响：

- `accelerating_voltage_kv`：决定 `lambda`，直接缩放斑点半径；
- `lattice_constant_angstrom`：决定倒易点间距，影响环半径；
- `zone_axis`：决定可见反射切片（满足 zone law 的点）；
- `max_miller_index`：控制反射枚举范围和计算量；
- `debye_waller_b_ang2`：抑制高阶反射强度；
- `image_size` 与 `spot_sigma_px`：控制图像采样与斑点展宽。

## R13

该 MVP 的“保证”类型：

- 近似比保证：N/A（非优化近似问题）；
- 概率成功率保证：N/A（算法确定性，无随机采样）。

可执行保证：

- 波长、消光规则、几何比例、中心对称性四类检查同时通过；
- 失败时 `assert` 抛错并返回非零退出码，便于自动化验证。

## R14

常见失效模式：

1. 把晶体结构误设为 bcc/simple cubic，会导致“错误消光”；
2. 区轴输入与样品坐标系不一致，反射集合会错位；
3. `max_miller_index` 过小导致高阶环缺失；
4. `spot_sigma_px` 过大使环间分辨率下降；
5. 误用单位（Å 与 m 混淆）会使半径数量级错误。

## R15

可扩展方向：

- 将 fcc 基元替换为任意多原子基元并引入原子散射因子 `f_j(g)`；
- 加入厚度和动态散射近似（Howie-Whelan / multislice）；
- 扩展到任意晶系和任意区轴自动索引；
- 输出图像到文件并接入实验斑点检测/索引流程；
- 引入噪声模型和探测器 MTF 进行更真实的 forward simulation。

## R16

相关主题：

- X 射线衍射（XRD）与中子衍射的结构因子对比；
- Ewald 球几何与 Laue 条件；
- 选区电子衍射（SAED）、汇聚束电子衍射（CBED）；
- 多重散射理论与多片层（multislice）算法。

## R17

`demo.py` 功能清单：

- `electron_wavelength_relativistic`：相对论电子波长；
- `enumerate_miller_indices` + `apply_zone_law`：反射集合生成；
- `fcc_structure_factor`：系统性消光判定核心；
- `build_reflection_table`：强度与探测器坐标计算；
- `rasterize_pattern` + `radial_profile`：图样生成与诊断；
- `run_validations`：四项自动断言验证。

运行方式：

```bash
cd Algorithms/物理-凝聚态物理-0472-电子衍射_(Electron_Diffraction)
uv run python demo.py
```

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `DiffractionConfig` 固定电压、晶格常数、区轴、Miller 截断与成像参数，建立单次仿真上下文。  
2. `electron_wavelength_relativistic` 根据相对论动量公式计算电子波长 `lambda`，把加速电压映射到衍射尺度。  
3. `enumerate_miller_indices` 生成立方索引网格，`apply_zone_law` 只保留满足 `hu+kv+lw=0` 的反射。  
4. `fcc_structure_factor` 对 fcc 四个基元点逐项求复指数和，得到 `F(hkl)` 并通过 `|F|^2` 自动体现消光。  
5. `build_reflection_table` 计算 `g`、Debye-Waller 衰减后的强度、以及 `x=L*lambda*g_x`/`y=L*lambda*g_y` 探测器坐标，形成 `pandas` 表。  
6. `rasterize_pattern` 将离散反射强度投到像素网格并用 `gaussian_filter` 做点扩散近似，得到归一化二维衍射图。  
7. `radial_profile` 通过半径分箱计算图像径向平均，用于检查环状强度分布是否合理。  
8. `run_validations` 执行四类断言（波长范围、fcc 消光、`R220/R200` 比值、中心反演对称），`main` 打印反射表和诊断摘要并在通过时输出成功信息。  

第三方库仅提供数值基础能力：`numpy` 做线代/数组，`scipy` 做常数与高斯滤波，`pandas` 负责反射表组织；物理判据与算法流程均在源码中显式实现，没有把“电子衍射”当成黑箱函数调用。
