# X射线散射 (X-ray Scattering)

- UID: `PHYS-0448`
- 学科: `物理`
- 分类: `凝聚态物理`
- 源序号: `471`
- 目标目录: `Algorithms/物理-凝聚态物理-0471-X射线散射_(X-ray_Scattering)`

## R01

X 射线散射（X-ray Scattering）研究的是入射 X 射线与材料中电子密度相互作用后产生的角分布强度。  
在凝聚态物理里，最核心的可观测量之一是静态结构因子 `S(q)`，它把“实空间原子排布”映射到“倒空间散射峰”。

本条目实现一个最小但完整的粉末散射 MVP：  
- 用 Debye 散射方程从原子坐标直接计算 `S(q)`；  
- 对比晶体（fcc）与非晶（随机点云）谱线差异；  
- 自动提取峰位并验证 fcc 低阶峰位置；  
- 用机器学习回归和 PyTorch 反演完成“前向+逆向”闭环。

## R02

该算法在凝聚态中的意义：

- 晶体结构鉴定：有序结构会出现明显 Bragg 峰；
- 无序程度评估：非晶通常只有宽峰/弱峰；
- 参数反演：可从散射谱反推晶格常数、热振动参数等；
- 为更复杂方法（Rietveld、PDF、全谱拟合）提供可审计的最小基线。

本实现刻意不依赖专用黑盒散射软件，强调源码透明性。

## R03

MVP 任务定义：

- 输入（程序内置）：
  - 一个有限 fcc 超胞原子坐标；
  - 一个等密度随机点云（作为非晶对照）；
  - `q` 采样网格、噪声幅度、平滑参数等。
- 输出：
  - 晶体与非晶的 `S(q)` 曲线；
  - 峰值表（`q_peak, S_peak, prominence`）；
  - 理论峰位与检测峰位对照表；
  - 反演得到的晶格常数 `a_fit` 与 Debye-Waller 参数 `B_fit`。

## R04

核心物理公式（本实现显式展开）：

1. Debye 粉末散射结构因子  
`S(q) = 1 + (2/N) * sum_{i<j} sin(q r_ij)/(q r_ij)`

2. Debye-Waller 衰减（简化到相干项包络）  
`S_dw(q) = 1 + (S(q)-1) * exp(-B q^2)`

3. fcc 低阶理论峰位置（粉末）  
`q_hkl = (2pi/a) * sqrt(h^2 + k^2 + l^2)`，且 fcc 仅允许 `(h,k,l)` 全奇或全偶。

4. 本实现中 `sin(qr)/(qr)` 用 `np.sinc(qr/pi)` 数值稳定地计算。

## R05

复杂度（记原子数 `N`、`q` 点数 `M`）：

- 距离预计算：`O(N^2)`；
- Debye 求和：`O(M * N^2)`（通过上三角距离压缩为 `M * N(N-1)/2`）；
- 峰值检测：`O(M)`；
- 线性回归：`O(P)`（`P` 为峰数，通常远小于 `M`）；
- Torch 反演：`O(E * M * N^2)`（`E` 为 epoch）。

本条目取 `N=108`（fcc `3x3x3` 超胞），可在 CPU 上快速完成。

## R06

`demo.py` 交付的最小闭环：

1. 生成 fcc 与非晶两套结构；
2. 计算并平滑两条 `S(q)` 曲线；
3. 提取晶体峰并对照 fcc 理论峰位；
4. 用 `LinearRegression` 评估“理论峰 -> 检测峰”一致性；
5. 用 `torch` 对 `a` 与 `B` 做全谱反演；
6. 输出表格和诊断指标，并执行断言作为自动验收。

## R07

优点：

- 物理关系清晰：从 `r_ij` 到 `S(q)` 全链路可追踪；
- 同时包含前向模拟（谱生成）和逆向估计（参数拟合）；
- 含晶体/非晶对照，物理可解释性直观；
- 第三方库仅用于数值工具，不替代核心物理逻辑。

局限：

- 采用有限尺寸超胞，会有峰展宽与边界效应；
- 只做单原子 fcc，未引入元素散射因子 `f_j(q)`；
- Debye-Waller 仅做简化包络，不含各向异性热振动张量。

## R08

前置知识：

- 结构因子与 Debye 散射方程；
- fcc 反射选择规则；
- 基本数值优化和峰值检测。

运行环境（本仓库默认依赖已覆盖）：

- Python `>=3.10`
- `numpy`
- `scipy`
- `pandas`
- `scikit-learn`
- `torch`

## R09

适用场景：

- 教学演示“原子坐标 -> 粉末散射曲线”；
- 快速验证结构有序/无序差异；
- 作为更复杂衍射拟合前的算法 sanity-check。

不适用场景：

- 实验级定量精修（Rietveld/PDF 全参数模型）；
- 多元素、多相混合与复杂仪器函数拟合；
- 需要严格误差传播与系统误差建模的正式分析。

## R10

正确性直觉与可检验现象：

1. `q -> 0` 时，`S(q)` 应接近有限体系的相干上限（约 `N` 量级）；
2. 晶体应出现显著峰，而非晶峰更宽更弱；
3. fcc 首峰位置应接近 `q_111 = (2pi/a)*sqrt(3)`；
4. 低阶理论峰与检测峰应近似线性对应（斜率接近 1，截距接近 0）；
5. 若反演有效，`a_fit` 应接近 `a_true`。

这些都在 `demo.py` 的断言中直接编码。

## R11

数值稳定策略：

- 使用 `np.sinc(qr/pi)` 避免 `qr -> 0` 时直接除零；
- 对加噪后的 `S(q)` 做轻度 `gaussian_filter1d` 平滑，提升峰检测稳定性；
- Torch 反演对 `a, B` 用 `softplus` 做正值约束；
- 对最少峰数量做保护判断，避免退化输入导致无意义回归。

## R12

关键参数（`XRayConfig`）：

- `lattice_constant_angstrom`：真值晶格常数；
- `n_cells`：超胞尺度，控制原子数与峰锐度；
- `q_min/q_max/n_q`：散射向量采样范围与分辨率；
- `debye_waller_true`：真实热振动衰减参数；
- `gaussian_noise_std`：观测噪声强度；
- `smoothing_sigma`：谱平滑宽度；
- `peak_prominence`：峰值检测阈值；
- `torch_epochs/torch_lr`：反演优化强度。

默认参数目标是“运行快 + 结果稳定 + 物理现象清晰”。

## R13

保证类型说明：

- 近似比保证：N/A（非组合优化问题）；
- 概率成功率保证：N/A（流程本身是确定性的，噪声由固定随机种子生成）。

可执行保证（脚本断言）：

- 首峰位置误差在阈值内；
- 晶体峰强显著高于非晶；
- 峰位回归 `R^2` 足够高；
- Torch 反演可恢复晶格常数并达到可接受 MSE。

## R14

常见失效模式：

1. `n_cells` 太小导致峰不明显，回归点不足；
2. 噪声过大或平滑过弱，峰检测不稳定；
3. `q_max` 过低导致高阶峰缺失；
4. Torch 初值离真值太远或学习率不合适，反演收敛变慢。

排查顺序建议：

1. 先看 `num_detected_peaks`；
2. 再看 `peak_regression_R2` 和首峰误差；
3. 最后看 `torch_lattice_fit_A` 与 `torch_fit_mse`。

## R15

工程化扩展方向：

- 引入元素散射因子 `f_j(q)` 与多组分体系；
- 从有限超胞切换到周期边界 + RDF 路径，降低有限尺寸效应；
- 增加仪器分辨函数、背景项与峰形函数；
- 对接实验数据文件（如 `.xy/.dat/.h5`）做真实拟合流程。

## R16

相关算法与主题：

- XRD Rietveld refinement；
- Pair Distribution Function (PDF) 分析；
- 电子衍射/中子衍射结构因子对比；
- 反问题优化（全谱拟合、贝叶斯反演、多目标约束）。

## R17

`demo.py` MVP 功能清单：

- `generate_fcc_positions`：构造有序 fcc 超胞；
- `generate_amorphous_positions`：构造无序对照结构；
- `pair_distances_upper`：提取所有 `i<j` 距离；
- `debye_structure_factor`：显式实现 Debye 方程；
- `first_fcc_reflections`：生成理论峰位置；
- `find_peaks` + `LinearRegression`：峰识别与一致性评估；
- `torch_fit_lattice_and_dw`：反演 `a` 与 `B`；
- `main`：打印表格、执行断言、给出 PASS/FAIL。

运行方式：

```bash
cd Algorithms/物理-凝聚态物理-0471-X射线散射_(X-ray_Scattering)
uv run python demo.py
```

脚本无需交互输入。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `XRayConfig` 固定晶格常数、超胞规模、`q` 网格、噪声、平滑和 Torch 优化参数，建立单次实验上下文。  
2. `generate_fcc_positions` 与 `generate_amorphous_positions` 分别生成有序/无序坐标集；`pair_distances_upper` 计算所有 `r_ij`。  
3. `debye_structure_factor` 对每个 `q` 显式计算 `S(q)=1+(2/N)sum_{i<j}sin(qr_ij)/(qr_ij)`，并可选乘以 `exp(-B q^2)` 衰减相干项。  
4. `main` 对晶体曲线加入高斯噪声并平滑，同时计算非晶平滑曲线，形成“有序 vs 无序”对照。  
5. `find_peaks` 从晶体 `S(q)` 提取峰位，`first_fcc_reflections` 生成理论 fcc 低阶峰 `q_hkl`，构建峰位对照表。  
6. `LinearRegression` 对“理论峰位 -> 检测峰位”做线性拟合，输出 `R^2` 与 RMSE 作为几何一致性指标。  
7. `torch_fit_lattice_and_dw` 将 Debye 方程写成可微计算图，使用 Adam 迭代优化 `a` 与 `B`，最小化模型谱与观测谱的 MSE。  
8. `main` 汇总打印峰值表、理论对照、反演参数和诊断指标，并执行断言（首峰误差、晶/非晶对比、回归质量、反演精度）作为最终验收。  

第三方库角色是“数值基础设施”而非物理黑盒：`numpy/scipy` 负责数组、平滑和峰检；`pandas` 负责表格；`scikit-learn` 负责线性回归；`torch` 负责可微优化。Debye 散射核心计算在源码中完全显式实现。
