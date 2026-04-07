# 全息原理 (Holographic Principle)

- UID: `PHYS-0380`
- 学科: `物理`
- 分类: `量子引力`
- 源序号: `399`
- 目标目录: `Algorithms/物理-量子引力-0399-全息原理_(Holographic_Principle)`

## R01

全息原理的核心主张是：一个区域可包含的最大信息量（熵）与其边界面积成正比，而不是与体积成正比。  
本条目用一个可运行数值 MVP 把这一主张转成可验证流程：在球对称近似下，显式计算并对比

- Bekenstein bound（有限能量系统的熵上界）；
- Bekenstein-Hawking 面积律（黑洞熵）；
- 体积标度 toy 模型（对照组）。

## R02

MVP 目标是打通以下闭环：

1. 由半径 `R` 生成几何量 `A=4πR^2`, `V=4πR^3/3`；
2. 由 `R` 得到等半径 Schwarzschild 质量 `M_s(R)=c^2R/(2G)`；
3. 令系统质量 `M=f*M_s`（`0<f<1`，未塌缩），计算 `E=Mc^2`；
4. 计算 Bekenstein 上界 bits 与黑洞熵 bits；
5. 验证 `S_Bekenstein/S_BH ≈ f`；
6. 回归检验面积律斜率约 2、体积模型斜率约 3；
7. 输出自动 PASS/FAIL。

## R03

本实现使用的关键公式（SI 制）：

1. Planck 面积：`l_p^2 = G*hbar/c^3`
2. Schwarzschild 等半径质量：`M_s(R)=c^2R/(2G)`
3. Bekenstein bound：
`S <= 2πk_B E R/(hbar c)`
4. Bekenstein-Hawking 熵：
`S_BH = k_B A/(4 l_p^2)`
5. bits 转换：
`bits = S/(k_B ln 2)`

据此得到
`bits_Bekenstein = 2πER/(hbar c ln2)`
与
`bits_BH = A/(4l_p^2 ln2)`。

## R04

理论一致性关系（本条目的核心可检验命题）：

当 `M = f * M_s(R)` 且 `E=Mc^2` 时，

- `bits_Bekenstein / bits_BH = f`

特别地，当 `f=1`（达到黑洞阈值）时，

- `bits_Bekenstein = bits_BH`

这把“Bekenstein 上界”和“黑洞面积律”在球对称阈值处直接连接起来。

## R05

`demo.py` 的输入输出约定（无交互）：

- 输入（脚本内配置）：
1. 半径区间 `R in [1e-2, 1e6] m`，几何采样 `n=160`
2. 质量比例 `f in [0.05, 0.95]`（随机但固定种子）
3. 体积熵密度 toy 参数 `rho_bits = 1e66 bits/m^3`

- 输出：
1. 数据预览表（`R`, `f`, `bits_Bekenstein`, `bits_BH`）
2. 三种回归器的标度斜率结果（sklearn/scipy/torch）
3. toy 体积模型与面积律交叉半径（理论值与数值值）
4. 自动验证表与 `Validation: PASS/FAIL`

## R06

算法主流程（高层）：

1. 生成 `R` 网格并计算 `A, V`；
2. 计算每个 `R` 对应的 `M_s(R)`；
3. 采样子临界比例 `f`，得到 `M=fM_s` 与 `E=Mc^2`；
4. 计算 `bits_Bekenstein` 与 `bits_BH`；
5. 构造对照 `bits_volume = rho_bits * V`；
6. 在 `log10(S)-log10(R)` 空间回归标度斜率；
7. 比较 toy 模型与面积律交叉点（理论/数值）；
8. 执行阈值断言并给出 PASS/FAIL。

## R07

复杂度分析（`N` 为半径采样数，默认 160）：

- 数据生成：`O(N)`
- 指标计算：`O(N)`
- 线性回归：`O(N)`（每个回归器）
- Torch 拟合：`O(TN)`，`T` 为迭代步数（默认 1200）

总时间复杂度约 `O(TN)`，空间复杂度 `O(N)`。在默认配置下可秒级运行。

## R08

数值稳定与实现细节：

1. 半径采用几何间隔（`geomspace`），覆盖多数量级；
2. 在对数回归前所有熵值均严格正，避免 `log` 非法输入；
3. 使用固定随机种子保证复现；
4. 回归使用三条独立路径（sklearn/scipy/torch）交叉校验，降低单实现偏差风险。

## R09

最小工具栈与职责：

- `numpy`：向量化计算几何量、熵和比例；
- `scipy.constants`：`G, hbar, c` 等 SI 常数；
- `scipy.stats.linregress`：一条独立线性回归路径；
- `pandas`：结果表格组织与打印；
- `scikit-learn`：`LinearRegression` 与 `R^2`；
- `torch`：Adam 拟合 `logS = a logR + b`，独立验证斜率。

## R10

运行方式：

```bash
cd Algorithms/物理-量子引力-0399-全息原理_(Holographic_Principle)
uv run python demo.py
```

脚本无命令行参数、无交互输入。

## R11

关键输出字段说明：

1. `ratio_bekenstein_to_bh`：`bits_Bekenstein / bits_BH`
2. `mass_fraction`：`f = M/M_s`
3. `ratio_volume_to_bh`：体积 toy 熵相对黑洞面积熵的比值
4. `slope`：`log10(S)` 对 `log10(R)` 的回归斜率
5. `R_cross_theory`：由解析方程求得的 toy/面积律交叉半径
6. `R_cross_numeric`：离散样本中首次超过面积律的半径

## R12

内置验收条件（`demo.py` 会自动检查）：

1. `max|ratio_bekenstein_to_bh - mass_fraction| < 1e-11`
2. 所有子临界样本满足 `bits_Bekenstein < bits_BH`
3. 面积律斜率误差 `max|slope-2| < 2e-3`
4. 体积模型斜率误差 `max|slope-3| < 2e-3`
5. `R_cross_numeric / R_cross_theory` 偏差小于 `0.15`

全部通过才输出 `Validation: PASS`，否则抛出异常终止。

## R13

结果应体现的物理特征：

1. 在固定 `R` 下，`f` 越大，`bits_Bekenstein` 越接近 `bits_BH`
2. `f<1` 时，Bekenstein 上界严格低于黑洞面积熵
3. 黑洞熵随半径呈平方标度（斜率约 2）
4. 常熵密度体积模型呈立方标度（斜率约 3）
5. 大尺度下体积模型会超过面积律（出现交叉半径）

## R14

模型边界与简化：

1. 仅做球对称、静态、半经典级别的数量级验证
2. 未涉及动态时空、全量量子引力微观态计数
3. toy 体积熵密度 `rho_bits` 仅用于标度对照，不是实在物质模型
4. 不处理旋转/带电黑洞与灰体谱等细节

## R15

可能失败模式：

1. 若修改参数导致 `f>=1`，将失去“子临界”前提，检查 2 可能失败
2. 若采样点过少或范围太窄，回归斜率精度下降
3. 若将 `torch_steps` 设得过低，Torch 斜率可能偏离阈值
4. 若改动 `rho_bits` 与采样区间不匹配，数值交叉点可能不存在

## R16

可扩展方向：

1. 扩展到非球形边界并比较最小包络面积与信息上限
2. 加入 Kerr/Reissner-Nordstrom 情况下的参数化面积律
3. 结合 AdS/CFT 子模型，把边界自由度计数映射到具体场论样本
4. 把 toy 体积熵替换为更真实的热场/凝聚态态密度模型

## R17

交付状态：

1. `README.md`：R01-R18 已完整填写
2. `demo.py`：可直接运行，且包含自动验证逻辑
3. `meta.json`：任务元数据保持与 `PHYS-0380` 一致

目录已具备独立验证条件。

## R18

`demo.py` 源码级算法流（8 步，非黑盒）：

1. `build_dataset` 生成半径网格 `R`，并由 `sphere_area/sphere_volume` 计算 `A,V`。  
2. `schwarzschild_mass_from_radius` 计算阈值质量 `M_s(R)`，再用随机 `f` 构造子临界质量 `M=fM_s` 与能量 `E=Mc^2`。  
3. `bekenstein_bound_bits` 按 `2πER/(hbar c ln2)` 逐点计算上界 bits。  
4. `bekenstein_hawking_bits` 按 `A/(4 l_p^2 ln2)`（其中 `l_p^2=G*hbar/c^3`）逐点计算黑洞面积熵 bits。  
5. 直接构造 `ratio_bekenstein_to_bh`，并与输入 `mass_fraction=f` 做误差比对，验证 `ratio≈f`。  
6. `run_scaling_regressions` 在 `log10(S)-log10(R)` 空间对面积律和体积对照分别做三路拟合：`sklearn`、`scipy`、`torch`，提取斜率并检查是否接近 2 与 3。  
7. `theoretical_crossing_radius` 解解析交叉半径，`first_numeric_crossing_radius` 从离散样本找首个超越点，比较理论/数值一致性。  
8. `main` 汇总 5 条验收断言并打印 `Validation: PASS/FAIL`；任一失败则 `raise RuntimeError`，确保结果可自动验证。  

第三方库只承担通用数值/回归职责，核心物理关系（Schwarzschild 阈值、Bekenstein bound、面积律、交叉半径公式）均在源码中显式展开。
