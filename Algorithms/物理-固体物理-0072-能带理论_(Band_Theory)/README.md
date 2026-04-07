# 能带理论 (Band Theory)

- UID: `PHYS-0072`
- 学科: `物理`
- 分类: `固体物理`
- 源序号: `72`
- 目标目录: `Algorithms/物理-固体物理-0072-能带理论_(Band_Theory)`

## R01

能带理论研究的是电子在周期势场中的量子态：原子级离散能级在晶体中展开为随波矢 `k` 变化的能带 `E_n(k)`。本条目实现一个 1D 最小可运行模型，直接展示“周期势 -> Bloch 哈密顿量 -> 能带与带隙”的算法链路。

## R02

MVP 聚焦于近自由电子框架下的平面波展开法（plane-wave expansion）：

1. 周期势采用 `V(x)=2V1*cos(G0*x)+2V2*cos(2*G0*x)`。
2. 在倒空间基 `|k+G_n>` 下构造有限维哈密顿量。
3. 对每个 `k` 点求本征值，得到前若干条能带。
4. 输出能带样本、带隙、对称性误差与拟合指标。

## R03

核心物理方程：

1. `H = -d^2/dx^2 + V(x)`（无量纲单位，等效 `ħ^2/2m = 1`）。
2. 平面波基中矩阵元
   `H_{nm}(k) = (k + G_n)^2 * delta_{nm} + V_{G_n-G_m}`。
3. 周期势傅里叶分量仅在 `±G0, ±2G0` 非零：
   `V_{±G0}=V1, V_{±2G0}=V2`。
4. 带隙定义（本例）：布里渊区边界处 `Eg = E2 - E1`。

## R04

算法直觉：

1. `V=0` 时，`E=(k+G_n)^2` 是互不耦合的抛物线。
2. 周期势引入 `G` 与 `G±mG0` 的耦合，抛物线在交叉点发生反交叉。
3. 反交叉直接打开带隙，这就是“绝缘/导体差异”的最小数学原型。
4. 在 1D 上先做通流程，有助于后续推广到 3D 晶体与真实势函数。

## R05

本实现的输入与可控参数：

1. `lattice_constant`：晶格常数 `a`。
2. `v1, v2`：周期势傅里叶幅值。
3. `n_harmonics`：平面波截断半径 `N`（基维度 `2N+1`）。
4. `n_kpoints`：第一布里渊区采样点数。
5. `n_bands_report`：输出的低能带条数。

默认参数是“快速可复现”优先，而非高精度材料计算设置。

## R06

复杂度估计：

1. 单个 `k` 点需要对 `M x M`（`M=2N+1`）Hermitian 矩阵对角化，代价约 `O(M^3)`。
2. 全部 `k` 点总成本约 `O(n_k * M^3)`。
3. 本默认值 `N=4, M=9, n_k=241`，因此运行非常快，适合教学和验证。

## R07

`demo.py` 的流程总览：

1. 生成倒格矢索引 `n=-N...N` 与 `k` 网格。
2. 构造势能傅里叶字典 `V_q`。
3. 对每个 `k` 构造 Bloch 哈密顿量并求本征值。
4. 汇总 `bands[k_index, band_index]`。
5. 计算带隙、对称误差、中心区 RMSE、有效质量拟合。
6. 输出两个 DataFrame + 自动断言。

## R08

工具栈（最小但不黑箱）：

1. `numpy`：矩阵构造、数组运算。
2. `scipy.linalg.eigh`：Hermitian 特征值求解。
3. `pandas`：结果表格整理。
4. `scikit-learn`：`mean_squared_error` 计算 RMSE。
5. `torch`：自动微分拟合 `E(k)≈E0+alpha*k^2`，提取曲率。

## R09

关键函数接口：

- `build_fourier_coefficients(v1, v2)`
- `build_bloch_hamiltonian(k, g0, n_indices, coeffs)`
- `solve_band_structure(config)`
- `fit_effective_mass_torch(k, e_band, steps, lr)`
- `build_report_tables(result)`
- `run_checks(result)`

每个函数都对应一个明确物理步骤，避免“单函数大黑盒”。

## R10

正确性校验策略：

1. **反演对称**：`E_n(k)=E_n(-k)`，用最大绝对误差量化。
2. **带隙打开**：区边界 `Eg>0`。
3. **微扰一致性**：弱势近似下 `Eg ≈ 2|V1|`。
4. **中心区近自由电子性**：首带与 `k^2` 的 RMSE 较小。
5. **Torch 拟合可收敛**：损失应降到阈值内。

## R11

异常与边界处理：

1. `n_harmonics < 1`、`n_kpoints < 5`、`n_bands_report` 非法时抛 `ValueError`。
2. `n_bands_report > 2N+1` 时抛 `ValueError`。
3. 任何非有限数（NaN/Inf）将触发断言失败。
4. 若带隙与理论偏差过大，断言会阻止“看似可跑但物理错误”的结果。

## R12

运行方式：

```bash
cd Algorithms/物理-固体物理-0072-能带理论_(Band_Theory)
uv run python demo.py
```

脚本无交互输入，运行后会打印：

1. 参数摘要。
2. 采样能带表（若干 `k` 点）。
3. 指标表（带隙、RMSE、拟合参数、对称误差）。
4. `All checks passed.`

## R13

输出指标解释：

1. `band_gap_zone_boundary`：区边界一、二带差值。
2. `predicted_gap_2|V1|`：弱势近似预估。
3. `gap_relative_error`：数值带隙与预估相对误差。
4. `center_rmse_vs_free`：首带在中心窗口相对 `k^2` 的 RMSE。
5. `inversion_symmetry_max_error`：`E(k)` 与 `E(-k)` 的最大差。
6. `alpha_quadratic_fit`、`effective_mass_ratio_m*/m_free`：由 Torch 拟合得到的曲率与有效质量比。

## R14

最小验收标准：

1. `README.md` 与 `demo.py` 中无占位符残留。
2. `uv run python demo.py` 可直接运行。
3. 终端出现 `All checks passed.`。
4. 指标满足：`Eg>0`、反演误差极小、拟合损失有限。

## R15

参数调优建议：

1. 增大 `v1` 会显著拉大第一带隙。
2. `v2` 主要影响更高能带形状与远离区边界的弯曲。
3. 增大 `n_harmonics` 提升截断精度，但立方增加对角化成本。
4. `n_kpoints` 太低会让带边定位和拟合变差。

## R16

与其他固体模型关系：

1. 与紧束缚（tight-binding）互补：本模型更接近近自由电子极限。
2. 与 Kronig-Penney 同属 1D 周期势建模，但本实现直接在倒空间离散。
3. 与真实 DFT 的关系：都基于 Bloch 定理，但本条目没有自洽电荷密度和交换关联项。
4. 因此它更适合作为“能带计算的算法原型”。

## R17

可扩展方向：

1. 升级到 2D/3D 倒格矢集合。
2. 引入自旋轨道耦合与多轨道基底。
3. 对接 Wannier 化与有效模型提取。
4. 加入可视化（整条 `E-k` 曲线图）和单元测试。
5. 用实验/第一性原理数据反标定 `V_q`。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `solve_band_structure` 先创建 `k` 网格和倒格矢索引 `n=-N...N`，并调用 `build_fourier_coefficients` 得到有限个非零 `V_q`。
2. 对每个 `k`，`build_bloch_hamiltonian` 逐元素写出 `H_nm=(k+G_n)^2*delta_nm+V_{n-m}`，明确展示动能对角项和势能耦合项。
3. `scipy.linalg.eigh` 对每个 Hermitian `H(k)` 求特征值；SciPy 在底层分发到 LAPACK 的对称本征求解路径（如 `syevr/syevd` 家族，取决于构建与驱动），并返回按升序排列的本征值。
4. 代码截取每个 `k` 的前 `n_bands_report` 条，拼成二维数组 `bands[k_index, band_index]`。
5. 通过区边界点的 `E2-E1` 计算数值带隙，并与弱势近似 `2|V1|` 比较，形成可解释物理校验。
6. 用 `sklearn.metrics.mean_squared_error` 计算中心窗口内 `E1(k)` 对自由电子 `k^2` 的 RMSE，量化“近自由电子”程度。
7. `fit_effective_mass_torch` 把 `E1(k)` 在小 `|k|` 区域拟合为 `E0+alpha*k^2`：Torch 用 autograd + Adam 迭代更新参数，输出 `alpha` 与拟合损失，再转成有效质量比 `m*/m_free=1/alpha`。
8. `run_checks` 统一断言对称性、带隙、拟合和有限性，`main` 打印结果并给出 `All checks passed.` 作为验收信号。
