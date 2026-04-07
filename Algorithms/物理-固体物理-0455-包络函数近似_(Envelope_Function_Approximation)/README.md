# 包络函数近似 (Envelope Function Approximation)

- UID: `PHYS-0435`
- 学科: `物理`
- 分类: `固体物理`
- 源序号: `455`
- 目标目录: `Algorithms/物理-固体物理-0455-包络函数近似_(Envelope_Function_Approximation)`

## R01

包络函数近似（Envelope Function Approximation, EFA）是半导体能带理论中连接微观 Bloch 态与器件尺度量子态的核心方法。  
在带边附近，电子总波函数可写成：

\[
\Psi(\mathbf{r}) = u_{n\mathbf{k}_0}(\mathbf{r}) \, F(\mathbf{r})
\]

其中 \(u_{n\mathbf{k}_0}\) 是快速振荡的晶格周期部分，\(F\) 是缓慢变化的包络函数。  
本条目给出一个一维异质结量子阱的最小可运行实现，直接求解包络函数本征值问题。

## R02

本 MVP 的问题设定：

1. 几何：一维有限深方势阱（量子阱）；
2. 势能：阱内 \(V=0\)，阱外 \(V=V_0\)；
3. 有效质量：阱内/垒区分别取常数 \(m_w, m_b\)；
4. 边界：计算域两端施加 Dirichlet（\(\psi=0\)）；
5. 目标：求束缚态能级与包络函数，并验证物理与数值一致性。

默认参数模拟典型 GaAs/AlGaAs 风格量级：`m_w/m0=0.067`、`m_b/m0=0.092`、`V0=0.30 eV`。

## R03

采用 BenDaniel-Duke 形式的包络函数方程：

\[
\left[-\frac{\hbar^2}{2}\frac{d}{dz}\left(\frac{1}{m^*(z)}\frac{d}{dz}\right)+V(z)\right]\psi_n(z)=E_n\psi_n(z)
\]

这一定义天然处理了质量突变界面；在离散形式下可保证厄米性。  
与“常质量 Schr\"odinger 方程”相比，关键差异是动能算符含有 \(1/m^*(z)\) 的空间变化项。

## R04

离散化策略（中心差分 + 节点间平均）：

- 网格：`z_i = z_min + i*dz`；
- 节点间逆质量：
  \[
  \left(\frac{1}{m}\right)_{i+1/2} = \frac12\left(\frac{1}{m_i}+\frac{1}{m_{i+1}}\right)
  \]
- 三对角系数：
  - 主对角：`V_i + pref*(inv_left + inv_right)`
  - 次对角：`-pref*inv_right`
  - `pref = hbar^2/(2*dz^2)`

得到实对称三对角哈密顿量，适合稳定高效本征求解。

## R05

最小工具栈与职责：

- `numpy`：网格、剖面、差分离散与后处理；
- `scipy.linalg.eigh_tridiagonal`：三对角厄米本征值求解；
- `pandas`：状态表和扫描结果表输出；
- `torch.linalg.eigvalsh`：独立后端交叉验证谱一致性；
- `scikit-learn`：线性回归拟合 `E1 ~ a*(1/L^2) + b`。

第三方库只提供线代与回归基础算子；模型构建、离散细节、物理校验路径都在源码显式实现。

## R06

`demo.py` 运行后输出四类结果：

1. 参数与网格信息（`dz`、内点规模、束缚态数量）；
2. 势能/质量剖面采样表；
3. 前几个束缚态表（能量、无限深阱参考、阱内概率、本征残差）；
4. 阱宽扫描表与线性拟合指标（`slope`、`intercept`、`R^2`）。

脚本末尾包含断言，全部通过后打印 `All checks passed.`。

## R07

正确性闭环分为数值与物理两条线：

- 数值线：
  1. SciPy 与 PyTorch 光谱差 `max|E_scipy-E_torch|`；
  2. 本征方程残差 `||Hψ-Eψ||/||Eψ||`；
  3. 多态归一与正交性检查（积分重叠矩阵）。
- 物理线：
  1. 基态能量应满足 `0 < E1 < V0`；
  2. 有限阱束缚态能级应低于同宽无限深阱参考；
  3. 基态应主要局域在阱内（`P_in_well` 高）。

## R08

默认配置 `EnvelopeConfig`：

- `half_domain_nm = 25.0`
- `well_width_nm = 10.0`
- `barrier_height_ev = 0.30`
- `m_well_over_m0 = 0.067`
- `m_barrier_over_m0 = 0.092`
- `n_grid = 801`
- `n_report_states = 3`
- `scan_lowest_eig_count = 18`

设计理由：网格足够细以解析低能束缚态，同时保持计算量轻量。

## R09

复杂度分析（记内点数为 `N`）：

- 剖面构造与三对角装配：`O(N)`；
- 三对角本征分解（全谱）：约 `O(N^2)`；
- PyTorch 稠密交叉验证（一次）：`O(N^3)`（仅用于验证，不用于主路径）；
- 阱宽扫描（只取低能若干本征值）可近似看作 `O(K * N^2)`，`K` 为扫描点数。

默认参数下可在普通 CPU 快速完成。

## R10

运行方式（无交互）：

```bash
cd Algorithms/物理-固体物理-0455-包络函数近似_(Envelope_Function_Approximation)
uv run python demo.py
```

或在仓库根目录执行：

```bash
uv run python Algorithms/物理-固体物理-0455-包络函数近似_(Envelope_Function_Approximation)/demo.py
```

## R11

输出解读建议：

1. `bound-state count`：先确认存在束缚态；
2. `Lowest bound states`：
   - `E_bound_ev`：实际束缚态能量；
   - `E_bound / E_inf`：应小于 1；
   - `P_in_well`：越接近 1 说明越局域；
   - `relative_residual`：应接近机器精度；
3. `max|E_scipy - E_torch|`：后端一致性；
4. `Width scan` 拟合：关注 `slope>0`、`R^2` 高，体现量子限域随宽度变化趋势。

## R12

物理含义（本模型可直接展示）：

- 包络函数把复杂晶格细节压缩到有效质量与带边势垒中；
- 阱宽减小时，基态能量升高（量子限域增强）；
- 势垒有限时，波函数会穿透到阱外，因此能量低于无限深阱参考；
- 不同区域质量差异会改变动能项权重，进而影响束缚能级分布。

## R13

适用范围与限制：

适用：
- 带边附近、单带主导、包络缓慢变化的半导体异质结构；
- 量子阱/量子线器件的快速原型分析。

限制：
- 一维单带模型，未含多带耦合、应变、自旋轨道与自洽电势；
- 界面粗糙、无序、强非抛物性等复杂效应未纳入；
- 不能替代全量 k·p 多带或第一性原理计算。

## R14

常见错误与排查：

1. 直接写成 `-(hbar^2/2m)d2/dz2`，忽略 `m*(z)` 空间变化；
2. 界面处质量平均方式错误导致哈密顿量非厄米；
3. 网格太粗使低能级偏差明显；
4. 计算域太短导致边界条件污染束缚态；
5. 未统一单位（nm、m、eV、J）引发量级错误。

## R15

可扩展方向：

1. 在势阱中加入外电场，求解 Stark 位移；
2. 加入自洽 Poisson 方程形成 Poisson-Schr\"odinger 闭环；
3. 扩展到双阱/超晶格并提取劈裂与隧穿耦合；
4. 推广到多带 k·p 包络函数框架；
5. 引入温度依赖参数 \(m^*(T), V_0(T)\)。

## R16

与相邻模型关系：

- 与 `有效质量近似`（0454）：0454 侧重从 \(E(k)\) 拟合质量张量；本条目固定质量后求空间量子态；
- 与 `k·p 微扰`（0453）：EFA 常可由 k·p 低能展开导出；
- 与 `近自由电子/紧束缚`：前者描述能带形成，本条目描述给定带边参数后的器件尺度束缚态；
- 与 NEGF：EFA 可提供低成本器件哈密顿量输入。

## R17

本条目最小交付清单（已满足）：

1. `README.md` 完整填充 `R01-R18`；
2. `demo.py` 可直接运行，无需交互输入；
3. 使用 `numpy/scipy/pandas/scikit-learn/torch` 最小工具链；
4. 显式实现 BenDaniel-Duke 离散哈密顿量；
5. 提供数值后端交叉验证与物理一致性断言。

## R18

`demo.py` 源码级算法流程（9 步）：

1. `EnvelopeConfig` 固定势阱宽度、势垒高度、两区有效质量和网格规模。  
2. `build_grid()` 生成实空间网格；`build_profiles()` 构造分段常数 `V(z)` 与 `m*(z)`。  
3. `build_bdd_tridiagonal()` 按 BenDaniel-Duke 公式构造三对角哈密顿量：主对角含 `inv_left+inv_right`，副对角含 `-inv_right`。  
4. `solve_tridiagonal()` 调用 `scipy.linalg.eigh_tridiagonal` 对该厄米三对角矩阵求本征值和本征矢（主数值求解路径）。  
5. `normalize_state()` 将内点本征矢补零到全域并做积分归一；`residual_relative_norm()` 计算 `Hψ-Eψ` 相对残差。  
6. 主流程按 `E<V0` 识别束缚态，计算每个态的阱内概率 `P_in_well`，并与 `infinite_well_energy_ev()` 的无限深阱参考比较。  
7. `max_scipy_torch_spectrum_diff()` 将同一三对角矩阵显式扩展为稠密矩阵，调用 `torch.linalg.eigvalsh` 复算全谱，与 SciPy 逐点对比。  
8. `scan_ground_state_vs_width()` 扫描多个阱宽，提取基态能量后用 `LinearRegression` 拟合 `E1 ~ a*(1/L^2)+b`，得到斜率和 `R^2`。  
9. `main()` 汇总 `pandas` 表格并执行断言（后端一致性、残差、束缚态区间、有限阱低于无限阱、局域性与拟合质量），通过后输出 `All checks passed.`。  

说明：虽然调用了 SciPy/Torch 的本征求解器，但哈密顿量离散结构、边界处理、束缚态判定、误差指标与物理验证路径都在源码中逐步展开，不是黑盒一行调用。
