# 受激辐射 (Stimulated Emission)

- UID: `PHYS-0468`
- 学科: `物理`
- 分类: `量子光学`
- 源序号: `491`
- 目标目录: `Algorithms/物理-量子光学-0491-受激辐射_(Stimulated_Emission)`

## R01

受激辐射的可计算核心是：

1. 在光子通量 `Phi` 存在时，激发态粒子会以 `R_stim ∝ Phi` 的速率发生受激跃迁；
2. 自发辐射速率 `R_sp` 不依赖外场强度；
3. 当 `R_stim > R_sp` 时，受激过程成为主导机制，介质更接近“受控放大”而非“随机辐射”。

本条目用最小模型把这三点直接数值化，而不是停留在定性描述。

## R02

要解决的最小问题分成两部分：

1. 速率交叉：验证 `R_stim / R_sp` 随 `Phi` 从小于 1 穿越到大于 1，并估计交叉通量；
2. 增益阈值：验证介质净增益 `g_net = sigma * DeltaN - alpha` 在 `DeltaN` 穿越阈值后由负变正，导致 `I_out / I_in` 从小于 1 变大于 1。

对应地，`demo.py` 同时输出“速率阈值”和“放大阈值”两组可验收指标。

## R03

MVP 建模边界（机制优先）：

- 使用单频近似和一维传播，不展开全 Maxwell-Bloch 方程；
- 使用等效受激截面 `sigma`、自发寿命 `tau_sp`、损耗 `alpha`；
- 用反转密度 `DeltaN = N2 - N1` 表征介质增益能力；
- 忽略横模、频率失谐、碰撞展宽、热透镜等工程细节。

该边界适合做“受激辐射是否主导”的第一性数值检查。

## R04

`demo.py` 实现的方程：

1. 自发辐射速率（单位体积）：`R_sp = N2 / tau_sp`
2. 受激辐射速率（单位体积）：`R_stim = sigma * Phi * N2`
3. 速率交叉通量：`Phi* = 1 / (sigma * tau_sp)`（由 `R_stim = R_sp` 得到）
4. 净增益系数：`g_net = sigma * DeltaN - alpha`
5. 传播方程：`dI/dz = g_net * I`
6. 闭式传播：`I_out = I_in * exp(g_net * L)`

因此增益阈值为 `DeltaN_th = alpha / sigma`。

## R05

脚本输出 5 类结果：

1. `Phi*` 理论值与数值估计值及相对误差；
2. `DeltaN_th` 理论值与数值估计值及相对误差；
3. 阈值下/阈值上 `gain_ratio = I_out / I_in` 对比；
4. `ln(gain_ratio)` 对 `DeltaN` 的线性斜率与理论 `sigma*L` 的误差；
5. 最终 `Validation: PASS/FAIL`。

## R06

实现策略：

- `StimulatedEmissionConfig`：集中管理模型参数与采样范围；
- `spontaneous_rate / stimulated_rate`：显式实现两类跃迁速率；
- `build_rate_ratio_table`：扫描光子通量并构建 `R_stim/R_sp` 数据表；
- `propagate_intensity_numeric`：用 `scipy.integrate.solve_ivp` 求解 `dI/dz`；
- `build_gain_table`：扫描 `DeltaN`，同时计算数值解与闭式解；
- `make_validation_report`：汇总阈值、斜率、误差并给出 PASS/FAIL。

## R07

优点：

- 直接覆盖受激辐射最关键的“速率主导切换 + 净增益阈值”；
- 模型短小，公式与代码一一对应，便于自动验收；
- 数值解与闭式解并行，能自检积分实现正确性。

局限：

- 未包含相干极化动力学与失谐效应；
- 未建模饱和增益、脉冲传播和空间烧孔；
- 参数是教学量级，非某一具体器件标定模型。

## R08

前置知识：

- 爱因斯坦 A/B 系数与受激/自发辐射概念；
- 光放大中的增益系数与损耗系数；
- 一阶常微分方程与指数增益关系。

依赖环境：

- Python 3.10+
- `numpy`
- `scipy`
- `pandas`

## R09

设增益扫描点数为 `M`、速率扫描点数为 `K`。

- 每个 `DeltaN` 需要一次一维 ODE 积分，成本近似 `O(1)`；
- 增益扫描总复杂度 `O(M)`；
- 速率扫描与统计复杂度 `O(K)`；
- 总复杂度 `O(M + K)`，空间复杂度 `O(M + K)`。

在默认参数下（`M=31, K=25`）运行开销很小。

## R10

数值稳定性与鲁棒性处理：

1. ODE 积分设置 `rtol=1e-9`、`atol=1e-12`；
2. 同时保留闭式解 `exp(gL)`，并记录 `ode_vs_exact_relerr`；
3. 阈值估计采用“邻域线性插值”而非离散点硬切换；
4. 误差归一化时使用安全下界，避免极小分母导致异常放大。

## R11

默认参数（见 `demo.py`）：

- `sigma = 3.0e-20 m^2`
- `tau_sp = 10 ns`
- `alpha = 25 1/m`
- `L = 0.03 m`
- `DeltaN` 扫描：`[0, 1.8e21]` 共 `31` 点
- `Phi` 扫描：围绕 `Phi*` 的 `10^-2` 到 `10^2` 共 `25` 点

由此得到：

- `Phi*_theory = 1/(sigma*tau_sp) = 3.333...e27`
- `DeltaN_th,theory = alpha/sigma = 8.333...e20`

## R12

内置验证门槛：

1. `DeltaN_th` 相对误差 `< 3%`；
2. 阈下 `gain_ratio < 1` 且阈上 `gain_ratio > 1`；
3. `ln(gain_ratio)` 斜率相对误差 `< 2%`；
4. `Phi*` 相对误差 `< 3%`。

四项全部满足才输出 `Validation: PASS`。

## R13

保证类型说明：

- 该条目是“物理关系一致性验证”，不是最优化求解；
- 给定参数后输出是确定性的，不依赖随机采样；
- 保证含义是“与解析关系一致且满足误差阈值”，不是实验器件性能保证。

## R14

常见失效模式：

1. 单位混用（例如把 `sigma` 用成 `cm^2` 而 `DeltaN` 用 `m^-3`）；
2. `alpha`、`L` 设置过大导致指数上溢；
3. 扫描范围没有覆盖阈值，导致交叉估计失败；
4. 直接把材料参数套进简化模型而未做有效参数折算。

## R15

可扩展方向：

- 从常系数 `g_net` 扩展到随 `z` 变化的饱和增益 `g(I,z)`；
- 引入失谐项与线型函数，研究频率选择性受激辐射；
- 与速率方程联立，模拟泵浦-反转-输出的完整时域闭环；
- 加入噪声项，分析 ASE 与信噪比演化。

## R16

相关主题：

- 爱因斯坦系数与详细平衡；
- 光纤放大器与行波放大模型；
- 激光阈值条件与反转粒子数钳位；
- Maxwell-Bloch 方程与半经典激光理论。

## R17

运行方式：

```bash
cd Algorithms/物理-量子光学-0491-受激辐射_(Stimulated_Emission)
uv run python demo.py
```

预期输出包含：

- `theory/estimated Phi*` 与误差；
- `theory/estimated DeltaN_th` 与误差；
- 阈上/阈下 `gain_ratio`；
- `Validation: PASS`。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `StimulatedEmissionConfig` 定义 `sigma, tau_sp, alpha, L` 与两组扫描网格。  
2. `spontaneous_rate` 与 `stimulated_rate` 分别计算 `R_sp`、`R_stim`，把“受激是否主导”变成数值比值。  
3. `build_rate_ratio_table` 在对数网格扫描 `Phi`，生成 `stim_over_sp` 曲线。  
4. `net_gain` 构建 `g_net = sigma*DeltaN-alpha`，把反转与损耗统一到单一增益系数。  
5. `propagate_intensity_numeric` 用 `solve_ivp` 求解 `dI/dz=g_net*I`，得到数值 `I_out`。  
6. `propagate_intensity_closed_form` 给出解析 `I_out`，并在 `build_gain_table` 中与数值结果对照，形成自检误差。  
7. `linear_crossing` 与 `make_validation_report` 估计两类阈值（`Phi*`、`DeltaN_th`），同时检验斜率一致性。  
8. `main` 打印关键表格和误差，按四项标准汇总 `Validation: PASS/FAIL`。  

第三方库职责拆解：

- `numpy`：网格采样、拟合与插值辅助；
- `scipy.integrate.solve_ivp`：仅负责一维 ODE 数值积分；
- `pandas`：仅负责结果表格化与展示。

物理方程、阈值逻辑、验证规则全部在源码中显式实现，不依赖黑盒算法。
