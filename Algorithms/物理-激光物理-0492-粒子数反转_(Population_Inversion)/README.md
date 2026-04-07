# 粒子数反转 (Population Inversion)

- UID: `PHYS-0469`
- 学科: `物理`
- 分类: `激光物理`
- 源序号: `492`
- 目标目录: `Algorithms/物理-激光物理-0492-粒子数反转_(Population_Inversion)`

## R01

粒子数反转（Population Inversion）是激光增益介质的核心条件，指上能级粒子数超过下能级粒子数。对本条目而言，目标不是做全波 Maxwell-Bloch 大型仿真，而是实现一个可运行、可审计、可复现实验趋势的最小模型。

本 MVP 采用三能级速率方程，直接计算 `N2-N1`（上能级减下能级）的时间演化，并扫描泵浦强度展示“反转阈值”行为。

## R02

问题定义：给定三能级介质参数和泵浦速率，求解以下任务：

1. 随时间积分粒子数动力学 `N1(t), N2(t), N3(t)`；
2. 计算反转量 `DeltaN(t)=N2(t)-N1(t)`；
3. 判断稳态是否达到反转（`DeltaN_ss > 0`）；
4. 在泵浦速率网格上估计反转阈值，并与理论近似 `1/tau21` 对比。

## R03

物理建模（本实现约定）：

- `N1`：基态/激光下能级；
- `N2`：激光上能级；
- `N3`：泵浦带；
- `pump_rate`：`N1 -> N3` 泵浦速率；
- `tau32`：`N3 -> N2` 快弛豫寿命；
- `tau21`：`N2 -> N1` 自发衰减寿命；
- `stim_rate`：外部光场导致的受激跃迁等效速率。

假设总粒子数守恒：`N_total = N1 + N2 + N3`。

## R04

速率方程：

- `dN3/dt = pump_rate * N1 - N3/tau32`
- `dN2/dt = N3/tau32 - N2/tau21 - stim_rate * (N2 - N1)`
- `dN1/dt = -pump_rate * N1 + N2/tau21 + stim_rate * (N2 - N1)`

由三式相加可得 `d(N1+N2+N3)/dt = 0`，因此模型内建粒子数守恒。

## R05

阈值直觉：

在 `tau32` 很短时可近似 `N3` 为快变量，稳态分析给出反转符号近似由 `pump_rate - 1/tau21` 决定，因此理论阈值约为：

`pump_threshold ~= 1 / tau21`

脚本会同时输出：
- 理论阈值 `1/tau21`；
- 粗网格数值阈值（首个 `DeltaN_ss > 0` 的泵浦点）；
- 线性化阈值（对近零区域做线性回归外推）。

## R06

算法流程（高层）：

1. 固定一组物理参数并初始化 `N1=N_total, N2=N3=0`；
2. 用 `scipy.integrate.solve_ivp` 积分速率方程；
3. 构造时间序列 `N1,N2,N3,N_total,DeltaN`；
4. 用尾段平均近似稳态，得到 `DeltaN_ss`；
5. 扫描泵浦速率得到阈值曲线；
6. 打印守恒误差、阈值估计和一致性检查结果。

## R07

核心数据结构：

- `ThreeLevelLaserParams`：参数 dataclass；
- 轨迹 `DataFrame` 列：
  - `t, N1, N2, N3, N_total, inversion`；
- 扫描 `DataFrame` 列：
  - `pump_rate, inversion_ss, inversion_max, conservation_max_error, is_inverted`；
- `checks: Dict[str, bool]`：最终验收布尔项。

## R08

正确性校验设计：

- 数值有限性：轨迹全表 `isfinite`；
- 物理可行性：粒子数非负（容忍 `1e-9` 数值误差）；
- 守恒性：`max|N_total(t)-N_total(0)| < 1e-8`；
- 低泵浦不反转：扫描首点 `DeltaN_ss < 0`；
- 高泵浦有反转：扫描末点 `DeltaN_ss > 0`；
- 阈值一致性：数值阈值与理论阈值在网格分辨率内一致。

## R09

复杂度分析：

记时间采样点为 `T`，泵浦扫描点数为 `M`。

- 单次积分后处理复杂度约 `O(T)`；
- 扫描复杂度约 `O(M*T)`；
- 空间复杂度约 `O(T)`（每次轨迹表），扫描汇总 `O(M)`。

由于本任务是教学级 MVP，`M` 与 `T` 都取较小固定值，运行开销很低。

## R10

参数与默认值（`demo.py`）：

- `n_total=1.0`
- `tau32=5e-7 s`
- `tau21=1e-3 s`
- `stim_rate=450 s^-1`
- 代表性单次运行 `pump_rate=1400 s^-1`
- 扫描网格 `pump_rate in [0, 2200]`，共 12 点

这组参数可稳定展示“低泵浦负反转、高泵浦正反转”的 crossing 行为。

## R11

工程取舍：

- 采用三能级速率方程，不引入空间传播和腔模方程；
- `solve_ivp` 仅用于 ODE 数值积分，状态方程由脚本显式实现；
- 不做文件 IO，直接控制台输出，便于批量验证；
- 用 `sklearn.LinearRegression` 只做阈值局部线性外推，非核心物理求解器。

## R12

`demo.py` 主要函数职责：

- `validate_params`：参数合法性检查；
- `rate_equations`：返回三能级速率方程右端；
- `simulate_population`：调用 ODE 积分并生成轨迹表；
- `summarize_trajectory`：提取稳态统计和守恒误差；
- `run_pump_sweep`：泵浦扫描并汇总阈值相关量；
- `estimate_threshold_linear`：线性回归估计阈值；
- `first_positive_threshold`：粗网格首个反转点；
- `main`：组织运行、打印结果、执行最终检查。

## R13

运行方式：

```bash
cd Algorithms/物理-激光物理-0492-粒子数反转_(Population_Inversion)
uv run python demo.py
```

无需任何交互输入。

## R14

输出重点字段说明：

- 轨迹表：`N1,N2,N3` 与 `inversion=N2-N1` 的时间演化；
- `inversion_ss`：尾段平均稳态反转量；
- `conservation_max_error`：总粒子数守恒误差；
- `analytic_threshold`：理论阈值 `1/tau21`；
- `coarse_numerical_threshold`：粗网格阈值；
- `linearized_threshold`：线性外推阈值；
- `all_core_checks_pass`：总验收布尔值。

## R15

最小验收标准：

1. `README.md` 和 `demo.py` 无占位符残留；
2. `uv run python demo.py` 可直接运行结束；
3. 输出中 `all_core_checks_pass=True`；
4. 扫描结果体现阈值行为（低泵浦负反转，高泵浦正反转）。

## R16

当前 MVP 局限：

- 未耦合腔光子数动力学（未显式建模增益钳位和振荡建立过程）；
- 参数是教学尺度的无量纲/等效量，不对应特定材料数据库；
- 未考虑空间非均匀泵浦、横模、温度效应和谱线展宽。

## R17

可扩展方向：

1. 增加光子数方程，形成最小激光振荡模型；
2. 扩展到四能级模型并比较阈值差异；
3. 引入随机噪声源，分析起振延迟统计；
4. 做参数拟合：用实验曲线反推 `tau21/stim_rate`；
5. 将 ODE 扩展为 PDE（空间分布+传播）。

## R18

`demo.py` 源码级算法流程（8 步，非黑盒）：

1. `main` 构造 `ThreeLevelLaserParams`，设置代表性泵浦与扫描网格。
2. `simulate_population` 创建初值 `y0=[N_total,0,0]` 和时间网格 `t_eval`。
3. `solve_ivp` 在每个自适应步调用 `rate_equations`：先计算 `inversion=N2-N1`，再按三条速率方程返回 `[dN1,dN2,dN3]`。
4. 积分结束后，脚本把 `solution.y` 显式组装为 `DataFrame(t,N1,N2,N3,N_total,inversion)`，并未把物理量隐藏在库对象里。
5. `summarize_trajectory` 在轨迹尾段做平均，得到 `inversion_ss`，同时直接计算守恒误差 `max|N_total-N_total(0)|`。
6. `run_pump_sweep` 对每个 `pump_rate` 重复步骤 2-5，生成阈值扫描表并给出 `is_inverted` 标记。
7. `estimate_threshold_linear` 选取 `|inversion_ss|` 最小的 5 个扫描点，用 `LinearRegression` 拟合 `inversion_ss = a*pump + b`，再计算 `-b/a` 得线性化阈值。
8. `main` 汇总理论阈值、粗阈值、线性阈值及全部检查项，打印 `all_core_checks_pass` 作为最终验收信号。
