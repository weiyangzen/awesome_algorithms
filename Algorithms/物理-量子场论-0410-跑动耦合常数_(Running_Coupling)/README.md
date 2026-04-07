# 跑动耦合常数 (Running Coupling)

- UID: `PHYS-0391`
- 学科: `物理`
- 分类: `量子场论`
- 源序号: `410`
- 目标目录: `Algorithms/物理-量子场论-0410-跑动耦合常数_(Running_Coupling)`

## R01

跑动耦合常数描述的是：耦合参数会随能标 `mu` 改变，而不是常数。
在一圈近似下，常见形式可写为：

`d alpha / d ln(mu) = c * alpha^2`

其中 `c` 由理论决定：
- `c > 0`（典型 QED）时，耦合随能标升高而增大；
- `c < 0`（典型 QCD）时，耦合随能标升高而减小（渐近自由）。

本条目给出一个最小可运行 MVP，包含解析解与数值积分（RK4）对照。

## R02

本目录实现的问题定义：
- 输入：
  - 参考能标 `mu0 > 0`；
  - 初始耦合 `alpha0 > 0`；
  - 目标能标网格 `mu_grid`（正且严格单调）；
  - 一圈 beta 系数 `c`。
- 方程：
  - `d alpha / d ln(mu) = c * alpha^2`。
- 输出：
  - 数值积分得到的 `alpha(mu)`；
  - 解析解 `alpha_exact(mu)`；
  - 最大相对误差、单调性检查结果、Landau 极点估计。

`demo.py` 内置 QED-like 与 QCD-like 三个固定案例，无需交互输入。

## R03

数学基础：

1. 一圈 RG 方程：`d alpha / d ln(mu) = c * alpha^2`。  
2. 变量分离可得解析解：  
   `alpha(mu) = alpha0 / (1 - c * alpha0 * ln(mu / mu0))`。  
3. 极点位置（若分母为 0）：  
   `mu_pole = mu0 * exp(1 / (c * alpha0))`。  
4. QED 常见一圈系数：`c_qed = 2*Nf/(3*pi)`。  
5. QCD 常见一圈系数（按 `d/dln(mu)` 记法）：  
   `c_qcd = -(11 - 2*Nf/3)/(2*pi)`。

## R04

算法流程（高层）：

1. 为每个案例构造对数能标网格 `mu_grid = geomspace(mu0, mu_end, N)`。  
2. 用 RK4 在变量 `t = ln(mu)` 上积分 ODE，得到 `alpha_numeric`。  
3. 用闭式解计算 `alpha_exact`。  
4. 计算逐点相对误差并取最大值。  
5. 进行单调性检查（与 `c` 和扫描方向一致）。  
6. 计算一圈极点估计 `mu_pole`。  
7. 打印每个案例结果并做全局汇总。

## R05

核心数据结构：

- `CaseConfig`（`@dataclass`）：
  - `name`：案例名称；
  - `mu0`、`alpha0`：初始条件；
  - `mu_end`、`num_points`：扫描区间与离散粒度；
  - `c`：一圈 beta 系数。
- `CaseResult`（`@dataclass`）：
  - `alpha_end_numeric`、`alpha_end_analytic`；
  - `max_rel_error`；
  - `monotonic_ok`；
  - `pole_scale`。
- `numpy.ndarray`：保存 `mu_grid`、`alpha_numeric`、`alpha_exact`。

## R06

正确性要点：

- 方程与解析解一致：解析公式直接来自一阶可分离 ODE。  
- RK4 在 `ln(mu)` 网格上积分同一方程，误差可由解析解直接审计。  
- 通过 `max relative error` 验证数值实现没有偏离方程定义。  
- 通过单调性检查验证物理趋势：
  - QED-like UV：应上升；
  - QCD-like UV：应下降；
  - QCD-like IR（逆向扫描）：应上升。  
- 若区间跨越极点，解析分母会触零，代码会显式报错而非静默输出无效值。

## R07

复杂度分析（每个案例 `N = num_points`）：

- RK4 每步常数次函数评估，单案例时间复杂度 `O(N)`。  
- 解析解逐点计算，时间复杂度 `O(N)`。  
- 总体时间复杂度 `O(N)`，空间复杂度 `O(N)`（存储网格与耦合序列）。

## R08

边界与异常处理：

- `mu0`、`alpha0` 非正或非有限：`ValueError`。  
- `mu_grid` 非 1D、长度不足、非单调、含非正值：`ValueError`。  
- 解析解分母 `<= 0`（跨越 Landau 极点）：`ValueError`。  
- RK4 出现非有限或非正耦合：`RuntimeError`。  
- `nf <= 0`：`ValueError`（QED/QCD 系数函数中）。

## R09

MVP 取舍：

- 仅实现一圈 beta 函数，不做两圈/阈值匹配。  
- 用 `numpy` + 手写 RK4，避免黑盒求解器，便于审计。  
- 选 3 个固定案例覆盖 UV/IR 趋势与符号差异。  
- 不做参数拟合或实验数据同化，保持“最小但诚实”的算法演示。

## R10

`demo.py` 主要函数职责：

- `beta_coeff_qed / beta_coeff_qcd`：生成一圈系数 `c`。  
- `running_coupling_analytic`：计算闭式跑动解。  
- `running_coupling_rk4`：在 `ln(mu)` 上做 RK4 数值积分。  
- `estimate_one_loop_pole_scale`：估计一圈极点位置。  
- `monotonicity_check`：校验趋势方向。  
- `run_case`：执行单案例并输出关键指标。  
- `print_summary`：汇总所有案例的误差与通过状态。  
- `main`：组装案例并驱动全流程。

## R11

运行方式：

```bash
cd Algorithms/物理-量子场论-0410-跑动耦合常数_(Running_Coupling)
uv run python demo.py
```

脚本不读取输入参数，不会请求交互输入。

## R12

输出字段说明：

- `c`：方程 `dα/dlnμ = cα²` 中的系数。  
- `mu range`：扫描起止能标与离散点数。  
- `alpha(mu0)`：初始耦合。  
- `alpha(mu_end) numeric/analytic`：终点的数值解与解析解。  
- `max relative error`：全区间最大逐点相对误差。  
- `monotonic trend check`：趋势是否符合预期。  
- `one-loop pole scale estimate`：按一圈公式估计极点位置。  
- `overall pass flag`：综合误差阈值与单调性检查后的结果。

## R13

内置测试案例：

- `QED-like UV run`：`c > 0`，从 `1 GeV` 跑到 `1e6 GeV`，验证 UV 增长。  
- `QCD-like UV run`：`c < 0`，从 `M_Z` 跑到 `1e4 GeV`，验证 UV 变小（渐近自由）。  
- `QCD-like IR run`：`c < 0`，从 `M_Z` 跑到 `1 GeV`，验证 IR 增强。

建议补充测试：
- 故意把区间扩展到极点附近，验证报错路径；
- 把 `num_points` 降到很小，观察 RK4 误差上升；
- 对比 Euler / RK4 收敛阶差异。

## R14

可调参数与调参建议：

- `num_points`：网格点数，越大数值误差越小但耗时略增。  
- `mu0`、`mu_end`：决定扫描方向与跨度。  
- `alpha0`：初值，越接近极点区域越刚性。  
- `c`：理论参数，决定跑动方向和强度。

调参建议：
- 若 `max relative error` 偏大，优先提高 `num_points`。  
- 若接近极点，缩小扫描区间避免跨越分母零点。  
- 若需要更真实 QCD，下一步应加入两圈项和阈值匹配。

## R15

方法对比：

- 对比纯解析法：
  - 一圈方程有闭式解，解析法最准确；
  - 但数值积分框架可无缝扩展到两圈/阈值分段等无闭式情形。  
- 对比黑盒 ODE 求解器：
  - 黑盒调用更短；
  - 本实现更透明，可逐步审计 `beta`、步进和稳定性。  
- 对比高阶现象学工具：
  - 高阶工具更物理完备；
  - 本 MVP 更适合作为教学和算法基线。

## R16

典型应用场景：

- 量子场论课程中演示“耦合常数为何会跑动”。  
- RG 数值模块开发前的最小正确性基线。  
- 验证 UV/IR 趋势与 Landau 极点概念。  
- 作为后续两圈、阈值匹配、拟合任务的骨架代码。

## R17

可扩展方向：

- 加入两圈 beta：`dα/dlnμ = c1 α² + c2 α³`。  
- 加入阈值匹配（不同 `Nf` 分段）。  
- 支持从实验值反推 `Lambda` 参数。  
- 输出 CSV 并绘制 `alpha(mu)` 曲线。  
- 扩展到 `g`、`alpha_s`、`y_t` 等多耦合联立 RG 方程。

## R18

`demo.py` 源码级算法流程（9 步）：

1. `main` 调用 `beta_coeff_qed/beta_coeff_qcd` 计算一圈系数，并构造 3 个固定 `CaseConfig`。  
2. 对每个案例，`run_case` 用 `np.geomspace` 建立 `mu_grid`（对数能标网格）。  
3. `running_coupling_rk4` 先做网格与初值检查，再转为 `t = ln(mu)` 变量。  
4. 在每个网格步上执行 RK4：计算 `k1..k4`，并更新 `alpha[i]`。  
5. 同一网格上，`running_coupling_analytic` 计算闭式解 `alpha_exact = alpha0 / (1 - c alpha0 ln(mu/mu0))`。  
6. `relative_error` 计算逐点误差，`run_case` 取 `max_rel_error` 作为数值正确性指标。  
7. `monotonicity_check` 根据 `sign(c)` 与扫描方向检查序列趋势是否与 RG 预期一致。  
8. `estimate_one_loop_pole_scale` 给出 `mu_pole = mu0 * exp(1/(c*alpha0))` 的一圈估计。  
9. `print_summary` 汇总所有案例，输出最差误差、趋势检查和 `overall pass flag`。
