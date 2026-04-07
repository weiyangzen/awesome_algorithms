# 近日点进动 (Perihelion Precession)

- UID: `PHYS-0369`
- 学科: `物理`
- 分类: `广义相对论`
- 源序号: `388`
- 目标目录: `Algorithms/物理-广义相对论-0388-近日点进动_(Perihelion_Precession)`

## R01

近日点进动是广义相对论对“行星轨道不是严格闭合椭圆”的经典预测。  
在牛顿两体问题中，轨道在理想条件下按 `2π` 完整闭合；在广义相对论修正下，每一圈近日点会向前旋转一个很小角度 `Δϖ`，长期累积后可观测。

本目录的 MVP 目标是：
- 用显式可追踪的 ODE 数值积分得到每圈 `Δϖ`；
- 与标准弱场解析公式对照；
- 给出水星、金星、地球、火星的每圈与每世纪进动量。

## R02

问题建模输入：
- 中心天体参数：`M_sun`（太阳质量）；
- 行星轨道参数：半长轴 `a`、偏心率 `e`；
- 数值参数：积分圈数 `n_orbits`、误差容限 `rtol/atol`。

输出：
- `precession_numeric_arcsec_per_orbit`：数值积分得到的每圈进动（角秒）；
- `precession_formula_arcsec_per_orbit`：解析公式每圈进动（角秒）；
- 两者相对误差；
- 折算到每世纪的角秒进动量；
- 数值质量指标（守恒量漂移、ODE 残差 RMS）。

## R03

核心公式（弱场、Schwarzschild、测试粒子近似）：

1. 半通径：`p = a(1-e^2)`
2. 无量纲参数：`epsilon = 3GM/(c^2 p)`
3. 轨道方程（以方位角 `phi` 为自变量，`w = p/r`）：
   `d2w/dphi2 + w = 1 + epsilon * w^2`
4. 每圈进动解析近似：
   `Δϖ_GR = 6πGM / (a(1-e^2)c^2)`
5. Kepler 周期（用于换算每世纪）：
   `T = 2π * sqrt(a^3/(GM))`

## R04

算法高层流程：
1. 读取行星 `(a, e)` 并计算 `p`、`epsilon`。
2. 在近日点设初值 `w(0)=1+e, w'(0)=0`。
3. 用 `solve_ivp(DOP853)` 积分 ODE。
4. 用事件 `w'(phi)=0`（方向 `+ -> -`）捕获每次近日点角位置 `phi_k`。
5. 计算 `Δphi_k = phi_{k+1} - phi_k`，再得 `Δϖ_k = Δphi_k - 2π`。
6. 对 `Δϖ_k` 求均值，得到数值每圈进动。
7. 与解析公式比较，计算相对误差。
8. 结合轨道周期换算每世纪进动并输出表格。

## R05

`demo.py` 中的数据结构：
- `PlanetOrbit`：行星名、半长轴（AU）、偏心率。
- `run_planet_suite(...)`：批量处理多个行星，输出 `pandas.DataFrame`。
- `diagnostics` 字典：保存每个行星的数值稳定性指标。

关键函数：
- `precession_formula_rad_per_orbit`：解析公式；
- `integrate_precession_numeric`：数值主流程；
- `_build_perihelion_event`：近日点事件函数；
- `_invariant_energy_like`：ODE 第一积分漂移诊断。

## R06

正确性校验思路：
- 物理一致性：`Δϖ > 0`，且结果量级应与经典经验值一致；
- 数值一致性：数值结果应贴近解析弱场公式；
- 稳定性：守恒量漂移和微分方程残差应足够小。

脚本内置阈值：
- 水星每世纪进动在 `[40, 46]` 角秒范围；
- 解析 vs 数值最大相对误差 `< 2e-4`；
- 最大守恒量漂移 `< 2e-8`；
- 最大 ODE 残差 RMS `< 1e-8`。

## R07

复杂度分析：
- 设单行星积分实际步数为 `N_step`，事件数约为 `N_orbit`，采样点为 `N_sample`。
- 时间复杂度：`O(N_step + N_sample)`（每个行星）。
- 空间复杂度：`O(N_sample)`（密集采样用于诊断）。
- 总成本随行星数量 `K` 线性增长：`O(K * (N_step + N_sample))`。

## R08

边界条件与异常处理：
- 输入约束：`a > 0`、`0 <= e < 1`，否则抛 `ValueError`；
- 事件不足：若捕获到的近日点事件数过少，抛 `RuntimeError`；
- ODE 积分失败：直接抛错并携带 `solve_ivp` 消息；
- 验证阈值失败：抛 `RuntimeError`，避免输出“看起来正常但实际不可信”的结果。

## R09

MVP 取舍：
- 保留“可解释 + 可验证”的核心链路（ODE + 事件检测 + 解析对照）；
- 不引入黑箱天体力学库；
- 不做多体摄动、太阳扁率、自转拖拽等高阶修正；
- 聚焦广义相对论主导项，适合作为教学与基线验证。

## R10

`demo.py` 函数职责分解：
- 参数与几何量：
  - `semimajor_axis_m`, `semi_latus_rectum_m`, `gr_epsilon`。
- 理论公式：
  - `precession_formula_rad_per_orbit`, `orbital_period_days`。
- 数值积分核心：
  - `_gr_orbit_rhs`, `_build_perihelion_event`, `integrate_precession_numeric`。
- 结果组织：
  - `run_planet_suite` 生成 DataFrame；
  - `main` 负责运行、打印和阈值校验。

## R11

运行方式：

```bash
cd Algorithms/物理-广义相对论-0388-近日点进动_(Perihelion_Precession)
uv run python demo.py
```

脚本无交互输入，不依赖外部数据文件。

## R12

输出字段说明：
- `a_AU, e`：轨道参数；
- `epsilon`：无量纲 GR 修正强度；
- `period_days`：Kepler 周期（天）；
- `precession_formula_arcsec_per_orbit`：解析每圈进动；
- `precession_numeric_arcsec_per_orbit`：数值每圈进动；
- `relative_error_formula_vs_numeric`：两者相对误差；
- `precession_*_arcsec_per_century`：折算每世纪进动；
- `perihelion_events_used`：用于估计的近日点事件数量。

## R13

默认参数与样本：
- 太阳引力参数：`mu = G * M_sun`；
- 行星样本：`Mercury/Venus/Earth/Mars`；
- 每个行星积分圈数：`n_orbits = 320`；
- ODE 设置：`DOP853`, `rtol=1e-11`, `atol=1e-13`, `max_step=0.10`。

该设置在当前仓库环境下可稳定给出：
- 水星约 `42.98 arcsec/century`；
- 地球约 `3.84 arcsec/century`；
并与解析值高度一致。

## R14

常见失败模式与修复：
- 失败：事件数量不足。
  - 处理：增大 `n_orbits` 或 `phi_max`（代码中由 `n_orbits` 间接控制）。
- 失败：相对误差偏大。
  - 处理：收紧 `rtol/atol`，减小 `max_step`。
- 失败：守恒量漂移偏大。
  - 处理：检查积分精度参数并提升 `sample_count`。
- 失败：水星结果明显偏离 43 角秒/世纪。
  - 处理：核对单位（AU、SI 常数、角度换算）和轨道参数。

## R15

与其他实现路径对比：
- 对比纯解析公式：
  - 解析快，但不展示“如何从轨道演化中提取进动”。
- 对比天体力学黑箱包：
  - 黑箱方便，但不利于逐步审计每个物理步骤。
- 对比本实现：
  - 既有解析基准，又有事件驱动数值提取，透明且可验证。

## R16

适用场景：
- 广义相对论课程中的“水星近日点进动”复现实验；
- 数值 ODE 求解器精度与物理约束联合验证；
- 作为更复杂轨道模型（多体摄动、PN 更高阶）的基线单元。

不适用场景：
- 高精度天体历表生产；
- 需要包含全部摄动源的观测拟合任务。

## R17

可扩展方向：
- 引入其他摄动项（行星多体、太阳 J2、自旋拖拽）；
- 将 `solve_ivp` 与 `torch` 自动微分结合，做参数反演；
- 扩展为“进动随 `(a,e)` 扫描”的参数相图；
- 加入误差预算分解（模型误差 vs 数值误差）。

## R18

`demo.py` 源码级算法流（9 步，非黑箱）：
1. `main` 固定太阳常数、四个行星轨道参数和积分圈数。  
2. `run_planet_suite` 逐行星调用 `integrate_precession_numeric`。  
3. `integrate_precession_numeric` 先由 `a,e` 计算 `p=a(1-e^2)` 与 `epsilon=3GM/(c^2 p)`。  
4. 构造初值 `w(0)=1+e, w'(0)=0`，对应从近日点出发。  
5. 用 `solve_ivp(DOP853)` 积分 `w'' + w = 1 + epsilon w^2`。  
6. 通过事件函数 `w'(phi)=0` 且方向 `+ -> -` 抓取每个近日点 `phi_k`。  
7. 计算样本 `Δϖ_k = (phi_{k+1}-phi_k)-2π`，对其求均值得数值每圈进动。  
8. `precession_formula_rad_per_orbit` 给出解析值，并在 DataFrame 中对比相对误差、换算每世纪值。  
9. 用 `_invariant_energy_like` 漂移与 ODE 残差 RMS 评估数值质量；再执行阈值断言并输出 `Validation: PASS`。  
