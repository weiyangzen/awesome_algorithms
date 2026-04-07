# 霍金辐射 (Hawking Radiation)

- UID: `PHYS-0378`
- 学科: `物理`
- 分类: `量子引力`
- 源序号: `397`
- 目标目录: `Algorithms/物理-量子引力-0397-霍金辐射_(Hawking_Radiation)`

## R01

霍金辐射是半经典量子引力中的核心结果：在弯曲时空背景下，黑洞事件视界附近的量子场表现出近热辐射行为，使黑洞质量随时间减少。该条目目标是给出一个最小可运行数值 MVP，完成 `M -> T_H -> P_H -> dM/dt -> M(t)` 的可复现计算链路。

## R02

MVP 采用最简 Schwarzschild（无电荷、无自旋）近似，并引入一个有效发射率 `emissivity` 吸收灰体因子和粒子种类细节：

1. 黑洞按热辐射源处理（半经典近似）
2. 辐射功率采用 `P_H ~ 1/M^2` 缩放
3. 质量演化服从 `dM/dt = -P_H/c^2`
4. 与解析型 `dM/dt=-K/M^2` 进行数值交叉验证

## R03

`demo.py` 的输入/输出约定（无交互）：

- 输入（脚本内固定）：
1. 初始质量数组 `M0 = [2.5e5, 5e5, 1e6] kg`
2. 数值配置：`rtol=1e-8`, `atol=1e-10`
3. 采样区间：`t in [0, 0.999 * t_life]`

- 输出：
1. 初始状态摘要表（`T_H`, `P_H`, `r_s`, `A`, `nu_peak`, `lifetime`）
2. ODE 与解析解最大相对误差
3. 第一组质量的时间轨迹采样表

## R04

本实现使用的关键公式：

1. 霍金温度：
`T_H = hbar * c^3 / (8 * pi * G * M * k_B)`

2. 史瓦西半径与视界面积：
`r_s = 2GM/c^2`, `A = 4pi r_s^2`

3. 功率近似：
`P_H = emissivity * hbar * c^6 / (15360 * pi * G^2 * M^2)`

4. 质量损失：
`dM/dt = -P_H / c^2 = -K/M^2`

5. 解析质量轨迹与寿命：
`M(t) = (M0^3 - 3Kt)^(1/3)`, `t_life = M0^3/(3K)`

## R05

`demo.py` 主流程：

1. 定义常量与配置（`HawkingConfig`）
2. 由初始质量计算 `T_H`, `P_H`, `r_s`, `A_horizon`, `lifetime`
3. 通过 `solve_ivp` 积分 `dM/dt`
4. 同步计算解析解 `M_analytic(t)`
5. 计算 `|M_num - M_analytic| / M_analytic` 误差
6. 用 `pandas.DataFrame` 汇总并打印结果

## R06

正确性依据：

1. 温度公式、寿命公式和 `1/M^2` 缩放均来自标准霍金蒸发近似
2. ODE 右端直接由功率守恒 `dE= c^2 dM` 推出
3. 同一模型下存在闭式解析解，可与数值解一一对应比较
4. 输出包含 `max_rel_error`，可直接判断积分器是否偏离理论轨迹

## R07

复杂度分析（设时间采样点为 `N_t`）：

- 时间复杂度：`O(N_t)`（单变量常微分方程）
- 空间复杂度：`O(N_t)`（存储时间与质量轨迹）

本 MVP 的主要成本是 `solve_ivp` 评估 RHS，参数规模很小，CPU 上可瞬时运行。

## R08

数值稳定性处理：

1. 质量下界截断 `min_mass_kg`，避免 `1/M^2` 在 `M -> 0` 爆炸
2. 积分仅到 `0.999 * t_life`，避免终点奇异性主导误差
3. 相对误差分母加入下界 `max(M_analytic, 1e-30)` 防止除零
4. 使用较严格 `rtol/atol`，保证与解析解的可比较精度

## R09

单位与参数约定：

1. 质量单位：`kg`
2. 温度单位：`K`
3. 功率单位：`W`
4. 长度单位：`m`
5. 时间单位：`s`（并额外输出 `years`）

常数来自 `scipy.constants`，确保 SI 单位一致。

## R10

运行方式：

```bash
uv run python Algorithms/物理-量子引力-0397-霍金辐射_(Hawking_Radiation)/demo.py
```

或在目录内运行：

```bash
uv run python demo.py
```

## R11

输出字段说明：

1. `M0_kg`：初始黑洞质量
2. `T_H_K`：霍金温度
3. `P_H_W`：霍金辐射总功率近似
4. `r_s_m`：史瓦西半径
5. `A_horizon_m2`：视界面积
6. `nu_peak_Hz`：按 `B_nu` 表示下的谱峰频率
7. `lifetime_s / lifetime_years`：解析蒸发寿命
8. `max_rel_error`：ODE 轨迹相对解析解的最大误差

## R12

可预期结果特征：

1. `M` 越小，`T_H` 越高（`T_H ~ 1/M`）
2. `M` 越小，`P_H` 越大（`P_H ~ 1/M^2`）
3. 寿命随初始质量呈立方增长（`t_life ~ M0^3`）
4. 轨迹末段蒸发明显加速（因 `|dM/dt| ~ 1/M^2` 增大）

## R13

模型边界与简化：

1. 未处理 Kerr/Reissner-Nordstrom 黑洞（仅 Schwarzschild）
2. 未显式分解粒子物种与真实灰体谱（用 `emissivity` 汇总）
3. 未纳入反冲、自引力回馈与量子引力终态
4. 结果适合教学演示与算法验证，不用于高精度天体推断

## R14

可能失败模式：

1. 若将积分终点逼近 `t_life`，误差可能因奇异行为快速放大
2. 若把 `min_mass_kg` 设得过大，会扭曲末段轨迹
3. 若把 `rtol/atol` 放宽过多，`max_rel_error` 可能显著上升
4. 若给出极端大质量，寿命极长但数值本身仍可计算，仅物理可观测性降低

## R15

最小测试建议：

1. 正值性：`T_H > 0`, `P_H > 0`, `lifetime > 0`
2. 单调性：积分过程中 `M(t)` 应单调下降
3. 一致性：`max_rel_error` 应显著小于 1（通常远小于 `1e-3`）
4. 标度律：将 `M0` 放大 2 倍，寿命应近似放大 8 倍

## R16

可扩展方向：

1. 加入质量依赖灰体因子与粒子阈值，替代常数 `emissivity`
2. 从总功率扩展到频谱积分，输出分频段辐射率
3. 支持旋转黑洞参数，加入超辐射与角动量损失
4. 增加末态模型（例如 Planck 质量残留假设）比较不同终止条件

## R17

交付清单与状态：

1. `README.md`：R01-R18 完整填写
2. `demo.py`：可直接 `uv run python demo.py` 运行
3. `meta.json`：保持 UID/学科/分类/源序号与任务一致

本目录是独立可运行的最小实现，不依赖交互输入。

## R18

`demo.py` 的源码级算法流（不把第三方库视作黑箱）：

1. 在 `hawking_temperature_kelvin`、`hawking_power_w` 中按 SI 常数展开 `T_H` 与 `P_H` 的显式公式，形成 `M -> (T_H, P_H)` 映射。 
2. 在 `evaporation_constant_k` 构造 `K`，将能量关系转为一阶常微分方程 `dM/dt=-K/M^2`。 
3. `integrate_evaporation` 先用解析寿命 `t_life=M0^3/(3K)` 生成时间网格，并把积分上限裁到 `0.999*t_life` 以规避终点奇异。 
4. `solve_ivp` 采用自适应步长（RK 系列默认方法）迭代：每步调用 `mass_loss_rhs` 评估斜率，再据局部截断误差控制步长。 
5. `mass_loss_rhs` 在每次斜率评估时计算当前 `M` 的 `P_H(M)`，再除以 `c^2` 得到 `dM/dt`，并应用质量下界避免除零。 
6. 数值轨迹完成后，用 `analytic_mass_profile` 逐点计算闭式解 `M_analytic(t)=(M0^3-3Kt)^(1/3)`。 
7. 用向量化误差公式 `abs(M_num-M_analytic)/max(M_analytic,1e-30)` 得到逐点误差，并提取 `max_rel_error` 作为主诊断。 
8. `build_summary` 和 `sample_trajectory` 用 `pandas` 组织成结构化表格，确保输出可读、可复核、可用于自动验证。
