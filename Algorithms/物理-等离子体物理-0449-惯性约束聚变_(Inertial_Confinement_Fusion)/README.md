# 惯性约束聚变 (Inertial Confinement Fusion)

- UID: `PHYS-0429`
- 学科: `物理`
- 分类: `等离子体物理`
- 源序号: `449`
- 目标目录: `Algorithms/物理-等离子体物理-0449-惯性约束聚变_(Inertial_Confinement_Fusion)`

## R01

惯性约束聚变（ICF）的核心思路是：先用短脉冲高功率驱动（激光或离子束）把燃料靶丸快速压缩到高密度，再依靠“惯性约束”的短时间窗口完成聚变燃烧。  
本目录的 MVP 不追求完整辐射流体代码，而是实现一个可运行、可追踪、可检查的 `0D 热斑模型`：

- 给定球对称内爆轨迹 `R(t)`；
- 演化 DT 粒子数、alpha 粒子数与总内能；
- 同时计算压缩做功、alpha 自加热、轫致辐射损失；
- 输出烧蚀分数、峰值温度、`rhoR`、聚变功率与点火裕量指标。

## R02

本 MVP 要回答的不是“工程上能否点火”，而是“最小算法链路是否自洽”：

1. 内爆压缩是否能把热斑温度与密度推高到 DT 反应显著区间；
2. alpha 沉积功率与辐射损失的竞争关系如何随 `rhoR` 变化；
3. 燃料消耗与聚变产额是否与时间轨迹一致；
4. 在固定简化模型下，脚本能否稳定给出可复现指标。

## R03

状态变量选择为：

- `N_D`：氘核总数；
- `N_T`：氚核总数；
- `N_alpha`：alpha 粒子总数；
- `E`：热斑总内能（J）。

几何量采用给定轨迹而非自洽流体动量方程：

- `R(t)`：半余弦压缩到停滞半径，再线性回弹；
- `V(t)=4/3*pi*R(t)^3`。

该设计保留了 ICF 的主导能量学与反应动力学路径，同时把模型复杂度控制在单文件 MVP 可审计范围。

## R04

`demo.py` 使用的核心方程如下。

1. 反应速率（体积分布假设）

`R_f = N_D * N_T * <sigma v>(T) / V`

2. 粒子数演化

`dN_D/dt = -R_f`  
`dN_T/dt = -R_f`  
`dN_alpha/dt = R_f`

3. 内能演化

`dE/dt = P_comp + P_alpha - P_brem`

其中：

- `P_comp = -p * dV/dt`（压缩做功）；
- `P_alpha = f_alpha(rhoR) * E_alpha * R_f`；
- `P_brem ~ C_brem * Z_eff^2 * n_e * n_i * sqrt(T_eV) * V`。

4. 温度由理想等离子体内能反推

`E = 3/2 * (n_i + n_e) * k_B * T * V`

5. alpha 沉积分数

`f_alpha = 1 - exp(-rhoR / rhoR_stop)`。

## R05

DT 反应性 `<sigma v>` 没有调用黑盒核数据库接口，而是在源码中采用“对数插值的表格代理模型”：

- 温度网格 `T_keV in [1, 100]`；
- 给定一组代表性 `m^3/s` 反应性样本；
- 在 `log(T)-log(<sigma v>)` 空间线性插值。

这样做的目的：

- 保证代码完整自包含，无外部数据文件依赖；
- 维持 ICF 参数扫描所需的单调与量级合理性；
- 清晰展示“温度 -> 反应性 -> 反应功率”的算法链路。

## R06

算法流程（高层）如下：

1. 构造配置并做参数合法性检查；
2. 根据初始密度、半径和温度建立初始 `N_D, N_T, N_alpha, E`；
3. 将状态缩放为无量纲 `x_D=N_D/N0, x_T=N_T/N0, x_alpha=N_alpha/N0, eps=E/E0`，再用 `solve_ivp(Radau)` 积分；
4. 每个时间点由 `R(t)` 与状态量反算 `T, rho, rhoR, R_f, P_alpha, P_brem`；
5. 生成轨迹表与摘要指标；
6. 执行阈值检查（有限性、烧蚀分数、温度、`rhoR`、产额）；
7. 打印结果并给出 `Validation: PASS`。

## R07

复杂度（设时间采样点数 `N`、ODE 内部步数 `K`）:

- ODE 求解主成本：`O(K)`；
- 逐点后处理：`O(N)`；
- 表格与摘要统计：`O(N)`。

总体时间复杂度可写作 `O(K+N)`，空间复杂度 `O(N)`。默认参数在普通 CPU 上可秒级运行。

## R08

数值稳定性策略：

- 使用 `Radau` 处理燃烧与能量耦合的刚性区段；
- 对状态做无量纲缩放（`N/N0`, `E/E0`），降低刚性求解中的数值尺度病态；
- 通过 `max_step` 限制单步过大跨越；
- 对反应率增加“可用燃料耗尽上限”（`~ min(N_D,N_T)/1e-12`）避免局部数值爆冲；
- 对粒子数与内能使用软下限保护，避免数值负值导致非物理崩溃；
- 对温度与密度相关公式设置最小正值，防止除零和 `sqrt` 异常；
- 统一在后处理中检查 `NaN/Inf`。

## R09

模型取舍与边界：

- 这是 `0D` 热斑模型，不解 1D/2D 辐射流体偏微分方程；
- 几何轨迹 `R(t)` 是外给函数，未包含驱动激光吸收与烧蚀层动力学；
- `<sigma v>` 与轫致辐射采用工程近似代理，不替代高保真核数据；
- 未建模混合污染、热传导、电子-离子非平衡、非局域输运与流体不稳定性。

因此本实现适合作为“算法骨架与量纲检查”基线，而不是点火设计工具。

## R10

`demo.py` 的主要函数职责：

- `validate_config`：参数范围检查；
- `radius_and_velocity`：内爆/回弹轨迹与 `dR/dt`；
- `dt_reactivity_m3_s`：DT 反应性代理；
- `bremsstrahlung_power_density_w_m3`：辐射损失模型；
- `derived_quantities`：由状态恢复所有物理派生量；
- `rhs`：构建 ODE 右端；
- `run_simulation`：调用 `solve_ivp` 完成积分；
- `postprocess`：构建轨迹表和摘要指标；
- `run_checks`：自动验收；
- `main`：一键执行与打印结果。

## R11

运行方式（无交互）：

```bash
cd Algorithms/物理-等离子体物理-0449-惯性约束聚变_(Inertial_Confinement_Fusion)
uv run python demo.py
```

脚本会输出：

- 配置摘要；
- `Summary Metrics`（烧蚀分数、产额、峰值温度、峰值 `rhoR` 等）；
- `Trajectory Samples`（时间采样轨迹）；
- 末行 `Validation: PASS`。

## R12

关键输出指标解释：

- `burn_fraction`：平均燃耗分数；
- `yield_MJ`：按 `N_alpha * 17.6 MeV` 估算的总聚变产额；
- `peak_temperature_keV`：轨迹最高温度；
- `peak_rhoR_g_cm2`：轨迹最高面密度（alpha 约束关键指标）；
- `peak_fusion_power_GW`：峰值聚变总功率；
- `time_of_peak_power_ns`：峰值功率出现时间；
- `ignition_margin_at_stag = P_alpha/P_brem`（停滞时点火裕量指标）；
- `hot_tau_ns_above_5keV`：高温窗持续时间；
- `n_tau_m3_s`：简化的 `n*tau` 指标。

## R13

MVP 内置最小验收条件：

1. 所有输出必须是有限数；
2. `burn_fraction` 必须在 `[0,1)`；
3. `peak_temperature_keV > 1.0`；
4. `peak_rhoR_g_cm2 > 0.05`；
5. `yield_MJ > 0`。

任一不满足会抛 `RuntimeError`；全部满足则输出 `Validation: PASS`。

## R14

关键参数调节建议：

- `radius_stagnation_m` 越小，压缩比越高，通常 `rho` 与 `T` 更高；
- `t_stagnation_s` 决定压缩速率，过快可能导致能量项剧烈；
- `alpha_stop_areal_density_kg_m2` 决定 alpha 沉积效率；
- `rho_initial_kg_m3` 与 `temperature_initial_keV` 共同决定初始反应门槛；
- `max_step_s/rtol/atol` 控制求解稳定性与耗时。

调参顺序建议：先调几何压缩参数，再调输运/反应参数，最后收紧 ODE 容差。

## R15

与替代方案对比：

- 完整辐射流体代码：精度高但实现复杂、依赖重；
- 纯静态 Lawson 判据：简单但没有时间过程；
- 本 MVP（0D 时间演化）：
  - 优点：可运行、可解释、可做参数扫描；
  - 缺点：空间分布与多物理细节不足。

因此该版本更像“教学/原型中的动力学中间层”。

## R16

适用场景：

- ICF 基本量纲与机制教学（压缩、点火裕量、燃耗）；
- 快速参数敏感性探索（停滞半径、初始密度、停滞时刻）；
- 作为更高保真代码前的 sanity-check 基线。

不适用场景：

- 设计级点火窗口预测；
- 需要空间分辨的热点-壳层耦合分析；
- 需要对实验诊断信号进行高精度反演。

## R17

交付内容与约束对应：

- `README.md` 保持 `R01-R18` 全部完整；
- `demo.py` 无占位符，可直接运行；
- `meta.json` 保持与任务元数据一致；
- 目录内实现自包含，不依赖交互输入。

工具栈说明：

- `numpy`：数值计算；
- `scipy`：ODE 积分；
- `pandas`：结构化输出与表格打印。

## R18

`demo.py` 源码级算法流程（9 步）：

1. `main` 构造 `ICFConfig` 并调用 `validate_config`，锁定几何、初值、输运与求解参数。  
2. `run_simulation` 由初始密度和半径计算初始质量与粒子数，构造尺度 `N0,E0` 并建立无量纲状态 `y0=[x_D,x_T,x_alpha,eps]`。  
3. `solve_ivp(method="Radau")` 在 `[0, t_end]` 上积分 `rhs`，其中轨迹点由 `t_eval` 统一采样。  
4. `rhs` 每次调用 `derived_quantities`：先由 `radius_and_velocity` 得到 `R(t), dR/dt` 与 `V(t)`。  
5. `derived_quantities` 再由状态量反推出 `n_D,n_T,n_i,n_e,T,rho,rhoR`，并用 `dt_reactivity_m3_s` 得到 `<sigma v>`。  
6. 根据 `R_f=N_D*N_T*<sigma v>/V` 计算源项，并对 `R_f` 施加“燃料耗尽速率上限”后得到 `dx_D/dt,dx_T/dt,dx_alpha/dt,deps/dt`。  
7. 积分结束后 `postprocess` 对每个时间点重建轨迹表，计算 `burn_fraction,yield,peak_T,peak_rhoR,peak_power,n_tau` 等指标。  
8. `run_checks` 对有限性与最小物理阈值做自动验收，失败即抛异常阻止静默错误。  
9. `main` 打印摘要表和采样轨迹，最终输出 `Validation: PASS`，完成一次可复现实验闭环。  

第三方库并未把算法黑盒化：反应、沉积、辐射、做功和状态方程均在源码中显式写出；`scipy` 只承担通用 ODE 时间推进器角色。
