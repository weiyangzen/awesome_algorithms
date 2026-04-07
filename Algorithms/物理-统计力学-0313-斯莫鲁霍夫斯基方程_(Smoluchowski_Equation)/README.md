# 斯莫鲁霍夫斯基方程 (Smoluchowski Equation)

- UID: `PHYS-0310`
- 学科: `物理`
- 分类: `统计力学`
- 源序号: `313`
- 目标目录: `Algorithms/物理-统计力学-0313-斯莫鲁霍夫斯基方程_(Smoluchowski_Equation)`

## R01

斯莫鲁霍夫斯基方程（Smoluchowski Equation）描述过阻尼布朗粒子在势场中的概率密度演化，常被视为 Fokker-Planck 方程在惯性可忽略极限下的形式。  
本条目实现一个最小可运行 MVP：
- 一维谐振子势阱中的概率密度时间演化；
- 显式可审计的通量离散与反射边界；
- 输出质量守恒与向平衡分布收敛的数值证据。

## R02

本目录要解决的问题：
- 输入：固定的物理参数（`k, mu, kBT`）、空间网格和初始分布；
- 演化方程：
  `d_t p(x,t) = -d_x J(x,t)`，
  `J = -D d_x p + mu F(x) p`，其中 `D = mu*kBT`；
- 边界条件：`J(-L,t)=J(L,t)=0`（反射边界）；
- 输出：若干时刻的 `mass/mean_x/var_x/L1/KL` 指标与最终收敛检查。

`demo.py` 无命令行依赖，不需要交互输入。

## R03

本实现采用谐振子势：
- 势能：`U(x) = 0.5 * k * x^2`
- 力：`F(x) = -dU/dx = -k*x`

理论平衡态（有限区间上归一化后）满足玻尔兹曼形式：
`p_eq(x) ∝ exp(-U(x)/kBT)`。

因此，数值求解可通过以下信号验证合理性：
- 概率质量 `∫p dx` 近似保持 1；
- `L1(p(t), p_eq)` 随时间下降；
- 均值 `mean_x` 向 0 回归，方差向平衡值靠近。

## R04

Smoluchowski 连续形式可以写为“守恒律 + 通量”：
- 守恒律：`d_t p + d_x J = 0`
- 通量：`J = J_diff + J_drift`
- 扩散部分：`J_diff = -D d_x p`
- 漂移部分：`J_drift = mu F p`

这种写法的数值优势是：
- 直接保证“总量由边界通量决定”；
- 易于施加反射边界（边界通量置零）；
- 便于做有限体积/差分离散并检查质量守恒。

## R05

算法选择：一维均匀网格上的显式有限体积（face flux）推进。

离散策略：
1. 在单元面计算通量 `J_{i+1/2}`；
2. 用通量差更新单元中心概率密度；
3. 扩散项用中心差分；
4. 漂移项用上风格式（upwind）提高数值稳健性；
5. 每步后做非负剪裁+归一化作为安全护栏。

该方案比黑盒 PDE 求解器更透明，适合算法条目 MVP。

## R06

核心离散公式（`dx` 为空间步长，`dt` 为时间步长）：

- 面通量：
  `J_{i+1/2} = -D * (p_{i+1}-p_i)/dx + v_{i+1/2} * p_upwind`
  其中 `v_{i+1/2} = mu * F(x_{i+1/2})`。

- 更新式：
  `p_i^{n+1} = p_i^n - dt * (J_{i+1/2} - J_{i-1/2})/dx`

- 反射边界：
  `J_{1/2}=0`, `J_{N+1/2}=0`

- 时间步长选择（CFL 风格）：
  `dt <= safety * min(dx^2/D, dx/v_max)`。

## R07

复杂度分析（网格点数 `N`，步数 `T`）：
- 单步计算面通量与更新均为 `O(N)`；
- 总时间复杂度 `O(TN)`；
- 空间复杂度 `O(N)`（若干密度向量+通量向量）。

在默认参数下（`N=241`），脚本可在很短时间内完成。

## R08

数值稳定与边界处理：
- 通过 `choose_stable_dt` 同时考虑扩散和漂移约束；
- 使用上风漂移通量，降低非物理振荡；
- 边界面通量强制为 0，实现反射边界；
- 保留 `total_clipped_negative_mass` 指标追踪非负修正量；
- 若归一化质量非正或非有限，直接抛异常中止。

## R09

MVP 取舍：
- 仅做 1D 谐振子势，不扩展到任意势或高维；
- 仅做前向显式推进，不上隐式稀疏线性求解；
- 不调用 `scipy` 的黑盒 PDE 接口；
- 优先“可读、可跑、可审计”而非“最高精度/最高阶收敛”。

## R10

`demo.py` 主要函数职责：
- `SimulationConfig`：集中管理全部参数；
- `equilibrium_density`：构造玻尔兹曼平衡分布；
- `choose_stable_dt`：按 CFL 风格自动选步长；
- `compute_flux_faces`：在 staggered faces 计算总通量；
- `smoluchowski_step`：执行单步时间推进；
- `summarize_state`：计算质量、均值、方差、`L1/KL` 指标；
- `run_simulation`：主循环并记录检查点；
- `main`：打印配置、结果表和通过性检查。

## R11

运行方式（无交互输入）：

```bash
cd Algorithms/物理-统计力学-0313-斯莫鲁霍夫斯基方程_(Smoluchowski_Equation)
uv run python demo.py
```

或在仓库根目录直接运行：

```bash
uv run python Algorithms/物理-统计力学-0313-斯莫鲁霍夫斯基方程_(Smoluchowski_Equation)/demo.py
```

## R12

输出字段解释：
- `time`：检查点时间；
- `mass`：`∫p dx`（应接近 1）；
- `mean_x`：分布均值；
- `var_x`：分布方差；
- `l1_to_equilibrium`：`∫|p - p_eq| dx`；
- `kl_to_equilibrium`：`KL(p || p_eq)`（离散积分近似）；
- `total clipped negative mass`：数值护栏触发强度；
- `equilibrium approach check`：最终 `L1` 是否小于初始 `L1`。

## R13

建议最小验证集：
1. 默认参数（仓库内置）检查是否稳定收敛；
2. 将 `initial_mean` 改到正侧，验证均值仍回归 0；
3. 增大 `k_spring`（更陡势阱）验证 `dt` 自动变小；
4. 减小 `grid_size` 验证粗网格下仍可运行但精度下降。

异常测试建议：
- `grid_size < 5`、`total_time <= 0`、`kbt <= 0` 应抛错。

## R14

关键可调参数：
- `grid_size`：空间分辨率；
- `total_time`：仿真时长；
- `k_spring`：势阱强度；
- `mobility` 与 `kbt`：共同决定 `D=mu*kBT`；
- `cfl_safety`：稳定裕度（越小越稳，越慢）；
- `checkpoint_times`：输出采样时刻。

调参经验：
- 若出现较大负概率修正，降低 `cfl_safety` 或增大 `grid_size`；
- 若收敛慢，增大 `total_time`；
- 若要更细节曲线，可增加 `checkpoint_times` 密度。

## R15

与常见替代方案对比：
- 隐式差分/Crank-Nicolson：稳定性更强，可用更大 `dt`，但实现和线性求解更复杂；
- 谱方法：平滑势场下精度高，但边界和非线性处理更讲究；
- 粒子法（Langevin 轨道采样）：直观但统计噪声大，需要大量样本。

本条目选择显式通量法，优先透明度与教学可验证性。

## R16

典型应用背景：
- 胶体颗粒在外势中的扩散与漂移；
- 化学反应坐标上的过阻尼动力学近似；
- 生物分子在有效势能面上的概率演化；
- 作为更复杂 Fokker-Planck/随机动力学模型的基线模块。

## R17

可扩展方向：
1. 改为隐式或半隐式推进，提升刚性问题稳定性；
2. 支持任意势函数 `U(x)`（用户可注入）；
3. 扩展到二维/三维并使用稀疏算子；
4. 增加收敛阶实验（网格加密下误差标度）；
5. 与 Langevin 粒子模拟做交叉验证。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 构造 `SimulationConfig`，调用 `run_simulation`。  
2. `run_simulation` 创建空间网格 `x`，由 `D=mu*kBT` 得到扩散系数。  
3. `choose_stable_dt` 根据 `dx^2/D` 与 `dx/v_max` 计算稳定时间步长和总步数。  
4. 用 `gaussian_density` 初始化 `p(x,0)`，并由 `equilibrium_density` 计算有限区间平衡分布 `p_eq`。  
5. 每个时间步在 `compute_flux_faces` 中计算面通量：扩散中心差分 + 漂移上风通量，并在两端施加 `J=0`。  
6. `smoluchowski_step` 依据通量散度更新 `p`，执行非负剪裁与归一化，累计负概率修正量。  
7. 到达检查点时，`summarize_state` 计算 `mass/mean_x/var_x/L1/KL` 并写入 `DataFrame`。  
8. `main` 打印检查点表和最终判据（`final L1 < initial L1`），确认系统向玻尔兹曼平衡靠近。  

实现中没有调用第三方“黑盒 Smoluchowski 求解器”；通量构造、边界条件、时间推进与评估指标均在源码中显式展开。
