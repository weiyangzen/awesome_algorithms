# 涨落-耗散定理 (Fluctuation-Dissipation Theorem)

- UID: `PHYS-0302`
- 学科: `物理`
- 分类: `统计力学`
- 源序号: `305`
- 目标目录: `Algorithms/物理-统计力学-0305-涨落-耗散定理_(Fluctuation-Dissipation_Theorem)`

## R01

涨落-耗散定理（FDT）把“平衡态自发涨落”与“外扰下线性响应”建立定量联系。  
本目录选择最小可解析模型：过阻尼谐势阱中的 Langevin 粒子，验证经典 FDT 形式：

- 响应函数积分（阶跃响应）`χ(t)` 与平衡自相关 `C(t)=<x(0)x(t)>` 满足  
  `χ(t) = (C(0) - C(t)) / (k_B T)`。

这是一条可直接数值验证、且与统计力学课程高度一致的版本。

## R02

任务定义（本实现）：

- 输入：`k_B T, μ, k, dt, n_steps_eq, n_steps_resp, n_ensembles, step_force`。
- 输出：  
  1) 由平衡轨迹估计的 `C(t)`；  
  2) 由外加小阶跃力估计的 `χ_est(t)`；  
  3) 由 FDT 右端得到的 `χ_fdt(t)=(C(0)-C(t))/k_B T`；  
  4) 二者误差与解析解对照。

目标不是做大框架，而是给出可运行、可解释、可核验的最小原型（MVP）。

## R03

连续模型（过阻尼 OU 过程）：

`dx/dt = -μk x + μ f(t) + sqrt(2 μ k_B T) ξ(t)`，其中 `ξ` 为白噪声。

平衡（`f=0`）下有解析结果：

- `C(t) = (k_B T / k) exp(-μk t)`
- 对阶跃力 `f(t)=f0 Θ(t)`，线性响应（易感率）  
  `χ(t) = δ<x(t)>/f0 = (1/k)(1-exp(-μk t))`

因此 FDT 立刻给出：

- `(C(0)-C(t))/k_B T = (1/k)(1-exp(-μk t)) = χ(t)`。

## R04

离散化采用解析一致的 OU 更新（稳定且无条件）：

- `a = exp(-λ dt)`, `λ = μk`
- `x_{n+1} = a x_n + drift + σ η_n`, `η_n ~ N(0,1)`
- `σ = sqrt((k_B T / k)(1-a^2))`
- 无外力时 `drift = 0`
- 阶跃力时 `drift = (f0/k)(1-a)`

该离散形式保持正确平衡方差 `Var(x)=k_B T/k`，比欧拉法更稳健。

## R05

本 MVP 同时构造两条“独立证据链”：

- 涨落侧：从单条长平衡轨迹计算自相关 `C(τ)`（FFT 无偏估计）。
- 耗散侧：在大量并行样本上施加小阶跃力，测 `δ<x(t)>/f0` 得 `χ_est(t)`。

最后比较 `χ_est(t)` 与 `χ_fdt(t)`：

- `χ_fdt(t) = (C(0)-C(t))/k_B T`
- 误差指标：RMSE、最大绝对误差、相对 `L2` 误差。

## R06

输入/输出规格（`demo.py`）：

- 无交互输入，参数在 `FDTConfig` 中集中定义。
- 终端输出 JSON，包含：
  - 配置与离散系数；
  - 关键统计量（`var_eq_est`, `C0_est`）；
  - FDT 对比误差；
  - 若干采样时刻上的 `χ_est / χ_fdt / χ_theory`。

这样便于后续验证脚本直接解析。

## R07

算法伪代码：

1. 读取配置并计算 `a, σ, drift`。  
2. 生成平衡轨迹 `x_eq`（无外力）。  
3. 计算 `C(τ)`（FFT 自相关，无偏归一化）。  
4. 截取前 `n_steps_resp` 个滞后，构造 `χ_fdt=(C0-C)/k_B T`。  
5. 构造同噪声双系统（有/无阶跃力）并行演化，得到 `χ_est(t)`。  
6. 计算解析 `χ_theory(t)=(1/k)(1-exp(-μk t))`。  
7. 统计 `χ_est` 与 `χ_fdt` 的误差。  
8. 以 JSON 打印结果。

## R08

正确性直觉：

- 若系统满足线性响应与热平衡，涨落信息应完全决定耗散响应；
- 本模型是可解析 OU 过程，故存在“数值结果 vs 解析真值”的双重校验；
- 当 `n_steps_eq` 与 `n_ensembles` 增大时，`χ_est` 与 `χ_fdt` 应更接近。

因此该 demo 不仅“可运行”，还能实证体现 FDT 的核心思想。

## R09

复杂度：

- 平衡轨迹生成：时间 `O(N_eq)`，空间 `O(N_eq)`；
- 自相关（FFT）：时间 `O(N_eq log N_eq)`，空间 `O(N_eq)`；
- 阶跃响应并行仿真：时间 `O(N_resp * M)`，空间 `O(M)`，`M=n_ensembles`。

总耗时通常由 `O(N_resp * M)` 和 FFT 两部分主导。

## R10

误差来源与数值注意点：

- 相关函数长时尾部噪声会污染 `χ_fdt`；
- 阶跃力过大将偏离线性响应区间；
- `n_ensembles` 太小时，`χ_est` 方差较大；
- `dt` 太大时离散时间分辨率不足。

缓解：

- 增大 `n_steps_eq`、`n_ensembles`；
- 保持 `step_force` 小（如 `0.02~0.1`）；
- 选较小 `dt`（如 `1e-3~5e-3`）。

## R11

`demo.py` 主要结构：

- `FDTConfig`：参数容器；
- `compute_discrete_coefficients`：生成 `a, σ, drift`；
- `simulate_equilibrium_trajectory`：平衡轨迹；
- `autocorr_unbiased`：FFT 无偏自相关；
- `simulate_step_response_susceptibility`：并行估计 `χ_est`；
- `run_demo`：汇总误差和采样点结果；
- `main`：打印 JSON（`ensure_ascii=False`）。

## R12

运行：

```bash
uv run python Algorithms/物理-统计力学-0305-涨落-耗散定理_(Fluctuation-Dissipation_Theorem)/demo.py
```

预期：输出 JSON，`fdt_rmse` 和 `fdt_rel_l2` 为小量，说明数值上支持 FDT 关系。

## R13

关键输出字段说明：

- `var_eq_theory` / `var_eq_est`：平衡方差理论/估计；
- `chi_est_final` / `chi_fdt_final` / `chi_theory_final`：末时刻对照；
- `fdt_rmse`：`χ_est` 与 `χ_fdt` 的均方根误差；
- `fdt_rel_l2`：相对二范数误差；
- `samples`：若干时刻的逐点对照，便于人工检查趋势。

## R14

边界与失败场景：

- `k <= 0`、`mu <= 0`、`kbt <= 0`：物理无效；
- `n_steps_eq < 4`：相关函数无意义；
- `n_steps_resp < 2`：无法形成时间响应曲线；
- `n_ensembles < 2`：阶跃响应统计噪声极大。

`demo.py` 对上述参数会抛出明确异常，避免静默错误。

## R15

可扩展方向：

- 从标量 `x` 推广到多维线性系统（矩阵 FDT）；
- 比较不同积分器（欧拉/精确 OU 离散）的偏差；
- 加入 bootstrap 置信区间；
- 推广到频域形式：`Im χ(ω)` 与功率谱之间的 FDT。

## R16

最小验收清单：

1. `uv run python demo.py` 可直接运行；
2. `README.md` 与 `demo.py` 不含待填充占位符；
3. 输出中 `var_eq_est` 接近 `k_B T / k`；
4. `χ_est(t)` 与 `χ_fdt(t)` 曲线同向接近；
5. 解析 `χ_theory(t)` 与两者量级一致。

## R17

知识背景（概念层）：

- Kubo 线性响应理论；
- Langevin / Ornstein-Uhlenbeck 过程；
- 平衡态关联函数与易感率（susceptibility）关系；
- 统计误差与有限样本效应。

本目录优先强调“公式到代码”的可追踪实现，不依赖外部物理黑盒库。

## R18

`demo.py` 源码级流程（8 步，非黑箱）：

1. 在 `FDTConfig` 中设定 `μ, k, k_B T, dt` 与采样规模。  
2. `compute_discrete_coefficients` 计算 `a=exp(-μkdt)`、噪声强度 `σ`、阶跃漂移项。  
3. `simulate_equilibrium_trajectory` 用 for 循环逐步递推，生成平衡序列 `x_eq`。  
4. `autocorr_unbiased` 对 `x_eq` 去均值后做 `rfft -> 功率谱 -> irfft`，再按 `N-τ` 归一化得 `C(τ)`。  
5. 由 `χ_fdt(τ) = (C(0)-C(τ))/k_B T` 构造 FDT 预测响应曲线。  
6. `simulate_step_response_susceptibility` 维护两组并行粒子（同噪声），分别在“无力/阶跃力”下演化，取均值差除以 `f0` 得 `χ_est(t)`。  
7. 计算解析 `χ_theory(t)=(1/k)(1-exp(-μk t))`，并与 `χ_est`、`χ_fdt` 做 RMSE 与相对 `L2` 误差。  
8. 将配置、误差、采样时刻对照点打包成 JSON 输出，形成可复现实验结论。  
