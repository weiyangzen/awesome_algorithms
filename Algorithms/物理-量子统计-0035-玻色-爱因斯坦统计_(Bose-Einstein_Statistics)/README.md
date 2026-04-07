# 玻色-爱因斯坦统计 (Bose-Einstein Statistics)

- UID: `PHYS-0035`
- 学科: `物理`
- 分类: `量子统计`
- 源序号: `35`
- 目标目录: `Algorithms/物理-量子统计-0035-玻色-爱因斯坦统计_(Bose-Einstein_Statistics)`

## R01

玻色-爱因斯坦统计描述“不可分辨玻色子”在能级上的平均占据数：

`n(ε) = 1 / (exp((ε - μ)/k_B T) - 1)`。

当温度降低到临界温度 `T_c` 以下时，化学势逼近 `μ -> 0`，低能态占据迅速增长并形成玻色-爱因斯坦凝聚（BEC）。

本条目给出一个可运行的最小数值实验：
- 采用三维理想玻色气体的态密度 `g(ε) = A*sqrt(ε)`；
- 对给定温度求解满足粒子数守恒的 `μ(T)`；
- 输出激发态粒子数、凝聚数与占据数样本。

## R02

理想玻色气体在连续能谱近似下：

`N_ex(T, μ) = ∫ g(ε) / (exp((ε-μ)/T)-1) dε`，其中 `μ <= 0`。

在三维自由粒子模型里，`g(ε) = A*sqrt(ε)`，可得：
- `T > T_c`：无宏观基态凝聚，`N_ex = N_total`，需解 `μ<0`；
- `T <= T_c`：`μ = 0`，且 `N0 = N_total - N_ex(T, 0)`；
- 理论凝聚分数：`N0/N = 1 - (T/T_c)^(3/2)`（`T<T_c`）。

## R03

MVP 建模选择：
- 使用无量纲单位：`k_B=1, T_c=1, N_total=1`；
- 通过 `A = 1/(Gamma(3/2)*zeta(3/2)*T_c^(3/2))` 归一化态密度；
- 温度采样：`T = [0.4, 0.7, 1.0, 1.3, 1.8]`；
- 输出两张表：
  - 热力学汇总（`mu`, `N_ex`, `N0`, `cond_frac`）；
  - 代表性能级的平均占据数 `n(ε)`。

## R04

数值策略：
1. 先做变量替换 `ε=t^2`，把积分改写为
   `I(alpha)=∫ 2 t^2/(exp(t^2+alpha)-1) dt`，其中 `alpha=-μ/T`；
2. 用 `scipy.integrate.quad` 计算 `I(alpha)`；
3. 在 `T>T_c` 分支，用 `scipy.optimize.brentq` 解粒子数方程；
4. 在 `T<=T_c` 分支直接令 `μ=0`；
5. 汇总得到 `N_ex`, `N0`, `N0/N` 并做物理一致性检查。

## R05

`demo.py` 默认参数（`BEConfig`）：
- `tc = 1.0`
- `total_density = 1.0`
- `temperatures = (0.40, 0.70, 1.00, 1.30, 1.80)`
- `energy_levels = (0.02, 0.10, 0.30, 0.80, 1.50, 3.00)`
- 积分截断上限：`t_upper = 14.0`
- 积分精度：`quad_epsabs = 1e-9`, `quad_epsrel = 1e-8`
- 根搜索初始右端点：`alpha_upper_init = 60.0`

## R06

代码结构（`demo.py`）：
- `BEConfig`：集中管理温度、积分、输出参数；
- `density_prefactor_for_tc`：按 `T_c` 归一化态密度常数 `A`；
- `_bose_integrand_t` / `bose_integral`：计算玻色积分；
- `excited_density`：计算 `N_ex(T, μ)`；
- `solve_mu_above_tc`：在高温相解 `μ(T)`；
- `analyze_temperatures`：遍历温度并生成两张 `DataFrame`；
- `run_consistency_checks`：物理趋势断言；
- `main`：执行并打印结果。

## R07

伪代码：

```text
cfg <- default config
A <- normalize DOS prefactor from Tc

for T in temperatures:
  if T <= Tc:
    mu <- 0
  else:
    solve alpha from A*T^(3/2)*I(alpha) = N_total
    mu <- -alpha*T

  N_ex <- A*T^(3/2)*I(-mu/T)
  N0 <- max(N_total - N_ex, 0)
  cond_frac <- N0 / N_total

  for epsilon in energy_levels:
    n(epsilon) <- 1 / (exp((epsilon-mu)/T)-1)

run sanity checks and print tables
```

## R08

复杂度分析（`M` 为温度点数，`Q` 为根搜索中的积分调用次数）：
- 单次积分复杂度可视作 `O(K_quad)`（自适应求积采样点数）；
- 高温分支求 `μ` 需多次积分，约 `O(Q * K_quad)`；
- 总体时间复杂度约 `O(M * Q * K_quad)`；
- 空间复杂度 `O(M)`（主要保存输出表）。

默认参数规模很小，CPU 下可快速完成。

## R09

稳定性与可复现性处理：
- 用替换 `ε=t^2` 消除 `μ=0` 时积分核在 `ε=0` 的可积奇性；
- 使用 `np.expm1` 避免小指数差分的数值损失；
- 高温根搜索采用“动态扩展上界”确保 `brentq` 成功夹根；
- 不使用随机数，结果可重复。

## R10

与其他实现路径比较：
- 解析路径：可直接用多重对数 `Li_{3/2}` 写闭式，但对教学上“守恒方程如何数值求解”展示较少；
- 全量微观模拟：可做蒙特卡洛或格点玻色-哈伯德，更真实但超出 MVP 范围；
- 本实现：保留关键物理量（`μ(T)`, `N_ex`, `N0`）并展示最小可运行求解流程。

## R11

调参建议：
- 想看更细临界行为：在 `temperatures` 增加 `0.95, 1.00, 1.05`；
- 想更高积分精度：减小 `quad_epsabs/quad_epsrel`；
- 想更快运行：减小温度点数量或放宽积分精度；
- 若根求解失败：提高 `alpha_upper_init` 或 `t_upper`。

## R12

实现细节：
- `density_prefactor_for_tc` 通过 `Gamma(3/2)` 和 `zeta(3/2)` 直接归一化，避免手工常数误差；
- `_bose_integrand_t` 在 `u=t^2+alpha -> 0` 时使用极限值 `2`，避免 `0/0`；
- `run_consistency_checks` 同时检查：
  - `T<T_c` 时凝聚分数随 `T` 上升而下降；
  - 低温数值凝聚分数与理论 `1-(T/Tc)^(3/2)` 接近；
  - `T>T_c` 时 `μ<0` 且凝聚分数趋近 0。

## R13

运行方式：

```bash
cd "Algorithms/物理-量子统计-0035-玻色-爱因斯坦统计_(Bose-Einstein_Statistics)"
uv run python demo.py
```

或在项目根目录执行：

```bash
uv run python Algorithms/物理-量子统计-0035-玻色-爱因斯坦统计_(Bose-Einstein_Statistics)/demo.py
```

脚本无需交互输入。

## R14

输出解读：
- `mu`：化学势；高于临界温度时应为负值；
- `N_ex`：激发态粒子数（或密度）；
- `N0`：凝聚到基态的粒子数；
- `cond_frac`：凝聚分数 `N0/N_total`；
- `theory_cond_frac`：理想气体解析公式值；
- `n(e=...)`：指定能级的 BE 平均占据数。

典型趋势：`T` 越低，低能级占据越高；`T<T_c` 时出现非零 `N0`。

## R15

常见问题排查：
- 报根搜索失败：
  - 增大 `alpha_upper_init`；
  - 适当增大 `t_upper`，确保积分尾部收敛充分。
- 断言失败（通常在临界点附近）：
  - 增加温度点分辨率并检查排序；
  - 适当放宽误差阈值（`run_consistency_checks` 中容差）。
- 输出占据数异常大：
  - 这是低温、低能级下 BE 分布的正常行为，不是数值错误。

## R16

可扩展方向：
- 换用解析多重对数 `Li_s(z)` 与数值积分交叉验证；
- 加入有限体积离散能级，研究临界附近的有限尺寸效应；
- 计算热容、压强等更多热力学量并绘制温度曲线；
- 扩展到弱相互作用模型（如 Hartree-Fock 近似）做对比。

## R17

边界与限制：
- 当前模型是理想玻色气体，不含粒子相互作用；
- 使用连续态密度近似，不包含真实势阱细节；
- 临界点附近数值积分与求根精度会影响小量差值；
- 该 MVP 用于算法演示与教学，不替代高精度实验拟合。

## R18

`demo.py` 源码级流程（8 步）：
1. `main()` 创建 `BEConfig`，固定 `T_c`、温度点、积分参数和输出能级。  
2. `analyze_temperatures()` 先调用 `density_prefactor_for_tc()`，用 `Gamma` 和 `zeta` 算出态密度系数 `A`。  
3. 对每个温度 `T` 进入分支：`T<=Tc` 直接设 `μ=0`；`T>Tc` 调 `solve_mu_above_tc()`。  
4. `solve_mu_above_tc()` 定义残差 `A*T^(3/2)*I(alpha)-N_total`，先扩展右边界再用 `scipy.optimize.brentq` 求根 `alpha*`。  
5. `bose_integral(alpha)` 在每次残差评估中调用 `scipy.integrate.quad` 计算 `I(alpha)`，积分核来自 `_bose_integrand_t()` 的 `ε=t^2` 变换。  
6. 得到 `μ=-alpha*T` 后，`excited_density()` 计算 `N_ex`，再得到 `N0=max(N_total-N_ex,0)` 与 `cond_frac`。  
7. 同一循环中按 `energy_levels` 计算 `n(ε)=1/expm1((ε-μ)/T)`，构造占据数样本表。  
8. `run_consistency_checks()` 执行物理断言（低温凝聚分数单调、高温 `μ<0`、高温凝聚近零），最后 `main()` 打印两张结果表。
