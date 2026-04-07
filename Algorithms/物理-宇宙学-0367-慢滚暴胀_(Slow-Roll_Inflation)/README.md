# 慢滚暴胀 (Slow-Roll Inflation)

- UID: `PHYS-0349`
- 学科: `物理`
- 分类: `宇宙学`
- 源序号: `367`
- 目标目录: `Algorithms/物理-宇宙学-0367-慢滚暴胀_(Slow-Roll_Inflation)`

## R01

慢滚暴胀是单场暴胀模型中的近似机制：标量场 `phi` 在平缓势能 `V(phi)` 上缓慢滚动，使宇宙经历足够长的准指数膨胀。核心判据是慢滚参数很小：

- `epsilon_V = (M_pl^2/2) * (V'/V)^2 << 1`
- `|eta_V| = |M_pl^2 * V''/V| << 1`

在本 MVP 中采用约化普朗克单位 `M_pl = 1`，并用幂律势 `V(phi)=lambda*phi^p` 做最小可执行演示。

## R02

本条目关注的不是完整 CMB 参数拟合，而是“慢滚近似如何落地为数值算法”这一核心链路：

1. 由 `epsilon_V=1` 求暴胀结束点 `phi_end`；
2. 给定目标 e-fold `N_target` 反解枢轴尺度对应 `phi_star`；
3. 用标量振幅 `A_s` 反标定势能尺度 `lambda`；
4. 输出 `n_s`、`r` 等可观测量并做一致性交叉检验。

## R03

`demo.py` 的 MVP 范围：

- 模型族：`V(phi)=lambda*phi^p`，扫描 `p in {2/3, 1, 2, 3, 4}`；
- 目标参数：`N_target=55`，`A_s=2.1e-9`；
- 主要工具：`numpy`、`scipy`、`pandas`、`scikit-learn`、`PyTorch`；
- 数值任务：求根、积分、ODE 终止事件、线性回归估计谱倾角、autograd 导数校验；
- 输出结果：每个 `p` 的场值、慢滚参数、谱指数、张量比和可行性布尔标记。

## R04

关键公式（`M_pl=1`）：

1. 背景慢滚方程：
   - `3H^2 ≈ V(phi)`
   - `3H phidot + V'(phi) ≈ 0`
2. 势慢滚参数：
   - `epsilon_V = 0.5 * (V'/V)^2`
   - `eta_V = V''/V`
3. 结束条件：`epsilon_V(phi_end)=1`
4. e-fold：`N(phi_star->phi_end)=∫[phi_end, phi_star] (V/V') dphi`
5. 一阶可观测量：
   - `A_s ≈ V/(24*pi^2*epsilon_V)`
   - `n_s ≈ 1 - 6*epsilon_V + 2*eta_V`
   - `r ≈ 16*epsilon_V`

## R05

复杂度分析（`K` 为扫描的 `p` 个数）：

- `phi_end`：每个模型一次 1D 求根，约 `O(I_root)`；
- `phi_star`：外层求根 + 内层积分，约 `O(I_root * I_quad)`；
- ODE 交叉检验：约 `O(I_ode)`；
- 总体：`O(K * (I_root*I_quad + I_ode))`。

当前 `K=5`，在普通 CPU 上通常秒级完成。

## R06

输出表字段说明：

- `p`：幂律势指数；
- `phi_end`：满足 `epsilon_V=1` 的结束场值；
- `phi_star`：满足目标 e-fold 的枢轴场值；
- `lambda`：由 `A_s` 反标定的势能尺度；
- `N_quad` / `N_ode`：积分法与 ODE 法得到的 e-fold；
- `epsilon_v` / `epsilon_torch`：解析导数与 autograd 的 `epsilon_V`；
- `eta_torch`：autograd 得到的 `eta_V`；
- `n_s_analytic` / `n_s_regression`：解析公式与回归估计的谱指数；
- `r`：张量标量比；
- `plausible_under_simple_cut`：教学用简化筛选标签。

## R07

优势：

- 公式链路完整，理论量到观测量一一可追踪；
- 同时使用积分法与 ODE 法计算 e-fold，便于互验；
- 引入 PyTorch autograd 校验导数实现，降低手写导数错误。

局限：

- 仅单场、标准慢滚、一阶近似；
- 未纳入 reheating 不确定性与完整后验推断；
- 观测筛选阈值是教学型，不是正式参数拟合结论。

## R08

前置知识与依赖：

- 早期宇宙学中标量场动力学与 FRW 背景；
- 慢滚近似和 e-fold 概念；
- 数值分析中的积分、求根、常微分方程；
- Python 科学计算栈：`numpy/scipy/pandas/sklearn/torch`。

## R09

适用场景：

- 教学演示慢滚暴胀的可执行计算流程；
- 快速比较不同幂律势的 `n_s` 与 `r` 预测趋势；
- 为更复杂暴胀模型搭建可测试的最小框架。

不适用场景：

- 发表级精度 CMB 拟合；
- 多场耦合、非高斯性、非正则动能等扩展模型；
- 需要完整宇宙学参数后验的场景。

## R10

正确性直觉：

1. `epsilon_V=1` 给出“何时结束暴胀”的动力学边界；
2. `N_target` 固定后，`phi_star` 唯一对应“某尺度出场时刻”；
3. 固定 `phi_star` 后用 `A_s` 标定 `lambda`，将模型幅度锚定到观测量级；
4. `n_s` 与 `r` 直接由同一点慢滚参数给出；
5. 若积分法和 ODE 法对 `N` 一致，说明实现链路数值自洽。

## R11

数值稳定策略：

- 求根区间使用严格正值，避免 `phi=0` 奇点；
- 对 `phi_star` 的上界采用自适应扩展直到成功 bracket；
- ODE 中对极小 `phi` 做下界保护防止除零；
- 使用断言约束关键一致性：`N`、`A_s`、导数校验与单调趋势。

## R12

主要参数（`SlowRollConfig`）：

- `n_target`：目标 e-fold，默认 `55`；
- `target_as`：目标标量振幅，默认 `2.1e-9`；
- `p_values`：扫描的势指数集合；
- `ns_band` 与 `r_upper`：教学型可行性筛选阈值。

调参影响：

- 增大 `n_target` 往往会减小 `r`；
- 增大 `p` 通常提高 `epsilon_V`，从而推高 `r`；
- `target_as` 主要重标定 `lambda`，不直接改变 `n_s` 和 `r`。

## R13

保证性质声明：

- 近似比保证：N/A（并非组合优化算法）；
- 概率成功保证：N/A（当前流程为确定性数值计算）；
- 可重复性：同样参数与环境下输出应一致；
- 可运行性：`uv run python demo.py` 无需交互输入。

## R14

常见错误与规避：

1. `N` 积分上下限写反导致 e-fold 为负；
2. 漏做 `A_s` 标定使不同模型幅度不可比；
3. 误把 `phi_end` 当作 `phi_star`；
4. 忽略 `phi->0` 的数值不稳定问题；
5. 将教学阈值误读为严肃观测约束。

脚本通过事件终止、区间求根和断言机制降低上述风险。

## R15

可扩展方向：

1. 加入 Starobinsky、Natural inflation、Hilltop 等势；
2. 将 `N_target` 与 reheating 参数联动而非固定常数；
3. 引入慢滚二阶修正或直接求解扰动方程；
4. 对接更完整的宇宙学推断管线（如 MCMC+Boltzmann）；
5. 加入参数不确定性传播和敏感性分析。

## R16

相关主题：

- 慢滚参数层级与 Hubble-flow 参数；
- 标量/张量初始谱与 CMB 观测约束；
- reheating 对 `N_*` 映射的影响；
- 早期宇宙模型选择与可证伪性。

## R17

运行步骤：

```bash
cd Algorithms/物理-宇宙学-0367-慢滚暴胀_(Slow-Roll_Inflation)
uv run python demo.py
```

期望行为：

- 输出一个按 `p` 排序的数据表；
- 打印目标参数与简化筛选阈值；
- 输出启发式最优 `p`；
- 最后打印 checks passed 信息。

## R18

`demo.py` 的源码级算法流程可拆成 8 步（避免把第三方库当黑箱）：

1. `find_phi_end` 用 `scipy.optimize.brentq` 对函数 `epsilon_V(phi)-1` 求根，得到 `phi_end`。
2. `find_phi_star` 先构造 `objective(phi)=N(phi)-N_target`，其中 `N(phi)` 由 `scipy.integrate.quad` 计算 `∫(V/V')dphi`。
3. 若初始上界不满足异号条件，循环放大上界直到可 bracket，再用 `brentq` 求出 `phi_star`。
4. `calibrate_lambda` 代入 `A_s = V/(24*pi^2*epsilon_V)`，直接反解势能尺度 `lambda`。
5. `ns_r` 在 `phi_star` 处计算 `epsilon_V`、`eta_V`，再得到 `n_s` 和 `r`。
6. `efolds_by_ode` 用 `solve_ivp` 积分 `dphi/dN=-p/phi`，并用事件 `epsilon_V-1=0` 自动停止，得到 `N_ode`。
7. `torch_slowroll_at_star` 用 PyTorch autograd 自动求 `V'` 与 `V''`，独立复算 `epsilon_V`、`eta_V` 与解析结果对比。
8. `estimate_ns_with_regression` 在枢轴附近采样 `ΔN`，计算 `P_R`，再用 `sklearn.linear_model.LinearRegression` 拟合 `ln P_R` 对 `ln(k/k*)` 的斜率，得到回归版 `n_s` 并与解析值交叉验证。
