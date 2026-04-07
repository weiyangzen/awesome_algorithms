# 暴胀理论 (Inflation Theory)

- UID: `PHYS-0348`
- 学科: `物理`
- 分类: `宇宙学`
- 源序号: `366`
- 目标目录: `Algorithms/物理-宇宙学-0366-暴胀理论_(Inflation_Theory)`

## R01

暴胀理论是早期宇宙学中的一种动力学机制：在极短时间内出现近指数膨胀，以解释标准大爆炸模型下的视界问题、平坦性问题和遗迹问题。  
在单场慢滚近似中，宇宙由标量场 `phi` 与势能 `V(phi)` 主导，满足：

- `3H^2 M_pl^2 ≈ V(phi)`
- `3H phidot + dV/dphi ≈ 0`

本目录 MVP 采用 `M_pl = 1` 的约化单位，聚焦“如何把慢滚暴胀从公式变成可运行数值流程”。

## R02

这个算法条目的核心不是“复刻完整宇宙学参数拟合器”，而是实现一个最小、可追踪、可检验的暴胀计算管线：

1. 给定一族势能（这里选幂律 `V = lambda * phi^p`）；
2. 由慢滚参数确定“暴胀结束点”；
3. 由目标 e-fold 数反解“枢轴尺度出场点”；
4. 计算谱指数 `n_s`、张量标量比 `r`、标量振幅 `A_s`。

这正是暴胀模型和观测约束之间的最小闭环。

## R03

`demo.py` 的范围定义：

- 模型族：`V(phi)=lambda*phi^p`，扫描 `p in {2/3, 1, 2, 3, 4}`；
- 目标 e-fold：`N_target = 55`；
- 目标振幅：`A_s = 2.1e-9`（用于标定 `lambda`）；
- 数值方法：`scipy.optimize.brentq` + `scipy.integrate.quad` + `solve_ivp`；
- 输出：每个 `p` 的 `phi_end`、`phi_star`、`lambda`、`n_s`、`r` 等指标与可行性标记。

这是教学/验证级 MVP，不包含 reheating 细节、多场耦合、非高斯性等高级主题。

## R04

关键公式（`M_pl = 1`）：

1. 势能与导数：
   - `V(phi)=lambda*phi^p`
   - `V'(phi)=lambda*p*phi^(p-1)`
2. 势慢滚参数：
   - `epsilon_V = 0.5*(V'/V)^2 = 0.5*(p/phi)^2`
   - `eta_V = V''/V = p*(p-1)/phi^2`
3. 暴胀结束条件：
   - `epsilon_V(phi_end)=1`
4. e-fold 计算：
   - `N(phi_star->phi_end)=int_{phi_end}^{phi_star} (V/V') dphi`
5. 一阶慢滚观测量：
   - `A_s ≈ V/(24*pi^2*epsilon_V)`
   - `n_s ≈ 1 - 6*epsilon_V + 2*eta_V`
   - `r ≈ 16*epsilon_V`

`demo.py` 用这套公式进行反解和一致性检验。

## R05

复杂度（`K` 为势指数扫描数量）：

- `phi_end` 求根：每个模型约 `O(I_root)` 次函数调用；
- `phi_star` 求根：每次函数评估会调用一次积分，约 `O(I_root * I_quad)`；
- ODE 交叉验证：`O(I_ode)`；
- 总体：`O(K * (I_root*I_quad + I_ode))`。

在当前参数下 `K=5`，运行时间通常是秒级，适合作为可重复验证的最小样例。

## R06

脚本输出一个 `pandas` 表格，字段包括：

- `p`：势能指数；
- `phi_end`：暴胀结束场值；
- `phi_star`：枢轴尺度出场场值（满足目标 e-fold）；
- `lambda`：由目标 `A_s` 标定得到的势能尺度；
- `n_from_quad` / `n_from_ode`：积分法与 ODE 法得到的 e-fold；
- `epsilon_v`, `eta_v`, `as_pred`, `ns`, `r`：慢滚可观测量；
- `plausible_under_simple_cut`：基于简化阈值 `n_s` 区间与 `r` 上限的布尔标记。

## R07

优点：

- 从理论量到可观测量的链路完整且透明；
- 每一步都对应显式公式，便于审查与教学；
- 使用两种数值路径（积分与 ODE）交叉验证 e-fold。

局限：

- 仅单场、标准慢滚、一阶近似；
- 未做完整观测数据后验拟合；
- 未纳入 reheating 不确定性导致的 `N_target` 漂移。

## R08

前置知识：

- FRW 背景下标量场动力学；
- 慢滚近似与视界出场概念；
- 基础数值方法（积分、求根、常微分方程）。

运行依赖（来自项目 `pyproject.toml`）：

- Python `>=3.10`
- `numpy`
- `scipy`
- `pandas`

## R09

适用场景：

- 教学演示“暴胀模型 -> 观测参数”的最小可执行链路；
- 快速比较不同幂律势 `p` 的预测趋势；
- 在引入更复杂模型前做数值框架预热。

不适用场景：

- 需要高精度 CMB 参数推断（应使用专业 Boltzmann + MCMC 管线）；
- 研究多场、非正则动能、非高斯性等扩展理论；
- 直接作为发表级结果。

## R10

正确性直觉：

1. `epsilon_V=1` 定义暴胀终止，给出 `phi_end`；
2. 给定 `N_target` 反解 `phi_star`，对应“某尺度在终止前多少 e-fold 出场”；
3. 给定 `phi_star` 后，`A_s` 可反解 `lambda`，即振幅定标；
4. `n_s`、`r` 来自同一点的 `epsilon_V` 与 `eta_V`；
5. 用 ODE 再算一次终止 e-fold，若与积分一致，说明实现链路自洽。

`demo.py` 中断言正是围绕这 5 点构建。

## R11

数值稳定与工程细节：

- 对 `phi` 使用正区间根求解，避免奇点；
- `phi_star` 采用自适应上界扩展，保证 `brentq` 成功 bracket；
- ODE 右端加入最小正值保护，避免除零；
- 对关键等式使用严格断言（`N` 一致性、`A_s` 一致性、`r` 单调性）；
- 全流程无随机采样，输出可复现。

## R12

关键参数（`InflationConfig`）：

- `n_target`：目标 e-fold（默认 55）；
- `target_as`：目标标量振幅；
- `p_values`：待扫描的幂律指数集合；
- `ns_band`、`r_upper`：简化可行性阈值（仅教学用途）。

调参影响：

- 增大 `n_target` 通常降低 `r`、调整 `n_s`；
- 增大 `p` 会使 `epsilon_V` 变大，通常导致 `r` 上升；
- `target_as` 主要重新标定 `lambda`，不直接改变 `n_s` 与 `r`。

## R13

保证类型说明：

- 近似比保证：N/A（非组合优化问题）。
- 随机成功率保证：N/A（本实现是确定性数值流程）。

MVP 的可执行保证：

- `uv run python demo.py` 可直接运行，无交互输入；
- 每个模型都输出完整中间量与观测量；
- 内置断言验证了积分法/ODE 法一致性与若干物理趋势。

## R14

常见失效模式：

1. 混淆 `phi_end` 与 `phi_star` 的物理定义；
2. 漏掉 `A_s` 定标步骤，导致势能尺度不可比；
3. 把 `N` 积分上下限写反，得到负 e-fold；
4. 在 `phi -> 0` 附近数值发散；
5. 直接把本简化阈值当成严肃观测约束。

本实现通过根求解、方向明确的积分定义和断言检查降低这些风险。

## R15

可扩展方向：

1. 加入更多势模型（Starobinsky、hilltop、natural inflation）；
2. 引入 reheating 参数，把 `N_target` 从常数改为区间推断；
3. 对接 Boltzmann 求解器并与 CMB 数据做联合拟合；
4. 增加不确定性传播（参数先验 -> 后验分布）；
5. 把慢滚一阶公式升级到二阶修正或直接数值扰动演化。

## R16

相关主题：

- 慢滚近似与 Hubble flow parameters；
- 宇宙初始条件问题（视界/平坦性）；
- CMB 标量谱、张量谱与 B 模观测；
- reheating 与 e-fold 映射不确定性；
- 早期宇宙模型选择与贝叶斯比较。

## R17

运行方式（无交互）：

```bash
cd Algorithms/物理-宇宙学-0366-暴胀理论_(Inflation_Theory)
uv run python demo.py
```

交付核对：

- `README.md` 的 `R01-R18` 全部完成；
- `demo.py` 可直接执行并包含内置自检；
- `meta.json` 与任务元数据一致；
- 目录可独立用于最小验证。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 创建 `InflationConfig`，固定 `N_target`、`A_s` 目标与待扫描的 `p_values`。  
2. `run_grid` 逐个 `p` 调用 `find_phi_end`，用 `brentq` 解 `epsilon_V(phi)=1` 得到 `phi_end`。  
3. 同一 `p` 下，`find_phi_star` 再用 `brentq` 反解 `efolds_between(phi_star, phi_end)=N_target`。  
4. `efolds_between` 用 `scipy.integrate.quad` 计算 `int(V/V')dphi`，把理论 e-fold 关系落实为数值积分。  
5. 有了 `phi_star` 后，`calibrate_lambda` 按 `A_s = V/(24*pi^2*epsilon_V)` 反推出 `lambda`。  
6. `predict_observables` 计算 `epsilon_V`、`eta_V`、`n_s`、`r`、`A_s` 预测值，并按简化阈值打可行性标签。  
7. `efolds_by_ode` 用 `solve_ivp` 解 `dphi/dN = -V'/V` 到 `epsilon_V=1` 事件，得到 `n_from_ode` 作为交叉验证。  
8. `run_self_checks` 对 `N` 一致性、`A_s` 一致性、`r` 随 `p` 单调性做断言，最终输出可复验表格。  

第三方库拆解说明：`scipy` 仅提供通用积分/求根/ODE 数值内核；势能、慢滚参数、可观测量公式、模型扫描与物理判据都在源码中显式实现，不是黑盒调用。
