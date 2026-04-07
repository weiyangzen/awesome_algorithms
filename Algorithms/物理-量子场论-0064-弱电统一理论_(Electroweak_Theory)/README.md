# 弱电统一理论 (Electroweak Theory)

- UID: `PHYS-0064`
- 学科: `物理`
- 分类: `量子场论`
- 源序号: `64`
- 目标目录: `Algorithms/物理-量子场论-0064-弱电统一理论_(Electroweak_Theory)`

## R01

弱电统一理论把电磁相互作用与弱相互作用统一到规范群 `SU(2)_L x U(1)_Y` 下。  
在自发对称性破缺后，规范场重新组合为光子 `A`、弱中性玻色子 `Z`、带电玻色子 `W^±`，并给出 Weinberg 角 `theta_W`、电荷 `e`、`W/Z` 质量关系等可检验结构。

## R02

本条目目标：实现一个可运行、可审计的最小数值 MVP，展示弱电统一里最核心的“可计算链条”：

- `g1 (U(1)_Y)` 与 `g2 (SU(2)_L)` 的一圈 RG 跑动；
- 数值积分与解析解的一致性验证；
- 由 `(g1, g2)` 反推出 `sin^2(theta_W)`, `alpha_em`, `mW`, `mZ`, `rho`；
- 给出中性流 `Z-f-\bar f` 的向量/轴矢耦合系数表。

## R03

`demo.py` 输入输出约定（无交互）：

- 输入（脚本内固定）：
1. 参考能标 `mu_ref = MZ = 91.1876 GeV`；
2. `alpha_em(mu_ref) = 1/127.95`；
3. `sin^2(theta_W)(mu_ref) = 0.23122`；
4. Higgs 真空期望值 `v = 246.22 GeV`；
5. 一圈系数 `b1 = 41/6`, `b2 = -19/6`；
6. 能标网格 `mu_ref * geomspace(1, 100, 16)`。
- 输出：
1. 输入参数与参考点耦合汇总；
2. 跑动耦合与派生可观测量表；
3. `mu=MZ` 与 `mu=100*MZ` 的中性流耦合表；
4. 断言通过后输出 `All checks passed.`。

## R04

MVP 使用的数学模型：

1. 一圈规范耦合 RG 方程：
`d g_i / d ln(mu) = (b_i / 16pi^2) * g_i^3`。

2. 本条目采用（非 GUT 归一化）系数：
`b1 = 41/6`（`U(1)_Y`），`b2 = -19/6`（`SU(2)_L`）。

3. 一圈闭式解：
`1/g_i(mu)^2 = 1/g_i(mu0)^2 - (b_i/8pi^2) ln(mu/mu0)`。

4. 弱电混合与树级关系：
`sin^2(theta_W) = g1^2/(g1^2+g2^2)`，
`e = g1 g2 / sqrt(g1^2 + g2^2)`，
`mW = v g2/2`，
`mZ = v sqrt(g1^2+g2^2)/2`，
`rho = mW^2/(mZ^2 cos^2(theta_W)) = 1`（树级）。

5. 中性流耦合（常见归一化）：
`gV = T3 - 2Q sin^2(theta_W)`，`gA = T3`。

## R05

复杂度分析（能标点数 `M`）：

- 双耦合一阶 ODE 积分：`O(M)` 到 `O(M * step_factor)`；
- 解析解计算、误差统计、派生量构造：`O(M)`；
- 中性流耦合表（固定 4 种费米子）：`O(1)`；
- 空间复杂度：`O(M)`。

主开销来自 `solve_ivp` 对 `(g1, g2)` 的积分。

## R06

算法流程：

1. 从 `alpha_em` 与 `sin^2(theta_W)` 反推参考点 `g1_ref, g2_ref`；
2. 在 `t=ln(mu)` 空间建立 `g1,g2` 的一圈 beta 方程；
3. 使用 `solve_ivp` 积分得到数值跑动轨道；
4. 用闭式解独立计算 `g1,g2` 的解析跑动；
5. 计算数值/解析相对误差并验证；
6. 由跑动耦合构造 `sin^2(theta_W), alpha_em, mW, mZ, rho`；
7. 计算 `nu_e,e,u,d` 的 `gV,gA,gL,gR` 中性流耦合；
8. 打印表格并执行断言检查。

## R07

优点：

- 统一展示了“RG 跑动 -> 混合角 -> 质量关系 -> 中性流耦合”的完整链条；
- 数值解与解析解互校，结果可审计；
- 模型小、依赖轻、运行快。

局限：

- 仅含一圈与树级关系，不含两圈/高阶电弱修正；
- 未处理阈值匹配、Yukawa/Higgs 自耦合联立 RG；
- 中性流表是机制演示，不是精密电弱拟合。

## R08

前置知识与环境：

- 量子场论中的规范对称性、RG、树级质量关系；
- 常微分方程数值积分；
- Python `>=3.10`；
- 依赖：`numpy`, `pandas`, `scipy`。

## R09

适用场景：

- 课程中演示弱电统一的最小计算闭环；
- 快速检查 `g1/g2` 跑动方向与 `sin^2(theta_W)` 能标趋势；
- 作为更高阶电弱代码的 baseline。

不适用场景：

- 需要实验级精度（LEP/SLC 全局拟合）；
- 需要两圈以上 RG、阈值与方案依赖细节；
- 需要完整 SMEFT/BSM 参数扫描。

## R10

正确性直觉：

1. `b1>0` 导致 `g1` 随 `mu` 增大而增长；
2. `b2<0` 导致 `g2` 随 `mu` 增大而减小；
3. 因 `sin^2(theta_W)=g1^2/(g1^2+g2^2)`，该比值在本区间应上升；
4. 树级关系要求 `mW/mZ = cos(theta_W)` 且 `rho=1`；
5. 一圈方程有闭式解，可直接检验数值积分正确性。

## R11

数值稳定策略：

- 在 `t=ln(mu)` 空间积分，避免跨数量级直接积分引起的误差放大；
- 强制 `mu_grid` 正值、严格递增、起点等于 `mu_ref`；
- ODE RHS 对 `g1,g2` 施加微小正下界防止非物理负值；
- 显式检查 `solve_ivp.success` 与有限性；
- 用解析解相对误差阈值做回归断言。

## R12

关键参数与影响：

- `alpha_em_ref`、`sin2_theta_w_ref`：决定参考点 `g1_ref,g2_ref`；
- `b1,b2`：直接决定耦合跑动方向与速度；
- `higgs_vev_gev`：线性缩放 `mW,mZ`；
- `mu_grid` 范围：决定可见的跑动幅度；
- ODE 容差 `rtol/atol`：影响数值-解析误差。

调参建议：

- 先扩大 `mu_grid` 区间看趋势，再收紧误差阈值；
- 做高精度研究时应切换到两圈及阈值匹配版本。

## R13

- 近似比保证：N/A（非优化近似问题）。
- 随机成功率：N/A（全流程确定性）。

本 MVP 的可验证保证（由断言给出）：

- `g1/g2` 数值解与一圈解析解相对误差低于阈值；
- `g1` 上升、`g2` 下降、`sin^2(theta_W)` 上升；
- `rho` 与 `mW/mZ=cos(theta_W)` 在浮点精度内成立。

## R14

常见失效模式：

1. `mu_grid` 非递增或含非正值，导致 `ln(mu)` 非法；
2. beta 系数符号写反，造成跑动方向错误；
3. 将 `g1` 的归一化约定与 `b1` 系数混用；
4. 忽略 ODE 求解失败标志导致静默错误；
5. 将树级 `rho=1` 误读为包含辐射修正后的精密结论。

## R15

可扩展方向：

- 加入 `g3`, Yukawa, Higgs 自耦合，做多耦合联立 RG；
- 加入两圈 beta 与阈值匹配（top/Higgs/weak boson）；
- 输出 CSV 并绘图展示 `sin^2(theta_W)(mu)`；
- 扩展到 SMEFT 参数跑动。

## R16

相关条目：

- 自发对称性破缺（Higgs 机制）；
- Weinberg 角与中性流；
- 重整化群方程（RGE）；
- 标准模型规范耦合跑动。

## R17

`demo.py` 交付能力：

- 显式实现 `SU(2)_L x U(1)_Y` 一圈 RG 方程；
- 在同一脚本内完成“数值积分 vs 解析解”校验；
- 给出 `sin^2(theta_W)`, `alpha_em`, `mW`, `mZ`, `rho` 表；
- 输出代表性费米子的中性流耦合；
- 无交互，单命令运行。

运行方式：

```bash
cd Algorithms/物理-量子场论-0064-弱电统一理论_(Electroweak_Theory)
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程（8 步）：

1. `initial_couplings` 用输入的 `alpha_em_ref` 与 `sin2_theta_w_ref` 显式反推 `g1_ref,g2_ref,e_ref`。  
2. `one_loop_beta` 实现一圈 beta 基元 `dg/dln(mu)=(b/16pi^2)g^3`，并对非正耦合做输入检查。  
3. `integrate_gauge_couplings` 在 `t=ln(mu)` 变量下组装二维 RHS `(g1,g2)`，调用 `solve_ivp` 计算数值轨道，同时检查成功标志与物理性。  
4. `analytic_running` 用闭式公式 `1/g^2` 的线性关系独立算出解析轨道，避免把第三方积分器当黑盒结论。  
5. `electroweak_observables` 将 `(g1,g2)` 显式映射到 `sin^2(theta_W)`, `alpha_em`, `mW`, `mZ`, `rho` 与 `mW/mZ`。  
6. `neutral_current_couplings` 对 `nu_e,e,u,d` 用 `T3,Q` 逐项计算 `gV,gA,gL,gR`，形成中性流耦合表。  
7. `main` 汇总数值/解析误差与物理量表格，分别输出参考尺度和高尺度下的耦合信息。  
8. `main` 运行确定性断言（误差阈值、单调性、`rho` 与质量关系）并输出 `All checks passed.`。  

说明：`scipy.solve_ivp` 只负责通用 ODE 步进；方程形式、系数、可观测量构造和物理校验都在源码中显式实现。
