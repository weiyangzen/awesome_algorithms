# 量子色动力学 (Quantum Chromodynamics, QCD)

- UID: `PHYS-0063`
- 学科: `物理`
- 分类: `量子场论`
- 源序号: `63`
- 目标目录: `Algorithms/物理-量子场论-0063-量子色动力学_(Quantum_Chromodynamics,_QCD)`

## R01

量子色动力学（QCD）是描述夸克与胶子强相互作用的非阿贝尔规范理论，其规范群为 `SU(3)_c`。  
本条目聚焦一个可运行、可审计的核心数值闭环：

- 用 QCD beta 函数计算跑动耦合 `alpha_s(mu)`；
- 用微扰展开预测 `e+e- -> hadrons` 的 `R` 比；
- 从观测反推参数，验证“高能渐近自由、低能耦合增强”的机制。

## R02

本 MVP 要解决的问题：

1. 给定参考点 `(mu_ref, alpha_ref)` 与 `N_f`，数值积分两圈重整化群方程，得到 `alpha_s(mu)`；
2. 基于 `R = 3 * sum(q_f^2) * (1 + a + c2 a^2)`（`a = alpha_s/pi`）构造可观测量；
3. 用带噪声合成数据执行“正向预测 + 反演拟合”；
4. 用断言验证 QCD 关键性质（尤其是渐近自由）。

## R03

`demo.py` 的输入输出约定（无交互）：

- 输入（脚本内固定配置）：
1. `N_f=5`，`mu_ref=10 GeV`，`alpha_ref_true=0.18`；
2. 能标网格 `mu in [10, 400] GeV`（几何等距）；
3. 合成观测噪声 `sigma=0.004`；
4. 两圈 beta 方程积分。
- 输出：
1. `N_f` 对应的 `beta0,beta1` 与渐近自由判据；
2. 逐能标表格：`alpha_true`、`R_true`、`R_observed`、反演 `alpha_from_R`、拟合曲线；
3. 反推参数摘要（非线性拟合 `alpha_ref`、线性回归 `Lambda_QCD`）；
4. Torch 自动微分对 beta 导数的校验；
5. 全部断言通过后打印 `All checks passed.`。

## R04

采用的数学模型：

1. 两圈 QCD beta（`SU(3)`）：
`beta0 = 11 - 2N_f/3`，`beta1 = 102 - 38N_f/3`。

2. 在 `t = ln(mu)` 变量下：
`d alpha / dt = -b0 alpha^2 - b1 alpha^3`，
其中 `b0 = beta0/(2pi)`，`b1 = beta1/(4pi^2)`。

3. `e+e-` 强子截面比（简化到 `O(alpha_s^2)`）：
`R = 3 * sum_f(q_f^2) * [1 + a + c2 a^2]`，`a = alpha_s/pi`，
`c2 = 1.9857 - 0.1153 N_f`。

4. 反演 `alpha_s`：
由 `c2 x^2 + x - q = 0`（`x = alpha_s/pi`，`q = R/base - 1`）取正根。

## R05

复杂度分析（能标点数 `M`）：

- beta 系数与常数计算：`O(1)`；
- 一次 ODE 积分（固定维度一阶方程）：约 `O(M)` 到 `O(M * step_factor)`；
- `R` 计算与反演：`O(M)`；
- 线性回归（单特征）：`O(M)`；
- 非线性最小二乘拟合：`O(K * M)`，`K` 为迭代次数；
- 空间复杂度：`O(M)`。

## R06

MVP 算法闭环：

1. 先用给定 `alpha_ref_true` 生成“真值”跑动曲线 `alpha_true(mu)`；
2. 由 `alpha_true` 计算 `R_true` 并加高斯噪声得到 `R_observed`；
3. 对 `R_observed` 执行解析反演，得到 `alpha_from_R`；
4. 用 `scipy.optimize.least_squares` 拟合 `alpha_ref`，使模型 `R_fit` 逼近观测；
5. 用 `sklearn` 线性回归在一圈线性化关系中估计 `Lambda_QCD`；
6. 用 `torch.autograd` 校验 `d beta / d alpha` 与解析导数一致；
7. 用断言验证物理趋势与数值一致性。

## R07

优点：

- 贯通“场论方程 -> 可观测量 -> 参数反演”的完整链路；
- 所有关键物理方程在源码中显式实现，不依赖黑盒专用库；
- 同时使用确定性验证（断言）与多方法交叉检查（解析反演、非线性拟合、自动微分）。

局限：

- 仅示范固定 `N_f`、固定方案下的简化两圈运行；
- 未处理重夸克阈值匹配、非微扰区域与高阶修正；
- `R` 比采用近似系数，目标是教学演示而非精密现象学。

## R08

前置知识与环境：

- 量子场论中的重整化群（RG）与 beta 函数；
- 基础常微分方程数值积分；
- Python `>=3.10`；
- 依赖：`numpy`, `scipy`, `pandas`, `scikit-learn`, `torch`。

## R09

适用场景：

- 量子场论/粒子物理课程中的 QCD 跑动耦合演示；
- 教学或原型中验证“由观测反推耦合参数”的最小工作流；
- 后续扩展到更高圈数或阈值匹配前的基线实现。

不适用场景：

- 需要实验级精度的全局拟合；
- 低能强耦合区的非微扰计算（如格点 QCD 主问题）；
- 涉及复杂重整化方案依赖或多过程联合约束的分析。

## R10

正确性直觉：

1. `beta0 > 0` 时，小耦合区 `beta(alpha) < 0`，所以 `mu` 升高时 `alpha_s` 下降；
2. 对 `N_f=5`，应当出现渐近自由；对 `N_f=17`，`beta0 < 0`，应失去该性质；
3. `R` 比随 `alpha_s` 增大而增大，因此可用于反演耦合强度；
4. 自动微分得到的导数应与手工导数一致，能排查实现层面的系数错误。

## R11

数值稳定策略：

- 在 `ln(mu)` 空间积分，减少跨尺度刚性问题；
- 对 `alpha` 设置极小正下界，避免浮点误差导致非物理负值；
- 对反演中 `q = R/base - 1` 做下界截断，防止噪声触发负判别式；
- 采用严格 `rtol/atol` 并检查 `solve_ivp` 成功标记；
- 非线性拟合使用有界参数区间，限制到合理物理范围。

## R12

关键参数与影响：

- `n_f`：决定 beta 系数，主导渐近自由判据；
- `mu_ref_gev`, `alpha_ref_true`：定义参考边界条件；
- `energy_max_gev`, `num_points`：决定可观测曲线的覆盖范围与分辨率；
- `noise_sigma`：决定反演与拟合难度；
- `loops`：选择一圈或两圈方程。

调参建议：

- 提高稳定性：先加密 `num_points`，再适度降低 `noise_sigma`；
- 想看高能效应：增大 `energy_max_gev`；
- 想看模型偏差：把拟合用一圈、生成用两圈，比较系统误差。

## R13

- 近似比保证：N/A（非近似优化算法条目）。
- 随机成功率保证：N/A（随机性仅用于可控合成噪声，不影响流程可执行性）。

可验证保证（由断言给出）：

- `N_f=5` 渐近自由且 `N_f=17` 非渐近自由；
- `alpha_s(mu)` 随 `mu` 增大单调下降；
- 反推 `alpha_ref` 与真值偏差受限；
- `R` 比拟合 RMSE 低于阈值；
- Torch 自动微分导数与解析导数误差在机器精度范围。

## R14

常见失效模式：

1. `mu_grid` 非升序或含非正值，导致 `ln(mu)` 非法；
2. beta 系数归一化因子写错（`2pi`、`4pi^2` 混淆）；
3. `R` 反演根选择错误（选到负根）；
4. 噪声过大导致观测点超出模型可逆区间；
5. 忽略 ODE/拟合成功标记，产生静默错误。

## R15

工程扩展方向：

- 加入阈值匹配（分段 `N_f`）与更高圈 beta；
- 引入真实实验数据（替换合成数据）；
- 扩展到更多可观测量（喷注率、事件形状等）；
- 将拟合升级为贝叶斯后验估计并输出不确定度区间。

## R16

相关条目：

- 渐近自由（Asymptotic Freedom）；
- 重整化群方程（RGE）；
- 夸克-胶子理论中的跑动耦合 `alpha_s(mu)`；
- `e+e-` 强子产生比与微扰 QCD 修正。

## R17

`demo.py` 交付能力清单：

- 显式实现 QCD 一圈/二圈 beta 系数与 ODE；
- 生成可复现实验风格数据并执行参数反演；
- 同时提供非线性拟合与线性回归两条反推路径；
- 用 `torch` 自动微分做导数级一致性校验；
- 无交互，单命令可运行。

运行方式：

```bash
cd Algorithms/物理-量子场论-0063-量子色动力学_(Quantum_Chromodynamics,_QCD)
uv run python demo.py
```

## R18

`demo.py` 的源码级算法流程（9 步）：

1. `main` 创建 `QCDConfig` 并由 `validate_config` 校验参数边界。  
2. `qcd_beta_coefficients` 与 `reduced_beta_coefficients` 计算 `beta0,beta1,b0,b1`，确定 RG 方程系数。  
3. `integrate_running_alpha` 在 `t=ln(mu)` 变量上调用 `solve_ivp` 积分 `d alpha/dt = -b0 alpha^2 - b1 alpha^3`，生成 `alpha_true(mu)`。  
4. `qcd_r_ratio` 把 `alpha_true` 映射为理论 `R_true`，再在 `main` 中加噪声得到 `R_observed`。  
5. `invert_r_ratio_to_alpha` 对每个观测点解二次方程 `c2 x^2 + x - q = 0`，得到 `alpha_from_R`。  
6. `fit_alpha_ref_from_r_data` 用 `least_squares` 调整 `alpha_ref`，每次迭代都重新积分 RG 并最小化 `R_model - R_observed`。  
7. `fit_one_loop_lambda_with_sklearn` 在 `1/alpha` 与 `ln(mu)` 的线性关系上做回归，提取 `Lambda_QCD` 与 `b0` 斜率。  
8. `torch_beta_derivative` 用 `torch.autograd` 计算 `d beta / d alpha`，并与解析导数 `-2b0 alpha - 3b1 alpha^2` 对比。  
9. `main` 打印系数表、逐点报告与拟合摘要，最后执行断言（渐近自由、单调性、拟合误差、导数一致性），通过后输出 `All checks passed.`。
