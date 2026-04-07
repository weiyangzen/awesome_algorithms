# 维度正规化 (Dimensional Regularization)

- UID: `PHYS-0386`
- 学科: `物理`
- 分类: `量子场论`
- 源序号: `405`
- 目标目录: `Algorithms/物理-量子场论-0405-维度正规化_(Dimensional_Regularization)`

## R01

维度正规化（Dimensional Regularization）是一种处理量子场论发散积分的标准方法。核心思想是把积分维数从 `d=4` 解析延拓到 `d=4-2epsilon`，在 `epsilon>0` 时积分变为收敛，再通过 `epsilon -> 0` 的 Laurent 展开把发散显式化为 `1/epsilon` 极点。

本条目聚焦最小代表性积分：

`I_d(Delta)=mu^(2epsilon) ∫ d^d k/(2pi)^d * 1/(k^2+Delta)^2`

这里 `mu` 是重整化尺度，用来保持量纲一致。

## R02

在微扰量子场论中，环图积分几乎必然出现 UV 发散。维度正规化的优势是：

- 保持洛伦兹协变与规范对称结构（相比硬截止常更干净）；
- 发散结构统一编码为 `1/epsilon`、`1/epsilon^2` 等极点；
- 与 MS / MS-bar 重整化方案自然兼容，便于推导 beta 函数与跑动耦合。

## R03

本 MVP 解决三个具体问题：

- 用 Gamma 函数闭式公式计算维度正规化积分；
- 对 `epsilon -> 0` 做到 `O(epsilon^0)` 的 Laurent 近似，并验证误差阶；
- 实现 MS 与 MS-bar 扣除，数值确认有限部分与理论表达一致。

另外加入一个 `d<4` 的径向数值积分交叉验证，确保闭式公式不是“只写不验”。

## R04

关键公式如下（`alpha=2`）：

1. 通用欧氏积分恒等式：
`∫ d^d k/(2pi)^d * 1/(k^2+Delta)^alpha
 = 1/(4pi)^(d/2) * Gamma(alpha-d/2)/Gamma(alpha) * Delta^(d/2-alpha)`

2. 令 `d=4-2epsilon, alpha=2`：
`I_d(Delta)=mu^(2epsilon)/(4pi)^(2-epsilon) * Gamma(epsilon) * Delta^(-epsilon)`

3. Laurent 展开到常数项：
`I_d(Delta)=1/(16pi^2) * [1/epsilon - gamma_E + ln(4pi) + ln(mu^2/Delta)] + O(epsilon)`

4. 方案定义：
- `MS` 仅减去 `1/(16pi^2)*(1/epsilon)`；
- `MS-bar` 减去 `1/(16pi^2)*(1/epsilon - gamma_E + ln(4pi))`。

## R05

设 `n` 为 `epsilon` 采样点数量：

- 闭式积分逐点评估：`O(n)`；
- Laurent 近似与误差统计：`O(n)`；
- 数值径向积分（单次 `quad`）：在给定容差下可视为 `O(m)`（`m` 为自适应采样数）。

整体主流程时间复杂度 `O(n + m)`，空间复杂度 `O(n)`。

## R06

`demo.py` 的输出包括：

- `epsilon` 网格上 exact vs Laurent 的逐行对比；
- `|exact-laurent|/epsilon` 的误差阶检查（验证截断误差为 `O(epsilon)`）；
- MS / MS-bar 有限部分与理论值对比；
- 在 `d=3.2` 时，闭式公式与径向数值积分的相对误差。

脚本含 `assert`，可直接用于自动化校验。

## R07

优点：

- 选用量子场论最典型的一环标量积分，物理代表性强；
- 解析表达、近似展开、重整化减法、数值积分交叉验证形成闭环；
- 仅依赖 `numpy + scipy`，运行负担小。

局限：

- 只覆盖一类标量积分（未涉及张量积分与 Feynman 参数多尺度结构）；
- 仅展示到 `O(epsilon^0)`，未展开更高阶 `epsilon` 系数；
- 未实现符号推导引擎（例如自动 IBP / 主积分约化）。

## R08

前置知识：

- 复变函数与 Gamma 函数基础；
- `epsilon` 展开与重整化方案概念；
- 欧氏动量积分与球坐标分离。

环境要求：

- Python `>=3.10`
- `numpy`
- `scipy`

## R09

适用场景：

- 教学展示维度正规化与 MS/MS-bar 的最小数值实现；
- 快速验证某类一环积分的极点结构与有限项；
- 作为后续更复杂 QFT 计算（自能、顶角修正）的基线模板。

不适用场景：

- 多环图、阈值奇点、复杂质量谱；
- 需要自动化生成 Feynman 图并全流程符号化约化；
- 非微扰问题（维度正规化本身是微扰框架工具）。

## R10

正确性直觉：

1. 当 `epsilon>0`，积分的 UV 幂次发散被“降维”抑制，积分应收敛；
2. `epsilon -> 0` 时，发散以 `1/epsilon` 形式显式出现；
3. 减去 MS 或 MS-bar 极点项后，结果应趋向有限；
4. 在 `d<4` 区间，直接数值积分应与闭式 Gamma 函数表达一致。

## R11

数值稳定策略：

- 避免在 `epsilon=0` 直接计算，使用小正数（如 `1e-6`）检查极限；
- 用 `|exact-laurent|/epsilon` 而非绝对误差判断展开阶，防止误判；
- 径向积分仅在 `d<4` 执行，避免真实发散点；
- `quad` 采用较严格容差并检查返回误差上界。

## R12

关键参数：

- `delta`：传播子中的正参数（可理解为质量或外参组合）；
- `mu`：重整化尺度，控制有限项中的 `ln(mu^2/delta)`；
- `eps_grid`：用于展开误差评估的一组 `epsilon`；
- `d_test`：径向数值积分维度（需 `<4`）。

调参建议：

- 若想看更强极点主导，可减小 `epsilon`；
- 若想看高维接近临界发散，可让 `d_test` 更接近 `4`（但数值积分会更敏感）；
- 改变 `mu` 可直观看到有限项的尺度依赖。

## R13

- 近似比保证：N/A（非优化问题）。
- 随机成功率保证：N/A（流程确定性，不含随机采样）。

可验证保证：

- `Laurent` 截断误差按 `O(epsilon)` 缩放；
- MS / MS-bar 有限部分与理论极限一致；
- `d<4` 的数值径向积分与闭式公式相对误差在阈值内。

## R14

常见失效模式：

1. 把 `epsilon` 取为 0 导致 `Gamma(epsilon)` 极点直接溢出；
2. `delta<=0` 破坏当前欧氏正定假设，可能触发复相位/分支问题；
3. 在 `d>=4` 仍试图做无截断径向数值积分，会遇到真实 UV 发散；
4. 忽略 `mu^(2epsilon)` 因子会导致量纲与有限项常数错位。

## R15

可扩展方向：

- 增加 Feynman 参数化，处理两质量或外动量依赖的一环函数；
- 扩展到张量积分并演示 Passarino-Veltman 约化；
- 引入 `sympy` 进行更高阶 `epsilon` 系数自动展开；
- 扩展到多环主积分并结合 IBP/Laporta 工作流。

## R16

相关主题：

- Pauli-Villars 与硬截止正规化（对比对称性保持）；
- 重整化群与 beta 函数；
- Ward 恒等式与规范不变性；
- OPE 与短程发散结构。

## R17

MVP 功能清单：

- 实现一环标量积分闭式公式 `I_d(Delta)`；
- 实现 Laurent 到 `O(epsilon^0)` 近似；
- 实现 MS / MS-bar 扣除并数值验证有限部分；
- 用 `d<4` 径向积分做独立交叉校验；
- 用断言保证脚本可直接作为非交互校验程序。

运行方式：

```bash
cd Algorithms/物理-量子场论-0405-维度正规化_(Dimensional_Regularization)
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程（8 步）：

1. `exact_one_loop_integral`：根据 `d=4-2epsilon` 与 Gamma 闭式公式计算裸积分值。  
2. `laurent_up_to_constant`：构造 `1/epsilon + 常数` 的截断展开，用于近似对照。  
3. `ms_counterterm`：生成 MS 的纯极点反项。  
4. `msbar_counterterm`：生成 MS-bar 反项（极点加 `-gamma_E+ln4pi` 常数）。  
5. `sphere_surface_area`：计算 `S_(d-1)`，为径向积分分离角向体积因子。  
6. `numeric_radial_integral`：在 `d<4` 时用 `quad` 直接积分 `k^(d-1)/(k^2+Delta)^2`，得到独立数值结果。  
7. `main` 前半段：对 `eps_grid` 计算 exact 与 Laurent 并验证误差按 `O(epsilon)` 缩放。  
8. `main` 后半段：执行 MS/MS-bar 有限项校验，再做 `d=3.2` 的闭式 vs 数值交叉验证并断言通过。  

第三方库未被当作黑盒：`scipy.special.gamma` 仅提供特殊函数值，`scipy.integrate.quad` 仅执行基础数值积分；维度正规化、展开、反项定义与验证逻辑都在源码中显式实现。
