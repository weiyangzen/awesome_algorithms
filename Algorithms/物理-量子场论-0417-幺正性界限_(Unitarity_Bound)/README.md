# 幺正性界限 (Unitarity Bound)

- UID: `PHYS-0398`
- 学科: `物理`
- 分类: `量子场论`
- 源序号: `417`
- 目标目录: `Algorithms/物理-量子场论-0417-幺正性界限_(Unitarity_Bound)`

## R01

幺正性界限来自 `S^\dagger S = I`。对 `2 -> 2` 散射做部分波展开后，每个角动量道 `a_l(s)` 必须满足幺正约束。常用的微扰树图判据是：

`|Re a_l(s)| <= 1/2`

它用于判断给定有效理论参数是否已进入“强耦合或微扰失效”区域。

## R02

在量子场论建模中，幺正性界限常用于：

- 限制接触相互作用或有效算符系数的大小；
- 估计新物理模型参数的可用范围（例如耦合常数、截断尺度）；
- 作为“理论可行性”快速筛查，在做更高阶计算前排除明显不自洽参数点。

## R03

本条目采用的核心公式：

1. 部分波定义
   `a_l(s) = (1/(32*pi)) * int_{-1}^{1} dcos(theta) P_l(cos(theta)) M(s, cos(theta))`
2. 树图级微扰幺正性判据
   `|Re a_l(s)| <= 1/2`
3. 演示模型（实标量 toy model）
   `M(s,t) = lambda + g^2 / (m_med^2 - t)`
4. 质心系下（相同外线质量 `m_ext`）
   `t = -(1/2) * (s - 4 m_ext^2) * (1 - cos(theta))`

此外，常数振幅 `M=lambda` 时有解析结果：`a_0 = lambda/(16*pi)`，可用于数值积分正确性校验。

## R04

MVP 目标不是实现完整现象学工具链，而是完成一个可运行、可验证的最小闭环：

- 从给定模型参数计算 `a_0(s)`；
- 在能量网格上扫描 `|Re a_0|` 与界限余量；
- 用根求解自动找出“临界 `lambda`（刚好触及 `1/2`）”；
- 给出满足/违反界限的两组对比样例。

## R05

设能量采样点数为 `N_s`，角度积分点数为 `N_theta`：

- 单个 `a_l(s)` 计算复杂度：`O(N_theta)`；
- 全能量扫描：`O(N_s * N_theta)`；
- Brent 一维根求解：`O(K * N_s * N_theta)`，`K` 为迭代次数（通常很小）；
- 空间复杂度：`O(N_s + N_theta)`。

MVP 默认 `N_s=24`、`N_theta=4097`，运行成本很低。

## R06

`demo.py` 的主要输出：

- 接触模型解析解校验误差 `|a0_numeric - a0_closed|`；
- 临界 `lambda_crit`（满足 `max_s |Re a0(s)| = 1/2`）；
- `safe` 与 `risky` 两组参数的最大值、界限余量、是否违规；
- 头部数据表（`sqrt_s, a_l, abs_re_a_l, margin_to_bound, violates_bound`）。

## R07

实现包含以下函数模块：

- `mandelstam_t`：从 `s, cos(theta)` 计算 `t`；
- `tree_level_amplitude`：计算 toy 振幅 `M(s,t)`；
- `partial_wave_a_l`：数值积分得到 `a_l(s)`；
- `scan_unitarity`：在能量网格上统计界限余量；
- `verify_contact_formula`：用解析 `a_0=lambda/(16*pi)` 做一致性检查；
- `find_critical_lambda`：用 `scipy.optimize.brentq` 求临界 `lambda`；
- `summarize_case`：打印单个情形的诊断摘要。

## R08

依赖与运行环境：

- Python `>=3.10`
- `numpy`：数组与数值积分网格
- `scipy.special`：Legendre 多项式 `P_l`
- `scipy.optimize`：Brent 根求解
- `pandas`：结果表格整理与输出

无交互输入，脚本可直接执行。

## R09

适用范围与假设：

- 适用于教学/原型层面的树图幺正性快速筛查；
- 采用单道、实振幅 toy 模型，默认聚焦 `l=0`（`s` 波）；
- 不包含环图修正、非弹性开道、复振幅吸收部分等高阶效应；
- 因此结论应理解为“微扰可用性指示”，而非完整现象学结论。

## R10

正确性检查逻辑：

1. 常数振幅存在解析 `a_0`，先验证积分实现无系统偏差；
2. 对给定参数扫描 `|Re a_0|`，直接比较 `1/2` 界限；
3. 用 Brent 求解 `max_s |Re a_0| - 1/2 = 0`，得到临界耦合；
4. 选取 `safe`（小于临界）与 `risky`（大于临界）参数，断言一过一不过。

## R11

数值稳定与工程细节：

- 角度积分使用足够密的均匀网格（默认 `4097` 点）；
- 使用 `np.trapezoid` 避免旧接口弃用警告；
- 临界根求解前先做区间包络检测，不满足时自动扩张上界；
- 若 `lambda=0` 已违规或长期无法包络根，会抛出 `RuntimeError` 明确失败原因。

## R12

`demo.py` 默认参数：

- 能量网格：`s in [0.1, 36.0]`（24 点）
- 外线质量：`m_ext = 0`
- 中介质量：`m_med = 3.0`
- 交换耦合：`g = 1.2`
- 对比参数：
  - `safe: lambda = 0.6 * lambda_crit`
  - `risky: lambda = 1.05 * lambda_crit`

该设置可稳定展示“满足/违反”两类结果。

## R13

本算法条目不涉及近似比或随机成功率保证（非优化/随机算法问题）。

可验证保证是确定性的：

- 接触模型解析一致性断言；
- `safe` 满足 `|Re a_0| <= 1/2`；
- `risky` 触发 `|Re a_0| > 1/2`。

只要上述断言通过，MVP 的核心逻辑链路即被程序化验证。

## R14

常见失效模式：

1. 能量网格覆盖过窄，导致错过最危险能区；
2. 角度积分点太少，`a_l` 数值误差放大；
3. 参数导致极点接近积分域，数值条件变差；
4. 把树图判据误当作完整幺正结论（忽略高阶与非弹性过程）。

## R15

可扩展方向：

- 扩展到 `l>0` 多个部分波通道；
- 加入复振幅与 Argand 圆几何检查（`Im a_l = |a_l|^2 + ...`）；
- 扫描二维/三维参数空间并导出 CSV；
- 接入 EFT 算符基，给出 Wilson 系数的自动界限估计。

## R16

相关主题：

- 光学定理（optical theorem）
- 有效场论中的部分波幺正性
- 散射振幅解析性质（unitarity / analyticity / crossing）
- Higgs 与纵向规范玻色子散射中的幺正性约束

## R17

交付内容：

- `README.md`：完成 R01-R18，覆盖定义、公式、复杂度、边界与源码流程；
- `demo.py`：可直接运行的最小实现，包含断言验证；
- `meta.json`：与任务元数据保持一致。

运行方式：

```bash
cd Algorithms/物理-量子场论-0417-幺正性界限_(Unitarity_Bound)
uv run python demo.py
```

## R18

源码级算法流程拆解（对应 `demo.py`，共 8 步）：

1. `main` 构造能量网格 `s_values` 与模型基准参数（`g, m_med`）。
2. `verify_contact_formula` 调用 `partial_wave_a_l` 计算常数振幅情形，并与解析式 `lambda/(16*pi)` 对比，得到基准误差。
3. `find_critical_lambda` 定义目标函数 `f(lambda)=max_s |Re a0|-1/2`，先检查并扩张包络区间。
4. `scipy.optimize.brentq` 在包络区间内迭代逼近根：每次用二分保持包络、再用割线/反二次插值加速收敛，直到 `xtol/rtol` 满足停止条件。
5. 每次评价 `f(lambda)` 时，`scan_unitarity` 在全能量网格调用 `partial_wave_a_l`，获得 `abs_re_a_l` 最大值。
6. `partial_wave_a_l` 内部先由 `scipy.special.eval_legendre` 生成 `P_l(cos(theta))`，再构造振幅 `M(s,t)` 并用 `np.trapezoid` 完成角度积分。
7. `summarize_case` 对 `safe`/`risky` 两组参数分别输出最大 `|Re a0|`、余量和是否违规，并打印前几行表格。
8. 末尾断言检查三件事：接触模型误差足够小、`safe` 不违规、`risky` 违规；全部通过则输出 `All checks passed.`。

该流程中第三方库仅提供基础数值原语（Legendre、多项式积分、标量根求解），幺正性判据、扫描逻辑和验证策略均在源码中显式实现。
