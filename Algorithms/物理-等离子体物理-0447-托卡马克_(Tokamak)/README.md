# 托卡马克 (Tokamak)

- UID: `PHYS-0427`
- 学科: `物理`
- 分类: `等离子体物理`
- 源序号: `447`
- 目标目录: `Algorithms/物理-等离子体物理-0447-托卡马克_(Tokamak)`

## R01

托卡马克（Tokamak）是受控热核聚变中最常见的磁约束装置：通过环向磁场与等离子体电流共同形成螺旋磁场，将高温等离子体限制在环形真空室内。

本条目的 MVP 目标不是做工程级反应堆设计，而是实现一个可运行、可审计的 `0D` 功率平衡算法：给定设备与工况参数，计算稳态温度解、聚变增益 `Q`、以及 Lawson 三乘积指标。

## R02

`demo.py` 采用的物理抽象是 `0D`（体平均）模型：

1. 等离子体只用体平均温度 `T` 与密度 `n` 描述，不解空间分布。
2. 反应类型固定为 D-T，设 `n_D = n_T = n_e/2`。
3. 约束时间 `tau_E` 用 ITER98(y,2) 标度律估算。
4. 功率项包括：外加辅助加热 `P_aux`、α粒子自加热 `P_alpha`、输运损失 `P_transport`、韧致辐射 `P_brem`。

## R03

核心功率平衡方程：

`P_net(T) = P_aux + P_alpha(T) - [P_transport(T) + P_brem(T)]`

稳态条件是 `P_net(T)=0`。

代码中各项定义为：

1. `P_fusion = V * (n_e^2/4) * <sigma v>(T) * E_fusion`
2. `P_alpha = (3.5/17.6) * P_fusion`
3. `W_th = 3 * n_e * (T_eV * e) * V`
4. `P_transport = W_th / tau_E`
5. `P_brem = V * C_brem * Z_eff * n_e^2 * sqrt(T_eV)`

## R04

能量约束时间使用 ITER98(y,2) 经验标度：

`tau_E = 0.0562 * H98 * I_p^0.93 * B_t^0.15 * n20^0.41 * P^-0.69 * R^1.97 * eps^0.58 * kappa^0.78 * M^0.19`

其中：

1. `eps = a/R`
2. `P` 在本 MVP 中取 `P_aux`（教学简化）
3. 其余量在 `TokamakConfig` 中显式给定

这样可以把“设备规模/形状/运行点”与“功率平衡结果”联系起来。

## R05

算法流程（高层）：

1. 从配置参数计算 `tau_E`。
2. 在温度区间 `2~30 keV` 上构建扫描表，逐点计算功率项与 `P_net`。
3. 在扫描表中寻找符号变化区间。
4. 用 `scipy.optimize.brentq` 在每个区间内求 `P_net(T)=0` 根。
5. 取高温根作为 burn branch 工作点。
6. 计算 `Q = P_fusion / P_aux`、`nTtau`、`Lawson_ratio`、`ignition_margin`。
7. 输出样例表、根列表、工作点摘要与自动校验结论。

## R06

输入输出约定：

1. 输入：无交互输入，参数写在 `TokamakConfig`。
2. 输出：
   - 扫描样例表（温度、聚变功率、损失功率、净功率）
   - 稳态温度根列表
   - 高温工作点指标（`Q`、`nTtau` 等）
   - 检查项与 `Validation: PASS/FAIL`

这保证脚本可直接用于自动验证流水线。

## R07

正确性依据（面向本 MVP）：

1. 功率平衡形式直接来源于受控聚变 0D 能量平衡。
2. D-T 反应率、α加热比例、Bremsstrahlung 和热能表达都在源码中显式写出，可审计。
3. 稳态解通过“先扫描找变号、再区间求根”的数值路径得到，避免盲目初值迭代。
4. 输出同时给出低温根与高温根，符合点火问题常见的多平衡分支现象。

## R08

复杂度分析：

设温度网格点数为 `N`，求得根个数为 `K`。

1. 扫描计算功率表：`O(N)`。
2. 变号区间检测：`O(N)`。
3. 每个根的 Brent 求根：约 `O(M)` 次函数评估（`M` 为迭代步数，通常很小）。

总时间复杂度可写作 `O(N + K*M)`，空间复杂度 `O(N)`。

## R09

数值稳定性处理：

1. `T` 在反应率函数内做 `clip`，避免 `T -> 0` 时出现奇异。
2. 所有功率与表格最终通过 `finite_scan` 检查 NaN/Inf。
3. 根求解前先做离散变号筛选，保证 `brentq` 调用区间合法。
4. 根列表做去重处理，避免网格点命中导致重复根。

## R10

最小工具栈：

1. `numpy`：向量化物理量计算、网格扫描、数值判断。
2. `scipy.optimize.brentq`：单变量有界稳态求根。
3. `pandas`：结构化结果表打印。

没有调用黑箱聚变求解器，关键算法都在 `demo.py` 明确可见。

## R11

运行方式：

```bash
cd Algorithms/物理-等离子体物理-0447-托卡马克_(Tokamak)
uv run python demo.py
```

成功运行应在末尾看到 `Validation: PASS`。

## R12

关键输出字段说明：

1. `T_keV`：离子/电子体平均温度（keV）。
2. `P_fusion_MW`：D-T 聚变总功率。
3. `P_alpha_MW`：α 粒子加热功率。
4. `P_loss_MW`：总损失功率（输运 + Bremsstrahlung）。
5. `P_net_MW`：净加热功率（稳态根处应接近 0）。
6. `Q`：聚变增益 `P_fusion/P_aux`。
7. `nTtau_keV_s_m3`：Lawson 三乘积。
8. `Lawson_ratio`：`nTtau / 3e21`。
9. `ignition_margin`：`P_alpha/P_loss`，用于观察是否接近自持燃烧。

## R13

脚本内置验证项：

1. `finite_scan`：扫描表全为有限数。
2. `tau_E_positive`：约束时间正值。
3. `roots_found`：至少找到一个平衡根。
4. `two_branches_detected`：检测到低温/高温两分支（当前参数下应成立）。
5. `root_residual_small`：选定工作点满足 `|P_net| < 1e-8 MW`。
6. `gain_Q_gt_1`：工作点具有净聚变增益（`Q>1`）。
7. `lawson_ratio_positive`：Lawson 指标有效且为正。

## R14

模型边界与简化：

1. 0D 体平均模型忽略剖面、输运屏障、边缘局域模等空间物理。
2. D-T 反应率采用教学型解析近似，不是高精度核数据库插值。
3. `tau_E` 仅按标度律估算，未与总加热功率做全自洽耦合迭代。
4. 未纳入杂质辐射谱线损失、中性束沉积剖面、MHD 稳定性边界。

## R15

常见失败模式：

1. 温度扫描区间过窄，可能漏掉高温根。
2. 参数设置过激（如密度太低、损失太大）时可能无根。
3. 人为把 `P_aux` 设为非正值会使 `tau_E` 标度非法。
4. 反应率或辐射系数若改到非物理量级，`Q` 与根位置会失真。

## R16

可扩展方向：

1. 把 `tau_E` 从固定 `P_aux` 扩展为与总损失功率自洽迭代。
2. 引入更高保真 D-T 反应率拟合（如 Bosch-Hale 参数化）。
3. 增加外回路：电流驱动功率、氚增殖约束、壁负荷约束。
4. 从 0D 升级到 1D（半径方向）输运方程，研究剖面演化。

## R17

适用场景：

1. 受控聚变课程中的“托卡马克功率平衡”算法演示。
2. 参数敏感性扫描的快速前处理器。
3. 更复杂等离子体模拟前的单元测试基线。
4. 用于解释“低温根/高温根/点火裕度”的最小可运行示例。

## R18

`demo.py` 的源码级算法流（8 步，展开第三方调用而非黑箱）：

1. `TokamakConfig` 固化机器参数、加热功率、温度扫描区间；`tau_e_ipb98_seconds` 用显式幂律公式计算 `tau_E`。
2. `dt_reactivity_m3_s` 按解析式逐点计算 D-T `\langle\sigma v\rangle(T)`，并对低温做 `clip`。
3. `power_terms_at_temperature` 逐项计算 `P_fusion/P_alpha/P_brem/P_transport/P_loss/P_net`，形成单温度状态。
4. `build_power_scan` 在温度网格上批量调用上一步，构建 `pandas.DataFrame`，并附加 `nTtau` 与 `Lawson_ratio`。
5. `find_equilibrium_roots` 先在离散网格上找 `P_net` 变号区间，再对每个区间调用 `brentq`。
6. `brentq` 在每个合法区间内执行“二分保号 + 插值加速”的一维有界根搜索，返回满足 `P_net(T)=0` 的温度根。
7. `main` 选择高温分支根作为运行点，调用 `summarize_operating_point` 计算 `Q`、`ignition_margin`、`nTtau`。
8. `main` 最后执行 7 项检查并输出 `Validation: PASS/FAIL`，失败则非零退出，便于自动化验证。
