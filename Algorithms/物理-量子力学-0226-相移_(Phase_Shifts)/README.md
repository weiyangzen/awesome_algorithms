# 相移 (Phase Shifts)

- UID: PHYS-0225
- 学科: 物理
- 分类: 量子力学
- 源序号: 226
- 目标目录: Algorithms/物理-量子力学-0226-相移_(Phase_Shifts)

## R01

相移是量子散射里最核心的中间量之一。对中心势散射，完整散射信息可拆成各偏波 l 的相移 δ_l(E)，再由这些相移组装散射截面。

本条目实现一个可运行 MVP：针对球对称有限深方势阱，计算 l 等于 0、1、2 的相移，并对比解析边界匹配结果与数值积分结果。

## R02

MVP 的问题定义：
- 势函数：V(r) 在 r 小于 a 时等于负 V0，在 r 大于等于 a 时等于 0，且 V0 和 a 都为正。
- 输入（脚本内固定参数）：m、hbar、V0、a、能量网格、l_max、Numerov 网格参数。
- 输出：
1. 相移明细表（解析、数值、误差）。
2. 总截面对比表（解析相移 vs 数值相移）。
3. 验证指标与 Validation: PASS 或 FAIL。

脚本无需交互输入，直接运行即可复现。

## R03

中心势散射的关键关系：

1. 波数 k 等于 sqrt(2mE) 除以 hbar。
2. 偏波展开后，约化径向方程为：
u 的二阶导数加上 [2m(E-V)/hbar 平方 减 l(l+1)/r 平方] 乘 u 等于 0。
3. 外区 r 大于 a 时渐近形式：
u_l(r) 正比于 cos(δ_l) 乘 j_l(kr) 减 sin(δ_l) 乘 y_l(kr)。
4. 截断到 l_max 的总截面近似：
σ_tot 约等于 (4π/k 平方) 乘 sum[(2l+1) 乘 sin 平方(δ_l)]。

## R04

方势阱下每个 l 的解析相移来自 r 等于 a 处函数值和导数连续。

- 内区波数 q 等于 sqrt(2m(E+V0)) 除以 hbar。
- 记 j_l 和 y_l 为球贝塞尔函数，d_dx 代表对自变量求导。
- 解析匹配可写成：tan(δ_l) 等于 N 除以 D。

其中：
- N 等于 q 乘 j_l(ka) 乘 d_dx(j_l)(qa) 减 k 乘 j_l(qa) 乘 d_dx(j_l)(ka)
- D 等于 q 乘 y_l(ka) 乘 d_dx(j_l)(qa) 减 k 乘 d_dx(y_l)(ka) 乘 j_l(qa)

MVP 用 arctan2(N, D) 计算相移主值。

## R05

数值相移采用 Numerov 加渐近对数导数匹配：

1. 在区间 [r_min, r_max] 网格上，用 Numerov 推进径向方程得到 u(r)。
2. 在外区窗口 [match_r1, match_r2] 选取绝对值最大的 u 点，避开节点不稳定。
3. 用中心差分估计 u_derivative，构造 beta 等于 u_derivative 除以 (k 乘 u)。
4. 用 tan(δ_l) 等于 (beta 乘 j_l 减 d_j_l) 除以 (beta 乘 y_l 减 d_y_l) 得到数值相移。

这个做法比简单两点振幅比值更稳健。

## R06

相移存在 δ_l 与 δ_l 加 nπ 的等价性，因此误差不能直接做普通差值。

脚本采用 mod π 误差：
- diff 等于 δ_num 减 δ_ref。
- wrapped 等于 ((diff + π/2) mod π) 减 π/2。
- phase_abs_err 等于 wrapped 的绝对值。

这样比较的是物理等价类里的最小角差，避免分支跳变造成伪大误差。

## R07

demo.py 的高层流程：

1. 校验配置参数合法性。
2. 遍历能量网格 E。
3. 对每个 l 从 0 到 l_max：
- 计算解析相移。
- 数值积分并提取数值相移。
- 计算 mod π 相移误差并记表。
4. 由两组相移分别计算总截面并比较相对误差。
5. 汇总最大误差、均值误差、有限性检查，输出 PASS 或 FAIL。

## R08

时间复杂度主导项：

- 设能量点数为 N_E，偏波数为 N_l，径向网格点数为 N_r。
- 每个 (E,l) 的 Numerov 成本是 O(N_r)。
- 总体约为 O(N_E 乘 N_l 乘 N_r)。

空间复杂度：
- 单次积分存 r、u、g 等向量，约 O(N_r)。
- 结果表规模为 O(N_E 乘 N_l)。

## R09

稳定性设计：

1. 近原点用 u(r) 近似 r 的 l+1 次幂初始化，避免奇点直接求值。
2. Numerov 过程中加入大数缩放，防止极端溢出。
3. 外区匹配点在窗口内选绝对值最大的 u，降低 u 近零时对数导数发散风险。
4. 误差比较采用 mod π，减少分支不连续导致的误判。
5. finite_ok 检查确保没有 NaN 或 Inf。

## R10

技术栈（最小可用）：

- numpy：数组运算、网格与误差处理。
- scipy.special：spherical_jn 和 spherical_yn 及其导数。
- pandas：结构化输出表。
- 标准库 dataclasses。

第三方库未替代算法主线，Numerov、匹配与验证逻辑都在源码内显式实现。

## R11

运行方式：

cd Algorithms/物理-量子力学-0226-相移_(Phase_Shifts)
uv run python demo.py

预期输出：相移明细表、总截面对比表、验证指标与 Validation: PASS。

## R12

主要输出字段：

相移明细表：
- energy：入射能量
- ell：偏波量子数 l
- delta_analytic_rad、delta_numeric_rad：弧度相移
- delta_analytic_deg、delta_numeric_deg：角度相移
- phase_abs_err：按 mod π 定义的绝对误差

总截面对比表：
- k：波数
- sigma_analytic、sigma_numeric
- sigma_rel_err

## R13

脚本内置验收条件：

1. max_phase_abs_err 小于等于 phase_abs_tol（默认 6e-2）。
2. max_sigma_rel_err 小于等于 sigma_rel_tol（默认 2e-1）。
3. 全部数值有限（finite_ok 为 True）。

三项同时满足则 Validation: PASS，否则脚本以非零状态退出。

## R14

当前实现边界：

- 仅覆盖球对称方势阱，不含任意势函数反演与实验拟合。
- 只做 l 小于等于 2 的低阶偏波截断，不是全偏波精确和。
- 仅处理弹性散射、定态、单通道问题。
- 解析式和数值法比较的是单粒子非相对论框架。

## R15

可扩展方向：

1. 支持高斯势、Yukawa 势、Wood-Saxon 势等更通用势函数。
2. 扩展到更高 l_max 并做偏波收敛性扫描。
3. 用自适应步长或对数网格改进低能区稳定性。
4. 增加微分截面角分布 dσ/dΩ 输出与绘图。
5. 引入参数反演（由相移和截面反推势参数）。

## R16

典型应用场景：

- 核散射与低能有效势模型教学验证。
- 散射理论课程中的相移到截面数值实验。
- 更复杂散射代码（偏波求和模块）的单元测试基线。
- 量子散射算法工程中的快速 sanity check。

## R17

与其他实现路线对比：

- 相比纯解析推导：本条目给出可运行数值链路，便于回归测试。
- 相比纯黑箱求解器：相移提取、误差定义、稳定性处理都可审计。
- 相比仅数值积分：加入解析匹配结果作为内部参照，可及时发现符号和分支错误。

## R18

demo.py 源码级算法流程拆解（9 步）：

1. PhaseShiftConfig 定义物理参数、能量和偏波范围、Numerov 网格和验收阈值。
2. validate_config 做输入边界检查（正值、区间顺序、网格规模）。
3. phase_shift_analytic_square_well 调用 spherical_jn 和 spherical_yn 及导数，按边界匹配公式计算解析 δ_l。
4. numerov_radial_solution 构造 g(r)，以 u 近似 r 的 l+1 次幂初始化并用 Numerov 递推得到径向解。
5. phase_shift_from_asymptotic_log_derivative 在外区选稳健匹配点，算 beta=u_derivative/(k*u)，由贝塞尔基函数求数值 δ_l。
6. phase_error_mod_pi 把 δ_num-δ_ref 映射到 [-π/2, π/2)，得到物理等价下的最小误差。
7. total_cross_section 按偏波和式计算 σ_tot。
8. run_phase_shift_mvp 双循环遍历 E 和 l，生成两张 pandas 表并汇总最大和均值指标。
9. main 打印配置、明细表、总截面对比和验证结论，失败时 SystemExit(1)。
