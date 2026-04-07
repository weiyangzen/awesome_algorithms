# 希格斯机制 (Higgs Mechanism)

- UID: `PHYS-0067`
- 学科: `物理`
- 分类: `量子场论`
- 源序号: `67`
- 目标目录: `Algorithms/物理-量子场论-0067-希格斯机制_(Higgs_Mechanism)`

## R01

希格斯机制的核心是“自发对称性破缺 + 规范场吸收 Goldstone 模式”：
原本无质量的规范场在真空中获得非零质量，而标量谱中留下一个实标量激发（Higgs 模式）。

本条目提供一个可运行的最小数值原型（MVP），用二维实标量表示 Abelian Higgs toy model 的复标量场，展示：

- 墨西哥帽势的真空环
- 梯度流自动选真空（破缺方向随机）
- 通过“单位规旋转”得到物理谱
- Hessian 特征值给出 `m_G^2 ~ 0` 与 `m_H^2 > 0`
- `m_A^2 = g^2 v^2` 的规范玻色子质量关系

## R02

理论背景（树级、经典）采用 Abelian Higgs 结构：

- 复标量场 `phi`
- `U(1)` 规范场 `A_mu`
- 对称破缺势能 `V(phi)`

当势能在 `|phi|=v` 处达到极小值时，真空不再位于对称点 `phi=0`，而是落到真空环上的某一方向。该方向选择使全局表示中的角向激发成为 Goldstone 模式；在规范理论中该自由度被规范场吸收，体现为规范场质量项。

## R03

使用的连续模型：

- 拉氏量：`L = -1/4 F_{mu nu}F^{mu nu} + |D_mu phi|^2 - V(phi)`
- 协变导数：`D_mu = partial_mu - i g A_mu`
- 势能：`V(phi) = - (mu^2/2) |phi|^2 + (lambda/4) |phi|^4`，其中 `mu>0, lambda>0`

把复标量写成两个实分量 `phi=(phi1, phi2)` 后，MVP 数值部分直接在 `(phi1, phi2)` 平面上工作。解析真空期望值：

- `v = mu / sqrt(lambda)`
- `m_H^2 = 2 mu^2`
- `m_G^2 = 0`（被吸收前）
- `m_A^2 = g^2 v^2`

## R04

离散与算法策略（与 `demo.py` 一致）：

- 忽略时空依赖，只研究均匀场的势能形状与局域谱（0 维 toy model）
- 对多个随机初值运行显式梯度流：`phi <- phi - lr * grad V(phi)`
- 每个 trial 收敛到真空环上的某一点，统计半径和相位
- 选取最低势能 trial 作为代表真空，并做一次 `U(1)` 旋转到单位规方向（`phi2 -> 0`）
- 在该点计算 Hessian 矩阵并求特征值，提取 Goldstone/Higgs 两个标量方向的 `m^2`
- 以 `g` 与数值真空 `v_num` 计算 `m_A^2 = g^2 v_num^2`

## R05

关键假设与边界：

- 仅树级经典势能，未包含量子修正与重整化群流
- 仅 Abelian `U(1)` toy model，不直接等价于标准模型完整 `SU(2)_L x U(1)_Y`
- 只演示真空结构和质量生成关系，不求散射振幅或圈图
- 忽略时空传播、规范固定细节和鬼场动力学

## R06

`demo.py` 的输入输出约定：

- 输入：脚本内固定超参数（`mu, lambda, g, lr, n_trials` 等）
- 无交互输入，支持直接 `uv run python demo.py`
- 输出：
1. 多个随机初值的梯度流收敛表
2. 单位规真空与 Hessian 特征值
3. 解析值 vs 数值值对照表
4. 校验项与 `Validation: PASS/FAIL`

## R07

主流程（高层）如下：

1. 设置模型参数并计算解析参考值 `v, m_H^2, m_G^2, m_A^2`。
2. 采样多个随机初值 `(phi1, phi2)`。
3. 对每个初值执行梯度流直到梯度范数足够小。
4. 汇总每个 trial 的收敛点、半径、相位和势能。
5. 选取最低势能的收敛点作为代表真空。
6. 旋转到单位规方向并计算 Hessian 特征值。
7. 计算规范玻色子质量平方 `m_A^2=g^2 v_num^2`。
8. 执行数值一致性校验并输出 PASS/FAIL。

## R08

设：

- trial 数为 `T`
- 每次梯度流迭代步数上限为 `K`
- 环扫描点数为 `M`

则复杂度为：

- 时间复杂度：`O(T*K + M)`（每步仅 2 维向量运算，常数很小）
- 空间复杂度：`O(T + M)`（主要存储 trial 结果和一维扫描数组）

## R09

数值稳定性处理：

- 设置梯度范数阈值 `grad_tol` 作为终止条件
- 使用较小学习率 `learning_rate=0.05` 防止在真空环附近震荡
- 使用多初值 trial，避免单初值路径偶然性
- 采用 `safe_relative_error` 处理解析值接近零的相对误差
- Goldstone 模式允许小阈值误差（`|m_G^2| < 1e-3`）

## R10

最小工具栈：

- `numpy`：势能、梯度、Hessian、线性代数和随机采样
- `pandas`：结果表格化输出（trial 汇总 + 质量对照）

未使用高层量子场论黑箱包；算法核心公式和步骤都在 `demo.py` 显式实现。

## R11

运行方式：

```bash
cd Algorithms/物理-量子场论-0067-希格斯机制_(Higgs_Mechanism)
uv run python demo.py
```

脚本无需任何命令行参数或交互输入。

## R12

关键输出字段说明：

- `phi1_init, phi2_init`：trial 初始场分量
- `phi1_final, phi2_final`：梯度流收敛后的场分量
- `radius_final`：收敛点半径 `sqrt(phi1^2 + phi2^2)`
- `phase_deg`：收敛相位角（度）
- `steps`：该 trial 使用的迭代步数
- `grad_norm`：终止时梯度范数
- `V_final`：收敛点势能
- `Hessian eigenvalues [m_G^2, m_H^2]`：标量谱质量平方
- `Analytic vs Numerical` 表：`v, m_H^2, m_G^2, m_A^2` 的解析/数值/误差

## R13

内置校验条件（全部满足即 PASS）：

1. `ring_scan_min_close`: 一维半径扫描最小值接近解析 `v`
2. `selected_vacuum_radius_close`: 代表真空半径接近解析 `v`
3. `all_trials_converge_to_ring`: 所有 trial 都收敛到真空环邻域
4. `goldstone_mass_near_zero`: `|m_G^2| < 1e-3`
5. `higgs_mass_matches_theory`: `m_H^2` 相对误差 < `1e-2`
6. `gauge_mass_matches_theory`: `m_A^2` 相对误差 < `1e-2`
7. `higgs_mode_positive`: `m_H^2 > 0`

## R14

当前 MVP 局限：

- 仅演示势能和局域质量矩阵，不含完整时空动力学
- 不包含非阿贝尔群、规范固定路径积分、Faddeev-Popov 结构
- 未考虑圈修正、温度效应、相变动力学
- 与实验可观测量（截面、衰变宽度）仍有距离

## R15

可扩展方向：

- 升级到格点场论版本（含空间维度和梯度项）
- 加入有效势一圈修正（Coleman-Weinberg）
- 扩展到 `SU(2) x U(1)` 结构并区分 `W/Z` 质量来源
- 用自动微分框架（如 PyTorch）研究更复杂参数拟合
- 对比不同数值积分器（显式 Euler vs 自适应 ODE）

## R16

适用场景：

- 量子场论课程中“希格斯机制”概念可视化教学
- 快速验证参数变化对 `v, m_H, m_A` 的影响
- 在更大 QFT 仿真项目中充当可审计的最小单元测试
- 用于说明“自发破缺 + 规范吸收”这条算法化推理链

## R17

相关方法对比：

- 纯解析推导：最清晰但缺少“随机初值如何落入真空环”的数值直观
- 直接调用高层符号包：开发快，但容易把关键机制隐藏在黑箱中
- 本条目 MVP：保留解析结果，同时提供最小数值实验，验证质量关系并输出可检查日志

因此该实现定位为“教学与审计优先”的希格斯机制算法骨架。

## R18

`demo.py` 源码级算法流（8 步）：

1. `HiggsConfig` 固定 `mu, lambda_, gauge_coupling, learning_rate` 等超参数；`main` 先计算解析参考量 `v_analytic, m_H2_analytic, m_G2_analytic, m_A2_analytic`。
2. `potential` 按 `V = -mu^2|phi|^2/2 + lambda|phi|^4/4` 计算势能，`gradient` 返回 `(-mu^2 + lambda|phi|^2) * phi`。
3. 对每个随机初值调用 `gradient_flow`，执行显式迭代 `phi <- phi - lr * grad`，直到 `||grad|| < grad_tol` 或达到步数上限。
4. 把每个 trial 的初值、终值、半径、相位、步数、终止梯度和终值势能写入 `trial_df`，并选势能最低的 trial 作为代表真空。
5. `rotate_to_unitary_gauge` 计算真空相位 `theta` 并做二维旋转，把真空映射到单位规方向（数值上 `phi2 ~ 0`）。
6. `hessian` 在单位规真空处构建 `2x2` 质量矩阵，`np.linalg.eigvalsh` 提取特征值并识别 `m_G^2` 与 `m_H^2`；再用 `m_A^2 = g^2 v_num^2` 得到规范场质量平方。
7. 对 `v, m_H^2, m_G^2, m_A^2` 构建 `mass_df`，计算绝对误差和相对误差（零点附近用 `safe_relative_error` 稳定处理）。
8. 逐条执行 7 个一致性检查并打印 `[PASS]/[FAIL]`；全部通过输出 `Validation: PASS`，否则 `SystemExit(1)`。
