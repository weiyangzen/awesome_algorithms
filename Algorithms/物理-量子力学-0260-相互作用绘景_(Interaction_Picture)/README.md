# 相互作用绘景 (Interaction Picture)

- UID: `PHYS-0257`
- 学科: `物理`
- 分类: `量子力学`
- 源序号: `260`
- 目标目录: `Algorithms/物理-量子力学-0260-相互作用绘景_(Interaction_Picture)`

## R01

相互作用绘景（Interaction Picture）是介于薛定谔绘景与海森堡绘景之间的一种表述。它将哈密顿量拆成

`H(t) = H0 + V(t)`

并把由 `H0` 生成的“快演化”吸收到态矢变换中，仅让态在扰动项 `V` 的变换后版本 `V_I(t)` 下演化：

`i hbar d|psi_I>/dt = V_I(t) |psi_I>`，其中 `V_I(t) = U0^dagger(t) V(t) U0(t)`，`U0(t)=exp(-iH0t/hbar)`。

## R02

典型使用场景：

- 时间依赖微扰理论与 Dyson 级数推导
- 原子与光场耦合（旋转波近似、共振驱动）
- 在“可解主项 + 小扰动项”结构下做数值积分
- 量子信息中驱动门操作的近似分析与误差评估

## R03

核心数学关系：

1. 绘景变换
   `|psi_I(t)> = U0^dagger(t) |psi_S(t)>`
2. 相互作用绘景下的方程
   `i hbar d|psi_I>/dt = V_I(t) |psi_I>`
3. 反变换回薛定谔绘景
   `|psi_S(t)> = U0(t) |psi_I(t)>`
4. 若 `H = H0 + V` 与时间无关，薛定谔绘景精确解为
   `|psi_exact(t)> = exp(-i(H0+V)t/hbar)|psi(0)>`

因此可用“相互作用绘景数值积分”与“全哈密顿量精确指数”互相校验。

## R04

直观上，相互作用绘景做了“参考系旋转”：

- 先跟着 `H0` 的主振荡一起转，消去快速相位
- 再在这个旋转坐标系里只看扰动 `V` 的有效作用
- 若 `V` 较小或振荡快，积分更稳定且更容易看清主导机制

这和经典力学里换到共转坐标系处理小扰动是同类思想。

## R05

正确性要点：

1. 变换 `|psi_I> = U0^dagger |psi_S>` 是幺正变换，概率范数保持。
2. 对同一 `H=H0+V`，由相互作用绘景积分再反变换，理论上应与薛定谔绘景精确解一致。
3. `demo.py` 用两条路径比较 `||psi_from_I - psi_exact||_2`，作为数值正确性诊断。
4. 同时跟踪态范数偏差，确认数值积分没有产生明显非物理漂移。

## R06

复杂度（`n` 维希尔伯特空间，`m` 个输出时间点）：

- 单次矩阵指数 `expm(n x n)`：典型 `O(n^3)`
- `V_I(t)=U0^dagger V U0`：两次矩阵乘法，`O(n^3)`
- ODE 每一步右端计算：`O(n^2)`（矩阵向量乘），但若每步现算 `V_I` 则含 `O(n^3)` 成本
- 总体约 `O(m n^3 + N_step * n^3)`（教学级二维系统下可忽略）

## R07

标准实现步骤：

1. 设定 `H0`、`V`、初态 `|psi(0)>` 与时间网格。
2. 构造 `U0(t)=exp(-iH0t/hbar)`。
3. 计算 `V_I(t)=U0^dagger(t) V U0(t)`。
4. 在相互作用绘景积分 `d|psi_I>/dt = -(i/hbar) V_I(t)|psi_I>`。
5. 通过 `|psi_S>=U0|psi_I>` 回到薛定谔绘景。
6. 独立计算全哈密顿量精确解 `exp(-i(H0+V)t/hbar)|psi0>`。
7. 统计误差、范数偏差和观测量（如激发态概率）。

## R08

`demo.py` 的 MVP 设计：

- 模型：两能级系统，`H0=(Delta/2) sigma_z`，`V=(Omega/2) sigma_x`
- 路线 A：相互作用绘景 ODE 积分 + 反变换
- 路线 B：薛定谔绘景全哈密顿量精确矩阵指数
- 路线 C：一阶 Dyson 近似（非黑箱，显式离散积分）
- 输出：`time, P1_exact, P1_from_I, abs_err, P1_dyson1, dyson_abs_err, norm_err_I`

## R09

`demo.py` 接口约定：

- `unitary_from_hamiltonian(h, t, hbar=1.0) -> np.ndarray`
- `interaction_hamiltonian(h0, v, t, hbar=1.0) -> np.ndarray`
- `integrate_interaction_state(psi0, h0, v, t_eval, hbar=1.0) -> list[np.ndarray]`
- `recover_schrodinger_state(psi_i, h0, t, hbar=1.0) -> np.ndarray`
- `exact_schrodinger_state(psi0, h_total, t, hbar=1.0) -> np.ndarray`
- `dyson_first_order_state(psi0, h0, v, t, hbar=1.0, n_steps=400) -> np.ndarray`
- `excited_population(psi) -> float`

## R10

测试策略：

- 数值一致性：比较路线 A 与路线 B 的态矢差 `||psi_A-psi_B||_2`
- 概率一致性：比较激发态概率 `|psi_1|^2` 的误差
- 近似质量：对比一阶 Dyson 与精确解，确认其为“可用但有误差”的低阶近似
- 物理约束：检查 `||psi||^2` 是否接近 1

## R11

边界与异常处理：

- 初态非一维向量或零向量：抛出 `ValueError`
- `t_eval` 非一维或未单调递增：抛出 `ValueError`
- ODE 求解失败：抛出 `RuntimeError`
- Dyson 积分步数 `n_steps<=0`：抛出 `ValueError`

## R12

与相关方法关系：

- 与薛定谔绘景：完全等价，只是分配“时间依赖”的方式不同
- 与海森堡绘景：海森堡把时间依赖主要放到算符；相互作用绘景是折中
- 与微扰理论：Dyson 级数天然在相互作用绘景展开，是其主战场

## R13

示例参数（`demo.py`）：

- 自然单位：`hbar = 1`
- `Delta = 2.0`，`Omega = 0.7`
- 初态 `|0> = [1, 0]^T`
- 时间区间 `t in [0, 8]`，采样 11 个点
- Dyson 一阶离散积分步数 `n_steps = 600`

该组参数保证：

- 相互作用绘景与精确解误差很小（数值方法正确）
- 一阶 Dyson 近似有可见偏差（体现近似阶数影响）

## R14

工程实现注意点：

- `solve_ivp` 主输入是实向量，MVP 用“实部/虚部拼接”规避复杂 dtype 路径差异
- `V_I(t)` 每次 RHS 调用都需计算 `U0^dagger V U0`，高维问题应缓存或用结构化算法
- 一阶 Dyson 非幺正，长期演化下误差会积累，不应当作高精度替代

## R15

最小示例的物理含义：

- `H0` 让系统围绕 `z` 轴快速进动
- `V` 负责在 `|0>, |1>` 间耦合跃迁
- 相互作用绘景把 `H0` 的快相位提掉后，数值上更直接地积分“真实耦合效应”
- 输出表格中 `P1_from_I` 与 `P1_exact` 高度一致，证明绘景转换实现正确

## R16

可扩展方向：

- 替换为显式时间依赖扰动 `V(t)`（如脉冲驱动）
- 引入旋转波近似并与全模型数值解对照
- 扩展到多能级系统并比较稀疏矩阵加速收益
- 在开放系统中接入 Lindblad 项，形成“相互作用绘景 + 主方程”框架

## R17

本条目交付内容：

- `README.md`：完成 R01-R18，覆盖定义、正确性、复杂度、工程实现与源码流程
- `demo.py`：可直接运行、无交互输入的最小 MVP
- `meta.json`：保持与任务元数据一致

运行方式：

```bash
uv run python Algorithms/物理-量子力学-0260-相互作用绘景_(Interaction_Picture)/demo.py
```

或在目录内：

```bash
uv run python demo.py
```

## R18

源码级算法流程拆解（对应 `demo.py`，共 9 步）：

1. **输入标准化与校验**
   归一化 `|psi0>`，检查时间网格合法，确保积分起点与初值物理可行。

2. **构造可解主项演化算符 `U0(t)`**
   `unitary_from_hamiltonian` 调用 `expm(-iH0t/hbar)`，得到“参考系旋转”所需幺正矩阵。

3. **建立相互作用绘景哈密顿量**
   `interaction_hamiltonian` 计算 `V_I(t)=U0^dagger V U0`，把扰动映射到旋转参考系。

4. **复态转实向量以适配 ODE 求解器**
   `pack_state`/`unpack_state` 将复向量拆为 `[Re, Im]`，使 `solve_ivp` 可稳定处理。

5. **数值积分相互作用绘景动力学**
   在 `rhs_interaction_real` 中计算 `d|psi_I>/dt = -(i/hbar)V_I(t)|psi_I>`，`solve_ivp` 在 `t_eval` 上给出 `|psi_I(t)>`。

6. **反变换回薛定谔绘景**
   `recover_schrodinger_state` 执行 `|psi_S>=U0|psi_I>`，还原可与实验可观测量直接对应的态。

7. **独立生成精确参考解**
   `exact_schrodinger_state` 使用 `H_total=H0+V` 的矩阵指数给出基准轨迹，避免同源误差。

8. **构造一阶 Dyson 近似**
   `dyson_first_order_state` 通过离散梯形积分近似 `int_0^t V_I(t') dt'`，计算
   `|psi_I^(1)> = (I - i/hbar * integral)|psi0>` 再反变换，用于展示低阶近似误差。

9. **输出误差与观测量表**
   计算 `P1`、`|P1_from_I-P1_exact|`、`|P1_dyson-P1_exact|`、范数偏差，并汇总最大误差指标，完成算法正确性与近似阶数的联合验证。
