# 冯诺依曼方程 (von Neumann Equation)

- UID: `PHYS-0259`
- 学科: `物理`
- 分类: `量子力学`
- 源序号: `262`
- 目标目录: `Algorithms/物理-量子力学-0262-冯诺依曼方程_(von_Neumann_Equation)`

## R01

冯诺依曼方程描述闭合量子系统中密度矩阵 `rho(t)` 的时间演化：

`d rho / dt = -(i / hbar) [H, rho]`

其中 `H` 是哈密顿量，`[H, rho] = H rho - rho H` 是对易子。它是薛定谔方程在“密度矩阵表述”下的对应形式，可同时处理纯态和混态。

## R02

典型应用场景：

- 两能级系统（自旋-1/2、量子比特）的相干演化模拟
- 量子信息中密度矩阵轨迹计算与保真度分析
- 量子统计中的混态动力学研究
- Lindblad 主方程（开放系统）的无耗散极限基线

## R03

数学定义与基本约束：

- 密度矩阵满足：`rho = rho^dagger`、`Tr(rho)=1`、`rho >= 0`
- 闭合系统下，形式解为
  `rho(t) = U(t) rho(0) U^dagger(t)`
- 若 `H` 与时间无关，则
  `U(t) = exp(-i H t / hbar)`

这说明问题可以通过两条等价路线求解：

1. 直接用矩阵指数求 `U(t)`，再做共轭变换
2. 把 `rho` 展平后作为常微分方程数值积分

## R04

直观理解：

- `H` 生成系统“旋转”方向与频率
- 对易子 `[H, rho]` 衡量 `rho` 与 `H` 的不相容程度
- 若 `[H, rho]=0`，则 `d rho/dt = 0`，状态在该哈密顿量下不演化
- 若不对易，`rho` 在希尔伯特空间中做幺正轨道运动

## R05

正确性要点：

1. 幺正演化 `rho(t)=U rho0 U^dagger` 自动保持厄米性。
2. 迹守恒：`Tr(U rho0 U^dagger)=Tr(rho0)=1`。
3. 正半定性守恒：幺正相似变换不改变特征值。
4. 因此物理可行性（合法密度矩阵）在理论上保持。
5. `demo.py` 用数值诊断（迹偏差、厄米误差、最小特征值）对上述性质做计算验证。

## R06

复杂度分析（`n x n` 复矩阵）：

- 单次 `U rho U^dagger`：主要是矩阵乘法，`O(n^3)`
- 单次 `expm`：典型实现约 `O(n^3)`（常数较高）
- ODE 积分法：若时间步数为 `m`，总体约 `O(m n^3)`

对于教学级 MVP（两能级，`n=2`），运行开销可忽略，重点是验证物理不变量与算法一致性。

## R07

标准实现步骤：

1. 构造哈密顿量 `H` 与初态密度矩阵 `rho0`。
2. 校验 `rho0` 是否为合法密度矩阵。
3. 对每个时间点 `t` 计算 `U(t)=exp(-iHt/hbar)`。
4. 得到 `rho(t)=U rho0 U^dagger`。
5. 并行地将冯诺依曼方程写成实向量 ODE 做数值积分。
6. 比较两种路线结果误差。
7. 输出迹守恒、厄米性、正性等诊断量。

## R08

`demo.py` 的 MVP 设计：

- 依赖：`numpy` + `scipy`（`expm`, `solve_ivp`）
- 系统：二维量子比特（`sigma_x`, `sigma_z` 线性组合）
- 演化：
  - 路线 A：矩阵指数的封闭形式
  - 路线 B：微分方程积分的数值形式
- 验证：输出每个时间点的 `trace_dev`、`herm_err`、`min_eig`、`purity`

## R09

`demo.py` 接口约定：

- `commutator(a, b) -> np.ndarray`
- `validate_density_matrix(rho, atol=1e-10) -> None`
- `unitary_evolution(rho0, hamiltonian, t, hbar=1.0) -> np.ndarray`
- `von_neumann_rhs_real(t, y, hamiltonian, hbar=1.0) -> np.ndarray`
- `integrate_von_neumann(rho0, hamiltonian, t_eval, hbar=1.0) -> list[np.ndarray]`
- `density_diagnostics(rho) -> tuple[float, float, float, float]`

## R10

测试策略：

- 结构正确性：`rho0` 先过合法性校验
- 物理不变量：沿时间轴检查
  - `|Tr(rho)-1|` 应接近 0
  - `||rho-rho^dagger||_F` 应接近 0
  - `lambda_min(rho)` 不应显著小于 0
- 方法一致性：`||rho_unitary - rho_ode||_F` 应很小

## R11

边界条件与异常处理：

- `rho` 非方阵或非厄米：抛出 `ValueError`
- `Tr(rho)` 不为 1：抛出 `ValueError`
- `rho` 非正半定（最小特征值 < `-atol`）：抛出 `ValueError`
- ODE 求解失败：抛出 `RuntimeError`

## R12

与相关方程关系：

- 薛定谔方程描述态矢量 `|psi>` 的演化；冯诺依曼方程描述统计算符 `rho` 的演化
- 纯态情形下 `rho = |psi><psi|`，两者完全等价
- Lindblad 方程可看作冯诺依曼方程加入耗散项后的推广

## R13

示例参数选择（`demo.py`）：

- 设 `hbar = 1`（自然单位）
- `H = (omega/2) sigma_x + (delta/2) sigma_z`，其中 `omega=1.8`, `delta=0.6`
- 初态取带相干项的混态，既非完全纯态也非对角态，便于观察非平凡演化
- 采样时间 `t in [0, 6]` 上的 9 个点

## R14

工程实现注意点：

- 直接把复矩阵给 ODE 求解器可能受方法限制，MVP 用“实部+虚部拼接”保证稳健
- 数值误差会引入极小的非厄米成分，诊断时要给出容差
- `expm` 与 ODE 误差阈值应根据步长与容差配置理解，不应要求机器零误差

## R15

最小示例解释：

- 在二维系统中，若 `rho0` 与 `H` 不对易，则布洛赫向量会围绕由 `H` 决定的轴旋转
- `demo.py` 输出可看到纯度 `Tr(rho^2)` 基本保持常数，符合闭合系统幺正演化特征

## R16

可扩展方向：

- 引入时间依赖 `H(t)` 并改为非自治积分
- 加入 Lindblad 耗散项模拟开放系统
- 扩展到多量子比特张量积空间（维度增长后关注稀疏结构）
- 与量子电路仿真框架对接，比较门级与哈密顿量级演化

## R17

本条目交付说明：

- `README.md`：完成 R01-R18，覆盖定义、正确性、复杂度、工程实现和源码流程
- `demo.py`：可直接运行、无需交互输入的最小可行实现
- `meta.json`：保持与任务元信息一致

运行方式：

```bash
uv run python Algorithms/物理-量子力学-0262-冯诺依曼方程_(von_Neumann_Equation)/demo.py
```

或在目录内运行：

```bash
uv run python demo.py
```

## R18

源码级算法流程拆解（对应 `demo.py`，共 8 步）：

1. **输入物理化校验**
   `validate_density_matrix` 检查 `rho0` 的方阵性、厄米性、迹为 1、正半定性，确保初值合法。

2. **构造生成元**
   对每个时间点形成 `A(t) = -i H t / hbar`，这是幺正演化算符的指数生成元。

3. **`scipy.linalg.expm` 的阶数与缩放决策**
   `expm` 内部依据矩阵范数选择 Padé 近似阶数，并决定缩放因子 `s`，将问题化为对 `A/2^s` 求指数。

4. **Padé 有理逼近求局部指数**
   在缩放后矩阵上构造分子/分母多项式并做线性求解，得到 `exp(A/2^s)` 的高精度近似。

5. **平方回代恢复尺度**
   通过重复平方 `s` 次恢复 `U(t)=exp(A)`，即缩放-平方（scaling and squaring）主流程。

6. **幺正共轭推进密度矩阵**
   执行 `rho(t) = U rho0 U^dagger` 得到封闭系统解析路径下的状态。

7. **实向量化 ODE 交叉验证**
   `von_neumann_rhs_real` 将复矩阵导数拆成实/虚向量，交给 `solve_ivp`（RK45）在同一时间网格积分。

8. **一致性与不变量诊断输出**
   比较两条路径的 Frobenius 范数误差，并打印 `trace_dev / herm_err / min_eig / purity`，验证算法与物理约束一致。
