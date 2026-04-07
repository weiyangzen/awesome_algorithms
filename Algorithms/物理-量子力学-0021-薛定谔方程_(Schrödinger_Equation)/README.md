# 薛定谔方程 (Schrödinger Equation)

- UID: `PHYS-0021`
- 学科: `物理`
- 分类: `量子力学`
- 源序号: `21`
- 目标目录: `Algorithms/物理-量子力学-0021-薛定谔方程_(Schrödinger_Equation)`

## R01

时间依赖薛定谔方程描述闭合量子系统态矢量 `psi(t)` 的演化：

`i hbar d psi / dt = H psi`

其中 `H` 是哈密顿量。若 `H` 与时间无关，形式解为：

`psi(t) = exp(-i H t / hbar) psi(0)`

本条目用 1D 有限差分离散 `H`，给出可运行的数值 MVP。

## R02

典型应用场景：

- 量子波包在给定势场中的传播模拟
- 量子比特/多能级系统的相干演化基线计算
- 量子控制与绝热演化方案前的数值验证
- 与密度矩阵方程、路径积分近似的结果对照

## R03

离散化后的数学模型（`demo.py`）：

- 空间网格 `x` 上用中心差分近似二阶导数
- 构造拉普拉斯矩阵 `D2`
- 构造哈密顿量
  `H = -(hbar^2 / 2m) D2 + diag(V(x))`
- 时间演化写成复向量 ODE：
  `d psi / dt = -(i / hbar) H psi`

这样偏微分方程被转换为有限维线性常微分方程系统。

## R04

直观理解：

- `H` 决定系统“相位旋转”的频率结构
- 动能项来自波函数曲率，势能项来自位置能量地形
- 波包在势阱中演化表现为展宽、振荡与相位干涉
- 闭合系统的幺正演化应保持总概率（范数）不变

## R05

正确性要点：

1. 连续理论中，若 `H` 厄米，则演化算符 `U(t)=exp(-iHt/hbar)` 幺正。
2. 幺正性保证 `||psi(t)||_2` 守恒，概率归一性不变。
3. 对时间无关 `H`，能量期望 `E=<psi|H|psi>` 理论上守恒。
4. `demo.py` 用 `norm` 与 `E` 的时间序列验证守恒性质。
5. 用 `solve_ivp` 与 `expm` 两条路线交叉对照，检查数值实现一致性。

## R06

复杂度分析（网格点数 `n`，采样时刻数 `m`）：

- 构造稠密哈密顿量：`O(n^2)` 存储，`O(n^2)` 初始化
- 单次 `expm`：典型 `O(n^3)`
- `m` 个时刻直接做 `expm`：约 `O(m n^3)`
- ODE 路线每步矩阵向量乘约 `O(n^2)`，总成本与步数相关

本 MVP 取 `n=120`，可在普通 CPU 快速完成。

## R07

标准实现步骤：

1. 生成一维均匀网格 `x`。
2. 设定势函数 `V(x)`（示例为谐振子势）。
3. 组装有限差分哈密顿量并验证其厄米性。
4. 构造并归一化初态波包 `psi0`。
5. 路线 A：对每个 `t` 计算 `exp(-iHt/hbar) psi0`。
6. 路线 B：将复 ODE 拆成实系统后交给 `solve_ivp`。
7. 输出 `norm`、`<x>`、`<H>`、两路线 `L2` 误差。

## R08

`demo.py` 的 MVP 设计：

- 依赖：`numpy` + `scipy`（`scipy.linalg.expm`, `scipy.integrate.solve_ivp`）
- 模型：1D 谐振子势 `V(x)=0.5*m*omega^2*x^2`
- 初态：带动量的高斯波包（非平凡演化）
- 结果：输出 9 个时间点的守恒量与方法差异
- 运行方式：`uv run python demo.py`（无需交互输入）

## R09

`demo.py` 接口约定：

- `build_hamiltonian_1d(x, mass, hbar, potential_fn) -> np.ndarray`
- `validate_hamiltonian(hamiltonian, atol=1e-10) -> None`
- `normalize_state(psi, dx) -> np.ndarray`
- `unitary_evolution(psi0, hamiltonian, t, hbar=1.0) -> np.ndarray`
- `schrodinger_rhs_real(t, y, hamiltonian, hbar=1.0) -> np.ndarray`
- `integrate_schrodinger(psi0, hamiltonian, t_eval, hbar=1.0) -> list[np.ndarray]`
- `wave_diagnostics(psi, hamiltonian, x, dx) -> tuple[float, float, float]`

## R10

测试策略：

- 结构测试：`x` 是否均匀；`H` 是否方阵且厄米
- 初值测试：`psi0` 归一化后范数应为 1
- 物理量测试：`norm` 与 `E` 在时间轴上应基本稳定
- 一致性测试：`expm` 与 `solve_ivp` 的 `L2` 误差应保持小量级

## R11

边界条件与异常处理：

- 网格不是一维/点数过少：抛 `ValueError`
- 网格不均匀：抛 `ValueError`
- 哈密顿量非厄米：抛 `ValueError`
- 初态范数非正：抛 `ValueError`
- ODE 求解失败：抛 `RuntimeError`

数值边界方面，有限区间 `[-8, 8]` 近似无限域；若波包明显触边，需扩展区间或细化网格。

## R12

与相关方程关系：

- 薛定谔方程是态矢量演化方程
- 冯诺依曼方程是密度矩阵对应形式
- 海森堡图像把时间依赖转移到算符上
- 对时间无关 `H`，三种表述在可观测量预测上等价

## R13

示例参数选择（`demo.py`）：

- 自然单位：`hbar=1`, `m=1`
- 谐振子频率：`omega=1`
- 网格：`x in [-8, 8]`，`n=120`
- 初态：`x0=-1.5`, `sigma=0.8`, `k0=1.2`
- 时间采样：`t in [0, 2]` 共 9 点

这组参数能在小算力下展示稳定且非平凡的波包动力学。

## R14

工程实现注意点：

- `solve_ivp` 主接口针对实向量，MVP 采用“实部+虚部拼接”
- 概率积分使用 `sum(|psi|^2) * dx`，不是简单向量范数
- 期望值计算需用共轭内积 `vdot` 并考虑离散积分权重 `dx`
- 误差评估使用离散 `L2` 范数，便于跨方法比较

## R15

最小示例解释：

- 初态是偏移高斯波包，在谐振子势中会来回振荡
- 输出中的 `<x>` 会随时间变化，反映波包中心运动
- `norm(unitary)` 接近 1，说明幺正传播保持概率
- `L2(unitary, ode)` 很小，说明两条数值路线相互印证

## R16

可扩展方向：

- 改为时间依赖势 `V(x, t)`，研究受驱动系统
- 用 Crank-Nicolson 替代 `expm`，提升大规模时步效率
- 引入吸收边界（CAP）模拟散射与开放边界问题
- 扩展到二维/三维或多粒子张量积空间

## R17

本条目交付说明：

- `README.md`：完成 R01-R18 的算法说明与工程细节
- `demo.py`：提供可直接运行的最小可行实现
- `meta.json`：保持与任务元数据一致（UID、学科、分类、源序号、目录）

执行命令：

```bash
uv run python Algorithms/物理-量子力学-0021-薛定谔方程_(Schrödinger_Equation)/demo.py
```

或在目录内：

```bash
uv run python demo.py
```

## R18

源码级算法流程拆解（对应 `demo.py`，共 9 步）：

1. **离散化空间与算子**
   在 `build_hamiltonian_1d` 中用三对角中心差分构造 `D2`，再组装 `H = T + V`。

2. **验证生成元合法性**
   `validate_hamiltonian` 检查 `H` 为方阵且厄米，保证理论上可产生幺正演化。

3. **初始化物理初态**
   构造高斯波包并用 `normalize_state` 强制 `sum(|psi|^2)dx=1`。

4. **路径 A：矩阵指数传播准备**
   对每个时刻形成 `A(t) = -iHt/hbar`，交给 `scipy.linalg.expm`。

5. **`expm` 内核（缩放-平方 + Padé）**
   `expm` 先按矩阵范数选择缩放因子与 Padé 阶数，在缩放矩阵上做有理逼近，再重复平方恢复 `exp(A)`。

6. **路径 A：得到幺正解**
   执行 `psi(t)=exp(A)psi0`，得到每个采样时刻的波函数。

7. **路径 B：ODE 数值积分**
   `schrodinger_rhs_real` 将复 ODE 拆成实系统；`integrate_schrodinger` 通过 `solve_ivp(RK45)` 在同一 `t_eval` 网格积分。

8. **计算诊断量**
   `wave_diagnostics` 计算 `norm`、`<x>`、`<H>`；`l2_error` 计算两路径的离散 `L2` 差。

9. **输出一致性结论**
   打印逐时刻守恒量、`max L2(unitary, ode)` 与 `max |norm(ode)-1|`，形成可审查的最小数值证据链。
