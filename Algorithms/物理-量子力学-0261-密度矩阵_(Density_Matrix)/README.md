# 密度矩阵 (Density Matrix)

- UID: `PHYS-0258`
- 学科: `物理`
- 分类: `量子力学`
- 源序号: `261`
- 目标目录: `Algorithms/物理-量子力学-0261-密度矩阵_(Density_Matrix)`

## R01

密度矩阵是描述量子态的统一对象，既可表示纯态，也可表示混态。定义为：

`rho = sum_i p_i |psi_i><psi_i|`

其中 `p_i >= 0` 且 `sum_i p_i = 1`。  
本条目给出一个单量子比特最小可运行 MVP，展示：

- 由经典概率混合构造 `rho`；
- 验证 `rho` 的物理合法性（厄米、迹为 1、半正定）；
- 闭系统幺正演化 `rho(t)=U rho U†`；
- 开系统相位翻转噪声（Kraus 形式）导致纯度下降、熵上升。

## R02

问题范围（MVP）：

1. 维度固定为 2（单量子比特）。
2. 初态由两态集合（`|0>` 与 `|+>`）的概率混合生成，不直接硬编码矩阵。
3. 哈密顿量取 `H = (omega/2) sigma_x + (delta/2) sigma_z`，便于看到非平凡旋转。
4. 对多个时间点计算：
   - 幺正演化后的 `rho_u(t)`；
   - 经过相位翻转信道后的 `rho_n(t)`。
5. 输出并检查纯度、冯诺依曼熵、最小本征值、Bloch 向量范数。

## R03

数学模型：

1. **密度矩阵构造**  
   `rho = sum_i p_i |psi_i><psi_i|`。

2. **物理约束**  
   - `rho = rho†`（厄米）  
   - `Tr(rho)=1`（归一化）  
   - `rho >= 0`（半正定，本征值非负）

3. **幺正演化**  
   `U(t)=exp(-iHt/hbar)`，  
   `rho_u(t)=U(t) rho(0) U†(t)`。

4. **相位翻转噪声（phase-flip）**  
   `E(rho)=(1-p)rho + p Z rho Z`，等价 Kraus 形式：
   `K0=sqrt(1-p)I, K1=sqrt(p)Z, E(rho)=sum_k Kk rho Kk†`。

5. **诊断量**  
   - 纯度：`P = Tr(rho^2)`  
   - 熵：`S = -Tr(rho log rho)`（代码里以 `log2` 输出比特）  
   - 可观测量期望：`<A> = Tr(rho A)`。

## R04

物理直觉：

- 纯态对应 `P=1`，混态满足 `P<1`。
- 幺正演化只做“坐标旋转”，不改变本征值，因此纯度和熵保持常数。
- 噪声信道会衰减量子相干项，通常使纯度下降、熵上升。
- 对单比特，Bloch 球半径 `||r||` 与纯度一一相关：半径越小，态越混合。

## R05

正确性要点：

1. `density_from_ensemble` 先归一化概率与态矢，避免输入误差直接污染 `rho`。
2. `validate_density_matrix` 做三重检查：方阵 + 厄米 + `Tr=1`，再用最小本征值判定半正定。
3. `rho_u(t)=U rho U†` 结构上保持迹与半正定，代码中对每个时间点再次验证。
4. 相位翻转信道采用 Kraus 形式，天然保持完全正和迹保持（CPTP）。
5. `run_checks` 断言：
   - 幺正演化纯度恒定；
   - 噪声态纯度不超过幺正态；
   - 幺正演化熵恒定；
   - 全时刻最小本征值非负（容忍数值误差）。

## R06

复杂度分析（单比特 `d=2`）：

- 本征值分解（`eigvalsh`）复杂度 `O(d^3)`，这里是常数级。
- 矩阵指数 `expm` 复杂度 `O(d^3)`，同样是常数级。
- `T` 个时间点总体复杂度约 `O(T*d^3)`，即线性随时间采样点增长。
- 空间复杂度 `O(d^2 + T)`（矩阵和结果表）。

在默认 `T=8` 下，`uv run python demo.py` 可快速完成。

## R07

标准实现流程：

1. 构造 `|0>、|1>、|+>` 三个态矢并归一化。
2. 根据概率分布构造初始密度矩阵 `rho0`。
3. 构造哈密顿量 `H`。
4. 在时间网格上计算 `rho_u(t)=U rho0 U†`。
5. 对 `rho_u(t)` 施加相位翻转信道得到 `rho_n(t)`。
6. 对两类状态分别计算纯度、熵、最小本征值、Bloch 范数。
7. 汇总为 `pandas.DataFrame` 并打印。
8. 执行自动断言，输出 `All checks passed.` 作为验收信号。

## R08

`demo.py` 的 MVP 设计：

- 依赖最小化：`numpy`（线性代数）、`scipy.linalg.expm`（矩阵指数）、`pandas`（结果表展示）。
- 不依赖高层量子框架（如 qiskit/cirq），避免黑盒。
- 不需要命令行输入，脚本固定参数可重复运行。
- 输出同时包含：
  - 初态矩阵；
  - 哈密顿量；
  - 时间序列诊断表；
  - 自动检查结果。

## R09

核心函数接口：

- `normalize_state(state) -> np.ndarray`
- `density_from_ensemble(probabilities, states) -> np.ndarray`
- `validate_density_matrix(rho, atol=1e-10) -> None`
- `purity(rho) -> float`
- `von_neumann_entropy(rho, base=2.0) -> float`
- `expectation(rho, observable) -> float`
- `unitary_from_hamiltonian(hamiltonian, time, hbar=1.0) -> np.ndarray`
- `evolve_unitary(rho, hamiltonian, time, hbar=1.0) -> np.ndarray`
- `phase_flip_channel(rho, probability) -> np.ndarray`
- `bloch_vector(rho) -> tuple[float, float, float]`
- `run_density_matrix_mvp() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]`
- `run_checks(rho0, report) -> None`

## R10

测试策略：

1. **结构测试**：`rho` 必须是方阵且维度一致。
2. **物理性测试**：厄米性、`Tr=1`、最小本征值非负。
3. **动力学测试**：幺正演化下纯度、熵在全时间轴不变。
4. **噪声测试**：噪声后纯度不超过噪声前纯度。
5. **稳定性测试**：表格中全部关键指标有限 (`finite`) 且无异常中断。

## R11

边界条件与异常处理：

- 概率为空、含负值或总和非正：抛 `ValueError`。
- 态矢范数为 0 或含非有限数：抛 `ValueError`。
- 维度不一致（不同长度态矢混合）：抛 `ValueError`。
- `rho` 非方阵、非厄米、迹偏离 1、或非半正定：抛 `ValueError`。
- `hbar <= 0` 或噪声概率不在 `[0,1]`：抛 `ValueError`。

## R12

运行方式：

```bash
cd Algorithms/物理-量子力学-0261-密度矩阵_(Density_Matrix)
uv run python demo.py
```

脚本无交互输入，直接打印完整结果。

## R13

输出字段说明：

- `purity_unitary`：幺正态纯度（应与初值一致）。
- `entropy_unitary_bits`：幺正态熵（应保持常数）。
- `min_eig_unitary`：幺正态最小本征值（应非负）。
- `purity_noisy`：噪声态纯度（通常低于幺正态）。
- `entropy_noisy_bits`：噪声态熵（通常高于幺正态）。
- `min_eig_noisy`：噪声态最小本征值（应非负）。
- `bloch_norm_unitary` / `bloch_norm_noisy`：Bloch 半径对比，噪声后通常更小。

## R14

最小验收标准（默认参数应满足）：

1. `uv run python demo.py` 可直接运行完成。
2. 终端出现 `All checks passed.`。
3. `purity_unitary` 在全部时间点上恒定（数值容差内）。
4. `purity_noisy <= purity_unitary` 对全部时间点成立。
5. 所有 `min_eig_*` 不小于 `-1e-9`（数值误差容忍）。

## R15

关键参数与调参建议：

- `omega, delta`：决定 Bloch 球上的旋转轴与角速度；可修改以观察不同动力学。
- `times`：采样点越多，时间曲线越平滑，但输出更长。
- `p_noise`：噪声强度，接近 `0` 时接近闭系统，接近 `0.5` 时退相干明显。
- 初始混合比例（`0.55/0.45`）可改成更偏纯态或更偏混态，用于比较纯度基线。

## R16

与相关算法的关系：

- 与“薛定谔方程”相比：密度矩阵天然支持混态与噪声信道，不仅限于纯态向量。
- 与“量子主方程/Lindblad”相比：本条目实现了离散噪声信道版本，复杂度更低、实现更短。
- 与“量子过程层析”相比：本条目只前向模拟，不做信道反演估计。

## R17

可扩展方向：

1. 从单比特推广到双比特，并加入部分迹计算纠缠子系统态。
2. 增加其他信道：振幅阻尼、去极化、广义相位阻尼。
3. 用 Lindblad 微分方程实现连续时间开系统演化。
4. 增加观测算符集合与可视化（如 Bloch 轨迹图）。
5. 把断言扩展为可复用单元测试（`pytest`）。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `run_density_matrix_mvp` 构造基础态矢，并由 `density_from_ensemble` 生成 `rho0`。
2. `density_from_ensemble` 内部逐项执行 `pi * |psi_i><psi_i|` 并累加，得到初始混态。
3. `validate_density_matrix` 检查厄米性、迹和半正定；半正定通过 `eigvalsh` 最小本征值判定。
4. 每个时间点调用 `evolve_unitary`，其内部先 `unitary_from_hamiltonian` 计算 `U=expm(-iHt/hbar)`，再做 `U rho U†`。
5. 这里 `scipy.linalg.expm` 并非单行黑盒魔法，核心流程是：范数评估 -> 缩放矩阵 -> Pade 有理近似 -> 平方放大（scaling and squaring）得到最终 `U`。
6. 调用 `phase_flip_channel` 用 Kraus 算子 `K0,K1` 做 `sum_k Kk rho Kk†`，得到噪声态。
7. 对幺正态和噪声态分别计算纯度、冯诺依曼熵、最小本征值、Bloch 范数，写入 `DataFrame`。
8. `run_checks` 做物理一致性断言；`main` 打印矩阵和时间序列表，最后输出 `All checks passed.`。
