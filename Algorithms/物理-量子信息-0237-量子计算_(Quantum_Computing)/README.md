# 量子计算 (Quantum Computing)

- UID: `PHYS-0236`
- 学科: `物理`
- 分类: `量子信息`
- 源序号: `237`
- 目标目录: `Algorithms/物理-量子信息-0237-量子计算_(Quantum_Computing)`

## R01

量子计算是以量子比特（qubit）为信息载体、以幺正门操作为基本计算原语的一类计算模型。与经典比特不同，`n` 个 qubit 的纯态可表示为长度 `2^n` 的复向量，允许叠加与纠缠。

本条目的 MVP 聚焦“门模型（gate model）”的最小可运行实现：

- 用状态向量模拟 2-qubit 电路演化；
- 演示 Bell 态制备（纠缠）；
- 演示 2-qubit Grover 搜索（振幅放大）。

## R02

本问题可抽象为：给定初态 `|00>` 与一组量子门，计算末态及测量分布。

目标输出包括：

- Bell 实验的计算基概率与采样计数；
- Grover 一次迭代后的目标态概率；
- 归一化诊断（numpy 与 torch 双重检查）。

这对应量子计算中的“电路仿真 + 采样验证”基础工作流。

## R03

数学表示（纯态）采用列向量：

- 态：`|psi> in C^(2^n)`，约束 `||psi||_2 = 1`
- 门：幺正矩阵 `U`，满足 `U^dagger U = I`
- 演化：`|psi'> = U |psi>`
- 测量概率：`p_i = |psi_i|^2`

本 MVP 具体使用：

- `H` 门（Hadamard）制造叠加；
- `CNOT` 制造纠缠；
- `Rz(theta)=exp(-i theta/2 Z)`（由 `scipy.linalg.expm` 生成）；
- Grover 的 Oracle 与 Diffusion 算子。

## R04

核心直觉：

- 量子门本质是对复振幅向量的“旋转/反射”；
- Bell 电路把 `|00>` 变成纠缠态（测量呈强相关）；
- Grover 的一次迭代把目标基态振幅显著放大，在 `N=4` 情形可达到理论成功率 1。

因此，哪怕只做 2 qubit，仍可展示量子计算区别于经典穷举的关键机制。

## R05

正确性要点：

1. 初态由 `zero_state` 构造，天然归一化。
2. 所有门操作都按 `state <- U @ state` 线性更新。
3. 幺正门保持范数，`validate_state_vector` 做数值校验。
4. 概率由 `abs(state)^2` 计算并归一化，保证和为 1。
5. Bell 实验应满足 `P(00)+P(11)≈1`。
6. 2-qubit Grover 一次迭代理论上应使目标态概率接近 1。

## R06

复杂度（状态向量维度 `d=2^n`）：

- 构造全矩阵单门（`kron` 链）约 `O(d^2)` 存储、`O(d^2)` 生成；
- 一次门应用 `U @ state` 约 `O(d^2)`；
- `m` 个门总计约 `O(m d^2)`；
- 采样 `shots=S` 为 `O(S)`。

本条目固定 `n=2`，所以时间与空间开销都很小，重点是算法流程透明。

## R07

标准实现流程：

1. 定义基础门矩阵（`I, H, X, Z`）。
2. 构造 `|00>` 初态。
3. 通过 Kronecker 积扩展单比特门到全系统。
4. 用位串重映射构造 `CNOT` 置换矩阵。
5. 运行 Bell 电路并输出分布。
6. 构造 Grover Oracle 与 Diffuser。
7. 执行一次 Grover 迭代并输出目标态概率。
8. 用 torch 对末态范数做独立检查。

## R08

`demo.py` 的 MVP 设计决策：

- 依赖栈：`numpy + scipy + pandas + torch`。
- 不依赖量子专用框架（如 Qiskit），避免黑箱化。
- 所有关键算子（CNOT、Oracle、Diffuser）在源码中显式构造。
- 使用 `pandas.DataFrame` 打印可读概率表，便于验证脚本输出。

## R09

主要函数接口：

- `zero_state(n_qubits) -> np.ndarray`
- `validate_state_vector(state, atol=1e-10) -> None`
- `single_qubit_operator(gate, qubit, n_qubits) -> np.ndarray`
- `cnot_operator(n_qubits, control, target) -> np.ndarray`
- `rz_gate(theta) -> np.ndarray`
- `bell_state_experiment(...) -> (state, probs, table)`
- `grover_experiment(...) -> (state, probs, table, target_index)`
- `torch_norm_check(state) -> float`

## R10

测试与验收策略：

- 结构性检查：状态长度必须是 `2^n`，范数必须接近 1。
- Bell 检查：`P(00 or 11)` 应接近 1。
- Grover 检查：目标态概率应接近 1。
- 采样可复现：固定随机种子，避免结果抖动。
- 运行结束条件：若关键阈值未达标则抛出 `RuntimeError`。

## R11

边界条件与异常处理：

- `n_qubits <= 0`、门矩阵尺寸错误、qubit 越界时抛 `ValueError`。
- `CNOT` 的 `control == target` 禁止。
- `shots <= 0` 禁止。
- 任何态向量归一化失败会被 `validate_state_vector` 捕获。

## R12

与相关主题关系：

- 与“量子电路模型”直接对应：门级演化 + 末端测量。
- 与“量子纠缠”对应：Bell 态是最小纠缠示例。
- 与“量子搜索”对应：Grover 展示振幅放大机制。
- 与“哈密顿量模拟”相关：`Rz` 通过矩阵指数体现连续演化到离散门的连接。

## R13

`demo.py` 默认参数：

- Bell 采样：`shots=2000, seed=123`
- Grover 采样：`target_bits="10", shots=2000, seed=321`
- Bell 中加入 `Rz(pi/3)` 相位旋转（不破坏相关性）

这些参数在演示稳定性与输出可读性之间做了折中。

## R14

工程实现注意点：

- 本 MVP 使用“全矩阵法”，实现直观但不适合大规模 qubit。
- qubit 编号采用位串从左到右（`q0` 为最高位）约定，必须全程一致。
- 测量采样结果有统计波动，验收应优先看理论概率而非单次计数。
- 若要扩展到更多 qubit，应改为稀疏/张量网络或门分解执行，避免 `2^n` 爆炸。

## R15

最小示例解释：

- Bell 部分最终概率集中在 `|00>, |11>`，体现纠缠相关。
- Grover 部分对 2-qubit 的单目标搜索只需 1 次迭代即可把目标态概率推到 1（理想数学下）。
- 这两个实验分别覆盖了量子计算中“表示能力”和“算法增益”的最基本特征。

## R16

可扩展方向：

- 增加噪声信道（去相干、振幅阻尼）做 NISQ 风格模拟；
- 加入参数化量子电路与经典优化器形成 VQE/QAOA 雏形；
- 扩展到多轮 Grover 与多目标标记；
- 把全矩阵门替换为按位运算的就地更新以提升可扩展性。

## R17

交付与运行说明：

- `README.md`：完成 R01-R18，覆盖定义、建模、复杂度、实现和验证。
- `demo.py`：非交互式可运行 MVP。
- `meta.json`：与任务元数据保持一致。

运行方式（在仓库根目录）：

```bash
uv run python Algorithms/物理-量子信息-0237-量子计算_(Quantum_Computing)/demo.py
```

或进入目录后运行：

```bash
uv run python demo.py
```

## R18

源码级算法流程拆解（对应 `demo.py`，8 步）：

1. **初始化量子对象**
   定义 `I/H/X/Z` 基础门，并通过 `zero_state(2)` 生成 `|00>`。

2. **构造可复用门扩展器**
   `single_qubit_operator` 通过 Kronecker 积把 2x2 门扩展到 `2^n x 2^n`。

3. **显式构造两比特纠缠门**
   `cnot_operator` 逐基态位串重映射生成置换矩阵，避免黑箱调用。

4. **用 `scipy.linalg.expm` 构建相位旋转门**
   `rz_gate(theta)` 对生成元 `-i theta/2 Z` 做矩阵指数，得到 `Rz`。

5. **Bell 电路演化并测量**
   依次应用 `H(q0) -> Rz(q0) -> CNOT(q0->q1)`，再计算 `|amp|^2` 与采样计数表。

6. **Grover 算子显式展开**
   从均匀叠加态出发，构造 `Oracle`（目标态相位翻转）与 `Diffuser`（关于平均值反射）。

7. **执行一次 Grover 迭代并统计**
   应用 `Oracle` 后应用 `Diffuser`，输出目标态概率与采样分布。

8. **跨框架一致性校验与终止条件**
   用 `torch_norm_check` 复核态向量归一化；若 Bell/Grover 指标低于阈值则显式报错，否则打印 `MVP checks passed.`。
