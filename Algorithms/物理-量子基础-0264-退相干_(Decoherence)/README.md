# 退相干 (Decoherence)

- UID: `PHYS-0261`
- 学科: `物理`
- 分类: `量子基础`
- 源序号: `264`
- 目标目录: `Algorithms/物理-量子基础-0264-退相干_(Decoherence)`

## R01

退相干描述量子系统因与环境耦合而失去相位相干性的过程。它通常先破坏叠加态的相位信息（密度矩阵非对角项衰减），再在更长时间尺度上伴随能量弛豫。

本条目实现一个最小可运行 MVP：
- 以单量子比特纯退相干（pure dephasing）为对象；
- 手写 Kraus 通道作为核心算法，而非直接黑盒调用量子库；
- 用 `scipy/pandas/sklearn/torch` 做参考验证与误差度量。

## R02

本目录解决的具体问题：

1. 构造初态 `|+> = (|0>+|1>)/sqrt(2)` 的密度矩阵；
2. 实现纯退相干 CPTP 通道，满足 `rho_01(t)=rho_01(0)e^{-gamma t}`；
3. 给出 Lindblad 向量化参考解，用于交叉校验；
4. 跟踪相干度、纯度、冯诺依曼熵与 Bloch 分量随时间的变化；
5. 输出可审计表格与自动验收布尔结果。

## R03

数学模型（单比特纯退相干）：

- 主方程可写为
  `d rho / dt = (gamma/2) * (sigma_z rho sigma_z - rho)`；
- 等价地，非对角项指数衰减：
  `rho_01(t)=rho_01(0)e^{-gamma t}`，对角项不变；
- Kraus 形式：
  `rho(t)=K0 rho(0) K0^dagger + K1 rho(0) K1^dagger`，
  `K0=sqrt(1-p) I, K1=sqrt(p) sigma_z, p=(1-e^{-gamma t})/2`。

## R04

为何使用密度矩阵而不是态矢量：

- 退相干本质上是开放系统动力学，纯态矢量不足以表达与环境纠缠后的统计态；
- 密度矩阵可同时表达纯态与混态，天然支持 CPTP 通道；
- 通过 `Tr(rho^2)` 与熵可直接量化“从纯到混”的过程。

## R05

MVP 的核心观测量：

- `coherence_l1 = |rho01| + |rho10|`：量化相干性；
- `purity = Tr(rho^2)`：纯态为 1，混态更小；
- `entropy_bits = -Tr(rho log2 rho)`：纯态接近 0，比特极限混态接近 1；
- `bloch_x, bloch_y, bloch_z`：Bloch 球分量，纯退相干下 `z` 分量保持不变。

## R06

高层流程：

1. 构造 `rho0=|+><+|`；
2. 在时间网格上逐点施加手写 Kraus 退相干通道；
3. 同时用 `scipy.linalg.expm` 对 Lindblad 生成元指数化得到参考解；
4. 计算两者矩阵差异；
5. 统计相干度/纯度/熵/Bloch 分量并写入 `pandas.DataFrame`；
6. 用 `sklearn` 计算相干曲线与理论指数衰减的 MAE；
7. 用 `torch` 计算同一指数曲线并与 NumPy 对齐检查；
8. 汇总布尔验收项并打印结论。

## R07

核心数据结构：

- `rho: np.ndarray(shape=(2,2), complex)`：单比特密度矩阵；
- `t_grid: np.ndarray(shape=(N,))`：时间采样；
- `df: pandas.DataFrame`：每个时间点的观测量表格；
- `checks: dict[str, float|bool]`：误差与验收标志。

## R08

正确性校验设计：

- 物理合法性：每步检查 `rho` 的厄米性、迹为 1、半正定；
- 数值一致性：手写 Kraus 与 Lindblad 指数化参考解做 `||rho-rho_ref||` 比较；
- 理论一致性：`coherence_l1` 应满足 `e^{-gamma t}` 规律；
- 跨框架一致性：PyTorch 与 NumPy 对同一解析曲线结果一致。

## R09

复杂度：

- 单时间点演化是固定 2x2 线性代数，时间复杂度 `O(1)`；
- `N` 个时间点总体 `O(N)`；
- 表格存储空间复杂度 `O(N)`；
- 在本 MVP 中，成本主要来自循环次数而非矩阵规模。

## R10

边界与异常处理：

- `gamma < 0`、`t < 0`、`n_steps < 2` 直接报错；
- 非法态（非 2x2、非厄米、迹非 1、非半正定）拒绝进入通道计算；
- 对输出做对称化与迹归一化，降低浮点误差积累；
- 熵计算对特征值做 `eps` 裁剪，避免 `log(0)` 数值问题。

## R11

MVP 取舍：

- 仅实现单比特纯退相干，不扩展到多比特和一般噪声通道；
- 不引入外部量子计算框架（如 Qiskit/QuTiP），保持最小可审计实现；
- SciPy 只作为参考演化验证，不替代核心算法；
- 目标是“解释清楚 + 可运行 + 可验证”，不是完整量子噪声库。

## R12

`demo.py` 函数分工：

- `ket_to_density`：态矢量转密度矩阵；
- `is_valid_density_matrix`：物理合法性检查；
- `pure_dephasing_channel`：手写 Kraus 退相干通道；
- `lindblad_reference`：Lindblad 向量化指数参考解；
- `coherence_l1/purity/von_neumann_entropy_bits/bloch_components`：观测量计算；
- `run_experiment`：组织时间扫描、构建表格与验收指标；
- `main`：固定参数运行并打印摘要。

## R13

运行方式：

```bash
cd Algorithms/物理-量子基础-0264-退相干_(Decoherence)
uv run python demo.py
```

脚本无交互输入，直接输出轨迹样例与检查结果。

## R14

输出字段说明：

- `t`：演化时间；
- `coherence_l1`：l1 相干度（应随时间衰减）；
- `purity`：纯度（从 1 下降到约 0.5）；
- `entropy_bits`：冯诺依曼熵（从 0 上升到约 1）；
- `bloch_x/bloch_y/bloch_z`：Bloch 分量（`x` 衰减，`z` 保持）；
- `trace_real`：迹的实部（应约为 1）；
- `min_eig`：最小特征值（应非负，允许极小数值误差）。

## R15

最小验收标准：

- `all_checks_pass=True`；
- `max_reference_error < 1e-12`；
- `coherence_mae < 1e-12`；
- `trace_stays_one=True` 且 `positive_semidefinite=True`；
- `bloch_z_constant=True`（对应纯退相干不改变布居差）。

## R16

物理解释（本模型下）：

- 初态 `|+>` 在 Bloch 球赤道，具有最大相干性；
- 退相干让相位信息流向环境，导致 `x/y` 分量指数衰减；
- 由于无能量交换，`z` 分量与对角布居保持不变；
- 最终趋向经典混合态 `diag(1/2,1/2)`。

## R17

可扩展方向：

1. 加入振幅阻尼（T1）并与纯退相干（T2）联合建模；
2. 扩展到多比特并观察纠缠退相干；
3. 在不同噪声谱密度下比较指数/非指数衰减；
4. 引入实验测量噪声，做参数反演估计 `gamma`；
5. 将通道离散化为量子线路噪声门用于电路级仿真。

## R18

`demo.py` 源码级算法流程（9 步）：

1. `main` 设定 `gamma, t_max, n_steps` 并调用 `run_experiment`。
2. `run_experiment` 构造 `|+>`，由 `ket_to_density` 得到初态 `rho0`。
3. 在每个时间点 `t`，调用 `pure_dephasing_channel(rho0, gamma, t)`：
   先算 `lambda=e^{-gamma t}` 与 `p=(1-lambda)/2`，再组装 `K0, K1` 并执行 Kraus 求和。
4. 同一时间点调用 `lindblad_reference`：构造向量化生成元 `diag(0,-gamma,-gamma,0)`，用 `expm` 演化得到 `rho_ref`。
5. 计算 `||rho_t-rho_ref||` 作为核心参考误差，验证手写通道与主方程解一致。
6. 计算并记录 `coherence_l1/purity/entropy_bits/bloch_x,bloch_y,bloch_z/trace_real/min_eig` 到 `DataFrame`。
7. 循环结束后，用 `sklearn.mean_absolute_error` 比较观测相干曲线与理论 `e^{-gamma t}`。
8. 用 `torch.exp` 计算同一指数衰减并与 NumPy 结果对齐，得到 `torch_alignment_error`。
9. 汇总所有误差阈值与物理约束布尔项，生成 `all_checks_pass` 作为最终验收信号并打印。
