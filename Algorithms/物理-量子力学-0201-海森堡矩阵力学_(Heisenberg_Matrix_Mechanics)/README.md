# 海森堡矩阵力学 (Heisenberg Matrix Mechanics)

- UID: `PHYS-0200`
- 学科: `物理`
- 分类: `量子力学`
- 源序号: `201`
- 目标目录: `Algorithms/物理-量子力学-0201-海森堡矩阵力学_(Heisenberg_Matrix_Mechanics)`

## R01

海森堡矩阵力学把“可观测量”写成矩阵（算符），把动力学写成矩阵方程，而不是先从波函数偏微分方程出发。本条目的 MVP 用最经典的一维谐振子做演示，在截断 Fock 基（有限维 Hilbert 空间）里构造：

- 升降算符 `a, a†`
- 位置与动量算符 `X, P`
- 哈密顿量 `H`

然后直接验证矩阵力学的三件核心事实：

1. 对易关系结构；
2. 海森堡演化 `O(t)=U†OU`；
3. 谐振子能级与期望值轨迹。

## R02

MVP 的具体任务定义：

1. 在维度 `dim=24` 的截断基下构造谐振子矩阵算符，参数 `m=1.0, ω=1.7, ħ=1.0`；
2. 对时间网格 `t ∈ [0, 2π/ω]` 采样，比较：
   - 数值海森堡演化得到的 `X(t), P(t)`；
   - 谐振子解析闭式解 `X(t), P(t)`；
3. 用 `pandas` 输出误差表；
4. 用 `scikit-learn` 对能级 `E_n` 与量子数 `n` 做线性拟合，验证 `E_n = ħω(n+1/2)`；
5. 用 `PyTorch` 复算对易关系误差，做跨实现一致性核对。

## R03

为什么选这个设定：

- 海森堡矩阵力学最适合用离散能级系统演示，谐振子是标准样例；
- `a, a†` 在数值上构造简单，但能完整连接 `X,P,H` 与动力学；
- 可同时展示“形式正确性”（对易关系）与“动力学正确性”（`X(t),P(t)`）；
- 截断基会引入边界项，这能真实体现数值矩阵力学的工程约束，而不是理想化无限维推导。

## R04

本 MVP 用到的关键关系：

1. 升降算符矩阵元（数态基）：
`a_{n-1,n} = sqrt(n)`，`a† = a^†`。

2. 位置与动量：
`X = sqrt(ħ/(2mω)) (a + a†)`  
`P = i sqrt(mħω/2) (a† - a)`。

3. 哈密顿量：
`H = ħω (a†a + 1/2 I)`。

4. 海森堡演化：
`O(t) = U†(t) O U(t)`，`U(t)=exp(-iHt/ħ)`。

5. 谐振子闭式解：
`X(t)=X cos(ωt) + P/(mω) sin(ωt)`  
`P(t)=P cos(ωt) - mωX sin(ωt)`。

6. 截断基中的对易关系修正：
`[X,P] = iħ(I - N|N-1><N-1|)`（`N=dim`），用于解释边界误差来源。

## R05

算法主流程：

1. 校验输入参数（维度与物理参数正值）。
2. 构建 `a, a†`，再构建 `X, P, H, I`。
3. 计算 ` [X,P] `，同时对照
   - 理想无限维目标 `iħI`
   - 截断理论目标 `iħ(I-N|N-1><N-1|)`。
4. 用 `eigvalsh(H)` 求离散能级并线性回归 `E_n~n`。
5. 对多个时间点，使用 `scipy.linalg.expm` 计算 `U(t)`，做 `U†OU` 演化得到 `X_num(t), P_num(t)`。
6. 计算解析 `X_exact(t), P_exact(t)`，输出 Frobenius 范数误差。
7. 固定初态（截断相干态 `|α>`），比较 `<X>(t),<P>(t)` 与经典轨迹（Ehrenfest 检查）。
8. 打印可验收结果表和有限性检查标志。

## R06

正确性保障点：

- 结构层：`[X,P]` 同时做 NumPy 与 PyTorch 计算，防止单实现错误；
- 截断解释层：既报告到 `iħI` 的偏差，也报告到截断模型的偏差，区分“算法错误”和“基截断必然误差”；
- 动力学层：`U†OU` 与解析闭式解逐时刻比较；
- 统计层：能级线性回归斜率/截距与理论值直接对照；
- 轨迹层：期望值满足经典谐振子方程（Ehrenfest 关系）。

## R07

复杂度分析（`D=dim`, `T=时间采样点数`）：

- 构造算符：`O(D^2)`；
- 每个时间点做一次 `expm` 与矩阵乘法，主成本约 `O(D^3)`；
- 总演化成本约 `O(T*D^3)`；
- 能级求解 `eigvalsh` 约 `O(D^3)`。

在默认参数 `D=24, T=9` 下，运行时间很短，适合作为最小可复现示例。

## R08

边界与异常处理：

- `dim < 6` 直接报错，避免过小基导致演示失真；
- `m, ω, ħ <= 0` 直接报错；
- 期望值输出前统一做有限性检查；
- 对易关系不强行要求接近 `iħI`（因为截断必然失真），而是额外对照截断理论式。

## R09

MVP 取舍说明：

- 只做单粒子一维谐振子，不覆盖非线性势、时变哈密顿量或多体系统；
- 不求解薛定谔方程波函数演化，专注海森堡图像中的算符演化；
- 用有限维截断替代无限维 Hilbert 空间，因此边界项是预期现象；
- 优先“可运行 + 可校验”的小实现，不引入大型框架。

## R10

`demo.py` 主要函数职责：

- `validate_parameters`：参数合法性校验；
- `build_ladder_operators`：构造 `a, a†`；
- `build_harmonic_operators`：构造 `X,P,H,I`；
- `heisenberg_evolve`：执行 `U†OU` 演化；
- `harmonic_closed_form`：给出 `X(t),P(t)` 解析式；
- `coherent_state` / `expectation`：构造初态并计算期望；
- `build_operator_error_table`：输出算符层误差表；
- `build_expectation_table`：输出期望值层误差表；
- `fit_energy_levels`：能级线性回归；
- `torch_commutator_error`：PyTorch 对易关系复核；
- `truncated_commutator_model`：截断理论对易关系目标；
- `run_heisenberg_mvp`：组织整体流程；
- `main`：打印验收信息。

## R11

运行方式：

```bash
cd Algorithms/物理-量子力学-0201-海森堡矩阵力学_(Heisenberg_Matrix_Mechanics)
uv run python demo.py
```

脚本无交互输入，直接输出结果表。

## R12

输出字段说明：

- `commutator_error_numpy` / `commutator_error_torch`：
  `||[X,P]-iħI||_F`，体现与无限维理想关系的差异；
- `commutator_error_trunc_model`：
  `||[X,P]-iħ(I-N|N-1><N-1|)||_F`，应接近机器精度；
- `energy_fit`：
  回归斜率应接近 `ħω`，截距应接近 `0.5ħω`，`R2` 应接近 1；
- `Operator-level errors` 表：
  每个时间点的 `X/P` 演化误差与对易关系误差；
- `Expectation-level` 表：
  `<X>(t),<P>(t)` 与经典轨迹对照误差；
- `check_*_all_finite`：
  数值稳定性快速检查。

## R13

最小验收标准（本脚本默认应满足）：

1. `uv run python demo.py` 可直接运行；
2. `commutator_error_trunc_model` 在 `1e-12 ~ 1e-13` 量级；
3. `X_error_fro, P_error_fro` 在 `1e-12` 附近或更小；
4. 能级拟合 `R2` 接近 `1.0`，斜率与截距匹配理论值；
5. 两个 `check_*_all_finite` 都为 `True`。

## R14

关键参数与调参建议：

- `dim`：维度越大，截断边界影响越弱，但矩阵运算成本上升；
- `omega`：决定系统时间尺度 `T=2π/ω`；
- `times` 采样点：更多点可更细地看误差随时间变化；
- `alpha`（相干态参数）：控制期望值振幅，过大时需要更大 `dim` 才能稳定表示态。

经验上可先固定 `alpha`，逐步提高 `dim` 观察截断误差收敛。

## R15

与其他表述/方法的关系：

- 薛定谔图像：演化态 `|ψ(t)>`；海森堡图像：演化算符 `O(t)`；
- 路径积分：强调按路径求和；矩阵力学：强调代数结构与算符动力学；
- 对于谐振子，这三种表述应给出一致物理结论，本 MVP 主要展示海森堡表述的“矩阵可计算性”。

## R16

典型应用场景：

- 离散能级系统的算符动力学分析；
- 量子光学中 `a,a†` 代数与相干态演化；
- 数值量子课程中的“从理论公式到矩阵实现”教学演示；
- 更复杂体系（自旋链、截断场模态）之前的基线验证。

## R17

可扩展方向：

- 加入时变哈密顿量 `H(t)`，比较 Trotter 分解与直接指数法；
- 推广到多模谐振子或 Jaynes-Cummings 类模型；
- 用稀疏矩阵与 Krylov 子空间优化大维度演化；
- 引入耗散项（Lindblad 主方程）比较开闭系统动力学；
- 增加自动化误差-维度收敛扫描。

## R18

`demo.py` 的源码级流程（8 步）：

1. `run_heisenberg_mvp` 设置 `dim,m,ω,ħ` 并调用 `validate_parameters`。
2. `build_harmonic_operators` 内先调用 `build_ladder_operators` 构造 `a,a†`，再按公式组装 `X,P,H,I`。
3. 计算即时交换子 `comm=[X,P]`，分别与 `iħI` 及 `truncated_commutator_model` 对照，得到两类误差。
4. `fit_energy_levels` 调用 `np.linalg.eigvalsh` 求 `H` 的本征值，再用 `LinearRegression` 拟合 `E_n~n` 提取斜率、截距与 `R2`。
5. `build_operator_error_table` 遍历时间点；每个时间点调用 `heisenberg_evolve`，其中 `heisenberg_evolve` 用 `scipy.linalg.expm` 先得到 `U=exp(-iHt/ħ)`，再执行两次矩阵乘法 `U†OU`。
6. 同一时间点调用 `harmonic_closed_form` 生成 `X_exact,P_exact`，并计算 Frobenius 误差与交换子误差写入 `pandas` 表。
7. `build_expectation_table` 先由 `coherent_state` 构造初态，再用 `expectation` 计算 `<X>(t),<P>(t)`，并与经典轨迹公式逐点对比。
8. `main` 统一打印标量指标、两张数据表和 `check_*_all_finite`，形成完整可复现实验输出。
