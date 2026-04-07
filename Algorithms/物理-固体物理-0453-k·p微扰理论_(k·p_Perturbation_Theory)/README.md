# k·p微扰理论 (k·p Perturbation Theory)

- UID: `PHYS-0433`
- 学科: `物理`
- 分类: `固体物理`
- 源序号: `453`
- 目标目录: `Algorithms/物理-固体物理-0453-k·p微扰理论_(k·p_Perturbation_Theory)`

## R01

`k·p` 微扰理论是半导体能带理论中的经典有效模型方法：
- 先在高对称点（通常是 `Γ` 点）求布洛赫本征态；
- 再把哈密顿量对小波矢 `k` 展开，保留低阶项；
- 用少数关键能带和动量矩阵元，近似描述带边色散与有效质量。

本条目给出一个可运行、可审计的 MVP：用三带 toy 模型生成“参考能带”，再用二带 `k·p` 微扰模型在 `Γ` 附近逼近，并做定量验证。

## R02

MVP 范围（刻意收敛）：
1. 仅做 1D `k` 线（`-k_max` 到 `k_max`），不做完整 3D 布里渊区。
2. 用 3×3 对称哈密顿量 `(v,c,r)` 作为“全模型参考”。
3. 对远带 `r` 做二阶消去（Löwdin 思路），得到 2×2 有效 `(v,c)` 哈密顿量。
4. 比较三种模型误差：
   - `diag_baseline`（无带间耦合）；
   - `kp_perturbative`（二阶微扰参数）；
   - `kp_torch_refined`（在微扰初值上做小规模梯度拟合）。

## R03

本实现使用的核心公式：

1. 全模型哈密顿量（对称实矩阵）：
`H_full(k) = H0 + k*H1 + k^2*H2`

2. 二带 `k·p` 有效哈密顿量：
`H_eff(k) = [[Ev0 + Av_eff k^2, P_eff k], [P_eff k, Ec0 + Ac_eff k^2]]`

3. 远带二阶修正（示意）：
`Av_eff = Av + P_vr^2 / (Ev0 - Er0)`
`Ac_eff = Ac + P_cr^2 / (Ec0 - Er0)`

4. 能带来自本征值问题：
`H(k) u_n(k) = E_n(k) u_n(k)`

5. 近 `Γ` 有效质量拟合：
`E(k) ~= E(0) + s k^2`, 且 `m*/m0 = (hbar^2/2m0) / s`（价带取曲率绝对值）。

## R04

为什么这个构造能体现 `k·p` 本质：
- `k` 线性项（`P*k`）体现了带间动量耦合，是 `k·p` 的核心来源；
- 二阶项给出带曲率（有效质量）；
- 远带 `r` 不显式保留到最终模型，但通过二阶分母 `1/(E_n0-Er0)` 反馈到低能参数，符合微扰消元思想。

因此，即使是 toy 体系，也能演示“从多带到少带”的可解释降维。

## R05

`demo.py` 中的模型对应：
1. `full_three_band_hamiltonian`：构造三带参考哈密顿量。
2. `solve_full_bands`：逐个 `k` 求 3×3 本征值，得到参考 `E_full(k)`。
3. `perturbative_two_band_params`：把远带对角修正折叠进 `Av_eff/Ac_eff`。
4. `two_band_hamiltonian` + `solve_two_band`：求 2×2 有效能带。
5. `solve_diagonal_baseline`：去掉 `P_eff*k` 的对照组。

## R06

验证逻辑：
1. 先验证数值合法性：哈密顿量是否厄米（本例为实对称）。
2. 再验证物理/建模价值：`k·p` 在 `Γ` 附近误差必须小于阈值，且优于对角基线。
3. 最后验证可优化性：Torch 小规模拟合不应劣化微扰初值。

这三层检查分别覆盖“实现正确”“模型有效”“参数可调”。

## R07

脚本输出内容：
- `Effective-parameter table`：微扰初值与 Torch 拟合值；
- `Error metrics`：近 `Γ` 与全区间 RMSE；
- `Effective masses`：由 `k^2` 拟合得到的带边有效质量；
- `Band samples`：抽样 `k` 点的全模型与近似模型对比；
- `Checks` 与 `Validation: PASS/FAIL`：自动验收信号。

## R08

复杂度分析（`N_k` 为采样点数）：
1. 三带参考求解：每个 `k` 做一次 3×3 本征分解，成本 `O(N_k)`（矩阵维度固定）。
2. 二带模型求解：每个 `k` 做一次 2×2 本征分解，成本 `O(N_k)`。
3. 线性回归与 RMSE 计算：`O(N_k)`。
4. Torch 拟合：`O(steps * N_fit)`，其中 `N_fit` 为拟合窗口内点数。

在默认参数（`N_k=161`, `steps=900`）下可秒级运行。

## R09

数值稳定措施：
1. 配置检查：`ev0 < ec0 < er0`、`n_k` 为奇数（确保包含 `k=0`）。
2. 本征求解使用对称矩阵专用路径（`scipy.linalg.eigh`），避免一般复特征分解噪声。
3. Torch 中 2×2 本征值使用闭式公式，避免小矩阵反复调用黑盒分解。
4. 开根号项加 `1e-18` 防止极小负数数值噪声导致 `nan`。

## R10

MVP 技术栈：
- `numpy`：哈密顿量构建、网格与向量化计算；
- `scipy.linalg.eigh`：对称本征值分解；
- `pandas`：结果表格化输出；
- `scikit-learn`：线性回归与 RMSE 指标；
- `PyTorch`：梯度下降微调 `k·p` 有效参数。

没有调用任何材料计算黑盒软件；每个关键物理步骤都在源码中显式展开。

## R11

运行方式：

```bash
cd Algorithms/物理-固体物理-0453-k·p微扰理论_(k·p_Perturbation_Theory)
uv run python demo.py
```

脚本无需交互输入。若检查通过，末尾输出 `Validation: PASS`；失败则以非零退出码结束。

## R12

关键参数（`KPConfig`）：
1. 能量与曲率：`ev0/ec0/er0`, `av/ac/ar`。
2. 带间耦合：`p_cv/p_cr/p_vr`。
3. 采样窗口：`k_max/n_k`，拟合窗口 `fit_k_max`。
4. Torch 优化：`torch_steps/torch_lr/torch_seed`。
5. 验收阈值：`rmse_near_gamma_max`, `hermitian_tol`。

调参建议：增大 `p_cv` 会增强价带-导带反交叉效应；减小 `er0-ec0` 或 `er0-ev0` 会放大远带二阶修正。

## R13

内置验收条件：
1. `Hermitian residual <= hermitian_tol`。
2. `kp_perturbative` 的近 `Γ` RMSE 小于阈值（默认 `0.03 eV`）。
3. `kp_perturbative` 近 `Γ` RMSE 优于 `diag_baseline`。
4. `kp_torch_refined` 不劣于微扰初值。
5. 导带有效质量落在合理区间 `(0.02, 3.0) m0`。

全部满足才判定 `Validation: PASS`。

## R14

当前实现局限：
1. 仅 1D `k` 线，不含各向异性张量质量与多方向耦合。
2. 没有显式自旋-轨道耦合、简并处理与 Luttinger 参数体系。
3. 远带修正只做最简二阶对角项，未包含更高阶或能量依赖项。
4. 数据是 toy 模型，不直接对应真实材料参数。

## R15

可扩展方向：
1. 从 2 带扩展到 4 带/8 带 Kane 模型（含重空穴、轻空穴、分裂带）。
2. 引入自旋与 SOC，支持简并微扰与块对角化。
3. 对接 DFT/Wannier 导出的带结构数据做参数反演。
4. 增加 2D/3D `k` 路径与等能面可视化。
5. 在拟合中加入物理正则（参数符号、和规则、对称性约束）。

## R16

典型应用语境：
1. 半导体带边有效质量估计。
2. `Γ` 点附近非抛物性带结构快速近似。
3. 光电器件仿真中的低能有效哈密顿量构建。
4. 教学场景中演示“多带模型 -> 有效少带模型”的微扰思路。

## R17

与相邻方法的关系：
1. 与紧束缚/第一性原理相比：`k·p` 参数更少、近带边计算更快，但适用区间更局部。
2. 与纯抛物线有效质量模型相比：`k·p` 通过线性耦合项可描述带间混合与非抛物性。
3. 与经验赝势法相比：`k·p` 更强调解析结构与局域（`Γ` 点）展开。

本 MVP 选择“全模型对照 + 微扰约化 + 参数微调”的路线，便于同时检查物理解释性与数值可复现性。

## R18

`demo.py` 源码级算法流程（9 步）：
1. `main` 构造 `KPConfig`，`validate_config` 检查能级顺序、采样点和拟合窗口合法性。
2. 生成 `k_grid` 后，`solve_full_bands` 逐点构造 `H_full(k)` 并调用 `scipy.linalg.eigh` 得到三带参考谱。
3. 在 SciPy/LAPACK 路径里，对称本征求解会先把实对称矩阵约化到三对角形式，再求特征值/特征向量并回代到原基底（不是通用 `eig` 黑盒路径）。
4. `perturbative_two_band_params` 用二阶分母 `1/(E_n0-Er0)` 计算远带修正，得到 `av_eff/ac_eff/p_eff`。
5. `solve_two_band` 对每个 `k` 的 2×2 `H_eff(k)` 求本征值；`solve_diagonal_baseline` 去掉非对角耦合生成对照曲线。
6. `fit_effective_masses` 对 `E(k)-E(0)` 与 `k^2` 做 `LinearRegression(fit_intercept=False)`，回归斜率并换算 `m*/m0`。
7. `fit_two_band_with_torch` 把 `(av_eff, ac_eff, p_eff)` 设为可训练变量，使用 Adam 最小化拟合窗口内能带均方误差。
8. `metrics` 用 `sklearn.metrics.mean_squared_error` 计算近 `Γ` 与全区间 RMSE，并组织成 `pandas` 表。
9. `main` 执行 5 条阈值检查并打印 `Validation: PASS/FAIL`，失败则 `SystemExit(1)`。
