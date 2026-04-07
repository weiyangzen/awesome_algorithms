# 密度泛函理论 (Density Functional Theory, DFT)

- UID: `PHYS-0203`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `204`
- 目标目录: `Algorithms/物理-计算物理-0204-密度泛函理论_(Density_Functional_Theory,_DFT)`

## R01

密度泛函理论（DFT）的核心思想是：把多电子体系的基态性质写成电子密度 `n(r)` 的泛函，而不直接在 `3N` 维波函数空间求解。

本条目给出一个“可运行、可审计”的最小原型（MVP）：
在 1D 实空间上实现 orbital-free DFT（OF-DFT）自洽迭代，展示从初始密度到收敛密度和总能分解的完整链路。

## R02

MVP 采用的物理模型：

- 空间：1D 均匀网格（教学简化，不是材料级模型）
- 外势：谐振子势 `v_ext(x) = 0.5 * omega^2 * x^2`
- 电子相互作用：软库仑 Hartree 核
- 动能泛函：Thomas-Fermi + Weizsaecker 梯度修正
- 交换项：LDA exchange-only

这保证了 DFT 的关键组成（动能、外势、Hartree、交换）都在源码里显式出现。

## R03

本实现的总能泛函写为：

`E[n] = T_W[n] + T_TF[n] + E_ext[n] + E_H[n] + E_x[n]`

其中：

- `T_W[n] = (lambda_W/8) * integral (|grad n|^2 / n) dx`
- `T_TF[n] = C_TF * integral n^(5/3) dx`
- `E_ext[n] = integral v_ext(x) n(x) dx`
- `E_H[n] = 0.5 * integral integral n(x)n(x') / sqrt((x-x')^2 + a^2) dxdx'`
- `E_x[n] = C_x * integral n^(4/3) dx`

其中 `C_TF > 0`，`C_x < 0`。

## R04

数值离散策略：

- 使用 `Grid1D` 生成 `x` 和 `dx`
- 二阶中心差分构造 Laplacian 三对角矩阵
- 软库仑核预计算成稠密矩阵 `K`
- 每轮 SCF 解一维本征问题：
  `[-lambda_W/2 * Laplacian + v_eff[n]] phi = mu phi`
- 由 `n = phi^2` 得到新密度，并强制归一化到目标电子数

这里的 `phi = sqrt(n)` 让密度天然非负，避免直接在 `n` 上优化时的负值问题。

## R05

关键假设与边界：

- 使用 1D toy model 表达算法流程，不追求真实材料高精度
- 使用 3D 常见 TF/LDA 指数（`5/3`、`4/3`）作教学近似
- exchange-only，未加入相关泛函
- SCF 稳定策略为线性 mixing，不含 DIIS/Pulay
- 有限盒边界，未处理周期边界与 k 点

## R06

`demo.py` 输入输出约定：

- 输入：脚本内固定参数（网格、电子数、mix、阈值）
- 无交互输入，适合 `uv run python demo.py` 自动执行
- 输出包括：
1. SCF 尾部迭代日志
2. 最终能量分解汇总表
3. 逐项检查与 `Validation: PASS/FAIL`

## R07

SCF 主流程（高层）：

1. 初始化网格、外势、Laplacian、Hartree 核。
2. 用高斯函数初始化 `phi`，并归一化到给定电子数。
3. 用当前密度计算 `v_H`、`v_TF`、`v_x`，组装 `v_eff`。
4. 解最低本征态得到 `phi_out`（对应基态密度方向）。
5. 对 `phi` 线性混合并重新归一化。
6. 计算总能与分项、密度残差 `drho`、能量改变量 `dE`。
7. 满足双阈值（`drho` 和 `dE`）则终止，否则继续。

## R08

设网格点数为 `N`，迭代步数为 `K`：

- Hartree 势 `K @ n` 主导 `O(N^2)`
- 三对角最低本征态求解约 `O(N)` 到 `O(N^2)`（实现细节相关）
- 其余向量计算约 `O(N)`

综合可写为：

- 时间复杂度：`O(K * N^2)`
- 空间复杂度：`O(N^2)`（主要来自 Hartree 核矩阵）

## R09

数值稳定手段：

- 对密度做 `clip(n, 1e-14, +inf)`，避免幂函数和除法奇异
- 每轮都归一化 `phi`，严格维持电子数守恒
- 使用 `phi` 而不是直接 `n` 进行更新，天然保证 `n >= 0`
- 使用线性 mixing (`mix=0.03`) 降低 SCF 振荡
- 收敛采用 `drho + dE` 双条件，减少伪收敛

## R10

最小工具栈：

- `numpy`：网格、核矩阵、向量化数值计算
- `scipy.linalg.eigh_tridiagonal`：每轮最低本征态求解
- `pandas`：迭代日志与摘要表输出

未调用高层 DFT 软件包，算法路径在 `demo.py` 中可直接追踪。

## R11

运行方式：

```bash
cd Algorithms/物理-计算物理-0204-密度泛函理论_(Density_Functional_Theory,_DFT)
uv run python demo.py
```

成功时末行显示 `Validation: PASS`；
若任一检查失败，程序以非零状态退出。

## R12

关键输出字段说明：

- `E_total`：总能
- `T_w`：Weizsaecker 梯度动能
- `T_tf`：Thomas-Fermi 局域动能
- `E_ext`：外势能
- `E_H`：Hartree 能（正常应为正）
- `E_x`：交换能（正常应为负）
- `mu`：当前迭代最低本征值（化学势近似）
- `drho_L2`：相邻迭代密度差范数
- `N_integral`：密度积分电子数
- `N_error`：电子数误差

## R13

内置验证条件：

1. SCF 收敛
2. `drho_L2 < 2e-4`
3. `|N_integral - N_e| < 1e-6`
4. `T_w > 0`
5. `T_tf > 0`
6. `E_H > 0`
7. `E_x < 0`
8. 密度最小值非负

全部满足时打印 `Validation: PASS`。

## R14

当前原型局限：

- 1D 教学模型，不等价于真实 3D 固体/分子计算
- 交换相关仅含交换（无相关）
- OF-DFT 本身对化学键细节与壳层结构刻画有限
- 未包含赝势、周期边界、k 点、温度展宽等工程要素

## R15

可扩展方向：

- 加入相关泛函（LDA correlation / GGA）
- 引入更稳健的 SCF 加速（DIIS/Anderson）
- 用 FFT 卷积或 Poisson 求解加速 Hartree 计算
- 升级到 2D/3D 网格
- 与 Kohn-Sham 分支联动，比较 OF-DFT 与 KS-DFT 结果差异

## R16

适用场景：

- DFT 入门教学与算法可视化讲解
- 自洽迭代控制策略（mix、阈值、日志）快速试验
- 作为更大电子结构项目中的“可运行最小核”
- 用于验证能量分解与密度守恒是否实现正确

## R17

与相关方法简对比：

- Hartree-only：实现更简单，但缺少交换项，误差通常更大
- OF-DFT（本实现）：成本低、结构透明、适合教学和大规模粗筛
- KS-DFT：精度和适用性通常更好，但计算成本和实现复杂度更高

本条目优先目标是“可解释 + 可运行 + 可验证”的 DFT 算法骨架。

## R18

`demo.py` 源码级算法流（8 步）：

1. `Grid1D` 和 `OFDFTConfig` 定义网格与 SCF 超参数；初始化 `phi` 并按电子数归一化。
2. `build_laplacian_tridiagonal` 构造二阶差分 Laplacian，`build_soft_coulomb_kernel` 预计算 Hartree 核。
3. 每轮迭代先由 `density = phi^2` 得到密度，再用 `hartree_potential` 与 `lda_local_terms` 计算 `v_H / v_TF / v_x`。
4. 组装 `v_eff = v_ext + v_H + v_TF + v_x`，调用 `solve_ground_state` 用 `eigh_tridiagonal` 取最低本征态。
5. 对 `phi_out` 做绝对值和归一化，随后与旧 `phi` 线性 mixing 得到 `phi_new`。
6. `total_energy` 计算 `E_total, T_w, T_tf, E_ext, E_H, E_x`，并记录 `mu, drho_L2, dE, N_integral`。
7. 若 `drho_L2` 与 `dE` 同时小于阈值则判定收敛；否则进入下一轮。
8. `main` 打印迭代尾表、最终摘要与 8 项检查，最终输出 `Validation: PASS/FAIL`。
