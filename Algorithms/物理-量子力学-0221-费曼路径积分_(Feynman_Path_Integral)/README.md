# 费曼路径积分 (Feynman Path Integral)

- UID: `PHYS-0220`
- 学科: `物理`
- 分类: `量子力学`
- 源序号: `221`
- 目标目录: `Algorithms/物理-量子力学-0221-费曼路径积分_(Feynman_Path_Integral)`

## R01

费曼路径积分把量子传播幅写成“对所有路径求和”。本条目采用一维谐振子在欧氏时间（虚时）下的最小可运行版本：

- 将传播核 `K_E(x_f, x_i; beta)` 做时间切片离散；
- 把中间路径积分转成高斯型多维积分；
- 用线性代数（行列式与线性方程组）直接求离散路径积分值；
- 对照谐振子的解析欧氏传播核，验证收敛行为。

## R02

MVP 的任务定义：

1. 固定参数 `m=1.0, omega=1.25, beta=1.10, hbar=1.0, x_i=0.35, x_f=-0.20`；
2. 对 `n_slices in [4, 6, 8, 12, 16, 24, 32, 48, 64]` 计算离散路径积分传播核；
3. 计算每个切片数对应的绝对误差和相对误差；
4. 用 `scikit-learn` 对 `log(error)` 与 `log(n_slices)` 做线性拟合，估计收敛阶；
5. 用 `PyTorch` 对一个参考切片数（32）复算传播核，交叉核验 `NumPy` 结果；
6. 用 `pandas` 输出可直接验收的收敛表。

## R03

为什么选“欧氏时间 + 谐振子”作为路径积分 MVP：

- 实时路径积分振荡严重（符号问题），最小示例容易数值不稳定；
- 欧氏时间权重 `exp(-S_E/hbar)` 为正，能稳定演示“路径积分到可计算算法”的全过程；
- 谐振子同时有解析传播核，能提供严格对照基准，不会停留在黑箱数值结果；
- 离散作用量是二次型，正好体现路径积分与高斯积分、线性代数之间的核心联系。

## R04

本实现使用的关键公式：

1. 欧氏传播核：
`K_E(x_f,x_i;beta) = \int_{x(0)=x_i}^{x(beta)=x_f} Dx exp(-S_E[x]/hbar)`。

2. 欧氏作用量（谐振子）：
`S_E = \int_0^beta [ m/2 (dx/dtau)^2 + m omega^2 x^2 /2 ] dtau`。

3. 时间切片 `dtau=beta/N` 后，离散作用量写为
`S_E = 0.5 y^T A y - b^T y + c0`，其中 `y=(x_1,...,x_{N-1})`。

4. 高斯积分闭式：
`\int exp(-(1/(2hbar)) y^T A y + (1/hbar) b^T y) dy`
`= ((2pi hbar)^{(N-1)/2}/sqrt(det A)) * exp((1/(2hbar)) b^T A^{-1} b)`。

5. 谐振子解析欧氏核（用于对照）：
`K_exact = sqrt(m omega / (2pi hbar sinh(omega beta)))`
`* exp(-m omega ((x_f^2+x_i^2) cosh(omega beta)-2x_f x_i)/(2hbar sinh(omega beta)))`。

## R05

算法流程：

1. 校验输入参数（正值、有限值、切片数下限）；
2. 计算解析核 `K_exact`；
3. 对每个 `n_slices` 构造离散二次型 `(A,b,c0)`；
4. 用 Cholesky 分解计算 `log(det(A))` 并求解 `A^{-1}b`，代入高斯积分闭式；
5. 合并离散路径积分的归一化因子，得到 `K_discrete`；
6. 计算 `abs_error`、`rel_error`，写入 `pandas` 表；
7. 对 `log(error)` vs `log(n_slices)` 做线性拟合，估计收敛阶；
8. 用 `PyTorch` 在参考切片点复算一次，验证数值实现一致性。

## R06

正确性保障机制：

- 解析对照：每个切片数都与闭式解析核逐点比较；
- 收敛规律：通过对数线性拟合检查误差随切片加密是否按幂律下降；
- 双实现核验：`NumPy` 与 `PyTorch` 对同一离散系统独立计算，比较差值；
- 正定性检查：对 Hessian `A` 做 Cholesky 分解，若失败则立即报错。

## R07

复杂度（单个切片数记为 `N`，切片档位数记为 `L`）：

- 构造三对角矩阵 `A`：`O(N)` 存储/填充；
- 求解 `A^{-1}b` 与 `det(A)`（致密线代路径）：`O(N^3)`；
- 整体扫描：`O(L*N_max^3)`；
- 本脚本默认 `N_max=64`，运行开销很小，适合作为教学级 MVP。

## R08

边界与异常处理：

- `mass/omega/beta/hbar <= 0` 直接报错；
- 标量参数非有限值直接报错；
- `n_slices < 2` 报错；
- 收敛拟合至少需要 3 个切片点；
- 若 `A` 非正定（Cholesky 失败）则报错，避免继续产生伪结果；
- 最后输出 `check_convergence_table_all_finite` 做有限性快检。

## R09

MVP 取舍：

- 只做一维谐振子欧氏核，不覆盖多体、场论或实时振荡积分；
- 只计算两端点传播核，不做完整路径采样可视化；
- 采用致密矩阵求解以保持代码简洁，未做稀疏/Krylov 优化；
- 强调“离散作用量 -> 高斯积分 -> 可验收误差表”的完整闭环。

## R10

`demo.py` 主要函数职责：

- `validate_parameters`：参数合法性检查；
- `exact_euclidean_kernel`：解析欧氏传播核；
- `build_discrete_action_system`：构造离散作用量的 `(A,b,c0)`；
- `discrete_euclidean_kernel_numpy`：NumPy 路径积分主计算；
- `discrete_euclidean_kernel_torch`：PyTorch 交叉复算；
- `build_convergence_table`：生成误差数据表；
- `fit_convergence_order`：估计误差收敛阶；
- `run_path_integral_mvp`：组织全流程并打包结果；
- `main`：打印汇总、表格、核验指标。

## R11

运行方式：

```bash
cd Algorithms/物理-量子力学-0221-费曼路径积分_(Feynman_Path_Integral)
uv run python demo.py
```

脚本无交互输入。

## R12

输出字段说明：

- `exact_kernel`：解析欧氏传播核；
- `convergence_fit.order`：拟合得到的误差收敛阶（理论上应为正，越大收敛越快）；
- `convergence_fit.R2`：对数线性拟合质量；
- 收敛表列：
  - `n_slices`：切片数；
  - `dtau`：欧氏时间步长 `beta/n_slices`；
  - `kernel_discrete`：离散路径积分结果；
  - `kernel_exact`：解析结果；
  - `abs_error` / `rel_error`：误差指标；
- `NumPy vs PyTorch cross-check`：同一切片点的双实现差值；
- `check_convergence_table_all_finite`：结果表有限性检查。

## R13

最小验收标准（默认参数）：

1. `uv run python demo.py` 可直接运行；
2. 收敛表中 `abs_error` 随切片总体下降；
3. 拟合 `order > 0` 且 `R2` 较高（通常接近 1）；
4. `NumPy vs PyTorch` 差值在双精度舍入误差量级；
5. `check_convergence_table_all_finite = True`。

## R14

调参建议：

- 增大 `n_slices`：离散误差降低，但线代成本升高；
- 增大 `beta`：传播核衰减更强，误差与条件数表现会变化；
- 改变 `x_i, x_f`：可测试不同边界条件下的数值稳定性；
- 改变 `omega`：控制势阱曲率，影响时间步长需求；
- 若要更大 `N`，可改成三对角专用求解器或稀疏格式。

## R15

与其他量子表述的关系：

- 薛定谔方程法：直接解偏微分方程演化波函数；
- 算符法：在 Hilbert 空间处理算符与本征态；
- 路径积分法：在配置空间对路径求和。

本条目展示的是路径积分在“可数值实现”层面的具体落地，不依赖把第三方库当黑箱求解器。

## R16

应用场景：

- 量子统计中的欧氏时间传播核/配分函数离散近似；
- 路径积分教学中的“从形式定义到数值算法”示例；
- 更复杂势函数离散作用量算法的基线验证；
- 为后续蒙特卡罗路径采样（PIMC）做解析可控对照。

## R17

可扩展方向：

- 从谐振子扩展到非线性势并引入数值积分/采样；
- 从一维推广到多维耦合系统；
- 加入实时路径积分与相位重加权，研究符号问题；
- 用稀疏矩阵和专用三对角算法提升大 `N` 性能；
- 增加自动扫描脚本，输出误差-成本 Pareto 曲线。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `run_path_integral_mvp` 设定物理参数与切片列表，并调用 `validate_parameters`。
2. `build_convergence_table` 先调用 `exact_euclidean_kernel` 得到解析基准 `K_exact`。
3. 对每个 `n_slices`，`build_discrete_action_system` 组装离散作用量二次型 `S_E = 0.5 y^T A y - b^T y + c0`。
4. `discrete_euclidean_kernel_numpy` 使用 Cholesky 分解提取 `log(det(A))` 并求解 `A^{-1}b`，代入高斯积分闭式，得到离散传播核 `K_discrete`。
5. 将 `K_discrete` 与 `K_exact` 比较，记录 `abs_error`、`rel_error` 并汇总成 `pandas` 表。
6. `fit_convergence_order` 对 `log(abs_error)` 与 `log(n_slices)` 做 `LinearRegression`，估计收敛阶和 `R2`。
7. `discrete_euclidean_kernel_torch` 在参考切片点重复第 4 步（基于 `torch.linalg`），与 NumPy 结果做一致性核验。
8. `main` 打印参数、拟合结果、收敛表、双实现差值与有限性检查，形成完整可复现实验输出。
