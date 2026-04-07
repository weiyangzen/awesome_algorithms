# 玻恩-奥本海默近似 (Born-Oppenheimer Approximation)

- UID: `PHYS-0201`
- 学科: `物理`
- 分类: `量子化学`
- 源序号: `202`
- 目标目录: `Algorithms/物理-量子化学-0202-玻恩-奥本海默近似_(Born-Oppenheimer_Approximation)`

## R01

玻恩-奥本海默近似（BO 近似）的核心思想是利用“核远重于电子”的质量尺度分离：
- 先把核位置 `R` 视为参数，求电子本征问题；
- 再把电子能量作为核的有效势能面，求核的振动态。

本条目给出一个可运行、可审计的最小 MVP：
- 体系选用 `H2+`（两核一电子）；
- 电子部分用最小 LCAO（1s-1s）显式公式；
- 核部分在 BO 势能面上做 1D 有限差分本征求解。

## R02

问题定义（原子单位制）如下：
- 输入：核间距区间 `[R_min, R_max]`、离散网格数、质子质量、振动能级截断数。
- 电子方程（固定 `R`）：
  - `H_e(R) psi_e(r;R) = E_e(R) psi_e(r;R)`
- BO 势能面：
  - `V_BO(R) = E_e(R) + 1/R`
- 核方程：
  - `[-(1/(2mu)) d^2/dR^2 + V_BO(R)] chi_n(R) = E_n chi_n(R)`

输出：
- 平衡核距 `R_eq`；
- BO 势能最低值 `V_min`；
- 最低若干振动能级 `E_n`；
- 数值一致性检查（力、曲率、能级单调性等）。

## R03

本 MVP 使用的电子最小基模型：
- 重叠积分：
  - `S(R) = exp(-R) * (1 + R + R^2/3)`
- 电子哈密顿量矩阵元：
  - `H_AA(R) = -1/2 - 1/R + (1 + 1/R) exp(-2R)`
  - `H_AB(R) = -1/2 S(R) - exp(-R)(1+R)`
- 成键态电子能：
  - `E_e(R) = (H_AA + H_AB) / (1 + S)`
- BO 势能：
  - `V_BO(R) = E_e(R) + 1/R`

该模型是教学级近似，重点在于完整展示 BO 算法链路，而非化学精度。

## R04

物理与建模假设：
- 只考虑 `H2+`，且核运动限制在核间距 `R` 的 1D 径向自由度；
- 电子采用最小 1s 基组（不含极化/高角动量函数）；
- 忽略非绝热耦合项（这是 BO 近似本身的前提）；
- 核薛定谔方程在有限区间上施加盒边界（Dirichlet）。

## R05

高层算法流程：
1. 在连续区间内最小化 `V_BO(R)` 得到 `R_eq` 与 `V_min`。
2. 生成核坐标网格 `R_i`，计算 `V_BO(R_i)`。
3. 构造有限差分三对角哈密顿量 `T + V`。
4. 对三对角矩阵求最低 `n` 个本征值，得到振动能级。
5. 在 `R_eq` 邻域做二次拟合估计局域力常数。
6. 用 `torch` 自动微分计算 `dV/dR` 与 `d^2V/dR^2`，交叉检查平衡点性质。
7. 输出汇总表、能级表、势能采样点和验证结果。

## R06

`demo.py` 核心函数职责：
- `overlap_1s / h_aa / h_ab`：电子最小基积分/矩阵元公式。
- `electronic_energy_sigma_g`：电子成键态本征值 `E_e(R)`。
- `bo_potential`：构造 `V_BO(R)`。
- `find_equilibrium`：`scipy.optimize.minimize_scalar` 寻找势能最小值。
- `solve_nuclear_motion`：组装三对角核哈密顿量并调用 `eigh_tridiagonal`。
- `fit_local_quadratic`：`sklearn` 二次回归估计局域曲率。
- `torch_force_and_curvature`：`torch.autograd` 导数与 Hessian。
- `validate_results`：最小数值正确性检查。

## R07

复杂度（设核网格点数为 `N`）：
- 势能构造：`O(N)`；
- 三对角本征求解（取前 `k` 个低能级）：典型 `O(Nk)` 到 `O(N^2)` 区间，MVP 规模下近似线性可用；
- 局域拟合：`O(M)`（`M` 为窗口内点数，远小于 `N`）；
- 自动微分：常数级（单点标量图）。

空间复杂度主要为网格数组和三对角向量：`O(N)`。

## R08

数值稳定与边界处理：
- 通过 `R_min=0.5` 避免 `1/R` 奇点；
- 优化区间和离散区间一致地覆盖成键区和解离区；
- 对局域拟合窗口内样本数做下限检查（至少 6 点）；
- 验证振动能级严格递增，防止本征求解异常；
- 若最小化失败，直接抛出异常而不是静默继续。

## R09

MVP 取舍说明：
- 优先“透明可审计”：公式均在源码中显式展开，不依赖量化化学黑盒库；
- 保留最小物理正确性（PES 最小值、核振动量子化）；
- 不追求化学精度基准（缺少更大基组与电子相关）；
- 不实现多原子多自由度核动力学，仅演示 BO 主干机制。

## R10

最小工具栈及用途：
- `numpy`：向量化公式、网格与数组运算；
- `scipy`：
  - `minimize_scalar` 求 `R_eq`；
  - `eigh_tridiagonal` 求核振动态；
- `pandas`：结果表格化输出；
- `scikit-learn`：局域二次拟合提取力常数；
- `torch`：自动微分计算力与曲率，做独立交叉验证。

## R11

运行方式：

```bash
cd Algorithms/物理-量子化学-0202-玻恩-奥本海默近似_(Born-Oppenheimer_Approximation)
uv run python demo.py
```

脚本不需要任何命令行参数，也不会请求交互输入。

## R12

输出字段说明：
- `R_eq_bohr`：BO 势能最小对应核距；
- `E_electronic_eq_hartree`：平衡核距处电子能；
- `V_BO_min_hartree`：总 BO 势最小值；
- `force_at_eq_hartree_per_bohr`：`-dV/dR`（应接近 0）；
- `k_fit_hartree_per_bohr2`：局域二次拟合的力常数；
- `k_autograd_hartree_per_bohr2`：自动微分二阶导；
- `omega_harmonic_au / zpe_harmonic_hartree`：谐振近似频率与零点能；
- `E_vib0_total_hartree`：核基振动态总能；
- `mean_R_vib0_bohr`：基振动态期望核距。

## R13

最小测试建议：
1. 直接运行默认配置，检查 `Validation: PASS`。
2. 将 `n_grid` 降到较小值（如 200），观察振动能级变化与稳定性。
3. 将 `fit_window` 改得过小（如 `0.01`），确认触发“样本不足”异常。
4. 改大 `R_max`（如 12.0）验证解离端势能趋近 `-0.5 Ha` 附近。

## R14

可调参数（`BOConfig`）：
- `r_min, r_max`：核坐标求解区间；
- `n_grid`：核方程离散精度；
- `optimize_left, optimize_right`：平衡核距搜索区间；
- `n_vib_levels`：输出振动能级数；
- `fit_window`：局域拟合窗口半宽。

调参经验：
- `n_grid` 越大，振动能级收敛越稳定但耗时增加；
- `fit_window` 过大时会偏离局域谐振区，过小时拟合噪声增大。

## R15

与“非 BO 处理”相比：
- BO 近似把电子与核耦合问题分层，计算成本大幅降低；
- 代价是忽略非绝热耦合，无法描述强锥形交叉/电荷转移耦合区的细节；
- 在本条目这种轻量分子与基态附近问题中，BO 近似通常是有效的一阶工作模型。

## R16

典型应用场景：
- 量子化学课程中解释“电子先求解、核运动后求解”的分层思想；
- 作为更复杂电子结构程序的教学参考基线；
- 在小体系上快速估计平衡核距、势阱深度和低振动态结构。

## R17

可扩展方向：
- 电子部分替换为更高精度方法（HF/DFT/CI）并重建 PES；
- 加入非绝热耦合项，升级到超越 BO 的核动力学；
- 从 1D 核距扩展到多维正则振动坐标；
- 用稀疏本征求解或 DVR/FEM 提升核方程规模能力；
- 引入真实分子数据并与实验振动频率对比。

## R18

`demo.py` 源码级算法流（8 步）：
1. `main` 创建 `BOConfig`，确定核距区间、网格密度、拟合窗口和输出能级数。  
2. `find_equilibrium` 调用 `scipy.optimize.minimize_scalar`，以 `bo_potential(R)` 为目标函数在有界区间内搜索最小值，得到 `R_eq` 与 `V_min`。  
3. `solve_nuclear_motion` 在 `R` 网格上显式计算 `V_BO(R)`，组装三对角离散哈密顿量：主对角 `1/(mu*dr^2)+V_i`，副对角 `-1/(2mu*dr^2)`。  
4. `solve_nuclear_motion` 再调用 `scipy.linalg.eigh_tridiagonal` 仅提取最低若干本征值/本征矢，得到振动能级 `E_v` 与基态波函数。  
5. `fit_local_quadratic` 选取 `R_eq ± window` 的局域点，用 `PolynomialFeatures(deg=2) + LinearRegression` 拟合 `V(R)`，解析得到力常数 `k=2*coef_x2`。  
6. `torch_force_and_curvature` 在 `R_eq` 处复写同一势能表达式，用 `torch.autograd.grad` 连续求一阶导和二阶导，得到力与曲率用于独立校验。  
7. `harmonic_observables` 依据 `omega=sqrt(k/mu)` 计算谐振频率和零点能，并与数值振动态 `E_v` 并列报告。  
8. `validate_results` 执行 7 项检查（平衡点区间、势阱深度、力近零、曲率正、能级递增等），最终输出 `Validation: PASS/FAIL` 并决定退出码。
