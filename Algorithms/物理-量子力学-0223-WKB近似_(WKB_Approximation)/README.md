# WKB近似 (WKB Approximation)

- UID: `PHYS-0222`
- 学科: `物理`
- 分类: `量子力学`
- 源序号: `223`
- 目标目录: `Algorithms/物理-量子力学-0223-WKB近似_(WKB_Approximation)`

## R01

WKB 近似（Wentzel-Kramers-Brillouin）是半经典方法，用于近似求解一维定态薛定谔方程。它在势能缓慢变化、局域德布罗意波长变化不剧烈时有效，可给出束缚态量子化条件和隧穿指数估计。

## R02

典型适用场景：
- 高量子数束缚态的能级近似（如非简谐势）。
- 势垒隧穿概率的指数级估计。
- 需要在“解析洞察”和“数值代价”间做平衡的物理建模。

## R03

核心思想：
- 设波函数形如 `psi(x)=A(x)exp(i S(x)/hbar)`，把相位 `S` 做 `hbar` 展开。
- 0 阶主导项给出经典动量 `p(x)=sqrt(2m(E-V(x)))`。
- 在允许区得到振荡解，在禁阻区得到指数衰减/增长解。
- 通过转折点连接公式，得到 Bohr-Sommerfeld 量子化条件。

## R04

本目录 `demo.py` 的输入输出约定：
- 输入：无交互输入；脚本内部固定参数 `hbar=m=lambda=1`，势能 `V(x)=x^4/4`。
- 输出：
  - 有限差分数值本征能 `E_numeric`（作为参考“真值”）。
  - WKB 量子化能级 `E_WKB`。
  - 每个能级的绝对误差与相对误差百分比。

## R05

伪代码（本实现）：

```text
for n in 0..N-1:
    define target = pi*hbar*(n + 1/2)
    solve I(E) - target = 0 by brentq
    where I(E) = integral_{x1(E)}^{x2(E)} sqrt(2m(E-V(x))) dx

build tridiagonal Hamiltonian H by finite difference
solve lowest N eigenvalues as E_numeric
report E_numeric vs E_WKB and errors
```

## R06

正确性要点：
- Bohr-Sommerfeld 条件 `∫ p dx = pi*hbar(n+1/2)`来自 WKB 在双转折点问题的匹配条件。
- 对给定 `n`，`I(E)` 随 `E` 单调上升，故根查找（`brentq`）有唯一物理解。
- 有限差分离散哈密顿量是实对称三对角矩阵，最低本征值可稳定提取并用于交叉验证 WKB 结果。

## R07

复杂度（令网格内点数为 `M`、输出能级数为 `K`）：
- WKB 部分：每个能级一次一维求根，单次成本约为“若干次一维积分”，总体约 `O(K * C_quad * C_root)`。
- 数值本征值：三对角本征求解最低 `K` 个特征值，主成本近似 `O(M*K)` 到 `O(M^2)`（取决于底层 LAPACK 路径和选择模式）。
- 空间复杂度：`O(M)`（三对角存储）。

## R08

与其它方法对比：
- 相比纯数值求解（射击法/全矩阵本征），WKB 直接给出“能级如何随量子数变化”的结构性解释。
- 相比纯解析近似，WKB 在非简谐势（如四次势）上仍具可操作性。
- 对低量子数和转折点附近，WKB 精度通常不如高激发态；因此本 MVP 用数值本征值进行误差校核。

## R09

数据结构与函数划分：
- `quartic_potential(x)`：势能函数。
- `wkb_action(E)`：动作积分 `I(E)`。
- `wkb_energy_level(n)`：一维根求解得到 WKB 能级。
- `numerical_levels(...)`：构造三对角哈密顿量并求最低若干本征值。
- `main()`：汇总、打印比较表。

## R10

边界与异常处理：
- `n < 0` 会抛 `ValueError`。
- `n_levels <= 0` 或网格过小会抛 `ValueError`。
- 若根区间扩展失败（极端参数导致无法包根）会抛 `RuntimeError`。
- 动作积分中对数值误差导致的微小负值做了截断保护（`remaining <= 0` 时返回 0）。

## R11

MVP 选型说明：
- 使用 `numpy`：数组与数值运算。
- 使用 `scipy.integrate.quad`：计算 WKB 动作积分。
- 使用 `scipy.optimize.brentq`：求解量子化方程。
- 使用 `scipy.linalg.eigh_tridiagonal`：求三对角哈密顿量低能级。

该组合小而完整，既展示半经典算法，也提供数值基准，不依赖重型框架。

## R12

运行方式（在本目录下）：

```bash
uv run python demo.py
```

脚本不需要任何交互输入。

## R13

预期输出特征：
- 打印 6 个低能级（`n=0..5`）的 `E_numeric` 与 `E_WKB` 对比表。
- 输出每行误差 `abs_err` 与 `rel_err(%)`。
- 最后给出平均相对误差和最大相对误差。
- 一般会看到：高能级（较大 `n`）误差小于低能级，符合半经典近似直觉。

## R14

常见实现错误：
- 把量子化条件写成 `2*pi*(n+1/2)`（多一倍）。
- 忘记在根查找前做“包根区间扩展”，导致 `brentq` 失败。
- 有限差分离散中二阶导符号写反，导致负能谱或不合理结果。
- 网格太粗或区间太短，使数值“真值”本身不稳定。

## R15

最小测试清单：
- 运行性：`uv run python demo.py` 可直接完成，无交互。
- 单调性：`E_numeric` 与 `E_WKB` 均应随 `n` 严格上升。
- 物理性：所有输出能级应为正值。
- 近似性：相对误差应整体处于可接受范围，且高 `n` 趋势通常更好。

## R16

可扩展方向：
- 增加“势垒隧穿”案例，验证 `T ~ exp(-2/hbar * integral kappa dx)`。
- 引入转折点统一近似（Airy matching）改善低能级附近精度。
- 扩展到位置依赖质量或径向方程的 Langer 修正。
- 把误差随 `n` 的标度关系做拟合与可视化分析。

## R17

局限与取舍：
- WKB 本质是渐近近似，低量子数和急剧变化势能下会偏差较大。
- 转折点邻域需匹配处理，单一区域公式不可直接硬套。
- 本 MVP 选择了四次势和一维静态问题，强调可读与可运行，不覆盖多维/时变情形。

## R18

源码级算法流程（本实现，非黑箱分解）：
1. 在 `quartic_potential` 定义 `V(x)=x^4/4`，给 WKB 和数值本征值模块共用同一势能。  
2. 对每个量子数 `n`，在 `wkb_energy_level` 里先构造目标值 `pi*hbar*(n+1/2)`。  
3. 调 `wkb_action(E)` 计算动作积分：先由 `E=V(x)` 得转折点 `x_turn`，再用 `quad` 计算 `∫_{-x_turn}^{x_turn} sqrt(2m(E-V)) dx`。  
4. 用 `brentq` 解方程 `I(E)-target=0`；通过倍增上界 `hi` 保证根被区间夹住后再收敛。  
5. 在 `numerical_levels` 中构造一维网格，对 `-(hbar^2/2m)d2/dx2 + V(x)` 做中心差分离散，得到三对角矩阵的主对角与副对角。  
6. 调 `eigh_tridiagonal(..., select="i")` 仅提取最低 `K` 个本征值，作为数值参考能级 `E_numeric`。  
7. 在 `main` 中逐级比较 `E_WKB` 和 `E_numeric`，计算 `abs_err` 与 `rel_err`，并检查能级严格单调上升。  
8. 打印结果表与总体误差统计，形成“半经典近似 + 数值校核”的最小闭环。  
