# 勒让德多项式 (Legendre Polynomials)

- UID: `PHYS-0143`
- 学科: `物理`
- 分类: `数学物理`
- 源序号: `143`
- 目标目录: `Algorithms/物理-数学物理-0143-勒让德多项式_(Legendre_Polynomials)`

## R01

勒让德多项式 `P_n(x)` 是区间 `[-1, 1]` 上的一组正交多项式，满足勒让德微分方程：

`(1-x^2)y'' - 2xy' + n(n+1)y = 0`

它是球坐标分离变量中最核心的特殊函数之一，既有清晰解析结构，也非常适合做数值算法入门和验证基准。

## R02

在数学物理中，勒让德多项式常见于：

- 拉普拉斯方程在球坐标下的角向解；
- 万有引力势与电势的多极展开；
- 球谐函数 `Y_l^m` 的 `m=0` 特例；
- 光散射、量子角动量相关的解析推导。

因此它既是理论工具，也是工程计算中常见的“基函数模块”。

## R03

本条目的最小数学定义采用三条：

1. 初值：`P_0(x)=1`, `P_1(x)=x`
2. 三项递推：

`P_n(x) = ((2n-1)xP_{n-1}(x) - (n-1)P_{n-2}(x))/n`

3. Rodrigues 公式：

`P_n(x) = 1/(2^n n!) * d^n/dx^n[(x^2-1)^n]`

MVP 中递推作为主计算路径，Rodrigues 作为独立交叉验证路径。

## R04

关键性质（用于验收）如下：

- 端点值：`P_n(1)=1`, `P_n(-1)=(-1)^n`；
- 正交性：

`∫_{-1}^{1} P_m(x)P_n(x) dx = 0 (m!=n)`

`∫_{-1}^{1} P_n(x)^2 dx = 2/(2n+1)`

- `P_n` 在 `(-1,1)` 内有 `n` 个互异实根。

这些性质都在 `demo.py` 中有对应的数值检查项。

## R05

本 MVP 的目标不是“只调一个库函数拿答案”，而是建立可审计闭环：

- 用递推显式构造 `P_0..P_n`；
- 用 Rodrigues 公式单独重建；
- 用 Gauss-Legendre 积分验证正交归一关系；
- 用微分方程残差验证方程层正确性；
- 用截断勒让德级数近似 `exp(x)` 演示实际数值用途。

## R06

`demo.py` 输入输出约定：

- 输入：无命令行参数，脚本内固定配置（最高阶数、积分阶数、采样网格）；
- 输出：
1. 每个阶数的误差表（递推/ Rodrigues/ ODE/ 根）；
2. 正交性误差摘要；
3. `exp(x)` 的勒让德展开系数与重构误差；
4. 通过断言后打印 `All checks passed.`。

脚本可直接用于自动化验证，无交互步骤。

## R07

算法主流程（高层）为：

1. 在网格上用三项递推同时计算 `P_0..P_N`；
2. 对每个 `n` 用 Rodrigues 公式构造多项式并对比 `scipy.special.eval_legendre`；
3. 计算勒让德方程残差 `((1-x^2)P_n''-2xP_n'+n(n+1)P_n)`；
4. 用 `roots_legendre` 提供的根检查 `P_n(root)≈0`；
5. 用 Gauss-Legendre 求积构造 Gram 矩阵检查正交性；
6. 计算 `exp(x)` 的截断勒让德系数并重构函数；
7. 输出误差与断言结果。

## R08

设最高阶为 `N`，采样点数为 `M`，积分节点数为 `Q`：

- 递推评估：`O(NM)`；
- Rodrigues 逐阶构造+评估（本实现用于小 `N` 验证）：约 `O(N^2 + NM)`；
- 正交 Gram 计算：`O(NQ + N^2Q)`；
- 投影系数计算：`O(NQ)`；
- 空间复杂度：`O(NM)`。

对教学/算法验证规模（`N<=几十`）完全可接受。

## R09

数值稳定性策略：

- 主路径使用三项递推，避免高阶幂展开直接计算带来的巨大消去误差；
- 正交积分使用 Gauss-Legendre 节点，避免普通等距积分在高阶下精度下降；
- ODE 残差在 `[-0.98,0.98]` 内评估，减少端点附近舍入放大影响；
- 所有计算统一 `float64`。

## R10

MVP 技术栈：

- `numpy`：向量化计算、递推、求积节点；
- `pandas`：结果表格化输出；
- `scipy.special`：`eval_legendre` 与 `roots_legendre` 作为对照验证；
- `math`/标准库：阶乘与基本结构。

核心算法（递推、Rodrigues、投影、误差验收）在源码中显式实现，不依赖黑箱流程。

## R11

运行方式：

```bash
cd Algorithms/物理-数学物理-0143-勒让德多项式_(Legendre_Polynomials)
uv run python demo.py
```

预期结果是打印三段报告并以 `All checks passed.` 结束。

## R12

主要输出字段含义：

- `n`: 多项式阶数；
- `max|rec-scipy|`: 递推结果与 `eval_legendre` 的最大差值；
- `max|rodrigues-scipy|`: Rodrigues 结果与 `eval_legendre` 的最大差值；
- `max|ODE residual|`: 勒让德微分方程残差最大值；
- `max|P_n(root)|`: 在理论根处的函数值残差；
- `max offdiag integral error`: 正交性非对角项误差；
- `max diag integral error`: 对角归一值误差；
- `max abs reconstruction error`: 截断展开重构 `exp(x)` 的最大绝对误差；
- `L2 reconstruction error`: `[-1,1]` 上的二范数误差。

## R13

`demo.py` 内置断言包含以下正确性门槛：

1. 递推与 SciPy 基准一致（`<1e-11`）；
2. Rodrigues 与 SciPy 一致（`<1e-10`）；
3. 微分方程残差足够小（`<1e-9`）；
4. 根位置函数值足够接近 0（`<1e-12`）；
5. 正交 Gram 的非对角/对角误差满足阈值（`<1e-12`）；
6. `exp(x)` 截断展开达到给定误差上限。

全部通过才输出成功标记。

## R14

当前实现局限：

- 只覆盖普通勒让德多项式 `P_n`，未扩展到陪伴勒让德函数 `P_l^m`；
- 主要面向中低阶演示，高阶极限精度与超大规模性能未专门优化；
- 未包含带权最小二乘噪声场景中的鲁棒投影策略。

## R15

可扩展方向：

- 扩展到陪伴勒让德函数并连接球谐函数；
- 引入 Clenshaw 递推，提高高阶级数求值稳定性；
- 加入 `scipy.sparse` 或并行策略做大规模谱方法原型；
- 加入与 Chebyshev/Jacobi 基函数的误差与条件数对比。

## R16

典型应用场景：

- 电磁/引力势的球谐与多极展开；
- 谱方法求解边值问题（基函数展开）；
- 高斯-勒让德积分节点与权重的理论基础；
- 量子力学角向方程教学与验证。

## R17

本目录 MVP 功能清单：

- [x] 递推生成 `P_n(x)`；
- [x] Rodrigues 公式独立重建；
- [x] 与 `scipy.special.eval_legendre` 数值交叉验证；
- [x] 勒让德 ODE 残差验证；
- [x] 正交性与归一积分验证；
- [x] 根性质验证；
- [x] `exp(x)` 截断展开演示；
- [x] 无交互、可直接 `uv run python demo.py` 执行。

## R18

`demo.py` 的源码级算法流程（9 步）：

1. `legendre_values_recurrence` 按 `P_0,P_1` 初值和三项递推一次性构造 `P_0..P_N`。  
2. `legendre_poly_rodrigues` 通过 Rodrigues 公式把 `P_n` 转成幂基 `poly1d` 多项式。  
3. 主循环中将递推值与 `scipy.special.eval_legendre` 对比，形成第一组一致性误差。  
4. 同一循环中把 Rodrigues 结果与 SciPy 基准对比，形成第二组独立误差。  
5. `legendre_ode_residual` 计算 `(1-x^2)P_n''-2xP_n'+n(n+1)P_n`，检查方程层残差。  
6. 利用 `roots_legendre` 给出的理论根，回代递推结果检查 `P_n(root)` 是否接近零。  
7. `orthogonality_errors` 使用 Gauss-Legendre 求积构造 Gram 矩阵，提取正交性误差和归一误差。  
8. `legendre_projection_coefficients` + `evaluate_legendre_series` 计算 `exp(x)` 的截断勒让德展开并重构，输出 `max/L2` 误差。  
9. `main` 汇总打印表格与指标，并用断言做最终验收，全部通过后输出 `All checks passed.`。

其中第三方库仅提供基础数值算子与对照基准，核心算法路径与验证逻辑都在源码中可追踪。
