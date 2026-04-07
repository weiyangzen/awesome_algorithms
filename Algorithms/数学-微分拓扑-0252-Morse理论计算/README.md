# Morse理论计算

- UID: `MATH-0252`
- 学科: `数学`
- 分类: `微分拓扑`
- 源序号: `252`
- 目标目录: `Algorithms/数学-微分拓扑-0252-Morse理论计算`

## R01

Morse 理论把“光滑函数的临界点结构”与“流形拓扑不变量”联系起来。核心思想是：
- 选一个 Morse 函数（所有临界点都非退化）。
- 计算每个临界点的 Morse 指数（Hessian 负特征值个数）。
- 统计各指数的临界点数量 `m_k`，并与 Betti 数 `b_k` 比较。

本目录的 MVP 选用二维环面 `T^2` 上的标准函数，完成从临界点求解到 Morse 不等式验证的全流程计算。

## R02

问题设定：在
\[
T^2=(\mathbb{R}/2\pi\mathbb{Z})^2
\]
上定义函数
\[
f(x,y)=\cos x+\cos y.
\]
这是经典可计算的 Morse 函数候选：定义简单、临界点有限、Hessian 易分析，非常适合做最小可运行实现。

## R03

临界点条件来自梯度为零：
\[
\nabla f(x,y)=(-\sin x,-\sin y)=(0,0)
\Rightarrow
\sin x=0,\ \sin y=0.
\]
在 `T^2` 的基本域 `[0,2\pi)\times[0,2\pi)` 内，可得候选临界点坐标来自 `{0,\pi}\times{0,\pi}`。

`demo.py` 并没有直接硬编码结论，而是用多初值数值求根（`scipy.optimize.root`）去恢复这些点。

## R04

Hessian 为
\[
H_f(x,y)=
\begin{bmatrix}
-\cos x & 0 \\
0 & -\cos y
\end{bmatrix}.
\]
在临界点处，若 Hessian 特征值都非零，则该点非退化，函数为 Morse 函数。当前函数在 4 个临界点处特征值均为 `\pm 1`，因此全部非退化。

## R05

Morse 指数定义为 Hessian 的负特征值个数：
- `index=0`：局部极小值
- `index=1`：鞍点
- `index=2`：局部极大值

对于 `f(x,y)=cos x + cos y`，理论上应得到：
- 一个 `index=0`
- 两个 `index=1`
- 一个 `index=2`
即 `m_0=1, m_1=2, m_2=1`。

## R06

环面 `T^2` 的 Betti 数为：
\[
b_0=1,\quad b_1=2,\quad b_2=1.
\]
Morse 弱不等式要求：
\[
m_k \ge b_k.
\]
本例中恰好 `m_k=b_k`，因此这是一个“perfect Morse function”的典型示例。

## R07

除了弱不等式，还检查强不等式（截断交错和）：
\[
\sum_{k=0}^{p}(-1)^{p-k}m_k \ge
\sum_{k=0}^{p}(-1)^{p-k}b_k,
\quad p=0,1,2.
\]
脚本会逐项输出 `lhs/rhs/holds`，避免只给结论不展示检验过程。

## R08

欧拉示性数一致性也会被验证：
\[
\chi(T^2)=\sum_k(-1)^k b_k=0,
\quad
\sum_k(-1)^k m_k=0.
\]
这一步是对临界点计数与指数分类的额外 sanity check。

## R09

伪代码：

```text
input: Morse function f on T^2
build seeds on [0,2pi)^2 grid
for each seed:
    solve grad(f)=0 by root solver
    normalize point into [0,2pi)
    if ||grad|| < tol: collect
remove duplicates
for each critical point:
    compute Hessian eigenvalues
    if any eigenvalue near 0: raise error (non-Morse)
    index = number of negative eigenvalues
count m_k by index
compare m_k with Betti numbers of T^2
check weak/strong inequalities and Euler characteristic
print tables
```

## R10

`demo.py` 默认配置：
- 多初值网格：`grid_size=11`（121 个种子）
- 梯度阈值：`grad_tol=1e-8`
- 非退化阈值：`eig_tol=1e-10`

这些参数足以在很短时间内稳定恢复 4 个临界点，并避免数值边界噪声导致误判。

## R11

输出数据结构：
- 临界点表（`pandas.DataFrame`）：`x, y, f(x,y), index, lambda_min, lambda_max`
- 计数字典：`m_0,m_1,m_2`
- 不等式表（`DataFrame`）：`kind, k, lhs, rhs, holds`
- 欧拉示性数检查：`chi_from_morse`, `chi_from_betti`

这样既有机器可读结果，也有直接可审阅文本。

## R12

实现函数分工：
- `gradient / hessian / morse_function`：数学定义
- `find_critical_points`：多初值求根与周期归一化
- `classify_critical_points`：特征值分解与指数判定
- `morse_counts`：按指数统计
- `check_morse_inequalities`：弱/强不等式逐条构造
- `main`：流程编排与结果打印

函数边界清晰，便于后续替换函数或流形参数化。

## R13

复杂度（设初值数为 `S`，每次求根平均迭代步数为 `I`，维度固定为 2）：
- 求根阶段约 `O(S*I)`。
- Hessian 特征分解在 2x2 情况是常数开销，总计 `O(C)`（`C` 为临界点个数）。
- 去重对当前小规模问题可视为 `O(C^2)`，但 `C` 很小。

整体运行成本低，适合作为教学与验证原型。

## R14

运行方式：

```bash
python3 demo.py
```

脚本无交互输入，单次执行即可输出完整检验报告。

## R15

预期结果特征：
- 临界点数量应为 `4`
- 指数分布应为 `m_0=1, m_1=2, m_2=1`
- 弱不等式与强不等式的 `holds` 都应为 `True`
- 欧拉示性数应满足 `chi_from_morse == chi_from_betti == 0`

若不满足，通常意味着求根参数、周期归一化或 Hessian 判断实现有误。

## R16

常见错误与规避：
- 把 `T^2` 当作普通平面，忘记对根做 `mod 2pi` 归一化，导致重复点统计错误。
- 只检查 `result.success` 不检查 `||grad||`，可能接收伪收敛点。
- 不处理近零特征值，误把退化点当作 Morse 临界点。
- 只验证弱不等式，不验证强不等式和欧拉一致性，难以及时发现计数错误。

本实现分别用归一化、梯度阈值、`eig_tol` 和多重校验覆盖这些风险。

## R17

可扩展方向：
- 换成其他流形/函数（如球面上的高度函数）并复用同一框架。
- 引入梯度流离散，构造 Morse-Smale 复形的边界算子近似。
- 对随机扰动函数做 Monte Carlo 统计，观察 `m_k` 分布稳定性。
- 用符号工具（如 `sympy`）与数值结果交叉验证临界点和 Hessian。

## R18

`demo.py` 源码级算法流程（8 步）：
1. 在 `main` 中调用 `find_critical_points(grid_size=11)`，对 `[0,2pi)^2` 生成 121 个初值种子。
2. 对每个种子使用 `scipy.optimize.root` 求解 `gradient(z)=0`；若 SciPy 不可用，则回退到解析临界点集合，保证脚本可运行。
3. 对每个求根结果调用 `normalize_point`，把坐标映射回基本域，并把靠近 `0/pi/2pi` 的值吸附到标准角，消除周期边界抖动。
4. 用 `||gradient(p)|| <= grad_tol` 再过滤一次，只保留真正满足驻点条件的候选点。
5. 调用 `deduplicate_points` 去掉重复临界点，得到有限唯一集合。
6. 在 `classify_critical_points` 中对每个点计算 Hessian 特征值；若最小绝对特征值 `<= eig_tol` 则判为退化并报错，否则按负特征值个数赋予 Morse 指数。
7. 用 `morse_counts` 统计 `m_0,m_1,m_2`，并在 `check_morse_inequalities` 中逐条构造弱/强不等式与 `holds` 布尔列。
8. 最后计算 `chi_from_morse` 与 `chi_from_betti`，并打印临界点表、Morse 多项式 `M(t)`、不等式表和欧拉一致性结论。
