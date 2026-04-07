# Mortar有限元

- UID: `MATH-0449`
- 学科: `数学`
- 分类: `数值分析`
- 源序号: `449`
- 目标目录: `Algorithms/数学-数值分析-0449-Mortar有限元`

## R01

Mortar 有限元（Mortar FEM）是一类用于**非匹配网格域分解**的弱耦合方法。它允许相邻子域在公共界面上使用不同网格，并通过拉格朗日乘子在弱意义下施加连续性约束，从而避免“必须节点对节点对齐”的严格要求。

本项目给出一个最小可运行 MVP：在二维 Poisson 方程上，把单位方形沿 `x=0.5` 分成两个子域，界面两侧 `y` 向网格数量不同（非匹配），再用 mortar 约束把两个子域拼接成一个整体问题。

## R02

目标 PDE：

- 区域：`Ω = (0,1) x (0,1)`
- 方程：`-Δu = f`
- 边界：`u = 0`（外边界齐次 Dirichlet）
- 验证解：`u*(x,y) = sin(πx) sin(πy)`
- 对应右端：`f = 2π² sin(πx) sin(πy)`

域分解：

- 左子域 `ΩL = (0,0.5) x (0,1)`
- 右子域 `ΩR = (0.5,1) x (0,1)`
- 界面 `Γ = {0.5} x (0,1)`

离散上，`ΩL` 与 `ΩR` 使用各自独立结构化网格，`Γ` 上节点不对齐。

## R03

Mortar 形式的离散鞍点系统可写为：

- 子域刚度方程：`AL uL = fL`，`AR uR = fR`
- 界面弱约束：`∫Γ λ (uL - uR) ds = 0`

离散后得到 KKT 系统：

```text
[ AL   0   BL^T ] [uL]   [fL]
[ 0   AR  -BR^T ] [uR] = [fR]
[ BL -BR    0   ] [λ ]   [0 ]
```

其中 `BL, BR` 是界面耦合矩阵，`λ` 是拉格朗日乘子自由度。

## R04

本实现使用的 mortar 策略：

1. 子域内部：标准 `Q1` 双线性有限元离散。
2. 界面乘子空间：**分片常数基**（按较粗界面剖分定义）。
3. 约束方式：在每个乘子区间上约束 `uL-uR` 的加权积分（平均意义连续）。
4. 结果特征：界面处不是逐点强连续，而是满足弱连续，数值上可通过 jump 范数检查。

## R05

适用场景：

- 多物理/多尺度耦合，局部需要加密网格而全局不想重剖分。
- 子域由不同求解器或不同网格生成器独立产生。
- 非匹配网格拼接（例如几何复杂接触面、局部补丁更新）。

不适合直接套用的场景：

- 只做单一规则网格，且接口天然对齐；此时常规 conforming FEM 更简单。

## R06

`demo.py` 输入输出约定：

- 输入：无命令行参数、无交互输入（脚本内部固定配置）。
- 内部配置：
  - 左网格：`nx=10, ny=8`
  - 右网格：`nx=7, ny=5`
  - 乘子段数：`n_lambda=min(8,5)=5`
- 输出：终端打印
  - 两个子域自由度数
  - `L2` 误差（左右子域）
  - 界面 jump 的 `L2` 与 `max`
  - `||lambda||_2`

## R07

复杂度（设总自由度约为 `N = NL + NR + Nλ`）：

- 装配：`O(N)` 到 `O(N * q)`（`q` 为单元积分点常数）。
- 线性求解：本 MVP 使用稠密 `numpy.linalg.solve`，复杂度 `O(N^3)`。
- 存储：`O(N^2)`（稠密矩阵）。

说明：工程版通常会改为稀疏存储 + Krylov/分块预条件，复杂度会显著改善。

## R08

稳定性与误差要点：

- Mortar 约束把“界面不匹配”转为“弱耦合积分方程”，避免硬对点带来的插值失配。
- 乘子空间选择影响稳定性（离散 inf-sup 条件）。本例用分片常数仅作教学 MVP。
- 误差来源包含：子域离散误差 + 界面投影/约束误差。
- 诊断建议：
  - 子域 `L2` 误差看 PDE 近似质量。
  - 界面 jump 范数看耦合质量。

## R09

实现流程（对应 `demo.py`）：

1. 定义子域网格与边界条件。
2. 在每个子域组装 `Q1` 刚度矩阵与载荷。
3. 在界面上构造 mortar 耦合矩阵 `BL, BR`。
4. 拼装 KKT 鞍点线性系统。
5. 求解 `(uL, uR, λ)`。
6. 回填到完整节点场。
7. 计算 `L2` 误差与界面 jump 指标。
8. 输出诊断结果。

## R10

伪代码：

```text
build left_mesh, right_mesh
AL, fL = assemble_subdomain(left_mesh)
AR, fR = assemble_subdomain(right_mesh)
BL = assemble_mortar(left_mesh, interface=right_side)
BR = assemble_mortar(right_mesh, interface=left_side)

KKT = [[AL, 0,  BL^T],
       [0,  AR, -BR^T],
       [BL, -BR, 0   ]]
rhs = [fL, fR, 0]

solve KKT * [uL, uR, lambda] = rhs

evaluate L2 error on each subdomain
evaluate interface jump norms
print diagnostics
```

## R11

运行方式：

```bash
python3 Algorithms/数学-数值分析-0449-Mortar有限元/demo.py
```

环境依赖：

- Python 3
- `numpy`

本 MVP 不依赖 `scipy`，便于在最小环境直接运行。

## R12

关键函数说明：

- `assemble_subdomain_system`：子域 `Q1` 单元积分与全局装配。
- `assemble_mortar_matrix`：在界面上按乘子分段做高斯积分，构造 `B`。
- `trace_value`：在界面任意 `y` 上计算离散迹值，供 jump 评估。
- `compute_l2_error`：对照解析解做数值积分误差。
- `compute_interface_jump_l2`：统计弱耦合后界面不连续程度。

## R13

一次实测输出（当前仓库环境，`2026-04-07`）：

```text
Mortar FEM MVP (2D Poisson, nonmatching interface grids)
left mesh  : nx=10, ny=8, free dof=70
right mesh : nx=7, ny=5, free dof=28
lambda dof : 5
L2 error (left)  = 4.032902e-03
L2 error (right) = 5.356069e-03
interface jump L2  = 1.171564e-02
interface jump max = 3.228110e-02
||lambda||_2       = 3.544814e-02
```

解读：

- 两侧误差处于 `1e-3` 到 `1e-2` 量级，符合该网格尺度下预期。
- jump 不为 0，说明是“弱连续”而非“强逐点连续”，符合 mortar 机制。

## R14

当前 MVP 的边界与局限：

- 乘子空间为分片常数，精度与稳定性不是最强配置。
- 使用稠密线性代数，只适合小规模教学示例。
- 几何限定为规则矩形和直界面，未覆盖曲线界面与复杂网格。

## R15

与常见替代方法比较：

- 对齐网格 conforming FEM：
  - 优点：系统更简单。
  - 缺点：要求网格匹配，局部重构成本高。
- Nitsche 耦合：
  - 优点：不引入额外乘子未知量。
  - 缺点：需要罚参数调参，稳定性与一致性平衡更敏感。
- Mortar FEM：
  - 优点：对非匹配接口自然、理论成熟。
  - 缺点：形成鞍点系统，求解器与空间配对设计更复杂。

## R16

工程实现建议：

- 规模提升时切换到稀疏矩阵与迭代法（MINRES/GMRES + 分块预条件）。
- 为 `BL/BR` 装配加入缓存与矢量化，减少界面积分重复开销。
- 做乘子空间/主从侧敏感性测试，检查 inf-sup 健壮性。
- 增加网格收敛实验（h-refinement）验证阶数。

## R17

可扩展方向：

- 从结构化 `Q1` 扩展到非结构三角形 `P1/P2`。
- 从标量 Poisson 扩展到弹性力学、Stokes 或多物理耦合。
- 引入双重 mortar、Nitsche-mortar 混合等高级接口策略。
- 并行化子域装配与分块求解（Schur complement / FETI 思路）。

## R18

`demo.py` 的源码级算法流（8 步）：

1. `StructuredSubdomain` 定义每个子域的网格拓扑、坐标映射、边界判定。  
2. `build_free_dof_map` 过滤 Dirichlet 节点，构建自由度映射。  
3. `assemble_subdomain_system` 在每个单元用 `2x2` 高斯积分：
   - 计算 `Q1` 形函数与雅可比；
   - 形成局部刚度 `ke` 与载荷 `fe`；
   - 汇总到子域全局矩阵 `A` 和向量 `b`。  
4. `assemble_mortar_matrix` 在界面上按乘子分段积分，得到 `BL/BR`：
   - 采样界面点 `y`；
   - 找到其所在界面边段；
   - 计算界面迹基函数值并累积积分。  
5. 在 `main` 中按 KKT 结构拼装大矩阵：
   - 左上角放 `AL/AR`；
   - 右上/左下放 `±B^T` 与 `±B`；
   - 乘子块为零矩阵。  
6. 调用 `numpy.linalg.solve` 一次解出 `(uL, uR, λ)`。  
7. `expand_to_full` 把自由度解回填到完整节点数组。  
8. `compute_l2_error` 与 `compute_interface_jump_l2` 计算误差和界面 jump，并打印结果。

这 8 步对应 mortar 方法从“子域离散”到“弱界面耦合”再到“诊断验证”的完整闭环。
