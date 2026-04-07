# Berlekamp算法

- UID: `MATH-0085`
- 学科: `数学`
- 分类: `计算代数`
- 源序号: `85`
- 目标目录: `Algorithms/数学-计算代数-0085-Berlekamp算法`

## R01

Berlekamp 算法用于在有限域 `GF(p)` 上分解一元多项式。它通过构造 Berlekamp 子代数（满足 `g(x)^p ≡ g(x) mod f(x)` 的多项式集合），把“因式分解”转化为“线性代数 + 最大公因子”问题。

## R02

适用场景：
- 需要在有限域上做多项式不可约分解（编码理论、密码学、计算代数系统）。
- 需要对中小规模 `GF(p)[x]` 问题给出可验证、可复现的分解结果。
- 作为 Cantor-Zassenhaus 等随机分解算法的基础对照实现。

## R03

核心思想：
- 对 `deg(f)=n`，构造 `n×n` 的 Berlekamp 矩阵 `Q`，其中第 `i` 列对应 `x^(p*i) mod f` 在基 `1,x,...,x^(n-1)` 下的系数。
- 求解线性方程 `(Q - I)v = 0` 的零空间，得到 Berlekamp 子代数的一组基。
- 若零空间维数为 1，则 `f` 在 `GF(p)` 上不可约（在平方自由前提下）。
- 若维数大于 1，取子代数中的非平凡元素 `g`，对 `a in GF(p)` 计算 `gcd(f, g-a)`，得到非平凡因子并递归分解。

## R04

输入输出约定（对应 `demo.py`）：
- 输入：
  - 素数 `p`（有限域 `GF(p)`）
  - 多项式系数数组（升幂）：`[c0, c1, ..., cn]` 表示 `c0 + c1*x + ... + cn*x^n`
- 输出：
  - 因子列表（每个因子都标准化为首一多项式）
  - Demo 中还会打印回乘校验结果，保证分解正确性

## R05

伪代码：

```text
BerlekampFactor(f, p):
  ensure p is prime
  f <- monic(f)
  if gcd(f, f') != 1: report "need square-free decomposition first"

  return FactorSquareFree(f, p)

FactorSquareFree(h, p):
  if deg(h) <= 1: return [h]
  B <- basis(nullspace(Q(h)-I))
  if dim(B) == 1: return [h]    # irreducible

  find nontrivial split d via gcd(h, g-a)
  return FactorSquareFree(d, p) + FactorSquareFree(h/d, p)
```

## R06

正确性要点：
- 对平方自由多项式 `f`，Berlekamp 子代数维数等于不可约因子个数。
- `(Q-I)v=0` 恰好刻画了 `g^p ≡ g (mod f)` 的系数条件，因此零空间基覆盖全部可用分裂方向。
- 若 `g` 不是常数，则存在 `a` 使 `gcd(f, g-a)` 产生真因子（非 `1` 且非 `f`）。
- 递归拆分并做首一化后，最终得到不可约因子乘积，回乘可恢复原多项式。

## R07

复杂度（`n = deg(f)`）：
- 构造 Berlekamp 矩阵：约 `O(n^3 log p)`（包含多项式幂模与模约简）。
- 高斯消元求零空间：`O(n^3)`。
- 每轮 `gcd` 与递归分解依赖因子结构，通常低于或同阶于上述主项。
- 总体可视作中小规模问题上的 `O(n^3)` 主导实现。

## R08

与其他分解路线对比：
- 与 Kronecker（整数系数枚举）相比：Berlekamp 更适合有限域场景，不走整数试因子。
- 与 Cantor-Zassenhaus 相比：Berlekamp 更“线代驱动”、结构清晰；Cantor-Zassenhaus 在大规模随机分解中常更快。
- 与黑箱 CAS 调用相比：本实现可直接看到每一步矩阵与 gcd 分裂逻辑，便于教学与调试。

## R09

数据结构设计：
- 多项式：`List[int]` 升幂系数，所有运算后都取模 `p` 并裁剪尾零。
- 矩阵：`List[List[int]]`，在 `GF(p)` 上做 Gauss-Jordan 消元。
- 因子工作栈：`todo`（待继续分解）与 `done`（已不可约或不可再分）。

## R10

边界与异常处理：
- `p` 非素数：抛出 `ValueError`。
- 零多项式：抛出 `ValueError`（分解不定义）。
- 非平方自由：检测 `gcd(f, f') != 1` 后抛出 `ValueError`，提示先做平方自由分解。
- Demo 固定样例运行，无交互输入。

## R11

本目录 MVP 实现说明：
- 仅使用 Python 标准库（`random`、`typing`），避免外部环境依赖。
- 实现了完整最小链路：
  - `GF(p)[x]` 基础运算（加减乘除、求导、gcd、幂模）
  - Berlekamp 矩阵构造
  - 模 `p` 零空间求解
  - 基于 `gcd(f, g-a)` 的递归拆分
- 采用固定随机种子 `random.Random(0)`，保证示例可复现。

## R12

运行方式：

```bash
python3 demo.py
```

脚本会自动执行 3 组样例：
- `GF(2)` 可约 6 次多项式
- `GF(5)` 可约 4 次多项式
- `GF(2)` 不可约 3 次多项式

## R13

预期输出结构：
- 每个 Case 都会打印：
  - 有限域 `GF(p)`
  - 输入多项式 `f(x)`
  - 分解得到的因子列表
  - `product of factors equals f(x) [OK]` 校验
- 末尾打印 `All demo cases finished successfully.` 表示所有样例通过。

## R14

常见实现错误：
- 把 `Q` 的列/行定义写反，导致 `(Q-I)v=0` 解空间错误。
- 多项式未做模 `p` 归一化，出现负系数或尾零污染。
- 忘记首一化（monic），导致分解结果比较和回乘校验混乱。
- 非平方自由输入直接递归，可能卡在无法分裂或产生错误结论。

## R15

最小测试清单：
- 可约样例：至少 2 组不同 `p` 的多项式可正确拆分并回乘。
- 不可约样例：输出单个因子（即自身首一化）。
- 鲁棒性：`p` 非素数、零多项式、非平方自由输入都应触发明确异常。
- 复现性：同一输入重复运行，因子集合一致。

## R16

可扩展方向：
- 增加平方自由分解（square-free decomposition），完整支持重复因子情形。
- 引入 Cantor-Zassenhaus 作为大规模分解加速后端。
- 优化幂模与矩阵部分（例如位运算优化 `GF(2)`，或稀疏矩阵策略）。
- 增加与 `sympy` 结果交叉验证的测试脚本。

## R17

局限与取舍：
- 当前 MVP 主要面向教学与验证，未做极致性能优化。
- 对大次数多项式，纯 Python 的 `O(n^3)` 线代会较慢。
- 非平方自由输入暂不在本脚本内自动分解。
- 采用小而透明的实现，优先保证可读性与可检查性。

## R18

源码级算法流程（本实现，无黑箱第三方）：
1. `berlekamp_factor` 检查 `p` 是否素数，并把输入多项式归一化为首一形式。  
2. 计算 `gcd(f, f')`，若不是 1 则判定为非平方自由并报错退出。  
3. `berlekamp_matrix` 构造 `Q-I`：第 `col` 列来自 `x^(p*col) mod f` 的系数向量。  
4. `nullspace_mod_p` 对 `Q-I` 做模 `p` 的 Gauss-Jordan 消元，求出零空间基。  
5. 若零空间维数为 1，当前多项式不可再分，直接作为不可约因子保存。  
6. 若维数大于 1，`split_factor` 从零空间基构造候选 `g`，遍历 `a in GF(p)` 计算 `d = gcd(f, g-a)`。  
7. 一旦找到 `0 < deg(d) < deg(f)`，就用多项式除法得到另一因子 `f/d`，并把两者压回待分解栈。  
8. 重复步骤 3-7，直到栈中都变成线性或不可继续分裂的多项式。  
9. 对最终因子做排序、打印，并在 `demo.py` 中回乘校验 `∏ factor_i == f`。  
