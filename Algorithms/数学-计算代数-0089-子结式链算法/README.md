# 子结式链算法

- UID: `MATH-0089`
- 学科: `数学`
- 分类: `计算代数`
- 源序号: `89`
- 目标目录: `Algorithms/数学-计算代数-0089-子结式链算法`

## R01

子结式链算法（Subresultant PRS）用于在不引入大量分式系数膨胀的前提下，构造两个一元多项式的“带控制缩放”的余式链，并据此得到 `gcd` 与结果式信息。

给定 `f,g ∈ K[x]`（本实现取 `K=Q`），链 `P0,P1,...,Pk,0` 的最后一个非零项与 `gcd(f,g)` 仅差一个可逆常数因子。对其首一化（monic）后即可得到标准 gcd。

本目录 MVP 重点是“可审计实现”：不调用 CAS 黑箱接口，直接实现伪除法与子结式 PRS 递推。

## R02

有限计算问题定义：

- 输入：两个一元多项式 `f(x), g(x)`（整系数，内部转 `Fraction` 精确计算）。
- 输出：
  - 子结式链 `P0, P1, ..., Pt`；
  - 首一化 gcd `gcd_monic`；
  - 统计信息（迭代次数、伪除法步数）。
- 判定标准：
  - `gcd_monic` 与预期 gcd 一致；
  - `gcd_monic | f` 且 `gcd_monic | g`（在 `Q[x]` 上余数为 0）。

## R03

背景与意义：

- 直接欧几里得除法在整系数环上会频繁引入分母，造成中间表达膨胀。
- 伪余式链（PRS）通过乘首项幂避免即时分式，子结式 PRS 再通过特定缩放因子控制增长。
- 子结式链是计算代数中 gcd/resultant 的经典工程基线，常用于符号计算内核和代数消元流程。

## R04

核心数学对象（本实现对应）：

1. 伪余式 `prem(A,B)`：反复执行 `R <- lc(B)*R - lc(R)*x^t*B`，直到 `deg(R) < deg(B)`。
2. 递推量：
   - `delta_i = deg(P_{i-1}) - deg(P_i)`
   - `alpha_i = lc(P_i)^(delta_i+1)`
   - `R_i = prem(alpha_i * P_{i-1}, P_i)`
3. 子结式缩放：
   - `beta_1 = (-1)^(delta_1+1)`
   - `beta_i = -lc(P_{i-1}) * psi_i^(delta_i), i>=2`
   - `psi_1 = -1`
   - `psi_{i+1} = (-lc(P_i))^(delta_i) / psi_i^(delta_i-1)`
4. 新项：`P_{i+1} = R_i / beta_i`。

## R05

`demo.py` 的数据结构：

- `Poly = list[Fraction]`：稠密表示（高次到低次）。
- `Case`：案例名、输入多项式、期望 gcd。
- `PRSStats`：
  - `iterations`：链递推轮数；
  - `pseudo_division_steps`：伪除法内层消元步数。

选择 `Fraction` 的目的是保证每一步严格精确，可用于自动断言验证。

## R06

示例说明（`nontrivial-gcd`）：

- `f = (x^2-1)(x-2) = x^3-2x^2-x+2`
- `g = (x^2-1)(x+3) = x^3+3x^2-x-3`
- 链中得到 `P2 = 5x^2-5`，首一化后即 `x^2-1`。

这体现了子结式链的核心作用：中间可能带常数缩放，但最终非零项与 gcd 同伴随类。

## R07

时间复杂度（单变量、朴素乘加实现）：

- 伪除法一次内循环是“首项消元”，若项数规模为 `n`，单步多项式运算约 `O(n)` 到 `O(n^2)`（视实现而定）。
- 一次 `prem` 需要若干首项消元，整体可近似为 `O(n^2)` 级。
- 整条 PRS 长度最多为次数下降链，近似 `O(deg(f)+deg(g))`。

本 MVP 面向教学与小规模验证，不追求大规模最优复杂度。

## R08

空间复杂度：

- 主体是链中各多项式的系数存储，约为 `O(sum_i deg(P_i))`。
- 伪除法只维护当前余式与若干临时项，不需要大型矩阵。
- 由于使用 `Fraction`，分子分母可能增长，实际内存还受整数位宽影响。

## R09

正确性要点：

1. 每轮通过 `prem(alpha_i*P_{i-1}, P_i)` 保证余式次数下降。  
2. `beta_i` 与 `psi_i` 的缩放保证链保持子结式 PRS 的等价关系。  
3. 链终止于零多项式时，最后非零项与 `gcd(f,g)` 只差可逆常数。  
4. 对最后非零项做 `make_monic` 得到规范 gcd。  
5. 脚本额外用 `field_remainder` 验证 gcd 同时整除 `f,g`。

## R10

边界与异常处理：

- `f=g=0`：抛出 `ValueError`（gcd 未定义）。
- 伪余式/域余式除数为零：抛出 `ZeroDivisionError`。
- 输入次数顺序：若 `deg(f)<deg(g)` 自动交换。
- 单个零多项式输入：链退化为单项，`gcd` 为另一多项式首一化。

## R11

运行方式：

```bash
cd Algorithms/数学-计算代数-0089-子结式链算法
python3 demo.py
```

特性：

- 无交互输入；
- 固定案例可复现；
- 内置断言失败会直接报错，便于自动化校验。

## R12

输出字段解读：

- `P0, P1, ...`：子结式链各项。
- `computed gcd (monic)`：由链最后非零项首一化得到。
- `expected gcd (monic)`：测试用例期望值。
- `stats.iterations`：外层递推轮数。
- `stats.pseudo_division_steps`：伪除法总首项消元次数。
- `checks: PASS`：`gcd` 数值一致且整除性验证通过。

## R13

最小测试集（`demo.py` 已覆盖）：

1. `nontrivial-gcd`  
   - 输入：`(x^2-1)(x-2)` 与 `(x^2-1)(x+3)`  
   - 期望 gcd：`x^2-1`
2. `coprime`  
   - 输入：`x^3+x+1` 与 `x^2-1`  
   - 期望 gcd：`1`

每个案例都检查：

- 结果 gcd 与期望一致；
- 结果 gcd 在 `Q[x]` 上整除两输入。

## R14

可扩展方向：

- 系数域从 `Q` 扩展到 `GF(p)`（把 `Fraction` 替换为模域运算）。
- 多项式表示从稠密改为稀疏字典，提升高次稀疏场景效率。
- 接入快速乘法（Karatsuba/FFT）降低大规模代价。
- 在链基础上继续实现结果式计算与平方因子分解流程。

## R15

与“直接 CAS 一行调用”对比：

- 本实现优势：
  - 每个缩放与余式步骤透明；
  - 易调试、易教学、易嵌入轻量项目；
  - 无第三方符号库依赖。
- 本实现限制：
  - 仅支持一元多项式；
  - 性能与鲁棒性不及成熟 CAS 内核；
  - 未实现高级优化策略。

定位是“诚实 MVP”：小而完整，可跑可验，可继续迭代。

## R16

应用场景：

- 一元多项式 gcd 与整除链分析；
- 结果式与消元算法教学；
- 代数系统中符号模块的 baseline 实现；
- 需要精确有理计算且不希望引入大型 CAS 依赖的脚本任务。

## R17

`demo.py` 函数映射：

- 基础运算：
  - `trim`, `degree`, `lc`
  - `poly_sub`, `poly_mul_scalar`, `poly_mul_xk`
- 核心算法：
  - `poly_pseudo_remainder`
  - `subresultant_chain`
  - `gcd_from_chain`
- 验证与展示：
  - `field_remainder`, `poly_divides`
  - `poly_to_str`, `run_case`, `main`

整体仅依赖 Python 标准库：`dataclasses`, `fractions`, `typing`。

## R18

源码级流程（对应 `demo.py`，8 步）：

1. `main` 构造两个固定测试用例，并逐个调用 `run_case`。  
2. `run_case` 把整系数数组转为 `Fraction` 多项式，调用 `subresultant_chain` 生成链。  
3. `subresultant_chain` 初始化 `P0,P1`，若次数顺序不满足则先交换。  
4. 每轮计算 `delta_i` 与 `alpha_i=lc(P_i)^(delta_i+1)`，再调用 `poly_pseudo_remainder(alpha_i*P_{i-1}, P_i)` 得到 `R_i`。  
5. 用 `beta_i`（首轮用 `(-1)^(delta_1+1)`，后续用 `-lc(P_{i-1})*psi_i^(delta_i)`）缩放 `R_i` 得到 `P_{i+1}`。  
6. 更新 `psi`：`psi_{i+1}=(-lc(P_i))^(delta_i)/psi_i^(delta_i-1)`，继续迭代直到生成零多项式。  
7. `gcd_from_chain` 取最后非零项并首一化，得到规范 gcd。  
8. `run_case` 用 `field_remainder` 校验 gcd 同时整除 `f,g`，输出链、gcd 与统计并断言通过。  

这 8 步即本 MVP 的完整算法执行路径，没有依赖外部黑箱符号求解器。
