# 快速幂算法/二进制幂 (Binary Exponentiation)

- UID: `MATH-0015`
- 学科: `数学`
- 分类: `数论`
- 源序号: `15`
- 目标目录: `Algorithms/数学-数论-0015-快速幂算法／二进制幂_(Binary_Exponentiation)`

## R01

快速幂（Binary Exponentiation）用于高效计算 `a^n`，核心目标是把指数 `n` 的线性次乘法（`O(n)`）降为对数级（`O(log n)`）。

## R02

核心思想是利用指数的二进制展开：
- 若 `n` 的某一二进制位为 `1`，就把当前幂因子乘入结果。
- 每处理一位，底数自乘一次（`a <- a*a`），指数右移一位（`n >>= 1`）。

## R03

数学分解：
`n = b0*2^0 + b1*2^1 + ... + bk*2^k`，其中 `bi ∈ {0,1}`。
因此：
`a^n = Π(a^(2^i))^bi`。
算法只需依次构造 `a^(2^i)` 并在 `bi=1` 时乘入。

## R04

输入与输出定义：
- 输入：整数 `base`，非负整数 `exp`，可选正整数 `mod`。
- 输出：
  - 无模版本返回 `base^exp`。
  - 模幂版本返回 `(base^exp) mod mod`。

## R05

前置条件与约束：
- `exp` 必须是非负整数。
- 若使用模幂，`mod` 必须为正整数。
- 本 MVP 专注整数幂；浮点幂、分数幂不在范围内。

## R06

伪代码：

```text
function binpow(base, exp):
    result = 1
    while exp > 0:
        if exp is odd:
            result = result * base
        base = base * base
        exp = exp // 2
    return result

function mod_binpow(base, exp, mod):
    base = base % mod
    result = 1 % mod
    while exp > 0:
        if exp is odd:
            result = (result * base) % mod
        base = (base * base) % mod
        exp = exp // 2
    return result
```

## R07

正确性要点（循环不变式）：
- 设初始值为 `A`、`N`，循环任意时刻保持：
  `result * base^exp = A^N`（模幂版本是在模 `mod` 意义下成立）。
- 当 `exp` 为奇数时，把一个 `base` 从 `base^exp`“移”到 `result`，不变式仍成立。
- 然后执行 `base <- base^2` 与 `exp <- floor(exp/2)`，等价于把剩余指数按二进制右移，仍保持不变式。
- 当 `exp = 0`，有 `base^0 = 1`，故 `result = A^N`，算法正确。

## R08

复杂度分析：
- 时间复杂度：`O(log exp)`。
- 空间复杂度：`O(1)`（不计大整数本身存储）。

## R09

与朴素连乘对比：
- 朴素法：需要 `exp` 次乘法。
- 快速幂：只需约 `2*log2(exp)` 量级的乘法/取模操作。
- 当 `exp` 很大（如 `10^9` 量级）时，性能差异非常显著。

## R10

典型应用场景：
- 模幂计算：`a^n mod m`（密码学、同余方程、组合计数）。
- 矩阵快速幂与线性递推的基础思想（将“平方-乘”推广到矩阵乘法）。
- 需要高频幂运算的竞赛与工程代码。

## R11

边界情况：
- `exp = 0` 时结果应为 `1`（模幂时为 `1 % mod`）。
- `base = 0, exp > 0` 时结果为 `0`。
- `base = 0, exp = 0` 在多数编程语言中约定为 `1`，本实现遵循该约定。
- `mod = 1` 时模幂结果恒为 `0`。

## R12

本目录 `demo.py` 的实现内容：
- `binary_exponentiation(base, exp)`：整数快速幂。
- `mod_binary_exponentiation(base, exp, mod)`：整数模幂。
- `run_demo_cases()`：自动跑样例并对照 Python 内置 `pow` 做正确性校验。

## R13

运行方式：

```bash
python3 demo.py
```

脚本无交互输入，直接输出每个样例与校验结论。

## R14

MVP 输出解读：
- 每个样例会显示输入参数、算法结果、内置 `pow` 参考结果。
- 若二者一致，标记为 `PASS`；否则标记为 `FAIL`。
- 全部通过后会输出汇总成功信息。

## R15

本实现覆盖的测试点：
- 小规模基础样例：如 `2^10 = 1024`。
- 大指数样例：验证 `O(log n)` 路径。
- 含负底数样例：验证奇偶次幂符号。
- 模幂样例：含大指数与 `mod=1` 的边界值。

## R16

常见错误与规避：
- 忘记在循环里“先判断奇偶再平方右移”，会导致结果错位。
- 模幂版本若不在每步取模，数值会膨胀并拖慢性能。
- 没有限制 `exp >= 0`，会引入非终止或语义不清问题。

## R17

工程实践建议：
- Python 中优先使用内置 `pow(base, exp, mod)` 作为生产实现（C 层优化）。
- 教学、面试、算法库实现中建议手写快速幂，便于迁移到不支持三参 `pow` 的语言。
- 对超大整数运算应关注乘法成本；快速幂减少的是乘法次数，不是单次乘法代价。

## R18

`demo.py` 中手写模幂流程可拆为 8 个具体步骤：
1. 参数检查：`exp >= 0` 且 `mod > 0`。
2. 初始化：`result = 1 % mod`，`base = base % mod`。
3. 进入循环：当 `exp > 0` 持续迭代。
4. 检查最低位：`exp & 1` 判断当前二进制位是否为 `1`。
5. 若最低位为 `1`：执行 `result = (result * base) % mod`。
6. 平方推进：`base = (base * base) % mod`，构造下一位对应的幂因子。
7. 位移指数：`exp >>= 1`，处理下一个二进制位。
8. 循环结束返回 `result`，其值即 `(初始 base^初始 exp) mod mod`。

这 8 步就是“按二进制位消费指数”的源码级实现，不依赖第三方黑箱函数。
