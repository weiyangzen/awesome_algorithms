# 莫比乌斯反演公式

- UID: `MATH-0026`
- 学科: `数学`
- 分类: `数论`
- 源序号: `26`
- 目标目录: `Algorithms/数学-数论-0026-莫比乌斯反演公式`

## R01

莫比乌斯反演公式（Mobius Inversion Formula）是数论中的经典“卷积反演”工具。它用于在“按约数求和”与“恢复原函数”之间建立可逆关系，核心反演核为莫比乌斯函数 `mu(n)`。

## R02

典型使用场景：
- 已知 `f(n) = sum_{d|n} g(d)`，要求 `g(n)`。
- 已知某些按 gcd/约数分组的计数式，想去掉重复贡献。
- 需要把“前缀式约数累加信息”还原成“原子项贡献”。

## R03

基础定义与公式：
- 狄利克雷卷积：`(a * b)(n) = sum_{d|n} a(d)b(n/d)`。
- 常数函数 `1(n)=1`。
- 单位元 `e(1)=1, e(n>1)=0`。
- 若 `f = g * 1`，则 `g = f * mu`，即
  `g(n) = sum_{d|n} mu(d) f(n/d)`。

## R04

`mu(n)` 定义：
- `mu(1)=1`。
- 若 `n` 含平方因子（存在素数 `p` 使 `p^2|n`），则 `mu(n)=0`。
- 若 `n` 是 `k` 个互异素数乘积，则 `mu(n)=(-1)^k`。

## R05

正确性要点：
- `mu` 是常数函数 `1` 在狄利克雷卷积下的逆元，即 `1 * mu = e`。
- 因为 `f = g * 1`，两边再与 `mu` 卷积：
  `f * mu = g * (1 * mu) = g * e = g`。
- 因此可从 `f` 唯一恢复 `g`。

## R06

本目录 MVP 的算法设计：
- 用线性筛 `O(N)` 计算 `mu(1..N)`。
- 用双层枚举倍数计算 `f(n)=sum_{d|n}g(d)`，复杂度 `O(N log N)`。
- 再用同样的“按倍数累加”结构实现反演恢复 `g`，复杂度 `O(N log N)`。
- 增加一个应用验证：`sum mu(k) floor(N/k)^2` 计算互质有序对数量。

## R07

数据结构选择：
- `list[int]` 存储 `mu/f/g`（1-indexed 语义，`0` 位保留）。
- `primes: list[int]` 存储线性筛过程中的素数。
- `is_composite: list[bool]` 标记合数。

## R08

复杂度分析：
- 线性筛 `mu`：时间 `O(N)`，空间 `O(N)`。
- 约数卷积构造 `f`：时间 `O(N log N)`，空间 `O(N)`。
- 反演恢复 `g`：时间 `O(N log N)`，空间 `O(N)`。
- 互质对计数：时间 `O(N)`（已知 `mu` 时）。

## R09

边界与鲁棒性：
- `N < 1` 时线性筛返回最小合法数组。
- `mu[d]=0` 的项在反演时可直接跳过，避免无效计算。
- demo 使用断言验证“恢复序列”等式与“应用公式”正确性。

## R10

伪代码（恢复问题）：

```text
input: N, g[1..N]
mu = linear_sieve_mobius(N)
for d in 1..N:
  for m in {d,2d,3d,...<=N}:
    f[m] += g[d]
for d in 1..N:
  if mu[d] == 0: continue
  for m in {d,2d,3d,...<=N}:
    g_rec[m] += mu[d] * f[m/d]
assert g_rec == g
```

## R11

`demo.py` 实现内容：
- `linear_sieve_mobius(limit)`：线性筛求 `mu`。
- `divisor_sum_convolution(g)`：构造 `f(n)=sum_{d|n}g(d)`。
- `mobius_inversion(f, mu)`：反演恢复 `g`。
- `count_coprime_pairs_ordered(n, mu)`：互质有序对公式。
- `brute_force_coprime_pairs_ordered(n)`：小规模暴力校验。

## R12

运行方式：

```bash
cd Algorithms/数学-数论-0026-莫比乌斯反演公式
python3 demo.py
```

脚本无交互输入，直接输出验证结果和样例表。

## R13

预期输出特征：
- 出现 `Mobius inversion recovery check: PASS`。
- 打印前 12 项 `mu(n), g_true(n), f(n), g_recovered(n)` 对照表。
- 出现 `Coprime ordered-pair counting check: PASS`，并显示公式值与暴力值一致。

## R14

示例验证点：
- 设 `g(n)=n^2+3n+1`，先构造 `f` 再反演，逐项恢复原 `g`。
- 取 `N=20`，比较
  `sum_{k=1}^N mu(k) floor(N/k)^2`
  与暴力枚举 `gcd(a,b)=1` 的有序对数量。

## R15

常见错误：
- 把反演式误写成 `sum mu(n/d)f(d)` 后索引实现不一致。
- 忘记 `mu(1)=1` 或线性筛中 `i%p==0` 分支未 `break`。
- 把 0-index 与 1-index 混用，导致整列错位。

## R16

可扩展方向：
- 与欧拉函数、狄利克雷卷积模板结合，做多题复用。
- 在更大规模下可做分块求和优化（如整除分块）。
- 用于 gcd 分类计数、互素计数、约数和反推等问题。

## R17

与替代方法对比：
- 直接解方程组可行但通常更慢且实现复杂。
- 基于约数关系的递推在某些场景可用，但通用性不如莫比乌斯反演。
- 莫比乌斯反演统一、可复用，且与筛法组合后工程效率高。

## R18

本实现的源码级算法流（8 步）：
1. 设定上界 `N`，初始化 `mu/is_composite/primes`。
2. 执行线性筛：遇到新素数置 `mu[p]=-1`，并按最小素因子规则更新合数 `mu`。
3. 构造目标原函数 `g_true(n)`（demo 用 `n^2+3n+1`）。
4. 用“枚举约数 `d`，累加到所有倍数 `m`”构造 `f(m)=sum_{d|m}g_true(d)`。
5. 用同样的倍数枚举做反演：`g_rec(m)+=mu(d)*f(m/d)`，跳过 `mu(d)=0`。
6. 断言 `g_rec == g_true`，验证反演链路正确。
7. 使用 `sum mu(k) floor(N/k)^2` 计算互质有序对个数。
8. 用 `gcd` 暴力统计同一问题并断言一致，完成应用级正确性闭环。
