# 组合数计算 - Lucas定理

- UID: `MATH-0549`
- 学科: `数学`
- 分类: `组合数学`
- 源序号: `549`
- 目标目录: `Algorithms/数学-组合数学-0549-组合数计算_-_Lucas定理`

## R01

本条目实现“组合数计算 - Lucas 定理”的最小可运行版本（MVP）：
- 目标是计算 `C(n, k) mod p`（`p` 为质数）；
- 核心方法是 Lucas 分解：把 `n, k` 写成 `p` 进制后，拆成低位小组合数乘积；
- 实现强调可解释、可验证、可直接运行。

`demo.py` 内置固定样例与小范围穷举校验，不需要任何交互输入。

## R02

问题定义：
- 输入：非负整数 `n`、`k`，以及质数 `p`。
- 输出：`C(n, k) mod p`。

约定：
- 若 `k > n`，返回 `0`；
- 若 `n, k` 非法（负数、非整数）或 `p` 非质数，抛出异常；
- 只覆盖 Lucas 定理的标准前提：`p` 是质数。

## R03

数学基础：

1. Lucas 定理（`p` 为质数）：
   - 设
     - `n = n_0 + n_1 p + n_2 p^2 + ...`
     - `k = k_0 + k_1 p + k_2 p^2 + ...`
   - 则
     - `C(n,k) ≡ Π_i C(n_i, k_i) (mod p)`。
2. 小规模组合数（`0 <= n_i, k_i < p`）可由阶乘与逆元计算：
   - `C(a,b) mod p = fact[a] * inv_fact[b] * inv_fact[a-b] mod p`。
3. 逆元通过费马小定理得到：
   - `x^(p-2) mod p = x^(-1) mod p`（`x` 非零）。

## R04

算法流程（高层）：
1. 校验输入并确认 `p` 是质数。
2. 若 `k > n`，直接返回 `0`。
3. 预处理 `0..p-1` 的 `fact` 和 `inv_fact`（对同一 `p` 缓存复用）。
4. 循环分解 `n` 与 `k` 的 `p` 进制最低位：`n_i = n % p`，`k_i = k % p`。
5. 若某位 `k_i > n_i`，根据 Lucas 定理直接返回 `0`。
6. 计算该位小组合数 `C(n_i,k_i) mod p`，乘入结果。
7. 执行 `n //= p`、`k //= p`，继续处理更高位。
8. 循环结束后返回累计乘积。

## R05

核心数据结构：
- `fact: tuple[int, ...]`
  - 含义：`i! mod p`，`i=0..p-1`。
- `inv_fact: tuple[int, ...]`
  - 含义：`(i!)^{-1} mod p`。
- `lru_cache`
  - 对 `_factorials_mod_prime(p)` 做缓存，避免同一 `p` 重复预处理。
- `rows: list[tuple[int, int, int, int, int]]`
  - `lucas_trace` 的逐位追踪记录 `(位索引, n_i, k_i, term, cumulative)`。

## R06

正确性要点：
- Lucas 定理把全局 `C(n,k) mod p` 分解为各位独立的小组合数乘积；
- 若任一位 `k_i > n_i`，该位组合数为 `0`，整体乘积为 `0`；
- 小组合数部分使用阶乘与逆阶乘的标准恒等式，且 `p` 为质数保证逆元存在（非零元）；
- 代码在 `n<=40`、多个质数上与 `math.comb(n,k) % p` 穷举对照，作为实现级验证。

## R07

复杂度分析：
- 设 `L = O(log_p n)` 是 `p` 进制位数。
- 单次查询（已缓存该 `p` 的阶乘表）：
  - 时间复杂度 `O(L)`；
  - 空间复杂度 `O(1)`（不计缓存）。
- 首次遇到某个 `p` 的预处理：
  - 时间 `O(p)`；
  - 额外缓存空间 `O(p)`。

## R08

边界与异常处理：
- `n, k, p` 不是整数：`TypeError`。
- `n < 0` 或 `k < 0`：`ValueError`。
- `p <= 1` 或 `p` 非质数：`ValueError`。
- `k > n`：返回 `0`。
- `n = k = 0`：返回 `1`。

## R09

MVP 取舍：
- 使用 Python 标准库实现，不依赖外部黑盒函数。
- 仅实现“模质数”版本 Lucas，不扩展到合数模（扩展 Lucas/CRT）。
- 通过小范围穷举对照保障正确性，而非引入复杂测试框架。
- 额外提供 `lucas_trace`，便于教学和排错。

## R10

`demo.py` 函数职责：
- `_is_prime`：质数判定（试除法）。
- `_validate_inputs`：输入合法性检查。
- `_factorials_mod_prime`：按 `p` 预处理并缓存 `fact/inv_fact`。
- `_small_comb_mod_prime`：计算 `0<=n,k<p` 时的小组合数模 `p`。
- `lucas_binom_mod`：Lucas 主算法。
- `lucas_trace`：输出逐位计算轨迹。
- `_check_small_against_math_comb`：穷举对照测试。
- `_print_case_table` / `_print_trace`：展示固定样例与追踪表。
- `main`：串联测试与展示。

## R11

运行方式：

```bash
cd Algorithms/数学-组合数学-0549-组合数计算_-_Lucas定理
uv run python demo.py
```

脚本不会请求输入，执行后直接输出校验和样例结果。

## R12

输出字段说明：
- `n`、`k`、`p`：组合数参数与模数。
- `C(n,k) mod p`：Lucas 算法结果。
- `check`：
  - `PASS`：已与 `math.comb(n,k)%p` 对照且一致；
  - `FAIL`：对照不一致（理论上不应出现）；
  - `SKIP`：`n` 太大，跳过直接对照，仅展示 Lucas 结果。
- `Lucas Digit Trace`：显示每一位 `(n_i,k_i)` 的小组合数项与累积乘积。

## R13

内置测试覆盖：
- 穷举校验：`n<=40`，`p in {2,3,5,7,11,13,17,19}`，逐个 `k` 对照 `math.comb`。
- 边界校验：`k>n` 必须返回 `0`。
- 展示样例：
  - 小规模：`(10,3,7)`、`(20,10,13)`；
  - 中规模：`(100,50,13)`、`(1000,123,17)`；
  - 大规模：`(10^18+12345, 10^12+567, 97)`。

## R14

可调参数：
- `_check_small_against_math_comb(max_n=40, primes=[...])`：
  - 可调 `max_n` 与质数集合，平衡覆盖度与运行时间。
- `_print_case_table(..., direct_check_max_n=300)`：
  - 控制哪些样例与 `math.comb` 做直接对照。
- `cases`：
  - 可添加或替换业务关心的 `(n,k,p)`。

## R15

方法对比：
- Lucas（本实现）：
  - 优点：适合超大 `n,k` 的模质数组合数；
  - 缺点：依赖 `p` 为质数，且首次需 `O(p)` 预处理。
- 直接 `math.comb(n,k)%p`：
  - 优点：实现简单；
  - 缺点：当 `n,k` 极大时构造大整数成本很高。
- 递推杨辉三角：
  - 优点：直观；
  - 缺点：对大 `n` 不现实，且不利用模质数结构。

## R16

典型应用：
- 竞赛/工程中的大规模组合计数取模（模质数）。
- 数论与组合数学教学中的 Lucas 定理演示。
- 需要频繁查询同一质数模下多个 `C(n,k)` 的场景（可复用缓存）。
- 作为扩展 Lucas（合数模 + CRT）的基础模块。

## R17

可扩展方向：
- 增加“扩展 Lucas”：处理 `mod m`（`m` 非质数）并接 CRT。
- 针对超大质数 `p` 改为分块或按需策略，降低一次性 `O(p)` 内存压力。
- 增加批量 API：同一 `p` 下批处理多个 `(n,k)`。
- 增加性能基准，比较 Lucas 与直接 `math.comb` 在不同规模下的耗时。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 首先调用 `_check_small_against_math_comb`，在小范围内做穷举正确性对照。
2. 对每个质数 `p`，`lucas_binom_mod` 先经 `_validate_inputs` 校验输入并确认 `p` 是质数。
3. `lucas_binom_mod` 处理快捷边界：`k>n` 直接返回 `0`。
4. `_factorials_mod_prime(p)` 首次被调用时构造 `fact` 与 `inv_fact`，并通过 `lru_cache` 缓存。
5. 主循环里取 `n_i=n%p`、`k_i=k%p`，逐位展开 `p` 进制。
6. 若某位 `k_i>n_i`，立即返回 `0`；否则 `_small_comb_mod_prime(n_i,k_i,p)` 计算该位组合项。
7. 将该位项乘入 `result`，再执行 `n//=p`、`k//=p` 进入下一位，直到两者都为 `0`。
8. 返回最终 `result`；随后 `_print_case_table` 与 `_print_trace` 分别展示数值结果和逐位轨迹。
