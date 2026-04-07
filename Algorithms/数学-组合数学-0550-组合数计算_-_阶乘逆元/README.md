# 组合数计算 - 阶乘逆元

- UID: `MATH-0550`
- 学科: `数学`
- 分类: `组合数学`
- 源序号: `550`
- 目标目录: `Algorithms/数学-组合数学-0550-组合数计算_-_阶乘逆元`

## R01

本条目实现“组合数计算 - 阶乘逆元”的最小可运行版本（MVP）：
- 在素数模 `p` 下，预处理 `fac[i] = i! mod p` 与 `ifac[i] = (i!)^{-1} mod p`；
- 单次查询 `C(n,k) mod p` 以 `O(1)` 返回；
- 通过小范围恒等式与 `math.comb` 取模对照验证正确性。

`demo.py` 为非交互脚本，执行后直接输出结果表和检查结论。

## R02

问题定义：
- 输入：
  - 固定构建参数：素数 `mod` 与预处理上界 `max_n`；
  - 查询参数：`n, k`。
- 输出：
  - `C(n,k) mod mod`。

约定：
- 当 `k < 0` 或 `k > n` 时返回 `0`；
- 当 `0 <= k <= n <= max_n` 时返回标准组合数的模值；
- 本 MVP 约束 `max_n < mod`，避免阶乘出现 `0 mod mod` 而不可逆。

## R03

数学基础：
1. 组合数定义：
   - `C(n,k) = n! / (k!(n-k)!)`
2. 模逆元（费马小定理）：
   - 若 `p` 为素数且 `a % p != 0`，则 `a^(p-2) % p = a^{-1} % p`
3. 因而在 `n < p` 时可写为：
   - `C(n,k) mod p = fac[n] * ifac[k] * ifac[n-k] mod p`
4. 对称性：
   - `C(n,k) = C(n,n-k)`，可减少访问位置。

## R04

算法流程（高层）：
1. 校验 `mod` 是否为素数、`max_n` 是否非负且 `max_n < mod`。
2. 预处理 `fac[0..max_n]`：`fac[i] = fac[i-1] * i % mod`。
3. 先算 `ifac[max_n] = pow(fac[max_n], mod-2, mod)`。
4. 倒推 `ifac`：`ifac[i-1] = ifac[i] * i % mod`。
5. 查询时校验 `n,k`，处理越界 `k` 返回 `0`。
6. 用 `k=min(k,n-k)` 做对称压缩。
7. 返回 `fac[n] * ifac[k] % mod * ifac[n-k] % mod`。

## R05

核心数据结构：
- `fac: list[int]`：长度 `max_n+1`，保存阶乘模值。
- `ifac: list[int]`：长度 `max_n+1`，保存阶乘逆元模值。
- `FactorialInverseComb`：
  - 持有 `mod`、`max_n`、`fac`、`ifac`；
  - 对外提供 `comb(n,k)` 查询。

## R06

正确性要点：
- `fac` 的构造是阶乘定义的直接递推。
- `ifac[max_n]` 由费马小定理得到确切逆元（因 `fac[max_n] != 0 (mod p)`）。
- 倒推式 `ifac[i-1] = ifac[i] * i % p` 等价于 `(i-1)!^{-1} = i!^{-1} * i`。
- 因此每个 `ifac[t]` 都是 `t!` 的逆元。
- 查询公式等价于 `n! / (k!(n-k)!)` 的模域乘法表达，所以结果正确。

## R07

复杂度分析：
- 预处理时间：`O(max_n)`。
- 预处理空间：`O(max_n)`（两条数组）。
- 单次查询时间：`O(1)`。
- 单次查询空间：`O(1)`。

## R08

边界与异常处理：
- `mod`、`max_n` 不是整数：`TypeError`。
- `max_n < 0`：`ValueError`。
- `mod` 非素数或 `mod <= 2`：`ValueError`。
- `max_n >= mod`：`ValueError`（本实现未启用 Lucas 扩展）。
- `n` 或 `k` 非整数：`TypeError`。
- `n < 0` 或 `n > max_n`：`ValueError`。
- `k` 超出 `[0,n]`：返回 `0`。

## R09

MVP 取舍：
- 采用纯 Python + 标准库实现，保持可读与可移植。
- 明确实现预处理与查询公式，不调用第三方黑盒组合数 API 作为主路径。
- 只做必要的验证与表格输出，不引入大型框架或复杂工程结构。
- 未实现 `n >= mod` 场景（需 Lucas 或扩展算法），保证实现边界清晰。

## R10

`demo.py` 函数职责：
- `_is_prime_trial_division`：判素数（构造期校验用）。
- `_validate_mod_and_max_n`：构造参数合法性检查。
- `FactorialInverseComb._precompute`：生成 `fac` 与 `ifac`。
- `FactorialInverseComb.comb`：`O(1)` 查询 `C(n,k) mod p`。
- `_self_check_small`：对小范围值做 `math.comb` 与恒等式校验。
- `_print_case_table`：打印固定样例与对照信息。
- `main`：组织预处理、校验和展示。

## R11

运行方式：

```bash
cd Algorithms/数学-组合数学-0550-组合数计算_-_阶乘逆元
uv run python demo.py
```

脚本无需任何输入，直接运行并输出结果。

## R12

输出字段说明：
- `n`、`k`：组合参数。
- `C(n,k) mod p`：MVP 算法输出。
- `reference`：
  - 当 `n <= 300` 且 `0<=k<=n`，显示 `math.comb(n,k) % p`；
  - 否则显示 `-`（避免超大精确整数带来的额外开销）。
- `check`：
  - `PASS/FAIL`：与参考值对比结果；
  - `N/A`：未启用参考值对照。

## R13

内置测试覆盖：
- 小范围全覆盖：`n<=160` 的所有 `k`，逐项与 `math.comb` 取模对照。
- 对称性检查：`C(n,k)=C(n,n-k)`。
- 行和恒等式：`sum_k C(n,k) = 2^n (mod p)`。
- 固定展示样例覆盖小中大规模与越界 `k` 情况。

## R14

可调参数：
- `main` 中 `mod`（默认 `1_000_000_007`）。
- `main` 中 `max_n`（默认 `200_000`）。
- `_self_check_small(engine, limit_n=160)` 的校验上限。
- `cases` 列表中的展示查询集合。

调参建议：
- 开发期可下调 `max_n` 和 `limit_n` 加快迭代；
- 需要更多查询时优先提升 `max_n`，不改变查询复杂度。

## R15

方法对比：
- 阶乘逆元法（本实现）：
  - 预处理后单次查询 `O(1)`，适合大量查询；
  - 需要素数模与 `n < mod`（在本 MVP 约束内）。
- 递推法（Pascal）：
  - 无需模逆元，定义直观；
  - 单次常为 `O(nk)` 或 `O(n^2)` 量级。
- 连乘约分法：
  - 单点查询常较高效；
  - 多查询时不如预处理法复用性高。
- Lucas 定理：
  - 可处理 `n >= p`；
  - 实现更复杂，不属于当前最小 MVP。

## R16

典型应用：
- 竞赛与工程中的批量 `nCk mod p` 查询。
- 组合计数 DP 中频繁访问二项系数。
- 概率模型离散化时的系数计算。
- 作为 Lucas、NTT、组合恒等式模块的基础设施。

## R17

可扩展方向：
- 增加 Lucas 定理支持，去掉 `n < mod` 限制。
- 支持多模数组合（CRT 合并）以扩大适用范围。
- 增加批量 API（向量化输入）和微基准测试。
- 缓存多组 `mod/max_n` 预处理实例，服务多任务场景。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 设定 `mod=1_000_000_007` 与 `max_n=200000`，构造 `FactorialInverseComb`。
2. `__post_init__` 调用 `_validate_mod_and_max_n`，确认素数模和上界合法。
3. `_precompute` 正向循环生成 `fac[i] = fac[i-1] * i % mod`。
4. `_precompute` 使用 `pow(fac[max_n], mod-2, mod)` 得到 `ifac[max_n]`。
5. `_precompute` 逆向循环得到全部 `ifac[i-1] = ifac[i] * i % mod`。
6. `_self_check_small` 对 `n<=160` 的所有 `k` 调用 `comb`，并与 `math.comb(n,k)%mod`、对称性、行和恒等式逐项核对。
7. `_print_case_table` 逐条样例调用 `comb`，输出结果与可得参考值的比对状态。
8. `comb` 内部按 `fac[n] * ifac[k] * ifac[n-k] mod mod` 常数时间返回，最终脚本打印 PASS 并结束。
