# 离散对数 - Baby-step Giant-step

- UID: `MATH-0019`
- 学科: `数学`
- 分类: `数论`
- 源序号: `19`
- 目标目录: `Algorithms/数学-数论-0019-离散对数_-_Baby-step_Giant-step`

## R01

问题定义：给定整数 `a, b, m`，求最小非负整数 `x` 使得 `a^x ≡ b (mod m)`。这就是经典离散对数问题（Discrete Logarithm Problem, DLP）。

## R02

输入输出约定（本 MVP）：
- 输入：`base=a`、`target=b`、`modulus=m`。
- 输出：
  - 若存在解，返回最小非负整数 `x`。
  - 若不存在解，返回 `None`。
  - 若 `gcd(a, m) != 1`，标准 BSGS 不适用，抛出 `ValueError`。

## R03

核心思想：把指数 `x` 分块表示为 `x = i*n + j`，其中 `n ≈ sqrt(m)`。
- Baby step：预计算 `a^j (mod m)`，`j = 0..n-1`，存入哈希表。
- Giant step：把目标变形成 `b * (a^{-n})^i (mod m)`，`i = 0..n`。
- 当 giant 值命中 baby 表时，得到 `x = i*n + j`。

## R04

正确性要点：
- 若 `a^x ≡ b (mod m)` 且 `x = i*n + j`，则
  `a^(i*n+j) ≡ b`，等价于 `a^j ≡ b*(a^{-n})^i`。
- 因为 `j` 被限制在 `[0, n-1]`，`i` 枚举 `[0, n]`，覆盖了 `x` 的平方根分块搜索空间。
- 哈希命中即构成同余等式，两边恢复后可验证候选解。

## R05

复杂度：
- 时间复杂度：`O(sqrt(m))`。
- 空间复杂度：`O(sqrt(m))`。
相比朴素枚举 `O(m)` 有显著提升。

## R06

适用前提与边界：
- 本实现使用 `pow(base, -n, modulus)`，要求 `base` 在模 `m` 下可逆，即 `gcd(base, m)=1`。
- `modulus <= 1` 无意义，直接报错。
- `target == 1` 时平凡解 `x=0`。
- 非互素场景应使用扩展算法（exBSGS），本 MVP 明确不覆盖。

## R07

伪代码：

```text
BSGS(a, b, m):
  if m <= 1: error
  a <- a mod m, b <- b mod m
  if b == 1: return 0
  if gcd(a, m) != 1: error

  n <- floor(sqrt(m-1)) + 1
  table <- empty hash

  cur <- 1
  for j in [0..n-1]:
      if cur not in table: table[cur] = j
      cur <- cur * a mod m

  factor <- a^(-n) mod m
  gamma <- b
  for i in [0..n]:
      if gamma in table:
          j <- table[gamma]
          x <- i*n + j
          if a^x mod m == b: return x
      gamma <- gamma * factor mod m

  return None
```

## R08

MVP 设计说明：
- 语言：Python 3（标准库即可）。
- 文件：
  - `demo.py`：算法实现 + 固定测试样例。
  - `README.md`：原理、边界、复杂度、验证方式。
- 运行方式：`python3 demo.py`，不需要交互输入。

## R09

示例（有解）：
- `2^x ≡ 22 (mod 29)`，输出 `x=26`。
- `5^x ≡ 8 (mod 23)`，输出 `x=6`。
- `10^x ≡ 17 (mod 19)`，输出 `x=8`。

## R10

示例（无解或不适用）：
- 无解：`2^x ≡ 3 (mod 7)`，在 `{1,2,4}` 循环子群中不存在 `3`。
- 不适用：`6^x ≡ 9 (mod 15)`，`gcd(6,15)=3`，标准 BSGS 前提不满足。

## R11

实现细节：
- `bsgs_discrete_log`：核心求解函数。
- `_format_case_result`：单个样例执行与格式化输出。
- `main`：批量运行预置样例，并打印验证信息。

## R12

可运行性与复现：
- 环境：`python3`。
- 命令：

```bash
python3 Algorithms/数学-数论-0019-离散对数_-_Baby-step_Giant-step/demo.py
```

- 输出包含每个样例的 `x`、幂模校验值和 `verified=True/False`。

## R13

测试策略：
- 正例：多个不同模数的可解样例。
- 反例：无解样例。
- 前提校验：非互素样例触发错误路径。
- 一致性：对返回的 `x` 统一执行 `pow(a, x, m) == b mod m` 二次确认。

## R14

常见错误：
- 把 `n` 设得过小导致搜索覆盖不全。
- 忘记检查 `gcd(a,m)=1` 就调用逆元。
- 哈希表冲突处理不当，覆盖更小 `j` 导致非最小解。
- 命中后未做最终 `pow` 校验。

## R15

与其他方法对比：
- 朴素枚举：实现最简单，但 `O(m)` 太慢。
- Pollard's Rho for DLP：期望时间也可达平方根级，但实现更复杂、随机性更强。
- BSGS：确定性强、实现直观，代价是 `O(sqrt(m))` 内存。

## R16

扩展方向：
- 支持 `gcd(a,m) != 1` 的扩展 BSGS（exBSGS）。
- 对超大模数引入分布式哈希或外存表。
- 与椭圆曲线离散对数问题（ECDLP）做结构差异对照。

## R17

工程化建议：
- 若用于教学或面试，保留当前最小实现最利于讲解。
- 若用于实际密码分析实验，应补充：
  - 输入规模上限与超时策略。
  - 结果缓存与持久化。
  - 更系统的随机化回归测试。

## R18

`demo.py` 的源码级算法流程（8 步）：
1. 在 `bsgs_discrete_log` 中做参数规范化：`a,b` 取模，处理 `modulus<=1` 与 `target==1`。
2. 检查 `gcd(a,m)==1`，不满足则抛 `ValueError`，避免无逆元路径。
3. 计算块大小 `n = isqrt(m-1)+1`，确定 baby/giant 的分割尺度。
4. 构建 baby 哈希表：循环计算 `a^j mod m`，记录最早出现的 `j`。
5. 计算 giant 乘子 `factor = a^{-n} mod m`（通过 `pow` 的负指数逆元能力）。
6. 从 `gamma=b` 开始做 giant 迭代：每轮检查 `gamma` 是否在 baby 表中。
7. 命中时生成候选 `x=i*n+j`，再用 `pow(a,x,m)==b` 做最终一致性校验并返回。
8. 若 giant 完整遍历仍未命中，返回 `None`；`main` 汇总打印每个测试样例的结果。
