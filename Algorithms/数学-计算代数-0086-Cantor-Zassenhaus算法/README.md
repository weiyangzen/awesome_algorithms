# Cantor-Zassenhaus算法

- UID: `MATH-0086`
- 学科: `数学`
- 分类: `计算代数`
- 源序号: `86`
- 目标目录: `Algorithms/数学-计算代数-0086-Cantor-Zassenhaus算法`

## R01

Cantor-Zassenhaus 算法用于在有限域 `GF(q)` 上分解多项式（通常先处理为首一、平方自由）。

本目录 MVP 聚焦 `GF(p)`（`p` 为奇素数）下的可复现实现，完整走通两段流程：

- Distinct-Degree Factorization（DDF）：按不可约因子次数分块；
- Equal-Degree Factorization（EDF）：对同次数块做随机分裂（Cantor-Zassenhaus 核心）。

## R02

问题定义（本实现）：

- 输入：`GF(p)` 上首一且平方自由的多项式 `f(x)`；
- 输出：`f(x)` 的首一不可约因子列表；
- 约束：
  - 当前 MVP 仅实现 `p` 为奇素数的 EDF 路径；
  - `f` 必须平方自由（脚本会检查，不满足则报错）。

## R03

为什么这条算法重要：

- 在有限域计算代数中，多项式分解是核心基础操作；
- 相比纯暴力试除，Cantor-Zassenhaus 具有更好的随机化效率；
- 它是很多上层任务的底层组件，如编码理论、有限域构造、符号计算等。

## R04

核心数学分解框架：

1. 先保证 `f` 平方自由（`gcd(f, f') = 1`）。
2. DDF：
   - 令 `x` 为多项式变量，迭代计算 `h <- h^p mod f`；
   - 用 `gcd(f, h - x)` 提取“所有不可约因子次数为 d”的块。
3. EDF（Cantor-Zassenhaus）：
   - 对一个“全部因子次数均为 d”的块 `F`，随机取 `a(x)`；
   - 计算 `h = a^((p^d-1)/2) mod F`；
   - 取 `gcd(h-1, F)` 得到非平凡分裂后递归。

## R05

`demo.py` 的数据表示：

- 多项式用升幂系数数组表示：`[a0, a1, ..., an]` 对应 `a0 + a1*x + ... + an*x^n`；
- 系数均按 `mod p` 归一；
- `Case` 数据类保存测试集（名称、素域、期望因子）；
- `numpy.random.Generator` 用于 EDF 随机多项式采样并固定种子复现。

## R06

正确性要点（MVP 级）：

1. DDF 使用 `x^(p^d) - x` 在 `GF(p)` 上筛出“次数整除 d”的不可约因子块；
2. EDF 在等次数块上，利用有限域乘法群指数 `(p^d-1)/2` 构造二值划分；
3. 随机 `a(x)` 以高概率给出非平凡 `gcd`，递归直到每块次数为 `d`；
4. 结果因子相乘应回到原多项式，且每个因子应不可约（demo 里有小规模穷举验证）。

## R07

时间复杂度（概览）：

- 多项式乘法若按朴素卷积：`O(n^2)`；
- 模幂 `a^e mod f` 由二进制幂构成，约 `O(log e)` 次乘模；
- DDF + EDF 总体为随机多项式时间，工程上通常明显快于试除法；
- 本 MVP 以“可读实现”为优先，不引入 FFT 乘法等高级优化。

## R08

空间复杂度：

- 朴素多项式运算主要存储若干 `O(n)` 系数数组；
- 递归 EDF 额外占用与分解深度相关的栈空间；
- 总体可视为 `O(n)` 到 `O(n^2)` 的实现级开销（取决于中间对象保留方式）。

## R09

有限域算术细节：

- 系数逆元通过 `pow(c, p-2, p)`（费马小定理）计算；
- `poly_monic` 统一首一化，减少比较和校验歧义；
- `poly_divmod / poly_gcd / poly_pow_mod` 全部显式在源码中实现，不依赖黑盒分解库。

## R10

边界与异常处理：

- 除零多项式会抛 `ZeroDivisionError`；
- 非平方自由输入会在 `square_free_check` 直接拒绝；
- 导数为零（`p` 次幂结构）场景在本 MVP 明确标注为不支持；
- `p=2` 的 EDF 特例未实现，当前仅支持奇素数域。

## R11

运行方式：

```bash
cd Algorithms/数学-计算代数-0086-Cantor-Zassenhaus算法
python3 demo.py
```

脚本无交互输入，会自动运行内置案例并打印分解结果与校验状态。

## R12

输出解读：

- `Input f(x)`：输入多项式；
- `factor[i]`：算法产出的因子及其次数；
- `Rebuild check`：所有因子相乘是否还原输入；
- `Irreducible check`：小规模穷举检验每个因子是否不可约；
- 最后 `All checks passed.` 表示用例整体通过。

## R13

demo 覆盖的最小测试集：

- `GF(5)` 下混合次数：`1 + 2 + 2`；
- `GF(5)` 下等次数块：`3 + 3`（用于触发 EDF 随机分裂）；
- 每个案例都做：
  - 因子乘回验证；
  - 与期望因子集合比较；
  - 不可约性穷举校验。

## R14

可调参数与复现策略：

- 随机种子固定为 `20260407`，保证同环境可复现；
- `max_trials` 控制 EDF 最大随机尝试次数；
- 可替换测试因子集合扩展到更高次数，但本 MVP 保持小规模、可验证优先。

## R15

与 Berlekamp 等方法对比（简要）：

- Berlekamp：线性代数主导，适合小特征域经典实现；
- Cantor-Zassenhaus：随机化分裂，工程实现通常更简洁；
- 本条目选择 Cantor-Zassenhaus，是因为它能在较短代码中展示 DDF + EDF 的完整流程。

## R16

应用场景：

- 代数编码（如 BCH/RS 相关构造中的多项式处理）；
- 计算机代数系统中的有限域运算内核；
- 密码学与有限域实验教学中的算法演示；
- 作为后续高级分解器（如大规模优化实现）的基线版本。

## R17

`demo.py` 关键函数映射：

- 基础运算：`poly_add / poly_sub / poly_mul / poly_divmod / poly_gcd / poly_pow_mod`；
- 输入合法性：`square_free_check`；
- 分解阶段：`distinct_degree_factorization`（DDF）、`equal_degree_factorization`（EDF）；
- 总流程：`factor_square_free_monic`；
- 验证与展示：`is_irreducible_bruteforce`、`run_case`、`main`。

## R18

源码级算法流程拆解（对应 `demo.py`，9 步）：

1. `main` 固定随机种子，装载两个 `GF(5)` 示例因子集合并构造输入多项式。  
2. `run_case` 调用 `factor_square_free_monic` 进入分解流程。  
3. `square_free_check` 计算 `gcd(f, f')`，若非 1 则拒绝（确保平方自由前提）。  
4. `distinct_degree_factorization` 迭代 Frobenius `h <- h^p mod g`，并用 `gcd(g, h-x)`提取各次数块。  
5. 对每个次数块 `(d, block)` 调用 `equal_degree_factorization`。  
6. EDF 随机采样 `a(x)`，计算 `h = a^((p^d-1)/2) mod block`，再做 `gcd(h-1, block)`。  
7. 若得到非平凡因子，则递归分裂 `block` 与 `block/g`，直到次数降为 `d`。  
8. 全部因子返回后做首一化和排序，并在 `run_case` 中执行“乘回原式 + 与期望因子比对 + 不可约性穷举验证”。  
9. 所有案例通过后输出 `All checks passed.`，形成可自动验证的最小可运行实现。  

说明：第三方库 `numpy` 仅用于随机数生成与复现控制，分解算法本身由源码逐步实现，没有调用现成黑盒分解 API。
