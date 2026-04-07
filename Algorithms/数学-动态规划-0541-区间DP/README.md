# 区间DP

- UID: `MATH-0541`
- 学科: `数学`
- 分类: `动态规划`
- 源序号: `541`
- 目标目录: `Algorithms/数学-动态规划-0541-区间DP`

## R01

区间 DP（Interval Dynamic Programming）用于处理“答案由子区间组合而成”的优化问题。  
本条目采用最经典的矩阵链乘法（Matrix Chain Multiplication）作为 MVP：
- 给定矩阵维度序列 `dims=[p0,p1,...,pn]`，代表 `n` 个矩阵：
  - `A1: p0 x p1`
  - `A2: p1 x p2`
  - ...
  - `An: p(n-1) x pn`
- 目标：找到最优加括号方式，使总标量乘法次数最小。

`demo.py` 提供：
- 区间 DP 求最小代价；
- 最优分割点回溯得到括号表达式；
- 小规模暴力搜索对照校验正确性。

## R02

问题定义（本目录实现）：
- 输入：正整数维度序列 `dims`，长度至少为 2。  
- 输出：
  - `min_cost`：最小乘法代价；
  - `parenthesization`：一个最优括号方案；
  - `dp_table`：区间 DP 代价表；
  - `split_table`：每个区间最优分割点。

约束与约定：
- 使用 0-based 区间下标 `i..j` 表示矩阵 `A(i+1)..A(j+1)`；
- 空间中仅考虑合法矩阵链（由 `dims` 定义，天然可乘）；
- `demo.py` 不需要任何交互输入，直接运行固定样例。

## R03

状态定义与转移：

1. 设 `n = len(dims)-1`（矩阵个数）。  
2. 定义 `dp[i][j]`：计算矩阵链 `Ai..Aj`（1-based）所需的最小乘法次数。  
3. 边界：`dp[i][i] = 0`（单个矩阵无需乘法）。  
4. 转移：
   `dp[i][j] = min_{k in [i, j-1]} (dp[i][k] + dp[k+1][j] + p_{i-1} * p_k * p_j)`。

在代码的 0-based 表达中：
`cost = dp[i][k] + dp[k+1][j] + dims[i] * dims[k+1] * dims[j+1]`。

## R04

算法流程（自底向上区间 DP）：
1. 校验 `dims` 是长度 `>=2` 的有限正整数一维序列。  
2. 初始化 `dp` 为 `n x n`，对角线置 0，其余置为大数。  
3. 初始化 `split` 为 `n x n`，记录最佳分割位置。  
4. 按区间长度 `length=2..n` 枚举。  
5. 对每个区间 `[i,j]`，枚举分割点 `k in [i,j-1]`，计算候选代价。  
6. 若候选更优，则更新 `dp[i][j]` 与 `split[i][j]`。  
7. 最终答案为 `dp[0][n-1]`。  
8. 使用 `split` 递归重建一个最优括号表达式。

## R05

核心数据结构：
- `dims: np.ndarray[int64]`
  - 维度数组，长度 `n+1`。  
- `dp: np.ndarray[int64]`
  - `dp[i,j]` 最小代价，仅上三角有效。  
- `split: np.ndarray[int64]`
  - `split[i,j]` 记录区间 `[i,j]` 的最优断开点 `k`。  
- `CaseResult`（字典）
  - 保存每个样例的输入、DP 结果、暴力校验结果与最优括号。

## R06

正确性要点：
- 最优子结构：区间 `[i,j]` 的最优划分必然由某个 `k` 将其拆成两个最优子区间。  
- 无后效性：`dp[i][j]` 仅依赖更短区间状态，与形成该状态的路径无关。  
- 穷尽分割：对每个 `[i,j]` 枚举全部 `k`，不会漏掉最优方案。  
- 自底向上次序正确：长度从小到大，保证转移时子问题已求解。  
- 回溯正确：`split` 记录每个区间的最优切分，递归恢复得到与 `dp` 一致的最优括号。

## R07

复杂度分析：
- 设矩阵个数为 `n`。  
- 时间复杂度：
  - 区间长度 `O(n)`；
  - 起点 `i` 枚举 `O(n)`；
  - 分割点 `k` 枚举 `O(n)`；
  - 总计 `O(n^3)`。  
- 空间复杂度：
  - `dp` 与 `split` 各 `O(n^2)`；
  - 递归恢复括号深度 `O(n)`；
  - 总空间 `O(n^2)`。

## R08

边界与异常处理：
- `dims` 不是一维数组：报错。  
- `len(dims) < 2`：报错。  
- 存在非正数或非有限值：报错。  
- 只有一个矩阵（`len(dims)==2`）时：
  - 最小代价为 0；
  - 括号表达式为 `A1`。  
- 当数值过大可能导致 `int64` 溢出时，MVP 不做高精大整数优化（现实场景可改 Python `int` 列表实现）。

## R09

MVP 取舍：
- 使用 `numpy` 保存 DP 表，核心转移逻辑手写，不依赖黑盒优化器。  
- 问题规模选教学友好的小中规模，强调可审计与可验证。  
- 增加“暴力递归 + 记忆化”仅用于小规模校验，不作为主算法。  
- 不引入 Knuth 优化/四边形不等式等高级优化，保持标准区间 DP 形态。

## R10

`demo.py` 主要函数：
- `validate_dims`：检查输入维度合法性。  
- `matrix_chain_interval_dp`：核心区间 DP，输出 `min_cost/dp/split`。  
- `reconstruct_parenthesization`：由 `split` 回溯最优括号。  
- `brute_force_min_cost`：小规模基准真值（递归 + `lru_cache`）。  
- `format_upper_triangle`：友好打印 DP 上三角。  
- `run_case`：单样例执行 + 期望值/暴力校验。  
- `main`：运行预置样例并给出汇总。

## R11

运行方式：

```bash
cd Algorithms/数学-动态规划-0541-区间DP
uv run python demo.py
```

脚本不读取命令行参数，不请求用户输入。

## R12

输出说明：
- `dims`：输入维度序列。  
- `matrix_count`：矩阵个数 `n`。  
- `min_cost`：区间 DP 给出的最优代价。  
- `optimal_parenthesization`：最优括号形式。  
- `expected_cost`：如果样例配置了理论值则显示。  
- `brute_force_cost`：小规模时暴力校验值。  
- `dp_table_upper_triangle`：DP 表上三角（`-` 表示下三角无定义）。  
- `Summary`：全部样例通过状态。

## R13

内置最小测试集：
- `CLRS classic`: `dims=[30,35,15,5,10,20,25]`，期望 `15125`。  
- `Two-way split`: `dims=[10,20,30]`，期望 `6000`。  
- `GeeksforGeeks classic`: `dims=[5,10,3,12,5,50,6]`，期望 `2010`。  
- `Single matrix`: `dims=[8,13]`，期望 `0`。

附加建议：
- 再加随机小规模样例（如 `n<=7`），持续与暴力法自动对照。  
- 再加非法输入样例，验证异常路径。

## R14

可调参数与建议：
- `run_case(..., brute_force_limit=8)`：仅当矩阵数 `<=8` 执行暴力校验。  
- `format_upper_triangle(..., width=10)`：控制表格对齐宽度。  
- 样例规模建议：
  - 教学/调试：`n=4..8`；
  - 性能观察：逐步增加到 `n=100+`，重点看 `O(n^3)` 增长。

## R15

方法对比：
- 对比全枚举括号（Catalan 规模）：
  - 全枚举复杂度指数级；
  - 区间 DP 降到 `O(n^3)`。  
- 对比记忆化搜索（Top-Down）：
  - 复杂度同阶；
  - 本实现 Bottom-Up 更利于表格可视化与教学。  
- 对比高级优化版本：
  - 某些满足额外结构条件的问题可做更快优化；
  - 矩阵链乘法的一般形态通常仍用标准 `O(n^3)` 实现。

## R16

典型应用：
- 矩阵链乘法顺序优化（编译器/线性代数库中的算子重排思想）。  
- 其他区间型问题建模参考：
  - 多边形三角剖分；
  - 戳气球（Burst Balloons）；
  - 最优二叉搜索树等。  
- 动态规划教学中的“区间枚举 + 分割点转移”标准模板。

## R17

可扩展方向：
- 增加 Top-Down 版本并对照调用栈与缓存命中率。  
- 输出完整最优决策树（不仅是一条括号字符串）。  
- 增加随机样例压力测试并记录耗时曲线。  
- 支持“给定具体矩阵对象”时的真实乘法次数统计与内存占用估计。  
- 针对可满足特定不等式的子问题探索更快优化策略。

## R18

`demo.py` 源码级算法流（8 步）：
1. `main` 定义固定样例列表（包含已知最优值），逐个调用 `run_case`。  
2. `run_case` 先把 `dims` 传给 `matrix_chain_interval_dp`，得到 `min_cost`、`dp`、`split`。  
3. `matrix_chain_interval_dp` 内部调用 `validate_dims`，把输入转成 `np.int64` 并做合法性检查。  
4. 在 `matrix_chain_interval_dp` 中初始化 `dp/split`，设置对角线 `dp[i,i]=0`。  
5. 按区间长度从 2 到 `n` 递增，三重循环枚举 `(i,j,k)`，执行核心转移：
   `dp[i,j] = min(dp[i,k] + dp[k+1,j] + dims[i]*dims[k+1]*dims[j+1])`。  
6. 转移时同步记录最佳 `k` 到 `split[i,j]`，用于后续恢复决策。  
7. `run_case` 调用 `reconstruct_parenthesization(split,0,n-1)` 递归构建最优括号字符串。  
8. 当规模不大时，`run_case` 额外调用 `brute_force_min_cost` 做真值对照；全部样例通过后 `main` 输出汇总。
