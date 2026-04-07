# 稳定婚姻问题 - Gale-Shapley

- UID: `MATH-0494`
- 学科: `数学`
- 分类: `组合数学`
- 源序号: `494`
- 目标目录: `Algorithms/数学-组合数学-0494-稳定婚姻问题_-_Gale-Shapley`

## R01

稳定婚姻问题（Stable Marriage Problem）要求在两侧人数相同、每人对另一侧给出严格偏好序的前提下，找到一个**稳定匹配**：不存在一对男女（或一般化为 proposer/receiver）彼此都更偏好对方而不是当前匹配对象。

本条目给出 Gale-Shapley 的最小可运行 MVP：
- 手写提案方主导（proposer-optimal）的延迟接受算法；
- 输出匹配后进行阻塞对检查，验证稳定性；
- 在小规模下穷举全部稳定匹配，验证“提案方最优”结论。

## R02

本目录实现的问题定义：
- 输入：
  - `proposer_prefs in Z^(n*n)`：第 `i` 行是提案方 `i` 对接收方的偏好排列；
  - `receiver_prefs in Z^(n*n)`：第 `j` 行是接收方 `j` 对提案方的偏好排列。
- 约束：
  - 两个矩阵都是 `n*n`；
  - 每一行都是 `0..n-1` 的排列（严格偏好、无重复、无缺失）。
- 输出：
  - `proposer_to_receiver` 与 `receiver_to_proposer` 两个互逆匹配数组；
  - 提案次数 `proposal_count`；
  - 稳定性与提案方最优性验证结果（在 `main` 中打印）。

## R03

核心数学概念：
1. 匹配 `mu`：把每个提案方映射到唯一接收方，反向也唯一。  
2. 阻塞对 `(p, r)`：`p` 比 `mu(p)` 更喜欢 `r`，且 `r` 比 `mu(r)` 更喜欢 `p`。  
3. 稳定匹配：不存在任何阻塞对。  
4. Gale-Shapley 定理：在严格偏好下算法必终止并返回稳定匹配；若由提案方发起，则结果对提案方在所有稳定匹配中逐个体最优。

## R04

算法主流程（提案方版本）：
1. 初始化所有提案方为“未匹配”，每人下一次要提案的偏好下标设为 0。  
2. 从未匹配队列取一个提案方 `p`。  
3. `p` 向其尚未提案过的最高偏好接收方 `r` 提案。  
4. 若 `r` 当前空闲，则暂时接受 `p`。  
5. 若 `r` 已有对象 `p_old`，比较 `r` 对 `p` 与 `p_old` 的偏好：
   - 更喜欢 `p`：改配 `p`，`p_old` 重新入队；
   - 否则拒绝 `p`，`p` 继续保留未匹配状态并入队。  
6. 队列为空时结束，得到完整匹配。

## R05

核心数据结构：
- `proposer_prefs[n, n]` / `receiver_prefs[n, n]`：偏好序。  
- `receiver_rank[n, n]`：`receiver_rank[r, p]` 表示接收方 `r` 对提案方 `p` 的排序位置，便于 `O(1)` 比较两位追求者。  
- `next_choice_idx[n]`：每位提案方下次提案到其偏好列表的哪个位置。  
- `proposer_match[n]` / `receiver_match[n]`：双向匹配数组。  
- `deque`：未匹配提案方队列。  
- `StableMarriageResult`：统一封装输出结果。

## R06

正确性要点：
- 终止性：每位提案方最多向每位接收方提案一次，总提案次数不超过 `n^2`。  
- 匹配合法性：接收方每时刻最多暂时接受 1 人，最终形成双射。  
- 稳定性：若输出存在阻塞对可导出矛盾（被阻塞对中的接收方在算法过程中必曾拒绝过更差对象）。  
- 提案方最优性：在所有稳定匹配中，提案方得到其可达最优对象；`demo.py` 通过穷举稳定匹配在小规模上做机器校验。

## R07

复杂度分析：
- 时间复杂度：`O(n^2)`（最多 `n^2` 次提案，每次常数时间比较与更新）。  
- 空间复杂度：`O(n^2)`（主要来自偏好矩阵与 rank 矩阵）。  
- 若开启穷举稳定匹配用于验证，额外复杂度约 `O(n! * n^2)`，仅适合小规模。

## R08

边界与异常处理：
- 偏好矩阵不是二维/非方阵：抛 `ValueError`。  
- 偏好行包含越界编号、重复或缺失：抛 `ValueError`。  
- 匹配数组形状不对或非双射：在验证阶段抛 `ValueError`。  
- 理论上不会出现“有人提完全部对象仍未匹配”，若出现视为实现错误并抛 `RuntimeError`。

## R09

MVP 取舍：
- 只实现经典一对一、两侧等规模、严格完整偏好；不覆盖并列偏好、不可接受对象、容量约束。  
- 核心算法不调用外部匹配黑盒，全部源码可读。  
- 使用 `numpy` 与 `pandas` 仅做数据表示与输出，不替代 Gale-Shapley 核心逻辑。

## R10

`demo.py` 主要函数职责：
- `validate_preferences`：偏好矩阵合法性校验。  
- `build_rank_matrix`：把“顺序偏好”转为“排名查询表”。  
- `gale_shapley`：算法主循环。  
- `is_perfect_matching`：检查结果是否为双射匹配。  
- `find_blocking_pairs`：检测阻塞对。  
- `enumerate_stable_matchings`：小规模穷举稳定匹配。  
- `evaluate_proposer_optimality`：验证提案方最优。  
- `matching_to_dataframe`：格式化结果表。  
- `run_case` / `main`：组织样例并打印结果。

## R11

运行方式：

```bash
cd Algorithms/数学-组合数学-0494-稳定婚姻问题_-_Gale-Shapley
uv run python demo.py
```

脚本使用内置样例，无需任何交互输入。

## R12

输出字段说明：
- `participants per side`：每侧人数 `n`。  
- `total proposals`：算法实际提案次数（`<= n^2`）。  
- `stable matchings found by brute force`：穷举得到的稳定匹配数量。  
- `blocking pairs in GS output`：GS 输出中的阻塞对数，正确实现应为 `0`。  
- `proposer-optimal verified`：是否验证为提案方逐个体最优。  
- 表格列：
  - `proposer` / `receiver`：匹配对；
  - `p_rank(0-best)`：该提案方拿到对象在其偏好中的排名；
  - `r_rank(0-best)`：该接收方对当前对象的排名。

## R13

建议最小测试集：
- 手工 `4x4` 教学样例：便于人工复核过程。  
- 随机 `6x6` 固定种子样例：覆盖一般输入。  
- 异常样例（建议单元测试补充）：
  - 非方阵偏好；
  - 行内重复元素；
  - 越界编号。

## R14

可调项：
- `main` 中案例规模 `n2`（当前为 `6`）。  
- 随机种子 `np.random.default_rng(20260407)`，用于可复现比较。  
- `enumerate_stable_matchings(max_n=8)` 的穷举上限，避免阶乘爆炸。

调参建议：
- 仅想看算法本体，可把穷举验证关掉或调小规模；
- 想强化正确性审计，可保留 `n<=8` 的穷举校验。

## R15

方法对比：
- 对比暴力搜索所有匹配：
  - 暴力需 `n!` 级别；
  - Gale-Shapley 仅 `O(n^2)`。  
- 对比“随机配对后修补”：
  - 启发式不保证稳定；
  - Gale-Shapley 有严格稳定性与最优性结论。  
- 对比“接收方提案版本”：
  - 也稳定，但最优性偏向接收方而非提案方。

## R16

典型应用：
- 学校-学生、医院-住院医等双边匹配。  
- 招聘与候选人双向选择。  
- 双边推荐系统中的稳定分配基线。  
- 机制设计课程中的经典可验证案例。

## R17

可扩展方向：
- 不完整偏好（允许不可接受对象）。  
- 并列偏好（ties）与弱稳定概念。  
- 多对一容量约束（医院-住院医/college admissions）。  
- 战略行为与机制抗操纵性分析。  
- 在更大规模上加入性能剖析与日志追踪。

## R18

`demo.py` 源码级流程（8 步）：
1. `main` 构造一个手工 `4x4` 和一个固定随机种子 `6x6` 的偏好矩阵。  
2. `run_case` 调用 `gale_shapley`，在 `deque` 上维护未匹配提案方。  
3. `gale_shapley` 中每轮让提案方按 `next_choice_idx` 向下一个偏好对象提案。  
4. 接收方用 `receiver_rank` 常数时间比较“新提案者 vs 当前对象”，决定保留谁。  
5. 队列清空后得到 `proposer_to_receiver / receiver_to_proposer`，并返回提案次数。  
6. `is_perfect_matching` 与 `find_blocking_pairs` 对结果做结构与稳定性检查。  
7. `enumerate_stable_matchings` 对 `n<=8` 的实例穷举所有双射，筛出全部稳定匹配。  
8. `evaluate_proposer_optimality` 比较 GS 结果与全部稳定匹配中每位提案方的最优排名，给出可审计结论。
