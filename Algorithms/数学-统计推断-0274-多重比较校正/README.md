# 多重比较校正

- UID: `MATH-0274`
- 学科: `数学`
- 分类: `统计推断`
- 源序号: `274`
- 目标目录: `Algorithms/数学-统计推断-0274-多重比较校正`

## R01

多重比较校正（multiple comparison correction）解决的是同一批数据上进行大量假设检验时，第一类错误（假阳性）累积失控的问题。

如果每个检验都用显著性水平 `alpha=0.05`，做 `m` 次检验时至少出现一次假阳性的概率会显著上升，近似为：
`1 - (1 - alpha)^m`。例如 `m=24` 时该值约为 `0.71`，远高于 0.05。

## R02

本任务的 MVP 输入输出定义：

- 输入：`m` 个原始 p 值 `p1...pm`，以及显著性水平 `alpha`。
- 输出：
  - 每个假设在不同校正方法下的调整后 p 值；
  - 每个假设是否拒绝原假设（reject / not reject）；
  - 方法级汇总统计（拒绝总数、真阳性数、假阳性数）。

## R03

本实现提供三种经典校正：

- Bonferroni：控制 FWER（family-wise error rate），最保守。
- Holm-Bonferroni：同样控制 FWER，但通常比 Bonferroni 更有检验力。
- Benjamini-Hochberg（BH）：控制 FDR（false discovery rate），在高维场景常更实用。

## R04

核心数学对象：

- `FWER = P(V >= 1)`，`V` 是假阳性个数。
- `FDR = E[V / max(R,1)]`，`R` 是拒绝总数。

三种方法的关键公式：

- Bonferroni：`p_i <= alpha/m` 等价于 `p_i^adj = min(m*p_i, 1)`。
- Holm：按升序 `p_(1)<=...<=p_(m)`，逐步比较 `p_(i) <= alpha/(m-i+1)`，遇到首次失败后停止拒绝。
- BH：找最大 `k` 使 `p_(k) <= (k/m)*alpha`，拒绝前 `k` 个；其调整后 p 值通过反向累积最小化构造。

## R05

MVP 的计算流程：

1. 用 `scipy.stats.ttest_ind` 生成一组原始 p 值（模拟 24 个双样本检验）。
2. 分别调用三个“白盒”实现函数：
   - `bonferroni_correction`
   - `holm_bonferroni_correction`
   - `benjamini_hochberg_correction`
3. 将结果拼成 `pandas.DataFrame`，展示最小 p 值的若干假设。
4. 输出方法级统计，比较不同校正策略的“保守程度 vs 检出能力”。

## R06

正确性直觉：

- Bonferroni 通过把单检验阈值缩小到 `alpha/m`，用并联合并界控制“至少一次误拒”的概率。
- Holm 在最小 p 值处最严格，随后逐步放宽阈值，但依然维持 FWER 控制。
- BH 不要求“零误拒”，而是控制误拒占比的期望，因此通常能拒绝更多假设。

## R07

复杂度分析（`m` 为检验个数）：

- Bonferroni：`O(m)`。
- Holm：排序主导，`O(m log m)`。
- BH：排序主导，`O(m log m)`。
- 空间复杂度均为 `O(m)`。

## R08

数值与实现细节：

- 所有调整后 p 值都用 `clip(..., 0, 1)` 限制在合法区间。
- Holm 调整后 p 值使用 `maximum.accumulate` 保证单调性。
- BH 调整后 p 值使用“反向累积最小值”保证单调性。
- 采用固定随机种子 `seed=7`，确保 demo 输出可复现。

## R09

边界情况处理：

- `m=1` 时三种方法退化为单检验。
- p 值为 0 或接近 1 时，clip 可避免越界。
- 若 BH 无任何 `p_(i)` 满足阈值条件，则拒绝集合为空。
- 若出现并列 p 值，排序后的逆置换仍可稳定映射回原顺序。

## R10

代码结构（`demo.py`）：

- `CorrectionResult`：统一承载 `adjusted_pvalues` 与 `rejected`。
- 三个校正函数：独立、可单测。
- `simulate_multiple_tests`：生成具有“部分真效应”的合成数据。
- `attach_corrections`：将校正结果并回表格。
- `summarize`：统计真发现与假发现。
- `main`：一键运行、打印关键结果。

## R11

运行方式：

```bash
uv run python Algorithms/数学-统计推断-0274-多重比较校正/demo.py
```

程序无交互输入，会直接输出：

- 最小原始 p 值前 12 个假设的详细表格；
- 三种方法的拒绝统计对比。

## R12

输出解读建议：

- `p_bonf / p_holm / p_bh`：对应方法下的调整后 p 值。
- `rej_* = True`：在当前 `alpha` 下拒绝原假设。
- 典型现象：`BH` 的拒绝数通常最多，`Bonferroni` 最少，`Holm` 居中。

## R13

方法比较（工程视角）：

- 若场景对“任何假阳性都极其敏感”（如高风险确认实验），优先 FWER 控制方法。
- 若场景允许少量假阳性以换取更高检出率（如探索性筛选），BH 更常用。
- Holm 在“严格控制错误率 + 不想过度保守”之间是很实用的折中。

## R14

局限与注意事项：

- 本 demo 假设检验之间独立或弱相关；强相关结构下方法性质会变化。
- 统计效应和样本量由模拟参数决定，不能直接外推到真实业务。
- 仅演示了单一检验类型（双样本 t 检验），并未覆盖非参数检验或配对设计。

## R15

最小验证清单：

- 代码可运行：`uv run python .../demo.py`。
- `README.md` 与 `demo.py` 不含未填充占位符。
- 三种方法均输出调整后 p 值与拒绝布尔值。
- 调整后 p 值都在 `[0, 1]`。
- Holm/BH 的排序域内调整后 p 值单调性由累积操作保证。

## R16

可扩展方向：

- 增加 Benjamini-Yekutieli（相关性更保守）与 Storey q-value。
- 引入真实数据集，支持按分组批处理并输出图形化报告。
- 增加置换检验或 bootstrap p 值并接入同样的校正框架。

## R17

结论：

多重比较校正是把“单检验显著性”转化为“批量决策可信度”的关键步骤。该 MVP 给出一套可复现、可读、可扩展的最小实现，用于快速比较 FWER 与 FDR 两类控制思路在实际数据上的行为差异。

## R18

下面按源码层面追踪 `demo.py` 的算法流（非黑盒）：

1. `simulate_multiple_tests` 逐个构造 `m` 个假设：部分索引注入真实效应，其他保持零效应。
2. 对每个假设调用 `scipy.stats.ttest_ind` 得到原始 p 值，组成向量 `pvalues`。
3. Bonferroni 分支直接做向量运算：`adjusted = min(m*p, 1)`，并按 `p <= alpha/m` 给出拒绝标记。
4. Holm 分支先对 p 值升序排序，按 step-down 阈值 `alpha/(m-i+1)` 逐位比较，一旦失败立刻停止后续拒绝。
5. Holm 的调整后 p 值先算 `raw_adj=(m-i+1)*p_(i)`，再做前向 `maximum.accumulate` 保证有序性，最后映射回原索引。
6. BH 分支同样先排序，计算临界线 `(i/m)*alpha`，找到最大通过下标 `k`，令前 `k` 个为拒绝。
7. BH 的调整后 p 值先算 `raw_adj=(m/i)*p_(i)`，再做反向 `minimum.accumulate` 得到单调 q-value，再映射回原顺序。
8. `attach_corrections` 汇总三种结果到同一表；`summarize` 进一步结合 `H0_true` 统计真发现/假发现并打印。

这 8 步覆盖了从原始检验到批量决策的完整链路，且每一步都在代码中可定位、可替换、可单测。
