# LDA主题模型

- UID: `CS-0107`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `236`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0236-LDA主题模型`

## R01

LDA（Latent Dirichlet Allocation，隐狄利克雷分配）是经典概率主题模型，用于在无监督条件下从文档集合中发现潜在主题。

核心假设：
- 每篇文档是多个主题的混合；
- 每个主题是词表上的概率分布；
- 每个词位先采样主题，再由该主题采样词。

本目录的 MVP 同时提供：
- 手写 `Collapsed Gibbs Sampling`（非黑盒）；
- `scikit-learn` 的变分 Bayes（VB）LDA 作为基线对照。

## R02

问题定义（本实现）：
- 输入：文档 token 序列、主题数 `K`、先验 `alpha/beta`、迭代次数。
- 输出：
  - 文档-主题分布 `theta (D x K)`；
  - 主题-词分布 `phi (K x V)`；
  - 每个主题 Top 词；
  - 困惑度与恢复误差指标。

`demo.py` 采用可复现实验语料（脚本内合成），不依赖外部下载数据，不需要交互输入。

## R03

LDA 生成过程（简化）：
1. 对每个主题 `k`：`phi_k ~ Dir(beta)`。
2. 对每篇文档 `d`：`theta_d ~ Dir(alpha)`。
3. 对文档中每个词位 `n`：
   - 采样主题 `z_dn ~ Categorical(theta_d)`；
   - 采样词 `w_dn ~ Categorical(phi_{z_dn})`。

Collapsed Gibbs 的条件分布：

`p(z_dn=k | z_-dn, w) ∝ (n_dk^-dn + alpha) * (n_kw^-dn + beta) / (n_k^-dn + V*beta)`

其中 `n_dk`、`n_kw`、`n_k` 分别是文档-主题计数、主题-词计数、主题总计数。

## R04

推断方法说明：
- 手写部分：Collapsed Gibbs（离散采样，计数更新直观，便于理解 LDA 本体）。
- 对照部分：`sklearn.decomposition.LatentDirichletAllocation`（变分 Bayes，工程常用）。

这样做的目的：
- 避免“只调用库函数”的黑盒实现；
- 同时保留工程可比性，验证手写实现是否达到合理质量。

## R05

复杂度（Collapsed Gibbs）：
- 记总 token 数 `N`、主题数 `K`、迭代轮数 `T`。
- 时间复杂度：`O(T * N * K)`。
- 空间复杂度：`O(D*K + K*V + N)`。

瓶颈主要是对每个 token 计算 `K` 维条件概率。`K` 与 `N` 增大时，训练时间线性上升。

## R06

小例子（概念层面）：
- 若一篇文档同时包含“chip, model, gpu”和“team, score, match”，
- LDA 不会把文档硬分到单类，而是给出主题混合，例如：
  - `theta_d = [0.62(tech), 0.35(sports), 0.03(food)]`。

这也是主题模型与硬聚类的关键区别：文档可由多个主题共同解释。

## R07

优势：
- 无监督发现语料结构；
- 输出有解释性（主题 Top 词 + 文档主题配比）；
- 适合探索式分析与特征构建。

局限：
- 词袋假设忽略词序；
- 主题数 `K` 需要人工设定；
- 高频噪声词、预处理质量会显著影响主题质量。

## R08

实现依赖与前置知识：
- `numpy`：采样与矩阵计算；
- `pandas`：结果表格化输出；
- `scipy`：匈牙利算法做主题对齐；
- `scikit-learn`：VB-LDA 对照基线。

建议先理解：
- 狄利克雷分布及共轭先验；
- 计数式 Gibbs 更新；
- 困惑度与主题可解释性指标。

## R09

适用场景：
- 新闻、论文、评论、工单文本的主题发现；
- 检索系统中的文档低维语义表征；
- 推荐系统中的兴趣主题提取。

不适用或需谨慎：
- 强依赖语序的任务（如翻译、问答生成）；
- 极短文本（主题信号弱）；
- 未做清洗时噪声词过多的语料。

## R10

正确性检查要点：
1. 每个 token 的重采样必须执行“减旧计数 -> 采样 -> 加新计数”。
2. `theta`、`phi` 每行概率和应接近 1。
3. 困惑度应为有限正数。
4. 主题置换对齐后，恢复误差应在合理范围。
5. 结果应可复现（固定随机种子）。

## R11

数值稳定与工程细节：
- 采样概率若出现非有限值或和为 0，回退均匀分布，避免崩溃；
- 所有概率计算使用 `float64`；
- 概率下限使用 `1e-12` 防止 `log(0)`；
- 通过 `Dirichlet` 平滑（`alpha/beta`）防止零概率项。

## R12

关键参数与调优：
- `K`：主题数，过小会混叠、过大易碎片化。
- `alpha`（文档-主题先验）：
  - 小值：文档更稀疏、少数主题主导；
  - 大值：文档主题更平均。
- `beta`（主题-词先验）：
  - 小值：主题更尖锐；
  - 大值：主题更平滑。
- `gibbs_iters / vb_max_iter`：影响收敛和耗时。

## R13

与其他方法关系：
- 对比 pLSA：LDA 通过文档级先验增强泛化能力；
- 对比 NMF：NMF偏代数分解，LDA提供生成式概率解释；
- 对比 K-Means：K-Means多为硬分配，LDA为主题混合。

## R14

常见失败模式与防护：
- 主题词不清晰：增加文本清洗、停用词处理、短语建模；
- 主题混叠：调整 `K`、降低 `alpha`、增加迭代；
- 结果不稳定：固定随机种子，增加迭代与多次重启；
- 高频泛词主导：加入词频截断或 TF-IDF 预筛。

## R15

工程落地建议：
- 把预处理与建模参数版本化（词表、停用词、`K/alpha/beta`）；
- 同时看定量指标与人工可解释性（Top 词）；
- 对线上任务记录主题漂移，定期重训；
- 把 `theta` 作为下游特征输入分类/推荐模型。

## R16

可扩展方向：
- Online LDA（流式或大规模语料）；
- 监督主题模型（sLDA）；
- 层次主题模型（hLDA）；
- 与词向量/Transformer 表征结合，做混合式主题发现。

## R17

本目录 `demo.py` 的 MVP 设计：
- 合成已知真值语料（可精确评估恢复能力）；
- 手写 Collapsed Gibbs 训练 `theta/phi`；
- 使用 `sklearn` VB-LDA 作为工程基线；
- 用匈牙利算法对齐主题顺序后比较：
  - `theta_mean_l1`、`phi_mean_l1`、`token_perplexity`；
- 输出主题 Top 词、文档主题分布预览与质量守卫断言。

运行方式（无交互）：

```bash
cd Algorithms/计算机-机器学习／深度学习-0236-LDA主题模型
uv run python demo.py
```

## R18

源码级算法流拆解（`demo.py` + `scikit-learn`，8 步）：
1. `build_synthetic_corpus` 先定义 3 组词表主题，采样 `theta_true` 与 `phi_true`，再生成 token 级语料。  
2. `initialize_gibbs_state` 为每个 token 随机赋主题，并建立 `ndk/nkw/nk` 三类计数。  
3. `gibbs_sample` 在每个词位执行“减旧计数 -> 按 `(n_dk+alpha)*(n_kw+beta)/(n_k+V*beta)` 计算概率 -> 采样新主题 -> 加新计数”。  
4. `estimate_theta_phi` 从最终计数恢复带平滑项的 `theta/phi`。  
5. `align_topics` 使用 `scipy.optimize.linear_sum_assignment` 解决主题置换问题，使估计主题与真值主题一一对应。  
6. `evaluate_against_truth` 在对齐后计算 `theta_mean_l1`、`phi_mean_l1` 与 token-level 困惑度。  
7. `LatentDirichletAllocation.fit`（`sklearn/decomposition/_lda.py`）内部先做输入检查与参数初始化（`_init_latent_vars`），随后在 EM 循环中调用 `_e_step` 与 `_em_step` 更新文档-主题和主题-词统计。  
8. 在 `_e_step` 中，`_update_doc_distribution` 对每篇文档执行变分更新；最终 `transform/perplexity/components_` 输出 VB 推断结果，再与手写 Gibbs 结果并排比较。  

这里库函数只用于对照基线，核心 LDA 采样过程已在本地代码显式实现，并非黑盒交付。
