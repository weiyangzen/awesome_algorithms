# 隐狄利克雷分配 (LDA)

- UID: `MATH-0333`
- 学科: `数学`
- 分类: `机器学习`
- 源序号: `333`
- 目标目录: `Algorithms/数学-机器学习-0333-隐狄利克雷分配_(LDA)`

## R01

隐狄利克雷分配（Latent Dirichlet Allocation, LDA）是一种经典的概率主题模型，用于从文档集合中自动发现“潜在主题”。

直观上，LDA 假设：
- 每篇文档是若干主题的混合；
- 每个主题是词汇表上的概率分布；
- 文档中的每个词先选主题，再由该主题生成词。

本目录 MVP 采用“手写 Collapsed Gibbs Sampling”完成 LDA 推断，不调用 `sklearn` 的黑盒 LDA 训练器，重点展示核心计数更新与条件采样机制。

## R02

本实现问题定义：
- 输入：
  - 文档集合（`List[List[int]]`，每篇文档为词 id 序列）；
  - 主题数 `K`；
  - 先验超参数 `alpha`、`beta`；
  - 采样轮数 `n_iters`。
- 输出：
  - 文档-主题分布 `theta`（形状 `D x K`）；
  - 主题-词分布 `phi`（形状 `K x V`）；
  - 每个主题的 Top-N 关键词；
  - 训练语料困惑度（perplexity）。

`demo.py` 中语料由脚本自动合成，无需交互输入。

## R03

LDA 生成模型（对每篇文档 `d`）：
1. 文档主题分布：`theta_d ~ Dir(alpha)`
2. 每个主题词分布：`phi_k ~ Dir(beta)`
3. 对文档中每个词位 `n`：
   - `z_dn ~ Categorical(theta_d)`
   - `w_dn ~ Categorical(phi_{z_dn})`

Collapsed Gibbs 采样在积分掉 `theta/phi` 后，对词位主题变量 `z_dn` 的条件分布为：

`p(z_dn = k | z_-dn, w) ∝ (n_dk^-dn + alpha) * (n_kw^-dn + beta) / (n_k^-dn + V*beta)`

其中：
- `n_dk`：文档 `d` 中分配到主题 `k` 的词数；
- `n_kw`：主题 `k` 生成词 `w` 的次数；
- `n_k`：主题 `k` 的总词数；
- `V`：词表大小。

## R04

算法流程（高层）：
1. 构造或读取语料，转成词 id 序列。  
2. 为每个词位随机初始化主题。  
3. 建立并维护 `n_dk / n_kw / n_k` 计数矩阵。  
4. 逐词位执行 Gibbs 重采样：先减旧计数，再按条件分布采样新主题，再加新计数。  
5. 重复第 4 步共 `n_iters` 轮。  
6. 由最终计数估计 `theta` 与 `phi`。  
7. 输出主题关键词、文档主题分布与困惑度。

## R05

核心数据结构：
- `LDAConfig`（`@dataclass`）：
  - `n_topics, alpha, beta, n_iters, seed`；
  - 合成语料规模参数：`n_docs, doc_len_min, doc_len_max`；
  - 展示参数：`top_n_words`。
- `docs: List[np.ndarray]`：每篇文档的词 id 序列。
- `z_dn: List[np.ndarray]`：每篇文档每个词位的主题指派。
- 计数矩阵：
  - `ndk`：`(D, K)`；
  - `nkw`：`(K, V)`；
  - `nk`：`(K,)`。

## R06

正确性与稳定性要点：
- 每次重采样都严格执行“减旧 -> 采样 -> 加新”，保证计数一致性；
- 采样概率使用浮点归一化，若总和异常则回退均匀分布，避免数值崩溃；
- `theta`、`phi` 都带狄利克雷平滑（`alpha/beta`），避免零概率；
- 展示阶段做主题置换对齐（topic permutation alignment），仅改变输出顺序，便于与参考主题对照；
- 运行后用断言检查：
  - `theta`、`phi` 行和接近 1；
  - 困惑度为有限正数。

## R07

复杂度分析：
- 设文档总词数为 `N`，主题数为 `K`，采样轮数为 `T`。
- 时间复杂度：
  - 每个词位重采样需计算 `K` 个主题概率，单轮约 `O(NK)`；
  - 总体约 `O(TNK)`。
- 空间复杂度：
  - 计数矩阵与主题指派存储约 `O(DK + KV + N)`。

## R08

边界与异常处理：
- 参数检查：`n_topics > 1`、`alpha > 0`、`beta > 0`、`n_iters > 0`；
- 语料检查：空文档集合直接报错；
- 数值检查：采样概率求和异常时进行保护性回退；
- 输出检查：若概率矩阵不归一或困惑度非法，抛出 `RuntimeError`。

## R09

MVP 取舍：
- 采用合成语料，保证脚本自包含可复现；
- 只实现最核心的 Collapsed Gibbs，不引入在线 LDA、变分 Bayes、分布式训练；
- 不依赖外部 NLP 预处理器（分词、停用词、词形还原等），聚焦算法本体；
- 输出以解释性为主：Top 词 + 文档主题分布 + 困惑度。

## R10

`demo.py` 主要函数职责：
- `build_synthetic_corpus`：构造带已知主题结构的词表与文档；
- `initialize_state`：随机初始化主题并建立计数；
- `gibbs_sample`：执行 Collapsed Gibbs 主循环；
- `estimate_theta_phi`：由计数恢复 `theta/phi`；
- `compute_perplexity`：计算训练语料困惑度；
- `top_words_per_topic`：提取每个主题 Top 关键词；
- `main`：组织流程、打印摘要并做基本校验。

## R11

运行方式：

```bash
cd Algorithms/数学-机器学习-0333-隐狄利克雷分配_(LDA)
python3 demo.py
```

无需命令行参数，不需要交互输入。

## R12

输出解释：
- `Config`: 本次实验配置；
- `Vocabulary size / Documents`: 语料规模；
- `Topic k top words`: 每个主题概率最高的关键词；
- `Document-topic mixtures`: 前几篇文档的 `theta_d`（每篇文档属于各主题的概率）；
- `Training perplexity`: 模型在训练语料上的困惑度（越低通常越好）。

## R13

建议最小测试集：
- 默认合成语料（已内置）：验证主题可恢复性；
- 稀疏文档测试：缩短文档长度，观察主题混合不确定性上升；
- 主题数错配测试：
  - `K` 小于真实主题数时，主题会混叠；
  - `K` 大于真实主题数时，可能出现冗余主题。

## R14

关键参数与调优建议：
- `n_topics`：主题数，过小会欠分解，过大可能出现碎片主题；
- `alpha`：文档-主题先验，
  - 小 `alpha` 倾向“每篇文档少数主题主导”；
  - 大 `alpha` 倾向“主题更均匀混合”；
- `beta`：主题-词先验，
  - 小 `beta` 主题更尖锐；
  - 大 `beta` 主题更平滑；
- `n_iters`：迭代轮数，太小未收敛，太大计算成本上升。

## R15

方法对比：
- 对比 pLSA：
  - pLSA 缺少文档级先验，泛化到新文档时不如 LDA 自然；
  - LDA 的 Dirichlet 先验让模型更稳健。
- 对比 NMF：
  - NMF 也是主题发现工具，但多为代数分解视角；
  - LDA 提供完整概率生成解释。
- 对比聚类（KMeans）：
  - 聚类常把文档硬分配到单类；
  - LDA 允许一篇文档由多个主题混合生成。

## R16

典型应用场景：
- 新闻/论文/评论语料的主题发现；
- 文档检索中的主题表示构建；
- 推荐系统中的用户兴趣主题建模；
- 大规模文本数据的探索式分析与标签辅助。

## R17

可扩展方向：
- 由 Gibbs 扩展到变分推断（VB / Online VB）以支持更大语料；
- 引入真实文本预处理管线（分词、去停用词、n-gram）；
- 增加主题一致性指标（Topic Coherence）与多模型比较；
- 支持增量训练与模型持久化；
- 扩展到监督主题模型（sLDA）或层次主题模型（hLDA）。

## R18

`demo.py` 的源码级算法流（8 步，非黑盒）如下：

1. `main` 读取 `LDAConfig`，调用 `build_synthetic_corpus` 生成词表、文档与真实主题参考结构。  
2. `initialize_state` 为每个词位随机分配主题，并同步构建三类计数：`ndk / nkw / nk`。  
3. `gibbs_sample` 进入外层迭代（共 `n_iters` 轮），按文档和词位双层循环遍历全部 token。  
4. 对当前 token，先把旧主题计数从 `ndk / nkw / nk` 中减去，得到“剔除当前词位”的条件统计量。  
5. 依据公式 `(n_dk+alpha)*(n_kw+beta)/(n_k+V*beta)` 计算该 token 在每个主题下的未归一化概率。  
6. 归一化后执行离散采样得到新主题，再把该主题加回计数矩阵，并更新 `z_dn`。  
7. 采样结束后调用 `estimate_theta_phi` 从最终计数恢复 `theta` 与 `phi`（含平滑项）。  
8. `top_words_per_topic` 输出每个主题关键词，`compute_perplexity` 评估训练困惑度，最后打印结果并做一致性校验。

以上步骤全部在本地源码中显式实现，没有把 LDA 训练过程交给第三方黑盒函数。
