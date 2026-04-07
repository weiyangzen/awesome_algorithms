# 词嵌入 - Word2Vec

- UID: `MATH-0334`
- 学科: `数学`
- 分类: `NLP`
- 源序号: `334`
- 目标目录: `Algorithms/数学-NLP-0334-词嵌入_-_Word2Vec`

## R01

Word2Vec 是将离散词符映射到连续向量空间的经典方法。本条目实现的是其最常见训练形式之一：
- Skip-gram（给定中心词预测上下文词）
- Negative Sampling（把多分类 softmax 近似为二分类对比目标）

本目录提供可直接运行、可审计的最小 MVP：
- 纯 NumPy 手写训练循环，不调用第三方黑盒 `word2vec`；
- 固定随机种子的玩具语料，保证输出可复现；
- 自动质量检查（语义相似度与类比任务）确保结果具备基本可用性。

## R02

问题定义（无监督表示学习）：
- 输入：分词后的语料 `C = {w_1, w_2, ..., w_T}`。
- 输出：每个词 `w` 的稠密向量 `v_w in R^d`。

目标是让“在上下文中经常共现”的词在向量空间更接近，使向量能支持：
- 余弦相似度检索（nearest neighbors）；
- 类比推理（例如 `king - man + woman ≈ queen`）。

## R03

SGNS（Skip-gram with Negative Sampling）核心目标函数：

对于一个正样本词对 `(c, o)`（中心词 `c` 与真实上下文词 `o`）以及 `K` 个负样本词 `n_i`，最大化：

`log sigma(u_o^T v_c) + sum_{i=1..K} log sigma(-u_{n_i}^T v_c)`

其中：
- `v_c` 来自输入嵌入矩阵 `W_in`；
- `u_o, u_{n_i}` 来自输出嵌入矩阵 `W_out`；
- `sigma(x)=1/(1+exp(-x))`。

负样本按 `P(w) ∝ count(w)^0.75` 采样，是实践中常用配置。

## R04

`demo.py` 的高层流程：

1. 构造固定种子语料（多个语义簇：royal/animal/fruit/vehicle）。
2. 建词表并将句子编码为词 ID。
3. 按窗口生成 `(center, context)` 训练词对。
4. 初始化 `W_in/W_out`，构建负采样分布。
5. 多轮 epoch 执行 SGD：每步做 1 正样本 + K 负样本更新。
6. 训练结束后合并并归一化向量得到最终词嵌入。
7. 输出近邻词与类比结果。
8. 执行内置断言检查并给出通过标记。

## R05

核心数据结构：
- `TrainingReport`：训练摘要（词表规模、样本对数量、epoch、最终平均损失）。
- `SkipGramWord2Vec`：主模型类，包含：
  - `word_to_id_ / id_to_word_`：词表映射；
  - `_w_in / _w_out`：输入与输出嵌入矩阵；
  - `_neg_probs`：负采样概率分布；
  - `embeddings_`：训练后可直接用于检索/类比的归一化向量。

## R06

正确性要点：
- 正样本项推动 `u_o^T v_c` 变大，使真实共现词对更相似；
- 负样本项推动 `u_n^T v_c` 变小，抑制“错误共现”；
- 同时维护 `W_in` 与 `W_out`，训练后合并二者可提升表示质量；
- 用余弦相似度与类比任务做最小验证，确保向量空间具备语义结构。

## R07

设：
- 训练词对数 `P`；
- 负样本数 `K`；
- 向量维度 `d`；
- 训练轮数 `E`。

复杂度近似为：
- 时间：`O(E * P * K * d)`（每个样本需要若干点积与向量更新）；
- 空间：`O(|V| * d)`（主要是两套嵌入矩阵 `W_in/W_out`）。

在本 MVP 中，语料较小，因此可在秒级训练完成。

## R08

边界与异常处理：
- 参数非法（如 `embedding_dim<=0`、`epochs<=0`）会抛 `ValueError`；
- 空语料或句长不足 2 会抛 `ValueError`；
- `min_count` 过滤后词表过小会拒绝训练；
- 训练后接口前置检查，未 `fit` 调用会抛 `RuntimeError`；
- 查询不存在词时抛 `KeyError`，避免静默返回错误结果。

## R09

MVP 取舍：
- 保留：Skip-gram + Negative Sampling 的核心数学与梯度更新。
- 省略：分层 softmax、子词建模（fastText）、大规模并行优化、磁盘流式语料。
- 目的：用最小实现展示算法本体，便于阅读、复现实验与后续扩展。

## R10

`demo.py` 主要函数职责：
- `build_toy_corpus`：构造固定种子、语义可分的训练语料。
- `SkipGramWord2Vec.fit`：词表构建、样本生成、参数初始化与训练总循环。
- `SkipGramWord2Vec._generate_skipgram_pairs`：基于窗口展开正样本词对。
- `SkipGramWord2Vec._sample_negatives`：按 `count^0.75` 分布采负样本并排除正标签。
- `SkipGramWord2Vec._sgd_step`：执行单步损失与梯度更新（核心算法）。
- `most_similar / analogy / similarity`：训练后向量空间查询接口。
- `run_quality_checks`：内置语义检查与断言门槛。
- `main`：串联训练、展示、验证。

## R11

运行方式：

```bash
cd Algorithms/数学-NLP-0334-词嵌入_-_Word2Vec
uv run python demo.py
```

无需命令行参数，无需交互输入。

## R12

输出字段说明：
- `Vocabulary size`：词表大小。
- `Training pairs`：窗口展开后的正样本词对数。
- `Epochs`：训练轮数。
- `Final average loss`：最后一轮平均训练损失。
- `Nearest to 'word'`：词的最近邻及余弦相似度。
- `Analogy: king - man + woman`：类比任务候选结果。
- `All checks passed.`：所有质量断言通过。

## R13

内置最小测试与质量门槛：
1. 损失必须为有限值（非 NaN/Inf）。
2. 词表规模与样本对数量需达到最低规模，防止退化语料。
3. 语义相似度检查：
   - `sim(king, queen)` 明显高于 `sim(king, banana)`；
   - `sim(dog, cat)` 明显高于 `sim(dog, train)`。
4. 类比检查：`king - man + woman` 的前若干候选中必须出现 `queen`。

## R14

关键参数与调参建议：
- `embedding_dim`：向量维度。增大可提升表达能力但更慢。
- `window_size`：上下文窗口。大窗口更偏主题，小窗口更偏句法。
- `negatives`：每个正样本配套的负样本数。增大通常更稳但更耗时。
- `learning_rate`：SGD 学习率，过大易震荡，过小收敛慢。
- `epochs`：训练轮数，语料小可适当增加。
- `min_count`：词频截断，影响词表大小与噪声词占比。

## R15

与相关方法对比：
- 对比 CBOW：
  - CBOW 用上下文预测中心词，训练更快；
  - Skip-gram 对低频词常更友好。
- 对比 GloVe：
  - GloVe基于全局共现矩阵分解；
  - Word2Vec-SGNS 基于局部窗口采样与在线优化。
- 对比 fastText：
  - fastText 引入子词 n-gram，能更好处理未登录词；
  - 本实现仅词级建模，结构更简单。

## R16

典型应用场景：
- 传统 NLP 特征工程（文本分类、聚类、关键词扩展）。
- 检索中的语义召回与查询扩展。
- 作为更复杂模型（RNN/CNN/Transformer）的初始化词向量。
- 教学与算法验证：快速理解分布式语义表示的基本机制。

## R17

可扩展方向：
- 引入子采样（subsampling）以降低高频词干扰。
- 增加学习率衰减与 mini-batch 向量化提升训练效率。
- 支持外部语料读取与分词流水线。
- 增加词向量可视化（PCA/t-SNE）与更系统评测集。
- 扩展到 CBOW、Hierarchical Softmax、fastText 子词版本。

## R18

`demo.py` 的源码级算法流程（8 步，非黑盒）如下：

1. `build_toy_corpus` 生成固定随机种子的多语义簇句子，保证训练可复现。  
2. `fit` 调用 `_build_vocab` 统计词频并构建 `word_to_id`，再把句子编码成 ID 序列。  
3. `_generate_skipgram_pairs` 用滑动窗口把每个中心词展开成多个 `(center, context)` 正样本。  
4. 初始化 `W_in/W_out`，并按 `count^0.75` 构造负采样分布 `_neg_probs`。  
5. 每个 SGD step：`_sample_negatives` 先采 K 个负样本（排除正上下文词）。  
6. `_sgd_step` 计算 `log sigma(u_o^T v_c)` 与 `sum log sigma(-u_n^T v_c)` 对应损失，并对 `v_c/u_o/u_n` 求梯度更新。  
7. 多轮训练结束后将 `W_in + W_out` 合并，并做行归一化得到最终 `embeddings_`。  
8. `main` 调用 `most_similar/analogy` 展示结果，再由 `run_quality_checks` 执行语义与类比断言，确认实现有效。
