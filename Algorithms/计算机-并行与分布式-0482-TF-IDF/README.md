# TF-IDF

- UID: `CS-0320`
- 学科: `计算机`
- 分类: `并行与分布式`
- 源序号: `482`
- 目标目录: `Algorithms/计算机-并行与分布式-0482-TF-IDF`

## R01

TF-IDF（Term Frequency - Inverse Document Frequency）是经典文本特征加权方法：
- 用 TF 衡量词在当前文档中的重要性；
- 用 IDF 抑制全局高频“泛词”，突出区分度高的词。

本目录提供的是一个可运行、可审计的最小 MVP：核心计算由 `numpy + scipy.sparse` 手写完成，并且使用“分片 Map + 汇总 Reduce”的并行统计流程，体现并行与分布式场景中的常见计算分解方式。

## R02

问题定义：
- 输入：文档集合 `D = {d_1, d_2, ..., d_N}`。
- 中间量：每个词在文档内的词频 `tf(t, d)`，以及词跨文档出现次数 `df(t)`。
- 输出：稀疏矩阵 `X in R^{N x |V|}`，其中 `X[d,t]` 为词 `t` 在文档 `d` 的 TF-IDF 值。

目标是把文本离散符号映射到可计算向量空间，服务检索、相似度、聚类和下游模型。

## R03

本实现使用平滑 IDF 与 L2 归一化，公式为：

- `tf(t, d) = count(t, d)`（原始词频）
- `idf(t) = log((1 + N) / (1 + df(t))) + 1`
- `w(t, d) = tf(t, d) * idf(t)`
- `x_d = w_d / ||w_d||_2`

这样可避免 `df=0` 的数值问题，并保证文档向量可直接用点积表示余弦相似度。

## R04

`demo.py` 的总体流程：

1. 构造分布式系统主题的确定性语料。
2. 正则分词并转成 token 列表。
3. 按分片把文档分配给并行 map 任务（线程池）。
4. 每个分片统计：文档词频 Counter 与本地 DF。
5. reduce 阶段合并为全局 `doc_term_counts` 与 `global_df`。
6. 构建 CSR 稀疏 TF 矩阵，计算 IDF 并得到 TF-IDF。
7. 对每行做 L2 归一化。
8. 输出每个文档 top-k 关键词，并运行质量断言。

## R05

核心数据结构：
- `TfidfReport`：记录 `n_documents / vocab_size / nnz / num_partitions / max_abs_diff_vs_sklearn`。
- `MapReduceTfidf`：核心实现类，包含：
  - `vocabulary_`：词表（按词典序）；
  - `term_to_id_`：词到列索引映射；
  - `idf_`：IDF 向量；
  - `matrix_`：最终 CSR TF-IDF 矩阵。
- map 阶段中间结构：`list[(doc_id, Counter)] + Counter(df)`。

## R06

正确性关键点：
- DF 统计使用“文档是否出现”而不是总词频，避免 IDF 被重复词误导。
- map/reduce 分离后再全局聚合，保证并行分片不会破坏全局统计定义。
- 归一化后每行向量模长为 1（非空行），相似度计算稳定。
- 用 `scikit-learn` 的同参数 `TfidfVectorizer` 做数值对照，最大绝对误差接近机器精度，验证手写实现正确。

## R07

复杂度分析（`N`=文档数，`L`=总 token 数，`V`=词表大小，`nnz`=稀疏非零元数）：
- 分词与 map 统计：`O(L)`。
- reduce 聚合：`O(nnz)`。
- 构建 TF 稀疏矩阵：`O(nnz)`。
- 计算 IDF：`O(V)`。
- TF-IDF 乘权与 L2 归一化：`O(nnz + N)`。
- 总体时间：`O(L + nnz + V)`；空间：`O(nnz + V + N)`。

## R08

边界与异常处理：
- `num_partitions <= 0` 或 `num_workers <= 0` 直接 `ValueError`。
- 文档列表为空或存在空文档时拒绝运行。
- 聚合后词表为空则抛 `RuntimeError`，防止产出无意义矩阵。
- 质量检查中若控制词缺失、数值非有限、相似度关系异常，都会触发断言失败。

## R09

MVP 取舍：
- 保留：TF/DF/IDF 的核心定义、稀疏矩阵实现、并行 map + reduce 聚合、可复现实验。
- 省略：分布式集群通信、增量索引持久化、复杂分词与语言特定预处理。
- 原则：优先小而诚实的最小闭环，保证代码易读、可验证、可扩展。

## R10

`demo.py` 主要函数职责：
- `tokenize`：正则分词。
- `_map_partition`：单分片 map 统计。
- `MapReduceTfidf.fit_transform`：全流程训练入口。
- `MapReduceTfidf._build_statistics_map_reduce`：并行 map + reduce 聚合。
- `MapReduceTfidf._l2_normalize_rows`：稀疏矩阵行归一化。
- `top_terms_table`：生成每文档 top-k 词表。
- `validate_against_sklearn`：与 sklearn 数值对齐校验。
- `run_quality_checks`：断言式质量门槛。

## R11

运行方式：

```bash
cd Algorithms/计算机-并行与分布式-0482-TF-IDF
uv run python demo.py
```

无需命令行参数，无需交互输入。

## R12

输出字段说明：
- `Documents`：文档总数 `N`。
- `Vocabulary size`：词表大小 `|V|`。
- `Non-zero entries`：稀疏矩阵非零元 `nnz`。
- `Partitions`：map 分片数量。
- `Max abs diff vs sklearn`：手写实现与 sklearn 对照的最大绝对误差。
- `Top TF-IDF terms per document`：每个文档 TF-IDF 最高的词及权重。
- `All checks passed.`：全部内置校验通过。

## R13

内置最小测试与质量门槛：
1. 结果矩阵行列规模必须与文档数、词表数一致。
2. 矩阵值必须全为有限数（非 NaN/Inf）。
3. 稀有词 `consensus` 的 IDF 必须高于高频词 `distributed`。
4. 语义相似性检查：`MapReduce Primer` 与 `Spark Pipeline` 的相似度高于与 `Image Training` 的相似度。
5. 与 sklearn 基准的最大绝对误差必须不超过 `1e-12`。

## R14

关键参数与调参建议：
- `num_partitions`：分片数，过小并行度不足，过大调度开销增加。
- `num_workers`：map 并行 worker 数，建议不超过机器核数与分片数。
- 分词规则（`TOKEN_PATTERN`）：会直接影响词表与 DF 统计，是 TF-IDF 质量的重要前置。
- 词表裁剪策略（本 MVP 未启用）：真实项目可增加 `min_df/max_df` 以降噪。

## R15

相关方法对比：
- 对比 Bag-of-Words：TF-IDF 增加 IDF 抑制，区分度更高。
- 对比 BM25：BM25 进一步考虑文档长度归一与饱和项，检索效果通常更好但实现更复杂。
- 对比神经嵌入（如 BERT 向量）：TF-IDF 成本低、可解释强；语义泛化能力通常弱于深度模型。

## R16

典型应用场景：
- 搜索与召回阶段的关键词权重构建。
- 文档相似度去重与聚类初筛。
- 主题建模前的高权重词筛查。
- 传统机器学习文本分类（如线性模型）特征输入。

## R17

可扩展方向：
- 把 map/reduce 扩展到多进程或真实分布式执行框架。
- 增加 `min_df/max_df`、停用词、n-gram、子词切分。
- 支持增量更新（新增文档时仅局部更新 DF/IDF）。
- 用倒排索引与块压缩优化大规模稀疏存储。
- 加入标准检索评测（MAP/NDCG）与性能基准。

## R18

`demo.py` 的源码级算法流程（8 步，非黑盒）如下：

1. `build_demo_corpus` 生成固定语料，`tokenize` 把文本转换为 token 序列。  
2. `fit_transform` 调用 `_build_statistics_map_reduce`，先按 `num_partitions` 切分文档索引。  
3. map 阶段由 `_map_partition` 并行处理每个分片：输出每篇文档词频 Counter，以及“词是否在文档出现”的本地 DF 计数。  
4. reduce 阶段汇总各分片结果，得到全局 `doc_term_counts` 与 `global_df`。  
5. 按词典序建立 `vocabulary` 与列索引，遍历 `doc_term_counts` 组装 CSR 稀疏 TF 矩阵。  
6. 根据 `idf(t)=log((1+N)/(1+df(t)))+1` 计算 IDF 向量，并执行逐列乘权得到 TF-IDF。  
7. `_l2_normalize_rows` 对每行做 L2 归一化，得到可直接用于余弦相似度的向量。  
8. `validate_against_sklearn` 做数值对照，`run_quality_checks` 再执行 IDF 与语义关系断言，最终在 `main` 输出结果。  
