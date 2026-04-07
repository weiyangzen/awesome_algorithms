# 词嵌入 - GloVe

- UID: `MATH-0335`
- 学科: `数学`
- 分类: `NLP`
- 源序号: `335`
- 目标目录: `Algorithms/数学-NLP-0335-词嵌入_-_GloVe`

## R01

GloVe（Global Vectors for Word Representation）是一种基于全局词共现统计的词向量学习方法。它不直接做“下一词分类”，而是先构造词-词共现强度，再通过加权最小二乘把共现关系压缩到低维向量空间。

本目录给出可运行、可审计的最小 MVP：
- 纯 NumPy 手写共现构建与训练循环；
- 使用 AdaGrad 优化 GloVe 目标函数；
- 内置相似度与类比检查，保证实现不是空壳。

## R02

问题定义：
- 输入：分词后的语料 `C = {w_1, w_2, ..., w_T}`。
- 中间表示：窗口统计得到的共现矩阵元素 `X_ij`（词 `j` 出现在词 `i` 上下文中的加权次数）。
- 输出：每个词的向量 `v_i in R^d`，用于相似度检索与线性语义运算。

目标是让向量内积与偏置之和，逼近 `log(X_ij)`，使“高共现词对”在嵌入空间中保持可解释的几何关系。

## R03

GloVe 经典目标函数：

`J = sum_{i,j} f(X_ij) * (w_i^T w~_j + b_i + b~_j - log(X_ij))^2`

其中：
- `w_i`：中心词向量，`w~_j`：上下文词向量；
- `b_i, b~_j`：对应偏置；
- `f(x)`：权重函数，常用
  `f(x) = (x/x_max)^alpha (x < x_max), 1 (x >= x_max)`。

本实现采用 `x_max=10.0, alpha=0.75`，并使用 AdaGrad 做参数更新。

## R04

`demo.py` 的总体流程：

1. 构造确定性玩具语料（多个语义簇：royal/animal/fruit/vehicle/city）。
2. 统计词频并建立词表映射。
3. 基于滑动窗口生成方向性共现 `X_ij`（含距离衰减 `1/distance`）。
4. 初始化 `w_i, w~_j, b_i, b~_j`。
5. 按 GloVe 加权平方误差逐对训练，并用 AdaGrad 自适应步长。
6. 合并 `W + W_context` 得到最终词向量并归一化。
7. 输出近邻词与类比结果。
8. 运行断言式质量检查。

## R05

核心数据结构：
- `TrainingReport`：记录 `vocab_size / cooc_pairs / epochs / final_avg_loss`。
- `GloveEmbeddings`：主模型类，包含：
  - `word_to_id_ / id_to_word_`：词表映射；
  - `_w_main / _w_context`：两套词向量参数；
  - `_b_main / _b_context`：偏置参数；
  - `embeddings_`：训练后可直接检索的归一化向量。

## R06

正确性关键点：
- 共现强度经过 `log(X_ij)` 映射，减弱极端频次对优化的主导。
- `f(X_ij)` 对稀疏和超高频共现做平衡，减少噪声主导训练。
- 同时维护“中心词向量 + 上下文词向量”，最终合并 `W + W~` 可提升稳定性。
- 通过相似度与类比断言验证语义结构是否成立（例如 `king-queen`、`king-man+woman≈queen`）。

## R07

复杂度分析（设 `|V|` 为词表大小，`Nnz` 为非零共现对数量，`d` 为维度，`E` 为轮数）：
- 时间复杂度：`O(E * Nnz * d)`（每个共现对做一次点积与向量更新）。
- 空间复杂度：`O(|V| * d + Nnz)`（两套向量与共现稀疏存储）。

本 MVP 语料较小，可在秒级跑完。

## R08

边界与异常处理：
- 参数非法（如 `embedding_dim<=0`、`x_max<=0`、`epochs<=0`）立即抛 `ValueError`。
- 空语料或长度不足 2 的句子会拒绝训练。
- `min_count` 过滤后词表太小时直接报错。
- 共现矩阵为空时报 `RuntimeError`，防止静默产出无意义向量。
- 查询 OOV（未登录词）时报 `KeyError`，避免误用。

## R09

MVP 取舍：
- 保留：GloVe 的核心数学（全局共现 + 加权平方误差 + 双向量合并）。
- 省略：超大语料流式构建、稀疏矩阵并行训练、GPU 训练与分布式优化。
- 目的：在最小代码规模内完整呈现算法闭环，便于审计和教学复现。

## R10

`demo.py` 主要函数职责：
- `build_toy_corpus`：构造可复现实验语料。
- `GloveEmbeddings.fit`：词表、共现、参数初始化与训练总循环。
- `_build_cooccurrence_pairs`：窗口统计得到 `(i,j,X_ij)`。
- `_weight_fn`：实现 GloVe 权重函数 `f(x)`。
- `most_similar / similarity / analogy`：训练后语义查询接口。
- `run_quality_checks`：断言式质量门槛验证。
- `main`：串联训练、展示与校验。

## R11

运行方式：

```bash
cd Algorithms/数学-NLP-0335-词嵌入_-_GloVe
uv run python demo.py
```

无需参数，无需交互输入。

## R12

输出字段说明：
- `Vocabulary size`：词表大小。
- `Co-occurrence pairs`：非零共现对数量（`Nnz`）。
- `Epochs`：训练轮数。
- `Final average loss`：最后一轮平均加权平方误差。
- `Nearest to 'word'`：词语最近邻与余弦相似度。
- `Analogy: king - man + woman`：类比任务候选结果。
- `All checks passed.`：所有内置质量断言通过。

## R13

内置最小测试与质量门槛：
1. 最终损失必须是有限值（非 NaN/Inf）。
2. 词表规模与共现对数达到最低阈值，防止退化语料。
3. 语义相似度检查：
   `sim(king, queen) > sim(king, banana)`。
4. 类别相似度检查：
   `sim(dog, cat) > sim(dog, train)`。
5. 类比检查：`king - man + woman` 的前若干候选中必须出现 `queen`。

## R14

关键参数与调参建议：
- `embedding_dim`：向量维度，越大表达力越强但更慢。
- `window_size`：共现窗口，越大越偏主题，越小越偏局部语法。
- `x_max`：权重函数拐点，控制高频共现的“封顶”程度。
- `alpha`：权重函数曲率，常见值为 `0.75`。
- `learning_rate`：学习率，过大易振荡，过小收敛慢。
- `epochs`：训练轮数，受语料规模与目标收敛程度影响。

## R15

与相关方法对比：
- 对比 Word2Vec-SGNS：
  SGNS 侧重局部采样预测；GloVe 直接拟合全局共现统计。
- 对比 LSA/矩阵分解：
  GloVe 使用专门的加权损失与双向量参数，通常更适合词语语义任务。
- 对比 fastText：
  fastText 有子词建模，OOV 处理更强；本实现为纯词级表示。

## R16

典型应用场景：
- 传统 NLP 特征工程（分类、聚类、关键词扩展）。
- 语义检索和查询扩展的底层向量特征。
- 教学场景中的词向量机制演示。
- 小规模语料的快速语义原型验证。

## R17

可扩展方向：
- 使用真实大语料并引入更高效的稀疏存储/并行构建共现矩阵。
- 加入 mini-batch 向量化训练，减少 Python 循环开销。
- 支持词频截断、停用词策略与子词增强。
- 增加评测基准（词相似度榜单、词类比数据集）。
- 增加 PCA/t-SNE 可视化辅助诊断训练质量。

## R18

`demo.py` 的源码级算法流程（8 步，非黑盒）如下：

1. `build_toy_corpus` 生成固定随机种子的多语义簇语料，确保结果可复现。  
2. `fit` 调用 `_build_vocab` 统计词频并构建 `word_to_id`，再将句子编码成整数 ID。  
3. `_build_cooccurrence_pairs` 用窗口扫描每个句子，对每个词对累计 `X_ij += 1/distance`，得到稀疏共现统计。  
4. 初始化两套向量 `W_main/W_context` 与偏置 `b_main/b_context`，并初始化 AdaGrad 累积平方梯度。  
5. 训练循环中逐个共现对计算 `pred = w_i^T w~_j + b_i + b~_j`，再与 `log(X_ij)` 比较得到误差 `diff`。  
6. 通过 `f(X_ij)` 加权误差，按 GloVe 目标对 `w_i, w~_j, b_i, b~_j` 求梯度并做 AdaGrad 更新。  
7. 训练结束后将 `W_main + W_context` 合并并行归一化，生成最终 `embeddings_`。  
8. `main` 调用 `most_similar/analogy` 展示语义结果，再由 `run_quality_checks` 做断言验收。  
