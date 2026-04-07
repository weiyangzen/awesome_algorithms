# t-SNE

- UID: `CS-0105`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `234`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0234-t-SNE`

## R01

t-SNE（t-Distributed Stochastic Neighbor Embedding）是用于高维数据可视化的非线性降维算法。它的核心目标不是“保留全局欧氏距离”，而是尽量保留局部邻域结构，使高维中相近样本在低维（通常 2D）中仍然靠近。

本条目提供一个可运行、可审计的最小 MVP：
- 用 NumPy + SciPy 手写 exact t-SNE（高维高斯相似度 + 低维 Student-t 相似度 + KL 梯度下降）；
- 用 sklearn `TSNE(method="exact")` 做数值与结构对照；
- 输出信任度（trustworthiness）、10-NN 重叠率、KL 收敛轨迹与类别质心表。

## R02

问题定义：
- 输入：高维样本矩阵 `X in R^(n*d)`。
- 输出：二维嵌入坐标 `Y in R^(n*2)`。
- 目标：让低维分布 `Q` 逼近高维分布 `P`，最小化 `KL(P || Q)`。

在该 MVP 中，数据来自 `sklearn.datasets.load_digits`，先做标准化和 PCA(30 维) 预处理，再分别运行手写 t-SNE 与 sklearn exact t-SNE 并对比。

## R03

关键数学定义：

1. 高维条件概率（每个样本自适应带宽）：
`p_{j|i} = exp(-||x_i-x_j||^2 / (2*sigma_i^2)) / sum_{k!=i} exp(-||x_i-x_k||^2 / (2*sigma_i^2))`
其中 `sigma_i` 通过二分搜索满足给定 perplexity。

2. 对称联合概率：
`p_{ij} = (p_{j|i} + p_{i|j}) / (2n), p_{ii}=0`

3. 低维 Student-t 相似度：
`q_{ij} = (1 + ||y_i-y_j||^2)^(-1) / sum_{k!=l}(1 + ||y_k-y_l||^2)^(-1), q_{ii}=0`

4. 优化目标：
`C = KL(P||Q) = sum_{i!=j} p_{ij} * log(p_{ij}/q_{ij})`

5. 梯度：
`dC/dy_i = 4 * sum_j (p_{ij} - q_{ij}) * (1 + ||y_i-y_j||^2)^(-1) * (y_i - y_j)`

## R04

`demo.py` 的执行流程：

1. 加载 digits 数据并随机抽样 420 个样本。
2. 标准化后做 PCA 到 30 维，降低噪声和计算开销。
3. 手写 exact t-SNE：构建 `P`，执行带 early exaggeration 与 momentum 的梯度下降。
4. 调用 sklearn exact t-SNE（非 Barnes-Hut）作为参考实现。
5. 计算并输出：`trustworthiness@10`、两嵌入 10-NN 重叠率、训练快照和类别质心。
6. 执行内置质量门槛，全部通过后输出 `All checks passed.`。

## R05

核心数据结构：
- `TSNEConfig`：超参数集合（perplexity、learning rate、迭代轮数、early exaggeration 等）。
- `IterationRecord`：每一轮记录 `iteration/kl/grad_norm/momentum`。
- `TSNEResult`：训练产物（最终嵌入、历史轨迹、最终 KL、高维相似度矩阵 `P`）。

这些结构让 MVP 具备可解释性和可审计性，而不是只返回一组坐标。

## R06

正确性关键点：
- 高维概率 `P` 通过逐样本二分搜索精确匹配 perplexity，而不是固定全局带宽；
- 低维概率 `Q` 使用重尾 Student-t 分布，缓解 crowding problem；
- 目标函数使用 `KL(P||Q)`，强调“高维邻居不要被拆散”；
- 训练采用 early exaggeration + momentum，提升早期簇分离和优化稳定性；
- 与 sklearn exact 版本做结构性对照，防止实现偏离。

## R07

复杂度分析（exact t-SNE）：
- 计算高维距离矩阵：`O(n^2 * d)`；
- 概率矩阵构建（含二分搜索）：约 `O(n^2 * S)`，`S` 为二分迭代步数；
- 每次梯度迭代需要全样本对交互：`O(n^2)`；
- 总训练复杂度：`O(T * n^2)`；
- 空间复杂度：`O(n^2)`（主要为 `P/Q` 与距离矩阵）。

因此本 MVP 适合中小规模样本演示，不追求大规模近似加速。

## R08

边界与异常处理：
- 校验 `X` 维度、有限值约束（拒绝 `NaN/Inf`）；
- 限制 `perplexity` 在 `(1, n-1)` 范围内；
- 校验学习率、迭代次数、early exaggeration 参数合法；
- 对低维归一化分母下溢进行显式报错，避免 silent failure；
- `kNN` 对比函数检查样本数一致、`k` 合法。

## R09

MVP 取舍：
- 保留：exact 算法主干、可追踪训练轨迹、参考实现对照与质量门槛；
- 省略：Barnes-Hut / FFT 近似、多线程、GPU、参数网格搜索、可视化绘图文件输出；
- 原则：优先小而完整、能验证算法机制，不做工程化大系统。

## R10

`demo.py` 主要函数职责：
- `compute_joint_probabilities`：构建高维对称概率矩阵 `P`；
- `_binary_search_conditional_probs`：按 perplexity 逐样本搜索带宽；
- `kl_and_gradient`：计算 KL 与梯度；
- `tsne_exact`：主训练循环（early exaggeration + momentum + adaptive gains）；
- `history_snapshot`：提取收敛过程关键帧；
- `knn_overlap`：比较两个嵌入的局部邻域一致性；
- `run_quality_checks`：执行门槛验收；
- `main`：串联数据准备、训练、对照、输出。

## R11

运行方式：

```bash
cd Algorithms/计算机-机器学习／深度学习-0234-t-SNE
uv run python demo.py
```

脚本无需交互输入，直接输出对照结果与验收状态。

## R12

输出字段说明：
- `samples/pca_features/classes`：样本量、输入维度和类别数；
- `[Metrics]`：
  - `custom_exact.final_kl`：手写实现最终 KL；
  - `trustworthiness@10`：局部结构保真度（越高越好）；
- `10-NN overlap(custom vs sklearn)`：两个嵌入在 10 近邻上的平均交集比例；
- `[Custom Training Snapshot]`：若干关键迭代点的 KL、梯度范数、动量；
- `[Class Centroids in 2D Embedding Space]`：每个类别在两种嵌入中的质心坐标；
- `All checks passed.`：内置质量检查全部通过。

## R13

内置最小测试与质量门槛：
1. 训练历史长度至少 50，避免异常过早停止；
2. KL 序列必须有限且最终值低于初始值；
3. 手写与 sklearn 的 `trustworthiness@10` 均需 >= 0.90；
4. 两者 trustworthiness 差距不超过 0.08；
5. 两嵌入 10-NN 重叠率需 >= 0.30。

## R14

关键参数与调参建议：
- `perplexity`：控制局部邻域规模，常见范围 5~50；
- `learning_rate`：过小收敛慢，过大可能震荡；
- `max_iter`：迭代轮数，影响收敛充分性；
- `early_exaggeration` 与 `early_exaggeration_iters`：控制早期簇分离强度；
- `momentum`（初始/后期）：影响轨迹平滑性和收敛速度。

建议先固定 `perplexity=30`，再调学习率和迭代轮数观察 KL 下降曲线是否稳定。

## R15

与相关方法对比：
- 对比 PCA：PCA 强调全局线性投影，t-SNE 更强调局部邻域保真；
- 对比 UMAP：UMAP 常在大规模与速度上更有优势，t-SNE 在局部视觉聚团上常更直观；
- 对比直接黑盒调用：本实现显示了概率构建和梯度细节，便于教学和审计。

## R16

典型应用场景：
- 高维特征的探索式可视化（图像、文本嵌入、表征学习中间层）；
- 聚类前的数据形态检查与异常样本直观定位；
- 论文/报告中展示“同类是否局部聚集”的直观证据。

注意：t-SNE 结果主要用于可视化解释，不应直接当作下游监督学习特征替代。

## R17

可扩展方向：
- 增加 Barnes-Hut 或 FFT 近似以支持更大规模样本；
- 支持多随机种子复现实验并统计稳定性区间；
- 增加参数扫描（perplexity、learning_rate）与自动报告；
- 输出图像文件并叠加类别标注、密度等可视化层；
- 增加与 UMAP/PCA 的统一基准脚本。

## R18

本条目避免把 sklearn 当黑盒，源码级流程可拆为 8 步：

1. 在 `demo.py::compute_joint_probabilities` 中先用成对距离构建每个样本到其余样本的距离行。  
2. `demo.py::_binary_search_conditional_probs` 对每一行执行二分搜索，调整 `beta=1/(2*sigma^2)` 使熵匹配目标 perplexity。  
3. `demo.py::compute_joint_probabilities` 把条件概率对称化并归一化，得到 `P`（对角为 0）。  
4. `demo.py::tsne_exact` 初始化 `Y`（PCA + 小尺度），进入迭代；前期使用 `P * early_exaggeration`。  
5. `demo.py::kl_and_gradient` 计算 Student-t 分布下的 `Q`、`KL(P||Q)` 与矩阵化梯度。  
6. `demo.py::tsne_exact` 用 adaptive gains + momentum 更新 `Y`，并在每轮后中心化嵌入。  
7. 参考实现侧，`sklearn.manifold._t_sne` 的主路径是：`TSNE.fit_transform -> _fit -> _tsne -> _gradient_descent`，其中 exact 模式会通过 `_joint_probabilities` 构建 `P`，并在 `_kl_divergence` 中计算目标和梯度。  
8. `demo.py::run_quality_checks` 将手写实现与 sklearn exact 做定量对照（trustworthiness 与 kNN 重叠），确认流程与结果一致性。  
