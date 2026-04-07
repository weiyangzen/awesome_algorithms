# k-means++

- UID: `MATH-0230`
- 学科: `数学`
- 分类: `机器学习`
- 源序号: `230`
- 目标目录: `Algorithms/数学-机器学习-0230-k-means++`

## R01

`k-means++` 是 `k-means` 的初始化改进算法。核心思想是：
- 第一个中心随机选；
- 之后每个中心按“到最近已选中心的平方距离”(`D(x)^2`)做加权采样；
- 再进入标准 Lloyd 迭代（分配样本 + 更新中心）。

这个策略通常比完全随机初始化更稳定，能显著降低陷入差局部最优的概率。

## R02

问题定义（本目录实现）：
- 输入：
  - 数据矩阵 `X in R^{n x d}`；
  - 聚类数 `k`；
  - 随机种子 `seed`；
  - 最大迭代轮数 `max_iter`；
  - 中心位移阈值 `tol`。
- 输出：
  - 聚类中心 `centers in R^{k x d}`；
  - 样本簇标记 `labels in {0,...,k-1}^n`；
  - 最终惯性值 `inertia = sum_i ||x_i - mu_{label_i}||^2`；
  - 迭代历史 `history`（每轮惯性、中心位移、空簇数量）。

`demo.py` 生成固定合成数据并直接运行，不需要交互输入。

## R03

关键数学关系：

1. 初始化采样概率（`k-means++`）：
   - 对未被选为中心的样本 `x_i`，令 `D(x_i)^2 = min_c ||x_i - mu_c||^2`；
   - 采样概率 `p_i = D(x_i)^2 / sum_j D(x_j)^2`。
2. 分配步骤：
   - `label_i = argmin_c ||x_i - mu_c||^2`。
3. 更新步骤：
   - `mu_c = mean({x_i | label_i = c})`。
4. 目标函数（惯性）：
   - `J = sum_i ||x_i - mu_{label_i}||^2`。
5. 终止条件：
   - `||M_new - M_old||_F <= tol` 或达到 `max_iter`。

## R04

算法流程（高层）：
1. 检查输入矩阵与参数合法性。  
2. 用 `k-means++` 选出 `k` 个初始中心。  
3. 对所有样本计算到各中心的平方距离并分配标签。  
4. 依据标签重新计算每个簇中心。  
5. 若出现空簇，用随机样本重置该簇中心。  
6. 记录本轮 `inertia`、中心位移、空簇数。  
7. 若中心位移小于阈值则停止，否则继续。  
8. 输出最终中心、标签、迭代历史与评估指标。

## R05

核心数据结构：
- `X: np.ndarray`，形状 `(n_samples, n_features)`。  
- `centers: np.ndarray`，形状 `(k, n_features)`。  
- `labels: np.ndarray`，长度 `n_samples`，取值 `0..k-1`。  
- `HistoryItem = (iter, inertia, center_shift, empty_clusters)`：
  - `iter`：轮次；
  - `inertia`：当前轮的平方误差和；
  - `center_shift`：本轮中心位移范数；
  - `empty_clusters`：本轮空簇数量。
- `result: dict`：返回 `centers/labels/inertia/n_iter/history/init_indices`。

## R06

正确性要点：
- `k-means++` 的 `D(x)^2` 采样让新中心更倾向远离已有中心，提升初值质量。  
- 每轮“分配 + 更新”不会增加 `k-means` 目标（理论上单调不增，数值误差除外）。  
- 代码在每轮显式记录 `inertia`，可观察收敛趋势。  
- 空簇分支避免 `mean(empty)` 导致 `nan`，保证后续迭代可继续。  
- 使用固定随机种子，保证结果可复现、可审计。

## R07

复杂度分析（`n` 样本、`d` 维、`k` 簇、`T` 轮）：
- 初始化（`k-means++`）：每加一个中心需要一次到新中心距离更新，约 `O(n d)`，共 `k` 次，约 `O(n k d)`。  
- 单轮 Lloyd：
  - 距离与分配：`O(n k d)`；
  - 更新中心：`O(n d)`（按簇求均值）；
  - 主导项仍是 `O(n k d)`。
- 总时间复杂度：`O(n k d + T n k d)`，常写作 `O(T n k d)`。  
- 空间复杂度：
  - 数据与中心 `O(n d + k d)`；
  - 距离矩阵（本实现显式构造）`O(n k)`。

## R08

边界与异常处理：
- `X` 非二维、为空、含 `nan/inf`：抛 `ValueError`。  
- `k <= 0` 或 `k > n_samples`：抛 `ValueError`。  
- `max_iter <= 0` 或 `tol < 0`：抛 `ValueError`。  
- `std <= 0` 或样本数配置非正（合成数据生成时）：抛 `ValueError`。  
- `k-means++` 若出现总权重近零（样本几乎重合），回退为随机选点，避免除零。

## R09

MVP 取舍：
- 只依赖 `numpy`，不调用 `scikit-learn` 的黑盒聚类器。  
- 聚焦核心流程：`k-means++` 初始化 + Lloyd 主循环 + 空簇处理。  
- 不实现多次重启（`n_init`）、并行化、加速索引结构（如 KD-Tree）。  
- 评估使用合成数据与可解释指标（中心 RMSE、标签最优置换准确率）。

## R10

`demo.py` 主要函数职责：
- `check_data_matrix`：数据合法性检查。  
- `squared_distances_to_centers`：批量计算样本到中心平方距离。  
- `assign_labels`：最近中心分配。  
- `kmeans_plus_plus_init`：实现 `D(x)^2` 加权采样初始化。  
- `recompute_centers`：均值更新与空簇重置。  
- `kmeans_fit`：完整训练循环与历史记录。  
- `make_blob_data`：生成固定可复现合成数据。  
- `best_center_rmse`：中心与真值中心最优排列误差。  
- `best_label_accuracy`：标签最优置换准确率。  
- `run_case/main`：组织案例、输出日志与汇总结果。

## R11

运行方式：

```bash
cd Algorithms/数学-机器学习-0230-k-means++
python3 demo.py
```

脚本无交互输入，直接打印两组案例的过程与汇总。

## R12

输出字段说明：
- `iter`：迭代轮次。  
- `inertia`：当前轮簇内平方误差和。  
- `center_shift`：中心整体位移（Frobenius 范数）。  
- `empty_clusters`：该轮空簇个数。  
- `init_indices`：`k-means++` 初始化时选中的样本下标。  
- `final inertia`：最终目标值。  
- `center RMSE`：预测中心与真实中心最优匹配下的均方根误差。  
- `label accuracy`：簇标签最优置换后的准确率。  
- `Summary`：最大中心误差、最小标签准确率、平均迭代轮次与通过标记。

## R13

建议最小测试集（脚本已内置）：
- `Well-separated 2D blobs (k=3)`：簇间距大，验证初始化和收敛稳定性。  
- `Moderately-overlapping 2D blobs (k=4)`：存在重叠，验证鲁棒性和指标退化可解释性。

建议补充测试：
- 极端重复样本（检查 `total <= 1e-18` 分支）；  
- `k = n_samples`（每个点成为独立簇）；  
- 高维稀疏场景（观察距离计算开销）。

## R14

可调参数：
- `k`：簇数量。  
- `seed`：随机种子（影响初始化与空簇重置）。  
- `max_iter`：最大迭代轮次。  
- `tol`：中心位移阈值。  
- `std`（数据构造参数）：控制簇重叠度。

调参建议：
- 若结果不稳定，先固定 `seed` 再比较；  
- 若收敛过慢，增大 `tol` 或减小 `k`；  
- 若欠拟合（簇过粗），增大 `k` 并观察 `inertia` 降幅与解释性。

## R15

与相关方法对比：
- 对比随机初始化 `k-means`：
  - `k-means++` 初始中心更分散，通常更快收敛、结果更稳。  
- 对比 `k-medoids`：
  - `k-means` 用均值，计算快；`k-medoids` 用样本点代表，抗离群点更强但成本更高。  
- 对比 GMM（EM）：
  - GMM 是软聚类并建模协方差，表达力更强；`k-means++` 更简单、训练更快。

## R16

典型应用场景：
- 用户分群与画像初始化。  
- 图像颜色量化（调色板压缩）。  
- 向量检索前的粗聚类分桶。  
- 作为更复杂模型（如 GMM）的初始化步骤。

## R17

可扩展方向：
- 增加 `n_init` 多次重启并选最优 `inertia`。  
- 使用 `k-means||` 做大规模并行初始化。  
- 增量式/小批量 `mini-batch k-means` 提升大数据吞吐。  
- 引入 `sample_weight` 支持加权样本。  
- 加入肘部法、轮廓系数等自动选 `k` 评估流程。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 定义两组固定二维高斯簇配置，包含真实中心、样本数、噪声强度和随机种子。  
2. `run_case` 调用 `make_blob_data` 生成样本与真实标签，再调用 `kmeans_fit` 训练。  
3. `kmeans_fit` 先做输入校验，然后调用 `kmeans_plus_plus_init` 生成 `k` 个初始中心。  
4. `kmeans_plus_plus_init` 在每一轮根据当前 `closest_d2` 构造概率 `p_i = D(x_i)^2 / sum D^2`，采样下一个中心并更新 `closest_d2`。  
5. 回到 `kmeans_fit` 主循环：`assign_labels` 计算 `(n,k)` 距离矩阵并做 `argmin` 获得最近中心标签，同时计算 `inertia`。  
6. `recompute_centers` 对每个簇求均值更新中心；如果簇为空则随机重置中心并记录空簇计数。  
7. 计算 `center_shift = ||M_new - M_old||`，写入 `history`；若 `center_shift <= tol` 则停止，否则继续下一轮。  
8. 训练结束后，`run_case` 通过 `best_center_rmse` 与 `best_label_accuracy`（均做簇置换最优匹配）输出可解释评估，`main` 汇总并给出 `pass_flag`。
