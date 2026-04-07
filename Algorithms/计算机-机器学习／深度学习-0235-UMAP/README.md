# UMAP

- UID: `CS-0106`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `235`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0235-UMAP`

## R01

本条目实现一个“可解释、可运行、非黑盒”的 UMAP 最小 MVP。重点不是一行调用 `umap-learn`，而是把核心链路拆开实现：
1. 高维 kNN 图构建。
2. fuzzy simplicial set 概率图构建。
3. 低维谱初始化。
4. 带负采样的交叉熵式优化。

## R02

UMAP 的目标：在降维后尽量保留局部邻域结构，同时允许全局结构做适度弯曲。可把它理解为“把高维流形上的邻域关系压缩到低维平面”的概率匹配问题。

## R03

核心对象与符号：
- 原始数据：`X in R^(N x D)`
- 每个样本的近邻：`N_k(i)`
- 局部尺度参数：`sigma_i`
- 局部连接偏移：`rho_i`
- 高维边权（定向）：`p_{i->j} = exp(-(d(i,j)-rho_i)/sigma_i)`（若 `d(i,j)<=rho_i` 则为 1）
- 对称边权：`w_ij = p_{i->j} + p_{j->i} - p_{i->j}p_{j->i}`
- 低维相似度：`q_ij = 1 / (1 + a * ||y_i-y_j||^(2b))`

## R04

`demo.py` 输入/输出约定：
- 输入：脚本内置 `sklearn.datasets.load_digits`（1797 个样本，64 维）。
- 输出：
  1. 终端打印关键指标（`trustworthiness@10`、`neighbor_recall@10`）；
  2. 导出 `embedding.csv`，包含 `id/label/umap_x/umap_y`；
  3. 无需任何交互参数。

## R05

实现采用的工程参数：
- `n_neighbors=15`
- `n_components=2`
- `min_dist=0.1`
- `spread=1.0`
- `n_epochs=120`
- `negative_sample_rate=3`
- `random_state=20260407`

这些值用于演示流程连通性，不追求对全部数据集最优。

## R06

高维近邻图阶段：
- 用 `sklearn.neighbors.NearestNeighbors` 求每个样本的 `k+1` 邻居（包含自身）；
- 去掉第 1 列自身索引后得到 `k` 个真实邻居；
- 距离度量使用欧氏距离，得到 `knn_indices` 与 `knn_distances`。

## R07

平滑 kNN 距离（`smooth_knn_dist`）阶段：
- 对每个点二分搜索 `sigma_i`，使
`sum_j exp(-(d_ij-rho_i)/sigma_i) ≈ log2(k)`；
- `rho_i` 由局部连通性决定（默认连接到最近非零邻居）；
- 这一步让稠密区和稀疏区各自拥有不同“感知半径”，是 UMAP 适应非均匀密度的关键。

## R08

fuzzy simplicial set 构建：
1. 用 `sigma_i/rho_i` 把每条有向边转成概率强度 `p_{i->j}`；
2. 将有向图做并集对称化：
`W = P + P^T - P ⊙ P^T`；
3. 结果保存为 SciPy 稀疏矩阵，后续优化只在非零边上采样。

## R09

低维概率曲线参数 `a,b`：
- 低维核函数用 `q(d)=1/(1+a*d^(2b))`；
- `a,b` 不是写死，而是通过 `scipy.optimize.curve_fit` 拟合到目标曲线：
  - `d < min_dist` 区间近似保持 1；
  - `d >= min_dist` 区间指数衰减。

这样可把 `min_dist/spread` 直接映射到低维势场形状。

## R10

初始化阶段：
- 由图拉普拉斯 `L = I - D^{-1/2} W D^{-1/2}` 做特征分解；
- 取最小非平凡特征向量作为初始坐标（谱初始化）；
- 若特征分解失败，则回退到小噪声随机初始化，保证脚本稳健可跑。

## R11

优化阶段（NumPy + Torch 两段）：
1. NumPy 段：按边权采样正样本边，执行吸引梯度；对随机负样本执行排斥梯度；
2. Torch 段：对同一边集合做小批量交叉熵式精修（`Adam`），提高收敛平滑度。

这两段都显式写出了梯度与损失，不依赖黑盒降维 API。

## R12

时间复杂度（近似）：
- kNN 搜索：`O(N * k * D)`（实现依赖 sklearn 后端）
- 平滑半径搜索：`O(N * k * I)`，`I` 为二分迭代次数
- 稀疏图构建：`O(N * k)`
- 谱初始化：稀疏特征分解近似 `O(E)` 到 `O(E * t)`（`E` 为边数）
- SGD 优化：`O(epochs * samples_per_epoch * (1 + negative_sample_rate))`

## R13

数值稳定策略：
- 所有距离计算加 `1e-6` 防止除零；
- 梯度做 `clip([-4,4])`，避免爆炸；
- 每轮优化后做中心化 `Y -= mean(Y)`，抑制整体漂移；
- 断言下界：`trustworthiness > 0.80`、`neighbor_recall > 0.25`。

## R14

依赖栈分工：
- `numpy`：核心向量计算与 SGD 更新
- `scipy`：稀疏图和 `curve_fit`/`eigsh`
- `pandas`：结果表格与 CSV 导出
- `scikit-learn`：数据集、kNN、trustworthiness 指标
- `torch`：末段小批量可微精修

## R15

运行方式：

```bash
uv run python Algorithms/计算机-机器学习／深度学习-0235-UMAP/demo.py
```

或在该目录下：

```bash
uv run python demo.py
```

运行后会在当前目录生成 `embedding.csv`。

## R16

结果解读建议：
- `trustworthiness@10` 越高，说明低维邻域更忠实于高维近邻；
- `neighbor_recall@10` 衡量“低维前 10 邻居里有多少来自高维前 10 邻居”；
- 两者都不是分类准确率，它们评估的是几何结构保持效果。

## R17

MVP 边界与可扩展项：
- 这是教学化简版，不包含 `umap-learn` 的全部工程细节（如更完整的 epoch 调度与并行策略）；
- 可扩展方向：
1. 支持监督式 UMAP（标签图融合）；
2. 增加多种度量（cosine、manhattan）；
3. 引入更精细的负采样计划与 early stopping；
4. 增加可视化输出（散点图）与批量数据接口。

## R18

`demo.py` 的源码级算法流程可拆为 8 步：
1. `main()` 加载 digits 数据，构建 `SimpleUMAPConfig`。  
2. `simple_umap_fit_transform()` 调用 `NearestNeighbors` 得到 `knn_idx/knn_dist`。  
3. `smooth_knn_dist()` 对每个样本二分搜索 `sigma_i` 并计算 `rho_i`。  
4. `compute_fuzzy_simplicial_set()` 把定向概率边转成对称稀疏图 `W`。  
5. `find_ab_params()` 用 `curve_fit` 求低维曲线参数 `a,b`。  
6. `spectral_layout()` 对图拉普拉斯做稀疏特征分解，得到初始二维坐标。  
7. `optimize_embedding_numpy()` 执行正负采样更新；`refine_with_torch()` 做小批量可微精修。  
8. 回到 `main()` 计算 `trustworthiness/neighbor_recall`，导出 `embedding.csv` 并做断言。  

第三方库内部流程追踪（非黑盒声明）：
- `NearestNeighbors` 负责近邻索引检索，算法链路是“建索引 -> 查询邻居 -> 返回距离/索引”；
- `curve_fit` 负责非线性最小二乘迭代，链路是“当前参数 -> 残差评估 -> 参数更新”；
- `torch.optim.Adam` 负责“前向损失 -> 反向梯度 -> 参数更新”。

核心 UMAP 逻辑（图权重定义、对称化、优化目标、采样更新）在本目录源码中均为显式实现。
