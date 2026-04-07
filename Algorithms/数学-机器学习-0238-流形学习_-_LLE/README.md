# 流形学习 - LLE

- UID: `MATH-0238`
- 学科: `数学`
- 分类: `机器学习`
- 源序号: `238`
- 目标目录: `Algorithms/数学-机器学习-0238-流形学习_-_LLE`

## R01

LLE（Locally Linear Embedding，局部线性嵌入）是一种经典非线性流形学习方法。核心思想是：
- 高维数据在局部邻域内通常可近似为线性结构；
- 先在高维空间中学习每个点由其邻居线性重构的权重；
- 再在低维空间中寻找一组坐标，尽量保留这些重构权重关系。

与“直接保持距离”的方法不同，LLE更强调“局部线性重构关系”的不变性。

## R02

问题形式：
- 输入：样本矩阵 `X in R^(n x d)`；
- 输出：低维嵌入 `Y in R^(n x m)`，常见 `m=2`；
- 目标：在降维后仍保持每个样本由其 `k` 近邻线性表示的结构。

本目录 `demo.py` 使用 swiss-roll 高维扰动数据作为演示数据，固定随机种子、无交互输入。

## R03

LLE 的两阶段数学要点：

1. 局部重构权重学习：
   - 对每个点 `x_i` 及其邻域 `N(i)`，解约束最小二乘：
   - `min_w ||x_i - sum_j w_ij x_j||^2`，其中 `j in N(i)`；
   - 约束 `sum_j w_ij = 1`。

2. 全局嵌入求解：
   - 令稀疏权重矩阵为 `W`；
   - 最小化 `Phi(Y) = sum_i ||y_i - sum_j w_ij y_j||^2`；
   - 等价于对 `M = (I-W)^T(I-W)` 求最小非平凡特征向量。

## R04

高层流程（本 MVP）：
1. 生成并标准化高维数据。  
2. 计算每个样本的 `k` 近邻索引。  
3. 对每个样本构建局部 Gram 矩阵并解线性系统得到权重。  
4. 汇总为全局稀疏矩阵 `W`。  
5. 构建 `M=(I-W)^T(I-W)` 并做特征分解。  
6. 跳过最小平凡特征向量，取后续 `m` 个向量作为低维坐标。  
7. 输出 trustworthiness、邻域重叠率与局部重构误差。

## R05

关键数据结构：
- `LLEConfig`：`n_neighbors / n_components / reg / random_state`。  
- `LLEResult`：
  - `embedding`：低维嵌入；
  - `neighbor_indices`：每点近邻索引；
  - `local_weights`：局部重构权重；
  - `eigenvalues`：最小特征值序列；
  - `reconstruction_error`：局部重构 MSE。

## R06

正确性抓手：
- 每个样本的权重满足归一化约束 `sum_j w_ij = 1`；
- 权重求解使用带正则项的局部协方差矩阵，避免近奇异；
- `M` 由 `(I-W)^T(I-W)` 构造，理论上半正定；
- 嵌入取最小非平凡特征向量，剔除常量解对应方向；
- 运行末尾用断言检查有限值、形状和性能下界。

## R07

复杂度（`n` 样本数，`d` 特征维，`k` 邻居数）：
- kNN 构图：近似 `O(n log n * d)`（依赖近邻后端实现）；
- 局部权重求解：每点解 `k x k` 线性系统，约 `O(n * k^3)`；
- 特征分解：稀疏情形常用 `eigsh` 求前 `m+1` 小特征对，成本与迭代次数和稀疏度相关；
- 空间：
  - 邻接索引 `O(nk)`；
  - 权重矩阵 `W` 稀疏存储约 `O(nk)`。

## R08

边界与异常：
- `n_components + 1 >= n_samples` 会触发显式 `ValueError`；
- 局部矩阵可能病态，代码通过 `reg * trace(C)` 加对角稳定项；
- 线性方程求解失败时自动退化到 `lstsq`；
- 全局特征分解若 ARPACK 未完全收敛，先尝试用已收敛部分，否则退化为稠密 `eigh`。

## R09

MVP 取舍：
- 手写 LLE 主干逻辑，避免把算法封装成一行黑盒调用；
- 同时提供 `sklearn` LLE 与 `PCA` 作为基线，便于验证手写实现质量；
- 不实现 out-of-sample extension、鲁棒 LLE、Hessian LLE 等扩展；
- 目标是“可复现 + 可审计 + 结构透明”的最小可运行实现。

## R10

`demo.py` 主要函数映射：
- `build_dataset`：生成 3 维 swiss-roll 数据并标准化。  
- `knn_graph`：计算每个样本的 `k` 近邻。  
- `solve_local_weights`：逐点求解局部重构权重。  
- `build_weight_matrix`：构建全局稀疏 `W`。  
- `solve_global_embedding`：从 `M=(I-W)^T(I-W)` 求低维特征向量。  
- `local_reconstruction_error`：计算重构误差。  
- `neighbor_overlap_score`：计算高低维邻域重叠率。  
- `run_manual_lle`：串联手写 LLE 全流程。  
- `evaluate_embeddings`：输出手写 LLE、sklearn LLE、PCA 的指标表。  
- `main`：统一调度、打印、断言。

## R11

运行方式：

```bash
cd Algorithms/数学-机器学习-0238-流形学习_-_LLE
uv run python demo.py
```

脚本无需命令行参数，不读取交互输入。

## R12

输出解读：
- `Data shape`：输入矩阵维度；
- `Manual LLE config`：关键超参数；
- `Manual local reconstruction MSE`：局部线性重构误差；
- `Manual smallest eigenvalues`：用于确认谱分解行为；
- `Metrics` 表：
  - `trustworthiness`：低维邻域保持程度；
  - `neighbor_overlap`：kNN 重叠率；
- `embedding preview`：前 10 个样本二维坐标。

## R13

内置最小验证：
1. 手写嵌入矩阵形状必须等于 `(n_samples, 2)`；
2. 嵌入与特征值必须为有限数；
3. 局部重构误差必须小于上限；
4. 手写 LLE 的 trustworthiness 必须明显优于 PCA 基线；
5. 手写 LLE 与 sklearn LLE 的 trustworthiness 差距不能过大。

## R14

关键参数建议：
- `n_neighbors`：最敏感参数。小值更保局部细节，大值更平滑；
- `reg`：局部线性系统稳定项。太小可能不稳定，太大可能过度平滑；
- `n_components`：目标维度；
- 数据噪声和标准化方式会显著影响局部权重估计质量。

## R15

方法对比：
- 对比 PCA：
  - PCA 为全局线性映射；
  - LLE 可保留非线性流形局部结构。  
- 对比 Isomap：
  - Isomap 强调测地距离保真；
  - LLE 强调局部重构权重保真。  
- 对比 UMAP / t-SNE：
  - UMAP/t-SNE 更偏可视化优化目标；
  - LLE 目标函数更“代数化”，可直接转为特征值问题。

## R16

典型应用：
- 高维特征可视化与结构探索；
- 降维预处理后再做聚类/分类；
- 近似低维流形数据的几何分析；
- 教学场景中讲解“局部线性 -> 全局谱分解”的思路。

## R17

可扩展方向：
- 加入鲁棒权重估计（对离群点更稳）；
- 引入 Landmark LLE 处理超大样本；
- 增加 Hessian LLE / Modified LLE 对照实验；
- 加入 out-of-sample extension（新样本映射）。

## R18

`demo.py` 的源码级算法流（9 步）：
1. `build_dataset` 生成 swiss-roll 数据并做 `StandardScaler` 标准化。  
2. `knn_graph` 用 `NearestNeighbors` 计算每个样本的 `k` 近邻索引。  
3. `solve_local_weights` 对每个样本构造局部矩阵 `C = (X_N - x_i)(X_N - x_i)^T`。  
4. 在 `C` 上加 `reg * trace(C) * I` 稳定项后解 `Cw=1`，再归一化得到 `sum w = 1`。  
5. `build_weight_matrix` 将所有局部权重写入全局稀疏矩阵 `W`。  
6. `solve_global_embedding` 构建 `M=(I-W)^T(I-W)`，调用 `eigsh` 取最小 `m+1` 个特征对。  
7. 丢弃最小平凡特征向量，保留后续 `m` 个特征向量作为低维嵌入坐标。  
8. `evaluate_embeddings` 计算手写 LLE、`sklearn` LLE、PCA 的 `trustworthiness` 与邻域重叠率。  
9. `main` 输出指标和坐标预览，并用断言校验数值稳定性与相对性能门槛。
