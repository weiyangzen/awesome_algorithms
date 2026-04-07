# CUR分解

- UID: `MATH-0107`
- 学科: `数学`
- 分类: `数值线性代数`
- 源序号: `107`
- 目标目录: `Algorithms/数学-数值线性代数-0107-CUR分解`

## R01

CUR 分解是一类“可解释低秩近似”方法：对矩阵 `A` 直接抽取其真实列与真实行，构造

- `C`: 由 `A` 的若干列组成
- `R`: 由 `A` 的若干行组成
- `U`: 尺寸较小的耦合矩阵

使得 `A ≈ C U R`。与纯 SVD 不同，CUR 的左右因子保留了原始特征维度与样本维度的语义，常用于需要解释“选中了哪些列/行”的场景。

## R02

本目录 MVP 解决的问题是：给定实矩阵 `A ∈ R^{m×n}`，在目标秩 `k` 附近构造一个 CUR 近似，并评估重构误差。  
演示矩阵采用“低秩信号 + 小噪声”合成数据，便于稳定复现与量化对比。

目标输出：

- 选择的列索引、行索引
- CUR 重构相对 Frobenius 误差
- 与随机采样 CUR、截断 SVD 基线的误差对比

## R03

符号约定：

- `A`: 原矩阵，形状 `m×n`
- `k`: 目标秩（用于计算杠杆分数）
- `c, r`: 抽样列数与行数（通常 `c,r >= k`）
- `C = A[:, J]`，`R = A[I, :]`
- `U = pinv(C) @ A @ pinv(R)`

最终近似：

- `A_cur = C @ U @ R`

其中 `pinv` 是 Moore-Penrose 伪逆。

## R04

杠杆分数（leverage score）来自截断 SVD：

- `A ≈ U_k Σ_k V_k^T`
- 列杠杆分数：`p_j ∝ ||V_k[:, j]||_2^2`
- 行杠杆分数：`q_i ∝ ||U_k[i, :]||_2^2`

直觉：分数越高的行/列，对 rank-`k` 子空间贡献越大，优先选它们通常比均匀随机更稳定。

## R05

本实现使用确定性“Top-Leverage”采样：

1. 先做一次 `np.linalg.svd(A, full_matrices=False)`。
2. 取前 `k` 个左右奇异向量，计算行/列杠杆分数。
3. 分别选分数最高的 `c` 列与 `r` 行（去重后排序）。
4. 以 `U = pinv(C) @ A @ pinv(R)` 形成耦合矩阵。
5. 得到 `A_cur = C U R`。

同时提供均匀随机采样基线用于误差对照。

## R06

`demo.py` 关键函数：

- `make_low_rank_matrix`: 生成低秩 + 噪声测试矩阵
- `top_leverage_indices`: 基于截断 SVD 计算并选择高杠杆行/列
- `cur_decomposition`: 构造 `C,U,R` 并返回重构
- `relative_fro_error`: 计算相对 Frobenius 误差
- `truncated_svd_error`: 计算同阶最佳 SVD 近似误差（参考下界）
- `main`: 一次性运行并打印指标

## R07

正确性直觉：

- 若选中的列空间较好覆盖 `A` 的主列空间、选中的行空间较好覆盖主行空间，则 `C` 与 `R` 已捕捉主要结构。
- `U = pinv(C) A pinv(R)` 在最小二乘意义下把两侧子空间耦合起来。
- 因此 `C U R` 能逼近 `A` 的主能量部分，通常误差接近但不会优于同秩截断 SVD（后者是 Frobenius 最优）。

## R08

复杂度（`m×n` 矩阵，目标秩 `k`，采样 `c,r`）：

- 一次完整 SVD：约 `O(min(mn^2, m^2n))`（MVP 的主成本）
- 构造伪逆与耦合矩阵：约 `O(mc^2 + nr^2 + mnc + mnr)`（依实现路径而变）
- 重构与误差评估：`O(mnr + mn)`

MVP 优先“流程透明”，未做随机化 SVD 或增量优化。

## R09

数值稳定性要点：

- `C` 或 `R` 可能病态，伪逆会放大噪声；可在工程版加入截断或 Tikhonov 正则。
- 当 `c,r` 过小（接近或小于有效秩）时，重构误差会明显上升。
- 杠杆分数法通常优于均匀随机，尤其在谱衰减慢或信息集中时更明显。
- 建议固定随机种子，保证实验可复现。

## R10

默认参数（见 `CURConfig`）：

- `m=140, n=100`
- `true_rank=8`
- `target_rank=8`
- `n_cols=20, n_rows=20`
- `noise_std=0.02`
- `seed=7`

脚本会打印两种 CUR（杠杆分数/随机）及截断 SVD 的相对误差。

## R11

MVP 取舍：

- 只依赖 `numpy`，避免额外安装成本。
- 使用“先 SVD 再采样”的直接实现，便于核对数学对象。
- 不追求大规模最优性能，先保证算法链条可读、可跑、可验。
- 用随机基线和 SVD 基线进行最小但诚实的效果定位。

## R12

代码与数学映射关系：

- `A` 的构造：`make_low_rank_matrix`
- `U_k, V_k` 与杠杆分数：`top_leverage_indices`
- `C, U, R` 组装：`cur_decomposition`
- 误差度量：`relative_fro_error`, `truncated_svd_error`
- 结果汇总与输出：`main`

这样可直接从源码定位到 CUR 各数学部件。

## R13

运行方式：

```bash
cd Algorithms/数学-数值线性代数-0107-CUR分解
python3 demo.py
```

脚本无交互输入，运行后直接输出指标表和摘要。

## R14

预期现象：

- `CUR(top-leverage)` 误差通常小于 `CUR(uniform-random)`（不是严格保证，但在默认数据上通常成立）。
- 截断 SVD 误差通常最低（同阶最优基线）。
- 采样数量 `c,r` 增大时，CUR 误差一般下降但存储成本上升。

## R15

常见实现坑：

- 把列杠杆与行杠杆搞反，导致采样轴错误。
- 忘记对索引排序，影响输出可读性与调试稳定性。
- 直接用 `inv` 代替 `pinv`，在非方阵或病态矩阵上失败。
- 误把 `A ≈ CUR` 当成严格等式，没有报告误差指标。
- 未固定随机种子，导致基线结果不可复现。

## R16

可扩展方向：

- 用概率采样（按杠杆分数分布）替代 deterministic top-k。
- 引入随机化 SVD（RSVD）降低大矩阵成本。
- 在 `U` 求解中加入正则项以增强抗噪性。
- 扩展到稀疏矩阵/流式矩阵场景。
- 将 CUR 应用于特征选择、推荐系统解释性组件等任务。

## R17

与相关分解方法对比：

- SVD：误差最优但因子不具原始行列语义。
- QR with column pivoting：擅长列选择，但不同时强调行选择。
- NMF：强调非负可解释，但适用前提与优化目标不同。
- CUR：在“保持原始行/列可解释性”方面有直接优势。

## R18

`demo.py` 的源码级 CUR 算法流（非黑盒）：

1. `main` 创建 `CURConfig`，调用 `make_low_rank_matrix` 生成测试矩阵 `A`。
2. 在 `cur_decomposition(..., method="leverage")` 中对 `A` 做 SVD，得到 `U, s, Vt`。
3. `top_leverage_indices` 用 `U[:, :k]` 和 `Vt[:k, :]` 计算行/列杠杆分数并选出 `I, J`。
4. 按索引切片形成 `C = A[:, J]` 与 `R = A[I, :]`。
5. 通过 `np.linalg.pinv` 计算 `U_core = pinv(C) @ A @ pinv(R)`。
6. 重构 `A_cur = C @ U_core @ R`，再用 `relative_fro_error` 计算相对误差。
7. 重复一次 `cur_decomposition(..., method="uniform")` 作为随机采样基线。
8. `truncated_svd_error` 计算 rank-`k` 截断 SVD 误差，作为理论上更强的参照。
9. `main` 输出三者误差、存储规模与被选索引，形成完整可复现实验闭环。
