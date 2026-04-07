# Nyström方法

- UID: `MATH-0108`
- 学科: `数学`
- 分类: `数值线性代数`
- 源序号: `108`
- 目标目录: `Algorithms/数学-数值线性代数-0108-Nyström方法`

## R01

本条目实现 Nyström 方法的最小可运行版本（MVP），目标是把“大规模核矩阵低秩近似”这条主线完整打通：
- 在训练样本上构造 RBF 核矩阵的 Nyström 低秩近似；
- 用 Nyström 特征进行岭回归；
- 与完整核岭回归做误差对照，展示精度-成本权衡。

## R02

问题定义：
- 给定样本 `X = {x_i}_{i=1}^n`，核函数 `k(x, y)`，核矩阵 `K in R^{n x n}`；
- 当 `n` 较大时，直接存储/分解 `K` 代价高；
- 目标是用 `m << n` 个地标点（landmarks）构造近似 `K_hat`，并用于监督学习。

本实现选用：
- 核函数：RBF 核 `k(x,y)=exp(-gamma||x-y||^2)`；
- 任务：1D 非线性回归；
- 对照基线：完整 kernel ridge regression（KRR）。

## R03

Nyström 核心公式：
- 采样地标索引集合 `S`，`|S|=m`；
- 定义 `C = K[:, S] in R^{n x m}`，`W = K[S, S] in R^{m x m}`；
- 经典近似：`K_hat = C W^dagger C^T`。

为稳定实现，采用特征化写法：
1. 对 `W` 做特征分解 `W = U Lambda U^T`；
2. 取前 `r` 个正特征值 `Lambda_r` 与对应特征向量 `U_r`；
3. 定义 Nyström 特征 `Z = C U_r Lambda_r^{-1/2}`；
4. 则 `K_hat = Z Z^T`。

## R04

算法流程：
1. 生成训练/测试数据；
2. 计算完整 KRR 作为参考结果；
3. 随机采样 `m` 个地标点，构造 `C` 和 `W`；
4. 对 `W` 做特征分解并截断到 `rank`；
5. 构造 `Z_train` 与 `Z_test`；
6. 在 Nyström 特征空间做岭回归；
7. 计算 `||K-K_hat||_F / ||K||_F`；
8. 输出不同 `m/rank` 下的 RMSE 与误差差距。

## R05

核心数据结构：
- `NystromModel`（`dataclass`）：
  - `landmark_indices`：地标在训练集中的索引；
  - `landmarks`：地标样本矩阵；
  - `gamma`：RBF 参数；
  - `eigvecs`：`W` 截断特征向量 `U_r`；
  - `inv_sqrt_eigvals`：`Lambda_r^{-1/2}`。
- 矩阵对象：
  - `C`：`n x m`；
  - `W`：`m x m`；
  - `Z_train`：`n x r`。

## R06

正确性要点：
- 当 `W` 可逆且不截断时，`C W^{-1} C^T` 是 Nyström 标准形式；
- 当前实现用 `W` 的特征分解构造 `Z`，等价于显式使用 `W^dagger` 的稳定版本；
- 只保留正特征值，避免数值不稳定的除零/放大；
- 通过“完整 KRR vs Nyström-KRR”与 `Frobenius` 相对误差双重对照，验证近似有效性。

## R07

复杂度分析（`n` 样本数，`m` 地标数，`r` 截断秩）：
- 构造 `C`：`O(nm d)`（`d` 为特征维度）；
- 分解 `W (m x m)`：`O(m^3)`；
- 形成 `Z`：`O(nmr)`；
- 在特征空间做岭回归：解 `r x r` 线性系统，`O(r^3)`。

对比完整 KRR：
- 需要解 `n x n` 系统，约 `O(n^3)`；
- 当 `m,r << n` 时，Nyström 显著降低代价。

## R08

边界与异常处理：
- `m` 必须满足 `1 <= m <= n_train`；
- `rank >= 1`；
- `lam > 0`（岭回归正则）；
- 若 `W` 无正特征值，直接报错终止，避免构造无意义特征；
- 对平方距离做 `max(., 0)` 截断，降低浮点误差导致的负零小偏差。

## R09

MVP 取舍：
- 只实现“随机均匀采样地标”，不引入 k-means 或 leverage score 采样；
- 只用 `numpy`，不调用高阶黑盒核近似器；
- 聚焦最关键闭环：近似核矩阵 + 下游回归效果，不扩展到分类/在线场景。

## R10

`demo.py` 函数职责：
- `rbf_kernel`：构造核矩阵；
- `build_nystrom`：采样地标并生成 Nyström 特征；
- `transform_nystrom`：把新样本映射到 Nyström 特征空间；
- `fit_ridge_primal`：特征空间岭回归求解；
- `fit_kernel_ridge_dual` / `predict_kernel_ridge`：完整 KRR 基线；
- `run_single_case`：执行单组 `(m, rank)` 实验；
- `main`：批量运行 3 组预算并打印结果表。

## R11

运行方式：

```bash
cd Algorithms/数学-数值线性代数-0108-Nyström方法
python3 demo.py
```

脚本无交互输入，会自动完成数据生成、训练与评估。

## R12

输出解读：
- `m`：地标数量；
- `rank_eff`：实际保留秩（受 `W` 正特征值个数限制）；
- `relF(K,ZZ^T)`：核矩阵相对 Frobenius 误差，越小越好；
- `RMSE full KRR`：完整核岭回归在测试集 RMSE；
- `RMSE Nyström`：Nyström 近似模型 RMSE；
- `gap`：`RMSE Nyström - RMSE full KRR`，越接近 0 越好。

## R13

最小测试建议：
- 正常路径：默认三组参数应均可运行且输出有限数值；
- 参数非法：
  - `m=0`、`m>n_train`；
  - `rank=0`；
  - `lam<=0`；
- 退化观察：固定 `gamma/lam`，增大 `m/rank` 时，通常 `relF` 与 RMSE gap 下降。

## R14

关键参数说明：
- `gamma`：RBF 核带宽参数（越大越“局部”）；
- `lam`：岭回归正则强度；
- `m`：地标预算（内存/时间关键参数）；
- `rank`：截断秩（控制表达能力与稳定性）；
- `seed`：随机采样可复现实验。

调参经验：
- 先固定 `gamma, lam`，再增加 `m` 看近似误差是否足够；
- 当 `m` 固定时，`rank` 通常不宜超过 `m`，并应避开很小特征值。

## R15

与常见低秩策略对比：
- 随机傅里叶特征（RFF）：显式随机特征、与数据点无关；
- Nyström：直接利用训练样本子集，通常在同等特征数下近似核结构更直接；
- 完整 KRR：精度上限高但 `O(n^3)` 成本高。

本条目选择 Nyström 是因为它最贴近“核矩阵低秩近似”的数值线性代数视角。

## R16

典型应用场景：
- 大样本核方法加速（核岭回归、核 SVM 预处理）；
- 高斯过程近似中的核矩阵压缩；
- 谱聚类等依赖核/相似度矩阵的降成本计算。

## R17

可扩展方向：
- 更优地标策略：k-means centers、pivoted Cholesky、leverage score；
- 自适应秩选择：按累计谱能量自动确定 `rank`；
- 多核函数支持：多项式核、Matérn 核；
- 大规模工程化：分块计算 `C`，避免一次性内存峰值；
- 与 `scikit-learn` pipeline 接入做统一 benchmark。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `main` 调用 `make_regression_data` 生成训练/测试集，并设定 `gamma/lam` 与多组 `(m, rank)`。  
2. `run_single_case` 先调用 `fit_kernel_ridge_dual`，求解完整核系统 `(K + lam I) alpha = y`，作为精度参考。  
3. `build_nystrom` 随机选取 `m` 个地标，构造 `C = K[:,S]` 和 `W = K[S,S]`。  
4. 对 `W` 做 `eigh`，按降序取前 `rank_eff` 个正特征值/向量，形成 `U_r` 与 `Lambda_r`。  
5. 计算 `Z_train = C U_r Lambda_r^{-1/2}`，得到 Nyström 低维特征表示。  
6. `transform_nystrom` 对测试集构造 `Z_test`，并由 `fit_ridge_primal` 解 `(Z^T Z + lam I) w = Z^T y`。  
7. 用 `y_pred = Z_test w` 得到 Nyström 回归预测，同时构造 `K_hat = Z_train Z_train^T` 计算核近似误差。  
8. `main` 汇总打印每组参数的 `relF(K,ZZ^T)`、完整 KRR RMSE、Nyström RMSE 及 gap。  
