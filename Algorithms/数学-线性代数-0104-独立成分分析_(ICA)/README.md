# 独立成分分析 (ICA)

- UID: `MATH-0104`
- 学科: `数学`
- 分类: `线性代数`
- 源序号: `104`
- 目标目录: `Algorithms/数学-线性代数-0104-独立成分分析_(ICA)`

## R01

独立成分分析（Independent Component Analysis, ICA）用于从观测到的线性混合信号中恢复潜在的统计独立源信号。典型模型是盲源分离：
\[
X = AS
\]
其中 `S` 是未知独立源，`A` 是未知混合矩阵，`X` 是观测信号。ICA 在不预先知道 `A` 与 `S` 的情况下，估计解混矩阵 `W` 使得 `Y = WX` 近似恢复 `S`（允许排列与符号不确定性）。

## R02

本条目要解决的问题：
- 输入：线性混合后的多通道观测矩阵 `X`（本 MVP 中由固定随机种子自动生成）。
- 输出：估计的源信号 `S_est`、全局解混矩阵 `W_total`、恢复质量指标（相关系数、白化误差、恢复 MSE）。

`demo.py` 不需要任何交互输入，运行后自动完成数据生成、ICA 求解与结果评估。

## R03

核心数学关系：
- 观测模型：`X = AS`。
- 去中心化：`X_c = X - E[X]`。
- 白化变换：`Z = V X_c`，使 `Cov(Z) = I`。
- FastICA 固定点更新（`g(u)=tanh(u)`）：
\[
w_{new} = E[Z g(w^T Z)] - E[g'(w^T Z)]w
\]
- 对每个分量做单位化并与已求分量正交化（deflation），得到一组行向量 `W`。
- 最终解混矩阵：`W_total = W V`，估计源为 `S_est = W Z`。

ICA 的可辨识性依赖“源信号非高斯且相互独立”（最多允许一个高斯源）。

## R04

算法高层流程：
1. 构造三个相互独立且非高斯程度不同的源信号。
2. 随机采样可逆且条件数适中的混合矩阵 `A`，得到观测信号 `X`。
3. 对 `X` 去均值并通过协方差特征分解完成白化。
4. 在白化空间中逐个分量执行 FastICA 固定点迭代。
5. 每轮迭代后做 deflation 正交化，避免重复提取同一成分。
6. 迭代收敛后拼出 `W_total` 与 `S_est`。
7. 通过绝对相关矩阵 + 排列搜索对齐真值与估计结果。
8. 打印白化误差、匹配相关性、恢复 MSE 与成功判定。

## R05

核心数据结构：
- `s_true: np.ndarray (n_components, n_samples)`：真实源信号。
- `a_true: np.ndarray (n_components, n_components)`：真实混合矩阵。
- `x_obs: np.ndarray (n_components, n_samples)`：观测信号。
- `x_white: np.ndarray`：白化后的观测。
- `w_rows: np.ndarray`：FastICA 在白化空间估计到的解混行向量。
- `unmixing_total: np.ndarray`：映射 `x_obs -> s_est` 的全局解混矩阵。
- `abs_corr: np.ndarray`：`|corr(true_i, est_j)|` 相关矩阵。

## R06

正确性要点：
- 白化后 `Cov(Z)=I`，将问题从“任意线性混合”简化为“正交旋转识别”。
- 固定点更新以最大化非高斯性为目标，独立成分通常对应非高斯方向。
- Deflation 正交化保证不同 `w` 对应不同独立方向。
- ICA 天然存在排列与符号不确定性，因此评估时需先做“排列+符号对齐”。
- 若最小匹配相关系数高、恢复 MSE 低，则说明恢复成功。

## R07

复杂度分析（设 `m=n_components`, `T=max_iter`, `N=n_samples`）：
- 白化（协方差 + 特征分解）：`O(m^2N + m^3)`。
- FastICA 固定点迭代：每次迭代约 `O(mN)`，总计约 `O(m^2TN)`（deflation 模式逐分量求解）。
- 排列搜索：`O(m!)`，本 MVP 仅 `m=3`，开销可忽略。
- 空间复杂度：`O(mN + m^2)`。

## R08

边界与异常处理：
- 输入维度错误（非 2D）直接抛 `ValueError`。
- 行方差过小、协方差近奇异、`n_components` 非法时抛 `ValueError`。
- 固定点更新出现退化向量或迭代不收敛时抛 `RuntimeError`。
- 排列搜索限制 `n<=8`，避免暴力枚举失控。

## R09

MVP 取舍：
- 采用 `numpy` 手写 FastICA 关键流程，避免把算法完全交给第三方黑箱。
- 不引入复杂工程框架，仅保留演示 ICA 数学核心所需模块。
- 使用合成数据（可控真值）来直接量化恢复质量。
- 仅实现 deflation 版本，便于最小化代码规模并保持可读性。

## R10

`demo.py` 函数职责：
- `standardize_rows`：行级标准化。
- `generate_sources`：生成独立源信号。
- `make_mixture`：生成随机混合矩阵并构造观测。
- `whiten`：去中心化与白化。
- `fastica_deflation`：固定点迭代提取独立成分。
- `absolute_correlation_rows`：计算真值与估计的绝对相关矩阵。
- `best_permutation`：搜索最优排列。
- `align_estimated_sources`：符号和顺序对齐。
- `main`：串联流程并输出指标。

## R11

运行方式：

```bash
cd Algorithms/数学-线性代数-0104-独立成分分析_(ICA)
python3 demo.py
```

脚本默认固定随机种子，可重复复现实验结果。

## R12

输出字段说明：
- `True Mixing Matrix A`：用于生成观测信号的真实混合矩阵。
- `Estimated Global Unmixing Matrix W_total`：ICA 估计的解混矩阵。
- `Fixed-Point Iterations per Component`：每个独立成分收敛迭代步数。
- `|corr(true_i, est_j)| Matrix`：真值与估计的绝对相关矩阵。
- `Whitening error`：白化质量，越接近 `0` 越好。
- `Mean/Min matched |corr|`：恢复质量主指标，越接近 `1` 越好。
- `Aligned source recovery MSE`：对齐后的源信号误差，越小越好。

## R13

建议最小测试集：
- 默认 3 源信号配置（当前脚本）。
- 调整随机种子，验证不同混合矩阵下稳定性。
- 增加样本数 `n_samples`，观察收敛速度与精度变化。
- 人为制造病态混合矩阵，验证异常路径是否触发。

## R14

可调参数：
- `n_samples`：样本数，过小会导致统计量不稳定。
- `max_iter`：单分量最大迭代次数。
- `tol`：收敛阈值（`| |w_new^T w| - 1 |`）。
- `alpha`：`tanh(alpha*u)` 非线性强度。
- 混合矩阵筛选阈值（行列式与条件数）影响演示难度。

调参建议：先保证白化误差小，再观察最小匹配相关系数是否稳定高于阈值。

## R15

与相关方法对比：
- PCA：只保证方差方向正交，不能保证统计独立；更适合降维而非盲源分离。
- ICA：利用高阶统计信息，能在独立性假设下恢复潜在源。
- NMF：要求非负约束，解释性强，但不等价于独立成分分解。

## R16

典型应用：
- 语音分离（鸡尾酒会问题）。
- EEG/MEG 神经信号伪迹分离。
- 金融时间序列中的潜在因子分解。
- 图像与通信中的盲信号分离预处理。

## R17

可扩展方向：
- 从 deflation 扩展到 symmetric FastICA（并行更新所有分量）。
- 支持不同对比函数（如 `pow3`, `gauss`）并比较稳健性。
- 加入噪声模型与正则化，提升真实数据适应性。
- 使用 SciPy/Scikit-learn 做结果交叉验证，但保留手写核心流程作为主实现。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 固定 `n_components=3, n_samples=4000`，调用 `generate_sources` 得到 `s_true`。  
2. `make_mixture` 随机采样可逆矩阵 `a_true`，构造观测 `x_obs = a_true @ s_true`。  
3. `whiten` 对 `x_obs` 去均值并做特征分解，生成白化信号 `x_white` 与矩阵 `whitening`。  
4. `fastica_deflation` 对每个分量执行固定点迭代：`w_new = E[z g(w^Tz)] - E[g'(w^Tz)]w`。  
5. 在 `fastica_deflation` 内对 `w_new` 执行 deflation 正交化与单位化，满足收敛准则后写入 `w_rows`。  
6. `main` 计算 `s_est = w_rows @ x_white` 与全局解混矩阵 `unmixing_total = w_rows @ whitening`。  
7. `absolute_correlation_rows` 生成相关矩阵，`best_permutation` 找到最优排列，`align_estimated_sources` 完成符号与顺序对齐。  
8. `main` 输出白化误差、匹配相关系数、恢复 MSE，并给出布尔成功判定。  
