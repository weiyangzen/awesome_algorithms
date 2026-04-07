# SMO算法

- UID: `MATH-0303`
- 学科: `数学`
- 分类: `机器学习`
- 源序号: `303`
- 目标目录: `Algorithms/数学-机器学习-0303-SMO算法`

## R01

SMO（Sequential Minimal Optimization）是训练支持向量机（SVM）的一种经典分解算法。它把原本带等式约束与盒约束的二次规划问题，拆成一系列“只优化两个拉格朗日乘子 `alpha_i, alpha_j`”的小子问题，每个子问题都可解析更新，不需要调用通用 QP 求解器。

对二分类问题（标签 `y in {-1,+1}`），SMO 的目标是找到最大间隔超平面（线性核）或最大间隔决策边界（核方法）。

## R02

软间隔 SVM 的对偶问题：

- 最大化
  `W(alpha) = sum_i alpha_i - 1/2 * sum_{i,j} alpha_i alpha_j y_i y_j K(x_i, x_j)`
- 约束
  `0 <= alpha_i <= C`，`sum_i alpha_i y_i = 0`

其中：
- `C` 控制间隔与误分类惩罚的权衡；
- `K(.,.)` 是核函数（线性核、RBF 核等）。

SMO 的核心是：在一次迭代中固定其余变量，仅更新一对 `alpha`，并始终维持上述约束。

## R03

KKT 条件是 SMO 的迭代依据。记决策函数 `f(x)=sum_k alpha_k y_k K(x_k,x)+b`，则样本 `i` 的约束与条件可写为：

- 若 `alpha_i = 0`，应满足 `y_i f(x_i) >= 1`；
- 若 `0 < alpha_i < C`，应满足 `y_i f(x_i) = 1`；
- 若 `alpha_i = C`，应满足 `y_i f(x_i) <= 1`。

实现里通常通过误差 `E_i = f(x_i)-y_i` 检查是否违反 KKT，再决定是否触发 pair update。

## R04

为什么一次只更新两个变量：

- 因为存在等式约束 `sum_i alpha_i y_i = 0`，如果只改一个 `alpha_i` 会破坏该约束；
- 同时改 `alpha_i, alpha_j` 可以沿一条可行线段移动，约束可保持；
- 这使得每步子问题从高维 QP 降为一维裁剪问题，计算显著简化。

## R05

设选中 `(i,j)`，其它变量固定。根据约束可得：

`alpha_i_new = alpha_i_old + y_i y_j (alpha_j_old - alpha_j_new)`

再由目标函数沿该方向求极值得到 `alpha_j` 的无约束更新：

`eta = K_ii + K_jj - 2K_ij`

`alpha_j_new_unc = alpha_j_old + y_j (E_i - E_j) / eta`

其中 `K_ab = K(x_a, x_b)`。

## R06

`alpha_j_new` 需要投影回可行区间 `[L, H]`。区间由 `y_i` 与 `y_j` 是否同号决定：

- 若 `y_i != y_j`：
  `L = max(0, alpha_j_old - alpha_i_old)`
  `H = min(C, C + alpha_j_old - alpha_i_old)`
- 若 `y_i == y_j`：
  `L = max(0, alpha_i_old + alpha_j_old - C)`
  `H = min(C, alpha_i_old + alpha_j_old)`

最终 `alpha_j_new = clip(alpha_j_new_unc, L, H)`，再反算 `alpha_i_new`。

## R07

偏置 `b` 的更新（Platt 写法）：

- `b1 = b - E_i - y_i (alpha_i_new-alpha_i_old) K_ii - y_j (alpha_j_new-alpha_j_old) K_ij`
- `b2 = b - E_j - y_i (alpha_i_new-alpha_i_old) K_ij - y_j (alpha_j_new-alpha_j_old) K_jj`

选择规则：
- 若 `alpha_i_new` 在开区间 `(0,C)`，用 `b1`；
- 否则若 `alpha_j_new` 在 `(0,C)`，用 `b2`；
- 否则取平均 `(b1+b2)/2`。

## R08

核函数把线性间隔推广到非线性边界：

- 线性核：`K(x,z)=x^T z`；
- RBF 核：`K(x,z)=exp(-gamma ||x-z||^2)`。

本目录 MVP 在 `demo.py` 中同时跑线性核与 RBF 核，便于观察非线性数据上的效果差异。

## R09

时间与空间复杂度（简化估计）：

- 预计算核矩阵：时间 `O(n^2 d)`，空间 `O(n^2)`；
- 单轮遍历样本时，每次计算误差含 `O(n)` 内积，整体近似 `O(n^2)` 到 `O(n^3)`（取决于更新触发频率）；
- 总成本与 `max_iter`、`max_passes`、核类型、数据可分性有关。

教学 MVP 的重点是“步骤透明”，不是超大规模最优速度。

## R10

关键超参数：

- `C`：大 `C` 更强调训练集拟合，小 `C` 更强调间隔和正则；
- `tol`：KKT 违规容忍度，越小越严格；
- `max_passes`：连续“无更新轮次”阈值，达到则停止；
- `max_iter`：外层最大迭代保护；
- `gamma`（RBF）：越大边界越弯曲，过大可能过拟合。

## R11

停止条件（本实现）：

1. 连续 `max_passes` 轮没有任何 `alpha` 变化；或
2. 迭代轮数达到 `max_iter`。

同时用 `alpha_eps` 过滤数值噪声，避免把极小变化误判成有效更新。

## R12

数值稳定与工程细节：

- 对 `eta <= 1e-12` 的 pair 直接跳过，避免除零/病态更新；
- 对 `alpha` 更新后做 clip，保证盒约束不破坏；
- 使用标准化特征（`StandardScaler`）改善核尺度稳定性；
- 固定随机种子，便于复现实验结果。

## R13

本目录 MVP 任务定义：

- 数据：`make_moons(n_samples=260, noise=0.22)`；
- 标签：从 `{0,1}` 映射到 `{-1,+1}`；
- 模型：手写 `BinarySMO`，分别训练线性核与 RBF 核；
- 指标：`train_acc`、`test_acc`、`test_f1`、支持向量数、对偶目标值、训练耗时。

无交互输入，运行即输出结果表。

## R14

运行方式：

```bash
cd Algorithms/数学-机器学习-0303-SMO算法
uv run python demo.py
```

预期输出包含：
- 训练日志（迭代轮次、每轮更新数、support 数、dual objective）；
- 两种核函数的汇总表（`pandas.DataFrame` 打印）；
- 按测试准确率选出的最佳配置。

## R15

结果解读建议：

- 若 RBF `test_acc` 明显高于线性核，说明数据确实需要非线性边界；
- 若支持向量占比过高，可能表示边界复杂或 `C` 偏大；
- 若 `train_acc` 很高但 `test_acc` 下降，可尝试减小 `C` 或减小 `gamma`；
- 若长期无更新且精度低，可增大 `max_iter` 或调整 `tol`。

## R16

常见失败模式与排查：

- 失败：标签不是 `-1/+1`，导致更新公式符号错误。
  - 处理：在 `fit` 开头做强校验（本实现已校验）。
- 失败：`L == H` 频繁出现，优化停滞。
  - 处理：检查 `C` 是否过小、数据是否高度重复。
- 失败：`eta <= 0` 频繁出现。
  - 处理：通常是重复样本或核参数导致病态，可调 `gamma`。
- 失败：全部 `alpha` 近零。
  - 处理：增大 `max_iter`/`C`，并确认特征已标准化。

## R17

交付核对清单：

- `README.md` 的 `## R01 ... ## R18` 全部已填写；
- `demo.py` 已完成实现并可直接运行；
- 代码未调用 `sklearn.svm.SVC` 黑盒训练，而是显式实现 SMO pair update；
- `meta.json` 保持 UID/学科/子类/源序号与任务一致。

## R18

源码级算法流程（对应本目录 `demo.py`，9 步）：

1. `build_dataset` 生成 `make_moons` 二分类样本，做 `train_test_split` 与 `StandardScaler` 标准化。  
2. 在 `main` 中构造两组 `SMOConfig`（线性核、RBF 核），逐个调用 `run_experiment`。  
3. `BinarySMO.fit` 初始化 `alphas=0`、`b=0`，并预计算训练核矩阵 `K_train`。  
4. 外层迭代按样本扫描 `i`，通过 `E_i = f(x_i)-y_i` 与 `alpha_i` 判断是否违反 KKT。  
5. 若违反 KKT，随机选 `j!=i`，计算 `E_j`，据 `y_i/y_j` 关系求可行区间 `[L,H]`。  
6. 计算 `eta = K_ii + K_jj - 2K_ij`，得到 `alpha_j_new` 的无约束解后 clip 到 `[L,H]`。  
7. 用约束关系反算 `alpha_i_new`，再按 `b1/b2` 公式更新偏置 `b`。  
8. 循环直到连续 `max_passes` 轮无更新或达到 `max_iter`；训练中打印 `dual_objective` 与支持向量数。  
9. `predict` 基于支持向量集合计算决策函数，对测试集输出分类结果并汇总 `accuracy/F1` 等指标。  

以上 9 步均在本地源码可逐行追踪，不依赖第三方 SVM 训练黑盒。
