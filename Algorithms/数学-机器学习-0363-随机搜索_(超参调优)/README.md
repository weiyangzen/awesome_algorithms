# 随机搜索 (超参调优)

- UID: `MATH-0363`
- 学科: `数学`
- 分类: `机器学习`
- 源序号: `363`
- 目标目录: `Algorithms/数学-机器学习-0363-随机搜索_(超参调优)`

## R01

随机搜索（Random Search for Hyper-parameter Tuning）用于在给定预算下，从超参数空间中随机抽样候选配置，并通过交叉验证挑选表现最优的参数。  
它的核心优势是：
- 对连续空间（如 `C`、`gamma`）比密集网格更高效；
- 实现简单、易并行；
- 适合作为调参基线或第一阶段粗搜索。

本条目实现一个可复现、可直接运行的 MVP：
- 主路径：`sklearn` + `SVC` + 手写随机采样 + 分层 K 折评估；
- 兜底路径：无 `sklearn` 时使用 `numpy` KNN 随机搜索，保证脚本仍可运行。

## R02

问题定义（分类任务超参调优）：
- 输入：训练数据 `D={(x_i, y_i)}`、超参数搜索分布 `P(theta)`、搜索预算 `T`、折数 `K`。
- 过程：采样 `theta_1,...,theta_T ~ P(theta)`，对每个候选进行 K 折交叉验证，得到均值分数。
- 输出：
  - `best_params`：最优超参数；
  - `best_cv_f1`：最优候选的平均 CV F1；
  - `top_candidates`：候选排名表；
  - 测试集 `F1/Accuracy` 与 baseline 对比。

## R03

数学形式化：

1. 搜索空间 `Theta` 上定义采样分布 `P(theta)`。  
2. 对第 `t` 次采样得到的参数 `theta_t`，其 K 折评分记为 `s_{t,1},...,s_{t,K}`。  
3. 候选的经验风险（以 F1 最大化表示）为：
   `S_t = (1/K) * sum_{k=1..K} s_{t,k}`。  
4. 目标是：
   `t* = argmax_{1<=t<=T} S_t`，并取 `theta_{t*}`。  

随机搜索的关键不在“穷举”，而在“预算固定时用随机抽样提高有效覆盖概率”。

## R04

MVP 算法流程：
1. 生成固定随机种子的二分类合成数据并做分层训练/测试切分。  
2. 在参数分布中随机采样一个候选（`SVC` 的 `kernel/C/gamma/class_weight`）。  
3. 对该候选执行分层 K 折交叉验证，得到每折 F1。  
4. 汇总 `mean/std/min/max`，写入候选记录表。  
5. 重复采样与评估，直到达到 `n_iter` 预算。  
6. 按 `mean_f1`（同分看 `std_f1`）排序，选出最优候选。  
7. 用最优参数在完整训练集重训，并在测试集评估。  
8. 输出 Top-5、置信区间、以及相对 baseline 的增益。

## R05

核心数据结构：
- `SearchOutcome(dataclass)`：统一封装搜索结果；
- `records: list[dict]`：逐候选记录 `params`、`mean_f1`、`std_f1`、`fold_scores`；
- `top_candidates: pandas.DataFrame`：排序后的前 5 名候选；
- `seen: set[str]`：参数签名集合，避免重复候选；
- `x_train/x_test/y_train/y_test`：固定随机种子下的数据拆分结果。

## R06

正确性要点：
- 候选比较标准统一：所有候选都在同一数据、同一折数、同一评分函数下评估；
- 选择规则明确：按 `mean_f1` 最大选优，`std_f1` 作为稳定性次级排序；
- 防数据泄漏：标准化仅在训练折统计并应用到验证折；
- 可复现：随机种子固定后，采样顺序和结果稳定；
- 最终评估独立：测试集不参与搜索，仅用于最终泛化检查。

## R07

复杂度分析（主路径）：
- 记候选数 `T`、折数 `K`、单次模型训练成本 `C_fit`。  
- 时间复杂度约：`O(T * K * C_fit)`。  
- 空间复杂度主要来自：
  - 数据存储 `O(n*d)`；
  - 结果表 `O(T)`；
  - 单候选折分与模型中间状态（由学习器决定）。

结论：随机搜索通过控制 `T` 直接控制总算力开销。

## R08

边界与异常处理：
- `stratified_train_test_split_np` 检查维度与样本数匹配；
- `test_size` 需在 `(0,1)` 内；
- K 折时若某类样本数 `< n_splits`，会抛出 `ValueError`；
- 若随机采样失败导致无候选（理论上极低概率），抛出 `RuntimeError`；
- fallback KNN 中 `k/p/weighting` 都受控于合法采样范围。

## R09

MVP 取舍：
- 使用较小但真实的流程：随机采样 + CV + 独立测试；
- 不引入并行调度、持久化实验管理、早停策略，保持代码短小；
- 主路径用 `SVC` 展示连续超参采样，fallback 用 `numpy` KNN 保证可运行性；
- 重点在“流程透明和可复现”，而不是追求最高精度。

## R10

`demo.py` 模块职责：
- 数据与切分：`generate_dataset`、`stratified_train_test_split_np`；
- 评估基础：`binary_f1`、`binary_accuracy`、`approx_ci95`；
- 主路径随机搜索：
  - `sample_svc_params`
  - `evaluate_candidate_sklearn`
  - `run_random_search_sklearn`
- fallback 随机搜索：
  - `sample_knn_params`
  - `knn_predict`
  - `evaluate_candidate_knn_cv`
  - `run_random_search_numpy`
- `main`：统一组织实验并打印最终结果。

## R11

运行方式：

```bash
cd Algorithms/数学-机器学习-0363-随机搜索_(超参调优)
python3 demo.py
```

无需命令行参数，也不需要交互输入。

## R12

输出字段说明：
- `backend`：当前使用的实现路径（`sklearn-SVC` 或 `numpy-KNN-fallback`）；
- `n_candidates`：实际评估的随机候选数量；
- `best_params`：最优参数组合；
- `best_cv_f1`：最优候选在 K 折上的平均 F1；
- `best_cv_f1_95%_CI`：最优候选折分分数均值的近似置信区间；
- `tuned_test_f1 / tuned_test_accuracy`：最优模型在测试集指标；
- `baseline_test_f1`：默认参数基线；
- `f1_gain_vs_baseline`：调参收益；
- `Top-5 candidates`：候选排名明细。

## R13

最小测试集（可直接复用当前脚本）：
1. 正常流程：固定随机种子运行，检查是否输出 Top-5 和最优参数。  
2. 可复现性：连续运行两次，结果应一致。  
3. 异常测试（建议补充）：
   - `test_size=0`；
   - `n_splits` 大于某类样本数；
   - 人工构造非法输入维度。  
4. 退化环境测试：卸载 `sklearn` 后验证 fallback 路径仍可运行。

## R14

关键可调参数：
- `n_iter`：随机搜索预算；
- `n_splits`：交叉验证折数；
- `RANDOM_STATE`：复现性控制；
- `sample_svc_params` 的采样分布：
  - `C ~ 10^U(-2,2)`；
  - `gamma ~ 10^U(-4,0.5)`（仅 `rbf`）；
  - `kernel` 与 `class_weight` 离散随机选择。

经验建议：
- 先小预算试跑（如 10-20），确认流程正确后再加预算；
- 如果分数方差大，优先增大数据量或折数，再讨论扩大搜索范围。

## R15

方法对比：
- 对比网格搜索：
  - 网格：覆盖完整但易组合爆炸；
  - 随机：预算固定时通常更高效，尤其连续参数多时。  
- 对比贝叶斯优化：
  - 贝叶斯在样本效率上通常更强；
  - 随机搜索实现简单、并行友好、调试成本低。  
- 对比手工调参：
  - 随机搜索更系统、更可复现，主观偏差更小。

## R16

典型应用场景：
- 基线模型阶段的快速超参粗搜索；
- 实验平台/教学场景中展示“预算-性能”关系；
- 需要快速建立可审计调参流程的中小规模任务；
- 作为后续精细搜索（网格、贝叶斯、遗传算法）的前置阶段。

## R17

可扩展方向：
- 引入并行候选评估（多进程或集群）；
- 支持多指标联合选择（如 `F1 + AUC`）；
- 增加结果持久化（CSV/JSON）和可视化；
- 引入条件参数空间（例如仅在某 kernel 下启用某参数）；
- 用 `RandomizedSearchCV` / 贝叶斯优化替换当前手写采样器并保持同一评估接口。

## R18

`demo.py` 的源码级算法流（8 步，非黑盒）：
1. `main` 调用 `generate_dataset` 生成固定随机种子数据，再用 `stratified_train_test_split_np` 得到训练集与测试集。  
2. 根据环境选择 `run_random_search_sklearn` 或 `run_random_search_numpy`，两条路径都遵循“随机采样 + K 折评估 + 排序选优”闭环。  
3. 在主路径中，`sample_svc_params` 从混合空间随机采样一组参数（含连续对数尺度参数 `C/gamma`）。  
4. `evaluate_candidate_sklearn` 对每个候选做 `StratifiedKFold`，每折都重新构建 `Pipeline(StandardScaler + SVC)` 并计算 F1。  
5. `run_random_search_sklearn` 汇总每个候选的 `mean_f1/std_f1/min_f1/max_f1` 与 `fold_scores`，写入 `records`，最终排序得到 `best_params`。  
6. 选出最优后再次在完整训练集重训 tuned 模型，并在测试集计算 `tuned_test_f1/tuned_test_accuracy`，同时计算 baseline F1。  
7. `approx_ci95` 用最优候选的折分分数计算均值 95% 近似置信区间，配合 Top-5 表增强结果可解释性。  
8. fallback 路径中，`sample_knn_params + evaluate_candidate_knn_cv + knn_predict` 复现同样的随机搜索逻辑，保证即使无 `sklearn` 也能完整演示算法流程。
