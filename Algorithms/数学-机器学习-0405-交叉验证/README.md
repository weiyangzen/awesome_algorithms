# 交叉验证

- UID: `MATH-0405`
- 学科: `数学`
- 分类: `机器学习`
- 源序号: `405`
- 目标目录: `Algorithms/数学-机器学习-0405-交叉验证`

## R01

交叉验证（Cross-Validation, CV）是机器学习评估泛化能力的标准方法：
把数据分成多个互斥子集，轮流用其中一份做验证集、其余做训练集，再对多次结果做统计汇总。

本条目实现一个可运行、可审计的最小 MVP：
- 手写“分层 K 折”索引生成；
- 手写 `cross_validate_manual` 主循环；
- 用轻量 `NearestCentroidClassifier` 作为示例学习器；
- 输出每折准确率、均值、标准差、OOF 准确率；
- 额外比较两个候选参数（`metric_p=1/2`）展示 CV 选型作用。

## R02

问题定义（本目录实现）：
- 输入：
  - 特征矩阵 `X in R^(n*d)`；
  - 标签向量 `y in {0,1,...,C-1}^n`；
  - 分类器及其候选参数；
  - 折数 `K`、随机种子 `random_state`。
- 目标：
  - 估计模型泛化性能，并在候选参数中选出最佳配置。
- 输出：
  - 每折记录 `[(fold, train_size, valid_size, accuracy), ...]`；
  - 每个候选的 `mean_accuracy/std_accuracy/oof_accuracy`；
  - 最优参数及其折级明细。

`demo.py` 自动生成合成多分类数据，不需要交互输入。

## R03

数学描述：

1. 设样本索引集 `I={1,...,n}`，划分 `K` 个折 `V1,...,VK`，满足：
   - `Vi ∩ Vj = empty`（`i != j`）；
   - `union_k Vk = I`。
2. 第 `k` 折训练/验证集合：
   - `Train_k = I \ Vk`；
   - `Valid_k = Vk`。
3. 在每折训练模型得到得分 `s_k`（本实现为 accuracy）。
4. 折均值与标准差：
   - `mean = (1/K) * sum_k s_k`；
   - `std = sqrt((1/K) * sum_k (s_k - mean)^2)`。
5. OOF 准确率：
   - 每个样本仅在它所属验证折被预测一次，
   - `oof_acc = (1/n) * sum_i 1(y_i = y_i_hat_oof)`。

## R04

算法流程（高层）：
1. 校验 `X/y` 维度与有限值。  
2. 按类别收集索引并打乱。  
3. 对每个类别索引做 `array_split`，填入每个折的桶。  
4. 逐折构造 `(train_idx, valid_idx)`。  
5. 克隆估计器，在训练折拟合并在验证折预测。  
6. 记录折准确率，并写入 OOF 预测数组。  
7. 汇总折均值、标准差和 OOF 准确率。  
8. 遍历候选参数，选择平均准确率最高者。

## R05

核心数据结构：
- `FoldResult`（`dataclass`）
  - `fold`：折编号；
  - `train_size`：训练样本数；
  - `valid_size`：验证样本数；
  - `accuracy`：该折准确率。
- `fold_buckets: list[list[np.ndarray]]`
  - 分层拆分过程中的“每折-每类”索引桶。
- `oof_pred: np.ndarray`
  - 与 `y` 同长，记录每个样本的 OOF 预测。
- `summary_df: pandas.DataFrame`
  - 候选参数级别的汇总表（排名、均值、方差、OOF）。

## R06

正确性要点：
- 覆盖性：所有样本都应且只应在某一折验证集出现一次。  
- 互斥性：同一折训练集与验证集必须不相交。  
- 分层性：各类样本按比例分配到每折，降低类别不均衡偏差。  
- 可复现性：固定随机种子后，折划分与结果可重复。  
- 选择规则一致：候选间使用同一划分规则和同一评估指标。

## R07

复杂度分析：
- 记样本数 `n`、折数 `K`、单次拟合+预测代价 `C_fit`。  
- 分层拆分索引成本约 `O(n)`。  
- 单候选参数的 CV 成本约 `O(K * C_fit)`。  
- 若候选数为 `M`，总成本约 `O(M * K * C_fit)`。  
- 空间复杂度约 `O(n)`（索引、OOF 缓冲、折记录）。

## R08

边界与异常处理：
- `X` 非二维或 `y` 非一维：抛 `ValueError`。  
- `X/y` 样本数不一致：抛 `ValueError`。  
- `X` 含 `nan/inf`：抛 `ValueError`。  
- `n_splits < 2`：抛 `ValueError`。  
- 某类样本数小于折数：抛 `ValueError`。  
- `candidate_params` 为空：抛 `ValueError`。  
- `metric_p <= 0`：抛 `ValueError`。

## R09

MVP 取舍说明：
- 不依赖外部 CV 黑盒，主流程完全手写，便于学习和审计；
- 分类器采用最近质心（Nearest Centroid）而非复杂模型，突出 CV 本身；
- 指标只保留 accuracy，避免过度工程化；
- 不做并行和分布式，保持脚本小而真实；
- 数据采用可复现合成数据，避免额外数据依赖。

## R10

`demo.py` 函数职责：
- `NearestCentroidClassifier`：示例分类器（fit/predict）。  
- `validate_xy`：输入合法性检查。  
- `stratified_kfold_indices`：手写分层 K 折索引。  
- `accuracy_score_np`：准确率计算。  
- `cross_validate_manual`：执行单候选交叉验证。  
- `generate_synthetic_multiclass_data`：生成可复现测试数据。  
- `evaluate_candidates`：比较多个候选并选出最优。  
- `main`：组织流程并打印输出。

## R11

运行方式：

```bash
cd Algorithms/数学-机器学习-0405-交叉验证
python3 demo.py
```

脚本不读取命令行参数，不请求用户输入。

## R12

输出字段说明：
- `dataset`：数据集类型、样本数、特征数、类别数。  
- `n_splits/random_state`：折数和随机种子。  
- `Candidate summary`：候选参数级评估表：
  - `rank`：按平均准确率排序后的名次；
  - `candidate_id`：候选编号；
  - `params`：参数配置；
  - `mean_accuracy`：折均准确率；
  - `std_accuracy`：折准确率标准差；
  - `oof_accuracy`：OOF 全局准确率。  
- `Best params by mean CV accuracy`：最优候选参数。  
- `Fold details (best candidate)`：最优候选每折训练/验证样本数与准确率。

## R13

建议最小测试集：
1. 正常路径（已内置）
- 三分类合成数据；
- 候选参数 `metric_p in {1, 2}`；
- `K=5` 分层交叉验证。

2. 异常路径（建议补充）
- `n_splits=1`（应报错）；
- 删除某类样本使其小于 `K`（应报错）；
- 向 `X` 注入 `np.nan`（应报错）；
- `candidate_params=[]`（应报错）。

3. 重复性测试（建议补充）
- 固定种子多次运行，输出结果应一致。

## R14

关键可调参数：
- `n_splits`：折数，越大评估更稳定但成本更高。  
- `random_state`：影响分层打乱和可复现性。  
- `metric_p`：最近质心距离度量（L1/L2 或更高阶 Minkowski）。  
- `n_classes/n_per_class/n_features`：合成数据规模和难度。

调参建议：
- 教学演示可用 `K=5`；
- 数据更小时可尝试 `K=10`；
- 候选参数过多时优先减少搜索空间以控制成本。

## R15

方法对比：
- 对比单次留出法（Hold-out）：
  - CV 更稳健，方差更小；
  - 但训练次数更多。  
- 对比留一法（LOOCV）：
  - LOOCV 计算代价高，K 折更常用。  
- 对比分层与非分层 K 折：
  - 分层在类别分布不均时通常更可靠。

## R16

典型应用场景：
- 分类模型的离线评估与参数筛选；  
- 小样本场景下尽量充分利用数据；  
- 教学中解释“训练-验证轮换”机制；  
- 生产前的模型稳定性基线检查。

## R17

可扩展方向：
- 支持更多指标（F1、AUC、LogLoss）；  
- 增加重复 K 折评估（Repeated CV）；  
- 支持分组交叉验证（Group K-Fold）；  
- 增加时间序列 CV（滚动窗口）；  
- 增加结果落盘与可视化分析。

## R18

`demo.py` 源码级算法流程（8 步）：
1. `main` 调用 `generate_synthetic_multiclass_data` 生成固定随机种子的三分类数据。  
2. `main` 设置 `candidate_params=[{"metric_p":1}, {"metric_p":2}]`，并调用 `evaluate_candidates`。  
3. `evaluate_candidates` 逐个候选调用 `cross_validate_manual`，收集均值/标准差/OOF 指标。  
4. `cross_validate_manual` 先调用 `validate_xy` 校验输入，再通过 `stratified_kfold_indices` 获取每折索引。  
5. `stratified_kfold_indices` 对每个类别单独打乱并 `array_split` 到 `K` 个桶，保证分层划分。  
6. 每一折中，克隆 `NearestCentroidClassifier`，在训练折 `fit`（计算各类质心），在验证折 `predict`（按最小距离分配类别）。  
7. 用 `accuracy_score_np` 计算折准确率，并将验证样本预测写入 `oof_pred`，折循环结束后得到 `mean/std/oof_acc`。  
8. `evaluate_candidates` 根据 `mean_accuracy` 选出最佳参数，`main` 打印候选汇总表和最优候选折级明细。
