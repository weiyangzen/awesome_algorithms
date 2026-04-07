# 神经架构搜索 (NAS)

- UID: `MATH-0408`
- 学科: `数学`
- 分类: `AutoML`
- 源序号: `408`
- 目标目录: `Algorithms/数学-AutoML-0408-神经架构搜索_(NAS)`

## R01

神经架构搜索（Neural Architecture Search, NAS）要解决的问题是：
在给定计算预算下，自动从候选神经网络结构中找到泛化性能更好的架构，而不是完全依赖人工试错。

本目录实现一个最小可运行 NAS MVP：

- 用离散搜索空间定义一组 MLP 架构候选；
- 在验证集上比较候选性能并排序；
- 选出最优架构后在 `train+val` 上重训；
- 最后在独立测试集报告结果。

## R02

这里的 NAS 被建模为一个组合优化问题：

- 搜索变量：网络深度、隐藏层宽度、激活函数；
- 目标函数：验证集准确率最大化；
- 约束：固定训练轮数与固定搜索空间，不使用人工调参回路。

形式化可写为：
`a* = argmax_{a in A} Acc_val( Train(w; a, D_train), D_val )`，
其中 `A` 是架构集合，`w` 是训练得到的参数。

## R03

`demo.py` 中的离散搜索空间为 18 个候选：

- `depth in {1, 2, 3}`
- `hidden_dim in {16, 32, 64}`
- `activation in {relu, tanh}`

总规模：`3 * 3 * 2 = 18`。

每个候选都是一个前馈 MLP，输入维度 2（二分类平面点），输出维度 2（类别 logits）。

## R04

评分准则使用验证集准确率：

- 训练损失：`CrossEntropy(logits, y)`
- 验证指标：`val_acc = mean(argmax(logits)==y)`
- 选择策略：按 `val_acc` 降序排序，最高者入选

最终泛化性能由测试集准确率给出，但测试集不参与架构选择。

## R05

高层流程：

1. 生成并标准化二分类数据；
2. 切分 `train / val / test`；
3. 枚举所有候选架构；
4. 对每个架构独立训练并记录最佳验证准确率；
5. 汇总为表格并选出排名第一架构；
6. 用该架构在 `train+val` 上重训；
7. 在 `test` 上进行一次最终评估。

## R06

数据集使用 `sklearn.datasets.make_moons` 生成可复现实验数据：

- 样本数：`1500`
- 噪声：`0.25`
- 随机种子：`408`
- 划分：`60% train / 20% val / 20% test`

随后使用 `StandardScaler` 仅基于训练集拟合，再作用于验证集和测试集，避免数据泄漏。

## R07

候选模型结构细节：

- 每个隐藏层块为：`Linear -> Activation`；
- 激活函数由架构变量控制（`ReLU` 或 `Tanh`）；
- 输出层为 `Linear(hidden_dim, 2)`，不加 softmax（交叉熵内部处理）。

参数规模通过 `count_trainable_parameters` 统计，并写入结果表用于比较“性能-复杂度”关系。

## R08

搜索阶段训练配置：

- 优化器：`AdamW`
- 学习率：`1e-2`
- `weight_decay=1e-4`
- epoch：`10`
- batch size：`64`

重训阶段配置（用于最终模型）：

- 在 `train+val` 上训练 `25` 个 epoch；
- 学习率 `7e-3`，`weight_decay=5e-5`；
- 最终只报告一次测试集准确率。

## R09

复杂度分析（设候选数 `M`，每个架构参数量 `P`，训练步数 `T`）：

- 搜索成本约为 `O(M * T * P)`（忽略常数与 batch 切分项）；
- 存储成本约为 `O(P)`（逐个模型训练，不并发常驻）；
- 当 `M` 变大时，搜索时间线性增长，是离散 NAS 的主要瓶颈。

本 MVP 通过限制候选数到 18，保证脚本可快速运行。

## R10

输出结果包括：

- Top-10 架构表（按 `val_acc` 降序）；
- 每个候选的 `depth/hidden_dim/activation/params/best_epoch/val_acc`；
- 被选中架构及其 `Validation accuracy`；
- 重训后的 `Train+Val accuracy` 与 `Test accuracy`。

这使“搜索结果”与“最终落地性能”可同时审阅。

## R11

运行方式：

```bash
cd Algorithms/数学-AutoML-0408-神经架构搜索_(NAS)
uv run python demo.py
```

脚本不需要交互输入，运行结束后直接输出排名与最终指标。

## R12

`demo.py` 的函数分工：

- `build_dataset`：生成数据并做标准化与三段切分；
- `all_candidates`：构建离散架构空间；
- `build_model`：按架构描述实例化 MLP；
- `train_with_validation`：训练并记录最佳验证准确率；
- `run_search`：枚举并评估所有候选；
- `results_to_dataframe`：形成排序结果表；
- `retrain_best_and_test`：重训最优架构并评估测试集；
- `main`：组织端到端流程与质量门禁。

## R13

脚本内置了可执行质量门禁：

1. 搜索结果行数必须等于 18；
2. `val_acc` 必须在 `[0,1]` 且无 `NaN`；
3. 最优验证准确率必须不低于中位数（排序有效性）；
4. 最优 `val_acc > 0.85`；
5. 最终 `test_acc > 0.84`。

这些断言避免“脚本能跑但结果无效”的情况。

## R14

常见失效模式与应对：

- 失效 1：搜索空间太小，错过更优架构。  
  应对：扩展 `depth/width/activation` 或加入正则化选项。

- 失效 2：验证集过拟合，选择偏差大。  
  应对：增加重复切分、交叉验证或独立再验证。

- 失效 3：训练轮数过短，候选比较不公平。  
  应对：统一增加 epoch 或采用早停+学习率调度。

- 失效 4：单随机种子偶然性。  
  应对：多种子重复搜索并比较均值/方差。

## R15

与其他 AutoML 路线对比：

- 相比纯超参搜索：NAS 直接优化网络拓扑，而非仅优化学习率等训练参数；
- 相比可微 NAS（如 DARTS）：本实现更朴素、可解释、易复现，但效率较低；
- 相比进化 NAS：这里不做种群和变异，仅全枚举离散小空间，便于教学和验证。

## R16

可扩展方向：

- 将搜索算法从枚举升级为随机搜索、贝叶斯优化或进化策略；
- 将搜索空间扩展到卷积核大小、残差连接、归一化层类型；
- 使用多目标评分（精度 + 参数量 + 延迟）；
- 引入早停与代理评估，减少大空间搜索成本。

## R17

本 MVP 的边界与假设：

- 任务是二维 toy 数据二分类，不代表大规模视觉/NLP NAS 难度；
- 候选模型仅为 MLP，不含卷积、注意力或跳连；
- 训练预算固定且较小，结论用于演示流程而非追求 SOTA；
- 结果受搜索空间设计影响很大，不能脱离空间解释“NAS 好坏”。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `main` 调用 `set_global_seed(408)`，再通过 `build_dataset` 生成并标准化 `train/val/test` 张量。  
2. `all_candidates` 枚举 `depth x hidden_dim x activation`，得到 18 个 `ArchSpec`。  
3. `run_search` 逐个候选循环，对第 `i` 个候选先执行 `set_global_seed(10000+i)` 固定随机性。  
4. 每个候选通过 `build_model` 组装 `Linear -> Activation` 堆叠网络，并计算参数量。  
5. `train_with_validation` 使用 `AdamW + CrossEntropyLoss` 训练 10 个 epoch，每轮后计算验证准确率并保留最佳轮次。  
6. `results_to_dataframe` 汇总候选结果并按 `val_acc` 降序排序，`main` 取第一行作为最优架构。  
7. `retrain_best_and_test` 将 `train` 与 `val` 拼接后重训最优架构 25 个 epoch，再在 `test` 上评估最终准确率。  
8. `main` 打印排名与最终指标，并执行断言门禁（结果规模、数值合法性、性能阈值）确保流程正确结束。

这 8 步对应了一个完整的“架构定义 -> 搜索评估 -> 选择 -> 重训 -> 验证”最小 NAS 闭环。
