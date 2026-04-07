# 早停法

- UID: `MATH-0403`
- 学科: `数学`
- 分类: `机器学习`
- 源序号: `403`
- 目标目录: `Algorithms/数学-机器学习-0403-早停法`

## R01

早停法（Early Stopping）是机器学习中最常用的正则化策略之一：
- 训练阶段持续优化训练损失；
- 同时监控独立验证集指标（通常是 `val_loss`）；
- 当验证集长期不再提升时提前停止训练，避免继续拟合训练噪声。

它的核心不是“把模型训到最小训练误差”，而是“在泛化误差最低附近停止”。

## R02

本目录 MVP 聚焦的具体问题：

在带噪声二分类任务上，使用 NumPy 手写逻辑回归训练循环，并实现可审计的早停机制（`patience + min_delta + restore_best`），比较：
- 使用早停（基于验证集）
- 不使用早停（固定训练到 `max_epochs`）

输出比较包括：
- 最优验证损失出现的轮次；
- 是否提前停止、停止轮次；
- 验证集与测试集损失/准确率差异；
- 训练过程日志快照（`pandas.DataFrame`）。

## R03

实验设计选择：
- 数据：使用 NumPy 构造非线性二分类分数并注入标签噪声；
- 特征：二次多项式展开（增加特征维度，制造过拟合压力）；
- 划分：训练/验证/测试三段式切分，确保验证指标可用于停止判据；
- 优化：小批量梯度下降训练逻辑回归；
- 正则：仅轻微 `L2`，让“继续训练导致验证恶化”的现象更容易观察。

## R04

设样本为 `(x_i, y_i)`，`y_i in {0,1}`，模型为：

`p_i = sigmoid(w^T x_i + b)`

验证集早停监控的核心损失是二元交叉熵：

`L_val = (1/m) * sum_j [log(1 + exp(z_j)) - y_j * z_j]`

其中 `z_j = w^T x_j + b`。

早停判据：
- 若 `L_val < best_val_loss - min_delta`，记为“有改进”，更新最优参数并把等待计数 `wait=0`；
- 否则 `wait += 1`；
- 当 `wait >= patience` 时停止；
- 若 `restore_best=True`，训练结束后回滚到最佳验证轮对应参数。

## R05

算法高层流程：

1. 构造数据并划分 train/val/test。
2. 做特征标准化，保证梯度下降数值稳定。
3. 初始化参数 `w,b` 与早停状态（`best_val_loss, best_epoch, wait`）。
4. 每个 epoch 内按 mini-batch 计算梯度并更新参数。
5. 每个 epoch 结束后计算 `train_loss/train_acc` 与 `val_loss/val_acc`。
6. 用 `val_loss` 执行早停判据并记录日志行。
7. 触发 `patience` 时停止训练。
8. （可选）恢复最佳验证轮参数，评估验证与测试结果。

## R06

正确性要点：
- 验证集与训练集严格分离：早停只看验证指标，不看测试集；
- 损失实现使用 `logaddexp` 形式，避免 sigmoid 溢出；
- 每轮记录 `best_val_loss` 与 `wait`，可复核是否正确触发停止；
- `restore_best=True` 防止“最后几轮已过拟合”导致最终模型退化；
- 对照组（无早停）保持相同学习率/批大小/初始化种子，保证比较公平。

## R07

复杂度分析（设训练样本数 `N`、特征维度 `d`、训练轮数 `T`）：
- 单 batch 主要成本为矩阵向量乘，约 `O(B*d)`；
- 单 epoch 遍历全体训练样本，约 `O(N*d)`；
- 总训练复杂度约 `O(T*N*d)`；
- 存储开销：模型参数 `O(d)`，历史日志 `O(T)`。

早停减少的是有效 `T`，因此常直接减少总训练时间。

## R08

实现中的边界与异常处理：
- `X` 维度不是 2D、`y` 不是 1D：抛 `ValueError`；
- 训练与标签样本数不匹配：抛 `ValueError`；
- 学习率/轮数/批大小非法（非正）：抛 `ValueError`；
- `patience < 0`：抛 `ValueError`。

这些检查在 `fit_logistic_regression` 入口统一执行，避免训练后期才暴露问题。

## R09

MVP 取舍说明：
- 使用 `numpy + pandas`，不依赖高层黑盒训练器；
- 不调用任何现成分类器拟合接口来替代训练循环；
- 重点展示早停逻辑本身，不扩展到分布式训练、混合精度、复杂调度器；
- 保留一个可直接运行、可读、可审计的小实现。

## R10

`demo.py` 主要函数职责：
- `make_dataset`：生成带噪声二分类数据，做多项式扩展、切分、标准化。
- `sigmoid` / `bce_from_logits`：提供数值稳定的概率和损失计算。
- `fit_logistic_regression`：手写 mini-batch 训练循环并执行早停。
- `evaluate_dataset`：统一输出损失与准确率。
- `print_history_snapshot`：打印训练日志前后片段，便于肉眼审计。
- `main`：运行“有早停/无早停”对照实验并打印结果。

## R11

运行方式：

```bash
cd Algorithms/数学-机器学习-0403-早停法
python3 demo.py
```

脚本无需任何交互输入，运行后直接输出实验结果。

## R12

关键输出字段解释：
- `stopped_early`：是否触发提前停止。
- `stopped_epoch`：触发停止时的 epoch。
- `best_epoch`：验证损失最佳的 epoch。
- `best_val_loss`：历史最小验证损失。
- `val_loss/val_acc`：最终参数在验证集上的损失/精度。
- `test_loss/test_acc`：最终参数在测试集上的损失/精度。
- `(test_loss_without_es - test_loss_with_es)`：正值通常意味着早停改善泛化。

## R13

内置最小实验配置：
- 数据规模：`n_samples=1200`；
- 原始特征：`20` 维，二次扩展后约 `230` 维；
- 标签噪声：`flip_y=0.18`；
- 优化器：mini-batch GD（`batch_size=64`）；
- 训练上限：`max_epochs=300`；
- 早停参数：`patience=18`, `min_delta=1e-4`；
- 随机种子：`403`（可复现）。

## R14

调参建议：
1. `patience`：太小会过早停止，太大则接近不早停；可先在 `10~30` 搜索。
2. `min_delta`：定义“有效改进”的阈值；噪声大时可适当增大。
3. `learning_rate`：影响验证曲线平滑度；学习率过大会让早停判断被震荡误导。
4. `restore_best`：建议保持开启，否则停止点参数可能不是最佳泛化点。

## R15

与相关方法对比：
- 对比仅靠 `L2`：
  - `L2` 约束参数大小；
  - 早停约束有效训练时长；二者可叠加。
- 对比学习率衰减：
  - 衰减让训练“继续但更慢”；
  - 早停直接结束训练预算。
- 对比固定 epoch：
  - 固定 epoch 需要人为猜最佳轮次；
  - 早停由验证集自适应选择停止时间。

## R16

典型应用场景：
- 神经网络、梯度提升、线性模型等迭代式训练；
- 数据规模中等、验证集可稳定反映泛化趋势的任务；
- 需要节省训练预算并降低过拟合风险的工程场景。

## R17

可扩展方向：
- 监控多指标（如 `val_loss + val_f1`）并设计联合停止规则；
- 引入滑动平均/指数平滑降低验证噪声对判据的干扰；
- 支持 `k` 折交叉验证下的稳健早停策略；
- 扩展到 PyTorch 模型并复用同样的 `patience/min_delta/restore_best` 框架。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 调用 `make_dataset`：生成噪声二分类数据，做二次特征扩展、训练/验证/测试切分与标准化。
2. `fit_logistic_regression` 初始化参数 `w,b` 与早停状态变量：`best_val_loss=inf, best_epoch=0, wait=0`。
3. 进入 epoch 循环：每轮先打乱训练索引，再按 mini-batch 取数据。
4. 每个 batch 内计算 `logits -> sigmoid -> error`，得到梯度 `grad_w, grad_b` 并执行梯度下降更新。
5. 每轮结束后计算 `train_loss/train_acc` 与 `val_loss/val_acc`，把指标写入 `rows`。
6. 执行早停判据：若 `val_loss < best_val_loss - min_delta` 则更新最优快照并清零 `wait`，否则 `wait += 1`。
7. 当 `early_stopping=True` 且 `wait >= patience` 时跳出训练循环；若 `restore_best=True` 则回滚到最佳验证轮参数。
8. `main` 分别运行“有早停”和“无早停”两组训练，评估验证/测试损失与准确率，并打印历史快照用于审计。
