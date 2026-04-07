# 反向传播算法

- UID: `CS-0110`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `239`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0239-反向传播算法`

## R01

反向传播算法（Backpropagation）是训练神经网络的核心梯度计算方法。它通过链式法则把损失函数对输出层参数的梯度，逐层传回到隐藏层和输入侧参数，使梯度下降能够高效更新全部权重。

本目录 MVP 目标：
- 用 NumPy 手写一个单隐藏层 MLP 的前向与反向传播；
- 在 `make_moons` 二分类任务上完成训练和评估；
- 用数值梯度检查与 PyTorch 自动微分对齐校验反向传播实现正确性。

## R02

问题定义：
- 输入：`X in R^(n*d)`（样本特征）与 `y in {0,1}^n`（二分类标签）。
- 模型：`d -> h -> 1` 的 MLP，隐藏层 `tanh`，输出层 `sigmoid`。
- 输出：训练后的参数 `W1,b1,W2,b2`，以及在测试集上的 `loss/accuracy/f1`。

目标：在可解释、可审计的前提下，演示反向传播如何将误差信号逐层分解为参数梯度。

## R03

本实现使用带 `L2` 正则的二元交叉熵目标：

`L = -mean(y*log(p) + (1-y)*log(1-p)) + (l2/2)*(||W1||^2 + ||W2||^2)`

前向过程：
- `z1 = XW1 + b1`
- `a1 = tanh(z1)`
- `z2 = a1W2 + b2`
- `p = sigmoid(z2)`

反向传播核心梯度：
- `dz2 = (p - y)/n`
- `dW2 = a1^T dz2 + l2*W2`
- `db2 = sum(dz2)`
- `da1 = dz2 W2^T`
- `dz1 = da1 * (1-a1^2)`
- `dW1 = X^T dz1 + l2*W1`
- `db1 = sum(dz1)`

## R04

`demo.py` 的高层流程：
1. 生成并划分 `make_moons` 数据集，做训练集统计标准化。
2. 初始化手写 MLP 参数。
3. 迭代执行 `forward -> loss -> backward -> step`。
4. 周期性记录训练/测试指标到 `pandas.DataFrame`。
5. 训练结束后输出分类报告（`classification_report`）。
6. 执行数值梯度检查（finite difference）验证手写反向传播。
7. 执行 PyTorch autograd 梯度对齐检查。
8. 通过断言门槛后打印 `All checks passed.`。

## R05

核心数据结构：
- `BackpropConfig`：训练与校验参数（学习率、轮数、正则、梯度检查点数等）。
- `BackpropResult`：训练结果摘要（最终指标、最佳准确率、梯度误差）。
- `ManualMLP`：参数容器与算法核心（`forward/backward/step`）。

这些结构使得脚本不是“一次性计算”，而是可复现实验流程。

## R06

正确性关键点：
- 输出层梯度 `dz2 = (p-y)/n` 是 `sigmoid + BCE` 的简化结果；
- 隐层梯度依赖链式法则和 `tanh` 导数 `1-a1^2`；
- 正则项只对权重矩阵加梯度，不对偏置加梯度；
- 梯度检查通过“解析梯度 vs 数值梯度”比对，防止符号或系数错误；
- 再用 PyTorch 自动微分做独立参考，降低实现偏差风险。

## R07

复杂度分析（`n` 为样本数，`d` 为输入维度，`h` 为隐藏维度，`T` 为迭代轮数）：
- 单次前向：`O(n*d*h + n*h)`；
- 单次反向：`O(n*d*h + n*h)`；
- 单轮训练主成本：`O(n*d*h)`；
- 总时间复杂度：`O(T*n*d*h)`；
- 空间复杂度：`O(n*h + d*h + h)`（缓存激活与参数）。

## R08

边界与异常处理：
- 标准化时若某维方差接近 0，会替换为 1 防止除零；
- BCE 计算前对概率做 `clip`，避免 `log(0)`；
- 梯度检查使用中心差分并恢复原值，防止污染参数；
- 全流程无交互输入，固定随机种子保证可复现。

## R09

MVP 取舍：
- 保留：反向传播公式、参数更新、训练日志、双重梯度校验。
- 省略：mini-batch、动量/Adam、多层深网、GPU 训练工程优化。
- 原则：优先“最小但诚实”的算法演示，强调源码透明而不是框架堆叠。

## R10

`demo.py` 主要函数职责：
- `make_dataset`：构造并标准化数据。
- `ManualMLP.forward`：前向传播缓存中间量。
- `ManualMLP.backward`：按链式法则计算梯度。
- `ManualMLP.step`：执行梯度下降更新。
- `evaluate_model`：计算 `loss/acc/f1`。
- `gradient_check`：有限差分梯度检验。
- `torch_gradient_alignment`：与 PyTorch autograd 梯度对齐。
- `train_with_backprop`：组织训练循环与指标采样。
- `main`：串联流程并执行验收断言。

## R11

运行方式：

```bash
cd Algorithms/计算机-机器学习／深度学习-0239-反向传播算法
uv run python demo.py
```

脚本会自动训练并输出结果，无需手工输入。

## R12

输出字段说明：
- `[History Snapshot]`：若干 epoch 的 `train_loss/train_acc/test_loss/test_acc`；
- `[Test Classification Report]`：测试集 precision/recall/f1；
- `[Backprop Checks]`：
  - `gradcheck_max_rel_error`：数值梯度相对误差上界；
  - `torch_grad_max_abs_diff`：手写梯度与 autograd 最大绝对差；
- `All checks passed.`：所有验收条件满足。

## R13

内置最小验收门槛：
1. 最终训练损失必须低于初始训练损失；
2. 最佳测试准确率至少达到 `0.88`；
3. 数值梯度检查误差 `< 1e-4`；
4. 与 PyTorch autograd 梯度最大差 `< 1e-8`。

这些检查覆盖了“能学到”与“梯度算对了”两个核心维度。

## R14

关键超参数与建议：
- `hidden_dim`：隐藏层宽度，越大表达力越强但更易过拟合；
- `lr`：学习率，过大可能震荡，过小收敛慢；
- `epochs`：训练轮数，配合日志观察是否足够；
- `l2`：正则强度，抑制过拟合与参数爆炸；
- `noise`：数据难度，噪声越高任务越难。

调参顺序建议：先稳定 `lr`，再调 `hidden_dim` 与 `l2`，最后按指标调 `epochs`。

## R15

与相关方法对比：
- 感知机：只能线性可分，无法处理 `make_moons` 这类非线性边界；
- 逻辑回归：优化稳定但表达线性；
- 手写反向传播 MLP：可学习非线性决策边界；
- 纯框架黑盒训练：工程方便，但不利于理解梯度流。

本条目选“手写 + 对照校验”，兼顾学习价值与可信度。

## R16

典型应用场景：
- 神经网络训练原理教学；
- 新损失函数/新层结构的梯度推导验证；
- 在上大模型前，先做小规模可解释原型；
- 对自动微分结果做独立审计。

## R17

可扩展方向：
- 增加 mini-batch 与学习率调度；
- 增加多隐藏层并实现通用反向传播框架；
- 接入动量、RMSProp、Adam 对比优化行为；
- 增加决策边界可视化与更多数据集基准；
- 加入早停和交叉验证提升泛化稳定性。

## R18

`demo.py` 的源码级算法流程（8 步）：
1. `make_dataset` 生成 `make_moons` 数据并完成分层切分与标准化。  
2. `ManualMLP` 初始化 `W1,b1,W2,b2`，准备前向和反向计算。  
3. `forward` 计算 `z1 -> a1 -> z2 -> p`，并缓存中间张量供反向传播使用。  
4. `loss` 计算二元交叉熵与 `L2` 正则，得到当前目标值。  
5. `backward` 从 `dz2=(p-y)/n` 出发，按链式法则依次得到 `dW2/db2/dW1/db1`。  
6. `step` 用梯度下降更新参数，训练循环反复执行 `forward->backward->step`。  
7. `gradient_check` 用中心差分近似导数，逐点对比解析梯度；`torch_gradient_alignment` 再与 PyTorch autograd 对齐。  
8. `main` 汇总日志、输出分类报告并执行断言，全部满足后打印 `All checks passed.`。  
