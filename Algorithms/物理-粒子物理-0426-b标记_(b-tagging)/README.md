# b标记 (b-tagging)

- UID: `PHYS-0406`
- 学科: `物理`
- 分类: `粒子物理`
- 源序号: `426`
- 目标目录: `Algorithms/物理-粒子物理-0426-b标记_(b-tagging)`

## R01

`b` 标记（b-tagging）是在喷注重建后识别“该喷注是否来自 `b` 夸克”的分类任务。其核心思想是利用 `b` 强子寿命较长、次级顶点显著、半轻子衰变特征更明显等性质，把 `b`-jet 与 `c`/light-jet 区分开。

本条目提供一个可运行 MVP：
- 生成带 `b/c/light` 味标签的 toy 喷注特征数据；
- 用 NumPy 手写逻辑回归训练一个 `b` 标记器；
- 输出 ROC-AUC 与固定工作点（70%、80% b效率）的误标率。

## R02

问题定义（对应 `demo.py`）：
- 输入：喷注特征向量 `x in R^d`，其中 `d=7`。
- 标签：`y=1` 表示 b-jet，`y=0` 表示非 b-jet（`c + light`）。
- 模型输出：`P(y=1|x)`。
- 目标：在较高 b效率下，压低非 b 误标率，尤其是 light-jet mistag。

## R03

物理背景（简化）：
- `b` 强子典型寿命约 `O(1 ps)`，可在探测器内形成可分辨的位移衰变顶点。
- 因此 `b`-jet 在以下变量上通常更“硬”：
  - 轨迹冲击参数显著性（IP significance）更高；
  - 次级顶点质量、飞行距离显著性更高；
  - 次级顶点轨迹数更多。
- `c`-jet 介于 `b` 与 light 之间，是最常见混淆来源之一。

## R04

MVP 使用的 7 个特征：
- `ip3d_sig`：3D 冲击参数显著性。
- `sv_mass`：次级顶点不变质量（GeV）。
- `sv_flight_sig`：次级顶点飞行距离显著性。
- `n_sv_tracks`：次级顶点关联轨迹数。
- `soft_lepton_ptrel`：软轻子相对喷注轴横向动量（GeV）。
- `jet_pt`：喷注横向动量（GeV）。
- `jet_width`：喷注横向宽度（`eta-phi` 空间）。

## R05

模型与目标函数：

- 线性打分：`z = w^T x + b`
- 概率输出：`p = sigma(z) = 1 / (1 + exp(-z))`
- 二元交叉熵：
  `L = -mean(y log p + (1-y) log(1-p)) + (lambda/2) ||w||_2^2`

训练采用全批量梯度下降（full-batch GD），不依赖 `sklearn` 的分类器黑盒。

## R06

算法流程（高层）：
1. 用 `generate_toy_btag_dataframe` 生成 `b/c/light` 三类喷注样本并打乱。
2. 拆分训练/测试集（分层抽样，保证 b 占比稳定）。
3. 用训练集统计均值/方差，对特征做 z-score 标准化。
4. 执行 `fit_logistic_regression`：循环计算前向概率、梯度、参数更新。
5. 在测试集得到 `b` 概率分数。
6. 计算 ROC-AUC。
7. 在目标 b效率（70%、80%）下反推阈值，并报告非 b / c / light mistag。

## R07

复杂度分析：
- 设样本数 `N`，特征维度 `d`，训练轮数 `T`。
- 每轮梯度计算主要是矩阵向量乘法，时间复杂度 `O(Nd)`。
- 总训练复杂度 `O(TNd)`。
- 空间复杂度 `O(Nd)`（保存数据）+ `O(d)`（参数）。

## R08

正确性直觉：
- 如果 `b`-jet 特征在统计上与非 b-jet 可分，逻辑回归会学习到一个近似最优线性判别超平面。
- 由于输出是连续概率，可按不同实验工作点选择阈值，实现“效率-纯度”折中。
- 对于 `c` 与 `b` 相似度较高的情况，`c` mistag 往往高于 light mistag，这在输出中应可观察到。

## R09

伪代码：

```text
df <- generate_toy_btag_dataframe(seed)
X, y, flavor <- extract features and labels
train_idx, test_idx <- stratified split
mu, sigma <- fit standardization on X_train
X_train_z <- (X_train - mu) / sigma
X_test_z <- (X_test - mu) / sigma

initialize w=0, b=0
for t in [1..T]:
    p <- sigmoid(X_train_z @ w + b)
    grad_w <- X_train_z^T (p - y_train) / N + lambda * w
    grad_b <- mean(p - y_train)
    w <- w - lr * grad_w
    b <- b - lr * grad_b

score <- sigmoid(X_test_z @ w + b)
auc <- roc_auc(y_test, score)
for target_eff in {0.70, 0.80}:
    thr <- quantile(score[y_test==1], 1-target_eff)
    report b_eff and mistag rates at thr
```

## R10

数值与边界处理：
- `sigmoid` 前对 `z` 做 `[-40, 40]` 截断，防止指数溢出。
- 交叉熵对概率做 `eps` 裁剪，防止 `log(0)`。
- 标准化时对极小方差特征以 `1.0` 代替，避免除零。
- 工作点阈值函数检查 `target_eff in (0,1]` 且正类样本非空。

## R11

默认超参数：
- 数据规模：`n_b=2200, n_c=1400, n_light=2400`。
- 训练：`epochs=1200, lr=0.08, l2=1e-3`。
- 切分：`test_size=0.30`，`stratify=y`。
- 随机种子：`seed=426`（数据 + 切分均固定，保证可复现）。

## R12

`demo.py` 实现范围：
- 覆盖从数据生成、训练、评估到结果打印的完整链路。
- 不调用第三方分类器训练；模型更新由 NumPy 显式实现。
- 使用 `pandas` 组织输出表格，`sklearn` 仅用于 `train_test_split` 与 `roc_auc_score`。

## R13

运行方式：

```bash
cd Algorithms/物理-粒子物理-0426-b标记_(b-tagging)
uv run python demo.py
```

脚本无交互输入，执行后直接输出数据规模、AUC、工作点误标率和系数排序。

## R14

示例输出结构（数值会随样本波动但在固定 seed 下可复现）：

```text
=== Toy b-tagging dataset summary ===
flavor  n_jets
     b    2200
     c    1400
 light    2400

=== Model quality ===
Train size: 4200, Test size: 1800
ROC-AUC (test): 0.97xx
...

=== Working points ===
working_point  threshold  b_eff  mistag_nonb  mistag_c  mistag_light
    70% b-eff     ...     0.7000    ...         ...         ...
    80% b-eff     ...     0.8000    ...         ...         ...
```

## R15

最小验收清单：
- `README.md` 与 `demo.py` 已全部补全，无占位符残留。
- `uv run python demo.py` 可直接运行完成。
- 输出包含：
  - 数据集 flavor 统计；
  - ROC-AUC；
  - 至少两个工作点（70%、80%）的 b效率与 mistag；
  - 特征系数排序。

## R16

当前 MVP 局限：
- 使用 toy 分布，不等价于真实 ATLAS/CMS 探测器重建数据。
- 仅二分类（b vs non-b），未做多类别端到端训练。
- 线性模型无法捕捉复杂非线性关系（真实 b-tag 常用深度网络或 GBDT）。
- 未包含系统误差、域偏移、校准与 scale factor 流程。

## R17

可扩展方向：
- 引入更真实的 track/vertex 层级特征与图结构输入。
- 在相同数据上比较逻辑回归、GBDT、DNN（并保留可解释性对照）。
- 加入温度缩放/等值回归等概率校准。
- 增加按 `jet_pt/eta` 分桶的性能曲线与不确定度评估。
- 接入真实开放 HEP 数据格式，构建训练-验证-推理完整管线。

## R18

`demo.py` 源码级算法流（8 步，非黑盒）：
1. `generate_toy_btag_dataframe` 分 flavor 采样基础物理特征分布，并注入弱相关性，得到 `DataFrame`。  
2. `main` 提取 `X/y/flavor`，用 `train_test_split(stratify=y)` 生成训练与测试索引。  
3. `standardize_fit` 在训练集计算 `mean/std`，`standardize_apply` 对训练/测试分别做 z-score。  
4. `fit_logistic_regression` 初始化 `w,b` 后循环：`logits -> sigmoid -> error -> grad_w/grad_b -> 参数更新`。  
5. 在训练过程中周期性调用 `binary_cross_entropy` 记录 loss 检查收敛趋势。  
6. `predict_proba` 在测试集输出 `P(b-jet)` 分数，`roc_auc_score` 计算整体排序质量。  
7. `threshold_for_target_efficiency` 在正类分数分位数上反推阈值，实现固定 b效率工作点。  
8. `evaluate_working_point` 分别计算 `mistag_nonb/mistag_c/mistag_light`，并结合 `coefficient_table` 输出可解释结果。

说明：第三方库没有替代核心训练算法；核心分类器更新逻辑完全在源码中展开。
