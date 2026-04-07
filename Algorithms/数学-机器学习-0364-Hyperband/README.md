# Hyperband

- UID: `MATH-0364`
- 学科: `数学`
- 分类: `机器学习`
- 源序号: `364`
- 目标目录: `Algorithms/数学-机器学习-0364-Hyperband`

## R01

Hyperband 是一种面向“超参数优化 + 有限训练预算”的多保真搜索算法。  
核心思想：
- 把“训练资源”（如 epoch、样本数、迭代步）作为可分配预算；
- 先用低预算快速试很多配置，再逐轮淘汰差配置，把高预算留给好配置；
- 通过多个 bracket（不同探索/利用比例）覆盖不同策略偏好。

本目录 MVP 使用“手写逻辑回归 + mini-batch SGD”做二分类超参数搜索，资源定义为训练 epoch 数。

## R02

本实现求解的问题：
- 目标：
  - 在给定配置空间中最小化验证集损失 `val_logloss`；
- 输入：
  - 最大资源 `R=max_resource`；
  - 缩减率 `eta`；
  - 随机种子与配置采样空间；
- 输出：
  - Hyperband 找到的最优配置；
  - 全部评估记录（每次在某个资源层上得到的验证损失与准确率）；
  - 每个 bracket/stage 的淘汰统计；
  - 最优配置在 test 集上的最终指标（train+val 复训后评估）。

`demo.py` 固定参数运行，不需要任何交互输入。

## R03

关键数学关系（标准 Hyperband 记号）：

1. 最大 bracket 深度：
   - `s_max = floor(log_eta(R))`
2. 总预算常数：
   - `B = (s_max + 1) * R`
3. 对每个 bracket `s in {s_max, ..., 0}`：
   - `n = ceil((B / R) * eta^s / (s+1))`
   - `r = R * eta^{-s}`
4. 在 bracket 内第 `i` 轮（Successive Halving）：
   - `n_i = floor(n * eta^{-i})`
   - `r_i = r * eta^i`
5. 每轮按验证损失排序后，只保留前 `floor(n_i / eta)` 个配置进入下一轮。

其中 `eta` 越大，淘汰越激进。

## R04

算法流程（高层）：
1. 准备数据集并切分 train/val/test，做标准化。  
2. 设定 `R`、`eta`，计算 `s_max` 与 `B`。  
3. 从大 `s` 到小 `s` 枚举 bracket。  
4. 在每个 bracket 中随机采样 `n` 个配置。  
5. 在每个 stage 用资源 `r_i` 训练并评估全部候选。  
6. 按验证损失排序，保留 top 候选进入下一 stage。  
7. 记录所有 stage 统计与单次评估日志。  
8. 选出全局最优配置，并在 train+val 上用满资源复训，最后在 test 上报告泛化指标。

## R05

核心数据结构：
- `HyperConfig`：单个超参数配置（`alpha/eta0/batch_size/momentum/seed`）；
- `EvalRecord`：一次评估日志（`s/i/config_id/resource/val_logloss/val_accuracy`）；
- `StageRecord`：一个 stage 的统计摘要（目标候选数、资源层、评估数、保留数、best/median loss）；
- `DatasetPack`：`train/val/test` 数据容器；
- `HyperbandResult`：最终返回体（最优配置、验证与测试指标、全量日志）。

## R06

正确性与稳定性要点：
- 预算合法性检查：`max_resource>=1`、`eta>=2`；
- 资源离散化：`r_i` 会裁剪到 `[1, R]`，避免无效 epoch；
- 淘汰轮至少保留 1 个候选，避免空集合；
- 所有随机过程（数据构建、配置采样、epoch 打乱）都绑定种子，可复现；
- 通过 `logit/参数/梯度` 裁剪与概率下界保护，避免 `log_loss` 数值溢出。

## R07

复杂度分析：
- 若单次训练 `r` 个 epoch 的代价记为 `C(r)`，则一次 bracket 的代价约为
  - `sum_i n_i * C(r_i)`；
- Hyperband 总代价为各 bracket 之和。

在本实现中，`C(r)` 近似与 `r` 线性：
- `C(r) = O(r * N * d)`（`N` 为训练样本数，`d` 为特征维）；
- 额外排序开销约 `O(n_i log n_i)`，通常小于训练开销。

空间复杂度主要来自：
- 数据集缓存 `O(N*d)`；
- 日志记录 `O(#evaluations)`。

## R08

边界与异常处理：
- 参数越界：`resource_epochs<=0`、`eta<2`、`R<1` 会抛 `ValueError`；
- 训练失败保护：若未产生任何评估结果，会抛 `RuntimeError`；
- 数据/概率非数问题通过标准化与概率裁剪降低风险；
- 配置空间包含学习率、L2 系数、batch 大小和动量系数，覆盖“步长/正则/优化动力学”三个关键维度。

## R09

MVP 取舍：
- 只做单机串行 Hyperband，不做并行调度；
- 只示范二分类任务；
- 资源定义为 epoch，不扩展到样本子集或 wall-time；
- 不依赖 Ray Tune / Optuna Hyperband 调度器，核心逻辑全部手写可追踪；
- 配置空间保持小而代表性（学习率、L2 正则、batch 大小、动量）。

## R10

`demo.py` 主要函数职责：
- `build_dataset`：构造可复现的二分类数据并完成切分与标准化；
- `sample_config`：随机采样超参数配置；
- `train_for_resource`：给定 `resource_epochs` 训练并返回验证指标；
- `run_hyperband`：实现 Hyperband 外循环 + Successive Halving 内循环；
- `print_stage_table`：打印每个 stage 的预算与淘汰概览；
- `print_top_evaluations`：展示最优若干次评估；
- `format_config`：把配置整理为易读 dict；
- `main`：组织运行与汇总输出。

## R11

运行方式：

```bash
cd Algorithms/数学-机器学习-0364-Hyperband
python3 demo.py
```

无需命令行参数，无需交互输入。

## R12

输出字段说明：
- `Bracket/Stage summary`：
  - `s, i`：bracket 与其内部 stage 编号；
  - `target_n, target_r`：理论候选数与本层资源；
  - `evaluated, kept`：实际评估数与保留数；
  - `best_logloss, median_logloss`：本层最佳/中位验证损失。
- `Top evaluations`：
  - 按 `val_logloss` 排序的前若干次评估详情。
- `Best configuration`：
  - Hyperband 选出的最优超参数。
- `Best validation` 与 `Test (...)`：
  - 最优配置的验证结果，以及在 train+val 复训后的测试结果。

## R13

最小测试建议：
1. 固定默认参数运行一次，确认脚本可直接执行并有稳定输出；
2. 把 `max_resource` 改为 `9/27/81` 做对比，观察预算变化对结果的影响；
3. 把 `eta` 改为 `2` 与 `4`，观察“保留强度”变化；
4. 缩小或扩大配置空间范围，验证搜索鲁棒性；
5. 更换随机种子，观察结果方差。

## R14

关键参数与调优建议：
- `max_resource (R)`：
  - 越大，单配置训练更充分，但总耗时上升；
- `eta`：
  - 越大淘汰越激进，探索面更广但误淘风险更高；
- 配置空间范围：
  - `alpha`、`eta0` 的 log 范围直接决定可搜索区域；
- 数据规模：
  - 可通过样本量控制演示耗时；
- 评估指标：
  - 当前以 `val_logloss` 为主，可替换为 `1-accuracy` 或其他任务指标。

## R15

方法对比：
- 对比网格搜索：
  - 网格搜索通常给每个配置同等完整预算；Hyperband 会早停差配置，预算效率更高。
- 对比随机搜索：
  - 随机搜索虽然简单，但缺少分层淘汰机制；Hyperband 在同预算下可尝试更多初始配置。
- 对比贝叶斯优化：
  - 贝叶斯优化利用代理模型指导采样，单点评估更“聪明”；Hyperband实现简单且并行友好。

## R16

典型应用场景：
- 深度学习/线性模型超参数调优（epoch 可天然作为资源）；
- AutoML 的预算受限搜索；
- 快速实验筛选：先粗筛再精训；
- 训练成本高、配置空间较大时的首轮 baseline 调度器。

## R17

可扩展方向：
- 改为异步版（ASHA）以提高多 worker 利用率；
- 支持多目标（如精度 + 时延）排序；
- 引入更复杂的条件配置空间；
- 对接真实训练任务（PyTorch 模型）把资源定义为 step/epoch；
- 增加结果持久化（CSV/JSON）与可视化（收敛曲线、淘汰树）。

## R18

`demo.py` 的源码级算法流（8 步）：
1. `main` 固定 `dataset_seed/hyperband_seed/max_resource/eta`，调用 `build_dataset` 与 `run_hyperband`。  
2. `build_dataset` 生成二分类数据，做 train/val/test 分层切分与“基于 train 统计量”的手写标准化。  
3. `run_hyperband` 先计算 `s_max=floor(log_eta(R))` 与 `B=(s_max+1)R`，随后从 `s_max` 到 `0` 逐个处理 bracket。  
4. 每个 bracket 中，按公式计算 `n` 与 `r`，调用 `sample_config` 采样 `n` 个候选超参数。  
5. 在该 bracket 的每个 stage，计算 `n_i/r_i`，并对当前候选逐个调用 `train_for_resource`：
   - 用手写逻辑回归 mini-batch SGD 训练 `r_i` 个 epoch；
   - 计算 `val_logloss` 和 `val_accuracy`。  
6. 将每次评估写入 `EvalRecord`，并在 stage 内按 `val_logloss` 升序排序；若非最后一层，仅保留前 `floor(n_i/eta)` 个候选。  
7. 所有 bracket 结束后，得到全局最佳配置 `global_best_cfg`；再把 train+val 合并，用满资源 `R` 复训该配置并在 test 上评估。  
8. `main` 最后调用 `print_stage_table`、`print_top_evaluations` 与配置摘要输出，形成完整可审计结果。

该实现没有调用现成 Hyperband 调度器，预算分配、分层淘汰、评估记录与最终选优都在源码中逐步展开。
