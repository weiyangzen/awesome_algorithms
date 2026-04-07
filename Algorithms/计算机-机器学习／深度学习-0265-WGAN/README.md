# WGAN

- UID: `CS-0127`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `265`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0265-WGAN`

## R01

WGAN（Wasserstein GAN）是 GAN 的稳定化变体，用 Wasserstein-1 距离（Earth Mover Distance）替代原始 GAN 的 JS 散度目标。核心收益是：

1. 训练梯度在分布支撑集不重叠时仍更可用；
2. 损失值（更准确说 critic 差值）与样本质量相关性通常更强；
3. 更不容易出现原始 GAN 的早期梯度消失。

## R02

WGAN 的经典形式来自 2017 年论文《Wasserstein GAN》，关键思想是 Kantorovich-Rubinstein 对偶：

`W(P_r, P_g) = sup_{||f||_L <= 1} E_{x~P_r}[f(x)] - E_{x~P_g}[f(x)]`

其中 `f` 是 1-Lipschitz 函数。实践中用神经网络 `critic` 近似 `f`，并通过参数裁剪（weight clipping）粗略满足 Lipschitz 约束。

## R03

与原始 GAN 相比，WGAN 的两个本质变化：

1. 判别器改称 `critic`，输出不再是概率，而是实数分数；
2. 损失函数从 BCE 对抗改为线性差值目标。

训练目标可写为：

- Critic 最大化：`E[D(x_real)] - E[D(x_fake)]`
- Generator 最小化：`-E[D(x_fake)]`

在代码中通常以“最小化负号形式”实现。

## R04

本目录 `demo.py` 采用最小可运行 WGAN MVP：

1. 数据：`sklearn.datasets.load_digits`（8x8 灰度数字），离线可用；
2. 表示：图像展平为 64 维向量并归一化到 `[-1, 1]`；
3. 生成器：2 层 MLP + `tanh` 输出；
4. Critic：2 层 MLP + 线性输出；
5. 优化器：原论文推荐的 `RMSprop`；
6. 约束：每次 critic 更新后执行权重裁剪。

## R05

复杂度（单个 mini-batch）可近似理解为：

1. 设 batch 大小为 `B`，向量维度 `d=64`，隐藏层宽度 `h`；
2. 一次 critic 前向/反向主要代价约 `O(B * d * h + B * h^2)`；
3. 每轮批次里 critic 更新 `n_critic` 次，生成器更新 1 次；
4. 因此总训练开销约随 `n_critic` 线性放大。

这就是 WGAN 常见的“critic 训练更重”现象。

## R06

最小例子直观说明：

1. 从实分布采样一批真实向量 `x_real`；
2. 从高斯噪声采样 `z`，经生成器得到 `x_fake=G(z)`；
3. critic 计算 `D(x_real)` 与 `D(x_fake)`；
4. critic 损失为 `L_c = mean(D(fake)) - mean(D(real))`；
5. 生成器损失为 `L_g = -mean(D(fake))`；
6. critic 参数每步裁剪到 `[-c, c]`。

## R07

优点：

1. 训练曲线更平滑，调试可读性更好；
2. 相比原始 GAN，模式崩塌倾向通常减轻；
3. 目标函数与分布距离关联更直接。

局限：

1. 权重裁剪会限制 critic 表达能力；
2. 超参数（`clip_value`、`n_critic`）敏感；
3. 生成质量上通常不如后续 WGAN-GP、StyleGAN 等改进版本。

## R08

复现本 MVP 需要的前置知识：

1. GAN 对抗训练基本流程；
2. Wasserstein 距离与对偶形式的直觉；
3. PyTorch 张量与自动求导；
4. `DataLoader` mini-batch 训练；
5. 分布评估指标（如一维 Wasserstein 距离、直方图差异）。

## R09

适用场景：

1. 想验证 GAN 训练稳定化思路的教学/实验；
2. 需要一个比原始 GAN 更稳的最小基线；
3. 关注分布级拟合而不仅是单样本重建。

不适合场景：

1. 直接追求高保真图像 SOTA；
2. 大规模视觉生成生产部署（应优先更先进变体）；
3. 对 Lipschitz 约束精度要求很高但仍使用简单裁剪。

## R10

实现正确性的关键检查点：

1. Critic 输出不能接 `sigmoid`；
2. Critic 与 generator 损失符号不能写反；
3. critic 要比 generator 频繁更新（`n_critic > 1`）；
4. 必须在每次 critic 更新后执行参数裁剪；
5. 数据和生成输出范围要对齐（本例都在 `[-1, 1]`）。

## R11

数值稳定与训练稳定要点：

1. 使用较小学习率（本例 `5e-5`）；
2. 使用 `RMSprop` 与原版 WGAN 保持一致；
3. 固定随机种子减少波动；
4. 每轮监控 `wasserstein_est = E[D(real)]-E[D(fake)]`；
5. 对生成样本做方差检查，防止隐性模式崩塌。

## R12

本 MVP 的核心超参数：

1. `latent_dim=32`：噪声维度，过小会限制多样性；
2. `n_critic=5`：保证 critic 近似 Wasserstein 对偶更充分；
3. `weight_clip=0.01`：Lipschitz 约束强度；
4. `epochs=45`：在小数据上足够观察收敛趋势；
5. `batch_size=128`：兼顾稳定性与速度。

调参顺序建议：先 `lr`，再 `clip_value`，再 `n_critic`，最后网络宽度。

## R13

理论说明：

1. WGAN 仍是非凸优化，不保证全局最优；
2. 权重裁剪是 Lipschitz 约束的粗糙近似，理论与实践间有偏差；
3. 若 critic 太弱，会低估真实分布与生成分布差异；
4. 若裁剪过严，critic 容量不足，训练可能欠拟合。

## R14

常见失效模式与处理：

1. 失效：`wasserstein_est` 长期接近 0 且图像无结构。  
   处理：适当增大 critic 宽度或放宽 `weight_clip`。
2. 失效：生成样本方差极低（模式崩塌）。  
   处理：降低学习率、增大 `n_critic`、增加训练轮数。
3. 失效：损失出现 NaN/Inf。  
   处理：检查数据归一化范围，降低学习率并排查异常输入。
4. 失效：训练非常慢。  
   处理：减少 epochs 或隐藏层宽度。

## R15

工程实践建议：

1. 先用小数据集验证训练闭环，再迁移到更复杂数据；
2. 记录训练日志并保留分布指标而不只看 loss；
3. 给脚本加最小门禁（finite、方差下限、分布距离上限）；
4. 不要把“损失变小”误解为唯一目标，生成样本统计同样重要。

## R16

与相关模型关系：

1. GAN：最基础对抗框架，常用 BCE，对训练不稳定更敏感；
2. WGAN：把目标替换成 Wasserstein 对偶，训练更稳；
3. WGAN-GP：用梯度惩罚替代裁剪，通常更实用；
4. DCGAN：主要是卷积架构改造，可与 WGAN 目标组合。

本条目是“原始 weight-clipping WGAN”最小实现，不含 GP。

## R17

`demo.py` 的可运行特性：

1. 依赖 `numpy/pandas/scipy/scikit-learn/torch`；
2. 无需下载外部数据，离线即可执行；
3. 自动输出每轮训练摘要、最终统计表和评估指标；
4. 无交互输入，可直接用于自动化验证。

运行方式：

```bash
cd Algorithms/计算机-机器学习／深度学习-0265-WGAN
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程拆解（9 步）：

1. `load_digit_vectors` 读取 `load_digits` 数据，把 `8x8` 图像展平为 64 维并归一化到 `[-1,1]`，再切分训练/评估集。  
2. `Generator` 定义 `z -> x_fake` 的 MLP 映射，输出层用 `tanh` 保证生成向量与真实数据同范围。  
3. `Critic` 定义 `x -> score` 的 MLP 映射，输出单个实数，不做概率化。  
4. `train_wgan` 中每个 batch 先执行 `n_critic` 次 critic 更新：采样噪声生成 `fake`，计算 `mean(D(fake))-mean(D(real))` 并反传。  
5. 每次 critic 更新后调用 `clip_critic_weights`，把所有参数截断到 `[-weight_clip, weight_clip]`，近似满足 1-Lipschitz。  
6. 同一 batch 再执行 1 次 generator 更新：最小化 `-mean(D(G(z)))`，推动假样本得到更高 critic 分数。  
7. 每轮统计 `critic_loss`、`generator_loss` 与 `wasserstein_est = mean(D(real))-mean(D(fake))`，形成 `pandas` 日志表。  
8. 训练后 `generate_vectors` 采样生成向量，`evaluate_distribution` 用 `scipy.stats.wasserstein_distance`、直方图 L1、KL 与 PCA 均值差评估分布拟合。  
9. `run_sanity_checks` 对数值有限性、生成样本方差和 Wasserstein 上限做门禁，最后打印结果。  

该实现把 WGAN 关键机制（critic 线性目标、多步 critic 更新、权重裁剪）逐步展开，没有把核心训练逻辑交给黑盒函数。
