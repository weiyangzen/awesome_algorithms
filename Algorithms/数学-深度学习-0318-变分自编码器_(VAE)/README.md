# 变分自编码器 (VAE)

- UID: `MATH-0318`
- 学科: `数学`
- 分类: `深度学习`
- 源序号: `318`
- 目标目录: `Algorithms/数学-深度学习-0318-变分自编码器_(VAE)`

## R01

变分自编码器（Variational AutoEncoder, VAE）是一类概率生成模型。它把输入样本映射到一个连续潜变量分布（通常是高斯分布），再从潜变量重建样本。与普通自编码器只学习确定性编码不同，VAE通过概率建模获得可采样、可插值的潜空间，适合做生成、表征学习和异常检测。

## R02

核心思想是最大化观测数据对数似然 `log p_theta(x)` 的下界（ELBO）：

- 编码器 `q_phi(z|x)` 近似后验分布；
- 解码器 `p_theta(x|z)` 生成数据；
- 训练目标同时包含重建质量和潜空间正则化（使 `q_phi(z|x)` 靠近先验 `p(z)=N(0,I)`）。

## R03

概率图模型定义：

- 先验：`p(z)=N(0,I)`
- 似然：`p_theta(x|z)`
- 近似后验：`q_phi(z|x)=N(mu_phi(x), diag(sigma_phi^2(x)))`

重参数化技巧：

`z = mu + sigma * epsilon, epsilon ~ N(0, I)`

这样可把随机性移到 `epsilon`，从而对 `mu/sigma` 进行反向传播。

## R04

ELBO 推导要点：

`log p_theta(x) = ELBO(x) + KL(q_phi(z|x) || p_theta(z|x))`

因为 KL 非负，所以：

`ELBO(x) = E_{q_phi(z|x)}[log p_theta(x|z)] - KL(q_phi(z|x) || p(z)) <= log p_theta(x)`

训练时最小化负 ELBO，即：

`L = ReconLoss + beta * KL`

本项目默认 `beta=1.0`（标准 VAE）。

## R05

在 `demo.py` 中采用：

- 输入：`sklearn digits` 的 `8x8` 灰度图（展平为 64 维，归一化到 `[0,1]`）
- 重建项：`binary_cross_entropy(reduction="sum")`
- KL 项（高斯闭式）：
  `-0.5 * sum(1 + logvar - mu^2 - exp(logvar))`

每轮统计 `loss/recon/kl` 的每样本均值，并输出训练与测试指标。

## R06

MVP 网络结构（全连接）：

- Encoder: `64 -> 64 -> 32`（ReLU）
- 参数头：`mu_head: 32 -> 8`，`logvar_head: 32 -> 8`
- Decoder: `8 -> 32 -> 64 -> 64`（最后 Sigmoid）

该结构很小，CPU 下数秒可完成训练，便于验证算法流程而非追求 SOTA 性能。

## R07

伪代码：

```text
for epoch in 1..E:
  for x in train_loader:
    mu, logvar = Encoder(x)
    eps ~ N(0, I)
    z = mu + exp(0.5*logvar) * eps
    x_hat = Decoder(z)
    recon = BCE(x_hat, x)
    kl = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    loss = recon + beta * kl
    backprop(loss)
```

## R08

复杂度（单样本，单次前向）近似由全连接层主导：

- Encoder: `O(64*64 + 64*32 + 32*8*2)`
- Decoder: `O(8*32 + 32*64 + 64*64)`

总复杂度约 `O(d*h)` 级别（这里 d=64，h<=64），内存开销主要来自参数和 batch 激活。对本任务数据规模（1797 样本）非常轻量。

## R09

数值稳定性与工程细节：

- 使用 `logvar` 而非直接输出方差，避免负方差问题；
- 解码末层用 `Sigmoid`，与 BCE 的输入域一致；
- 固定随机种子（Python/NumPy/PyTorch）提升复现实验一致性；
- 以每样本均值输出指标，便于跨 batch size 比较。

## R10

与相关模型对比：

- AE：仅学习确定性 `z`，潜空间缺少概率约束，采样生成质量通常较差；
- VAE：有显式概率先验，采样稳定、插值自然，但常见重建偏模糊；
- GAN：生成锐利但训练更不稳定，且没有天然编码器（除变体外）。

本任务聚焦“可解释、可训练、可复现”的概率生成基线，因此选 VAE。

## R11

关键超参数建议：

- `latent_dim`: 4~16（过小欠表达，过大 KL 约束变弱）
- `beta`: 0.5~4（越大越强调潜空间规整，可能牺牲重建）
- `lr`: `1e-3` 起步，Adam 通常稳定
- `epochs`: 10~50（取决于数据规模和网络容量）

当前默认：`latent_dim=8, beta=1.0, lr=1e-3, epochs=12`。

## R12

代码实现说明（`demo.py`）：

- 数据：`load_digits()` + `train_test_split(stratify=y)`
- 框架：PyTorch 定义 `VAE`、损失函数和训练循环
- 评估：
  - 测试集 ELBO 分量（recon / kl）
  - 重建 MSE
  - 潜变量均值与 `N(0,1)` 的 Wasserstein 距离（SciPy）
- 日志：用 pandas 输出训练曲线末 3 轮，便于快速检查收敛趋势。

## R13

运行方式：

```bash
uv run python demo.py
```

期望输出包括：

- 训练配置（epoch、batch、latent_dim、beta）
- 最终 `train_loss/test_loss/test_recon/test_kl`
- `recon_mse`、`latent_wasserstein_to_N01`
- 训练曲线 tail（最后 3 轮）

## R14

结果解读建议：

- `test_recon` 下降：重建质量提升；
- `test_kl` 不应始终接近 0（否则可能后验塌缩）；
- `latent_wasserstein_to_N01` 越小表示潜变量更接近先验；
- `recon_mse` 与 `test_recon` 一起看，可粗判“生成规整性 vs 重建精度”的平衡。

## R15

常见问题排查：

- KL 很快接近 0：可能后验塌缩，可降低解码器能力或做 KL warmup；
- 重建很差：提高 hidden_dim/epochs，或降低 beta；
- 训练不稳定：调小学习率（如 `5e-4`）；
- 生成样本过于平均：潜变量维度过小或正则过强。

## R16

可扩展方向：

- 卷积 VAE（处理更大图像）
- `beta-VAE`（更强可解释潜因素）
- 条件 VAE（`p(x|z,y)`，可控生成）
- 与 flow 结合改进后验近似
- 在异常检测中使用重建误差 + 潜变量密度联合打分

## R17

边界与适用性：

- 该 MVP 适合教学和算法核验，不是工业级生成质量方案；
- 数据集较小（digits），结论主要用于流程验证；
- 概率生成模型可用于合成数据，但不应替代真实高风险决策数据；
- 若迁移到人脸/医疗等场景，需增加隐私、偏差和合规审查。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `build_dataloaders()` 调用 `load_digits()`，将像素缩放到 `[0,1]`，并通过 `train_test_split(..., stratify=y)` 划分训练/测试集。  
2. `VAE.__init__` 构造编码器、`mu/logvar` 双头和解码器，形成 `x -> (mu,logvar) -> z -> x_hat` 的可微路径。  
3. `forward()` 中先 `encode()` 得到 `mu/logvar`，再在 `reparameterize()` 用 `z = mu + exp(0.5*logvar)*eps` 把采样写成可反传形式。  
4. `vae_loss()` 计算两部分：`BCE(recon,x)` 与 KL 闭式项，并组合为 `total = recon + beta*kl`。  
5. `run_epoch(..., optimizer!=None)` 执行训练分支：`zero_grad -> forward -> loss.backward -> optimizer.step`，同时累计每样本指标。  
6. `run_epoch(..., optimizer=None)` 在评估分支仅前向统计测试损失，不更新参数。  
7. 训练结束后，`evaluate_latent_regularization()` 对测试集 `mu` 与标准正态样本逐维计算 `wasserstein_distance`，检查潜空间规整程度。  
8. `main()` 汇总并打印最终指标、重建误差、随机解码统计与训练曲线 tail，形成可复现、可验证的最小闭环。  
