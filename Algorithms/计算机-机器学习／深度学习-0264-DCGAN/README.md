# DCGAN

- UID: `CS-0126`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `264`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0264-DCGAN`

## R01

DCGAN（Deep Convolutional GAN）是在 GAN 框架上引入卷积结构的生成模型，核心思想是：

- 生成器 `G` 用反卷积（`ConvTranspose2d`）把噪声 `z` 逐步上采样成图像；
- 判别器 `D` 用卷积（`Conv2d`）把图像逐步下采样并判断真伪；
- 两者通过对抗训练共同逼近真实数据分布。

与全连接 GAN 相比，DCGAN 更适合图像任务，因为卷积更能表达局部纹理和空间结构。

## R02

标准 GAN 极小极大目标：

`min_G max_D E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]`

本目录实现采用常见的非饱和训练写法（BCE logits 形式）：

- 判别器：`L_D = BCE(D(x_real), 1) + BCE(D(G(z).detach()), 0)`
- 生成器：`L_G = BCE(D(G(z)), 1)`

其中 `BCEWithLogitsLoss` 直接作用于 logits，数值更稳定。

## R03

本 MVP 的输入与输出如下。

输入：

- 真实数据：`sklearn.datasets.load_digits()` 的手写数字（8x8 灰度图），经 `scipy.ndimage.zoom` 上采样到 32x32；
- 噪声先验：`z ~ N(0, I)`，维度 `latent_dim=64`；
- 网络结构：轻量 DCGAN（灰度单通道版本）。

输出：

- 训练后的生成器与判别器参数；
- 生成图像张量（`N x 1 x 32 x 32`）；
- 损失历史与分布评估指标（Wasserstein、像素直方图 L1、PCA 空间统计差异等）。

## R04

网络结构设计（与经典 DCGAN 思路一致）：

- `Generator`
  - `z(64x1x1)`
  - `ConvTranspose2d`: `64 -> 128`, 输出 4x4
  - `ConvTranspose2d`: `128 -> 64`, 输出 8x8
  - `ConvTranspose2d`: `64 -> 32`, 输出 16x16
  - `ConvTranspose2d`: `32 -> 1`, 输出 32x32
  - 末层 `Tanh`，输出范围 `[-1, 1]`
- `Discriminator`
  - `Conv2d`: `1 -> 32`, 32x32 -> 16x16
  - `Conv2d`: `32 -> 64`, 16x16 -> 8x8
  - `Conv2d`: `64 -> 128`, 8x8 -> 4x4
  - `Conv2d`: `128 -> 1`, 4x4 -> 1x1 logit

并使用 DCGAN 常见初始化：卷积权重 `N(0, 0.02)`，BN 权重 `N(1, 0.02)`。

## R05

训练策略：

- 每个 mini-batch 交替更新一次 `D`、一次 `G`；
- 判别器真实标签使用 `0.9`（one-sided label smoothing）；
- 优化器使用 Adam，`beta1=0.5`（DCGAN 常用配置）；
- 设备固定 CPU，保证在普通环境可直接复现。

这种策略重点是“稳定可跑通”，而非追求图像任务 SOTA 质量。

## R06

数据预处理流程：

1. 读取 digits 8x8 灰度图，缩放到 `[0,1]`；
2. 用 `scipy.ndimage.zoom(..., order=1)` 上采样到 32x32；
3. 划分训练集与评估集（`train_test_split`）；
4. 归一化到 `[-1, 1]` 以匹配生成器 `Tanh` 输出范围；
5. 训练时打包为 `DataLoader`，按 batch 随机打乱。

## R07

算法主流程：

1. 设定随机种子、超参数并加载真实图像；
2. 初始化 `Generator/Discriminator` 与权重；
3. 进入 epoch 循环；
4. 判别器阶段：用真实图和伪图计算 `L_D` 并更新 `D`；
5. 生成器阶段：重新采样噪声计算 `L_G` 并更新 `G`；
6. 记录每轮 `d_loss/g_loss` 与判别器真假平均概率；
7. 训练结束后由 `G` 采样生成评估图像；
8. 计算分布统计与质量指标，执行 sanity checks。

## R08

复杂度（粗略）：

- 设训练样本数 `N`、epoch 数 `E`、单次前向/反向卷积开销为 `C`；
- 训练复杂度约为 `O(E * N * C)`；
- 采样复杂度约为 `O(M * C_G)`（`M` 为生成样本数）；
- 主要内存开销来自 batch 激活图与模型参数，MVP 网络较小，CPU 环境可承受。

## R09

DCGAN 常见风险与本实现缓解点：

- 风险：判别器过强，生成器梯度不足。
  - 缓解：真实标签平滑 `0.9`。
- 风险：训练震荡。
  - 缓解：Adam `beta1=0.5`、较小学习率 `2e-4`。
- 风险：模式崩塌。
  - 缓解：监控生成图像方差、像素 Wasserstein 距离并设 sanity 阈值。

## R10

默认超参数（见 `DCGANConfig`）：

- `latent_dim=64`
- `g_channels=32`, `d_channels=32`
- `batch_size=128`
- `epochs=20`
- `lr_g=lr_d=2e-4`
- `beta1=0.5`
- `real_label_smooth=0.9`
- `max_train_samples=1500`
- `eval_samples=256`
- `device="cpu"`

这些参数以“可运行 + 可验证 + 速度适中”为导向。

## R11

评估指标设计：

- `pixel_wasserstein_1`：真实/生成像素值分布的一阶 Wasserstein 距离；
- `pixel_hist_l1`：像素直方图的平均 L1 差；
- `pixel_kl_real_to_gen`：像素分布 KL（`real -> generated`）；
- `pca_mean_l2`、`pca_var_l1`：在真实图像 PCA 子空间中的均值/方差偏移；
- `Distribution Summary`：均值、方差、图像亮度分位数、边缘能量。

指标目标是给出“是否学到真实统计结构”的可解释信号，而不仅盯一个 loss。

## R12

实现细节注意点：

- 判别器更新时必须对伪图 `detach()`，避免误更新生成器；
- 生成器和真实图像需在同一归一化区间（这里是 `[-1,1]`）；
- `Discriminator` 输出 logits，不应再手动套 `sigmoid` 进 BCE；
- 训练过程不追求 loss 单调下降，对抗模型更关注动态平衡和样本统计。

## R13

`demo.py` 的 MVP 特性：

- 依赖仅使用 `numpy/scipy/pandas/scikit-learn/torch`；
- 无需下载外部数据（digits 为本地可用内置数据集）；
- 无交互输入，`uv run python demo.py` 一次运行完成；
- 自动输出训练日志、统计表与指标，并包含基础失败门禁。

## R14

运行方式：

```bash
cd Algorithms/计算机-机器学习／深度学习-0264-DCGAN
uv run python demo.py
```

或在仓库根目录运行：

```bash
uv run python Algorithms/计算机-机器学习／深度学习-0264-DCGAN/demo.py
```

## R15

结果解读建议：

- `d_real_prob` 偏高、`d_fake_prob` 偏低是正常现象，但不应长期极端饱和；
- `pixel_wasserstein_1` 越小越好，表示像素分布更接近；
- `pixel_std`、`edge_energy` 若显著过低，通常意味着生成样本过平滑或塌缩；
- `pca_mean_l2`、`pca_var_l1` 可辅助判断生成图像是否偏离真实结构子空间。

## R16

验收标准（本目录）：

- `README.md` 与 `demo.py` 不含未填充占位符；
- `uv run python demo.py` 可直接运行完成；
- 输出包含训练尾部日志、分布统计和指标；
- sanity checks 未触发异常（有限值、非零方差、距离不过大）。

## R17

可扩展方向：

- 引入条件信息（cDCGAN）做按类别可控生成；
- 引入谱归一化或 WGAN-GP 提升稳定性；
- 将数据集替换为 FashionMNIST/CIFAR10（可用时）；
- 增加 FID/IS 等更贴近视觉质量的指标；
- 从灰度 32x32 扩展到多通道高分辨率生成。

## R18

`demo.py` 的源码级算法流可拆为 9 步：

1. `main()` 创建 `DCGANConfig` 并调用 `set_seed` 固定随机性。
2. `load_real_images` 读取 `load_digits()`，经 `ndimage.zoom` 把 8x8 变为 32x32，并归一化到 `[-1,1]`。
3. `build_train_loader` 将训练图像打包成 `DataLoader`，为对抗训练提供 mini-batch。
4. 初始化 `Generator` 与 `Discriminator`，再通过 `init_dcgan_weights` 应用 DCGAN 风格权重分布。
5. `train_dcgan` 中每个 batch 先执行判别器更新：
   - 从 `sample_latent` 采样噪声生成伪图；
   - 用 `BCEWithLogitsLoss` 计算真图/伪图损失并更新 `D`。
6. 同一 batch 再执行生成器更新：
   - 重新采样噪声并前向 `G`；
   - 以“伪图判为真”为目标计算 `g_loss` 并更新 `G`。
7. 每轮聚合 `d_loss/g_loss/d_real_prob/d_fake_prob`，写入 `pandas.DataFrame` 作为训练历史。
8. 训练完成后 `generate_images` 采样生成图像，`denormalize_to_unit` 将范围转回 `[0,1]`。
9. `evaluate_distribution` 计算 Wasserstein、直方图 L1、KL、PCA 差异；`run_sanity_checks` 做质量门禁并最终打印结果。

该流程完整展示了 DCGAN 的数据管线、卷积式生成/判别结构与交替优化细节，没有把核心训练步骤委托给黑盒 API。
