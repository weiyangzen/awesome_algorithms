# StyleGAN

- UID: `CS-0128`
- 学科: `计算机`
- 分类: `机器学习/深度学习`
- 源序号: `266`
- 目标目录: `Algorithms/计算机-机器学习／深度学习-0266-StyleGAN`

## R01

StyleGAN（Style-based GAN）是生成对抗网络的一类结构化改进，其核心目标不是仅“生成看起来像真的图像”，而是把潜变量拆成更可控的“风格”信号，使不同层分别控制从全局到局部的视觉属性。

本条目给出一个可运行 MVP，保留 StyleGAN 的关键机制：
- `z -> w` 映射网络（Mapping Network）；
- 每层风格调制（AdaIN）；
- 每层随机噪声注入；
- 风格混合（Style Mixing）；
- 截断技巧（Truncation Trick）。

为保证脚本离线可运行，`demo.py` 在合成的 `16x16` 灰度“真实分布”上训练，不依赖外部数据下载。

## R02

GAN 训练目标可写为：

`min_G max_D E_{x~p_data}[log D(x)] + E_{z~p(z)}[log(1 - D(G(z)))]`

在脚本中采用非饱和 logistic 形式（softplus 实现）：
- 判别器损失：`L_D = E[softplus(D(G(z)))] + E[softplus(-D(x))]`
- 生成器损失：`L_G = E[softplus(-D(G(z)))]`

StyleGAN 的关键是把 `G(z)` 改写为多层风格控制：
`w = f_map(z)`，然后每个生成层使用 `w` 进行调制，不同层可用不同 `w`（风格混合）。

## R03

StyleGAN 的核心思想（对应本 MVP）：

1. 潜变量解耦：`z` 先进入 Mapping Network 得到 `w`，让生成空间更线性、可编辑。
2. 分层风格控制：每个卷积层通过 AdaIN 的 `gamma/beta` 注入风格，控制特征统计。
3. 随机细节通道：每层加入噪声，增强微观纹理随机性。
4. 风格混合正则：同一张图在不同层使用 `w1/w2`，防止层间耦合过强。
5. 截断采样：用 `w_avg + psi*(w-w_avg)` 折中多样性与稳定性。

## R04

`demo.py` 的网络结构如下：

- `MappingNetwork(latent_dim=64, w_dim=64)`：
  - `PixelNorm -> Linear -> LeakyReLU` 共 3 层线性映射。
- `Generator`：
  - 可学习常量输入 `const(1,64,4,4)`；
  - 6 个 `StyledConv` 层（含两次上采样：`4->8->16`）；
  - `StyledConv` 内部执行：`Conv -> Noise -> LeakyReLU -> InstanceNorm -> AdaIN`；
  - `to_rgb(1x1 conv)` + `tanh` 输出 `1x16x16`。
- `Discriminator`：
  - 多层卷积 + `AvgPool` 下采样（`16->8->4`）+ 全连接输出真假 logits。

## R05

端到端流程：

1. 固定随机种子；
2. 构建合成“真实图像分布”（高斯 blob + 纹理）；
3. 初始化 `mapping/generator/discriminator` 与 Adam；
4. D 步：采样 `z`，生成假图（detach），更新判别器；
5. G 步：重新采样 `z`，端到端更新 `mapping + generator`；
6. 更新 `w_avg`（EMA）；
7. 每个 epoch 用 truncation 采样并打印统计；
8. 训练结束后做数值健壮性检查并输出样本摘要。

## R06

正确性依据：

- 对抗损失形式与 GAN 标准目标一致；
- `mapping -> AdaIN` 显式实现 StyleGAN 的风格控制路径；
- 每层独立噪声注入对应 StyleGAN 的随机细节来源；
- 风格混合由 `build_mixed_ws` 在层级上切换 `w1/w2`；
- 截断采样由 `sample_with_truncation` 显式实现；
- 最终断言覆盖 `loss finite`、`生成方差`、`均值偏差`，避免“能跑但退化”。

## R07

复杂度（单步）可近似分为：

- 生成器：`O(sum_l B * H_l * W_l * C_in_l * C_out_l * k^2)`；
- 判别器：同级卷积复杂度；
- 映射网络：`O(B * w_dim^2 * L_map)`，相对卷积成本较小。

本 MVP 采用 `16x16`、小通道数与小批量，CPU 下可在几十秒完成训练。

## R08

边界与健壮性处理：

- `Generator.forward` 检查风格层数量必须等于 `num_layers=6`；
- 数据归一化后保证像素在 `[-1,1]`；
- 训练后检查 `d_loss/g_loss` 为有限值；
- `gen_std` 过小触发“模式坍塌”告警；
- `gen_mean` 与真实分布均值偏差过大触发异常。

## R09

MVP 范围与取舍：

- 保留：映射网络、AdaIN、噪声、style mixing、truncation；
- 省略：渐进式分辨率增长、路径长度正则、R1/GP、更复杂判别器技巧；
- 目的：用最小可读代码演示 StyleGAN 机制闭环，而非复刻论文完整训练体系。

## R10

`demo.py` 主要模块职责：

- `TrainConfig`：集中管理实验超参数；
- `make_synthetic_real_images`：构造离线可复现实验数据；
- `MappingNetwork`：`z -> w`；
- `StyledConv`：卷积层中的风格调制与噪声注入；
- `Generator`：常量输入 + 分层风格生成；
- `Discriminator`：真假判别；
- `build_mixed_ws`：风格混合；
- `sample_with_truncation`：截断采样；
- `train_stylegan_mvp`：完整训练与检查流程；
- `main`：脚本入口。

## R11

运行方式：

```bash
cd Algorithms/计算机-机器学习／深度学习-0266-StyleGAN
uv run python demo.py
```

脚本无需交互输入。

## R12

输出字段说明：

- `device`：运行设备；
- `real dataset shape`：真实样本张量形状；
- `model params`：各子网络参数量；
- `epoch ... d_loss g_loss sample(...)`：训练损失与截断采样统计；
- `final stats`：真实/生成分布统计对比；
- `sample summary`：若干生成样本的均值/标准差/最大值；
- `All checks passed.`：训练与质量检查通过。

## R13

本条目默认实验配置：

1. 图像尺寸：`16x16` 灰度；
2. 数据规模：`1536` 个合成样本；
3. 批大小：`64`；
4. 训练轮数：`10`；
5. 优化器：`Adam(lr=1e-3, betas=(0.0, 0.99))`；
6. 风格混合概率：`0.8`；
7. 截断系数：`psi=0.7`。

一次实测可得到 `All checks passed.`。

## R14

关键超参数影响：

- `mixing_prob`：越高越强解耦，但过高可能降低收敛速度；
- `truncation_psi`：越小越稳定但多样性下降；
- `w_dim`：增大可提升表达能力，但会增加映射与调制开销；
- `epochs`：过小欠拟合，过大可能出现对抗振荡；
- `dataset_size` 与真实分布复杂度直接决定学习难度。

## R15

与常见 GAN 变体对比：

- 相对 DCGAN：StyleGAN 把潜变量控制显式分层，编辑性更好；
- 相对“直接 z 输入卷积栈”：StyleGAN 通过 `w` 空间通常更平滑；
- 相对 StyleGAN2/3：本实现更简化，未包含后续版本的抗伪影与正则改进。

## R16

典型应用场景：

- 人脸/角色/材质等可控生成任务原型；
- 研究潜空间插值与属性编辑；
- 教学中演示“生成器结构设计如何影响可控性”；
- 作为更复杂生成系统（条件控制、扩散先验）前置模块。

## R17

可扩展方向：

1. 加入 R1 正则或路径长度正则，提升稳定性；
2. 扩展到 RGB 高分辨率并引入更强判别器；
3. 增加条件标签输入形成 Conditional StyleGAN；
4. 接入 FID/KID 指标实现自动评估；
5. 将当前 AdaIN 版本替换为 StyleGAN2 的调制卷积实现。

## R18

`demo.py` 源码级算法流程（9 步）：

1. `main` 创建 `TrainConfig`，进入 `train_stylegan_mvp`。
2. `set_global_seed` 固定随机性；`make_synthetic_real_images` 生成 `[-1,1]` 的真实样本分布并装入 `DataLoader`。
3. 初始化 `MappingNetwork`、`Generator`、`Discriminator`，并创建 `g_opt/d_opt`。
4. 判别器更新阶段：
   - 调用 `build_mixed_ws` 由 `z1/z2` 产生分层风格；
   - `generator(ws)` 产假样本（`no_grad`）；
   - 计算 `softplus(fake) + softplus(-real)` 并反向更新 D。
5. 生成器更新阶段：
   - 重新采样 `z1/z2` 与 `ws`；
   - `generator(ws) -> discriminator(fake)`；
   - 计算 `softplus(-fake_logits)` 并反向更新 `mapping + generator`。
6. 使用当前 batch 的 `w_primary.mean` 对 `w_avg` 做指数滑动平均，供截断采样使用。
7. 每个 epoch 后调用 `sample_with_truncation`，执行 `w_trunc = w_avg + psi*(w-w_avg)` 并打印样本统计。
8. 训练结束后对真实/生成分布统计做对比，并执行健壮性断言（有限值、方差阈值、均值偏差阈值）。
9. 输出样本摘要与 `All checks passed.`，表示 MVP 训练流程闭环完成。
