# 扫描电子显微镜 (SEM)

- UID: `PHYS-0453`
- 学科: `物理`
- 分类: `材料科学`
- 源序号: `476`
- 目标目录: `Algorithms/物理-材料科学-0476-扫描电子显微镜_(SEM)`

## R01

问题定义：给定一组 SEM（Scanning Electron Microscopy）灰度图，自动估计材料表面缺陷程度。  
本 MVP 不依赖外部数据集，而是先合成“类 SEM”图像，再完成一条可运行的分析链路：去噪、分割、连通域统计、缺陷等级分类。

## R02

输入与输出：
- 输入：`n_images` 张 `image_size x image_size` 的合成 SEM 图（包含晶粒背景、孔洞、夹杂、划痕、噪声）。
- 中间输出：去噪图、Sobel 边缘强度图、KMeans 缺陷掩膜、连通域统计特征。
- 最终输出：二分类标签（`label=0/1`，低缺陷/高缺陷）及模型精度。

## R03

成像近似模型（简化）：
1. 用随机晶粒中心构造 Voronoi-like 微结构分区。
2. 每个晶粒赋予不同亮度，模拟取向/成分差异带来的背散射强度变化。
3. 对强度场做高斯平滑，近似探针卷积与束斑扩散。

## R04

缺陷与噪声模型：
- 缺陷：
  - 孔洞（暗圆斑）：降低局部灰度；
  - 夹杂（亮斑）：抬高局部灰度；
  - 划痕（细长暗带）：沿线条降低灰度。
- 噪声：
  - 泊松噪声（shot noise）：`Poisson(I * photon_count) / photon_count`；
  - 电子噪声：`N(0, sigma^2)`；
  - 最终裁剪到 `[0, 1]`。

## R05

预处理步骤：
1. `scipy.ndimage.gaussian_filter` 对噪声图去噪。
2. 保留灰度统计量：均值、标准差，为后续分类特征输入。

## R06

边缘提取（PyTorch）：
- 使用 `torch.nn.functional.conv2d` 施加 Sobel `kx/ky` 卷积核；
- 计算梯度幅值 `sqrt(gx^2 + gy^2)`；
- 以 `mean + std` 阈值统计 `edge_density`，描述表面纹理/边界复杂度。

## R07

无监督缺陷分割（scikit-learn）：
1. 先用强平滑估计背景，构造残差图 `residual = image - background`；
2. 对残差像素做 `KMeans(n_clusters=2)` 聚类；
3. 选择簇中心更暗的一类作为“疑似缺陷”；
4. 增加原图灰度门控（较暗像素）抑制误检；
5. 使用 `binary_closing` + `binary_opening` 做形态学清理。

## R08

连通域分析（SciPy）：
- `scipy.ndimage.label` 计算连通域；
- 过滤小区域（`min_region_area`）；
- 统计区域数、面积比、最大面积、平均周长、区域平均强度等。

## R09

特征工程（用于分类）：
- 几何类：`defect_area_ratio`, `region_count`, `max_region_area`, `mean_region_perimeter`
- 纹理类：`edge_density`
- 强度类：`std_intensity`

这些特征兼顾了缺陷规模、分布离散性与纹理粗糙度。

## R10

监督标签构造（合成数据真值）：
- 注入阶段先采样两种工况：`high_defect_mode` / `low_defect_mode`；
- 高缺陷工况使用更多孔洞、更大半径、更多划痕；
- 由该工况直接定义监督标签：高缺陷 `label=1`，低缺陷 `label=0`；
- 同时保留真实统计量（孔洞数、面积比等）用于分析。

## R11

基线模型 1（scikit-learn）：
- 模型：`LogisticRegression(max_iter=500)`；
- 划分：`train_test_split(..., stratify=y)`；
- 指标：Accuracy + Classification Report（precision/recall/f1）。

## R12

基线模型 2（PyTorch）：
- 模型：两层 MLP（`Linear -> ReLU -> Linear`）；
- 损失：`BCEWithLogitsLoss`；
- 优化：`Adam`；
- 目标：验证同一特征集在轻量神经网络上的可学习性。

## R13

时间复杂度（单张图，像素数 `P=H*W`）：
- KMeans（2类，迭代 `T`）：约 `O(P*T)`；
- Sobel 卷积：`O(P)`；
- 连通域标记：`O(P)`；
- 其余特征统计近似 `O(P)`。

总体瓶颈在 KMeans，整条管线仍为可快速运行的轻量 MVP。

## R14

数值稳定与工程处理：
- 全流程强度裁剪到 `[0,1]`；
- 边缘幅值中加入 `1e-12` 防止开根号数值问题；
- 标准化分母加入 `1e-6` 防止除零；
- 固定随机种子，保证复现。

## R15

运行方式（无交互）：

```bash
uv run python Algorithms/物理-材料科学-0476-扫描电子显微镜_(SEM)/demo.py
```

输出包括：
1. 数据集前几行样本；
2. 标签分布；
3. 分标签特征均值；
4. sklearn 与 PyTorch 两个基线模型精度。

## R16

结果解读建议：
- 若 `defect_area_ratio`、`region_count` 在 `label=1` 组显著更高，说明分割+连通域统计有效；
- 若两模型精度都高于随机基线（约 0.5），说明特征可区分缺陷程度；
- 若两模型差距较大，可据此判断线性可分性与非线性需求。

## R17

局限性与可扩展方向：
- 当前图像为合成数据，与真实 SEM 成像物理仍有差距；
- KMeans 仅按灰度分割，遇到复杂对比度/充电影响会退化；
- 后续可替换为：
  - 自适应阈值 + 多尺度滤波；
  - U-Net 类分割网络；
  - 结合 EDS/EBSD 的多模态特征。

## R18

源码级算法流（对应 `demo.py`，非黑箱拆解）：
1. `make_grain_microstructure`：构造晶粒中心，按最近中心生成 Voronoi-like 分区并赋亮度。
2. `inject_defects`：在强度场注入孔洞/夹杂/划痕，同时记录真值统计（面积比、计数、标签）。
3. `add_sem_noise`：叠加泊松 shot noise 与高斯电子噪声，得到观测图。
4. `gaussian_filter` 去噪，生成后续分析使用的平滑图。
5. `sobel_edge_density_torch`：用 PyTorch `conv2d` 计算 Sobel 梯度与边缘密度。
6. `segment_dark_defects_kmeans`：像素灰度聚 2 类，选暗类为缺陷掩膜，再做开闭运算清理。
6. `segment_dark_defects_kmeans`：先求残差暗区再聚类，叠加灰度门控后做开闭运算清理。
7. `component_stats`：连通域标记并提取区域数、面积比、周长等结构化特征。
8. `build_dataset`：对多张图重复 1-7，形成 `pandas.DataFrame` 特征表并生成标签列。
9. `train_sklearn_baseline` 与 `train_torch_baseline`：分别训练逻辑回归和 MLP，输出精度与报告。
