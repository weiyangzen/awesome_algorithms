# 缺失横能量 (Missing Transverse Energy)

- UID: `PHYS-0405`
- 学科: `物理`
- 分类: `粒子物理`
- 源序号: `425`
- 目标目录: `Algorithms/物理-粒子物理-0425-缺失横能量_(Missing_Transverse_Energy)`

## R01

缺失横能量（MET, Missing Transverse Energy）用于估计事件中“未被探测到”的横向动量，典型来源是中微子或新物理暗粒子。算法核心是对所有可见重建对象的横向动量做矢量和，再取反：
`vec(MET) = - sum(vec(pT_visible))`。

## R02

问题定义：给定一个事件的可见对象集合 `O={o_i}`，每个对象有 `(px_i, py_i)`，求
- `MET_x = -sum_i px_i`
- `MET_y = -sum_i py_i`
- `MET = sqrt(MET_x^2 + MET_y^2)`
- `phi_MET = atan2(MET_y, MET_x)`

并将重建结果与真值不可见动量（本 demo 中为中微子）进行对比。

## R03

粒子物理背景：
- 在强子对撞机中，入射束流沿 `z` 方向，初态横向总动量近似为 0。
- 若末态只含可见粒子，事件应满足横向动量平衡。
- 若出现中微子等不可见粒子，则可见动量和不再闭合，残差即表现为 MET。
- 因此 MET 是 `W->lv`、`top`、SUSY 等分析中的关键观测量。

## R04

`demo.py` 输入输出约定：
- 输入：脚本内部合成的玩具事件（无命令行参数、无交互）。
- 中间数据：
  - `TruthParticle`：真值粒子（含可见/不可见标记）。
  - `RecoObject`：施加探测器效应后的可见重建对象。
- 输出：
  - 重建对象表（`pt/eta/phi/px/py`）。
  - 三组 MET：真值不可见和、真值可见闭合、重建可见对象计算值。
  - 与真值偏差 `Delta|MET|` 与 `DeltaPhi`。

## R05

核心公式：

1. 分量定义
`MET_x = - sum_i px_i`, `MET_y = - sum_i py_i`

2. 模长与方位角
`MET = sqrt(MET_x^2 + MET_y^2)`
`phi_MET = atan2(MET_y, MET_x)`

3. 角差（用于重建-真值比较）
`DeltaPhi = wrap(phi_reco - phi_truth)`，其中 `wrap` 映射到 `[-pi, pi)`。

## R06

MVP 算法流程：
1. 生成一个含单中微子的真值事件，并构造可见反冲（轻子 + 喷注 + 软项）。
2. 通过动量闭合使真值层面满足 `pT_visible + pT_invisible ~= 0`。
3. 对可见真值粒子施加简化探测器接收范围与分辨率，得到重建对象。
4. 由重建对象计算 `MET_x/MET_y/MET/phi`。
5. 额外计算真值不可见动量和与真值可见闭合 MET。
6. 打印三者差异，验证实现链路是否合理。

## R07

复杂度分析：
- 设重建对象数为 `N`。
- MET 计算只需一次线性扫描：时间复杂度 `O(N)`。
- 额外空间主要是对象列表与少量标量：空间复杂度 `O(N)`。
- 事件生成和表格打印也是线性级别，整体仍为 `O(N)`。

## R08

正确性直觉：
- MET 本质是横向动量守恒在“只看可见粒子”条件下的残差。
- 若可见对象完美测量且无遗漏，`-sum(pT_visible)` 应逼近不可见粒子总 `pT`。
- 加入分辨率、阈值和接收范围后，重建 MET 会偏离真值，这正是实验中的真实特征。

## R09

伪代码：

```text
truth_particles <- generate_truth_event(seed)
reco_objects <- reconstruct_visible_objects(truth_particles, seed2)

met_reco <- MET(reco_objects)
met_truth_visible <- MET(all truth visible particles)
met_truth_invisible <- vector_sum(all truth invisible particles)

print reco object table
print met_truth_invisible, met_truth_visible, met_reco
print delta magnitude and delta phi
```

## R10

边界条件与数值处理：
- 空对象集合：`sum([])=0`，返回 `MET=0`。
- `atan2(0,0)`：Python 返回 `0.0`，可稳定处理零 MET 事件。
- `phi` 周期边界：统一通过 `wrap_phi` 和 `delta_phi` 处理 `-pi/pi` 跳变。
- 分辨率抽样可能给负尺度：代码中对缩放因子做 `max(scale, 0)` 保护。

## R11

默认参数（`demo.py`）：
- 事件随机种子：`truth=425`, `reco=426`（结果可复现）。
- 生成中微子 `pt` 范围：`35~110 GeV`。
- 重建分辨率（相对）：`muon 1%`, `jet 10%`, `soft 25%`。
- 接收范围：`|eta|<2.7`（muon），`4.5`（jet），`4.9`（soft）。
- 最低重建阈值：`3/10/0.5 GeV`（muon/jet/soft）。

## R12

当前实现覆盖范围：
- 覆盖 MET 的核心数学定义和事件级计算。
- 覆盖“真值 -> 重建 -> 指标对比”的最小闭环。
- 不依赖 ROOT/HEP 专用框架，便于教学和快速验证。
- 代码仅使用 `numpy`，最小依赖、可直接运行。

## R13

运行方式：

```bash
uv run python Algorithms/物理-粒子物理-0425-缺失横能量_(Missing_Transverse_Energy)/demo.py
```

或进入该目录后执行：

```bash
uv run python demo.py
```

## R14

示例输出结构（数值会随随机种子变化）：

```text
=== Missing Transverse Energy (MET) Demo ===
Truth particles: total=10, visible=9, invisible=1
Reconstructed visible objects: 6

name           kind         pt      eta      phi         px         py
...

--- MET Summary (GeV) ---
Truth invisible pT sum: MET=..., phi=..., (mx,my)=(..., ...)
Truth from visible closure: MET=..., phi=..., (mx,my)=(..., ...)
Reco MET from visible objects: MET=..., phi=..., (mx,my)=(..., ...)
Reco - Truth(|MET|): ... GeV
|Delta phi(reco, truth-invisible)|: ... rad
```

## R15

最小验收清单：
- `README.md` 与 `demo.py` 均已完成填充且无占位符残留。
- `uv run python demo.py` 可无交互运行结束。
- 输出包含重建对象表与三组 MET 对比。
- 输出中包含 `Reco - Truth(|MET|)` 和 `Delta phi`。

## R16

MVP 局限：
- 事件是玩具模型，不含真实 hard-process、parton shower、pileup。
- 探测器响应为参数化高斯近似，未做分区标定与非高斯尾处理。
- 未计算 MET significance、JES/JER 系统学、软项校正等实验级内容。

## R17

可扩展方向：
- 接入真实事件格式（ROOT/Parquet）并按对象类型读取重建量。
- 增加 Type-I MET 校正（喷注能标修正传播到 MET）。
- 评估系统不确定度并输出 covariance 或 significance。
- 加入多中微子拓扑与 pileup 抑制策略（PUPPI/CHS 风格简化实现）。

## R18

源码级流程拆解（对应 `demo.py`）：
1. `generate_truth_event` 采样中微子 `(pt, phi)`，并构造可见轻子/喷注/软项使横向动量近似闭合。
2. `_split_vector_into_soft_terms` 把目标反冲矢量拆成多个对象分量，最后一项强制闭合。
3. `reconstruct_visible_objects` 对可见真值对象施加 `eta` 接收、`pt` 阈值、能量和角度分辨率，生成 `RecoObject`。
4. `compute_met_from_visible` 线性遍历重建对象，计算 `MET_x=-sum(px)`、`MET_y=-sum(py)`。
5. 同函数内继续计算 `MET=sqrt(MET_x^2+MET_y^2)` 与 `phi=atan2(MET_y,MET_x)`。
6. `compute_met_from_truth_visible` 在真值可见对象上复用同一计算逻辑，得到“理想可见闭合”参考。
7. `compute_truth_invisible_vector` 对不可见真值粒子求和，得到“真值 MET 目标向量”。
8. `main` 汇总并打印三组结果及 `Delta|MET|`、`DeltaPhi`，完成端到端验证。
