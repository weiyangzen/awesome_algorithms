# 量能器 (Calorimeters)

- UID: `PHYS-0409`
- 学科: `物理`
- 分类: `粒子物理实验`
- 源序号: `429`
- 目标目录: `Algorithms/物理-粒子物理实验-0429-量能器_(Calorimeters)`

## R01

量能器（Calorimeter）的核心任务是把“粒子在探测器材料中的能量沉积”转换为“可标定的能量重建值”。  
本目录给出一个最小可运行 MVP，针对粒子物理实验中的常见 `ECAL + HCAL` 双系统场景，完成：
- 事件级响应模拟（电子与带电介子的不同簇射特征）
- 传统基线重建（采样分数反演）
- 数据驱动线性校准（可解释、非黑箱）
- 分箱分辨率评估（能量区间内的重建稳定性）

## R02

算法目标形式化为：
- 输入：每个事件的层级读出信号  
  `x = [ecal_l1, ecal_l2, hcal_l1, hcal_l2, ecal_signal_ratio, longitudinal_ratio, particle_type]`
- 输出：重建能量 `E_reco`

在 `demo.py` 中同时输出两种重建：
- 基线重建：`E_base = ecal_total/sf_ecal + hcal_total/sf_hcal`
- 校准重建：`E_cal = f_theta(x)`，其中 `f_theta` 为 `StandardScaler + Ridge` 线性模型

## R03

实验背景（粒子物理）：
- `ECAL` 对电磁簇射（如电子）响应强，`HCAL` 对强子簇射（如 pion）更关键
- 采样量能器只能测到沉积能量的一部分（采样分数效应）
- 真实系统有泄漏、非补偿响应、噪声和纵向形状差异

因此仅靠一个固定比例的能量反演通常有系统偏差，需引入校准步骤提高线性和分辨率。

## R04

时间复杂度（设测试样本数为 `N`，特征维度为 `d=7`）：
- 数据生成：`O(N)`
- 基线重建：`O(N)`
- 线性模型训练（Ridge，`d` 很小）：约 `O(N*d^2 + d^3)`，在本任务可近似视为 `O(N)`
- 评估与分箱统计：`O(N)`

整体复杂度近似线性于样本规模。

## R05

空间复杂度：
- 数据表保存 `N` 条事件和常数个字段，为 `O(N)`
- 模型参数规模 `O(d)`，可忽略
- 中间向量（预测值、残差）为 `O(N)`

总空间复杂度：`O(N)`。

## R06

示例计算链路（单事件）：
1. 抽样真实能量 `E_true` 与粒子类型（electron/pion）。
2. 根据粒子类型抽样 `(em_fraction, had_fraction, leakage_fraction)`。
3. 将 `ECAL/HCAL` 沉积分配到前后层（`l1/l2`），再乘采样分数得到读出均值。
4. 叠加随机噪声得到 `ecal_l1, ecal_l2, hcal_l1, hcal_l2`。
5. 由读出构造基线重建 `E_base`。
6. 用已训练校准模型输出 `E_cal`。
7. 与 `E_true` 对比，形成偏差与分辨率统计。

## R07

算法意义：
- 物理意义：把“簇射物理 + 探测器响应 + 读出噪声”串成可执行管线
- 工程意义：提供可扩展的能量标定 baseline（后续可接真实校准常数）
- 教学意义：展示从解析模型到数据驱动校准的过渡，而不是直接调用黑盒网络

## R08

直接依赖组件：
- `numpy`：数值计算与随机采样
- `pandas`：事件表管理与结果展示
- `scipy.stats.binned_statistic`：能量分箱统计
- `scikit-learn`：`StandardScaler + Ridge` 线性校准

本 MVP 未依赖深度学习框架，保持最小可解释实现。

## R09

适用前提与边界：
- 仅为玩具级模拟，不是实验离线重建软件替代品
- 粒子类型仅区分 `electron` 与 `pion`
- 几何效应、角度依赖、磁场、死区、时间漂移未显式建模
- 采样分数设为常数（未做通道级非均匀性）

适合用途：算法演示、流程验证、快速原型对比。

## R10

正确性支撑：
- 物理链路正确：`E_true -> 沉积 -> 采样 -> 噪声 -> 重建`
- 基线公式可解释：直接来自采样量能器反演思想
- 校准模型可解释：线性模型参数可读，不涉及不可追踪黑箱
- 评估指标完整：`MAE/RMSE/MAPE/Bias/Resolution/Linearity/R2`

## R11

数值稳定性处理：
- 信号和分母使用下限保护（如 `+1e-9`）防止除零
- 噪声叠加后对读出取 `max(0, value)` 防止非物理负能量
- 线性回归前标准化，降低不同尺度特征造成的病态
- 线性度拟合采用 `np.polyfit`，在测试集数量充足时稳定

## R12

真实开销（超越大 O）：
- 2500 事件规模下，CPU 运行通常是秒级
- 常数开销来自：
  - pandas DataFrame 构建
  - sklearn 训练管线（虽小但有对象封装成本）
  - 分箱统计和表格格式化输出
- 相比大规模实验框架，本 MVP 成本非常低

## R13

误差与不确定性来源：
- 主导误差：模型简化（簇射参数化、泄漏建模、常数采样分数）
- 次级误差：随机噪声抽样导致统计波动
- 数值误差：双精度下通常远小于上述物理/模型误差

因此结果解读应关注“趋势与相对改进”，而非逐事件精确复现真实探测器。

## R14

鲁棒性与失效模式：
- `n_events <= 0` 会被显式拒绝
- 非支持粒子类型会触发 `ValueError`
- 极端参数（过大噪声或不合理分数）会明显恶化线性与分辨率
- 若训练/测试分布偏移过大，线性模型会出现外推偏差

## R15

工程实现要点：
- 使用 `DetectorConfig` 集中管理采样分数和噪声参数
- 数据、重建、训练、评估分函数实现，便于单元替换
- 保留 `baseline` 与 `calibrated` 双轨输出，便于 A/B 对比
- 输出分箱分辨率表，避免只看全局平均指标掩盖局部问题

## R16

前驱与后继关系：
- 前驱：电磁/强子簇射理论、采样量能器响应模型、能量反演思想
- 同层：喷注重建前处理中的簇能标定、实验校准曲线拟合
- 后继：
  - 非线性校准（GBDT、NN）
  - 粒子流（Particle Flow）联合重建
  - 时间信息与纵向细粒度分层重建

## R17

运行说明（本目录）：
- 命令：
  - `cd "Algorithms/物理-粒子物理实验-0429-量能器_(Calorimeters)"`
  - `uv run python demo.py`
- 脚本无交互输入，默认固定随机种子，可重复运行
- 输出包含三部分：
  - 方法总体指标对比表
  - 测试集前 12 条事件预测预览
  - 分能量区间分辨率对比表

## R18

源码级算法流（对应 `demo.py`，8 步，非黑箱）：
1. `build_dataset` 逐事件调用 `simulate_event`，生成 `ecal/hcal` 四层读出与真值能量。
2. `simulate_event` 内部先用 `sample_shower_fractions` 按粒子类型抽样 `(em, had, leakage)`，再做纵向层分配和噪声叠加。
3. `build_dataset` 计算衍生特征 `ecal_signal_ratio`、`longitudinal_ratio`，形成训练表。
4. `train_test_split` 划分训练/测试集，`build_feature_matrix` 提取 7 维输入特征。
5. `baseline_reconstruction` 用固定采样分数直接反演得到 `E_base`。
6. `train_calibration_model` 训练 `StandardScaler + Ridge` 线性校准器并在测试集推理 `E_cal`。
7. `evaluate_predictions` 分别计算 `MAE/RMSE/MAPE/Bias/Resolution/Linearity/R2`，量化基线与校准性能差异。
8. `binned_resolution_table` 用 `scipy.stats.binned_statistic` 做分箱分辨率统计，输出各能区改进情况。

补充说明：第 6 步虽调用 sklearn 高层 API，但底层是标准线性代数求解的岭回归，不是不可解释黑盒模型。
