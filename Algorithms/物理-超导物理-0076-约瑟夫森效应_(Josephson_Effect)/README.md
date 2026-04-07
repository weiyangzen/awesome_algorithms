# 约瑟夫森效应 (Josephson Effect)

- UID: `PHYS-0076`
- 学科: `物理`
- 分类: `超导物理`
- 源序号: `76`
- 目标目录: `Algorithms/物理-超导物理-0076-约瑟夫森效应_(Josephson_Effect)`

## R01

约瑟夫森效应描述了两个超导体通过薄绝缘层（Josephson 结）耦合时的相位动力学与电流-电压行为。核心实验事实是：
- 直流约瑟夫森效应：在零电压下可通过超导隧穿电流；
- 交流约瑟夫森效应：在外加电压下相位线性演化，导致交流超电流；
- 在微波驱动下出现 Shapiro steps（离散电压平台）。

本目录给出一个最小可运行 MVP：基于过阻尼 RSJ（Resistively Shunted Junction）模型做数值积分，输出 I-V 曲线并自动检测 Shapiro 平台。

## R02

本实现的问题定义：
- 输入（脚本内固定参数，无交互）：
  - 归一化直流偏置电流网格 `I_dc`；
  - 交流驱动幅值 `i_ac` 与角频率 `omega`；
  - 数值积分参数 `dt`, `n_steps`, `burn_in_steps`。
- 输出：
  - 纯直流驱动下的归一化 I-V 曲线样本；
  - 直流+交流驱动下的归一化 I-V 曲线样本；
  - 检测到的 Shapiro steps（平台电压、对应电流区间、平台斜率）。

`demo.py` 可直接运行：`uv run python demo.py`。

## R03

采用归一化的过阻尼 RSJ 方程：

`dphi/dtau = i_dc + i_ac * sin(omega * tau) - sin(phi)`

其中：
- `phi`：超导序参量相位差；
- `i_dc`, `i_ac`：分别为直流和交流驱动电流（按临界电流归一化）；
- `tau`：归一化时间；
- 归一化平均电压满足 `V ~ <dphi/dtau>`（按约瑟夫森关系归一化后）。

Shapiro 台阶判据来自 `V ≈ n * omega`（`n=0,1,2,...`）。

## R04

算法总流程：
1. 生成电流网格 `currents = linspace(0, 2.5, 121)`。  
2. 在 `i_ac = 0` 下逐点积分 RSJ 方程，得到 `v_dc`。  
3. 在 `i_ac > 0` 下逐点积分 RSJ 方程，得到 `v_ac`。  
4. 对 `v_ac(I)` 计算数值导数 `dV/dI`。  
5. 逐个目标台阶 `V_n = n*omega` 做“电压接近 + 斜率接近 0”筛选。  
6. 将 True 掩码切分为连续区间，过滤太短区间。  
7. 汇总每个平台的电流边界、平均电压与平均斜率。  
8. 打印可读报告。

## R05

核心数据结构：
- `RSJConfig(dt, n_steps, burn_in_steps)`：积分配置。
- `currents: np.ndarray(shape=(N,))`：偏置电流扫描网格。
- `v_dc, v_ac: np.ndarray(shape=(N,))`：两种驱动下的平均电压。
- `steps: list[dict[str, float]]`：每个检测到的平台摘要：
  - `n`, `target_v`, `i_left`, `i_right`, `v_mean`, `mean_abs_dv_di`, `points`。

## R06

正确性要点：
- 相位动力学直接来自 RSJ 常微分方程，不依赖黑盒物理库。
- 平均电压通过后稳态窗口（burn-in 之后）对 `dphi/dtau` 取平均，避免初始瞬态污染。
- Shapiro steps 检测同时要求：
  - 电压接近 `n*omega`；
  - 局部斜率足够小（平台而非斜坡）。
- 连续区间分割保证输出的是“平台段”而非离散噪点。

## R07

复杂度分析（`N` 为电流网格点数，`T` 为积分步数）：
- 单次积分：`O(T)`。
- 一条 I-V 曲线：`O(N*T)`。
- 两条曲线：`O(2*N*T)`。
- 平台检测：`O(N * K)`，`K` 为检查的台阶数（默认 5）。
- 总体：`O(N*T + N*K)`，主导项为 `O(N*T)`；空间 `O(N)`。

## R08

边界与异常处理：
- `n_steps <= burn_in_steps` 会抛 `ValueError`，防止平均窗口为空。
- `dt <= 0` 会抛 `ValueError`，防止非法积分步长。
- 平台检测时若 `currents` 与 `voltages` 形状不一致会抛 `ValueError`。
- 若给定阈值下未检测到任何平台，会输出 "None detected"，而不是静默失败。

## R09

MVP 取舍说明：
- 只实现过阻尼 RSJ，不包含电容项（即不做完整 RCSJ）。
- 不加入热噪声与随机涨落，保持结果稳定可复现。
- 不绘图，仅输出文本摘要，确保在最小环境可运行。
- 重点是“可解释的算法闭环”，而不是实验级参数拟合。

## R10

`demo.py` 函数职责：
- `simulate_average_voltage`：单个偏置点上的时间积分与稳态平均。
- `sweep_iv_curve`：扫描电流网格，生成 I-V 数组。
- `_contiguous_true_segments`：把布尔掩码切成连续区间。
- `detect_shapiro_steps`：按目标电压和斜率阈值检测平台。
- `summarize_curve_samples`：提取代表性采样点用于打印。
- `estimate_switching_current`：估计从零电压态切换的阈值电流。
- `main`：组织配置、执行两组扫描并输出报告。

## R11

运行方式：

```bash
cd Algorithms/物理-超导物理-0076-约瑟夫森效应_(Josephson_Effect)
uv run python demo.py
```

脚本不会请求任何输入。

## R12

默认参数（归一化单位）：
- `dt = 0.05`
- `n_steps = 12000`
- `burn_in_steps = 3000`
- `currents in [0.0, 2.5], N=121`
- AC 驱动：`i_ac = 1.2`, `omega = 0.5`
- 平台检测：
  - `max_step_index = 4`
  - `voltage_tolerance = 0.08`
  - `slope_tolerance = 0.15`
  - `min_points = 3`

## R13

输出字段解释：
- `I`：偏置电流（归一化）。
- `<V>`：平均电压（归一化，约等于 `<dphi/dtau>`）。
- `Ic`：阈值法估计的切换电流（`<V> > 0.05` 的首个 `I`）。
- 平台条目：
  - `n`：第 `n` 个 Shapiro 台阶；
  - `target`：理论平台电压 `n*omega`；
  - `I in [left,right]`：检测到的平台电流区间；
  - `mean V`：平台段平均电压；
  - `mean |dV/dI|`：平台平坦程度指标。

## R14

内置验证思路：
- 纯 DC 下，低电流区域应接近零电压超导支路。
- 加入 AC 后，I-V 会出现接近 `n*omega` 的离散平台。
- 平台检测结果应给出多个 `n`（通常从 `0` 到若干正整数）。
- 如果调高 `i_ac`，平台通常更明显；如果调高 `slope_tolerance`，检测会更宽松。

## R15

与“黑盒调用”区别：
- 本实现没有调用现成 Josephson 结求解器。
- 从方程到数值结果全部在 `demo.py` 明确展开，可逐行审计：
  - 微分方程离散化；
  - 稳态窗口平均；
  - 平台判据与区间提取。
- 这满足“最小但诚实”的算法实现目标。

## R16

局限性：
- 过阻尼假设忽略了结电容与惯性效应。
- 未建模温度噪声、材料非理想性和器件参数离散性。
- 阈值法检测平台依赖经验参数，不是统计最优检测器。
- 归一化单位便于算法演示，但不能直接替代实验数据拟合。

## R17

可扩展方向：
- 升级到 RCSJ 模型（加入 `d^2phi/dtau^2` 项）。
- 引入噪声项，研究热激活与相位扩散。
- 增加参数扫描（`i_ac`, `omega`, 阻尼系数）并输出二维相图。
- 用 `pandas` 保存曲线数据，用 `matplotlib` 绘制 I-V 和平台标注图。
- 增加自动化测试，验证关键参数扰动下的平台稳定性。

## R18

`demo.py` 源码级算法流程（9 步）：
1. `main` 创建 `RSJConfig` 并定义电流网格与 AC 驱动参数。  
2. 调用 `sweep_iv_curve(..., i_ac=0)`，逐个 `I_dc` 进入 `simulate_average_voltage`。  
3. 在 `simulate_average_voltage` 内部，按欧拉法迭代相位：`phi <- phi + (i_dc + i_ac*sin(omega*t)-sin(phi))*dt`。  
4. 丢弃 burn-in 区间后累计 `dphi`，计算稳态平均 `V = <dphi/dtau>`。  
5. 重复第 2-4 步但设置 `i_ac=1.2, omega=0.5`，得到受微波驱动的 `v_ac`。  
6. `detect_shapiro_steps` 先用 `np.gradient` 计算 `dV/dI`，并为每个 `n` 构造掩码：`|V-n*omega|<=tol_v` 且 `|dV/dI|<=tol_s`。  
7. `_contiguous_true_segments` 将掩码切分为连续区间，过滤掉点数不足 `min_points` 的噪段。  
8. 对每个合法区间计算 `I` 边界、平均 `V` 和平均斜率，汇总为 step 记录。  
9. `main` 打印两组 I-V 代表点、`Ic` 估计值与全部 step 清单，形成“建模-积分-检测-解释”闭环。
