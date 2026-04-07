# 狭义相对论 (Special Relativity)

- UID: `PHYS-0041`
- 学科: `物理`
- 分类: `相对论`
- 源序号: `41`
- 目标目录: `Algorithms/物理-相对论-0041-狭义相对论_(Special_Relativity)`

## R01

狭义相对论研究惯性系之间的时空变换规律，核心是：
- 光速 `c` 在所有惯性系中不变；
- 物理规律在所有惯性系中形式相同。

本目录给出一个可运行的最小 MVP，用数值方式实现并验证四个关键结论：
- 一维洛伦兹变换；
- 闵可夫斯基间隔不变量；
- 时间膨胀与同时性相对性；
- 相对论速度叠加公式。

## R02

问题定义（本实现）：
- 输入：
  - 事件数组 `events`，每个事件写为 `[ct, x]`；
  - 速度参数 `beta = v / c`，满足 `|beta| < 1`；
  - 速度叠加中的 `u`、`v`（单位 m/s）。
- 输出：
  - 变换后的事件 `events'`；
  - 变换前后间隔平方 `s^2 = (ct)^2 - x^2` 的误差统计；
  - 时间膨胀与同时性相对性的数值验证结果；
  - 相对论速度叠加与经典叠加的对比结果。

`demo.py` 不读取交互输入，直接运行固定可复现实验。

## R03

核心数学关系：

1. 洛伦兹因子：
   `gamma = 1 / sqrt(1 - beta^2)`。
2. 一维洛伦兹变换（沿 `x` 方向）：
   - `ct' = gamma * (ct - beta * x)`
   - `x'  = gamma * (x  - beta * ct)`
3. 闵可夫斯基间隔：
   `s^2 = (ct)^2 - x^2`，在任意惯性系中不变。
4. 时间膨胀：
   `Delta t = gamma * Delta tau`（`Delta tau` 为固有时）。
5. 速度叠加：
   `w = (u + v) / (1 + uv/c^2)`，并且 `|w| < c`。

## R04

算法流程（高层）：
1. 校验 `beta` 的有限性与物理约束 `|beta| < 1`。  
2. 校验事件矩阵形状是否为 `(n, 2)` 且元素有限。  
3. 按洛伦兹公式对所有事件做向量化变换。  
4. 计算变换前后 `s^2`，统计最大绝对误差和相对误差。  
5. 构造固有时样例，验证 `Delta t = gamma * Delta tau`。  
6. 构造同一参考系“同时发生”的两事件，验证 `Delta t' != 0`。  
7. 计算速度叠加并与经典线性叠加比较。  
8. 输出所有实验结果和阈值检查结论。

## R05

核心数据结构：
- `events: np.ndarray(shape=(n,2), dtype=float)`：每行一个事件 `[ct, x]`。
- `intervals: np.ndarray(shape=(n,))`：各事件对应 `s^2`。
- `report: dict[str, float|bool]`：每组实验的误差与通过标记。
- 固定常量：
  - `C = 299792458.0`（光速，m/s）；
  - `EPS = 1e-12`（数值比较保护项）。

## R06

正确性要点：
- 变换公式直接来自狭义相对论标准一维洛伦兹变换。  
- 间隔不变性通过逐事件比较 `s^2_before` 与 `s^2_after` 验证。  
- 时间膨胀通过“固有时 -> 坐标时 -> 反推固有时”闭环验证。  
- 同时性相对性通过 `Delta t = 0` 但 `Delta t' != 0` 的构造性样例验证。  
- 速度叠加结果额外检查 `|w| < c`，避免出现超光速数值错误。

## R07

复杂度分析（`n` 为事件数）：
- 一次洛伦兹变换：`O(n)` 时间，`O(n)` 额外空间（输出数组）。
- 间隔不变量检查：`O(n)` 时间，`O(n)` 空间。
- 时间膨胀、同时性、速度叠加实验：`O(1)` 或 `O(k)`（`k` 为速度样本数）。
- 总体：`O(n + k)`，线性可扩展。

## R08

边界与异常处理：
- `beta` 非有限值或 `|beta| >= 1`：抛 `ValueError`。  
- `events` 不是二维或第二维不为 2：抛 `ValueError`。  
- `events`、`u`、`v` 中出现 `nan/inf`：抛 `ValueError`。  
- 速度叠加分母 `1 + uv/c^2` 过小：抛 `ZeroDivisionError`。  
- 所有检查都在计算前进行，避免产生不可解释结果。

## R09

MVP 取舍：
- 仅实现一维（`1+1` 维）洛伦兹变换，避免引入四维张量框架。  
- 不做广义相对论、加速参考系与引力效应。  
- 不依赖黑盒物理库，直接按公式实现，便于审计每一步来源。  
- 重点放在“公式 -> 代码 -> 数值验证”的最小闭环。

## R10

`demo.py` 主要函数职责：
- `validate_beta`：检查并规范化 `beta`。  
- `validate_events`：检查事件矩阵形状和有限性。  
- `gamma_from_beta`：计算洛伦兹因子 `gamma`。  
- `lorentz_transform_1d`：执行一维洛伦兹变换。  
- `minkowski_interval_sq`：计算 `s^2 = (ct)^2 - x^2`。  
- `velocity_addition`：执行相对论速度叠加。  
- `run_interval_invariance_demo`：间隔不变量实验。  
- `run_time_dilation_demo`：时间膨胀实验。  
- `run_relativity_of_simultaneity_demo`：同时性相对性实验。  
- `run_velocity_addition_demo`：速度叠加实验。  
- `main`：组织样例、运行并输出汇总。

## R11

运行方式：

```bash
cd Algorithms/物理-相对论-0041-狭义相对论_(Special_Relativity)
uv run python demo.py
```

脚本不会请求输入，也不依赖命令行参数。

## R12

输出字段说明：
- `beta`：参考系相对速度比例 `v/c`。  
- `gamma`：洛伦兹因子。  
- `max_abs_error`：间隔不变量的最大绝对误差。  
- `max_rel_error`：间隔不变量的最大相对误差。  
- `proper_time_s`：固有时（秒）。  
- `frame_time_s`：对应惯性系坐标时间（秒）。  
- `delta_t_prime_s`：变换后两事件时间差，用于显示同时性相对性。  
- `classical_sum_over_c`：经典速度和相对光速比。  
- `relativistic_sum_over_c`：相对论速度和相对光速比。  
- `all_sub_luminal`：随机速度叠加是否全部满足 `|w| < c`。

## R13

内置最小测试集：
- 间隔不变量：固定随机种子生成一批事件，分别在多个 `beta` 下变换并比较误差。  
- 时间膨胀：`[1, 10, 60]` 秒固有时样例。  
- 同时性相对性：同一系同时、异地两事件，验证变换后不再同时。  
- 速度叠加：
  - 经典与相对论公式对比（高速度示例）；
  - 随机 2000 对速度检查是否超光速。

## R14

可调参数：
- `betas`：不变量测试中使用的参考系速度列表。  
- `n_events`：随机事件数量。  
- `threshold`：误差通过阈值（默认 `1e-10` 量级）。  
- `simultaneity_dx_m`、`simultaneity_t0_s`：同时性实验的空间与时间尺度。  
- `random_velocity_pairs`：速度叠加随机检验样本数。

调参建议：
- 若想更严格检验数值稳定性，可增大 `n_events` 与速度接近 `|beta|=1` 的样本比例。  
- 若运行平台浮点差异较大，可适度放宽误差阈值。

## R15

与经典牛顿时空观对比：
- 经典变换（伽利略变换）下时间绝对，速度线性相加，可得到超光速结果。  
- 狭义相对论下时间和空间耦合，速度叠加是分式形式，天然约束 `|v| < c`。  
- 本 MVP 通过同一组数值样例直观展示两种理论的分歧位置。

## R16

典型应用场景：
- 高能粒子寿命与束流传播时间估算。  
- 卫星导航和高速运动系统中的时钟校正前置模型。  
- 物理建模课程中“公式到代码”的验证示例。  
- 作为更高维四矢量与场论数值实现的基础组件。

## R17

可扩展方向：
- 扩展到 `3+1` 维，支持任意方向速度矢量与完整洛伦兹矩阵。  
- 增加四动量变换与质能关系 `E^2=(pc)^2+(mc^2)^2` 的数值验证。  
- 增加固有时沿离散世界线积分（分段匀速近似）。  
- 结合 `matplotlib` 可视化闵可夫斯基图与光锥。  
- 增加单元测试与 CI 自动阈值校验。

## R18

`demo.py` 源码级流程（9 步）：
1. `main` 固定随机种子并定义物理常量、误差阈值和实验参数。  
2. `run_interval_invariance_demo` 生成事件矩阵 `events[n,2]`（列为 `ct,x`）。  
3. 对每个 `beta`，`lorentz_transform_1d` 先调用 `validate_beta/validate_events`，再用向量化公式得到 `events'`。  
4. `minkowski_interval_sq` 分别计算 `events` 与 `events'` 的 `s^2`，并统计最大绝对/相对误差。  
5. `run_time_dilation_demo` 调用 `gamma_from_beta`，执行 `Delta t = gamma * Delta tau` 与反推校验。  
6. `run_relativity_of_simultaneity_demo` 构造同一系同时两事件，变换后读取 `ct'` 差并换算为 `Delta t'`。  
7. `run_velocity_addition_demo` 先计算一个高速示例的经典和相对论结果，再批量随机检查叠加速度是否都小于光速。  
8. `velocity_addition` 内部显式检查分母 `1 + uv/c^2`，确保不会在奇异点附近静默失败。  
9. `main` 汇总各实验结果并输出 `PASS/FAIL` 结论，形成“实现-验证-解释”闭环。
