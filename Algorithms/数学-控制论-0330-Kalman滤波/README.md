# Kalman滤波

- UID: `MATH-0330`
- 学科: `数学`
- 分类: `控制论`
- 源序号: `330`
- 目标目录: `Algorithms/数学-控制论-0330-Kalman滤波`

## R01

Kalman 滤波（KF）用于线性高斯动态系统的在线状态估计。它把“系统动力学预测”和“带噪观测修正”结合在同一个递推框架里，在每一时刻输出当前最优线性无偏估计。

本目录的 MVP 采用二维平面匀速目标跟踪：
- 状态 `x=[px, py, vx, vy]`；
- 观测 `z=[px, py]`（仅位置可测，速度不可直接测）；
- 用合成数据验证滤波后误差是否明显优于原始量测。

## R02

离散线性高斯模型：

- 状态方程：`x_k = F x_{k-1} + w_{k-1}`, `w ~ N(0, Q)`
- 观测方程：`z_k = H x_k + v_k`, `v ~ N(0, R)`

Kalman 两步递推：

1. 预测（Predict）
   - `x_k^- = F x_{k-1}`
   - `P_k^- = F P_{k-1} F^T + Q`
2. 更新（Update）
   - `y_k = z_k - H x_k^-`（innovation）
   - `S_k = H P_k^- H^T + R`
   - `K_k = P_k^- H^T S_k^{-1}`
   - `x_k = x_k^- + K_k y_k`
   - `P_k = (I-K_kH)P_k^-(I-K_kH)^T + K_k R K_k^T`（Joseph 形式）

## R03

本实现中的具体矩阵定义：

- 采样间隔 `dt`
- 状态转移矩阵
  - `F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]`
- 观测矩阵
  - `H = [[1,0,0,0],[0,1,0,0]]`
- 加速度噪声离散映射矩阵
  - `G = [[0.5*dt^2,0],[0,0.5*dt^2],[dt,0],[0,dt]]`
- 过程噪声协方差
  - `Q = G * diag([q_acc, q_acc]) * G^T`
- 量测噪声协方差
  - `R = diag([r_pos^2, r_pos^2])`

这对应“匀速 + 高斯加速度扰动 + 高斯位置测量误差”的标准教学/工程基线模型。

## R04

`demo.py` 的主流程：

1. 固定 `n_steps/dt/q_acc/r_pos`，并设置随机种子。
2. 通过 `simulate_truth_and_measurements` 生成真值轨迹 `truths` 与观测 `zs`。
3. 初始化滤波器状态 `x_est` 与协方差 `P_est`（故意偏离真值）。
4. 对每个时刻先调用 `kf_predict` 得到 `x_pred/P_pred`。
5. 再调用 `kf_update` 融合观测，得到 `x_est/P_est`。
6. 记录创新范数、增益指标 `trace(KH)` 与状态估计序列。
7. 计算量测 RMSE、滤波后位置 RMSE、速度 RMSE。
8. 打印末状态对比与统计指标。

## R05

核心数据结构：

- `x`：长度 4 的状态向量 `[px, py, vx, vy]`。
- `P`：`4x4` 协方差矩阵，表示状态不确定性。
- `z`：长度 2 的量测向量 `[px, py]`。
- `F/H/Q/R`：Kalman 递推所需系统矩阵。
- `truths`：形状 `(T, 4)` 的真值轨迹。
- `zs`：形状 `(T, 2)` 的带噪观测。
- `estimates_arr`：形状 `(T, 4)` 的滤波估计轨迹。
- `innovation_norms`：每步创新二范数，反映观测与预测残差规模。

## R06

正确性关键点：

1. 模型匹配：数据生成和滤波采用同一线性模型族，验证递推公式本身。
2. 协方差传播：预测步使用 `FPF^T + Q`，确保过程噪声被显式计入。
3. 最优融合：更新步通过 `K = PH^TS^{-1}` 在模型与观测间自适应权衡。
4. 数值稳健：更新协方差用 Joseph 形式，减轻有限精度下的非对称/负定问题。
5. 可验证：同时输出 `measurement RMSE` 与 `filtered RMSE`，便于直接判断滤波收益。

## R07

复杂度分析（状态维 `n=4`，观测维 `m=2`，步数 `T`）：

- 单步预测：`O(n^3)`（矩阵乘法主导）。
- 单步更新：`O(n^2m + m^3)`，其中 `m=2` 时 `S` 求解成本很低。
- 总时间复杂度：`O(T*(n^3 + n^2m + m^3))`。
- 空间复杂度：
  - 递推状态本身 `O(n^2)`；
  - 若保留完整轨迹用于评估，则额外 `O(Tn)`。

在本任务规模下可实时运行。

## R08

边界条件与异常处理：

- `dt <= 0`、`q_acc < 0`、`n_steps <= 0`、`r_pos <= 0`：直接 `ValueError`。
- 关键矩阵/向量形状不一致：`check_matrix/check_vector` 抛错。
- 输入或中间结果出现 `nan/inf`：抛 `ValueError` 或 `RuntimeError`。
- 本实现假设线性高斯模型；若系统强非线性或噪声重尾，估计质量不再保证。

## R09

MVP 取舍说明：

- 只用 `numpy`，不依赖现成滤波库，完整保留算法解释权。
- 只实现单目标、线性观测、全状态协方差 KF。
- 不引入平滑器（RTS）、多模型（IMM）、数据关联等高级模块。
- 重点放在“最小可运行 + 可审计指标 + 清晰代码路径”。

## R10

`demo.py` 主要函数职责：

- `build_cv_model`：构造 `F/H/Q/G`。
- `simulate_truth_and_measurements`：生成真值与带噪观测。
- `kf_predict`：执行预测步。
- `kf_update`：执行更新步并返回创新与增益。
- `check_matrix/check_vector`：输入合法性检查。
- `run_kalman_demo`：组织整条实验流水线并输出指标。
- `main`：脚本入口，无交互参数。

## R11

运行方式（无交互输入）：

```bash
cd Algorithms/数学-控制论-0330-Kalman滤波
uv run python demo.py
```

或：

```bash
python demo.py
```

## R12

输出字段与 R04 流程对应：

- `steps`, `dt`：实验配置（R04 第 1 步）。
- `measurement position RMSE`：原始观测相对真值误差（R04 第 2 步）。
- `filtered position RMSE`：滤波后位置误差（R04 第 7 步）。
- `filtered velocity RMSE`：滤波恢复的速度误差（R04 第 7 步）。
- `mean innovation norm`：更新步平均残差尺度（R04 第 5-6 步）。
- `mean trace(KH)`：平均融合强度指标（R04 第 6 步）。
- `final true state / final estimated state`：末时刻状态对照（R04 第 8 步）。

预期现象：`filtered position RMSE` 通常小于 `measurement position RMSE`。

## R13

最小测试建议（与 R11/R14 一致，均可通过直接改常量复现）：

1. 基线测试：保持默认参数运行，确认脚本完成且滤波误差优于量测误差。
2. 高噪声量测测试：增大 `r_pos`，观察 Kalman 增益下降、对模型预测依赖增强。
3. 高过程噪声测试：增大 `q_acc`，观察滤波器更快跟随机动变化。
4. 初值偏差测试：增大初始状态偏差，验证是否仍能逐步收敛。
5. 异常输入测试：设置非法 `dt/r_pos`，确认触发异常而非静默失败。

## R14

关键可调参数：

- `q_acc`：过程噪声强度；越大表示系统机动性越强、模型不确定性越高。
- `r_pos`：量测噪声标准差；越大表示观测可信度越低。
- `x_est` 初值：初始状态猜测。
- `P_est` 初值：初始不确定性；越大意味着初始阶段更信观测。
- `n_steps`：评估长度。

调参原则（对应 R13）：

- 若滤波轨迹响应过慢，可增大 `q_acc` 或适度减小 `r_pos`。
- 若输出抖动偏大，可减小 `q_acc` 或增大 `r_pos`。
- 若初期震荡大，可增大 `P_est` 对角项，让滤波器更快修正初值偏差。

## R15

与相关方法对比：

- 对比滑动平均：
  - 滑动平均不建模动力学；
  - KF 同时建模状态演化和观测噪声，能估计不可直接观测的速度。
- 对比扩展 Kalman（EKF）：
  - KF 用于线性模型；
  - EKF 处理非线性但只是一阶近似，复杂度和调参难度更高。
- 对比粒子滤波（PF）：
  - PF 适合非高斯/多峰后验；
  - KF 在高斯线性条件下更高效且解析性更强。

## R16

适用场景与限制（与 R08 一致）：

- 适用：
  - 线性状态空间系统；
  - 噪声可近似为高斯；
  - 需要实时在线估计位置、速度等状态。
- 典型应用：
  - 目标跟踪、导航定位、传感器融合中的基础状态估计。
- 限制：
  - 强非线性系统应使用 EKF/UKF/PF；
  - 重尾或强异常噪声环境需配合鲁棒观测模型。

## R17

可扩展方向：

1. 增加 RTS smoother，得到离线平滑估计。
2. 增加异常观测门限（Mahalanobis gating）做鲁棒更新。
3. 扩展到多传感器异步融合（不同 `H/R`）。
4. 将单模型 KF 扩展为 IMM（交互多模型）应对机动目标。
5. 迁移到 EKF/UKF 处理非线性系统。

## R18

`demo.py` 源码级算法流（8 步，非黑箱）：

1. `run_kalman_demo` 设置实验参数并调用 `simulate_truth_and_measurements` 生成 `(truths, zs)`。
2. `simulate_truth_and_measurements` 每个时刻按 `x_k = F x_{k-1} + G a_k` 推进真值，再生成 `z_k = Hx_k + noise`。
3. `run_kalman_demo` 调用 `build_cv_model` 得到 `F/H/Q`，并构造 `R`、初始 `x_est/P_est`。
4. 每步先执行 `kf_predict`：计算 `x_pred = F x_est` 与 `P_pred = F P_est F^T + Q`。
5. 再执行 `kf_update`：先算创新 `y = z - Hx_pred` 与创新协方差 `S = H P_pred H^T + R`。
6. `kf_update` 用 `np.linalg.solve` 求解 Kalman 增益 `K = P_pred H^T S^{-1}`，避免直接求逆。
7. `kf_update` 更新状态 `x_est = x_pred + K y`，并用 Joseph 形式更新 `P_est` 保持数值稳定。
8. 主循环收集估计序列与诊断量，最后计算并打印量测/滤波 RMSE、平均创新范数和末状态对比。
