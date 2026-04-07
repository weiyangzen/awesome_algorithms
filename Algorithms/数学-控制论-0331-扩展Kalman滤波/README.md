# 扩展Kalman滤波

- UID: `MATH-0331`
- 学科: `数学`
- 分类: `控制论`
- 源序号: `331`
- 目标目录: `Algorithms/数学-控制论-0331-扩展Kalman滤波`

## R01

扩展 Kalman 滤波（EKF）是把线性 Kalman 滤波推广到“非线性状态方程或非线性观测方程”的经典方法。核心思想是:

1. 在当前估计点附近对非线性函数做一阶泰勒展开。
2. 用 Jacobian 近似局部线性系统。
3. 复用 Kalman 预测-更新框架。

本目录的 MVP 选择二维目标跟踪任务:
- 状态是线性的匀速模型 `[px, py, vx, vy]`
- 观测是非线性的雷达极坐标 `[range, bearing]`
- 用 EKF 估计真实轨迹并输出 RMSE。

## R02

目标问题可写为离散随机系统:

- 状态转移: `x_k = f(x_{k-1}) + w_{k-1}`, `w ~ N(0, Q)`
- 观测模型: `z_k = h(x_k) + v_k`, `v ~ N(0, R)`

与线性 KF 的区别是 `f` 或 `h` 至少有一个是非线性的。  
这里我们采用:
- `f`: 匀速运动（线性）
- `h`: 极坐标观测（非线性）

即 EKF 的非线性来源主要在观测侧。

## R03

本实现的状态变量与物理意义:

- `x = [px, py, vx, vy]^T`
- `px, py`: 平面位置
- `vx, vy`: 平面速度

离散时间间隔 `dt` 下:

`f(x) = [px + vx*dt, py + vy*dt, vx, vy]^T`

对应状态转移 Jacobian（也是线性模型矩阵）:

`F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]`

## R04

观测函数定义为雷达量测:

- `range = sqrt(px^2 + py^2)`
- `bearing = atan2(py, px)`

即:

`h(x) = [sqrt(px^2+py^2), atan2(py, px)]^T`

该函数非线性，导致无法直接使用标准线性 Kalman 更新，需要在当前预测点做局部线性化。

## R05

EKF 在每个时刻计算观测 Jacobian:

记 `r2 = px^2 + py^2`, `r = sqrt(r2)`，则

`H = dh/dx = [[px/r, py/r, 0, 0],[-py/r2, px/r2, 0, 0]]`

实现中为避免除零，使用 `eps` 下界保护 `r2 = max(r2, eps)`。  
这一步是 EKF 的关键近似来源: 用局部一阶导数替代真实非线性曲面。

## R06

预测步骤（Predict）:

1. `x_pred = f(x_est)`
2. `P_pred = F P F^T + Q`

其中:
- `Q` 为过程噪声协方差
- `P` 为状态估计协方差

在本 demo 中，`Q` 来自加速度噪声通过离散积分矩阵 `G` 投影:

`Q = G * diag([q_acc, q_acc]) * G^T`

## R07

更新步骤（Update）:

1. `z_pred = h(x_pred)`
2. 创新 `y = z - z_pred`，并对角度分量做 `wrap_angle`
3. `S = H P_pred H^T + R`
4. `K = P_pred H^T S^{-1}`
5. `x = x_pred + K y`
6. 协方差采用 Joseph 形式:
   `P = (I-KH)P_pred(I-KH)^T + K R K^T`

Joseph 形式相比 `P=(I-KH)P_pred` 更稳健，能减小数值不对称和负定风险。

## R08

噪声与初值设定（demo 参数）:

- `dt = 0.1`
- 步数 `n_steps = 120`
- 加速度噪声方差参数 `q_acc = 0.35`
- 量测噪声标准差:
  - `range`: `0.25`
  - `bearing`: `0.03`（弧度）
- 初始估计故意偏离真实值: `[0.3, -0.5, 0.0, 0.0]`
- 初始协方差: `diag([2,2,1,1])`

该配置能直观看到滤波收敛过程，同时保持 MVP 简洁。

## R09

数据生成方式:

1. 真实状态使用同一匀速矩阵 `F` 推进。
2. 每步叠加高斯加速度噪声（经 `G` 映射到状态）。
3. 由真实状态计算无噪声观测 `h(x_true)`。
4. 叠加高斯量测噪声得到 `z`。
5. 对观测角度做归一化，保持在 `[-pi, pi)`。

因此 demo 不依赖外部数据集，运行即可复现实验。

## R10

`demo.py` 结构（主要函数）:

- `wrap_angle`: 角度归一化
- `f_state`, `jacobian_f`: 状态模型与 Jacobian
- `h_measurement`, `jacobian_h`: 观测模型与 Jacobian
- `ekf_predict`: 预测步骤
- `ekf_update`: 更新步骤（含 Joseph 形式）
- `simulate_truth_and_measurements`: 合成真值与观测
- `run_ekf_demo`: 组织完整实验并输出指标

依赖仅 `numpy`，属于最小可运行实现。

## R11

运行方式（无交互输入）:

```bash
uv run python demo.py
```

或在已有 Python 环境中:

```bash
python demo.py
```

## R12

预期输出包含:

1. 运行完成标记
2. 步数和 `dt`
3. 位置 RMSE
4. 速度 RMSE
5. 最后一步真实状态与估计状态

不同机器上随机数实现可能存在细微差异，但在固定 seed 下应保持稳定可复现。

## R13

复杂度分析（状态维度 `n=4`，观测维度 `m=2`）:

- 预测: `O(n^3)`（矩阵乘法主导）
- 更新:
  - `S` 计算 `O(m n^2)`
  - `S^{-1}` 为 `O(m^3)`（此处 `m=2` 很小）
  - 其余约 `O(n^2 m)`

整体每步成本在本问题规模下很低，可实时运行。

## R14

适用场景:

- 模型“弱非线性”或采样足够高，使一阶近似可接受
- 需要在线递推、低延迟状态估计
- 噪声近似高斯，且协方差可建模

不适用或效果较差场景:

- 强非线性、多峰后验分布
- 线性化点远离真实状态（初值差或观测稀疏）
- 角度跳变严重且未正确处理角度残差

## R15

常见失败模式与对策:

1. `px, py` 接近 0 导致 Jacobian 分母趋零  
   对策: `r2` 加 `eps` 下界保护。
2. 角度创新跨 `pi` 边界导致错误大残差  
   对策: 对角度创新执行 `wrap_angle`。
3. 协方差数值不稳定、失去半正定  
   对策: 使用 Joseph 形式更新 `P`。
4. `Q/R` 配置失衡导致发散或过度平滑  
   对策: 结合残差统计调参。

## R16

可扩展方向:

1. 改为常加速度（CA）或转弯模型（CTRV）以提高运动拟合。
2. 引入多传感器融合（如雷达 + 里程计）。
3. 使用 NIS/NEES 做一致性检验和在线调参。
4. 与 UKF（无迹卡尔曼）或粒子滤波做对比实验。
5. 加入异常值抑制（门限、鲁棒损失、M-estimator）。

## R17

与标准 Kalman 滤波（KF）的差异总结:

- KF 假设 `f,h` 都线性，EKF 允许非线性。
- KF 的矩阵固定，EKF 每步都要在当前点计算 Jacobian。
- EKF 仅是一阶近似，精度依赖线性化点质量。
- 当系统本身线性时，EKF 会退化回普通 KF 形式。

本任务之所以选 EKF，是因为雷达观测 `range/bearing` 天然非线性。

## R18

`demo.py` 的源码级算法流（非黑箱，8 步）:

1. `run_ekf_demo` 设定 `dt / Q / R / 初始状态`，调用 `simulate_truth_and_measurements` 生成合成数据。
2. `simulate_truth_and_measurements` 中每步先用 `F` 与 `G` 推进真实状态，再由 `h_measurement` 生成无噪声观测并叠加噪声。
3. 主循环取当前观测 `z`，先调用 `ekf_predict`。
4. `ekf_predict` 内部调用 `f_state` 和 `jacobian_f`，计算 `x_pred` 与 `P_pred = FPF^T+Q`。
5. 回到主循环后调用 `ekf_update`。
6. `ekf_update` 内部先调用 `jacobian_h` 与 `h_measurement`，得到局部线性观测模型和预测观测。
7. `ekf_update` 计算创新 `y`（对角度分量 `wrap_angle`），再计算 `S`、`K`，更新状态 `x_upd`。
8. `ekf_update` 用 Joseph 形式更新 `P_upd` 并返回；主循环累计估计结果，最终计算并打印位置/速度 RMSE 与末状态对比。
