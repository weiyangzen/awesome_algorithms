# 线性响应理论 (Linear Response Theory)

- UID: `PHYS-0303`
- 学科: `物理`
- 分类: `统计力学`
- 源序号: `306`
- 目标目录: `Algorithms/物理-统计力学-0306-线性响应理论_(Linear_Response_Theory)`

## R01

线性响应理论研究“系统在弱外扰下的平均响应”。核心思想是：

- 外场足够弱时，响应可近似写成一阶（线性）项；
- 响应核可由平衡态涨落相关函数给出（涨落-耗散关系）。

本目录 MVP 选用 1D Ornstein-Uhlenbeck（OU）速度过程，直接做三方对照：

1. 数值直接受扰响应（施加小恒力）；
2. 基于平衡相关函数的线性响应预测；
3. OU 模型闭式解析解。

## R02

问题定义（本实现）：

- 输入：`mass, gamma, kbt, dt, n_steps, n_traj, force, seed`。
- 动力学：
  - 无扰动：`dv = -gamma * v dt + sqrt(2 gamma kBT / m) dW`
  - 受扰动：`dv = (-gamma * v + force / m) dt + sqrt(2 gamma kBT / m) dW`
- 输出：
  - `delta_direct(t) = <v>_pert - <v>_eq`
  - `delta_lrt(t) = (force/kBT) * ∫_0^t Cvv(s) ds`
  - `delta_analytic(t) = force/(m gamma) * (1-exp(-gamma t))`
  - 误差指标与采样表。

## R03

本 MVP 使用的关键公式：

1. 平衡相关函数（这里用两时点版本）：
   `Cvv(t) = <v(t) v(0)>_eq`。
2. 线性响应预测（FDT 形式）：
   `delta_lrt(t) = (force / kBT) * ∫_0^t Cvv(s) ds`。
3. OU 模型解析响应：
   `delta_analytic(t) = force/(m gamma) * (1 - exp(-gamma t))`。
4. 初值一致性检查：
   `Cvv(0) = <v^2> = kBT/m`。

以上关系让我们能在同一脚本中同时验证“数值动力学 -> 线性响应 -> 理论闭式解”的一致性。

## R04

适用前提与假设：

- 外场 `force` 足够小，线性近似成立；
- 初始分布来自平衡高斯分布；
- OU 过程满足平稳与马尔可夫条件；
- 以单自由度速度变量为示例，不含空间耦合与多体相互作用。

这使得代码简洁且可审计，重点突出线性响应理论本身，而非复杂系统细节。

## R05

数值离散与工程策略：

- 时间推进使用 Euler-Maruyama；
- 轨迹按“时间循环 + 轨迹向量化”实现；
- 无扰动与受扰动系统共用同一随机噪声（common random numbers），显著降低 `delta_direct` 估计方差；
- `Cvv(t)` 采用 `<v(t) v(0)>` 的 ensemble 估计（初态已在平衡分布）；
- 积分使用手写累计梯形法 `_cumulative_trapezoid`，避免黑箱积分器。

## R06

高层伪代码：

1. 读取并校验参数；
2. 从平衡高斯分布采样初始速度 `v_eq(0)`，复制得到 `v_pert(0)`；
3. 保存 `v0 = v_eq(0)` 作为相关函数参考；
4. 对每个时间步生成标准高斯噪声向量 `xi`；
5. 用同一 `xi` 更新 `v_eq` 与 `v_pert`；
6. 记录 `mean_eq(t)`、`mean_pert(t)`、`Cvv(t)=<v_eq(t)*v0>`；
7. 计算 `delta_direct`、`delta_lrt`、`delta_analytic`；
8. 汇总误差指标并输出表格 + JSON。

## R07

正确性直觉：

- 若线性响应成立，`delta_direct(t)` 应与 `delta_lrt(t)` 接近；
- 若 OU 离散误差可控且样本足够大，二者都应接近 `delta_analytic(t)`；
- `Cvv(0)` 应接近 `kBT/m`，否则初态平衡采样或噪声系数实现有误。

脚本内置两条 sanity check：

- `corr0` 相对误差不超过 10%；
- 末时刻 `direct` 与 `lrt` 相对误差不超过 25%。

## R08

复杂度分析（`N = n_steps`, `M = n_traj`）：

- 时间复杂度：`O(N * M)`（每步对长度 `M` 的向量做常数次运算）；
- 空间复杂度：`O(M + N)`（当前状态向量 + 时间序列统计量）；
- 不存储整块 `M x N` 轨迹矩阵，避免不必要内存占用。

## R09

主要误差来源：

- 统计误差：`n_traj` 太小时，均值与相关函数噪声大；
- 离散误差：`dt` 太大时 Euler-Maruyama 偏差上升；
- 截断误差：有限 `n_steps` 导致积分上限有限；
- 线性化误差：`force` 过大时高阶非线性响应不可忽略。

建议：先固定较小 `force`，再增加 `n_traj`、减小 `dt` 观察收敛。

## R10

`demo.py` 模块说明：

- `LinearResponseConfig`：集中管理物理与数值参数；
- `_validate_config`：参数合法性检查；
- `_simulate_ensemble_response`：核心动力学推进与统计采样；
- `_cumulative_trapezoid`：累计梯形积分；
- `run_linear_response_demo`：生成响应曲线与误差指标；
- `main`：执行 sanity check 并打印结果。

第三方库边界：

- `numpy` 仅用于数组与随机数；
- `pandas` 仅用于展示采样表；
- 不依赖任何“线性响应现成求解器”黑箱。

## R11

运行方式（无交互输入）：

```bash
uv run python Algorithms/物理-统计力学-0306-线性响应理论_(Linear_Response_Theory)/demo.py
```

程序会打印：

- 8 个时间点的对照表（direct / lrt / analytic）；
- 一段 JSON 指标摘要。

## R12

输出字段解释：

- `corr0_expected`：理论 `kBT/m`；
- `corr0_measured`：数值测得 `Cvv(0)`；
- `final_direct_delta`：末时刻直接响应；
- `final_lrt_prediction`：末时刻线性响应预测；
- `final_analytic`：末时刻解析值；
- `mae_direct_vs_lrt`：全时域平均绝对误差；
- `max_err_direct_vs_lrt`：全时域最大绝对误差；
- `mae_direct_vs_analytic`：direct 与解析均值误差；
- `final_rel_err_lrt`：末时刻 direct 相对 lrt 的误差。

## R13

最小验证清单：

1. `uv run python demo.py` 能直接完成，无交互；
2. 输出包含“Sample Table”和“JSON Summary”；
3. `corr0_measured` 与 `corr0_expected` 接近；
4. `final_rel_err_lrt` 通过阈值检查；
5. 适度增大 `n_traj` 后误差总体下降。

## R14

边界与失败场景：

- `mass/gamma/kbt/dt <= 0`：物理或数值参数非法；
- `n_steps < 2`：无法形成时间积分；
- `n_traj` 过小：统计噪声明显，断言可能失败；
- `force` 过大：线性响应前提被破坏；
- `dt` 过大：离散偏差导致与解析解偏离。

代码对前四类有显式参数校验或运行时 sanity check。

## R15

参数调优建议：

- 想更快：减小 `n_steps` 或 `n_traj`；
- 想更准：增大 `n_traj`、减小 `dt`；
- 想强调线性区：减小 `force`；
- 想更长时间平台：增大 `n_steps`。

经验上，`n_traj=5000, dt=0.002, n_steps=2500` 能在速度和稳定性间取得平衡。

## R16

与久保/Green-Kubo关系：

- 本实现是“时域响应版本”的线性响应演示；
- 若把观测量换成输运流并对相关函数积分，即得到 Green-Kubo 型输运系数估计；
- 因此本目录与 `Kubo Formula` 目录是同一理论框架下的两个切面：
  - 这里关注 `delta <B>(t)`；
  - Kubo 目录关注稳态输运系数（如扩散系数/电导率）。

## R17

可扩展方向：

1. 改成脉冲或正弦外场，验证卷积型响应核；
2. 从速度响应扩展到位置、能量流等观测量；
3. 使用 FFT 估计频域响应函数 `chi(omega)`；
4. 增加 block averaging / bootstrap 误差条；
5. 扩展为多粒子或耦合自由度系统。

## R18

`demo.py` 源码级算法流（8 步，非黑箱）：

1. `main` 构造 `LinearResponseConfig`，调用 `run_linear_response_demo`。  
2. `run_linear_response_demo` 先执行 `_validate_config`，拒绝非法参数。  
3. 进入 `_simulate_ensemble_response`：按 `kBT/m` 采样平衡初值 `v_eq(0)`，复制为 `v_pert(0)`，并保存 `v0`。  
4. 对每个时间步生成噪声向量 `xi`，构造 `noise = sqrt(2 gamma kBT / m * dt) * xi`。  
5. 用同一 `noise` 更新无扰动与受扰动系统（Euler-Maruyama），记录 `mean_eq`、`mean_pert`、`Cvv(t)=<v_eq(t)v0>`。  
6. 回到 `run_linear_response_demo` 计算 `delta_direct = mean_pert - mean_eq`。  
7. 调用 `_cumulative_trapezoid(Cvv, dt)` 并乘 `force/kBT` 得 `delta_lrt`，同时按 OU 闭式公式计算 `delta_analytic`。  
8. 计算误差指标、构造 `pandas` 采样表并输出；`main` 再做两条 sanity check 后打印表格与 JSON。  

第三方库仅承担基础数值与表格展示，不替代核心算法步骤。
