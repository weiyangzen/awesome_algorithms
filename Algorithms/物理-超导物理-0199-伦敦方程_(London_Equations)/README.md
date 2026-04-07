# 伦敦方程 (London Equations)

- UID: `PHYS-0198`
- 学科: `物理`
- 分类: `超导物理`
- 源序号: `199`
- 目标目录: `Algorithms/物理-超导物理-0199-伦敦方程_(London_Equations)`

## R01

伦敦方程（London Equations）是超导体电磁响应的最小现象学模型，核心是两条关系：

1. 第一伦敦方程：`dJ/dt = (n_s e^2 / m) E`
2. 第二伦敦方程：`curl J = -(n_s e^2 / m) B`

其中 `J` 为超电流密度，`n_s` 为超流电子密度。它直接解释了超导体的零电阻（稳态持久电流）和迈斯纳效应（磁场排斥）。

## R02

在超导物理中，伦敦方程的价值是把“宏观电磁现象”变成可计算的方程组：

- 由第二伦敦方程与麦克斯韦方程联立可得 `nabla^2 B = B / lambda_L^2`；
- `lambda_L`（伦敦穿透深度）控制磁场在超导体中的指数衰减尺度；
- 第一伦敦方程给出电场激发后的无耗散电流演化。

因此它是 Ginzburg-Landau 与 BCS 之前/之外最常用的工程级基线模型。

## R03

本目录 MVP 的目标是做一个可运行、可校验、非黑盒的最小实现：

- 数值求解一维 `B'' = B/lambda_L^2`（有限差分）；
- 与解析解 `B(x)=B0 exp(-x/lambda_L)` 对比，验证迈斯纳屏蔽；
- 对带噪声的 `B(x)` 合成数据拟合 `lambda_L`；
- 用第一伦敦方程积分 `J(t)`，演示脉冲电场后的持久电流。

## R04

本实现用到的关键公式：

1. `lambda_L^2 = m / (mu0 n_s e^2)`
2. `B'' = B/lambda_L^2`
3. 半无限体解析解：`B(x) = B0 exp(-x/lambda_L)`
4. 一维下由 `curl B = mu0 J` 得 `J = -(1/mu0) dB/dx`
5. 第一伦敦方程离散化（Euler）：`J[k] = J[k-1] + alpha E[k-1] dt`

其中 `alpha = n_s e^2 / m`，该脚本由 `lambda_L` 反推 `n_s` 保持参数自洽。

## R05

复杂度（设空间网格数 `N`，时间步数 `T`）：

- 有限差分线性系统求解：`O(N^3)`（当前直接 dense `solve`，`N=240` 时很快）；
- 一阶时间积分：`O(T)`；
- 参数拟合（非线性最小二乘）：近似 `O(kN)`，`k` 为迭代步。

在默认参数下，脚本可在普通 CPU 上快速完成。

## R06

`demo.py` 的输出包括：

- 核心指标表：
  - 有限差分解 vs 解析解相对误差；
  - 第二伦敦方程残差；
  - `lambda_L` 拟合值与相对误差；
  - 脉冲后电流终值与漂移幅度。
- 8 个采样点的 `x, B_fd, B_analytic, abs_diff` 对照表。
- 最后给出 `All sanity checks passed.` 作为可执行验收标记。

## R07

优点：

- 公式到代码映射直接，便于审查；
- 同时覆盖第一、第二伦敦方程；
- 拟合流程显式可读，不依赖封装好的超导专用黑盒。

局限：

- 只做一维、各向同性、局域伦敦模型；
- 未包含临界场、涡旋、非局域效应；
- 噪声数据为合成数据，不是实验实测。

## R08

前置知识与环境：

- 前置知识：基础电磁学、常微分/偏微分方程离散化、最小二乘拟合。
- Python：`>=3.10`
- 依赖：`numpy`、`scipy`、`pandas`

本 MVP 使用最小工具栈；核心方程离散和残差计算都在源码中显式实现。

## R09

适用场景：

- 迈斯纳效应的一维教学演示；
- 穿透深度 `lambda_L` 的快速原型估计；
- 后续更复杂超导模型的 baseline。

不适用场景：

- 需要二维/三维磁场分布与真实边界几何；
- 涉及强非线性、涡旋态、时间依赖 GL 方程；
- 直接替代实验数据反演管线。

## R10

正确性直觉：

1. 若第二伦敦方程成立，`B` 应在尺度 `lambda_L` 上指数衰减；
2. 有限差分与解析解一致，说明离散方程实现正确；
3. `dJ/dx + B/(mu0 lambda_L^2)` 接近零，说明二阶关系闭环；
4. 第一伦敦方程下，在 `E=0` 阶段 `dJ/dt=0`，电流应保持常值。

脚本断言对应这四条可执行物理检查。

## R11

数值稳定策略：

- 二阶导数残差在内部点（去边界 2 点）计算，减小边缘差分误差；
- 参数拟合添加边界约束 `lambda in [1e-10, 1e-5]`，避免无意义解；
- 使用固定随机种子，确保可重复；
- 通过 `assert` 给出显式失败条件，避免静默错误。

## R12

关键参数（`LondonConfig`）：

- `lambda_l`：真实穿透深度；
- `b0_surface`：表面磁场；
- `domain_factor`：计算域长度 `L = domain_factor * lambda_l`；
- `n_grid`：空间离散精度；
- `noise_std`：合成观测噪声；
- `e_pulse`、`dt`、`n_steps`、`pulse_steps`：第一伦敦方程时域模拟设置。

调参建议：

- 提高 `n_grid` 可降低空间离散误差；
- 增大 `domain_factor` 可更接近半无限体近似；
- 减小 `noise_std` 可提升 `lambda_L` 拟合稳定性。

## R13

保证类型说明：

- 近似比保证：N/A（非组合优化问题）。
- 随机成功率保证：N/A（主流程确定性；仅噪声生成受固定种子控制）。

本 MVP 的可执行保证是：

- 无交互输入；
- 可完整输出指标与样例表；
- 关键物理关系通过内置断言自动验证。

## R14

常见失效模式：

1. `lambda_l` 单位误用（nm 未转 m）；
2. 网格过粗导致二阶导数残差偏大；
3. 噪声设置过大导致拟合偏差超过断言阈值；
4. `pulse_steps >= n_steps` 导致时域模拟配置非法；
5. 把 `J = -(1/mu0) dB/dx` 的符号写反。

脚本对第 4 点做了显式输入检查。

## R15

可扩展方向：

1. 用三对角求解器替代 dense 求解，提高大网格效率；
2. 从一维扩展到二维有限元/有限差分；
3. 增加实验数据读取与误差条估计；
4. 纳入温度依赖 `lambda_L(T)`；
5. 连接 Ginzburg-Landau 参数（如 `kappa = lambda/xi`）做多模型对照。

## R16

相关主题：

- 迈斯纳效应（Meissner Effect）
- Ginzburg-Landau 理论
- BCS 理论与超流密度
- 穿透深度测量与参数反演
- 伦敦极限与非局域电磁响应

## R17

运行方式（无交互）：

```bash
cd Algorithms/物理-超导物理-0199-伦敦方程_(London_Equations)
uv run python demo.py
```

交付核对：

- `README.md` 的 `R01-R18` 已完整填写；
- `demo.py` 可直接运行并输出结果；
- `meta.json` 与任务元数据保持一致；
- 目录可独立验证。

## R18

`demo.py` 源码级算法流程（9 步）：

1. `main` 读取 `LondonConfig`，设定 `lambda_L`、表面磁场、网格和时域参数。  
2. 由 `length = domain_factor * lambda_L` 构造一维计算域。  
3. `solve_second_london_fd` 对 `B'' = B/lambda_L^2` 建立有限差分线性系统，并施加 `B(0)=B0` 与 `B(L)=B0 exp(-L/lambda_L)` 两端边界。  
4. 通过 `np.linalg.solve` 得到离散 `B_fd(x)`，并与解析 `B_ref(x)=B0 exp(-x/lambda_L)` 计算相对误差。  
5. `second_london_residual` 先由 `J=-(1/mu0)dB/dx` 得电流，再检验 `dJ/dx = -B/(mu0 lambda_L^2)` 的离散残差。  
6. 在 `B_ref` 上叠加高斯噪声得到合成观测 `B_obs`。  
7. `estimate_lambda_from_data` 调用 `scipy.optimize.curve_fit` 对 `B(x)=B0 exp(-x/lambda)` 做有界拟合，估计 `lambda_hat`。  
8. `simulate_first_london_current` 用 Euler 积分第一伦敦方程 `dJ/dt = alpha E(t)`，其中 `alpha` 由 `lambda_L -> n_s -> alpha` 自洽推导；脉冲结束后检查电流漂移。  
9. `main` 汇总指标为 `pandas.DataFrame`，打印样例剖面表，并执行断言作为自动验收。

第三方库拆解说明：`numpy` 用于网格、差分和线性代数；`scipy` 只用于物理常数与非线性拟合；`pandas` 仅用于终端表格展示。核心伦敦方程离散流程、残差定义和时域推进均在源码中显式实现，不是黑盒调用。
