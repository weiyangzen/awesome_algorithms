# 切伦科夫辐射 (Cherenkov Radiation)

- UID: `PHYS-0186`
- 学科: `物理`
- 分类: `电动力学`
- 源序号: `187`
- 目标目录: `Algorithms/物理-电动力学-0187-切伦科夫辐射_(Cherenkov_Radiation)`

## R01

切伦科夫辐射指带电粒子在介质中运动速度超过该介质中的光相速度时产生的相干电磁辐射。阈值条件是：

`beta * n(lambda) > 1`

其中 `beta=v/c`，`n(lambda)` 是折射率。满足条件时，辐射锥角满足：

`cos(theta_c) = 1 / (beta * n(lambda))`

## R02

在电动力学里，切伦科夫辐射是“介质电磁响应 + 相位匹配”共同作用的典型现象：
- 在粒子探测中用于速度判别（RICH、DIRC、阈值切伦科夫计数器）；
- 在核反应堆水池中产生可见蓝光；
- 在高能天体物理中用于地面大气切伦科夫望远镜信号重建。

它把“粒子动力学参数”转化为“可观测的光谱与角分布”。

## R03

本条目 MVP 目标是构建一个最小、可审计的数值闭环：
1. 在给定波长区间构建弱色散介质 `n(lambda)`；
2. 用 Frank-Tamm 公式直接计算 `d^2N/(dx d lambda)`；
3. 显式施加阈值掩码 `beta*n(lambda) > 1`；
4. 积分得到总光子产额 `dN/dx`；
5. 校验常折射率时数值积分与解析积分一致；
6. 输出不同 `beta` 的产额与平均辐射角。

## R04

`demo.py` 使用的核心方程：

1. 阈值条件：`beta*n(lambda)>1`。
2. 切伦科夫角：`theta(lambda)=arccos(1/(beta*n(lambda)))`。
3. Frank-Tamm 光子谱：
   `d^2N/(dx d lambda) = 2*pi*alpha*(1/lambda^2)*(1 - 1/(beta^2*n(lambda)^2))`。
4. 条件不满足时谱密度设为 `0`（物理上无辐射）。
5. 总产额：`dN/dx = ∫[lambda_min, lambda_max] d^2N/(dx d lambda) d lambda`。
6. 常折射率解析积分：
   `dN/dx = 2*pi*alpha*(1 - 1/(beta^2*n^2))*(1/lambda_min - 1/lambda_max)`。

## R05

设波长网格点数为 `m`，速度扫描点数为 `k`：
- 折射率构造：`O(m)`；
- 单个 `beta` 的谱计算与积分：`O(m)`；
- 全部扫描：`O(k*m)`。

本 MVP 中 `m=6001`，`k=4`，整体复杂度很小，适合快速回归验证。

## R06

脚本输出两组表格指标：
- `checks`：
  - 低于阈值时是否严格零辐射；
  - 自动构造的高于阈值 `beta` 下全波段是否都满足阈值；
  - 常折射率数值积分与解析积分相对误差；
  - `dN/dx` 随 `beta` 增大是否单调上升；
  - 色散折射率最小/最大值。
- `beta sweep`：每个 `beta` 对应的 `dN/dx`、平均角、最大角、有效波段占比。

运行结束由 `assert` 给出通过/失败，无需人工交互。

## R07

优点：
- 公式与代码一一对应，可直接审计；
- 同时覆盖阈值行为、谱积分、角分布与解析对照；
- 无黑盒仿真器依赖，便于教学与单元测试。

局限：
- 仅含单粒子、均匀介质、直线匀速近似；
- 未包含吸收、散射、探测器响应与量子效率；
- 未模拟实际光子传输几何（镜面、像面、时间分辨）。

## R08

前置知识：
- 麦克斯韦方程与介质折射率概念；
- 切伦科夫阈值与辐射锥角几何；
- 基本数值积分（梯形法）与数组运算。

运行依赖：
- Python `>=3.10`
- `numpy`
- `pandas`

## R09

适用场景：
- 快速估算给定波段下的切伦科夫光子产额数量级；
- 做探测器概念设计前的参数扫描（`beta`、波段、折射率模型）；
- 作为更复杂 Geant4/全链路仿真的前置 sanity check。

不适用场景：
- 需要精确到探测器像面、时间分辨和统计涨落的工程分析；
- 强吸收/强散射介质中的传播建模；
- 多粒子簇射、非均匀介质与复杂边界条件。

## R10

正确性直觉：
1. 若 `beta*n<=1`，相位无法形成稳定辐射锥，应无辐射；
2. 进入可辐射区后，`beta` 越大，`1-1/(beta^2 n^2)` 越大，总产额上升；
3. 谱密度有 `1/lambda^2` 权重，所以短波贡献更强；
4. 常折射率时积分有闭式解，数值结果应贴近解析结果。

这四点是脚本里断言逻辑的物理来源。

## R11

数值稳定与精度措施：
- 使用 6001 点均匀波长网格降低积分离散误差；
- 对阈值判断采用显式布尔掩码，避免非法 `arccos` 输入；
- `arccos` 参数使用 `clip` 到 `[-1,1]` 防止浮点越界；
- 解析对照采用相对误差阈值（`5e-5`）进行自动验收。

## R12

关键参数（`CherenkovConfig`）：
- `lambda_min_nm`, `lambda_max_nm`：谱积分波段；
- `n_lambda_samples`：积分离散精度；
- `n0`, `cauchy_b_um2`：Cauchy 色散模型参数；
- `betas_for_sweep`：速度扫描点。

调参建议：
- 要提高积分精度优先增大 `n_lambda_samples`；
- 更接近真实介质时，用实测 `n(lambda)` 替换 Cauchy 模型；
- 若研究阈值附近行为，应加密 `beta` 扫描点。

## R13

理论保证类型说明：
- 近似比保证：N/A（非优化问题）。
- 随机成功率保证：N/A（全流程确定性计算）。

本 MVP 的保证来自“数值一致性 + 解析一致性”：
- 阈值下严格零辐射；
- 常折射率下数值积分贴合解析积分；
- 扫描中 `dN/dx` 对 `beta` 单调增加。

## R14

常见失效模式：
1. 把阈值写成 `beta>1/n` 但忽略 `n(lambda)` 色散；
2. 单位混乱（nm 与 m 混用）导致谱值数量级错误；
3. 直接对所有波长调用 `arccos` 产生 `nan` 或非法值；
4. 将 `1/lambda^2` 错写为 `1/lambda`；
5. 只看单点角度，不做总产额积分，导致设计偏差。

## R15

可扩展方向：
- 引入介质吸收长度与探测器量子效率，计算可探测光子数；
- 把单粒子扩展到粒子能谱分布并卷积得到事件级光子统计；
- 加入几何光学传播，估计光子在探测面上的环形像；
- 与实验标定数据拟合 `n(lambda)` 或有效探测窗口。

## R16

相关主题：
- Frank-Tamm 理论；
- RICH/DIRC 粒子鉴别；
- 电磁辐射中的相干辐射机制；
- 色散关系与群速度/相速度区分。

## R17

`demo.py` 的 MVP 清单：
- `refractive_index_cauchy`：构建可计算的 `n(lambda)`；
- `cherenkov_condition`：逐波长阈值判定；
- `cherenkov_angle_rad`：生成有效波段辐射角；
- `frank_tamm_photon_density`：实现 Frank-Tamm 光子谱；
- `analytic_constant_index_yield`：常折射率解析积分；
- `run_demo`：组织断言与报表输出。

运行方式：

```bash
cd "Algorithms/物理-电动力学-0187-切伦科夫辐射_(Cherenkov_Radiation)"
uv run python demo.py
```

脚本无交互输入，成功时打印 `All checks passed.`。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `build_wavelength_grid` 生成 `[lambda_min, lambda_max]` 的高分辨率波长网格（单位 m），并做基础合法性检查。  
2. `refractive_index_cauchy` 在该网格上显式计算色散折射率 `n(lambda)=n0+b/lambda_um^2`。  
3. `cherenkov_condition` 逐点判断 `beta*n(lambda)>1`，得到可辐射波段掩码。  
4. `frank_tamm_photon_density` 按 Frank-Tamm 公式计算 `d^2N/(dx d lambda)`，并在掩码外强制置零。  
5. `integrate_photons_per_meter` 用 `np.trapezoid` 将谱密度积分成总产额 `dN/dx`。  
6. `analytic_constant_index_yield` 给出常折射率闭式结果，`run_demo` 将其与第 5 步数值积分做相对误差校验。  
7. `cherenkov_angle_rad` 在有效波段计算 `theta(lambda)=arccos(1/(beta*n(lambda)))`，并提取均值/最大角与有效波段占比。  
8. `run_demo/main` 汇总阈值、解析一致性、单调性与角度范围的断言结果，用 `pandas` 组织打印，形成可复现、可自动验收的非交互 MVP。  

第三方库仅承担数组与表格操作，物理核心流程（阈值、角度、谱、积分、解析对照）均在源码中逐步展开。
