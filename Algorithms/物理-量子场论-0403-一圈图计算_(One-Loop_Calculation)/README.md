# 一圈图计算 (One-Loop Calculation)

- UID: `PHYS-0384`
- 学科: `物理`
- 分类: `量子场论`
- 源序号: `403`
- 目标目录: `Algorithms/物理-量子场论-0403-一圈图计算_(One-Loop_Calculation)`

## R01

一圈图计算（One-Loop Calculation）是量子场论微扰展开中的第一层量子修正。  
本条目给出一个最小可运行 MVP：计算等质量标量泡图（bubble）在欧氏动量下的重整化有限主积分，并同时提供数值积分、解析闭式、低能展开三种结果交叉验证。

## R02

本目录要解决的问题：

1. 给出可直接数值求值的一圈主积分定义；
2. 实现该积分的解析闭式，避免只停留在黑盒数值；
3. 对比低能 `Q^2 << m^2` 级数展开，展示近似有效区间；
4. 输出结构化结果表并附带自动断言，确保 `uv run python demo.py` 可稳定复现。

## R03

本 MVP 采用的主积分（已在 `Q^2=0` 处做减法）是：

`B_hat(Q^2,m) = (1/(16*pi^2)) * ∫_0^1 dx ln(1 + x(1-x) Q^2/m^2)`。

其中：

- `Q^2 >= 0`：欧氏外动量平方；
- `m > 0`：回路内部粒子质量；
- 积分维度被 Feynman 参数化压缩为 1 维，便于审计实现。

## R04

引入无量纲变量 `a = Q^2/m^2`，可写成

`B_hat = J(a)/(16*pi^2)`，
`J(a) = ∫_0^1 ln(1 + a x(1-x)) dx`。

解析闭式：

`J(a) = 2 * ( sqrt((a+4)/a) * artanh(sqrt(a/(a+4))) - 1 ), a>0`，
`J(0)=0`。

`demo.py` 中 `bubble_master_analytic` 与 `_j_analytic_dimensionless` 即对应这条公式。

## R05

低能展开（`a << 1`）为：

`J(a) = a/6 - a^2/60 + O(a^3)`，
`B_hat(Q^2,m) = (1/(16*pi^2)) * (a/6 - a^2/60 + O(a^3))`。

这一步可用于：

- 快速估计小动量区域的一圈修正；
- 与精确值比对近似误差，验证程序行为符合理论预期。

## R06

算法实现分三条并行计算链：

1. 数值链：`scipy.integrate.quad` 对 `x in [0,1]` 积分；
2. 解析链：使用闭式公式直接计算；
3. 近似链：使用 `O(Q^4)` 级数展开。

三条链在同一组 `Q^2` 网格上输出并交叉校验。

## R07

脚本输出的物理量与辅助量：

- `B_numeric`：数值积分得到的主积分；
- `B_analytic`：闭式解析值；
- `B_series_O(q4)`：低能截断展开值；
- `delta_lambda = 3 * lambda^2 * B_hat`：示例化的一圈耦合修正尺度（toy 组合）；
- 各类误差列：用于自动审计。

## R08

复杂度（`N` 为 `Q^2` 采样点数量）：

- 每个点一次 1D `quad` 积分，成本记为 `Q`；
- 总时间复杂度约 `O(N*Q)`；
- 闭式与级数计算总计 `O(N)`；
- 空间复杂度 `O(N)`（以 `pandas.DataFrame` 保存结果）。

## R09

关键数据结构：

- `OneLoopConfig`：封装 `mass`、`coupling`、`q2_values`、积分容差；
- `pandas.DataFrame`：统一承载每个 `Q^2` 处的数值值、解析值、级数值和误差；
- 列字段包括：
  `q2`, `a=q2/m^2`, `B_numeric`, `B_analytic`, `B_series_O(q4)`, `delta_lambda` 及误差列。

## R10

边界与异常处理：

1. `mass<=0`、`coupling<=0`、`epsabs/epsrel<=0` 直接报错；
2. `q2_values` 必须是一维、有限、非负、非降序；
3. `q2=0` 单独走解析极限返回 0，避免数值微小噪声；
4. 小 `a` 区间解析函数切换到级数分支，避免 `atanh` 组合造成消差。

## R11

运行方式（无交互）：

```bash
cd Algorithms/物理-量子场论-0403-一圈图计算_(One-Loop_Calculation)
uv run python demo.py
```

或从仓库根目录运行：

```bash
uv run python Algorithms/物理-量子场论-0403-一圈图计算_(One-Loop_Calculation)/demo.py
```

## R12

输出解释：

1. 头部打印参数：`m`、`lambda` 与积分定义；
2. 主表按 `Q^2` 给出三套计算值及误差；
3. `abs_err_num_vs_ana` 用于验证数值积分实现是否正确；
4. `rel_err_series_vs_ana` 用于评估低能展开在当前动量点的可用性；
5. 若全部断言通过，最后打印 `All checks passed.`。

## R13

正确性验证（脚本内置断言）：

1. `max |B_numeric - B_analytic| < 1e-12`；
2. `B_hat(0,m)=0`（重整化减法定义）；
3. `B_hat` 对非负欧氏 `Q^2` 单调不减；
4. 小 `Q^2` 区域（本例 `Q^2<=0.1`）级数相对误差小于阈值。

这些断言同时检查“数值实现正确”和“物理趋势正确”。

## R14

参数调节建议：

- 想压低数值误差：减小 `epsabs`/`epsrel`；
- 想增强低能展开对比：在 `q2_values` 增加更小的 `Q^2` 点；
- 想观察高能行为：把 `q2_values` 延伸到更大值并关注级数误差快速增大；
- 想映射不同理论尺度：调整 `mass` 与 `coupling`。

## R15

和其他实现方式的比较：

- 直接 4 维动量积分：更贴原始图表达，但数值实现更重；
- 本实现（参数化后 1 维积分）：易复现、可审计、适合算法条目 MVP；
- 全黑盒振幅库：开发快，但不利于教学与源码级理解。

本条目刻意选择“少而透明”的路径。

## R16

典型应用场景：

1. 量子场论课程中演示一圈图从积分到可计算公式的落地；
2. 更复杂圈图代码的单元测试基准（先对照一圈主积分）；
3. 低能有效理论里估计圈修正量级；
4. 为后续重整化群/跑动耦合条目提供可复用计算模块。

## R17

可扩展方向：

1. 推广到不等质量情形 `m1 != m2`；
2. 加入 Minkowski 区域与阈值/虚部处理；
3. 扩展到完整 Passarino-Veltman 函数族（`A0/B0/C0/...`）；
4. 把 `Q^2` 扫描自动导出 CSV 与绘图；
5. 加入不同重整化方案（MS-bar、on-shell）对比。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 构造 `OneLoopConfig`，固定 `m`、`lambda`、`Q^2` 网格与积分容差。  
2. `build_report` 逐点遍历 `Q^2`，先调用 `bubble_master_numeric` 做 1D Feynman 参数数值积分。  
3. 同一点调用 `bubble_master_analytic`，内部使用 `_j_analytic_dimensionless` 的闭式 `atanh` 公式。  
4. 调用 `bubble_master_series_q2` 计算 `O(Q^4)` 低能展开近似。  
5. 调用 `one_loop_coupling_shift` 将 `B_hat` 映射为示例耦合修正 `delta_lambda`。  
6. 将上述标量与误差项写入 `DataFrame` 行，形成可审计结果表。  
7. `main` 对结果执行三类断言：数值-解析一致性、`Q^2=0` 与单调性、低能展开准确性。  
8. 断言全部通过后打印 `All checks passed.` 并结束。

说明：`scipy.quad` 只用于通用数值积分；一圈图的物理定义、闭式表达、级数展开与验证逻辑都在源码中显式实现，不是黑盒一键求解。
