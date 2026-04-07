# 重整化 (Renormalization)

- UID: `PHYS-0388`
- 学科: `物理`
- 分类: `量子场论`
- 源序号: `407`
- 目标目录: `Algorithms/物理-量子场论-0407-重整化_(Renormalization)`

## R01

重整化（Renormalization）是把“发散的裸参数”重新组织成“可测、有限、与实验标定点一致的参数”的程序。

在量子场论中，环图常带来 UV 发散；重整化并不是把发散“忽略”，而是显式引入正则化与反项，让预测量在选定的重整化条件下可计算、可比对、可迁移到别的能标。

## R02

本条目用一个最小可运行的 `phi^4` 一圈模型回答三个具体问题：

1. 给定重整化条件（`m_R`, `lambda_R`, `mu_sub`）时，裸参数 `m_B^2`, `lambda_B` 如何随截断 `Lambda` 变化；
2. 用每个 `Lambda` 对应的反项后，重整化后的四点耦合 `lambda_eff(p)` 是否对 `Lambda` 不敏感；
3. 若错误地把 bare 参数固定不动，会出现多大的截断依赖。

## R03

`demo.py` 的输入输出约定（无交互）：

- 输入（脚本内固定）：
1. `m_R=1.0`，`lambda_R=0.20`，`mu_sub=2.0`；
2. 截断扫描：`Lambda in {20,40,80,160,320,640}`；
3. 探测动量：`p in {0,2,5,10}`（欧氏动量模）。
- 输出：
1. 每个 `Lambda` 的 `bubble_sub`、`lambda_B`、`m_B^2`、减除点重构误差；
2. 采用“逐截断反项”时的 `lambda_eff(p)` 透视表；
3. “固定 bare 参数”的对照表；
4. 各 `p` 的 cutoff spread 汇总；
5. 断言通过后输出 `All checks passed.`。

## R04

MVP 的理论模型与公式（欧氏、硬截断）：

1. tadpole 积分：
`I_tad(Lambda,m) = [Lambda^2 - m^2 ln(1 + Lambda^2/m^2)] / (16 pi^2)`。

2. bubble 积分（费曼参数）：
`B(Lambda,m,p) = (1/(16 pi^2)) * int_0^1 dx [ln(1 + Lambda^2/M^2) - Lambda^2/(Lambda^2 + M^2)]`，
其中 `M^2 = m^2 + x(1-x)p^2`。

3. 一圈重整化条件（四点函数减除点 `p=mu_sub`）：
`lambda_R = lambda_B - 3 lambda_B^2 B(Lambda,m_R,mu_sub)`。

4. 两点函数质量条件（`p=0`）：
`m_R^2 = m_B^2 + (lambda_B/2) I_tad(Lambda,m_R)`，
故 `m_B^2 = m_R^2 - (lambda_B/2) I_tad(...)`。

5. 任意探测动量下的一圈有效耦合：
`lambda_eff(p) = lambda_B - 3 lambda_B^2 B(Lambda,m_R,p)`。

## R05

复杂度（设截断个数 `N_L`，探测动量个数 `N_p`，`quad` 自适应采样点均值为 `K`）：

- 每次 bubble 计算约 `O(K)`；
- 裸参数扫描约 `O(N_L * K)`；
- `lambda_eff` 扫描约 `O(N_L * N_p * K)`；
- 表格聚合约 `O(N_L * N_p)`；
- 空间复杂度约 `O(N_L * N_p)`。

这里主耗时是 `scipy.integrate.quad` 对费曼参数积分的自适应求积。

## R06

算法流程（本仓库实现）：

1. 用解析式计算 tadpole；
2. 用 `quad` 计算 bubble；
3. 在每个 `Lambda` 上解方程 `lambda_R = lambda_B - 3 lambda_B^2 B_sub` 得 `lambda_B`；
4. 根据质量条件求 `m_B^2`；
5. 在多个探测动量计算 `lambda_eff(p)`；
6. 再做一组“固定 bare 参数”对照扫描；
7. 输出三类表格（bare、renorm、naive）；
8. 断言减除点一致性与 cutoff 敏感度对比。

## R07

优点：

- 重整化关键机制是显式实现，不依赖单句理论描述；
- 同时展示“正确重整化”和“错误固定 bare”两条路径；
- 输出是结构化表格，便于自动检查与后续扩展。

局限：

- 仅一圈、仅 `phi^4`、仅单参数减除方案；
- 未包含波函数重整化、scheme 变换与多耦合联立 RG；
- 主要用于机制演示，不是精密现象学计算。

## R08

前置知识与环境：

- 量子场论中的正则化、反项、重整化条件；
- 一元求根与数值积分；
- Python `>=3.10`；
- 依赖：`numpy`, `pandas`, `scipy`。

## R09

适用场景：

- 课程或讨论中快速演示“裸参数吸收发散”；
- 对比不同 cutoff 下的重整化前后差异；
- 作为后续实现 Callan-Symanzik / RG 流的起点。

不适用场景：

- 需要多圈高精度 beta 函数和匹配条件；
- 需要非微扰结果（格点、Schwinger-Dyson 等）；
- 需要与真实实验数据直接拟合的生产级流程。

## R10

正确性直觉：

1. 截断增大时，裸参数会“漂移”，这是反项吸收 UV 结构的预期现象；
2. 在减除点 `mu_sub`，`lambda_eff(mu_sub)` 必须回到输入的 `lambda_R`；
3. 对其他 `p`，若重整化做对，跨 cutoff 的 spread 应显著小于 naive 固定 bare；
4. 若不重整化，`Lambda` 依赖会直接污染可比预测量。

## R11

数值稳定策略：

- 使用 `log1p` 避免 `ln(1+x)` 在小 `x` 下的精度损失；
- `quad` 设定较严格 `epsabs/epsrel`；
- `lambda_B` 求解先走 Newton，失败后回退 Brent 有界根查找；
- 对判别式 `1-12*B*lambda_R` 做先验检查，避免进入无实根区；
- 全流程用断言固定关键物理条件。

## R12

关键参数与影响：

- `lambda_R`：越大越可能逼近 `1-12*B*lambda_R=0`，导致微扰根失效；
- `mu_sub`：定义重整化条件所在标定点；
- `cutoffs`：决定裸参数漂移范围与剩余 cutoff 敏感度；
- `probe_momenta`：决定观察到的动量依赖形态；
- 求积/求根容差：影响运行时间与误差下界。

调参建议：

- 出现无实根时优先降低 `lambda_R` 或减小 cutoff 范围；
- 要更快可先放宽积分容差，再检查断言余量；
- 要更平滑的曲线可加密 `probe_momenta` 网格。

## R13

- 近似比保证：N/A（非组合优化条目）。
- 随机成功率保证：N/A（全流程确定性）。

本实现可验证保证：

- 每个 cutoff 上 `|lambda_eff(mu_sub)-lambda_R| < 1e-10`；
- 本参数下 `lambda_B` 随 cutoff 增大而上升；
- 本参数下 `m_B^2` 随 cutoff 增大而下降；
- 重整化方案的 cutoff spread 明显小于 naive 固定 bare 方案。

## R14

常见失效模式：

1. 选了过大 `lambda_R` 使判别式非正（无微扰实根）；
2. 忘记在每个 cutoff 重新解 bare 参数，导致假“重整化”；
3. 把欧氏 `p` 与 Minkowski 变量混用；
4. 积分容差过松导致减除点误差累积；
5. 把质量条件与四点条件使用的参数混搭（`m_R`/`m_B` 不一致）。

## R15

可扩展方向：

- 引入 `Z_phi` 与动量依赖自能，加入波函数重整化；
- 从本减除方案扩展到 `MS-bar` 并对比 scheme 依赖；
- 把离散 `p` 扫描改为连续网格并导出 CSV/图像；
- 增加两圈近似或与数值 RG 方程联立。

## R16

相关条目：

- 正则化（Regularization）；
- 反项与重整化条件；
- 重整化群方程（RGE）；
- beta 函数与跑动耦合；
- 渐近自由与 Landau pole。

## R17

`demo.py` 交付能力：

- 显式实现 tadpole / bubble 一圈积分；
- 显式求解 `lambda_B` 与 `m_B^2`；
- 输出“重整化 vs naive”双通道比较结果；
- 内置断言检查减除点与 cutoff 敏感度；
- 无交互、单命令可运行。

运行方式：

```bash
cd Algorithms/物理-量子场论-0407-重整化_(Renormalization)
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程（9 步）：

1. `tadpole_integral_cutoff` 用闭式公式计算 `I_tad(Lambda,m)`，显式暴露二次发散结构。  
2. `bubble_integral_cutoff` 对费曼参数 `x` 调用 `scipy.integrate.quad`，得到 `B(Lambda,m,p)`。  
3. `solve_bare_lambda` 构造方程 `lambda_R = lambda_B - 3 lambda_B^2 B_sub`，先 Newton、后 Brent 回退求根。  
4. `compute_bare_parameters` 在每个 cutoff 上计算 `lambda_B` 与 `m_B^2`，把重整化条件落到具体数值。  
5. `effective_renormalized_coupling` 用同一一圈公式计算任意探测动量下的 `lambda_eff(p)`。  
6. `build_report` 生成“逐 cutoff 重整化”数据表，并单独生成“固定 bare”naive 对照数据表。  
7. `build_report` 对两组结果按 `p` 聚合最小值/最大值，形成 spread（cutoff 敏感度）指标。  
8. `main` 打印 bare 参数表、重整化透视表、naive 透视表、spread 汇总表。  
9. `main` 执行断言：减除点一致性、`lambda_B` 与 `m_B^2` 漂移方向、重整化方案优于 naive。  

说明：`scipy` 只提供通用数值积分/求根算子；重整化条件、方程结构、反项吸收逻辑与验证准则都在源码中显式实现。
