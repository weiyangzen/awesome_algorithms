# 量子测量理论 (Quantum Measurement Theory)

- UID: `PHYS-0260`
- 学科: `物理`
- 分类: `量子基础`
- 源序号: `263`
- 目标目录: `Algorithms/物理-量子基础-0263-量子测量理论_(Quantum_Measurement_Theory)`

## R01

量子测量理论研究“如何把量子态映射为可观测结果，以及测量后系统如何更新”。
本条目给出一个可运行 MVP，覆盖三条主线：

1. **Born 规则**：从密度矩阵计算测量概率；
2. **测量回馈（state update）**：根据测量结果更新后验量子态；
3. **不可交换测量次序效应**：比较 `Z` 与 `X->Z` 的统计差异。

## R02

为什么用这个最小问题定义“量子测量理论”算法条目：

1. 既有严格数学定义（密度矩阵、POVM、Kraus 更新），又能直接做数值实验；
2. 可以在二维量子比特空间完整呈现“概率 + 回馈 + 序效应”；
3. 不依赖大型框架或黑盒求解器，适合作为后续量子信息算法的基础模块。

## R03

本实现使用的核心对象（二维复向量空间）：

1. 量子态（密度矩阵）`rho`：
   - `rho = rho^dagger`
   - `Tr(rho)=1`
   - `rho >= 0`
2. 测量（Kraus 表示）`{M_k}`，对应效应算符 `E_k = M_k^dagger M_k`；
3. 完备性条件：`sum_k E_k = I`；
4. 特例：投影测量（projective measurement）可取 `M_k = P_k`。

## R04

算法对应的物理方程：

1. Born 概率：`p_k = Tr(E_k rho)`；
2. 条件态更新（一次测量得到结果 `k` 后）：
   `rho' = (M_k rho M_k^dagger) / p_k`；
3. 若连续执行测量序列 `M^(1), M^(2), ...`，则把上一步 `rho'` 作为下一步输入。

本条目在 `demo.py` 中同时实现了：
- 投影 `Z` 测量、投影 `X` 测量；
- 一个两结果非投影 POVM（unsharp Z）。

## R05

复杂度（量子比特维度 `d=2`，测量结果数 `m`，总 shots 为 `N`）：

1. 单次概率计算：`O(m * d^3)`（小矩阵乘法与迹运算）；
2. 单次后验更新：`O(d^3)`；
3. `N` 次独立采样：`O(N * m * d^3)`；
4. 空间复杂度：`O(d^2)`（状态矩阵 + 常数级中间量，结果表除外）。

由于 `d=2`，实际运行主要由采样次数 `N` 决定。

## R06

`demo.py` 的模块组织：

1. `Measurement`：封装 Kraus 算符并生成 `effects()`；
2. `density_from_ket`：从态矢生成密度矩阵；
3. `born_probabilities`：计算并归一化 `p_k`；
4. `measure_once`：执行一次测量并返回 `(outcome, post_state)`；
5. `sample_prepared_state`：独立重复制备态的频率统计；
6. `collapse_repeatability`：验证投影测量重复性；
7. `sequence_final_distribution`：测量序列终态统计；
8. `main`：运行实验、打印表格、执行断言。

## R07

MVP 设计了三组可验证实验：

1. **Born 频率校验**：
   - `Z on |0>`
   - `Z on |+>`
   - `X on |0>`
   - `UnsharpZ(eta=0.80) on |0>`
2. **投影塌缩重复性**：对 `|+>` 连续做两次 `Z` 测量，统计第一次与第二次结果是否不一致；
3. **不可交换次序效应**：
   - 直接 `Z on |0>`（应接近确定性 0）；
   - `X then Z on |0>`（应接近均匀分布）。

## R08

可复现与数值稳健策略：

1. 固定随机种子 `base_seed=20260407`；
2. 概率在浮点误差下做非负裁剪后再归一化；
3. 每次态更新后执行 `Hermitian/trace/PSD` 检查；
4. 每个测量先检查完备性 `sum_k M_k^dagger M_k = I`；
5. 用断言约束关键结论，便于自动验证脚本判断通过/失败。

## R09

关键参数与默认设置：

1. Born 频率实验：`n_shots=30000`；
2. 塌缩重复性实验：`n_trials=5000`；
3. 序效应实验：`n_shots=20000`；
4. 非投影测量强度：`eta=0.80`（`0.5 <= eta <= 1.0`）。

经验上，样本数增大时频率误差按 `O(N^{-1/2})` 缩小。

## R10

运行方式（无交互输入）：

```bash
cd Algorithms/物理-量子基础-0263-量子测量理论_(Quantum_Measurement_Theory)
uv run python demo.py
```

或在仓库根目录执行：

```bash
uv run python Algorithms/物理-量子基础-0263-量子测量理论_(Quantum_Measurement_Theory)/demo.py
```

## R11

输出包含三段结构化结果：

1. **Born Rule Frequency Check**：
   每个 case/outcome 的理论概率、经验频率、绝对误差；
2. **Projective Collapse Repeatability**：
   连续两次相同投影测量的 mismatch rate；
3. **Non-Commuting Order Effect**：
   `Direct Z` 与 `X then Z` 的终端分布和参考值对照。

## R12

本 MVP 采用的指标：

1. `abs_error = |empirical_prob - theory_prob|`；
2. `max_born_error = max(abs_error)`（Born 校验总误差上界）；
3. `repeat_mismatch_rate`（投影重复测量一致性）；
4. 序效应偏差：`|p_{X->Z}(0) - 0.5|`；
5. 序效应幅度：`|p_{direct Z}(0) - p_{X->Z}(0)|`。

## R13

适用范围：

1. 量子测量基础教学与实验脚本化验证；
2. 量子信息算法中“测量模块”的最小可测试单元；
3. 对 Born 规则、POVM、塌缩与不可交换性做可重复演示。

不适用范围：

1. 多体系统大维度仿真（需更高效线性代数与张量网络工具）；
2. 开放系统连续时间测量（需 Lindblad/随机主方程框架）；
3. 实验噪声建模、读出误差标定等工程级量子硬件流程。

## R14

常见失效模式与排查：

1. **测量不完备**：`sum_k E_k != I`，会导致概率和不为 1；
2. **态矩阵不物理**：非厄米、迹不为 1、负特征值明显；
3. **把后验态与先验态混用**：序列测量结果会出现错误统计；
4. **样本数过小**：把抽样噪声误判为理论失效；
5. **结果标签顺序混乱**：统计表字段对齐错误。

## R15

扩展方向：

1. 扩展到高维 qudit 与多比特联合测量；
2. 增加测量噪声与读出错误模型；
3. 引入连续弱测量轨迹（stochastic update）；
4. 将 Kraus 更新与量子线路门操作串接，形成 end-to-end 模拟；
5. 输出更多信息论指标（熵变化、互信息、Fidelity）。

## R16

与相关算法的关系：

1. 与“薛定谔方程时间演化”互补：前者管幺正演化，测量理论管非幺正更新；
2. 与“量子态层析”相关：层析依赖重复测量统计反推 `rho`；
3. 与“量子纠错/量子控制”相关：综合征提取本质是测量流程设计；
4. 与“经典统计估计”相关：测量输出频率最终仍需采样统计分析。

## R17

最小可交付能力清单（本条目已覆盖）：

1. 显式实现 `Born` 概率计算；
2. 显式实现 `Kraus` 后验更新（非黑盒）；
3. 覆盖投影测量与 POVM 两种测量类型；
4. 给出连续测量序列统计（体现不可交换性）；
5. 输出结构化结果并提供自动断言验证。

## R18

`demo.py` 的源码级算法流程（8 步）：

1. `main()` 固定随机种子与样本规模，调用 `run_born_rule_suite()`、`collapse_repeatability()`、`sequence_final_distribution()` 三类实验。  
2. `run_born_rule_suite()` 构造 `|0>`、`|+>` 两个初态与三种测量（`Z`、`X`、`Unsharp Z`），先用 `assert_measurement_complete()` 校验完备性。  
3. 对每个 case，`sample_prepared_state()` 先调用 `born_probabilities()` 计算 `p_k=Tr(E_k rho)`，再用 `multinomial` 采样经验频率。  
4. 结果整理成 `pandas.DataFrame`，输出 `theory_prob / empirical_prob / abs_error`。  
5. `collapse_repeatability()` 对 `|+>` 做两次连续 `Z` 测量：每次测量通过 `measure_once()` 完成 outcome 采样与 `rho' = M rho M^dagger / p` 更新。  
6. `sequence_final_distribution()` 对 `|0>` 执行 `X->Z`，逐 shot 传播后验态并统计最后一步 `Z` 的分布。  
7. `main()` 额外计算 `Direct Z on |0>` 的基准分布，与 `X->Z` 结果并排输出，直接展示测量次序引起的统计变化。  
8. 执行断言：Born 最大误差阈值、投影重复性、`X->Z` 近均匀性与显著序效应差距；全部通过后打印 `All checks passed.`。  

说明：`numpy/pandas` 只提供数组与表格基础能力；Born 概率、Kraus 更新、序列测量传播与验证逻辑均在源码中显式实现，不是第三方黑盒一键调用。
