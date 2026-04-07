# GGA近似 (Generalized Gradient Approximation)

- UID: `PHYS-0207`
- 学科: `物理`
- 分类: `计算物理`
- 源序号: `208`
- 目标目录: `Algorithms/物理-计算物理-0208-GGA近似_(Generalized_Gradient_Approximation)`

## R01

GGA（Generalized Gradient Approximation）是密度泛函理论（DFT）中交换-相关泛函的一类近似。  
相较于只依赖局域电子密度 `n(r)` 的 LDA，GGA 额外引入密度梯度 `∇n(r)`，用来描述“非均匀电子气”中的空间变化效应。

本条目给出 exchange-only 的最小可运行实现：在 1D 模型密度上比较 `E_x^LDA` 与 `E_x^GGA`，并验证 GGA 在均匀极限和梯度增强行为上的关键性质。

## R02

MVP 的目标问题：

1. 给定一维电子密度剖面 `n(x)`；
2. 计算 LDA 交换能与 PBE 形式 GGA 交换能；
3. 验证以下物理/数值性质：
   - 均匀密度下 `GGA -> LDA`；
   - 增强因子 `F_x(s)` 满足 PBE 上下界；
   - 梯度越强，GGA 修正越明显；
   - 网格加密后能量收敛。

该问题规模小、可完全审计，不依赖外部电子结构黑箱程序。

## R03

本实现使用的 LDA 交换能表达式（原子单位风格）为：

- 每电子交换能：`eps_x^LDA(n) = -C_x * n^(1/3)`
- 常数：`C_x = 3/4 * (3/pi)^(1/3)`
- 交换能：`E_x^LDA = ∫ n(x) * eps_x^LDA(n(x)) dx`

在代码中，`lda_exchange_per_particle` 和 `exchange_energies` 直接实现了这组公式。

## R04

GGA（这里采用 PBE 交换增强形式）写为：

- `E_x^GGA = ∫ n(x) * eps_x^LDA(n(x)) * F_x(s(x)) dx`
- 约化梯度：`s = |∇n| / (2 * k_F * n)`
- `k_F = (3*pi^2*n)^(1/3)`
- 增强因子：`F_x(s) = 1 + kappa - kappa / (1 + (mu/kappa) * s^2)`

本脚本取常见 PBE 参数：
- `mu = 0.2195149727645171`
- `kappa = 0.804`

## R05

模型密度设计：

- 均匀剖面：`n(x) = constant`（用于验证均匀极限）；
- 非均匀剖面：高斯包络 + 余弦调制
  `n(x) = background + n0 * exp(-(x/width)^2) * (1 + amplitude * cos(qx))`。

这样可以用同一套代码覆盖“零梯度”“中等梯度”“较强梯度”三种情形。

## R06

`demo.py` 输入输出约定：

- 输入：全部内置在 `GGAConfig` 与密度参数中，无命令行交互；
- 输出：
1. 每个剖面的 `E_x_LDA`、`E_x_GGA`、修正量、`<s>`、`max(s)`、`min/max(Fx)`；
2. 粗细网格相对误差（LDA 与 GGA 各一项）；
3. 阈值检查项与最终 `Validation: PASS/FAIL`。

脚本可直接运行：`uv run python demo.py`。

## R07

算法主流程（高层）：

1. 构建均匀与非均匀电子密度 `n(x)`；
2. 有限差分计算 `grad n`；
3. 由 `n` 与 `grad n` 构造约化梯度 `s`；
4. 计算 PBE 交换增强因子 `F_x(s)`；
5. 计算并积分 LDA/GGA 交换能；
6. 汇总多剖面对比表；
7. 做均匀极限、边界约束、梯度趋势、网格收敛检查并给出 PASS/FAIL。

## R08

复杂度分析（`N` 为网格点数，`P` 为剖面个数）：

- 每个剖面：
  - 梯度计算 `O(N)`；
  - `s`、`F_x`、积分 integrand 评估 `O(N)`；
  - 数值积分 `O(N)`；
- 全部剖面复杂度 `O(P*N)`；
- 网格收敛对比（粗 + 细）仍是线性级，总体 `O(N)` 量级常数倍。

空间复杂度主要由若干长度为 `N` 的数组组成，为 `O(N)`。

## R09

数值稳定策略：

- 对密度使用 `density_floor` 下界，避免在极低密度区出现除零或不稳定；
- `s` 的分母 `2*k_F*n` 也使用最小值保护；
- 梯度使用 `np.gradient(..., edge_order=2)` 保持边界精度；
- 输出中显式检查 `F_x` 是否在理论边界内，防止实现错误被静默掩盖。

## R10

MVP 技术栈：

- Python 3
- `numpy`：数组运算、梯度、积分（`np.trapezoid`）
- `pandas`：结果表格化输出
- 标准库 `dataclasses`：配置与结果结构化

未调用任何 DFT 软件包黑箱。GGA 的核心计算链路全部在源码中显式实现。

## R11

运行方式：

```bash
cd Algorithms/物理-计算物理-0208-GGA近似_(Generalized_Gradient_Approximation)
uv run python demo.py
```

脚本不读取交互输入；阈值不满足时会返回非零退出码。

## R12

主要输出字段解释：

- `E_x_LDA`：LDA 交换能积分结果。
- `E_x_GGA`：GGA（PBE 交换增强）交换能积分结果。
- `GGA-LDA`：梯度修正项（通常为负，代表交换能更低）。
- `|corr|/|LDA|`：修正相对强度。
- `<s>`、`max(s)`：约化梯度统计量。
- `min(Fx)`、`max(Fx)`：增强因子区间。
- `relative ... difference (coarse vs fine)`：网格收敛度指标。

## R13

内置验收条件（全部通过才 PASS）：

1. `uniform limit: |E_GGA - E_LDA| < 1e-10`
2. `min(Fx) >= 1`
3. `max(Fx) <= 1 + kappa`
4. `smooth` 剖面满足 `E_GGA < E_LDA`
5. `sharp` 剖面的修正比强于 `smooth`
6. LDA 网格相对差 `< 5e-4`
7. GGA 网格相对差 `< 5e-4`

这组检查覆盖了理论边界、趋势正确性和数值收敛三类核心要求。

## R14

当前实现局限：

- 只做 exchange-only，未加入相关项 `E_c`；
- 只在给定密度上评估泛函，不做 Kohn-Sham 自洽迭代；
- 使用 1D 剖面近似演示，非真实 3D 周期固体/分子计算；
- 未实现自旋极化版本与其他 GGA 家族（如 B88/PW91 对比）。

## R15

可扩展方向：

- 加入相关泛函（例如 PBE correlation）并统计 `E_xc`；
- 引入简单 Kohn-Sham 迭代闭环（密度更新 + 混合）；
- 对比多种交换增强函数（PBE、revPBE、B88）；
- 扩展到 3D 网格与周期边界条件，并纳入动能与 Hartree 项形成更完整 DFT 教学原型。

## R16

应用背景与价值：

- GGA 是凝聚态与量化计算中最常用的交换-相关近似层级之一；
- 相比 LDA，GGA 对非均匀密度（表面、分子键区、低维结构）通常更可靠；
- 本 MVP 适合作为“从 LDA 升级到 GGA”的可复现实验模板与单元测试基线。

## R17

方法对比（简述）：

- LDA：最简单、计算稳定，但对密度梯度效应描述不足；
- GGA：引入 `∇n`，在多数体系上精度更好，代价略增；
- meta-GGA / hybrid：通常更精确但实现和计算成本更高。

本条目选择 PBE 交换 GGA，是为了在最小复杂度下体现“梯度增强”这个核心机制，并保持源码透明可审计。

## R18

`demo.py` 源码级流程拆解（9 步）：

1. `GGAConfig` 定义网格大小、`mu/kappa`、密度下界，统一控制数值环境。  
2. `make_grid` 生成 `[-x_max, x_max]` 一维离散网格。  
3. `uniform_density` 与 `gaussian_modulated_density` 构造三类测试密度（均匀/平滑梯度/强梯度）。  
4. `finite_difference_gradient` 用二阶边界差分计算 `∇n`。  
5. `reduced_gradient_s` 依公式 `s=|∇n|/(2*k_F*n)` 计算约化梯度，`k_F=(3*pi^2*n)^(1/3)`。  
6. `pbe_exchange_enhancement` 计算 PBE 交换增强因子 `F_x(s)`，并隐含满足 `1 <= F_x <= 1+kappa`。  
7. `exchange_energies` 显式构建 `n*eps_x^LDA` 与 `n*eps_x^LDA*F_x`，用 `np.trapezoid` 积分得到 `E_x^LDA` 与 `E_x^GGA`。  
8. `evaluate_profile` 汇总每个剖面的能量与梯度统计；`grid_convergence_check` 比较粗细网格下的相对误差。  
9. `main` 组装结果表、执行 7 条阈值检查并打印 `Validation: PASS/FAIL`。

说明：虽然使用了 `numpy/pandas` 基础数值工具，但 GGA 关键算法链（`n -> ∇n -> s -> F_x -> E_x`）是逐步显式实现，不是第三方黑箱一键求值。
