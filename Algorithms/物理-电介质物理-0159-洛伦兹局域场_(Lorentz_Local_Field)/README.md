# 洛伦兹局域场 (Lorentz Local Field)

- UID: `PHYS-0158`
- 学科: `物理`
- 分类: `电介质物理`
- 源序号: `159`
- 目标目录: `Algorithms/物理-电介质物理-0159-洛伦兹局域场_(Lorentz_Local_Field)`

## R01

洛伦兹局域场（Lorentz local field）用于描述介质内部某一分子实际感受到的电场 `E_loc`，它不同于宏观平均电场 `E_macro`。  
在各向同性、均匀介质与球形洛伦兹空穴近似下，有经典关系：

`E_loc = E_macro + P / (3 * epsilon_0)`

其中 `P` 是极化强度，`epsilon_0` 是真空介电常数。该关系是从微观极化与宏观介电响应建立桥梁的核心公式。

## R02

典型应用场景：

- 从分子极化率 `alpha` 与数密度 `N` 预测介电常数
- 推导 Clausius-Mossotti / Lorentz-Lorenz 关系
- 在介质物理、分子光学中估算局域增强电场
- 作为更复杂模型（各向异性、频散、非线性响应）的基准近似

## R03

基本变量与方程（各向同性标量形式）：

- 单分子诱导偶极矩：`p = alpha * E_loc`
- 宏观极化：`P = N * p = N * alpha * E_loc`
- 局域场公式：`E_loc = E_macro + P / (3 * epsilon_0)`

将后两式联立可得自洽关系：

`E_loc = E_macro / (1 - N*alpha/(3*epsilon_0))`

再结合 `P = epsilon_0 * chi_e * E_macro` 可得

`chi_e = (N*alpha/epsilon_0) / (1 - N*alpha/(3*epsilon_0))`

以及

`(epsilon_r - 1) / (epsilon_r + 2) = N*alpha / (3*epsilon_0)`。

## R04

直观解释：

- `E_macro` 是连续介质平均视角下的外场
- `P/(3*epsilon_0)` 是周围极化电荷在“洛伦兹球空穴中心”产生的附加场
- 若介质越易极化（`alpha` 大或 `N` 大），附加场越强，分子看到的 `E_loc` 越大
- 因而形成正反馈：更大 `E_loc` 导致更大 `P`，再反过来增强 `E_loc`

## R05

正确性要点（MVP 对应）：

1. 从 `P = N*alpha*E_loc` 与 `E_loc = E_macro + P/(3*epsilon_0)` 出发可直接代数消元得到 `E_loc` 闭式解。
2. 再由 `P = epsilon_0*chi_e*E_macro` 得到 `chi_e`，构成从微观到宏观参数的映射。
3. 由 `epsilon_r = 1 + chi_e` 与 Clausius-Mossotti 公式得到两条独立计算路径。
4. `demo.py` 对两条路径的 `epsilon_r` 做数值一致性检验，验证推导与实现。
5. 额外检查稳定区间 `x = N*alpha/(3*epsilon_0) < 1`，避免进入奇异点附近的非物理发散区。

## R06

复杂度分析（单材料标量版本）：

- 计算 `x, E_loc, P, chi_e, epsilon_r`：常数次算术操作，时间复杂度 `O(1)`，空间复杂度 `O(1)`。
- 若对 `m` 组材料参数批处理：时间复杂度 `O(m)`，空间复杂度 `O(m)`（保存结果表）。

本条目属于“公式驱动算法”，关键不在数值求解规模，而在物理约束与一致性验证。

## R07

标准实现流程：

1. 输入材料参数：`N`、`alpha`、`E_macro`。
2. 计算无量纲耦合因子 `x = N*alpha/(3*epsilon_0)`。
3. 校验 `x < 1`（保守上可要求 `< 0.95` 防止条件数恶化）。
4. 用闭式解求 `E_loc = E_macro/(1-x)`。
5. 计算 `P = N*alpha*E_loc`。
6. 由 `P/(epsilon_0*E_macro)` 得 `chi_e`。
7. 分别用 `epsilon_r = 1 + chi_e` 与 Clausius-Mossotti 得 `epsilon_r`。
8. 输出误差与诊断量。

## R08

`demo.py` 的 MVP 设计：

- 依赖：`numpy`（数值计算）+ `pandas`（结果表展示）
- 物理模型：各向同性线性介质、静态或准静态标量场近似
- 数据：3 组示例材料（稀薄气体、中等极化介质、高极化介质）
- 输出：
  - `x`（耦合因子）
  - `E_loc/E_macro`（局域场增强倍数）
  - `P`
  - 两条路径的 `epsilon_r`
  - 相对误差 `rel_err_eps_r`

## R09

`demo.py` 接口约定：

- `lorentz_coupling_factor(number_density, polarizability) -> float`
- `local_field(macro_field, number_density, polarizability) -> float`
- `polarization(number_density, polarizability, local_field_value) -> float`
- `susceptibility_from_micro(number_density, polarizability) -> float`
- `epsilon_r_from_chi(chi_e) -> float`
- `epsilon_r_clausius_mossotti(number_density, polarizability) -> float`
- `analyze_material(name, number_density, polarizability, macro_field) -> dict[str, float | str]`
- `main() -> None`

## R10

测试策略：

- 单元级逻辑检查：
  - 当 `alpha -> 0` 时，`E_loc -> E_macro`，`epsilon_r -> 1`
  - 当 `x` 增大时，`E_loc/E_macro = 1/(1-x)` 单调增大
- 一致性检查：
  - `epsilon_r(1+chi_e)` 与 `epsilon_r(CM)` 的相对误差应接近机器精度
- 稳定性检查：
  - `x >= 1` 时抛出异常，防止奇点附近失真结果被误用

## R11

边界条件与异常处理：

- `number_density < 0` 或 `polarizability < 0`：抛出 `ValueError`
- `x = N*alpha/(3*epsilon_0) >= 1`：抛出 `ValueError`
- `macro_field = 0` 允许，结果应给出零极化（但 `epsilon_r` 仍可由材料参数计算）
- 浮点比较采用小容差，避免把舍入误差误判为模型错误

## R12

与相关理论的关系：

- Clausius-Mossotti 关系是洛伦兹局域场 + 线性极化率模型的直接结果
- Lorentz-Lorenz 折射率关系是其在光学频段常见的对应形式（忽略磁响应时）
- 若介质各向异性，应将 `alpha` 与 `epsilon` 扩展为张量
- 若频率升高，应考虑复极化率与色散/吸收（`alpha(omega)`）

## R13

示例参数选择（`demo.py`）：

- `epsilon_0 = 8.8541878128e-12 F/m`
- `E_macro = 1.0e5 V/m`
- 三组参数：
  - 稀薄气体：`N=2.5e25`, `alpha=1.7e-40`
  - 中等极化介质：`N=3.0e27`, `alpha=1.2e-39`
  - 高极化介质：`N=8.0e27`, `alpha=1.4e-39`

这些参数让 `x` 横跨弱耦合到较强耦合区间，同时保持 `x < 1`，便于观察局域场增强趋势。

## R14

工程实现注意点：

- 单位必须统一为 SI（`N` 用 `m^-3`，`alpha` 用 `C·m^2/V`，电场 `V/m`）
- `x` 接近 1 时会出现数值爆炸，不代表实际材料一定可达到该极限
- 该模型忽略短程关联、晶格离散性与非线性效应，适合作为一阶近似
- 表格输出时保留科学计数法，避免弱极化量被格式化为 0

## R15

最小示例解释：

- 在稀薄气体参数下，`x` 很小，`E_loc` 与 `E_macro` 几乎相同，`epsilon_r` 仅略大于 1。
- 在高极化参数下，`x` 明显增大，`E_loc/E_macro` 提升显著，导致更高 `epsilon_r`。
- 两条 `epsilon_r` 计算路线仍然一致，说明算法推导链条闭合。

## R16

可扩展方向：

- 张量化版本：处理各向异性晶体中的局域场
- 频域版本：引入复数 `alpha(omega)` 预测色散与损耗
- 数据驱动版本：对实验 `epsilon_r` 数据反演 `alpha` 或有效 `N`
- 与分子动力学/第一性原理耦合，校正简单洛伦兹空穴近似

## R17

本条目交付说明：

- `README.md`：已完成 R01-R18，覆盖定义、推导、复杂度、边界、工程实现与源码级流程
- `demo.py`：提供可运行、无交互输入的最小 Python MVP
- `meta.json`：保持与任务元信息（UID、学科、分类、源序号、路径）一致

运行方式：

```bash
uv run python Algorithms/物理-电介质物理-0159-洛伦兹局域场_(Lorentz_Local_Field)/demo.py
```

或在目录内运行：

```bash
uv run python demo.py
```

## R18

源码级算法流程拆解（对应 `demo.py`，共 8 步）：

1. **参数读入与常量定义**  
   在 `main` 中定义材料参数列表、宏观场 `E_macro` 与常量 `epsilon_0`。

2. **计算耦合因子**  
   `lorentz_coupling_factor` 计算 `x = N*alpha/(3*epsilon_0)`，作为后续稳定性与增强强度核心指标。

3. **物理可行性检查**  
   在 `local_field` / `susceptibility_from_micro` 中检查 `N >= 0`、`alpha >= 0`、`x < 1`，不满足则抛出异常。

4. **局域场自洽求解**  
   使用闭式解 `E_loc = E_macro/(1-x)`，避免迭代误差与额外数值求解器依赖。

5. **由微观关系求极化强度**  
   `polarization` 计算 `P = N*alpha*E_loc`，得到可直接与宏观量关联的中间变量。

6. **路径 A：经电极化率得到介电常数**  
   `susceptibility_from_micro` 给出 `chi_e`，再经 `epsilon_r_from_chi` 得 `epsilon_r_A = 1 + chi_e`。

7. **路径 B：Clausius-Mossotti 直接计算**  
   `epsilon_r_clausius_mossotti` 用 `epsilon_r_B = (1 + 2x)/(1 - x)` 独立求值。

8. **一致性诊断与结果表输出**  
   `analyze_material` 计算 `rel_err_eps_r`，`main` 汇总为 `pandas.DataFrame` 打印，验证算法实现与理论等价性。
