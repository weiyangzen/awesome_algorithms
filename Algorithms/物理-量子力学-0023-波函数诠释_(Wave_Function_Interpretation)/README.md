# 波函数诠释 (Wave Function Interpretation)

- UID: `PHYS-0023`
- 学科: `物理`
- 分类: `量子力学`
- 源序号: `23`
- 目标目录: `Algorithms/物理-量子力学-0023-波函数诠释_(Wave_Function_Interpretation)`

## R01

本条目实现“波函数诠释”的最小可运行版本（MVP），核心目标是把抽象解释落到可计算链路：
- 构造一个可归一化的复波函数 `psi(x,t)`；
- 用 Born 规则 `rho(x)=|psi(x,t)|^2` 得到位置概率密度；
- 执行大量位置测量采样，并与理论概率分布做统计对比（`L1` 距离与 `KL` 散度）。

## R02

问题定义（MVP 范围）：
- 输入：
  - 一维无限深势阱长度 `L`；
  - 质量 `m`、约化普朗克常数 `hbar`、观察时刻 `t`；
  - 少量本征态复系数 `c_n`；
  - 空间离散点数 `n_grid` 与测量样本数 `n_samples`。
- 输出：
  - 归一化波函数 `psi`；
  - 概率密度 `rho` 及其积分值；
  - `<x>`、`<x^2>`、`Var(x)`；
  - 基于 Born 规则的测量样本与经验分布；
  - 经验分布与理论分布差异指标（`L1`、`KL`）。
- 约束：
  - 仅处理一维、无自旋、非相对论、封闭系统；
  - 不涉及塌缩动力学争论、环境退相干建模、量子测量装置细节。

## R03

数学模型：
1. 一维无限深势阱本征函数（`x in [0,L]`）：  
   `phi_n(x) = sqrt(2/L) * sin(n*pi*x/L)`。
2. 能级：  
   `E_n = n^2*pi^2*hbar^2 / (2*m*L^2)`。
3. 态叠加与相位演化：  
   `psi(x,t)=sum_n c_n * phi_n(x) * exp(-i*E_n*t/hbar)`。
4. Born 诠释：  
   `rho(x)=|psi(x,t)|^2`，并要求 `integral rho(x) dx = 1`。
5. 期望与方差：  
   `<x>=integral x*rho(x)dx`，`<x^2>=integral x^2*rho(x)dx`，`Var(x)=<x^2>-<x>^2`。

## R04

算法流程（MVP）：
1. 生成空间网格 `x`（均匀离散）。
2. 将用户给定的复系数向量 `c_raw` 归一化为 `c_normalized`。
3. 对每个本征态 `n` 计算 `phi_n(x)`、`E_n`、相位因子，并累加得到 `psi(x,t)`。
4. 计算 `rho=|psi|^2`，再做数值归一化。
5. 把连续密度转换为离散 PMF `p_theoretical`（用于蒙特卡洛采样）。
6. 根据 `p_theoretical` 采样 `n_samples` 次位置测量，得到 `empirical_prob`。
7. 计算 `<x>`、`Var(x)`、`L1`、`KL`，并执行阈值检查。
8. 打印简报与预览表。

## R05

核心数据结构：
- `numpy.ndarray`：
  - `x`、`psi`、`rho`、`theoretical_prob`、`empirical_prob`、`sampled_x`。
- `WaveFunctionResult`（`dataclass`）：
  - 集中保存全部状态数组与诊断量（归一化、矩、距离指标）。
- `pandas.DataFrame`：
  - 仅用于终端预览输出，不参与核心求解。

## R06

正确性要点：
- 通过 `c_raw -> c_normalized` 保证 `sum_n |c_n|^2 = 1`，保证态矢范数受控。
- 本征基 `phi_n` 在势阱内正交归一，叠加后 `psi` 仍可解释为有效态。
- 使用 `rho=|psi|^2` 且显式积分归一化，使 Born 概率解释在离散网格上成立。
- 采样分布直接来自 `p_theoretical`，因此经验分布在大样本下应收敛到理论分布。
- `L1` 与 `KL` 用于验证“测量统计符合波函数概率解释”这一核心命题。

## R07

复杂度分析：
- 记网格点数为 `G`，叠加本征态数为 `K`，采样次数为 `S`。
- 时间复杂度：`O(K*G + S)`。
- 空间复杂度：`O(G + S)`（主存储为状态数组与样本）。
- 对当前 MVP（`K=3`）而言，计算开销主要由 `G` 与 `S` 决定。

## R08

边界与异常处理：
- `L <= 0`、`mass <= 0`、`hbar <= 0` -> `ValueError`。
- `x` 不是严格递增的一维网格，或含 `nan/inf` -> `ValueError`。
- 系数向量为空、维度不对、范数为 0 -> `ValueError`。
- `n_grid < 32` 或 `n_samples < 1` -> `ValueError`。
- 概率向量质量和非正/非有限 -> `ValueError`。

## R09

MVP 取舍说明：
- 选用无限深势阱解析本征态，避免 PDE 求解器带来的实现复杂度。
- 重点放在“概率诠释与测量统计一致性”，不追求完整量子动力学框架。
- 不调用高层黑盒量子库（如一行封装的电路模拟器），而是逐步实现态叠加与采样。
- 这是一个“可解释、可复现、可验证”的教学与验证级最小实现。

## R10

`demo.py` 模块职责：
- `normalize_coefficients`：复系数归一化。
- `infinite_well_basis`：计算势阱本征函数 `phi_n(x)`。
- `build_wavefunction`：构造 `psi(x,t)`。
- `probability_density`：计算并归一化 `rho=|psi|^2`。
- `discrete_distribution_from_density`：连续密度到离散 PMF 的映射。
- `sample_position_measurements`：按 Born 概率采样测量结果。
- `compare_empirical_and_theoretical`：输出 `L1` 与 `KL`。
- `run_wave_function_interpretation_mvp`：串联全流程。
- `run_checks`：做结果断言；`main`：无交互运行入口。

## R11

运行方式：

```bash
cd Algorithms/物理-量子力学-0023-波函数诠释_(Wave_Function_Interpretation)
uv run python demo.py
```

脚本无需交互输入，直接输出统计报告并在通过检查后打印 `All checks passed.`。

## R12

输出字段解读：
- `basis coefficients c_n`：归一化后的本征态复幅度。
- `Integral rho dx`：概率密度积分，应接近 `1`。
- `<x>`、`<x^2>`、`std(x)`：位置分布的一阶/二阶统计量。
- `L1(empirical, theoretical)`：经验分布与理论分布的总变差量级。
- `KL(empirical || theoretical)`：经验分布相对理论分布的信息差。
- 预览表列：`x_m`、`psi_real`、`psi_imag`、`rho_theoretical`、`p_empirical`。

## R13

建议最小测试集：
- 正常场景：默认参数，检查归一化、方差正值、`L1/KL` 在阈值内。
- 大样本场景：增大 `n_samples`，`L1` 与 `KL` 应继续下降。
- 细网格场景：增大 `n_grid`，积分精度应提高。
- 异常输入场景：
  - `L <= 0`；
  - 系数全零；
  - 非递增 `x`；
  - `n_samples=0`。

## R14

关键可调参数：
- `L`：势阱尺度，影响本征波长与能级间距。
- `t`：观测时刻，影响各能级相位差。
- `c_raw`：各本征态占比与相位关系，决定干涉图样。
- `n_grid`：空间离散精度（越大越精细，越耗时）。
- `n_samples`：测量统计稳定性（越大越接近理论）。
- `seed`：随机种子，控制可复现实验。

## R15

方法对比：
- 对比“纯解析不采样”方式：
  - 纯解析只能给出 `rho`；
  - 本实现补上采样环节，可直接验证 Born 诠释的统计含义。
- 对比“黑盒量子框架一键模拟”：
  - 黑盒速度快但中间机制不透明；
  - 本实现源码可追踪，便于教学与审查。
- 对比“密度矩阵/开放系统模型”：
  - 本实现更轻量，适合单体纯态演示；
  - 密度矩阵更通用，但复杂度明显提升。

## R16

应用场景：
- 量子力学课程中 Born 规则与测量统计的可视化验证；
- 数值方法入门：从连续概率密度到离散采样；
- 量子算法前置教育：理解“振幅平方给概率”的核心概念；
- 实验数据教学对照：把理想理论分布与有限样本统计波动对比。

## R17

可扩展方向：
- 扩展到有限深势阱或谐振子势场；
- 增加时间序列 `t_0...t_N`，观察概率密度演化动画；
- 引入动量空间（傅里叶变换）并比较 `|psi(x)|^2` 与 `|phi(p)|^2`；
- 扩展到密度矩阵与退相干项；
- 用真实实验计数数据替换合成采样进行拟合检验。

## R18

源码级算法流（对应 `demo.py`，8 步）：
1. `main` 调用 `run_wave_function_interpretation_mvp`，设置 `L`、`t`、`n_grid`、`n_samples` 和随机种子。  
2. `run_wave_function_interpretation_mvp` 先创建网格 `x`，再在 `normalize_coefficients` 中把 `c_raw` 归一化，确保 `sum|c_n|^2=1`。  
3. `build_wavefunction` 遍历每个能级 `n`：计算 `E_n`、`phi_n(x)` 和相位 `exp(-iE_n t/hbar)`，累加得到复波函数 `psi(x,t)`。  
4. `probability_density` 计算 `rho=|psi|^2` 并用数值积分归一化，保证概率总和为 1。  
5. `expectation_x_moments` 计算 `<x>`、`<x^2>` 与 `Var(x)`，用于描述位置分布统计特征。  
6. `discrete_distribution_from_density` 把连续 `rho` 变为离散 `p_theoretical`，随后 `sample_position_measurements` 按该 PMF 进行蒙特卡洛采样，得到 `empirical_prob`。  
7. `compare_empirical_and_theoretical` 计算 `L1` 与 `KL`，量化经验分布和理论分布的一致性。  
8. `run_checks` 校验归一化与统计阈值，`main` 输出关键指标和预览表，最终打印 `All checks passed.`。  
