# 线性扰动理论 (Linear Perturbation Theory)

- UID: `PHYS-0357`
- 学科: `物理`
- 分类: `宇宙学`
- 源序号: `375`
- 目标目录: `Algorithms/物理-宇宙学-0375-线性扰动理论_(Linear_Perturbation_Theory)`

## R01

线性扰动理论（Linear Perturbation Theory）研究“微小密度起伏如何在宇宙膨胀背景中演化”。
在晚期大尺度结构形成中，常用线性密度对比度 `delta = delta_rho / rho_bar` 描述扰动，并写成

`delta(k, a) = D(a) * delta(k, a_init)`

其中 `D(a)` 是与尺度无关的线性增长因子（在无压冷暗物质、亚视界、GR 线性近似下成立）。本条目 MVP 的核心就是稳定、透明地数值求解 `D(a)`。

## R02

本任务目标是构建一个最小可运行管线，完成以下闭环：

1. 给定背景宇宙学参数 `Omega_m0, Omega_r0, Omega_Lambda0, Omega_k0`；
2. 数值积分线性增长方程，得到 `D(a)` 与增长率 `f=d ln D / d ln a`；
3. 在 `LambdaCDM` 下拟合经验关系 `f ~= Omega_m(a)^gamma`；
4. 用 Einstein-de Sitter (EdS) 解析解 `D(a)=a, f=1` 做基准自检。

脚本强调“可审计方程链”，而不是依赖黑盒宇宙学软件包。

## R03

MVP 采用的物理假设：

- 背景：FRW 均匀各向同性宇宙；
- 扰动：标量密度扰动的一阶线性近似；
- 物质：压力可忽略（冷物质主导结构增长）；
- 尺度：亚视界线性增长（忽略非线性并合与反馈）；
- 引力：广义相对论标准增长方程，不含修正引力参数化。

因此该实现是“背景宇宙 + 线性增长”的教学/验证级最小模型。

## R04

核心方程（变量 `x = ln a`）：

1. 背景膨胀率
`E(a)^2 = H(a)^2 / H0^2 = Omega_r0 a^-4 + Omega_m0 a^-3 + Omega_k0 a^-2 + Omega_Lambda0`

2. 线性增长方程
`d^2D/dx^2 + [2 + d ln H / dx] dD/dx - 3/2 * Omega_m(a) * D = 0`

3. 其中
`Omega_m(a) = [Omega_m0 a^-3] / E(a)^2`

4. 增长率定义
`f(a) = d ln D / d ln a = (dD/dx)/D`

5. 经验拟合
`f(a) ~= Omega_m(a)^gamma`（在 `LambdaCDM` 中常见 `gamma ~ 0.55`）。

## R05

数值算法设计（`demo.py`）：

- 把二阶方程改写为一阶系统：`y=[D, dD/dx]`；
- 用 `scipy.integrate.solve_ivp` 从 `a_init` 积分到 `a=1`；
- 初值使用物质时代近似 `D~a`，即 `D(a_init)=a_init`、`dD/dln a=a_init`；
- 结果归一化为 `D(a=1)=1`；
- 计算 `f(a)`、`Omega_m(a)`，并做 `gamma` 对数线性拟合；
- 并行运行 EdS case，验证解析关系 `D(a)=a` 与 `f=1`。

复杂度近似为 `O(N_eval)`，其中 `N_eval` 为输出采样点数（默认 800）。

## R06

脚本输出两部分：

1. `LambdaCDM` 关键红移节点表（`z=0,0.5,1,2,3,5`）
- `D(z)`
- `f(z)`
- `Omega_m(a)`
- `Omega_m(a)^gamma`

2. 汇总诊断
- `gamma` 拟合值和 RMS 误差；
- `f(z=0,1,3)`；
- EdS 的最大绝对误差（`|D-a|`, `|f-1|`）。

同时脚本含断言，若偏离预期物理行为会直接失败。

## R07

优点：

- 方程到代码映射直接，便于复核；
- 用 EdS 解析解做强约束回归测试；
- 结果能反映 `LambdaCDM` 中增长减缓与 `gamma` 拟合关系。

局限：

- 仅线性区，不包含非线性结构形成；
- 未处理重子声学振荡、中微子质量、标度依赖增长；
- 未接入观测数据拟合（如 RSD、弱透镜）流程。

## R08

前置知识：

- FRW 背景膨胀和密度参数定义；
- 常微分方程数值积分；
- 线性增长因子与增长率的物理意义。

运行依赖：

- Python `>=3.10`
- `numpy`
- `scipy`
- `pandas`

## R09

适用场景：

- 宇宙学课程中的线性增长演示；
- 构建更大 LSS 管线前的基线测试模块；
- 检查给定背景参数下增长历史是否物理合理。

不适用场景：

- 非线性功率谱与晕模型预测；
- 高精度 CMB/LSS 联合参数推断；
- 修正引力或多流体耦合扰动的研究级计算。

## R10

正确性直觉：

1. 线性方程系数完全由背景 `H(a)` 与 `Omega_m(a)` 决定；
2. EdS 下增长应严格回到解析解 `D=a`，因此是最直接真值对照；
3. `LambdaCDM` 今天的增长率应明显低于 1（暗能量抑制增长）；
4. 更高红移趋近物质主导，`f(z)` 应向 1 靠近；
5. `f` 与 `Omega_m^gamma` 在中低红移应近似吻合。

这些都被脚本内断言覆盖。

## R11

数值稳定性与实现细节：

- 积分变量选 `ln a`，跨越大动态范围更稳定；
- `a_init=1e-3` 避免 `a->0` 奇异端点；
- 对背景先检查 `E(a)^2 > 0`，提前排除不物理参数；
- 使用较严格容差 `rtol=1e-9`, `atol=1e-11`；
- `gamma` 拟合时过滤 `Omega_m(a)≈1` 区域，避免 `ln` 退化造成数值病态。

## R12

关键参数（`GrowthSolverConfig`）：

- `a_init`：积分起点尺度因子；
- `n_eval`：输出采样分辨率；
- `rtol/atol`：ODE 误差控制。

关键宇宙学参数（`CosmologyParams`）：

- `omega_m0`, `omega_r0`, `omega_lambda0`, `omega_k0`；
- 若 `omega_k0=None`，脚本自动用闭合关系推导。

调参建议：

- 若需要更平滑导出曲线，提高 `n_eval`；
- 若拟合残差偏大，可提高积分精度并收紧拟合红移范围。

## R13

保证类型说明：

- 近似比保证：N/A（非离散优化问题）；
- 随机成功率保证：N/A（确定性数值流程，无随机采样）。

工程可保证项：

- 脚本无交互输入，重复运行可复现；
- EdS 解析基准误差受断言约束；
- `LambdaCDM` 下增长率与 `gamma` 拟合范围受断言约束。

## R14

常见失败模式：

1. 把初值设在过晚时代，导致丢失纯增长模初态；
2. 忽略 `E(a)^2` 正性检查，导致 ODE 系数发散；
3. 在 `Omega_m(a)->1` 区域直接做对数拟合，数值不稳；
4. 混用 `dD/da` 与 `dD/dln a`，方程系数写错；
5. 未做归一化比较，导致不同模型 `D` 量纲基准不一致。

本实现分别用“早期初值 + 正性校验 + 过滤拟合 + 明确变量定义 + 归一化”应对。

## R15

可扩展方向：

1. 增加增长方程对修正引力参数 `mu(a,k)` 的支持；
2. 引入 `wCDM` 或 `w0wa` 暗能量参数化；
3. 接入观测量（`f sigma8`）并做最小参数拟合；
4. 将线性结果拼接到非线性修正模块；
5. 增加单元测试并导出 CSV/图像结果。

## R16

相关主题：

- 弗里德曼方程与背景宇宙学；
- 线性化爱因斯坦方程与牛顿势扰动；
- 结构形成、功率谱与转移函数；
- 红移空间畸变（RSD）与 `f sigma8`；
- 非线性增长与 N-body 模拟。

## R17

运行方式（无交互）：

```bash
cd Algorithms/物理-宇宙学-0375-线性扰动理论_(Linear_Perturbation_Theory)
uv run python demo.py
```

交付核对：

- `README.md` 的 `R01-R18` 已填写完整；
- `demo.py` 可直接执行并输出诊断结果；
- `meta.json` 与任务元数据一致；
- 目录可独立验证。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 构造 `LambdaCDM` 和 `EdS` 两组 `CosmologyParams`，并设置 `GrowthSolverConfig`。  
2. `solve_linear_growth` 先通过 `with_derived_curvature` 和 `validate_cosmology` 完成参数闭合与 `E(a)^2>0` 校验。  
3. 在 `x=ln a` 变量下，`growth_rhs` 把二阶增长方程拆成一阶系统 `y=[D, dD/dx]`。  
4. `solve_ivp` 只负责通用 ODE 时间推进；方程系数 `d ln H/d ln a`、`Omega_m(a)` 全由本地函数显式计算，不是黑盒增长求解器。  
5. 数值解返回后做 `D(a=1)=1` 归一化，计算 `f=(dD/dx)/D` 并整理为 `pandas.DataFrame`。  
6. `fit_growth_index_gamma` 在 `0<=z<=3` 上对 `ln f` 与 `ln Omega_m(a)` 做单参数最小二乘，得到 `gamma` 和 RMS。  
7. `run_eds_checks` 用解析真值 `D=a, f=1` 进行强约束断言；`run_lcdm_checks` 断言 `f(z)` 趋势与 `gamma` 合理区间。  
8. `make_report_table` 生成固定红移节点结果，`main` 打印表格和摘要，形成一次性可复验输出。  

第三方库边界说明：`numpy/pandas` 仅做数值与表格；`scipy.solve_ivp` 仅做通用积分步进。宇宙学背景、增长方程、拟合与物理判据都在源码中展开实现。
