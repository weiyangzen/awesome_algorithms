# 部分子分布函数 (Parton Distribution Functions)

- UID: `PHYS-0399`
- 学科: `物理`
- 分类: `QCD`
- 源序号: `418`
- 目标目录: `Algorithms/物理-QCD-0418-部分子分布函数_(Parton_Distribution_Functions)`

## R01

部分子分布函数（PDF）描述强子内部夸克/胶子的纵向动量分数分布：

`f_i(x, Q^2)` = 在尺度 `Q^2` 下、部分子类型 `i` 携带动量分数 `x` 的分布密度。

本条目实现一个最小可运行 MVP：使用可解释的 Beta 族参数化构造 `u_v, d_v, sea, g`，生成伪 DIS 数据 `F2(x,Q^2)`，再用非线性最小二乘反演 PDF 参数并检查和规则（sum rules）。

## R02

MVP 目标与范围：

- 构造参考尺度 `Q0^2` 的 PDF 参数化；
- 用简化的 `Q^2` 演化项模拟尺度依赖；
- 生成带噪声的结构函数 `F2` 伪数据；
- 用 `scipy.optimize.least_squares` 拟合参数；
- 验证价夸克数和规则与动量和规则；
- 输出拟合质量指标（`chi2/dof`、`RMSE`、`R^2`）与样例表格。

不覆盖完整全球拟合（如高阶 DGLAP、重味阈值、实验系统误差协方差矩阵）。

## R03

数学模型（与 `demo.py` 对应）：

1. 参考尺度参数化
   - `u_v(x)=N_u x^{a_u}(1-x)^{b_u}`
   - `d_v(x)=N_d x^{a_d}(1-x)^{b_d}`
   - `sea(x)=N_s x^{a_s}(1-x)^{b_s}`
   - `g(x)=N_g x^{a_g}(1-x)^{b_g}`
2. 价数归一化
   - `int_0^1 u_v(x) dx = 2`
   - `int_0^1 d_v(x) dx = 1`
3. 海夸克动量分数设为显式参数 `m_sea`：
   - `int_0^1 x sea(x) dx = m_sea`
4. 胶子动量由和规则确定：
   - `int_0^1 x[u_v+d_v+sea+g] dx = 1`
5. 结构函数（LO 简化）
   - `F2 = x[(4/9)(u_v+0.5*sea) + (1/9)(d_v+0.3*sea) + (1/9)(0.2*sea)]`
6. 简化尺度演化（现象学）
   - `f(x,Q^2)=f(x,Q0^2)*exp(k*log((Q^2+lambda^2)/(Q0^2+lambda^2))*(1-x))`

## R04

物理直觉：

- 小 `x` 区域受 `x^a` 指数控制，决定低动量分数尾部陡峭程度；
- 大 `x` 区域受 `(1-x)^b` 控制，决定接近 1 时衰减速度；
- 价夸克由数目守恒固定总数（`u_v:2, d_v:1`）；
- 海夸克和胶子通过动量守恒竞争总动量份额；
- 随 `Q^2` 升高，分布形状会变化，MVP 用简化指数项表达这种趋势。

## R05

正确性要点：

1. 价夸克归一化常数使用 Beta 函数解析计算，不靠数值凑参数。
2. 海夸克和胶子归一化由动量和规则闭合，避免“总动量漂移”。
3. 拟合残差按观测不确定度 `sigma` 标准化，`chi^2` 具有明确统计意义。
4. 完成拟合后再次做独立数值积分，检查 `u_v`、`d_v` 数目与总动量是否仍满足约束。
5. 结果输出同时给统计指标和样本点预测-观测对比，避免只看单个标量。

## R06

复杂度分析（`N` 个数据点，`P` 个拟合参数）：

- 单次模型前向计算：`O(N)`；
- Levenberg-Marquardt / Trust-Region 迭代每步需多次残差评估，经验复杂度 `O(K * N * P)`；
- 数值和规则检查（密网格积分 `M` 点）：`O(M)`；
- 总体内存复杂度约 `O(N + M)`。

在当前配置（`N=225, P=9`）下，CPU 上可秒级完成。

## R07

标准流程：

1. 设定“真值参数”并生成 `(x,Q^2)` 网格。
2. 用真值模型生成 `F2_true`，叠加高斯噪声得 `F2_obs` 与 `sigma`。
3. 定义参数边界与初值。
4. 调用 `least_squares` 最小化加权残差。
5. 计算 `chi2/dof`、`RMSE`、`R^2`。
6. 用拟合参数在密网格上做和规则积分验证。
7. 打印参数恢复结果、统计指标、前若干行预测表。
8. 执行断言，若明显违背物理或数值预期则报错。

## R08

`demo.py` 采用的最小工具栈：

- `numpy`：数组与向量化计算；
- `scipy`：`special.beta`（解析归一化）+ `optimize.least_squares`（拟合）；
- `pandas`：结果表格整理；
- `scikit-learn`：`RMSE` 与 `R^2` 指标；
- `PyTorch`：可选一致性检查（若环境安装则对比 NumPy 前向结果）。

脚本无交互输入，`uv run python demo.py` 可直接运行。

## R09

核心函数接口：

- `compute_normalization_constants(params) -> dict[str, float]`
- `pdf_components_at_q0(x, params) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
- `evolve_component(base_component, x, q2, k) -> np.ndarray`
- `f2_model(x, q2, params) -> np.ndarray`
- `make_synthetic_dataset(rng, true_params, x_points, q2_points) -> pd.DataFrame`
- `residual_vector(theta, x, q2, y_obs, sigma) -> np.ndarray`
- `fit_pdf_parameters(df, initial_theta, bounds) -> FitSummary`
- `sum_rule_diagnostics(params) -> dict[str, float]`
- `run_optional_torch_check(df, params) -> Optional[float]`

## R10

测试与验收策略：

- 参数恢复性：拟合后 `chi2/dof` 应在合理范围（默认 `< 2.5`）；
- 拟合质量：`R^2` 应足够高（默认 `> 0.92`）；
- 物理约束：
  - `int u_v dx ≈ 2`
  - `int d_v dx ≈ 1`
  - `int x(u_v+d_v+sea+g) dx ≈ 1`
- 数值一致性：若安装 PyTorch，要求与 NumPy 前向最大差异足够小；
- 全流程无人工输入，失败即抛出 `AssertionError`。

## R11

边界条件与异常处理：

- `x` 必须落在 `(0,1)`，`Q^2` 必须正；
- 形状指数要求满足 Beta 函数可积条件（通过参数边界保证）；
- 若推导出的胶子动量分数 `m_g <= 0`，直接视为非法参数并返回惩罚残差；
- `sigma` 下限截断（`>=1e-6`）避免除零；
- 参数越界由优化器边界处理，不靠手工裁剪。

## R12

与标准 QCD 全局拟合的关系：

- 本实现保留了“参数化 + 数据反演 + 物理和规则”主骨架；
- 省略了 NLO/NNLO 系数函数、严格 DGLAP 卷积演化、重味方案与实验系统学；
- 因此它是教学/工程入门版，而非可直接用于精密物理分析的生产级 PDF 拟合器。

## R13

示例配置（`demo.py` 默认）：

- `x`：`1e-3` 到 `0.7` 的 45 点对数网格；
- `Q^2`：`[2, 5, 10, 20, 50] GeV^2`；
- 数据点总数：`225`；
- 真值参数包含 9 个自由度（低/高 `x` 形状 + 海夸克动量 + 尺度斜率）；
- 噪声模型：`sigma = 0.03 * |F2_true| + 0.002`。

该配置可稳定展示“参数可恢复 + 约束可校验”的完整闭环。

## R14

工程实现注意点：

- 解析归一化优先：比每次积分归一化更快更稳定；
- `x` 端点避开 `0` 与 `1`，减少幂律奇异数值问题；
- 残差里对非法参数使用常数惩罚，避免优化器进入 NaN；
- 报告时同时展示“真值/拟合值/相对偏差”，便于快速诊断可辨识性；
- PyTorch 检查采用可选分支，不把其安装与否变成硬依赖。

## R15

预期输出特征：

- 能打印每个参数的真值与拟合值，误差在可接受范围内；
- 拟合统计如 `chi2/dof`、`RMSE`、`R^2` 表现良好；
- 和规则诊断值接近理论目标（2、1、1）；
- 若安装 PyTorch，会额外输出 NumPy/Torch 前向差异；
- 最后输出 `All checks passed.`。

## R16

可扩展方向：

- 用离散 `x` 卷积核实现 LO DGLAP 数值演化；
- 引入 charm/bottom 与阈值匹配；
- 用协方差矩阵和 nuisance parameters 处理系统误差；
- 使用自动微分（PyTorch/JAX）替代有限差分 Jacobian；
- 多数据集联合拟合（DIS + Drell-Yan + jet）并做不确定度传播。

## R17

本条目交付内容：

- `README.md`：R01-R18 全部完成，含模型、约束、测试和工程细节；
- `demo.py`：可直接运行的 PDF 拟合 MVP；
- `meta.json`：保留并与任务元数据一致。

运行方式：

```bash
cd Algorithms/物理-QCD-0418-部分子分布函数_(Parton_Distribution_Functions)
uv run python demo.py
```

## R18

源码级算法流程拆解（对应 `demo.py`，9 步）：

1. **参数语义化与约束准备**  
   `PDFParams` 记录 9 个可拟合参数；`PARAM_ORDER` 固定向量化顺序，方便优化器与物理参数互转。

2. **解析归一化常数计算**  
   `compute_normalization_constants` 用 Beta 函数计算 `N_u/N_d/N_s`，并由动量和规则反推 `m_g` 与 `N_g`。

3. **参考尺度分量构造**  
   `pdf_components_at_q0` 在 `Q0^2` 生成 `u_v, d_v, sea, g` 四个分布，若 `m_g<=0` 立即报错。

4. **简化尺度演化**  
   `evolve_component` 对每个分量施加 `exp(k * log-ratio * (1-x))`，得到任意 `Q^2` 下分布。

5. **结构函数前向模型**  
   `f2_model` 将分量组合为 LO 简化 `F2(x,Q^2)`，并对数组输入做向量化处理。

6. **伪数据生成**  
   `make_synthetic_dataset` 在 `(x,Q^2)` 网格上计算 `F2_true`，按给定噪声模型采样得到 `F2_obs` 与 `sigma`。

7. **残差定义与参数拟合**  
   `residual_vector` 返回 `(F2_pred - F2_obs)/sigma`；`fit_pdf_parameters` 调用 `least_squares` 迭代求解最优参数。

8. **拟合后诊断**  
   计算 `chi2/dof`、`RMSE`、`R^2`，并用 `sum_rule_diagnostics` 数值积分校验三条核心和规则。

9. **一致性检查与收尾**  
   `run_optional_torch_check`（可选）比较 Torch/NumPy 前向差异；`run_assertions` 对统计与物理阈值做自动验收，最后打印 `All checks passed.`。
