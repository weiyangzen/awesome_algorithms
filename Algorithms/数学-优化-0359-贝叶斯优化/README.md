# 贝叶斯优化

- UID: `MATH-0359`
- 学科: `数学`
- 分类: `优化`
- 源序号: `359`
- 目标目录: `Algorithms/数学-优化-0359-贝叶斯优化`

## R01

贝叶斯优化（Bayesian Optimization, BO）用于优化“评估代价高、梯度不可得、函数形状未知”的黑盒目标函数。  
它的核心思想不是盲目试点，而是维护一个概率代理模型（常见是高斯过程 GP），再用采集函数（Acquisition）在“探索不确定区域”和“利用当前最优附近”之间做权衡。

本目录的 MVP 聚焦 1 维连续变量最小化问题：
- 代理模型：手写 Gaussian Process 回归（RBF 核 + Cholesky 求解）；
- 采集函数：Expected Improvement (EI)；
- 候选点搜索：随机扫描 + 黄金分割局部搜索（纯 `numpy`）。

## R02

问题定义（本实现）：
- 目标：
  - `min_{x in [l, u]} f(x)`
  - 其中 `f` 为黑盒函数，只能点查询。
- 输入：
  - 搜索区间 `bounds=(l, u)`；
  - 初始采样点 `initial_points`；
  - 迭代次数 `n_iter`。
- 输出：
  - BO 找到的 `best_x, best_y`；
  - 全部观测 `(x_obs, y_obs)`；
  - 每轮日志（`x_next/y_next/best_x/best_y/EI`）。

`demo.py` 默认优化 Forrester 基准函数（定义域 `[0,1]`），无需交互输入。

## R03

关键数学关系：

1. GP 后验均值与方差（归一化标签空间）：
   - `mu(x) = k(x, X)^T K^{-1} y`
   - `sigma^2(x) = k(x,x) - k(x, X)^T K^{-1} k(x, X)`
2. RBF 核：
   - `k(x, x') = sigma_f^2 * exp(- ||x-x'||^2 / (2*l^2))`
3. EI（最小化场景）：
   - `I(x) = max(0, f_best - mu(x) - xi)`
   - `EI(x) = I(x) * Phi(z) + sigma(x) * phi(z)`，其中 `z = I(x)/sigma(x)`
4. BO 更新：
   - `x_{t+1} = argmax_x EI(x)`
   - 评估 `y_{t+1}=f(x_{t+1})` 后并入数据集，重建后验。

## R04

算法流程（高层）：
1. 用初始点集评估目标函数，得到首批观测。  
2. 基于观测拟合 GP 后验。  
3. 在定义域内优化 EI，得到下一个采样点。  
4. 查询黑盒函数得到新观测值。  
5. 把新点追加到样本集中。  
6. 重复步骤 2-5，直到达到预算轮数。  
7. 返回观测历史中的最优值。

## R05

核心数据结构：
- `GpPosterior`（`@dataclass`）：
  - `x_train`：训练输入矩阵 `(n, d)`；
  - `y_mean, y_std`：标签标准化参数；
  - `length_scale, signal_variance, noise_variance`：核超参数；
  - `chol_l`：`K` 的 Cholesky 下三角；
  - `alpha`：`K^{-1} y` 的稳定求解结果。
- `history: List[Dict[str, float]]`：每轮包含
  - `iter, x_next, y_next, best_x, best_y, ei`。
- `x_obs, y_obs`：当前所有观测点和目标值。

## R06

正确性要点：
- 数值稳定：
  - 核矩阵加 `noise_variance + jitter`，再 Cholesky 分解；
  - 对近零方差的 EI 做保护，避免除零。
- 可行性保证：
  - 采集优化的随机候选与黄金分割搜索都带边界裁剪，候选点始终在 `[l,u]` 内。
- 退化处理：
  - 如果候选与已有点重复，则回退到“距离已采样点最远”的随机点。
- 可验证性：
  - `demo.py` 会用高密度网格近似参考最优解，并报告 `x/y` 误差。

## R07

复杂度分析（设迭代到第 `t` 轮时样本数为 `n_t`）：
- GP 拟合（Cholesky）：`O(n_t^3)`。  
- 单点预测：`O(n_t)` 到 `O(n_t^2)`（实现中包含三角求解）。  
- 每轮采集函数搜索：
  - 随机扫描 `m` 点约 `O(m * n_t^2)`；
  - 局部优化重启 `r` 次，代价与函数评估次数线性相关。
- 总体时间主要由后期 `O(n_t^3)` 主导。  
- 空间复杂度：核矩阵 `O(n_t^2)`。

## R08

边界与异常处理：
- 输入合法性：
  - 向量/矩阵维度检查（`ensure_1d_vector`, `ensure_2d_array`）；
  - `bounds` 要满足 `lower < upper`；
  - `initial_points` 必须落在区间内；
  - `n_iter > 0`。
- 模型构建异常：
  - 若核矩阵非正定导致 Cholesky 失败，抛 `RuntimeError`。
- 采集函数异常：
  - `mu/sigma` 形状不一致时报错；
  - `xi < 0` 报错。

## R09

MVP 取舍：
- 只实现 1 维连续变量，强调可读和可验证。  
- GP 超参数固定（`length_scale=0.2` 等），不做边际似然自动调参。  
- 采集函数只实现 EI，不扩展 UCB/PI/Knowledge Gradient。  
- 不依赖黑盒 BO 框架，核心 GP + EI 全部源码可追踪。

## R10

`demo.py` 主要函数职责：
- `forrester_function`：测试目标函数。  
- `rbf_kernel`：构造 RBF 核矩阵。  
- `fit_gp_posterior`：拟合 GP 后验（标准化 + Cholesky + alpha）。  
- `gp_predict`：给定查询点返回后验均值和标准差。  
- `expected_improvement`：计算 EI。  
- `propose_next_point`：随机扫描 + 黄金分割局部搜索组合找最大 EI 点。  
- `bayesian_optimization`：主循环（拟合、提议、评估、更新）。  
- `approximate_true_minimum`：用密集网格做参考最优。  
- `main`：组织实验并打印结果。

## R11

运行方式：

```bash
cd Algorithms/数学-优化-0359-贝叶斯优化
python3 demo.py
```

脚本不需要命令行参数，也不需要交互输入。

## R12

输出字段说明：
- 迭代表：
  - `iter`：第几轮 BO 迭代；
  - `x_next`：本轮采样点；
  - `y_next`：该点目标值；
  - `best_x / best_y`：截至本轮的历史最优；
  - `EI`：提议点的采集函数值。
- 最终摘要：
  - `BO best x/y`：BO 找到的最优观测；
  - `Reference x/y`：密集网格近似最优；
  - `Absolute x/y error`：与参考的误差；
  - `Pass loose accuracy check`：MVP 的宽松通过标志。

## R13

建议最小测试集：
- 默认 Forrester（已内置）：验证多峰黑盒上的 BO 收敛能力。  
- 建议补充：
  - 光滑单峰函数（如 `(x-0.2)^2`），验证快速收敛；
  - 含噪目标（`f(x)+epsilon`），验证 `noise_variance` 的鲁棒性；
  - 极少初始点与较多初始点对比，观察探索/利用差异。

## R14

可调参数与建议：
- `length_scale`：
  - 小值更“局部敏感”，易过拟合；
  - 大值更平滑，易欠拟合。
- `noise_variance`：
  - 噪声大时应适当增大，提升稳定性。
- `xi`（EI 中探索系数）：
  - 大 `xi` 更探索，小 `xi` 更利用。
- `n_candidates` 与 `n_restarts`：
  - 值越大，采集搜索更充分，但计算更慢。

## R15

方法对比：
- 对比网格搜索：
  - 网格搜索不利用历史信息；BO 会在不确定且可能优的区域加密采样。
- 对比随机搜索：
  - 随机搜索纯探索；BO 通过 EI 在探索与利用间动态平衡。
- 对比梯度法：
  - 梯度法要求可导或可近似梯度；BO 只需函数值，适合真黑盒场景。

## R16

典型应用场景：
- 机器学习超参数调优（学习率、正则化、模型结构参数）。  
- 昂贵仿真参数校准（工程、材料、流体等）。  
- 在线实验中的低预算策略优化。  
- 无解析梯度、评估代价高的问题。

## R17

可扩展方向：
- 扩展到多维输入与批量并行采样（q-EI）。  
- 引入自动超参数学习（最大化 GP 边际似然）。  
- 支持约束贝叶斯优化（可行域概率模型）。  
- 支持异步/并行 worker 场景下的 pending 点处理。  
- 增加 UCB/PI/TS 等采集策略并做 benchmark。

## R18

`demo.py` 的源码级算法流（8 步）：
1. `main` 固定搜索区间 `[0,1]`、初始点与 BO 轮数，调用 `bayesian_optimization`。  
2. `bayesian_optimization` 先在初始点上评估黑盒 `forrester_function`，建立首批观测 `x_obs/y_obs`。  
3. 每轮先调用 `fit_gp_posterior`：
   - 标准化 `y`；
   - 通过 `rbf_kernel` 组装 `K`；
   - 对 `K + (noise+jitter)I` 做 Cholesky；
   - 解线性系统得到 `alpha`。  
4. 调用 `propose_next_point` 做采集优化第一阶段：随机生成候选点，使用 `gp_predict` + `expected_improvement` 批量算 EI，选当前最优候选。  
5. 在同函数内做第二阶段局部精修：在多个种子邻域内用 `golden_section_maximize` 直接最大化 `EI(x)`，拿到更优候选。  
6. 若候选与历史点重复，执行“最远随机点”回退策略，保证新采样点有效。  
7. 回到主循环，查询黑盒得到 `y_next`，把 `(x_next, y_next)` 追加到数据集，并更新历史最优。  
8. 迭代结束后返回最优结果；`main` 再用 `approximate_true_minimum`（密集网格）生成参考最优，输出误差和通过标记。

本实现不依赖 BO 框架与优化器黑盒：GP 建模、EI 计算、采集搜索、数据更新均在源码中可逐步追踪。
