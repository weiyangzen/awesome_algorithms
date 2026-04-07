# Cox比例风险模型

- UID: `MATH-0288`
- 学科: `数学`
- 分类: `生存分析`
- 源序号: `288`
- 目标目录: `Algorithms/数学-生存分析-0288-Cox比例风险模型`

## R01

Cox 比例风险模型（Cox Proportional Hazards Model）用于建模右删失生存数据中“协变量与风险率”的关系。它的核心思想是：
- 个体 `i` 的风险函数写为 `h(t|x_i) = h0(t) * exp(x_i^T beta)`；
- 其中 `h0(t)` 是未知基线风险函数，不需要参数化；
- `beta` 通过部分似然（partial likelihood）估计。

本条目提供一个最小可运行 MVP：
- 手写负对数部分似然、梯度、Hessian；
- 用 Newton-Raphson + 回溯线搜索拟合参数；
- 额外给出 Breslow 基线风险估计与 C-index 验证。

## R02

本目录实现的问题定义：
- 输入：
  - `X in R^(n*p)`：特征矩阵；
  - `time in R^n`：观测时间（必须为正）；
  - `event in {0,1}^n`：事件指示（1=事件发生，0=删失）；
  - `l2`、`max_iter`、`tol` 等训练参数。
- 输出：
  - 拟合参数 `beta_hat`；
  - 优化历史（`nll`、`grad_norm`、`step_norm`、`line_search_alpha`）；
  - Breslow 基线累计风险曲线；
  - C-index 评估值。

`demo.py` 直接生成可复现合成数据并运行，不需要交互输入。

## R03

数学基础（Breslow ties 近似）：

1. 比例风险模型：
   `h(t|x) = h0(t) * exp(x^T beta)`。
2. 部分对数似然：
   `l(beta) = sum_{i:event_i=1} [x_i^T beta - log(sum_{j:time_j>=time_i} exp(x_j^T beta))]`。
3. 负对数部分似然：
   `NLL(beta) = -l(beta) + (l2/2)*||beta||^2`。
4. 梯度：
   `grad = -sum_{i:event_i=1}(x_i - E_beta[x|R_i]) + l2*beta`。
5. Hessian：
   `H = sum_{i:event_i=1} Cov_beta[x|R_i] + l2*I`，半正定。

由于优化目标是凸（加 `l2` 时严格更稳定），Newton 迭代通常能快速收敛。

## R04

算法流程（高层）：
1. 检查输入维度、有限性、时间正值与事件标签合法性。  
2. 初始化 `beta = 0`。  
3. 每轮计算 `NLL/grad/Hessian`。  
4. 若 `||grad||` 小于阈值则停止。  
5. 解线性方程 `H * step = grad` 得 Newton 方向。  
6. 用 Armijo 回溯线搜索找步长 `alpha`。  
7. 更新 `beta <- beta - alpha*step`，记录历史。  
8. 结束后计算 C-index 与 Breslow 基线累计风险并打印结果。

## R05

核心数据结构：
- `CoxFitResult`（`dataclass`）：
  - `beta: np.ndarray`，最终参数；
  - `history: pd.DataFrame`，迭代轨迹；
  - `converged: bool`，是否触发收敛条件。
- `history` 的列：
  - `iter`：迭代轮次；
  - `nll`：当前负对数部分似然；
  - `grad_norm`：梯度范数；
  - `step_norm`：参数步长范数；
  - `line_search_alpha`：回溯线搜索最终步长。
- `baseline`（`pd.DataFrame`）：
  - `time`、`events_at_time`、`baseline_hazard_increment`、`baseline_cumulative_hazard`。

## R06

正确性要点：
- 目标函数与导数一致：`cox_nll_grad_hess` 同时输出 `NLL/grad/Hessian`，避免实现不一致。  
- 数值稳定：风险集分母使用 `log-sum-exp` 形式（减去 `max_eta` 再指数化）。  
- 下降保证：回溯线搜索使用 Armijo 条件，避免 Newton 步过大导致发散。  
- 凸性保障：`l2*I` 使 Hessian 更稳健，减轻共线性下的病态问题。  
- 可验证：脚本输出真值参数 vs 估计参数误差、以及 C-index。

## R07

复杂度分析（`n` 样本，`p` 特征）：
- 当前实现采用朴素风险集循环：
  - 每个事件样本都构造风险集并计算二阶矩，单次近似 `O(n*p^2)`；
  - 最坏总计 `O(n^2*p^2)`。
- Newton 迭代 `T` 轮时，总时间约 `O(T*n^2*p^2)`。  
- 空间复杂度：
  - 数据主存储 `O(np)`；
  - Hessian `O(p^2)`；
  - 历史 `O(T)`。

该复杂度适合教学与中小规模 MVP，不适合超大样本生产场景。

## R08

边界与异常处理：
- `X` 非二维、`time/event` 非一维、样本数不一致：抛 `ValueError`。  
- `time <= 0`、`event` 非 0/1、无事件样本：抛 `ValueError`。  
- `beta` 维度不匹配、`l2 < 0`、`max_iter <= 0`、`tol <= 0`：抛 `ValueError`。  
- 迭代中出现非有限 `nll/grad/hess`：抛 `RuntimeError`。  
- Hessian 奇异时：增加微小阻尼后再解线性方程。

## R09

MVP 取舍说明：
- 只处理右删失数据，不含左截断、区间删失、多状态过程。  
- ties 使用 Breslow 近似，不实现 Efron/Exact。  
- 优化器使用手写 Newton + 线搜索，不调用 survival 黑盒库。  
- 基线风险使用 Breslow 估计，展示可解释的风险分解结果。  
- 重点是“可读、可审计、可运行”的最小实现。

## R10

`demo.py` 函数职责：
- `validate_inputs`：检查输入合法性。  
- `make_synthetic_data`：生成带删失的可复现合成生存数据。  
- `cox_nll_grad_hess`：核心数学，计算 NLL/梯度/Hessian。  
- `cox_neg_log_partial_likelihood`：仅计算 NLL（用于线搜索）。  
- `fit_cox_newton`：Newton-Raphson 主循环。  
- `breslow_baseline_hazard`：估计基线风险增量与累计风险。  
- `concordance_index`：计算 Harrell C-index。  
- `main`：串联数据生成、训练、评估和打印。

## R11

运行方式：

```bash
cd Algorithms/数学-生存分析-0288-Cox比例风险模型
uv run python demo.py
```

脚本无命令行参数依赖，也不会请求任何交互输入。

## R12

输出字段说明：
- `n_samples/n_features`：数据规模。  
- `events/censoring_rate`：事件数量与删失比例。  
- `converged/iters`：优化是否收敛、迭代次数。  
- `[Coefficient recovery]`：
  - `coef_true`：合成数据真实系数；
  - `coef_hat`：估计系数；
  - `abs_error`：绝对误差。  
- `[Optimization trace]`：训练轨迹头尾部。  
- `C-index`：风险排序能力指标（越大越好，随机约 0.5）。  
- `baseline_*`：Breslow 基线风险增量与累计风险。

## R13

建议最小测试覆盖：
- 正常流程：默认数据、默认参数，验证可运行和收敛。  
- 正则强度测试：`l2` 取 `0` 与较大值，观察系数收缩与稳定性。  
- 维度鲁棒性：修改 `n_features`，验证函数适配性。  
- 异常输入：
  - `time` 含非正数；
  - `event` 含非 0/1；
  - `X/time/event` 行数不一致。

## R14

关键可调参数：
- `l2`：L2 正则强度。  
- `max_iter`：最大迭代轮数。  
- `tol`：收敛阈值（同时影响梯度/步长停止）。  
- `n_samples`、`n_features`：数据规模。  
- 合成数据参数：`baseline_rate`、删失尺度分位点。

调参建议：
- 收敛不稳定：增大 `l2` 或放宽 `tol`。  
- 收敛过慢：提高 `max_iter`，并检查特征尺度是否标准化。  
- 系数波动大：增大样本量或降低删失比例。

## R15

方法对比：
- 对比参数生存模型（如 Weibull AFT）：
  - AFT 需指定分布；
  - Cox 不需指定基线风险分布，更灵活。  
- 对比 Kaplan-Meier：
  - KM 适合无协变量的总体生存曲线；
  - Cox 可处理多协变量风险分层。  
- 对比黑盒库实现：
  - 黑盒接口更简洁；
  - 本实现更透明，便于教学、审计与二次开发。

## R16

典型应用场景：
- 医疗随访：患者特征对事件风险的相对影响评估。  
- 金融风控：客户违约/流失时间建模。  
- 工业可靠性：设备失效时间与工况因素关联分析。  
- 互联网产品：用户留存/流失时间与行为特征关系分析。

## R17

可扩展方向：
- ties 处理升级到 Efron / Exact。  
- 增加分层 Cox（stratified Cox）和时间变系数。  
- 用前缀和或排序技巧将风险集计算降到更低复杂度。  
- 增加稳健标准误、置信区间与显著性检验。  
- 支持批量实验与结果落盘（CSV/Parquet）。

## R18

`demo.py` 的源码级算法流程（8 步）：
1. `main` 调用 `make_synthetic_data` 生成标准化特征、观测时间、删失标签和真实 `beta`。  
2. `fit_cox_newton` 初始化 `beta=0`，并在每轮调用 `cox_nll_grad_hess` 计算当前 `NLL/grad/Hessian`。  
3. `cox_nll_grad_hess` 对每个事件样本构造风险集 `R_i={j|time_j>=time_i}`，计算 `s0/s1/s2`。  
4. 利用 `s0/s1/s2` 累加负对数部分似然、梯度与 Hessian，并叠加 `l2` 正则项。  
5. 回到 `fit_cox_newton`，解线性方程 `H*step=grad` 得 Newton 方向；若 Hessian 病态则加微阻尼。  
6. 使用 `cox_neg_log_partial_likelihood` 做 Armijo 回溯线搜索，找到满足下降条件的 `alpha`。  
7. 更新 `beta <- beta - alpha*step`，记录 `iter/nll/grad_norm/step_norm/alpha` 到 `history`，按阈值判停。  
8. 训练结束后，`main` 用 `breslow_baseline_hazard` 估计基线累计风险，并用 `concordance_index` 评估排序性能，最后打印系数恢复与轨迹摘要。
