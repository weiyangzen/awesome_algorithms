# Majorization-Minimization

- UID: `MATH-0393`
- 学科: `数学`
- 分类: `优化`
- 源序号: `393`
- 目标目录: `Algorithms/数学-优化-0393-Majorization-Minimization`

## R01

Majorization-Minimization（MM，主化-最小化）是一类“每轮先构造上界代理函数，再精确或近似最小化代理函数”的迭代优化框架。它的核心优势是：
- 原目标可能难直接最小化，但可被一个更易处理的上界函数替代；
- 若上界满足接触条件与上界条件，理论上可保证目标函数单调不增。

本条目给出一个最小可运行 MVP：
- 任务：`L1` 正则逻辑回归；
- MM 形式：用全局二次上界 majorize 平滑 logistic 损失；
- minimization 子问题：闭式近端更新（soft-threshold）。

## R02

本目录实现的问题定义：

- 输入：
  - 特征矩阵 `X in R^(n*d)`；
  - 二分类标签 `y in {0,1}^n`；
  - 正则系数 `lambda >= 0`；
  - 收敛参数 `tol`、`max_iter`。
- 目标函数：
  - `F(w,b) = (1/n) * sum_i [log(1 + exp(z_i)) - y_i * z_i] + lambda * ||w||_1`
  - `z_i = x_i^T w + b`。
- 输出：
  - 参数估计 `(w_hat, b_hat)`；
  - 迭代轨迹 `[(iter, objective, ||grad||, ||step||, nnz(w)), ...]`；
  - 训练/测试目标值、准确率、支持集恢复指标。

`demo.py` 内置固定随机种子和数据生成流程，无需交互输入。

## R03

数学要点（本实现采用的 MM 上界）：

1. 记平滑部分为 `g(theta)`，`theta = [w; b]`。对逻辑回归，`∇g` 是 Lipschitz 连续，常数可取：
   - `L = 0.25 / n * ||X_aug||_2^2`，其中 `X_aug = [X, 1]`。
2. 利用标准不等式（L-smooth）：
   - `g(theta) <= g(theta_k) + <∇g(theta_k), theta-theta_k> + (L/2)||theta-theta_k||^2`。
3. 由此构造 majorizer：
   - `Q_L(theta | theta_k) = g(theta_k) + <∇g(theta_k), theta-theta_k> + (L/2)||theta-theta_k||^2 + lambda*||w||_1`。
4. 最小化 `Q_L` 可得闭式更新：
   - `w_{k+1} = S_{lambda/L}( w_k - (1/L)∇_w g(theta_k) )`
   - `b_{k+1} = b_k - (1/L)∇_b g(theta_k)`
   - `S_tau(.)` 为软阈值算子（逐坐标）。

这等价于“MM 视角下的近端梯度迭代”，但本实现强调的是上界构造与单调性检查。

## R04

算法高层流程：

1. 校验 `X/y` 形状、标签取值、有限性与参数合法性。  
2. 初始化 `w=0, b=0`。  
3. 计算 `L = 0.25/n * ||X_aug||_2^2`。  
4. 计算当前梯度 `∇_w g, ∇_b g`。  
5. 用二次上界对应的闭式解更新 `w,b`（`w` 走 soft-threshold，`b` 走梯度步）。  
6. 计算新目标值并检查是否保持单调不增。  
7. 记录轨迹 `(iter, objective, ||grad||, ||step||, nnz)`。  
8. 满足收敛条件则停止，否则继续迭代。

## R05

核心数据结构：

- `HistoryItem = (iter, objective, grad_norm, step_norm, nnz)`：
  - `iter`：迭代编号；
  - `objective`：当前目标函数值；
  - `grad_norm`：平滑部分梯度范数；
  - `step_norm`：参数更新范数；
  - `nnz`：`w` 的非零元素个数。
- `history: List[HistoryItem]`：保存完整收敛轨迹。
- `data: Dict[str, np.ndarray]`：合成数据及参考稀疏真值（`w_true`）。

## R06

正确性与可审计性要点：

- 上界正确性：`g` 的 L-smooth 不等式保证 `Q_L` 为全局二次 majorizer。  
- 接触条件：当 `theta = theta_k` 时，`Q_L(theta_k | theta_k) = F(theta_k)`。  
- 最小化步骤正确：`w` 子问题是 `L1` 近端，闭式为 soft-threshold；`b` 无正则，直接梯度步。  
- 单调性：理论上 `F(theta_{k+1}) <= F(theta_k)`，代码中显式校验并在违例时抛错。  
- 结果可解释：输出稀疏度、支持集 precision/recall，便于评估 `L1` 的结构恢复效果。

## R07

复杂度分析：

设样本数 `n`、特征数 `d`、迭代轮数 `T`。

- 预处理：估计 `L` 需谱范数计算，通常约 `O(min(n,d)*n*d)`。
- 每轮迭代主成本：
  - 前向与梯度：`X @ w`、`X.T @ residual`，均为 `O(n*d)`；
  - 近端与统计量：`O(d)`。
- 总时间复杂度：`O(precompute_L + T*n*d)`。
- 空间复杂度：
  - 数据存储 `O(n*d)`；
  - 参数与中间量 `O(d)`；
  - 轨迹 `O(T)`。

## R08

边界与异常处理：

- `X` 非二维或含 `nan/inf`：`ValueError`。  
- `y` 非一维、含非法标签（非 0/1）或含 `nan/inf`：`ValueError`。  
- `X/y` 行数不一致：`ValueError`。  
- `lambda < 0`、`tol <= 0`、`max_iter <= 0`：`ValueError`。  
- 估计到的 `L` 非法（非有限或非正）：`RuntimeError`。  
- 迭代中若目标值上升超容差：`RuntimeError`。  
- 达到 `max_iter` 仍未满足停止准则：`RuntimeError`。

## R09

MVP 取舍说明：

- 仅依赖 `numpy`，不调用 `scikit-learn` 或 `scipy.optimize` 黑盒求解器。  
- 选择 `L1` 逻辑回归这一“可解释且可验证”的 MM 场景，而非覆盖全部 MM 变体。  
- 保留完整迭代轨迹与单调性断言，优先保证算法透明度。  
- 不做工程级特性（并行、稀疏矩阵加速、warm-start 批量任务），保持小而诚实。

## R10

`demo.py` 函数职责映射：

- `check_finite_matrix` / `check_binary_vector`：输入合法性检查。  
- `sigmoid`：数值稳定 sigmoid。  
- `soft_threshold`：`L1` 近端算子。  
- `logistic_objective`：目标函数计算。  
- `logistic_grad`：平滑项梯度计算。  
- `estimate_lipschitz_constant`：按 `X_aug` 谱范数估计全局 `L`。  
- `mm_l1_logistic`：MM 主循环（majorize + minimize）。  
- `make_synthetic_data`：固定种子生成可复现实验数据。  
- `support_metrics`：支持集恢复评估。  
- `main`：组织流程、打印轨迹和摘要。

## R11

运行方式：

```bash
cd Algorithms/数学-优化-0393-Majorization-Minimization
python3 demo.py
```

脚本不会读取交互输入，也不依赖命令行参数。

## R12

输出字段说明：

- 迭代表格：
  - `iter`：迭代编号；
  - `objective`：当前目标值 `F(w,b)`；
  - `||grad||`：平滑项梯度范数；
  - `||step||`：参数更新范数；
  - `nnz(w)`：当前 `w` 非零元素数量。
- 结果摘要：
  - `train/test objective`：训练/测试目标值；
  - `train/test accuracy`：训练/测试准确率；
  - `nnz(w_hat)`：最终稀疏度；
  - `support precision/recall`：支持集恢复质量；
  - `objective monotone non-increasing`：MM 单调性检查结果。

## R13

最小测试集（脚本内置）：

1. 随机稀疏线性可分度中等的二分类数据：
   - `n_samples=640`，`n_features=24`，固定随机种子。  
2. 训练/测试切分：
   - `train_ratio=0.75`。  
3. 评估维度：
   - 收敛轨迹是否单调；
   - 训练/测试准确率；
   - 支持集恢复 precision/recall。

建议补充异常测试：
- `y` 放入 `{-1,1}` 标签（应报错）；
- `lam=-1`（应报错）；
- 人工破坏 `X` 含 `nan`（应报错）。

## R14

关键参数与调参建议：

- `lam`：稀疏强度（越大越稀疏，偏差通常越大）。  
- `tol`：停止阈值（越小越严格）。  
- `max_iter`：最大迭代轮数。  
- `support_metrics` 中 `thresh`：判定非零系数的阈值。

经验建议：
- 先用中等 `lam`（如 `0.05 ~ 0.1`）观察 `nnz` 和 accuracy；
- 若出现过稀疏导致准确率下降，可下调 `lam`；
- 若追求更稳定收敛判断，可减小 `tol` 并增大 `max_iter`。

## R15

与相关方法对比：

- 对比纯梯度下降：MM/近端步骤可直接处理 `L1` 非光滑项，且具单调下降保证。  
- 对比经典 IRLS：IRLS 也可视作 MM，但常依赖线性系统求解；本实现每轮仅需矩阵向量乘，更轻量。  
- 对比牛顿/L-BFGS：二阶法在某些问题收敛更快，但实现与数值维护更复杂。  
- 对比 `sklearn` 黑盒：黑盒调用更短，但难展示“上界构造 -> 子问题最小化”的细节链路。

## R16

典型应用场景：

- 稀疏特征选择（高维二分类）。  
- 需要可解释稀疏模型且希望稳定下降的工业基线。  
- 教学场景中展示 MM 框架与近端算法的统一视角。  
- 作为更复杂 MM/EM/DC 算法的可验证起点。

## R17

可扩展方向：

- 加入 backtracking 版本 MM，自适应估计局部 `L`。  
- 扩展为 Elastic Net（`L1 + L2`）或 Group Lasso。  
- 支持稀疏矩阵输入与批量 warm-start。  
- 增加 ROC-AUC、PR-AUC 与正则路径（lambda grid）评估。  
- 用 PyTorch/JAX 自动微分替换手写梯度，保留 MM 近端更新框架。

## R18

`demo.py` 源码级算法流程（8 步）：

1. `main` 调用 `make_synthetic_data` 生成固定随机种子的训练/测试集与稀疏真值参数。  
2. `main` 设置 `lam/tol/max_iter` 后进入 `mm_l1_logistic`。  
3. `mm_l1_logistic` 先用 `check_finite_matrix`、`check_binary_vector` 做输入合法性验证，并初始化 `w=0,b=0`。  
4. `estimate_lipschitz_constant` 计算 `X_aug` 的谱范数，得到 logistic 平滑项的全局 Lipschitz 常数 `L`，即 majorizer 的二次系数。  
5. 每轮调用 `logistic_grad` 得到 `∇_w g` 与 `∇_b g`，并构造代理函数最小点：`w_tilde = w - grad_w/L`、`b_tilde = b - grad_b/L`。  
6. 对 `w_tilde` 调用 `soft_threshold(w_tilde, lam/L)` 得到 `w_next`，并取 `b_next=b_tilde`，完成 minimization 子问题。  
7. 通过 `logistic_objective` 计算新目标值，检查是否不大于上一轮（允许 `1e-12` 数值容差），再记录 `(iter, objective, ||grad||, ||step||, nnz)` 到 `history`。  
8. 返回 `w_hat,b_hat,history` 后，`main` 计算 accuracy、支持集 precision/recall 与单调性布尔值并打印汇总。
