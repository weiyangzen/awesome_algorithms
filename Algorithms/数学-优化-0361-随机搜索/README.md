# 随机搜索

- UID: `MATH-0361`
- 学科: `数学`
- 分类: `优化`
- 源序号: `361`
- 目标目录: `Algorithms/数学-优化-0361-随机搜索`

## R01

随机搜索（Random Search）是最基础的无梯度优化方法之一：
- 在给定可行域内随机采样候选解；
- 评估目标函数；
- 持续保留当前最优解。

本条目实现一个最小可运行、可复现、可审计的 Python MVP，用于连续变量的有界最小化问题。

## R02

本实现求解的问题定义：
- 输入：
  - 目标函数 `f(x)`（本 MVP 使用向量化批评估接口）；
  - 盒约束 `bounds`，形状 `(d,2)`，每维含 `[low_i, high_i]`；
  - 采样预算 `n_samples`；
  - 随机种子 `seed`；
  - 批大小 `batch_size`（控制一次评估多少样本）。
- 输出：
  - 发现的最优解 `best_x`；
  - 最优目标值 `best_value`；
  - 历史检查点 `[(samples_seen, best_value_so_far), ...]`。

## R03

数学基础（连续盒约束场景）：

1. 目标：
   `min f(x), x in Omega`，其中
   `Omega = [l_1,u_1] x ... x [l_d,u_d]`。

2. 随机搜索采样：
   - 令 `U^(k) ~ Uniform(Omega)`，独立同分布；
   - 用 `f(U^(k))` 评估候选优劣。

3. 当前最好值序列：
   `B_n = min_{1<=k<=n} f(U^(k))`。

4. 性质（直观）：
   - `B_n` 随 `n` 非增；
   - 在温和条件下，随样本增多，命中更优区域的概率提升。

随机搜索不依赖梯度和凸性，是可靠的基线方法。

## R04

算法流程（MVP）：
1. 检查 `bounds/n_samples/batch_size` 合法性。  
2. 用 `numpy.random.default_rng(seed)` 初始化随机源。  
3. 按批次生成 `k x d` 的 `U(0,1)` 随机数。  
4. 线性映射到真实边界：`X = low + U * (high-low)`。  
5. 批量评估 `values = f(X)`。  
6. 取本批最优并与全局最优比较，必要时更新 `best_x/best_value`。  
7. 在检查点记录“已评估样本数 + 当前最佳值”。  
8. 达到 `n_samples` 后输出结果。

## R05

核心数据结构：
- `bounds: np.ndarray(shape=(d,2))`：每维上下界；
- `X_batch: np.ndarray(shape=(k,d))`：本批候选样本；
- `values: np.ndarray(shape=(k,))`：本批目标值；
- `SearchResult(dataclass)`：
  - `best_x`、`best_value`、`samples_evaluated`、`history`。

## R06

正确性要点：
- 每轮更新都执行 `best_value = min(best_value, batch_best)`，因此最优记录不会变差；
- 最终 `best_value` 等于所有已评估样本目标值的最小值；
- 固定 `seed` 时，采样序列确定，实验可复现；
- 该方法不保证有限步达到全局最优，但保证给定预算下返回“已见最优”。

## R07

复杂度分析：
- 设维度 `d`，样本总数 `N`，一次函数评估均摊成本为 `C_f(d)`；
- 时间复杂度：`O(N * (d + C_f(d)))`（采样映射 + 目标评估）；
- 空间复杂度：`O(batch_size * d)`（批处理缓存）+ `O(log_points)`（历史）。

## R08

边界与异常处理：
- `bounds` 不是二维 `(d,2)` 或含 `nan/inf`：抛 `ValueError`；
- 某维 `low >= high`：抛 `ValueError`；
- `n_samples <= 0` 或 `batch_size <= 0`：抛 `ValueError`；
- 目标函数输出形状不为 `(k,)` 或含非有限值：抛 `RuntimeError`。

## R09

MVP 取舍：
- 只实现“纯随机均匀采样”基线，不加入自适应分布、退火或进化机制；
- 不调用 `scipy.optimize` 黑盒优化器，保留完全可读的源码流程；
- 使用 `numpy` 完成采样与向量计算，保证代码短小且可运行。

## R10

`demo.py` 模块职责：
- `validate_bounds`：检查边界矩阵合法性；
- `make_checkpoints`：生成历史记录检查点；
- `random_search`：执行随机搜索主循环并返回 `SearchResult`；
- `sphere_batch/rastrigin_batch/shifted_quadratic_batch`：三个演示目标函数；
- `run_case`：运行单案例并打印最优值、解向量、误差指标；
- `main`：组织固定案例并输出汇总结论。

## R11

运行方式：

```bash
cd Algorithms/数学-优化-0361-随机搜索
python3 demo.py
```

脚本不需要交互输入，直接打印三组固定实验结果。

## R12

输出字段说明：
- `best_value`：当前预算下找到的最小目标值；
- `best_x`：对应最优点；
- `known_optimum`：基准函数已知最优值（本例均为 `0`）；
- `optimality_gap`：`best_value - known_optimum`；
- `history checkpoints`：不同采样进度下的最佳值轨迹。

解读重点：
- `best_value` 应随样本数增加呈非增趋势；
- 多峰函数（如 Rastrigin）上，改进通常更慢且抖动更明显。

## R13

最小测试集（内置）：
1. `Sphere-2D`：单峰凸函数，验证基本收敛趋势；
2. `Rastrigin-2D`：多峰函数，验证全局探索能力；
3. `ShiftedQuadratic-5D`：维度略高、最优点不在原点，验证边界映射与泛化。

建议补充异常测试：
- 非法 `bounds`（如上下界倒置）；
- `n_samples=0`；
- 人为构造返回 `nan` 的目标函数。

## R14

可调参数与影响：
- `n_samples`：总预算，越大通常结果越好但耗时增加；
- `batch_size`：批处理大小，影响内存占用与向量化效率；
- `seed`：控制可复现性；
- `log_points`：历史记录粒度，影响日志详细程度。

实践建议：
- 先用小预算确认流程正确，再提高 `n_samples` 观察性能边界；
- 对高维问题，优先提高总预算而不是过度增大 `batch_size`。

## R15

方法对比：
- 对比网格搜索：
  - 网格在高维下组合爆炸；
  - 随机搜索在同预算下覆盖更灵活。
- 对比梯度下降：
  - 梯度法在光滑问题通常更高效；
  - 随机搜索不依赖导数，对非光滑/黑盒目标更稳健。
- 对比贝叶斯优化：
  - 贝叶斯优化样本效率更高，但实现复杂；
  - 随机搜索实现最简单，适合基线与并行粗搜。

## R16

典型应用：
- 黑盒函数调参（仿真器、不可导评分函数）；
- 超参数粗搜索的第一阶段基线；
- 作为复杂元启发式算法（SA/GA/PSO）的对照组；
- 教学中展示“预算-性能”关系。

## R17

扩展方向：
- 重要性采样或分层采样，提高命中高价值区域概率；
- 多起点局部优化混合（Random Search + Local Search）；
- 并行分布式采样；
- 引入约束处理（惩罚法/可行性投影）；
- 增加日志落盘与收敛可视化。

## R18

`demo.py` 源码级算法流（7 步，非黑盒）：
1. `main` 定义三组基准目标函数、边界、样本预算、随机种子与已知最优值。  
2. `run_case` 调用 `random_search`，并在返回后计算 `optimality_gap`。  
3. `random_search` 先执行 `validate_bounds`，提取 `low/high` 与维度信息。  
4. 通过 `default_rng(seed)` 生成 `U(0,1)` 批样本，再线性映射到真实搜索空间。  
5. 调用目标函数的批接口得到 `values`，检查形状与有限性。  
6. 用 `argmin(values)` 找本批最优，并与全局最优比较后更新 `best_x/best_value`。  
7. 按检查点记录历史，循环直到样本预算耗尽，最终返回 `SearchResult` 并打印结果。
