# 贝尔不等式 (Bell's Inequality)

- UID: `PHYS-0231`
- 学科: `物理`
- 分类: `量子基础`
- 源序号: `232`
- 目标目录: `Algorithms/物理-量子基础-0232-贝尔不等式_(Bell's_Inequality)`

## R01

贝尔不等式用于检验“局域实在论”是否能解释纠缠体系的统计相关性。最常用的是 CHSH 形式：

`S = E(a,b) + E(a,b') + E(a',b) - E(a',b')`

其中 `E(x,y)` 是两个观测结果（取值 `±1`）的相关函数。对任意局域隐变量模型，都有 `|S| <= 2`。量子力学允许在特定测量角下达到 `|S| = 2*sqrt(2)`（Tsirelson 上界）。

## R02

本条目目标是给出一个可运行、可验证的最小实现，用数值实验同时展示：

- 局域隐变量模型满足 CHSH 不等式；
- 量子纠缠（单态模型）可违反经典上界；
- 结果可通过固定随机种子复现，便于自动化验证。

## R03

数学设定（二维自旋测量，结果 `A,B in {-1,+1}`）：

1. 相关函数定义：`E(x,y) = <A_x * B_y>`。
2. CHSH 组合：`S = E(a,b) + E(a,b') + E(a',b) - E(a',b')`。
3. 经典局域实在论约束：`|S| <= 2`。
4. 量子单态（singlet）理论相关函数：
   `E_Q(theta_A, theta_B) = -cos(theta_A - theta_B)`。
5. 典型最大违背角度：
   `a=0, a'=pi/2, b=pi/4, b'=-pi/4`，此时 `|S_Q| = 2*sqrt(2)`。

## R04

经典上界的直观证明要点：

- 对每个隐变量样本 `lambda`，四个结果 `A(a,lambda), A(a',lambda), B(b,lambda), B(b',lambda)` 都是确定的 `±1`。
- 构造单次样本量
  `X = A(a)B(b) + A(a)B(b') + A(a')B(b) - A(a')B(b')`。
- 可整理为
  `X = A(a)[B(b)+B(b')] + A(a')[B(b)-B(b')]`。
- 因为 `B(b), B(b')` 为 `±1`，两括号中必有一个为 `0`，另一个为 `±2`，故 `X = ±2`。
- 对样本求平均得到 `S = <X>`，因此必有 `|S| <= 2`。

## R05

复杂度（`n` 为每个测量设定的采样次数）：

- 生成随机样本与计算四个相关函数：`O(n)`；
- 构建结果表（常数行）：`O(1)`；
- 总体时间复杂度：`O(n)`；
- 额外空间复杂度：`O(n)`（向量化采样）。

## R06

MVP 的两条并行路径：

1. **局域隐变量路径**
   - 用共享隐变量 `lambda ~ Uniform(0, 2pi)`；
   - 设定确定性响应 `sign(cos(theta-lambda))`；
   - 逐样本形成 CHSH 单次量 `X` 并取平均，确保经验 `S_local` 严格处于 `[-2,2]`。

2. **量子路径（单态统计）**
   - 直接使用理论相关函数 `E_Q=-cos(Δθ)`；
   - 再按该期望构造 `±1` 配对采样，得到经验相关 `E_Q_emp`；
   - 组合成 `S_quantum`，观察对经典界限的显著违背。

## R07

`demo.py` 的实现重点：

- 不依赖量子专用黑箱库；
- 所有核心公式（CHSH、`E_Q`、局域响应函数）在源码显式实现；
- 输出两张表：角度/相关函数表与 CHSH 汇总表；
- 增加统计显著性估计（`z` 分数与单侧 `p` 值）帮助解释数值违背强度。

## R08

运行环境：

- Python `>=3.10`
- 依赖：`numpy`, `pandas`, `scipy`

其中：

- `numpy` 负责向量化采样与数值运算；
- `pandas` 负责结构化展示输出；
- `scipy.stats` 仅用于把 `z` 分数转成 `p` 值。

## R09

`demo.py` 核心函数接口：

- `local_response(theta, lam) -> np.ndarray`
- `quantum_theory_correlator(theta_a, theta_b) -> float`
- `sample_quantum_correlator(theta_a, theta_b, n_shots, rng) -> float`
- `run_local_hidden_variable_experiment(angles, n_shots, rng) -> dict`
- `compute_chsh(correlators) -> float`
- `build_report(n_shots, seed) -> tuple[pd.DataFrame, pd.DataFrame]`

## R10

测试与校验策略：

1. **不变量校验**：局域模型的单次 `X` 必须满足 `|X|<=2`。
2. **理论一致性**：量子理论 `|S_theory|` 应接近 `2*sqrt(2)`。
3. **经验验证**：固定随机种子下 `|S_quantum_empirical|` 应稳定大于 `2`。
4. **统计解释**：输出 `z` 与 `p`，量化“超过经典界限”的置信程度。

## R11

边界条件与鲁棒性：

- `n_shots` 需为正整数，否则抛出 `ValueError`；
- 量子相关函数采样时，概率 `p_same` 会被夹到 `[0,1]`，避免浮点舍入越界；
- 若标准误差极小导致除零风险，`z` 分数回退为 `inf` 或 `0` 的安全值。

## R12

与量子基础主题的关系：

- Bell/CHSH 不等式是“可观测统计约束”，不依赖具体诠释学立场；
- 违背 `|S|<=2` 表明“局域+预定值”联合假设无法解释实验统计；
- 它与 Kochen-Specker、Leggett-Garg 等基础性 no-go 结果共同构成量子基础核心内容。

## R13

本实现参数选择：

- 角度：`a=0`, `a'=pi/2`, `b=pi/4`, `b'=-pi/4`（CHSH 违背最优组之一）；
- 每设定采样数：`n_shots = 20000`；
- 随机种子：`seed = 2026`。

该配置在普通笔记本上即可快速运行，并给出稳定、显著的量子违背结果。

## R14

工程注意事项：

- 若四个相关函数分别独立采样，局域模型经验 `S` 可能因噪声短暂越界；
- 本实现为局域路径使用“同一批 `lambda` 生成四项”并直接平均 `X`，从构造上避免伪越界；
- 量子路径按设定分别采样是物理上合理的，因为四组测量不可在同一对粒子上同时实现。

## R15

结果解读指引：

- `S_local` 接近 `2` 但不超过 `2`：符合局域隐变量极限；
- `S_quantum_theory` 约为 `2.828`：对应 Tsirelson 上界；
- `S_quantum_empirical` 若显著大于 `2` 且 `p` 很小，说明采样结果支持量子违背。

## R16

可扩展方向：

- 加入探测效率、噪声、可见度参数，研究 loophole 对违背的影响；
- 从 CHSH 扩展到 CH 不等式、Mermin 不等式（多体）；
- 用真实实验计数数据替换合成采样，形成数据分析脚本；
- 增加可视化（角度扫描 vs `S` 曲线）。

## R17

交付内容：

- `README.md`：R01-R18 全部完成；
- `demo.py`：可直接执行、无需交互；
- `meta.json`：保持与任务元数据一致。

运行命令：

```bash
cd Algorithms/物理-量子基础-0232-贝尔不等式_(Bell's_Inequality)
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程（8 步）：

1. **设定 CHSH 角度与随机源**
   初始化 `CHSHAngles`（`a,a',b,b'`）和固定种子 RNG，保证实验可复现。

2. **局域响应函数离散化**
   用 `local_response(theta, lambda)=sign(cos(theta-lambda))` 将连续角度映射到 `±1` 测量结果。

3. **同批隐变量构造四个经典相关项**
   对同一批 `lambda` 同时计算 `A(a),A(a'),B(b),B(b')`，得到 `E_ab,E_ab',E_a'b,E_a'b'`。

4. **逐样本构造 CHSH 单次量并求平均**
   计算 `X = A(a)B(b)+A(a)B(b')+A(a')B(b)-A(a')B(b')`，验证 `X in {-2,+2}` 后取均值得 `S_local`。

5. **量子理论相关函数显式计算**
   对每个角度对计算 `E_Q = -cos(theta_A-theta_B)`，并据此得到理论 `S_quantum_theory`。

6. **按理论相关生成量子采样数据**
   先采样 `A in {±1}`，再按 `P(B=A)=(1+E_Q)/2` 生成 `B`，确保 `E[A*B]=E_Q`，得到经验 `E_Q_emp`。

7. **组合经验 CHSH 与误差估计**
   将四个经验相关函数合成为 `S_quantum_empirical`，并由 `Var(E)= (1-E^2)/n` 估计 `SE(S)`。

8. **显著性计算与结果汇总输出**
   计算 `z=(|S|-2)/SE`，使用 `scipy.stats.norm.sf` 给出单侧 `p` 值，最后用 `pandas` 表格打印完整报告。

说明：本实现未调用任何 Bell 专用黑箱函数；CHSH 组合、局域约束构造、量子相关生成与统计检验均在源码中展开。
