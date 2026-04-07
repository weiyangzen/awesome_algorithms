# CHSH不等式 (CHSH Inequality)

- UID: `PHYS-0232`
- 学科: `物理`
- 分类: `量子基础`
- 源序号: `233`
- 目标目录: `Algorithms/物理-量子基础-0233-CHSH不等式_(CHSH_Inequality)`

## R01

CHSH 不等式是 Bell 不等式中最常用的实验形式之一，用于检验“局域实在论”与量子预测是否兼容。定义四个二值测量设置：

- Alice 端：`a, a'`
- Bob 端：`b, b'`
- 每次测量结果：`A, B in {-1, +1}`

相关函数记作 `E(x,y)=<A_x B_y>`，CHSH 组合为：

`S = E(a,b) + E(a,b') + E(a',b) - E(a',b')`

局域隐藏变量（LHV）模型必须满足：

`|S| <= 2`

## R02

本条目 MVP 目标：

1. 用可追踪的离散算法给出 LHV 上界（不依赖黑盒库）。
2. 对量子 singlet 态计算理想相关函数并得到 `|S| = 2*sqrt(2)`（Tsirelson 上界）。
3. 加一个有限采样版本，模拟实验统计涨落下仍可观测 CHSH 违背。
4. 代码无交互输入，`uv run python demo.py` 直接输出可复核结果。

## R03

数学模型分两部分：

1. 局域隐藏变量模型  
   预先给定 `A(a), A(a'), B(b), B(b') in {-1,+1}`。  
   对任一隐藏变量 `lambda`：
   `S_lambda = A(a)B(b) + A(a)B(b') + A(a')B(b) - A(a')B(b')`  
   可化为：
   `S_lambda = A(a)[B(b)+B(b')] + A(a')[B(b)-B(b')]`  
   因 `B(b), B(b')` 仅取 `+-1`，故 `|S_lambda|=2`，平均后有 `|S|<=2`。

2. 量子 singlet 预测  
   态 `|psi-> = (|01>-|10>)/sqrt(2)`，两端测量算符 `sigma·a` 与 `sigma·b`。  
   相关函数：
   `E(a,b)=<psi-| (sigma·a ⊗ sigma·b) |psi-> = -a·b`  
   取标准角度 `a=0, a'=pi/2, b=pi/4, b'=-pi/4`（同一平面），可得：
   `|S| = 2*sqrt(2) > 2`。

## R04

离散实现思路（对应 `demo.py`）：

- 经典端：穷举 16 个确定性策略 `(A(a),A(a'),B(b),B(b'))`，直接计算每个 `S`，取最大 `|S|`。
- 量子端：显式构造 Pauli 矩阵、`sigma·n`、Kronecker 张量积算符，计算 `<psi|O|psi>`。
- 采样端：对每个理想相关值 `E` 生成 `+-1` 样本，满足 `P(+1)=(1+E)/2`，再估计经验 CHSH。

这样可同时覆盖“理论上界”“理想量子值”“有限统计实验值”三条证据链。

## R05

正确性要点：

1. LHV 部分是穷举而非抽样，因此“最大不超过 2”是严格结论。
2. 量子部分按定义实现 `E(a,b)=<psi|sigma·a⊗sigma·b|psi>`，避免调用高层量子框架黑盒。
3. 选取的角度是 CHSH 最优解之一，理论应达到 Tsirelson 值 `2*sqrt(2)`。
4. 采样部分只影响统计噪声，不改变期望值，样本数足够大时应稳定违背 2。
5. 脚本内置断言：经典不越界、理想值贴合理论、采样值实际违背。

## R06

复杂度（设采样每个设置 `N` 次）：

- LHV 穷举：`16` 个策略，时间 `O(1)`，空间 `O(1)`。
- 理想量子相关：4 个设置，每次 4x4 复矩阵运算，时间 `O(1)`，空间 `O(1)`。
- 采样估计：4 组伯努利抽样，时间 `O(N)`，空间 `O(N)`（当前实现保存临时向量）。

主耗时来自采样阶段，整体近似 `O(N)`。

## R07

标准流程：

1. 计算并输出经典 LHV 最优 `S`。
2. 用 singlet 态 + Pauli 算符计算理想量子 4 个相关函数。
3. 组合得到理想 CHSH 值。
4. 以固定随机种子做有限采样，估计实验 CHSH。
5. 打印三组结果（经典 / 理想量子 / 采样量子）。
6. 执行断言并输出 `All checks passed.`。

## R08

`demo.py` 输入输出约定：

- 输入：无命令行参数、无交互输入（参数写在代码中）。
- 输出：
  - CHSH 定义与 LHV 上界说明；
  - 三组相关函数与 `S` 值；
  - Tsirelson 值 `2*sqrt(2)`；
  - 自动检查通过标记。

依赖仅 `numpy`，保持最小可复现。

## R09

核心函数职责：

- `_chsh_from_correlators`：由四个相关值拼装 `S`。
- `classical_lhv_max_bound`：穷举 16 个确定性局域策略。
- `_pauli` / `_spin_observable` / `_singlet_state`：显式构造量子对象。
- `quantum_correlator`：计算 `<psi|sigma·a⊗sigma·b|psi>`。
- `quantum_ideal_chsh`：使用最优角度得到理想 CHSH。
- `sample_binary_products`：按给定期望 `E` 采样 `+-1` 数据。
- `quantum_sampled_chsh`：得到有限样本估计 CHSH。
- `run_checks`：执行三项一致性断言。

## R10

测试策略（脚本内自动完成）：

1. 经典边界测试：`|S_classical| <= 2`。
2. 理想量子值测试：`||S_ideal|-2*sqrt(2)| < 1e-9`。
3. 实验可观测违背测试：`|S_sampled| > 2` 且有安全裕量（默认 `>2.5`）。

此外固定随机种子确保结果可重现，便于批量验证。

## R11

边界与异常处理：

- `sample_binary_products` 会校验输入相关值必须在 `[-1,1]`。
- 若内部实现错误导致概率越界，将立即抛出 `ValueError`。
- 若数值结果不满足物理预期，`run_checks` 抛出 `AssertionError`。
- 若经典穷举意外无结果（理论上不会发生），抛出 `RuntimeError`。

## R12

与量子基础主题的关系：

- Bell 定理：给出“任何局域实在论都受不等式约束”的可检验形式。
- 纠缠：singlet 态的非经典相关是 CHSH 违背的核心资源。
- 非定域性：CHSH 违背并不允许超光速通信，但否定了局域隐藏变量完备描述。
- Tsirelson 上界：量子理论允许的最大违背是 `2*sqrt(2)`，仍小于一般无信号理论上界 4。

## R13

示例参数（`demo.py`）：

- 测量方向（x-y 平面）：
  - `a = 0`
  - `a' = pi/2`
  - `b = pi/4`
  - `b' = -pi/4`
- 采样配置：
  - `shots_per_setting = 30000`
  - `seed = 7`

该配置通常会输出接近 `2.8` 的采样 `|S|`，稳定高于 2。

## R14

工程实现注意点：

1. 量子相关计算保持“算符级显式实现”，可直接审查每一步线性代数操作。
2. 采样阶段不单独生成 Alice/Bob 原始比特，而直接采样乘积变量 `AB`，这是最小 MVP 的等价简化。
3. 输出同时打印四个相关函数，便于人工核查符号与角度映射是否一致。
4. 随机数使用 `np.random.default_rng(seed)`，避免全局随机状态污染。

## R15

最小验收信号：

- 经典条目 `|S|` 不超过 `2`；
- 理想量子条目 `|S|` 接近 `2.828427`；
- 采样量子条目也明显大于 `2`；
- 末行打印 `All checks passed.`。

这四点构成“从算法到物理结论”的最小闭环。

## R16

可扩展方向：

1. 增加探测效率、噪声和退相干模型，研究违背阈值。
2. 扩展到 CH（Clauser-Horne）或多方 Mermin 不等式。
3. 用真实实验计数数据替代理想相关值采样。
4. 扫描任意测量角，绘制 `S(theta)` 曲线并寻找最优方向。
5. 加入统计置信区间和 p-value，形成实验报告模板。

## R17

交付清单：

- `README.md`：R01-R18 全部填充，含定义、算法、验证与工程说明；
- `demo.py`：可直接运行的 CHSH 最小实现；
- `meta.json`：与本任务元数据保持一致。

运行方式：

```bash
cd Algorithms/物理-量子基础-0233-CHSH不等式_(CHSH_Inequality)
uv run python demo.py
```

## R18

`demo.py` 源码级算法流程（8 步，非黑盒）：

1. `classical_lhv_max_bound` 枚举 `A(a),A(a'),B(b),B(b')` 的 16 种 `+-1` 组合。  
2. 对每个组合调用 `_chsh_from_correlators` 计算 `S`，记录最大 `|S|`。  
3. `_pauli` 和 `_spin_observable` 显式构造 `sigma·n` 测量算符，`_singlet_state` 构造二比特 singlet 态。  
4. `quantum_correlator` 通过 `op = kron(sigma·a, sigma·b)` 和 `<psi|op|psi>` 计算单个理想相关函数。  
5. `quantum_ideal_chsh` 用四组标准角度得到 `(Eab, Eab', Ea'b, Ea'b')` 并组合成理想 `S`。  
6. `sample_binary_products` 把每个理想 `E` 转成 `P(+1)=(1+E)/2`，生成有限样本经验相关值。  
7. `quantum_sampled_chsh` 汇总四个经验相关值，得到采样版 `S`。  
8. `main` 打印三组报告并在 `run_checks` 中断言：经典不越界、理想值等于 Tsirelson、采样值仍违背 CHSH。
