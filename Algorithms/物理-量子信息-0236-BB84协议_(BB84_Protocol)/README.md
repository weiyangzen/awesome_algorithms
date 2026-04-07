# BB84协议 (BB84 Protocol)

- UID: `PHYS-0235`
- 学科: `物理`
- 分类: `量子信息`
- 源序号: `236`
- 目标目录: `Algorithms/物理-量子信息-0236-BB84协议_(BB84_Protocol)`

## R01

问题定义：实现一个可运行、可追踪、无量子 SDK 黑盒依赖的 BB84 最小 MVP。

本条目的目标不是做实验级硬件仿真，而是用最小代码闭环复现 BB84 的核心统计行为：
- 无窃听时 QBER（量子误码率）低，安全密钥长度为正；
- intercept-resend 窃听时 QBER 升高，渐近密钥率下降到 0。

## R02

BB84 的核心是“编码基随机 + 错基测量扰动可检出”：
- Alice 在 `Z/X` 两组共轭基中随机选择一组编码比特；
- Bob 随机选择测量基；
- 仅保留 Alice/Bob 同基测量的样本（sifting）；
- 公布其中一部分用于参数估计（QBER），剩余部分用于纠错与隐私放大。

如果 Eve 进行 intercept-resend：
- Eve 不知道 Alice 基，只能随机测量；
- Eve 错基测量会随机化态并向 Bob 传播额外误码；
- 该扰动在公开抽样的 QBER 中可见。

## R03

本目录 MVP 的功能边界：
1. 采用单量子比特态矢量（`2` 维复向量）显式表示每个脉冲；
2. 显式实现 `Z/X` 基态准备、投影测量、测后态塌缩；
3. 显式实现 Eve 截获重发与信道 bit-flip 噪声；
4. 完成 sifting、QBER 抽样估计、渐近密钥率估计；
5. 使用二元随机哈希矩阵执行隐私放大。

这保证了协议主流程可逐函数追踪，而不是调用“几行 API 即得结果”的黑箱。

## R04

脚本使用的关键数学对象与公式：

1. 基态：
`|0> = [1,0]^T`, `|1> = [0,1]^T`

2. X 基态：
`|+> = (|0> + |1>) / sqrt(2)`
`|-> = (|0> - |1>) / sqrt(2)`

3. 测量概率（以 X 基为例）：
`p(0) = |<+|psi>|^2`, `p(1) = |<-|psi>|^2`

4. 二元熵：
`H2(p) = -p log2 p - (1-p) log2(1-p)`

5. 渐近 BB84 密钥率近似：
`r = max(0, 1 - 2H2(Q))`
其中 `Q` 为估计 QBER。

## R05

复杂度分析（`N` 为发送脉冲数，`n_raw` 为筛选后原始密钥长度）：
- 量子态准备/测量循环：`O(N)`；
- sifting 与 QBER 统计：`O(N)`；
- 隐私放大（二元矩阵乘法）：`O(final_key_len * n_raw)`；
- 空间复杂度：`O(N + final_key_len * n_raw)`。

默认参数下（`N=4096`）运行很快，适合本地验证与回归测试。

## R06

`demo.py` 的协议过程：
1. Alice 生成随机比特与随机基；
2. 每个脉冲构造 2 维量子态（`|0>|1>|+>|->` 之一）；
3. Eve 以概率 `eve_intercept_prob` 执行拦截重发；
4. 信道以概率 `channel_bit_flip` 施加 Pauli-X；
5. Bob 随机选基测量，得到测量比特；
6. 公共信道公开基，保留同基样本；
7. 抽样估计 QBER，剩余位作为 raw key；
8. 估计纠错泄露并做隐私放大，得到最终密钥。

## R07

建模假设（MVP 级）：
- 单光子抽象，不含多光子脉冲与诱骗态；
- 噪声只含独立 bit-flip，不含偏振漂移、探测器暗计数等器件级效应；
- 纠错阶段使用信息泄露近似，不实现 CASCADE/LDPC 实码；
- 安全率采用渐近公式，不包含 finite-key 修正。

因此该实现用于教学与算法验证，不是工程部署安全评估。

## R08

运行环境与依赖：
- Python `>=3.10`
- `numpy`
- `pandas`

运行方式：

```bash
cd Algorithms/物理-量子信息-0236-BB84协议_(BB84_Protocol)
uv run python demo.py
```

无需交互输入。

## R09

脚本输出两组对照场景：
- `clean_channel`：`eve_intercept_prob=0.0`
- `intercept_resend`：`eve_intercept_prob=1.0`

每组给出：
- `sifted_len`、`sample_size`、`raw_len`
- `qber_sample`、`qber_raw_before_ec`、`qber_total_sifted`
- `secret_fraction`、`leak_ec_bits`、`final_key_len`
- `eve_intercepted`、`channel_flips`

最后打印 `All checks passed.` 表示断言全部通过。

## R10

正确性检查（内置断言）：
1. clean 场景 `qber_sample < 8%`；
2. intercept-resend 场景 `qber_sample > 18%`；
3. intercept-resend 场景 `final_key_len == 0`；
4. clean 场景 `final_key_len > 64`。

这组断言约束了 BB84 的核心趋势：
- 无窃听时可提取密钥；
- 强窃听时安全率坍缩。

## R11

默认参数（`BB84Params`）：
- `n_qubits = 4096`
- `channel_bit_flip = 0.01`
- `sample_fraction = 0.20`
- `seed = 2026`

场景切换只改 `eve_intercept_prob`：
- clean: `0.0`
- attacked: `1.0`

参数意图：
- 4096 让统计波动足够小；
- 1% 噪声避免“完全理想信道”假设；
- 强攻击场景保证 QBER 提升显著。

## R12

一次运行（默认参数）通常呈现如下规律：
- clean：`qber_sample` 接近低百分比，`final_key_len` 显著大于 0；
- intercept-resend：`qber_sample` 常在约 `20%~30%` 区间，`final_key_len` 变为 0。

由于脚本固定随机种子，指标可复现，便于回归测试。

## R13

安全解释边界：
- 本实现验证的是“协议统计现象”而非“完整可组合安全证明”；
- 渐近密钥率公式适合大样本近似，不代表有限码长最优界；
- 纠错开销用 `n_raw * H2(Q)` 近似，未包含具体码效率与交互轮次。

因此该结果应被解读为算法级 sanity check。

## R14

常见失败模式：
1. 基编码映射错误（把 `|+>|->` 与 bit 对应写反）；
2. 测量后态未按测量基重置，导致统计偏差；
3. sifting 掩码误用（未严格使用 `alice_bases == bob_bases`）；
4. QBER 抽样与留存索引重叠，污染参数估计；
5. 密钥率公式未做 `max(0, ·)` 截断，导致负长度。

这些问题都会直接反映为断言失败或统计异常。

## R15

可扩展方向：
- 引入 depolarizing/phase-flip 噪声并比较不同误差来源；
- 引入有限码长修正与置信区间估计；
- 使用 Toeplitz 哈希替代密集随机矩阵以降低隐私放大开销；
- 扩展到 decoy-state BB84 与器件模型。

## R16

相关算法/主题：
- 量子密钥分发（QKD）总体框架；
- E91 协议与基于纠缠的参数估计；
- 量子不可克隆定理；
- 纠错码与隐私放大；
- 量子网络与密钥中继。

## R17

本目录交付物：
- `README.md`：R01-R18 完整说明；
- `demo.py`：可直接运行的 BB84 MVP；
- `meta.json`：任务元数据。

自检命令：

```bash
cd Algorithms/物理-量子信息-0236-BB84协议_(BB84_Protocol)
uv run python demo.py
```

预期：打印两组场景表格并最终显示 `All checks passed.`。

## R18

`demo.py` 源码级算法流（8 步）：
1. `BB84Params` 固定样本规模、噪声、窃听概率与随机种子，建立可复现实验配置。
2. `state_from_bit_basis` 将 `(bit, basis)` 显式映射到 `|0>|1>|+>|->` 四个纯态，完成 Alice 制备。
3. `measurement_probabilities` 与 `measure_state` 显式计算投影概率并生成测量结果/测后态，分别用于 Eve 与 Bob 的测量。
4. `simulate_bb84` 在每个脉冲上执行 Eve intercept-resend：随机选基测量并按测得比特重发新态。
5. `apply_channel_noise` 按独立概率施加 Pauli-X，模拟 bit-flip 信道。
6. Bob 随机选基测量得到 `bob_bits`，随后按 `alice_bases == bob_bases` 做 sifting，并拆分“公开抽样位”和“留存 raw key”。
7. 用抽样位计算 `qber_sample`，再通过 `binary_entropy` 与 `r=max(0,1-2H2(Q))` 估计纠错泄露、安全率与 `final_key_len`。
8. `privacy_amplification` 构造随机二元哈希矩阵并执行模 2 乘法压缩密钥，`run_sanity_checks` 对 clean/attacked 行为做断言，`main` 输出结果。

说明：脚本只使用 `numpy/pandas` 做线性代数与表格化，协议关键步骤（态准备、测量、扰动、筛选、密钥率、隐私放大）都在源码中显式展开。
