# 量子密钥分发 (Quantum Key Distribution)

- UID: `PHYS-0234`
- 学科: `物理`
- 分类: `量子信息`
- 源序号: `235`
- 目标目录: `Algorithms/物理-量子信息-0235-量子密钥分发_(Quantum_Key_Distribution)`

## R01

问题定义：实现一个可运行、可追踪的 BB84 量子密钥分发最小 MVP。

目标是用离散随机仿真复现 BB84 的核心现象：
- 无窃听时，抽样量子误码率（QBER）较低，可提取正长度安全密钥；
- 拦截重发（intercept-resend）窃听时，QBER 显著上升，安全密钥率坍缩到 0。

## R02

理论背景：BB84 基于“非正交基不可同时无扰测量”与“测量扰动可检测”。

协议核心结构：
- Alice 随机选择比特与基（Z/X）制备量子态；
- Bob 随机选基测量；
- 公共信道仅公开基，不公开比特，并筛选同基事件（sifting）；
- 用部分筛选位做参数估计得到 QBER，再执行纠错与隐私放大。

## R03

本 MVP 计算任务：
1. 生成 Alice/Bob 基与比特；
2. 建模 Eve 的 intercept-resend 攻击；
3. 建模物理信道独立比特翻转噪声；
4. 执行基筛选和抽样 QBER 估计；
5. 依据 BB84 渐近密钥率公式估计可提取密钥长度；
6. 用二元随机哈希矩阵做隐私放大输出最终密钥。

## R04

建模假设（MVP 级别）：
- 单光子、无多光子脉冲、无探测器效率失配；
- 基集合仅 `Z/X`，并用离散概率规则代替连续量子态波函数演化；
- 信道噪声简化为独立 bit-flip，概率 `channel_bit_flip`；
- 纠错阶段不实现完整级联/LDPC，仅做泄露位数估计与“理想纠错后密钥一致”近似。

这些假设适用于算法教学与 sanity check，不等价于工程级 QKD 系统安全评估。

## R05

关键公式：

1. 二元熵函数：
`H2(p) = -p log2 p - (1-p) log2(1-p)`

2. BB84 渐近密钥率近似（对称误码、单向纠错）：
`r = max(0, 1 - 2H2(Q))`

其中 `Q` 取抽样估计 `qber_sample`。在脚本中：
- `leak_ec_bits = ceil(n_raw * H2(Q))`
- `final_key_len = floor(r * n_raw)`

## R06

`demo.py` 的协议流程：
1. Alice 生成 `alice_bits` 与 `alice_bases`；
2. Eve 按 `eve_intercept_prob` 决定是否拦截并随机选基测量、重发；
3. 信道按 `channel_bit_flip` 注入翻转错误；
4. Bob 随机选基测量得到 `bob_bits`；
5. 公开基并筛选 `alice_bases == bob_bases` 的位；
6. 公布随机抽样子集估计 `qber_sample`；
7. 估计纠错泄露并计算密钥率；
8. 用随机二元矩阵模 2 乘法做隐私放大。

## R07

复杂度分析（设发送比特数为 `N`，原始密钥长度约 `n_raw`）：
- 时间复杂度：`O(N + final_key_len * n_raw)`，主开销是隐私放大矩阵乘法；
- 空间复杂度：`O(N + final_key_len * n_raw)`，主要来自中间数组和哈希矩阵。

在默认参数下（`N=4096`），运行耗时很低，适合本地快速验证。

## R08

数值与统计稳定性策略：
- 固定随机种子 `seed`，确保复现实验；
- 使用足够样本（`n_qubits=4096`）降低 QBER 抽样波动；
- 分别给出 `qber_sample`、`qber_raw_before_ec`、`qber_total_sifted`，避免单指标误读；
- 用 clean/attacked 两场景对照进行行为回归检查。

## R09

适用场景：
- 量子信息课程中 BB84 协议流程演示；
- QKD 管线的离散仿真 baseline；
- 评估窃听对 QBER 与密钥率的一阶影响。

不适用场景：
- 需要器件级建模（暗计数、失配、偏振漂移、有限码长安全证明）；
- 工程部署前的安全认证或吞吐评估。

## R10

脚本内置正确性检查：
1. clean 场景 `qber_sample < 8%`；
2. intercept-resend 场景 `qber_sample > 18%`；
3. intercepted 场景 `final_key_len == 0`；
4. clean 场景 `final_key_len > 64`。

这些断言同时约束“协议物理趋势正确”与“可提取密钥行为正确”。

## R11

默认参数（`BB84Params`）：
- `n_qubits=4096`
- `channel_bit_flip=0.01`
- `sample_fraction=0.2`
- `seed=7`
- 场景 A：`eve_intercept_prob=0.0`
- 场景 B：`eve_intercept_prob=1.0`

参数意图：
- 保留小幅物理噪声（1%）以避免“零误码理想化”；
- 通过强攻击场景让 QBER 信号足够显著。

## R12

一次实测输出（`uv run python demo.py`）：

- clean 场景：
  - `sifted_len = 2053`
  - `qber_sample = 0.244%`
  - `qber_total_sifted = 0.585%`
  - `secret_fraction = 0.951`
  - `final_key_len = 1561`

- intercept-resend 场景：
  - `sifted_len = 2053`
  - `qber_sample = 26.098%`
  - `qber_total_sifted = 25.231%`
  - `secret_fraction = 0.000`
  - `final_key_len = 0`

结果符合 BB84 预期：窃听导致明显扰动并使安全密钥率归零。

## R13

正确性边界说明：
- 本实现验证的是“协议统计行为”而非“完整信息论安全证明”；
- 使用渐近密钥率公式，未覆盖有限码长修正项；
- 纠错阶段采用近似泄露计数，不代表真实协议栈的具体码构造效率。

因此它是教学/原型级验证，不是生产安全结论。

## R14

常见失败模式与修复：
- 失败：`n_qubits` 太小，QBER 波动大导致断言不稳定。
  - 修复：提升到 `>= 2000`，并固定 `seed`。
- 失败：攻击场景 QBER 不升反降。
  - 修复：检查 Eve 与 Bob 的“错基测量取随机位”逻辑。
- 失败：隐私放大输出长度异常。
  - 修复：检查 `binary_entropy` 边界处理和 `final_key_len` 的 `max(0, ...)` 截断。

## R15

工程化建议：
- 把采样估计与总体统计同时记录，降低采样偶然性误判；
- 对多个 `seed` 做批量 Monte Carlo，输出置信区间而非单点值；
- 在更真实链路中替换为显式纠错码（CASCADE/LDPC）并记录实际信息泄露。

## R16

可扩展方向：
- 引入诱骗态（decoy-state）与光子数分布模型；
- 加入有限码长安全分析（finite-key correction）；
- 引入探测器模型（暗计数、效率、死时间）；
- 把隐私放大替换为结构化 Toeplitz 哈希并做性能优化。

## R17

本目录交付说明：
- `demo.py`：可直接运行的 BB84 MVP，无交互输入；
- `README.md`：R01-R18 完整说明；
- `meta.json`：与任务元信息保持一致。

运行方式：

```bash
cd Algorithms/物理-量子信息-0235-量子密钥分发_(Quantum_Key_Distribution)
uv run python demo.py
```

## R18

`demo.py` 源码级算法流拆解（8 步，非黑盒）：
1. `BB84Params` 固化样本规模、窃听概率、信道噪声和采样比例，保证实验可复现。
2. `simulate_bb84` 先生成 `alice_bits/alice_bases`，把“随机制备”落实为显式二元数组。
3. 在 Eve 分支中，先按 `eve_mask` 决定拦截位置，再根据 `eve_bases == signal_bases` 区分“同基无误差/错基随机化”，并把测得结果重发到信道状态 `signal_bits/signal_bases`。
4. 对每个脉冲独立采样 `flip_mask` 实现信道 bit-flip，形成物理噪声注入后的信号。
5. Bob 用随机基测量：同基直接读出，错基随机取值，得到 `bob_bits`。
6. 通过 `sift_mask = (alice_bases == bob_bases)` 完成基筛选，再对筛选位随机抽样得到 `qber_sample`，其余位作为 `raw_key`。
7. 先用 `H2(Q)` 计算纠错泄露 `leak_ec_bits`，再用 `r = max(0, 1 - 2H2(Q))` 计算可提取安全比率并得到 `final_key_len`。
8. `privacy_amplification` 显式构造随机二元哈希矩阵并执行模 2 乘法压缩原始密钥，`main` 最后用 clean/attacked 断言验证协议行为。

整个实现可逐函数追踪，不依赖外部 QKD 黑盒库。
