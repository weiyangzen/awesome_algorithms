# 差错控制算法

- UID: `CS-0216`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `369`
- 目标目录: `Algorithms/计算机-计算机网络-0369-差错控制算法`

## R01

本条目实现一个可运行的网络差错控制 MVP：`CRC 差错检测 + 停等 ARQ（Stop-and-Wait）`。
目标是用最小代码展示链路层/传输层常见闭环：
- 发送端给数据帧附加 CRC；
- 信道引入比特翻转；
- 接收端做 CRC 校验并返回 ACK/NACK；
- 发送端在 NACK 时重传，直到成功。

## R02

问题定义（MVP 范围）：
- 输入：
1. `payloads`：待发送字节序列（每个元素在 `[0, 255]`）；
2. `generator_bits`：CRC 生成多项式比特串（示例 `10011`）；
3. `error_plan`：确定性误码计划，键为 `(packet_index, attempt)`，值为翻转位位置列表；
4. `max_retries`：单包最大重传次数。
- 输出：
1. `decoded_payloads`：成功接收后的载荷序列；
2. `records`：每次发送尝试的详细日志（ACK/NACK、翻转位、帧比特串）；
3. `stats`：总尝试数、重传数、帧长、goodput。

## R03

算法规则：
- 帧格式：`seq(1 bit) + payload(8 bits) + crc(k bits)`；
- CRC 编码：对 `data_bits * x^k` 进行 GF(2) 模 2 长除，余数作为 CRC；
- 信道模型：按 `error_plan` 指定位置翻转比特；
- 判决规则：接收帧 CRC 余数全 0 则 `ACK`，否则 `NACK`；
- ARQ 机制：`NACK` 触发重传同一帧，直到成功或超出 `max_retries`。

## R04

`demo.py` 主流程：
1. 标准化并校验 CRC 多项式；
2. 为每个 payload 构造带 CRC 的发送帧；
3. 按尝试轮次注入误码并进行 CRC 校验；
4. 若校验通过则解析 `seq/payload` 并确认 ACK；
5. 若校验失败则记录 NACK 并进入下一次重传；
6. 汇总所有尝试记录，计算重传率与 goodput；
7. 用固定样例断言关键统计量；
8. 打印时间线和最终校验结果。

## R05

关键数据结构：
- `AttemptRecord`：单次发送尝试的完整状态；
- `decoded_payloads: List[int]`：成功接收的数据；
- `records: List[AttemptRecord]`：仿真全日志；
- `stats: Dict[str, float]`：聚合性能指标；
- `np.ndarray`：位数组、CRC 运算、固定断言中的序列比较。

## R06

正确性直觉：
- CRC 提供“传输后是否被破坏”的快速判据；
- 停等 ARQ 提供“检测失败就重发”的恢复机制；
- 在误码是暂态且重试上限足够时，系统能收敛到成功接收；
- 因此该组合实现“检测 + 恢复”的最小闭环差错控制。

## R07

复杂度分析（设包数为 `P`，每包最多尝试 `R` 次，帧长 `L`）：
- 时间复杂度：`O(P * R * L)`（每次尝试含一次 CRC 除法）；
- 空间复杂度：`O(P * R)`（保存全部尝试日志）。

在本 MVP 中 `L` 很小（13 bit），因此实现重点在可解释性而非极限性能。

## R08

边界与异常处理：
- `payload` 不在 `[0,255]`：抛 `ValueError`；
- `generator_bits` 非 0/1、长度不足、首尾不是 1：抛 `ValueError`；
- `max_retries <= 0`：抛 `ValueError`；
- `flip_positions` 越界：抛 `ValueError`；
- 超过最大重传仍失败：抛 `RuntimeError`。

## R09

MVP 设计取舍：
- 采用 `CRC + Stop-and-Wait`，避免直接调用黑箱协议栈；
- 误码使用确定性 `error_plan`，便于可复现实验和断言；
- 使用 `numpy` 做位运算与序列校验，工具栈最小；
- 不扩展突发误码统计模型、滑动窗口 ARQ、FEC 译码器，保持代码聚焦。

## R10

`demo.py` 函数职责：
- `bits_from_int / bits_to_int`：整数与位序列互转；
- `normalize_generator_bits`：CRC 多项式参数合法化；
- `mod2_divide`：GF(2) 长除核心；
- `crc_encode / crc_is_valid`：CRC 编码与校验；
- `apply_error_pattern`：注入误码；
- `build_frame / parse_frame`：帧封装与解析；
- `simulate_stop_and_wait_arq`：停等 ARQ 主循环；
- `run_demo / main`：固定样例、断言与打印入口。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机网络-0369-差错控制算法
uv run python demo.py
```

脚本无需交互输入，直接打印时间线和校验结果。

## R12

输出说明：
- `Transmission timeline`：逐次尝试的 `pkt/seq/payload/attempt/flips/ACK-NACK`；
- `decoded payloads`：接收端最终恢复的数据序列；
- `stats`：
1. `total_packets`：原始包数；
2. `total_attempts`：总发送尝试次数；
3. `retransmissions`：重传次数；
4. `frame_bits`：每帧总 bit 数；
5. `goodput`：有效载荷比特占已发送总比特比例；
- `All checks passed.`：表示内置断言全部通过。

## R13

最小验证清单：
- 校验 `decoded_payloads == payloads`；
- 校验每个 packet 仅被 ACK 一次；
- 校验每包尝试次数为预期 `[2,1,3,1]`；
- 校验总尝试数 `7`、重传数 `3`；
- 校验所有 NACK 均来自注入误码场景。

## R14

当前 demo 固定参数：
- `payloads = [0x3A, 0xC1, 0x07, 0xBE]`；
- `generator_bits = [1,0,0,1,1]`（`x^4 + x + 1`）；
- `error_plan = {(0,1):[2], (2,1):[10], (2,2):[7]}`；
- `max_retries = 6`。

该参数覆盖：
- 一次单重传成功；
- 一次双重传成功；
- 无误码直接成功；
- CRC 与 ARQ 协同工作的完整路径。

## R15

与其他差错控制思路对比：
- 本实现：`CRC(检测) + ARQ(重传恢复)`，实现简单、可解释性高；
- Hamming/Reed-Solomon：偏向前向纠错（FEC），可在接收端直接纠错但编码复杂度更高；
- Go-Back-N/Selective Repeat：窗口化 ARQ，吞吐更高但状态机更复杂。

## R16

适用场景：
- 课程教学中的差错控制入门演示；
- 需要可复现实验日志的协议原型验证；
- 解释“误码检测”与“重传恢复”如何形成闭环。

不适用场景：
- 高吞吐链路仿真（应使用滑动窗口 ARQ）；
- 强突发误码信道下的工业级纠错设计（应考虑更强 FEC/交织）；
- 需要真实物理层时序与缓冲建模的场合。

## R17

可扩展方向：
- 将停等 ARQ 扩展为 Go-Back-N 或 Selective Repeat；
- 在 `error_plan` 之外增加随机误码与统计置信区间；
- 引入突发误码模型（Gilbert-Elliott）；
- 对比不同 CRC 多项式的漏检概率；
- 增加 FEC（如 Hamming(7,4)）并比较延迟/吞吐/开销。

## R18

`demo.py` 源码级算法流（8 步，非黑箱）：
1. `main` 调用 `run_demo`，配置 payload、CRC 多项式与误码计划。  
2. `run_demo` 进入 `simulate_stop_and_wait_arq`，先由 `normalize_generator_bits` 校验多项式合法性。  
3. 对每个包用 `build_frame` 构造 `seq + payload`，再调用 `crc_encode` 附加 CRC。  
4. 每次发送尝试按 `(packet_index, attempt)` 从 `error_plan` 取翻转位，`apply_error_pattern` 生成接收帧。  
5. 接收端调用 `crc_is_valid`，其内部通过 `mod2_divide` 计算 syndrome；零余数判 ACK，非零判 NACK。  
6. ACK 时用 `parse_frame` 解析并保存 payload；NACK 时继续下一次尝试，直到达到 `max_retries`。  
7. 所有包处理完后统计 `total_attempts/retransmissions/goodput`，并输出完整 `AttemptRecord` 时间线。  
8. `run_demo` 对解码结果、每包尝试次数、总重传次数执行固定断言，全部通过后打印 `All checks passed.`。  
