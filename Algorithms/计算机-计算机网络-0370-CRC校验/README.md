# CRC校验

- UID: `CS-0217`
- 学科: `计算机`
- 分类: `计算机网络`
- 源序号: `370`
- 目标目录: `Algorithms/计算机-计算机网络-0370-CRC校验`

## R01

CRC（Cyclic Redundancy Check，循环冗余校验）是链路层和存储系统常用的差错检测方法。核心思想是把比特串看作 GF(2) 上的多项式，发送端追加余数，接收端复算余数是否为 0，以判断传输过程是否发生比特错误。

## R02

本条目 MVP 目标：实现一个可运行的 CRC 编码与校验流程（不依赖协议栈黑箱）。

输入：
- `payload`：原始字节串（示例使用 `b"CRC"`）；
- `generator_bits`：生成多项式比特串（示例 `10011`，对应 `x^4 + x + 1`）；
- `flip_positions`：信道误码注入位置。

输出：
- 发送码字（数据 + CRC）；
- 每个误码场景下的 syndrome（余数）与是否通过校验；
- 最小正确性断言结果。

## R03

数学模型（GF(2)）：
- 数据多项式 `M(x)`，生成多项式 `G(x)`，其阶为 `r`；
- 发送端计算 `T(x) = x^r * M(x) + R(x)`，其中 `R(x)` 为 `x^r * M(x)` 对 `G(x)` 的余数；
- 接收端对收到的 `T'(x)` 做除法：若 `T'(x) mod G(x) = 0`，判定“未检测到错误”，否则判定“检测到错误”。

## R04

`demo.py` 的主流程：
1. 解析并校验生成多项式；
2. 将 `payload` 转为 bit 序列；
3. 用 GF(2) 长除法计算 CRC 并拼接为码字；
4. 构造无误码、单比特误码、双比特误码、突发误码场景；
5. 对每个场景复算 syndrome 并输出判决；
6. 执行断言：无误码必须通过、所有单比特误码必须被检出。

## R05

关键数据结构：
- `np.ndarray(dtype=np.uint8)`：统一承载 bit 向量；
- `CheckRecord`：单个场景的记录（名称、翻转位置、是否通过、syndrome）；
- `List[CheckRecord]`：汇总所有实验场景输出。

## R06

正确性直觉：
- 发送端将余数补到数据后，合法码字可被 `G(x)` 整除；
- 任意传输扰动等价于叠加误差多项式 `E(x)`；
- 接收端实际检查的是 `(T(x) + E(x)) mod G(x)`；
- 当 `E(x)` 不是 `G(x)` 的倍数时，余数非 0，从而检出错误。

## R07

复杂度分析（数据长度 `n`，生成多项式长度 `m`）：
- 编码：`O(n * m)`（滑动异或长除）；
- 校验：`O((n + m - 1) * m)`；
- 空间：`O(n + m)`。

本 MVP 规模很小，重点放在可解释性与可验证性。

## R08

边界与异常处理：
- `generator_bits` 含非 `0/1`、长度小于 2、首尾不是 `1`：抛 `ValueError`；
- `payload` 为空：抛 `ValueError`；
- `flip_positions` 越界：抛 `ValueError`；
- 除法输入维度不正确或被除数比除数短：抛 `ValueError`。

## R09

MVP 设计取舍：
- 只实现 CRC 核心，不叠加 ARQ/FEC，保证主题聚焦；
- 误码采用确定性注入，保证每次运行输出一致；
- 使用 `numpy` 做位运算，代码短、可读且易断言；
- 不调用第三方 CRC 黑箱接口，保证算法细节可追踪。

## R10

`demo.py` 函数职责：
- `parse_generator_bits`：多项式参数合法化；
- `bits_from_bytes`：字节到 bit 序列转换；
- `mod2_divide`：GF(2) 长除法核心；
- `crc_encode / crc_syndrome / crc_check`：编码、求 syndrome、校验；
- `flip_bits`：误码注入；
- `find_detected_two_bit_error / find_detected_burst_error`：自动挑选可检出的示例场景；
- `run_demo / main`：运行实验、断言与打印。

## R11

运行方式：

```bash
cd Algorithms/计算机-计算机网络-0370-CRC校验
uv run python demo.py
```

脚本无需交互输入，直接打印实验与校验结果。

## R12

输出说明：
- `Generator bits`：本次使用的 CRC 生成多项式；
- `Payload` / `Data bits` / `Codeword bits`：编码前后数据；
- `Case results`：每个误码场景的翻转位、是否通过、syndrome；
- `All checks passed.`：表示内置断言全部通过。

## R13

最小验证清单：
- 无误码场景必须 `valid=True`；
- 所有单比特误码都必须 `valid=False`；
- 至少存在一个双比特误码示例被检出；
- 至少存在一个突发误码示例被检出；
- `README.md` 与 `demo.py` 不包含模板占位符。

## R14

当前 demo 固定参数：
- `payload = b"CRC"`；
- `generator_bits = "10011"`；
- 自动搜索一个可检出的双比特翻转 `(i, j)`；
- 自动搜索一个可检出的 4 比特突发翻转区间。

这样可以避免手工挑样本导致的偶然漏检。

## R15

与其他简单校验方式对比：
- 奇偶校验：开销极低，但检测能力弱；
- 加和校验（Checksum）：实现简单，但对某些模式错误敏感性不足；
- CRC：在类似开销下对突发误码检测能力更强，工程上应用最广。

## R16

适用与不适用场景：
- 适用：链路层帧校验、文件块完整性检查、教学演示；
- 不适用：需要纠错（而非只检测）的场景，应结合 FEC 或 ARQ。

## R17

可扩展方向：
- 对比不同多项式（CRC-8/CRC-16/CRC-32）的误检率；
- 增加随机误码仿真并统计检测概率；
- 与 Stop-and-Wait/GBN 结合，形成“检测 + 重传”闭环；
- 增加 bit 反射、初值、异或输出等工程参数。

## R18

`demo.py` 源码级算法流（8 步，非黑箱）：
1. `main` 调用 `run_demo`，设定 `payload` 与 `generator_bits`。  
2. `parse_generator_bits` 校验多项式合法性并转为 `np.ndarray`。  
3. `bits_from_bytes` 把字节串展开为 MSB-first 的 bit 序列。  
4. `crc_encode` 在数据后补 `r` 个 0，并调用 `mod2_divide` 做 GF(2) 长除得到余数，再拼接为码字。  
5. `find_detected_two_bit_error` 与 `find_detected_burst_error` 在当前码字上搜索可检出的示例误码位置。  
6. `flip_bits` 对码字注入误码，`crc_syndrome` 复算余数，`crc_check` 根据余数是否全 0 给出判决。  
7. `run_demo` 汇总 `CheckRecord` 列表，逐行打印每个场景的翻转位、校验结果和 syndrome。  
8. 最后执行断言：无误码必须通过、所有单比特错误必须失败、示例双比特和突发错误必须失败；全部通过后输出 `All checks passed.`。  
