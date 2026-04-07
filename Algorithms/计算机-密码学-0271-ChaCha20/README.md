# ChaCha20

- UID: `CS-0130`
- 学科: `计算机`
- 分类: `密码学`
- 源序号: `271`
- 目标目录: `Algorithms/计算机-密码学-0271-ChaCha20`

## R01

`ChaCha20` 是一种 ARX（Add-Rotate-XOR）结构的同步流密码，由 Daniel J. Bernstein 设计。  
它使用 256-bit 密钥、96-bit nonce（IETF 变体）和 32-bit 计数器，通过 20 轮混合生成伪随机密钥流，再与明文按字节异或实现加解密。

## R02

本条目的目标是给出一个可运行、可审阅的最小实现：
- 从零实现 quarter round 与 20 轮 block function；
- 基于块函数拼接 keystream，完成加密/解密；
- 通过 RFC 8439 测试向量与 round-trip 断言验证正确性。

## R03

核心思想：
- ChaCha20 不直接“变换明文”，而是先生成同长度密钥流 `K`；
- 加密：`C = P XOR K`；
- 解密：`P = C XOR K`；
- 安全性来自“在同一 key 下，nonce+counter 不能重复”，否则会泄漏 `P1 XOR P2`。

## R04

状态与轮函数（IETF 96-bit nonce 版）：
- 内部状态是 `16 x uint32`（总 512 bit）；
- 前 4 字是常量 `"expand 32-byte k"`；
- 中间 8 字来自 256-bit key；
- 第 13 字是 32-bit block counter；
- 后 3 字是 96-bit nonce；
- 20 轮 = 10 次 double-round（列轮 + 对角轮），每个 quarter round 由加法、异或、循环左移（16/12/8/7）组成。

## R05

输入输出定义（对应 `demo.py`）：
- `chacha20_block(key, counter, nonce) -> bytes(64)`：生成单个 64 字节密钥块；
- `chacha20_encrypt(key, nonce, plaintext, counter=1) -> bytes`：执行 keystream XOR；
- 同一函数可用于解密：`plaintext = encrypt(key, nonce, ciphertext, counter)`。

## R06

数据结构与实现约束：
- 全部使用 Python 原生整数与 `bytes`；
- 32-bit 运算通过 `& 0xFFFFFFFF` 显式截断；
- 字节序严格使用 little-endian（与 RFC 一致）；
- key 必须 32 字节，nonce 必须 12 字节，counter 在 `uint32` 范围内。

## R07

模块划分：
- `_rotl32`：32-bit 循环左移；
- `_quarter_round`：原地更新 4 个状态字；
- `_le_bytes_to_u32_words / _u32_words_to_le_bytes`：字节与 32-bit 字转换；
- `chacha20_block`：20 轮核心混合与 feed-forward；
- `chacha20_encrypt`：按 64 字节块扩展 keystream 并 XOR；
- `_self_test_*`：内置测试向量与回归检查；
- `main`：固定样例演示，保证无交互可运行。

## R08

复杂度分析：
- 设消息长度为 `n` 字节；
- `chacha20_block` 每块执行固定 20 轮，视为常数成本 `O(1)`；
- 整体加密需处理 `ceil(n/64)` 个块，总时间复杂度 `O(n)`；
- 输出缓冲与明文同长度，额外空间复杂度 `O(n)`（不计输入输出本身时可视为 `O(1)` 工作内存）。

## R09

正确性要点（直观）：
- 加密与解密同构，均为 XOR 同一密钥流；
- 对任意字节 `x` 与密钥流字节 `k`，有 `(x XOR k) XOR k = x`；
- 因此只要 `(key, nonce, counter 起点)` 相同，解密必恢复原文。

## R10

运行方式（无交互）：

```bash
cd Algorithms/计算机-密码学-0271-ChaCha20
uv run python demo.py
```

## R11

预期输出特征：
- 打印明文长度、密文长度、截断 hex 信息；
- `decryption ok: True`；
- 无异常退出（同时说明 quarter-round、block 向量、自定义 round-trip 三项断言均通过）。

## R12

边界与异常处理：
- key 长度错误、nonce 长度错误会抛 `ValueError`；
- counter 越界（<0 或超过 `uint32`）会抛 `ValueError/OverflowError`；
- 空明文 `b""` 可正常处理并返回空密文；
- 代码中显式处理计数器递增后的溢出，避免静默重复 keystream。

## R13

安全注意事项：
- `ChaCha20` 只提供机密性，不提供完整性/认证；
- 实际系统应使用 `ChaCha20-Poly1305` 等 AEAD 模式；
- 在相同 key 下重复 nonce 是严重错误，会泄漏不同明文的异或关系；
- 本 MVP 为教学实现，未覆盖常量时间、密钥擦除、侧信道防护等工程安全细节。

## R14

与相近方案对比（简述）：
- 对比 AES-CTR：两者都是“分离 keystream 与 XOR”的流化方案；ChaCha20 在纯软件平台通常更友好；
- 对比 OTP：OTP 需与消息等长真随机密钥，工程不可扩展；ChaCha20 用短密钥 + nonce + 计数器扩展密钥流；
- 对比 Salsa20：ChaCha20 继承并改进 round 结构（扩散效率更高，分析与部署更广泛）。

## R15

MVP 设计取舍：
- 不依赖第三方密码库，避免“黑盒调用几行结束”；
- 不实现 XChaCha20、Poly1305、随机 nonce 管理器，保持单文件可读；
- 重点保证“算法路径可追踪 + 内置测试可复现 + 运行零交互”。

## R16

自检清单：
- `README.md` 与 `demo.py` 无占位符残留；
- `uv run python demo.py` 可直接运行；
- RFC quarter-round 向量断言通过；
- RFC block-function 向量断言通过；
- 自定义明文 round-trip 断言通过。

## R17

可扩展方向：
- 增加 `ChaCha20-Poly1305` 一体化 AEAD 示例；
- 增加 `XChaCha20`（24-byte nonce）支持；
- 增加分片流式接口（文件/网络流）；
- 增加与标准库/第三方实现的交叉测试与性能基准。

## R18

源码级算法流程拆解（对应 `demo.py`，未使用第三方黑盒）：
1. 读取 `key(32B) + counter(uint32) + nonce(12B)`，拼成 16 个 `uint32` 初始状态（常量、密钥、计数器、nonce）。  
2. 复制一份 `working_state`，在其上执行 10 次 double-round（每次包含 4 个列 quarter round + 4 个对角 quarter round）。  
3. 每个 quarter round 按固定顺序做 `add mod 2^32 -> xor -> rotl`，旋转位数依次是 `16, 12, 8, 7`。  
4. 20 轮结束后，把 `working_state[i] + initial_state[i] (mod 2^32)` 做 feed-forward，得到最终 16 字输出状态。  
5. 将 16 个 `uint32` 按 little-endian 序列化为 64 字节 keystream block。  
6. 对明文分 64 字节分块，逐块生成 keystream，并对对应字节执行 XOR 得到密文。  
7. 解密时重复相同 keystream 生成与 XOR 流程，利用 XOR 自反性恢复原文。  
8. `main` 通过 RFC 向量与 round-trip 断言，验证实现从轮函数到整体验证链路都正确。  
