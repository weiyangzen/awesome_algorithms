# Salsa20

- UID: `CS-0131`
- 学科: `计算机`
- 分类: `密码学`
- 源序号: `272`
- 目标目录: `Algorithms/计算机-密码学-0272-Salsa20`

## R01

Salsa20 是 Daniel J. Bernstein 设计的流密码（stream cipher），核心特点是通过 ARX（Add-Rotate-XOR）操作生成伪随机密钥流，再与明文逐字节异或完成加解密。它不依赖 S 盒和查表，结构简单、实现高效，适合软件场景。

## R02

算法输入是 `key + nonce + block_counter`，输出是 64 字节密钥流块。  
实际加密时执行：
`ciphertext = plaintext XOR keystream`
解密同理：
`plaintext = ciphertext XOR keystream`
因为异或可逆，且只要密钥流一致就能还原原文。

## R03

本目录 MVP 约定：
- 密钥长度：16 或 32 字节（推荐 32 字节）
- nonce 长度：8 字节
- 计数器：64-bit block counter
- 块大小：64 字节
- 接口：
  - `salsa20_keystream(key, nonce, length, counter=0) -> bytes`
  - `salsa20_xor(key, nonce, data, counter=0) -> bytes`

## R04

内部状态为 16 个 32-bit 无符号整数（共 512 bit），按 little-endian 解析与回写。  
32 字节密钥模式下状态布局：
`[c0, k0, k1, k2, k3, c1, n0, n1, b0, b1, c2, k4, k5, k6, k7, c3]`
其中：
- `c*` 是常量 `"expand 32-byte k"` 分解后的 4 个字
- `k*` 是密钥字
- `n*` 是 nonce 字
- `b*` 是 block counter 的低/高 32 位

## R05

QuarterRound（四元组变换）是 Salsa20 的基本非线性操作：
- `z1 = y1 XOR ROTL(y0 + y3, 7)`
- `z2 = y2 XOR ROTL(z1 + y0, 9)`
- `z3 = y3 XOR ROTL(z2 + z1, 13)`
- `z0 = y0 XOR ROTL(z3 + z2, 18)`
其中加法为 `mod 2^32`，`ROTL` 为循环左移。  
20 轮由 10 个 double-round 组成，每个 double-round = column round + row round。

## R06

伪代码（与 `demo.py` 对应）：

```text
function SALSA20_HASH(state[16]):
    x = copy(state)
    repeat 10 times:
        x = COLUMN_ROUND(x)
        x = ROW_ROUND(x)
    for i in 0..15:
        out[i] = (x[i] + state[i]) mod 2^32
    return serialize_le(out)

function KEYSTREAM(key, nonce, length, counter):
    out = []
    while len(out) < length:
        state = BUILD_STATE(key, nonce, counter)
        out += SALSA20_HASH(state)   # 64 bytes
        counter += 1
    return out[:length]

function XOR_CRYPT(key, nonce, data, counter):
    ks = KEYSTREAM(key, nonce, len(data), counter)
    return data XOR ks
```

## R07

正确性要点：
- `salsa20_xor` 对同一 `(key, nonce, counter)` 产生同一密钥流。
- 异或逆元性质：`(P XOR K) XOR K = P`。
- 因此加密后再用同参数执行一次 `salsa20_xor` 必然得到原文。
`demo.py` 中用断言验证了这一点。

## R08

复杂度：
- 时间复杂度：`O(n)`，`n` 为数据字节数；每 64 字节对应固定 20 轮运算。
- 空间复杂度：`O(1)`（不计输出缓冲），核心仅维护常量规模状态数组。

## R09

安全使用注意事项：
- 同一密钥下，`nonce` 绝不能重复（重复会复用密钥流，导致明文泄露风险）。
- 计数器不能回绕复用。
- Salsa20 本身仅提供机密性，不提供完整性与认证；工程上应与 MAC 或 AEAD 方案组合。
- 本目录实现用于教学与算法理解，不替代经过审计的生产密码库。

## R10

典型应用场景：
- 需要高性能软件流加密的场景
- 嵌入式/跨平台环境（ARX 运算友好）
- 作为教学案例理解流密码结构、ARX 设计与 nonce/counter 管理

## R11

MVP 设计取舍：
- 采用纯 Python 实现，避免第三方黑盒，便于逐行追踪算法。
- 同时支持 16/32 字节密钥，覆盖 Salsa20 常见接口。
- 提供可复现实验输出：密文哈希、密钥流前缀、单比特翻转雪崩统计。
- 不引入 I/O 和交互参数，保证 `uv run python demo.py` 可直接运行。

## R12

运行方式：

```bash
uv run python demo.py
```

预期输出包含：
- `ciphertext_sha256=...`
- `keystream_head_16=...`
- `decryption_ok=True`
- `avalanche_bits_in_64B=.../512`

## R13

输出解释：
- `ciphertext_sha256`：用于固定输入下的结果指纹，便于回归测试。
- `keystream_head_16`：首 16 字节密钥流，可用于快速比对实现一致性。
- `decryption_ok`：验证加解密往返正确。
- `avalanche_bits_in_64B`：将密钥翻转 1 bit 后，64 字节密钥流中变化的 bit 数，通常接近 256/512（约 50%）。

## R14

边界与异常处理：
- 非法密钥长度（非 16/32）会抛出 `ValueError`。
- 非法 nonce 长度（非 8）会抛出 `ValueError`。
- 负长度密钥流请求会抛出 `ValueError`。
- counter 需在 `[0, 2^64)` 范围内。

## R15

测试建议（本目录已覆盖最小核心）：
- 功能测试：加密再解密应恢复原文。
- 稳定性测试：固定输入下输出哈希应稳定。
- 边界测试：空消息、1 字节消息、跨多个 64 字节块消息。
- 参数异常测试：错误 key/nonce/counter 触发异常。

## R16

与相近算法简对比：
- 对比 RC4：Salsa20 不依赖早期偏差明显的状态置换机制，设计更现代。
- 对比 ChaCha20：二者同属 ARX 流密码，ChaCha20 在轮函数常数/旋转参数和扩散路径上做了改进，工程生态更广；Salsa20 仍是理解该家族的基础。
- 对比 AES-CTR：都可作为流加密使用，但内部构造不同（分组密码计数器模式 vs 原生流密码）。

## R17

局限与可扩展方向：
- 本实现未集成认证机制，可扩展为 `Encrypt-then-MAC` 示例。
- 未做性能优化，可进一步用 `numpy`/`numba`/C 扩展提升吞吐。
- 未内置官方测试向量校验，可补充标准向量进行一致性对照。
- 可扩展到 XSalsa20（更长 nonce）以降低 nonce 管理压力。

## R18

`demo.py` 的源码级流程拆解（8 步）：
1. 参数校验与状态构建：`_build_state` 检查 key/nonce/counter，并按规范拼接 16 个 32-bit 字。
2. 基元准备：`_rotl32` 与 `_quarterround` 定义 ARX 核心操作。
3. 轮函数执行：`_salsa20_hash` 对状态执行 10 次 double-round（先列后行）。
4. 前馈求和：将轮后状态与初始状态逐字 `mod 2^32` 相加，得到 64 字节块输出。
5. 密钥流拼接：`salsa20_keystream` 按块递增 counter，循环生成直到满足请求长度。
6. 加解密统一：`salsa20_xor` 以密钥流与输入字节异或，兼容加密和解密。
7. 自检验证：`main` 中执行 round-trip 断言，确保 `decrypt(encrypt(P)) == P`。
8. 可观测指标输出：打印密文哈希、密钥流前缀与雪崩统计，便于回归和行为检查。
