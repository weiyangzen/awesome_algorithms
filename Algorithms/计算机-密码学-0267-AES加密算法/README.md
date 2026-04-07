# AES加密算法

- UID: `CS-0129`
- 学科: `计算机`
- 分类: `密码学`
- 源序号: `267`
- 目标目录: `Algorithms/计算机-密码学-0267-AES加密算法`

## R01

`AES (Advanced Encryption Standard)` 是对称分组密码标准。它把固定长度明文块（128 bit）在密钥控制下变换为密文块，核心目标是同时满足机密性、抗差分攻击和抗线性攻击。
本目录实现的是可运行、可读、可验证的 `AES-128` 最小 MVP。

## R02

本条目解决的问题：
- 输入：16 字节密钥与任意长度字节串。
- 输出：可复现的加密结果，以及可逆的解密结果。
- 约束：不依赖黑箱密码库，直接在源码中实现 AES 轮函数与密钥扩展。

## R03

AES-128 的核心结构是 `Substitution-Permutation Network (SPN)`：
- `SubBytes`：非线性字节替换（S-Box）。
- `ShiftRows`：行移位扩散。
- `MixColumns`：列混淆（GF(2^8) 线性变换）。
- `AddRoundKey`：与轮密钥异或。

轮数为 `10`：首轮仅加轮密钥，中间 `9` 轮完整变换，最后一轮去掉 `MixColumns`。

## R04

关键数学对象：
- 字节域：`GF(2^8)`，不可约多项式 `x^8 + x^4 + x^3 + x + 1`（即 `0x11B`）。
- `MixColumns` 乘法矩阵（加密）：

```text
[02 03 01 01]
[01 02 03 01]
[01 01 02 03]
[03 01 01 02]
```

- `KeyExpansion`：由 16 字节初始密钥扩展到 11 组轮密钥（共 176 字节）。

## R05

高层伪代码：

```text
round_keys = KeyExpansion(key)
state = plaintext_block
state = AddRoundKey(state, round_keys[0])
for r in 1..9:
  state = SubBytes(state)
  state = ShiftRows(state)
  state = MixColumns(state)
  state = AddRoundKey(state, round_keys[r])
state = SubBytes(state)
state = ShiftRows(state)
state = AddRoundKey(state, round_keys[10])
return state
```

解密按逆序执行 `InvShiftRows / InvSubBytes / AddRoundKey / InvMixColumns`。

## R06

时间复杂度：
- 单块加密/解密是常数轮数（10 轮），每轮固定 16 字节运算，因此为 `O(1)`（相对块大小常数）。
- 对长度为 `n` 的消息（按 16 字节分块）是 `O(n)`。

## R07

空间复杂度：
- 轮密钥固定 `176` 字节。
- 每次处理一个 16 字节状态。
- 总体额外空间 `O(1)`（不含输入输出缓冲）。

## R08

本 MVP 参数与实现选择：
- 密钥规格：`AES-128`（16 字节密钥）。
- 分组大小：`16` 字节。
- 示例模式：`ECB + PKCS#7`（用于演示可运行性，不代表生产安全默认）。
- 正确性基准：内置 `FIPS-197 Appendix C.1` 测试向量。

## R09

实现边界与假设：
- 仅实现 `AES-128`，未扩展到 `AES-192/256`。
- 模式只给出 ECB 封装，主要用于展示分组算法本体。
- 输入输出均为 `bytes`，无交互式参数输入。
- 目标是教学与验证，不是高性能密码库替代品。

## R10

运行方式（非交互）：

```bash
uv run python demo.py
```

程序会先执行标准测试向量，再执行一次 ECB 往返验证并打印结果。

## R11

输出解读：
- `AES-128 known-answer test: PASS`：表示与 FIPS 向量一致。
- `Block ciphertext` 应为 `69c4e0d86a7b0430d8cdb78070b4c55a`。
- `ECB round-trip : PASS`：表示加密后再解密可恢复原文。

## R12

快速正确性检查清单：
- 固定向量加密结果必须精确匹配。
- 单块解密必须恢复原始明文。
- PKCS#7 去填充需拒绝非法填充长度或填充字节。
- 非 16 字节倍数密文解密应报错。

## R13

常见失败模式：
- 状态矩阵索引方向错误（行主序/列主序混淆）。
- `ShiftRows` 与 `MixColumns` 组合次序错误。
- 密钥扩展中 `RotWord/SubWord/Rcon` 次序错误。
- 逆变换顺序不对，导致“可加密不可解密”。

## R14

与常见方案对比：
- 相比直接调用第三方库：本实现可读性和可审计性更高，但性能更低。
- 相比 DES：AES 分组与密钥安全性更强，已成为主流标准。
- 相比流密码（如 ChaCha20）：AES 是分组密码，本实现通过 ECB 演示分组流程，不覆盖 AEAD 场景。

## R15

工程实践建议：
- 生产环境优先使用成熟库和安全模式（如 GCM/CTR + 完整性校验）。
- 保留标准向量回归测试，避免重构引入细微错误。
- 若追求性能，优先改为底层实现（SIMD/AES-NI）而非 Python 逐字节循环。

## R16

可扩展方向：
- 增加 `CBC/CTR/GCM` 等分组工作模式。
- 支持 `AES-192/256` 密钥长度。
- 增加更多 NIST 测试向量和批量随机回归。
- 增加命令行参数以处理文件输入输出（保持非交互默认）。

## R17

依赖与自包含性：
- 依赖：仅 Python 标准库（无第三方包）。
- 输入：`demo.py` 内置样例与测试向量。
- 输出：标准输出打印校验结果和样例密文。

目录内 `README.md + demo.py + meta.json` 即可独立完成验证。

## R18

`demo.py` 源码级算法流程（9 步）：

1. `main()` 设定 FIPS-197 AES-128 测试向量（16 字节 key / 16 字节 plaintext / 期望 ciphertext）。
2. `expand_key()` 通过 `RotWord -> SubWord -> Rcon` 迭代把初始密钥扩展为 11 组轮密钥。
3. `encrypt_block()` 对初始状态先执行一次 `_add_round_key(state, round_keys[0])`。
4. 进入第 1-9 轮：依次调用 `_sub_bytes -> _shift_rows -> _mix_columns -> _add_round_key`。
5. 最后一轮仅执行 `_sub_bytes -> _shift_rows -> _add_round_key`（不做列混淆）。
6. `decrypt_block()` 从最后一组轮密钥反向恢复：`_inv_shift_rows -> _inv_sub_bytes -> _add_round_key -> _inv_mix_columns`，最后回到第 0 轮密钥。
7. `_gf_mul()` 在 `GF(2^8)` 中完成 `MixColumns/InvMixColumns` 所需的有限域乘法。
8. `encrypt_ecb()` 与 `decrypt_ecb()` 使用 `_pkcs7_pad/_pkcs7_unpad` 处理任意长度消息，并分块调用单块加解密。
9. `main()` 对比标准向量与往返结果，全部通过后打印 `PASS`，形成可重复验证的最小闭环。
